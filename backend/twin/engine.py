"""The DSU digital twin: live state, anomaly detection, alerts and work orders.

Every tick the engine reads the replayed datasets at the current simulated
time, layers on live evidence (camera detections, student reports, injected
demo scenarios) and turns rule + ML detections into alerts with rupee and CO2
impact. Alerts become facilities work orders that close only with proof.
"""
import itertools
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as C
from twin.clock import SimClock, to_dataset_time
from twin.geo import Campus, haversine_m
from twin.models import Baselines, EnergyIForest, LeakDetector, OccupancyModel, save_metrics

TEAM = {"water": "Plumbing", "energy": "Electrical", "waste": "Housekeeping", "resource": "Facilities"}
# Pressure sensors stand in for the campus water zones.
PRESSURE_ZONES = ["soe-a-block", "soe-b-block", "cdsimer-block-a", "cdsimer-hospital", "boys-hostel", "soe-food-court"]
# Bin fill rates (% per hour) by hour of day for each kind of location.
BIN_PROFILE = {
    "canteen": [0.3] * 7 + [2, 3, 3, 2, 3, 6, 8, 6, 3, 3, 4, 3, 2, 3, 2, 1, 0.5],
    "academic": [0.2] * 8 + [2, 3, 3, 3, 3, 4, 3, 3, 3, 2, 1, 0.5, 0.3, 0.3, 0.2, 0.2],
    "hostel": [0.5] * 6 + [1, 3, 4, 2, 1, 1, 3, 3, 1, 1, 1, 2, 3, 4, 3, 2, 1, 0.5],
    "hospital": [1] * 7 + [2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1.5, 1, 1, 1],
}


@dataclass
class Alert:
    id: str
    kind: str          # energy | water | waste | resource
    rule: str
    asset: str
    title: str
    detail: str
    severity: str      # warn | crit
    source: str        # sensor-ml | camera | student | demo-scenario
    created: str
    updated: str
    lng: float
    lat: float
    inr_per_day: float = 0.0
    co2_kg_per_day: float = 0.0
    status: str = "open"   # open | assigned | resolved | self-cleared
    evidence: dict = field(default_factory=dict)
    work_order: str | None = None


@dataclass
class WorkOrder:
    id: str
    alert_id: str
    team: str
    title: str
    created: str
    deadline: str
    status: str = "assigned"   # assigned | done | overdue
    proof: dict | None = None
    note: str = ""


class Twin:
    def __init__(self):
        self.lock = threading.RLock()
        self.clock = SimClock(speed=60)
        self.campus = Campus()
        self.buildings = pd.read_pickle(C.CACHE / "buildings.pkl").interpolate(limit_direction="both").fillna(0)
        self.rooms = pd.read_pickle(C.CACHE / "rooms.pkl").interpolate(limit_direction="both").fillna(0)
        self.water = pd.read_pickle(C.CACHE / "water.pkl")
        self.bld_ids = sorted({c[0] for c in self.buildings.columns})
        self.room_ids = sorted({c[0] for c in self.rooms.columns})
        # Keep each meter's real load shape but scale its annual total to an Indian benchmark intensity.
        for b in self.bld_ids:
            p = self.campus.buildings[b]
            target = p["area_m2"] * max(p["floors"], 1) * C.EUI_KWH_M2_YEAR.get(p["type"], 100)
            col = (b, "elec_kwh")
            self.buildings[col] = self.buildings[col] * (target / max(float(self.buildings[col].sum()), 1.0))

        self.base = {}
        self.iforest = {}
        for b in self.bld_ids:
            e = self.buildings[(b, "elec_kwh")]
            self.base[(b, "elec")] = Baselines(e)
            self.base[(b, "water")] = Baselines(self.buildings[(b, "water_l")])
            self.iforest[b] = EnergyIForest(e)
        self.occupancy = OccupancyModel()
        mapping = __import__("json").loads((C.CACHE / "mapping.json").read_text(encoding="utf-8"))
        self.leak = LeakDetector(self.water, mapping["water"].get("leak_onsets", {}).values())
        self.metrics = {"occupancy": self.occupancy.metrics, "leak": self.leak.metrics}
        save_metrics(self.metrics)

        self.alerts: dict[str, Alert] = {}
        self.orders: dict[str, WorkOrder] = {}
        self.scenarios: list[dict] = []
        self.camera_occupancy: dict[str, dict] = {}   # room -> {people, at}
        self.bin_obs: dict[str, dict] = {}            # bin -> {fill, at, source}
        self.bin_collected: dict[str, datetime] = {}
        self.bin_pickups = [7, 15]
        self.events: list[dict] = []                  # history for hotspots / reports
        self._ids = itertools.count(1)

    # ---------- data access ----------
    def _row(self, frame, sim, year, freq):
        t = pd.Timestamp(to_dataset_time(sim, year)).floor(freq)
        i = frame.index.get_indexer([t], method="nearest")[0]
        return frame.iloc[i], frame.index[i]

    def building_series(self, b, sim, hours=24):
        t = pd.Timestamp(to_dataset_time(sim, 2017)).floor("h")
        s = self.buildings[b].loc[t - pd.Timedelta(hours=hours - 1): t]
        base = [self.base[(b, "elec")].expected(ts) for ts in s.index]
        stamps = [(sim - timedelta(hours=len(s) - 1 - i)).strftime("%H:00") for i in range(len(s))]
        return {"t": stamps, "kwh": s["elec_kwh"].round(1).tolist(), "baseline": [round(x, 1) for x in base],
                "water_l": s["water_l"].round(0).tolist()}

    # ---------- live evidence ----------
    def add_scenario(self, kind, target=None):
        sim = self.clock.now()
        with self.lock:
            if kind == "leak":
                target = target or "boys-hostel"
                self.scenarios.append({"kind": "leak", "target": target, "lph": 600.0, "start": sim})
            elif kind == "ac_night":
                target = target or self.room_ids[0]
                self.scenarios.append({"kind": "ac_night", "target": target, "kw": 3.2, "start": sim})
            elif kind == "bin_overflow":
                target = target or "bin-foodcourt-1"
                self.bin_obs[target] = {"fill": 100, "at": sim, "source": "demo-scenario"}
            elif kind == "night":
                self.clock.jump_to(sim.replace(hour=2, minute=0) + timedelta(days=1 if sim.hour >= 2 else 0))
            return {"kind": kind, "target": target}

    def observe_people(self, room, people):
        with self.lock:
            self.camera_occupancy[room] = {"people": int(people), "at": self.clock.now()}

    def observe_bin(self, bin_id, fill, source):
        with self.lock:
            self.bin_obs[bin_id] = {"fill": float(fill), "at": self.clock.now(), "source": source}

    # ---------- bins ----------
    def _bin_type(self, bin_id):
        return self.campus.buildings.get(self.campus.nearest_building(bin_id), {}).get("type", "academic")

    def bin_rate(self, bin_id, hour):
        return BIN_PROFILE.get(self._bin_type(bin_id), BIN_PROFILE["academic"])[hour % 24]

    def bin_fill(self, bin_id, sim):
        last = sim.replace(minute=0, second=0, microsecond=0)
        while last.hour not in self.bin_pickups:
            last -= timedelta(hours=1)
        if bin_id in self.bin_collected and self.bin_collected[bin_id] > last:
            last = self.bin_collected[bin_id]
        fill, t, source = 0.0, last, "model"
        obs = self.bin_obs.get(bin_id)
        if obs and obs["at"] > last:
            fill, t, source = obs["fill"], obs["at"], obs["source"]
        while t < sim:
            step = min(timedelta(hours=1), sim - t)
            fill += self.bin_rate(bin_id, t.hour) * step.total_seconds() / 3600 * 3.2
            t += step
        return min(fill, 100.0), source

    def bin_state(self, sim):
        out = {}
        for b in self.campus.of_type("bin"):
            fill, source = self.bin_fill(b, sim)
            hours, f, h = 0, fill, sim.hour
            while f < 100 and hours < 48:
                f += self.bin_rate(b, h) * 3.2
                h, hours = h + 1, hours + 1
            lng, lat = self.campus.location(b)
            out[b] = {"fill": round(fill), "source": source, "eta_full_h": hours if fill < 100 else 0,
                      "lng": lng, "lat": lat, "status": "crit" if fill >= 95 else "warn" if fill >= 75 else "ok"}
        return out

    def pickup_route(self, sim):
        bins = self.bin_state(sim)
        todo = [b for b, s in bins.items() if s["fill"] >= 60 or s["eta_full_h"] <= 3]
        pos = self.campus.location("meter-main-inlet")
        route = []
        while todo:
            nxt = min(todo, key=lambda b: haversine_m(*pos, bins[b]["lng"], bins[b]["lat"]))
            route.append(nxt)
            pos = (bins[nxt]["lng"], bins[nxt]["lat"])
            todo.remove(nxt)
        coords = [self.campus.location("meter-main-inlet")] + [[bins[b]["lng"], bins[b]["lat"]] for b in route]
        dist = sum(haversine_m(*a, *b) for a, b in zip(coords, coords[1:]))
        return {"stops": route, "coords": coords, "distance_m": round(dist)}

    # ---------- alerts ----------
    def _raise(self, kind, rule, asset, title, detail, severity, source, inr_day, co2_day, evidence, sim):
        key = f"{kind}:{rule}:{asset}"
        existing = next((a for a in self.alerts.values()
                         if f"{a.kind}:{a.rule}:{a.asset}" == key and a.status in ("open", "assigned")), None)
        stamp = sim.isoformat(timespec="minutes")
        if existing:
            existing.updated, existing.detail, existing.evidence = stamp, detail, evidence
            existing.inr_per_day, existing.co2_kg_per_day = round(inr_day), round(co2_day, 1)
            if severity == "crit":
                existing.severity = "crit"
            return existing
        lng, lat = self.campus.location(asset.split("/")[0])
        a = Alert(id=f"A{next(self._ids):04d}", kind=kind, rule=rule, asset=asset, title=title, detail=detail,
                  severity=severity, source=source, created=stamp, updated=stamp, lng=lng, lat=lat,
                  inr_per_day=round(inr_day), co2_kg_per_day=round(co2_day, 1), evidence=evidence)
        self.alerts[a.id] = a
        self.events.append({"kind": kind, "asset": asset, "lng": lng, "lat": lat, "at": stamp, "rule": rule})
        return a

    def _clear(self, kind, rule, asset):
        for a in self.alerts.values():
            if a.kind == kind and a.rule == rule and a.asset == asset and a.status == "open" and a.source == "sensor-ml":
                a.status = "self-cleared"

    def _detect(self, sim, state):
        # Building energy: robust z-score vs hour-of-week baseline, confirmed by Isolation Forest.
        for b, s in state["buildings"].items():
            ts = s["_ts"]
            z = self.base[(b, "elec")].zscore(ts, s["power_kw"])
            excess = s["power_kw"] - s["baseline_kw"]
            after_hours = sim.hour >= 21 or sim.hour < 6 or sim.weekday() == 6
            if z > 3.5 and excess > max(2, 0.15 * s["baseline_kw"]) and self.iforest[b].is_anomaly(ts, s["power_kw"]):
                kwh_day = excess * (10 if after_hours else 6)
                self._raise("energy", "load-spike", b,
                            f"{'After-hours' if after_hours else 'Abnormal'} power draw in {s['name']}",
                            f"{s['power_kw']:.0f} kW vs {s['baseline_kw']:.0f} kW expected (z={z:.1f}).",
                            "crit" if z > 6 else "warn", "sensor-ml",
                            kwh_day * C.TARIFF_INR_PER_KWH, kwh_day * C.GRID_KG_CO2_PER_KWH,
                            {"z": round(z, 1), "iforest": True, "kw": round(s["power_kw"], 1)}, sim)
            elif z < 2:
                self._clear("energy", "load-spike", b)

            # Building water: night-time flow far above the usual night floor => tap left running / leak.
            wb = self.base[(b, "water")]
            if (sim.hour < 5) and s["water_lph"] > max(wb.expected(ts) * 2.2, wb.night_floor * 3) and s["water_lph"] > 150:
                extra = s["water_lph"] - wb.expected(ts)
                kl_day = extra * 24 / 1000
                self._raise("water", "night-flow", b, f"Continuous night water flow in {s['name']}",
                            f"{s['water_lph']:.0f} L/h at {sim:%H:%M}, normal is {wb.expected(ts):.0f} L/h. Likely leak or running tap.",
                            "crit", "sensor-ml", kl_day * C.WATER_INR_PER_KL, kl_day * C.WATER_KG_CO2_PER_KL,
                            {"lph": round(s["water_lph"]), "expected_lph": round(wb.expected(ts))}, sim)

        # Rooms: AC / lights running with nobody there (camera or occupancy model) or after hours.
        for r, s in state["rooms"].items():
            load = s["ac_kw"] + s["light_kw"]
            after_hours = sim.hour >= 20 or sim.hour < 7 or sim.weekday() == 6
            # Camera evidence wins; without a camera only after-hours use is flagged (the sensor
            # occupancy model is shown for context but too uncertain on its own to raise alerts).
            empty = s["people"] == 0
            if load > 1.0 and (empty or after_hours):
                why = "camera sees nobody" if empty else "after hours"
                kwh_day = load * (12 if after_hours else 4)
                self._raise("energy", "empty-room", r, f"AC/lights on in empty room {r.split('/')[1]}",
                            f"{s['ac_kw']:.1f} kW AC + {s['light_kw']:.1f} kW lighting while {why}.",
                            "warn" if load < 3 else "crit", "camera" if s["people"] == 0 else "sensor-ml",
                            kwh_day * C.TARIFF_INR_PER_KWH, kwh_day * C.GRID_KG_CO2_PER_KWH,
                            {"ac_kw": s["ac_kw"], "light_kw": s["light_kw"], "occ_prob": s["occ_prob"], "people": s["people"]}, sim)
            elif load < 0.5 or not (empty or after_hours):
                self._clear("energy", "empty-room", r)

        # Campus water main: sustained excess inflow (leak detector), localised by pressure drop.
        w = state["water"]
        if w["leak_score"] > 1 or w["scenario_lph"] > 0:
            zone = w["likely_zone"]
            excess = max(w["mnf_excess_lph"], w["scenario_lph"])
            kl_day = excess * 24 / 1000
            self._raise("water", "main-leak", zone, f"Suspected pipe leak near {self.campus.buildings[zone]['name']}",
                        f"Night-time minimum inflow is {excess:.0f} L/h above the 14-night baseline; biggest pressure drop at this zone.",
                        "crit", "demo-scenario" if w["scenario_lph"] else "sensor-ml",
                        kl_day * C.WATER_INR_PER_KL, kl_day * C.WATER_KG_CO2_PER_KL,
                        {"leak_score": round(w["leak_score"], 2), "excess_lph": round(excess)}, sim)

        # Bins: overflow now, or forecast to overflow before the next pickup.
        for b, s in state["bins"].items():
            if s["fill"] >= 95:
                self._raise("waste", "overflow", b, f"{b} is overflowing",
                            f"Fill {s['fill']}% ({s['source']}). Overflowing bins attract litter and pests.",
                            "crit", s["source"] if s["source"] != "model" else "sensor-ml",
                            150, 8 * C.WASTE_KG_CO2_PER_KG, {"fill": s["fill"]}, sim)
            elif s["fill"] < 50:
                self._clear("waste", "overflow", b)

    # ---------- snapshot ----------
    def state(self):
        with self.lock:
            sim = self.clock.now()
            st = {"time": sim.isoformat(timespec="minutes"), "speed": self.clock.speed, "buildings": {}, "rooms": {}}

            row, ts = self._row(self.buildings, sim, 2017, "h")
            for b in self.bld_ids:
                p = self.campus.buildings[b]
                kw, wl = float(row[(b, "elec_kwh")]), float(row[(b, "water_l")])
                for sc in self.scenarios:
                    if sc["kind"] == "leak" and sc["target"] == b:
                        wl += sc["lph"]
                st["buildings"][b] = {"name": p["name"], "type": p["type"], "power_kw": round(kw, 1),
                                      "baseline_kw": round(self.base[(b, "elec")].expected(ts), 1),
                                      "water_lph": round(wl), "water_baseline_lph": round(self.base[(b, "water")].expected(ts)),
                                      "_ts": ts}

            rrow, _ = self._row(self.rooms, sim, 2019, "15min")
            for r in self.room_ids:
                v = {m: float(np.nan_to_num(rrow[(r, m)])) for m in ("ac_kw", "light_kw", "plug_kw", "temp_c", "rh", "lux")}
                for sc in self.scenarios:
                    if sc["kind"] == "ac_night" and sc["target"] == r:
                        v["ac_kw"] += sc["kw"]
                cam = self.camera_occupancy.get(r)
                people = cam["people"] if cam and (sim - cam["at"]) < timedelta(minutes=30) else None
                v["occ_prob"] = round(self.occupancy.prob(v["temp_c"], v["rh"], v["lux"]), 2)
                v["people"] = people
                st["rooms"][r] = {k: round(x, 2) if isinstance(x, float) else x for k, x in v.items()}

            wrow, wts = self._row(self.water, sim, 2018, "5min")
            scen_lph = sum(sc["lph"] for sc in self.scenarios if sc["kind"] == "leak")
            score = self.leak.score_at(wts) + scen_lph / self.leak.threshold_at(wts)
            drops = {}
            for i, zone in enumerate(PRESSURE_ZONES):
                col = f"p{i + 1}_m"
                if col in wrow:
                    drops[zone] = float(self.water[col].loc[wts - pd.Timedelta(days=7)] - wrow[col]) if (wts - pd.Timedelta(days=7)) in self.water.index else 0.0
            leak_targets = [sc["target"] for sc in self.scenarios if sc["kind"] == "leak"]
            likely = leak_targets[-1] if leak_targets else max(drops, key=drops.get) if drops else PRESSURE_ZONES[0]
            st["water"] = {"inlet_lph": round(float(wrow.inlet_lph) + scen_lph), "expected_lph": round(self.leak.expected(wts)),
                           "leak_score": round(score, 2), "likely_zone": likely, "scenario_lph": scen_lph,
                           "mnf_excess_lph": round(self.leak.excess_at(wts)),
                           "pressures": {z: round(float(wrow[f"p{i + 1}_m"]), 2) for i, z in enumerate(PRESSURE_ZONES) if f"p{i + 1}_m" in wrow}}

            st["bins"] = self.bin_state(sim)
            self._detect(sim, st)
            self._check_deadlines(sim)

            for b, s in st["buildings"].items():
                s.pop("_ts")
                s["score"], s["grade"] = self.green_score(b, s)
                open_a = [a for a in self.alerts.values() if a.status in ("open", "assigned") and a.asset.split("/")[0] == b]
                s["status"] = "crit" if any(a.severity == "crit" for a in open_a) else "warn" if open_a else "ok"
                s["alerts"] = [a.id for a in open_a]

            st["kpis"] = self.kpis(sim, st)
            st["alerts"] = [asdict(a) for a in sorted(self.alerts.values(), key=lambda a: a.updated, reverse=True)[:60]]
            st["orders"] = [asdict(o) for o in sorted(self.orders.values(), key=lambda o: o.created, reverse=True)]
            st["scenarios"] = [{"kind": s["kind"], "target": s["target"]} for s in self.scenarios]
            return st

    def green_score(self, b, s):
        score = 100.0
        ratio = s["power_kw"] / max(s["baseline_kw"], 1)
        score -= min(max(ratio - 1, 0) * 60, 30)
        wratio = s["water_lph"] / max(s["water_baseline_lph"], 1)
        score -= min(max(wratio - 1, 0) * 30, 20)
        for a in self.alerts.values():
            if a.status in ("open", "assigned") and a.asset.split("/")[0] == b:
                score -= 15 if a.severity == "crit" else 7
        score = max(round(score), 0)
        grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
        return score, grade

    def kpis(self, sim, st):
        t = pd.Timestamp(to_dataset_time(sim, 2017)).floor("h")
        day = self.buildings.loc[t.normalize(): t]
        kwh = float(day.xs("elec_kwh", axis=1, level=1).sum().sum())
        water_l = float(day.xs("water_l", axis=1, level=1).sum().sum())
        open_a = [a for a in self.alerts.values() if a.status in ("open", "assigned")]
        scores = [s["score"] for s in st["buildings"].values()]
        return {
            "power_kw": round(sum(s["power_kw"] for s in st["buildings"].values())),
            "energy_today_kwh": round(kwh), "cost_today_inr": round(kwh * C.TARIFF_INR_PER_KWH),
            "co2_today_kg": round(kwh * C.GRID_KG_CO2_PER_KWH), "water_today_kl": round(water_l / 1000, 1),
            "open_alerts": len(open_a), "crit_alerts": sum(a.severity == "crit" for a in open_a),
            "waste_inr_per_day": round(sum(a.inr_per_day for a in open_a)),
            "green_score": round(sum(scores) / len(scores)) if scores else 0,
            "bins_attention": sum(1 for b in st["bins"].values() if b["status"] != "ok"),
        }

    # ---------- work orders ----------
    def create_order(self, alert_id, note=""):
        with self.lock:
            a = self.alerts[alert_id]
            if a.work_order:
                return self.orders[a.work_order]
            sim = self.clock.now()
            o = WorkOrder(id=f"WO{next(self._ids):04d}", alert_id=a.id, team=TEAM[a.kind], title=a.title,
                          created=sim.isoformat(timespec="minutes"),
                          deadline=(sim + timedelta(hours=C.DEADLINE_H[a.kind])).isoformat(timespec="minutes"), note=note)
            self.orders[o.id] = o
            a.work_order, a.status = o.id, "assigned"
            return o

    def complete_order(self, order_id, proof):
        """Close an order once its proof photo has passed verification."""
        with self.lock:
            o = self.orders[order_id]
            a = self.alerts[o.alert_id]
            o.status, o.proof = "done", proof
            a.status = "resolved"
            sim = self.clock.now()
            if a.kind == "waste":
                self.bin_collected[a.asset] = sim
                self.bin_obs.pop(a.asset, None)
            self.scenarios = [s for s in self.scenarios
                              if not (s["target"] == a.asset or (a.rule == "main-leak" and s["kind"] == "leak"))]
            if a.rule == "empty-room":
                self.scenarios = [s for s in self.scenarios if s["target"] != a.asset]
            return o

    def _check_deadlines(self, sim):
        for o in self.orders.values():
            if o.status == "assigned" and sim > datetime.fromisoformat(o.deadline):
                o.status = "overdue"

    def resolve(self, alert_id, status="resolved"):
        with self.lock:
            self.alerts[alert_id].status = status

    # ---------- history ----------
    def backfill(self, days=30):
        """Replay the detectors over the last N days to seed history (hotspots, reports)."""
        sim = self.clock.now()
        end = pd.Timestamp(to_dataset_time(sim, 2017)).floor("h")
        start = end - pd.Timedelta(days=days)
        hist = []
        for b in self.bld_ids:
            e = self.buildings[(b, "elec_kwh")].loc[start:end]
            base = self.base[(b, "elec")]
            how = e.index.dayofweek * 24 + e.index.hour
            z = (e.values - base.median[how]) / (1.4826 * base.mad[how])
            excess = e.values - base.median[how]
            lng, lat = self.campus.location(b)
            for ts, zz, ex in zip(e.index, z, excess):
                if zz > 3.5 and ex > 2:
                    at = sim - (end - ts).to_pytimedelta()
                    hist.append({"kind": "energy", "asset": b, "lng": lng, "lat": lat, "at": at.isoformat(timespec="minutes"), "rule": "load-spike"})
        for d in range(days, 0, -1):
            for b in self.campus.of_type("bin"):
                peak = max(self.bin_fill(b, (sim - timedelta(days=d)).replace(hour=h, minute=0))[0] for h in (6, 14, 23))
                if peak >= 95:
                    lng, lat = self.campus.location(b)
                    hist.append({"kind": "waste", "asset": b, "lng": lng, "lat": lat,
                                 "at": (sim - timedelta(days=d)).isoformat(timespec="minutes"), "rule": "overflow"})
        self.events = hist + self.events
        return len(hist)

    def history_summary(self, days=7):
        """Energy, water and cost over the last N simulated days, per building."""
        sim = self.clock.now()
        end = pd.Timestamp(to_dataset_time(sim, 2017)).floor("h")
        win = self.buildings.loc[end - pd.Timedelta(days=days): end]
        prev = self.buildings.loc[end - pd.Timedelta(days=2 * days): end - pd.Timedelta(days=days)]
        out = {}
        for b in self.bld_ids:
            kwh, pk = float(win[(b, "elec_kwh")].sum()), float(prev[(b, "elec_kwh")].sum())
            exp = sum(self.base[(b, "elec")].expected(ts) for ts in win.index)
            out[b] = {"name": self.campus.buildings[b]["name"], "kwh": round(kwh), "expected_kwh": round(exp),
                      "vs_expected_pct": round(100 * (kwh / exp - 1), 1) if exp else 0,
                      "vs_prev_week_pct": round(100 * (kwh / pk - 1), 1) if pk else 0,
                      "water_kl": round(float(win[(b, "water_l")].sum()) / 1000, 1),
                      "night_kwh": round(float(win[(b, "elec_kwh")][(win.index.hour >= 21) | (win.index.hour < 6)].sum()))}
        daily = win.xs("elec_kwh", axis=1, level=1).sum(axis=1).resample("D").sum()
        return {"days": days, "buildings": out, "daily_kwh": [round(float(x)) for x in daily.values][-days:],
                "total_kwh": round(sum(v["kwh"] for v in out.values())),
                "total_water_kl": round(sum(v["water_kl"] for v in out.values()), 1)}

    # ---------- what-if simulator ----------
    def simulate(self, ac_setpoint_delta=0.0, after_hours_cut=0.0, led_retrofit=False, fix_leaks=False,
                 solar_kwp=0.0, pickups_per_day=2):
        """Annualised savings from interventions, computed over a full year of replayed data."""
        elec = self.buildings.xs("elec_kwh", axis=1, level=1)
        total_kwh = float(elec.sum().sum())
        idx = elec.index
        night = (idx.hour >= 21) | (idx.hour < 6) | (idx.dayofweek == 6)
        base_floor = elec.quantile(0.05)
        after_hours_excess = float((elec[night] - base_floor).clip(lower=0).sum().sum())

        # AC and lighting shares of load come from the metered CU-BEMS rooms.
        rooms_all = self.rooms.T.groupby(level=1).sum().T
        tot_r = float(rooms_all[["ac_kw", "light_kw", "plug_kw"]].sum().sum())
        ac_share = float(rooms_all["ac_kw"].sum()) / tot_r if tot_r else 0.5
        light_share = float(rooms_all["light_kw"].sum()) / tot_r if tot_r else 0.15

        saved = {
            "ac_setpoint": total_kwh * ac_share * C.AC_SAVING_PER_DEG * max(ac_setpoint_delta, 0),
            "after_hours": after_hours_excess * after_hours_cut,
            "led_retrofit": total_kwh * light_share * 0.5 if led_retrofit else 0.0,
            "solar": solar_kwp * C.SOLAR_KWH_PER_KWP_DAY * 365,
        }
        kwh_saved = sum(saved.values())
        leak_kl = 0.0
        if fix_leaks:
            open_leaks = [a for a in self.alerts.values() if a.kind == "water" and a.status in ("open", "assigned")]
            leak_kl = sum(a.evidence.get("excess_lph", a.evidence.get("lph", 0)) for a in open_leaks) * 24 * 365 / 1000
            leak_kl = leak_kl or float(self.water.leak_m3h.mean()) * 24 * 365 * 0.2
        bins = len(self.campus.of_type("bin"))
        overflow_hours = max(0.0, 6.0 - 2.5 * (pickups_per_day - 1)) * bins * 365
        result = {
            "kwh_saved_year": round(kwh_saved),
            "inr_saved_year": round(kwh_saved * C.TARIFF_INR_PER_KWH + leak_kl * C.WATER_INR_PER_KL),
            "co2_t_saved_year": round((kwh_saved * C.GRID_KG_CO2_PER_KWH + leak_kl * C.WATER_KG_CO2_PER_KL) / 1000, 1),
            "water_kl_saved_year": round(leak_kl),
            "breakdown_kwh": {k: round(v) for k, v in saved.items()},
            "baseline_kwh_year": round(total_kwh),
            "pct_energy_saved": round(100 * kwh_saved / total_kwh, 1) if total_kwh else 0,
            "bin_overflow_hours_year": round(overflow_hours),
            "assumptions": {"ac_share": round(ac_share, 2), "light_share": round(light_share, 2),
                            "ac_saving_per_deg": C.AC_SAVING_PER_DEG, "tariff": C.TARIFF_INR_PER_KWH,
                            "grid_factor": C.GRID_KG_CO2_PER_KWH, "solar_yield": C.SOLAR_KWH_PER_KWP_DAY},
        }
        return result
