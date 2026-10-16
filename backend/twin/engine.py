"""The DSU digital twin: live state, anomaly detection, alerts and work orders.

Every tick the engine reads the replayed datasets at the current simulated
time, layers on live evidence (camera detections, student reports, injected
demo scenarios) and turns rule + ML detections into alerts with rupee and CO2
impact. Alerts become facilities work orders that close only with proof.
"""
import itertools
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config as C
from twin.clock import SimClock, to_dataset_time
from twin.geo import Campus, haversine_m
from twin.models import Baselines, EnergyIForest, LeakDetector, LoadForecaster, OccupancyModel, save_metrics

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
    status: str = "open"   # open | assigned | resolved | self-cleared | dismissed | rejected
    evidence: dict = field(default_factory=dict)
    work_order: str | None = None
    history: list = field(default_factory=list)     # incident timeline: [{at, wall, event, by, detail}]
    feedback: dict | None = None                    # operator label: {verdict, reason, by, at}


@dataclass
class WorkOrder:
    id: str
    alert_id: str
    team: str
    title: str
    created: str
    deadline: str
    status: str = "assigned"   # assigned | acknowledged | in_progress | awaiting_verification | done | reopened | cancelled
    proof: dict | None = None
    note: str = ""
    overdue: bool = False
    history: list = field(default_factory=list)
    comments: list = field(default_factory=list)


# Allowed work-order moves (server-side). Proof upload handles awaiting_verification -> done itself.
ORDER_FLOW = {
    "assigned": {"acknowledged", "in_progress", "cancelled"},
    "acknowledged": {"in_progress", "cancelled"},
    "in_progress": {"awaiting_verification", "cancelled"},
    "awaiting_verification": {"in_progress", "done"},
    "reopened": {"acknowledged", "in_progress", "cancelled"},
    "done": {"reopened"},
    "cancelled": set(),
}
CLOSED = ("resolved", "self-cleared", "dismissed", "rejected")
FEEDBACK = {"correct", "false_positive", "wrong_category", "duplicate", "insufficient_evidence", "uncertain"}


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

        mapping = json.loads((C.CACHE / "mapping.json").read_text(encoding="utf-8"))
        if not self._load_models():
            self.base = {}
            self.iforest = {}
            self.forecaster = {}
            for b in self.bld_ids:
                e = self.buildings[(b, "elec_kwh")]
                self.base[(b, "elec")] = Baselines(e)
                self.base[(b, "water")] = Baselines(self.buildings[(b, "water_l")])
                self.iforest[b] = EnergyIForest(e)
                self.forecaster[b] = LoadForecaster(e)
            self.occupancy = OccupancyModel()
            self.leak = LeakDetector(self.water, mapping["water"].get("leak_onsets", {}).values())
            self._save_models()
        fc = list(self.forecaster.values())
        self.metrics = {"occupancy": self.occupancy.metrics, "leak": self.leak.metrics,
                        "forecast": {"dataset": "BDG2 hourly meters (rescaled), test Nov-Dec",
                                     "method": "gradient-boosted trees, day-ahead (features >= 24 h old)",
                                     "buildings": len(fc),
                                     "mape_pct": round(float(np.mean([f.mape for f in fc])), 1),
                                     "naive_mape_pct": round(float(np.mean([f.mape_naive for f in fc])), 1)}}
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
        # Water tanks: simulated level + float-switch pump, driven by the fed buildings' demand.
        self.tanks = {t: {"level_kl": 0.6 * cfg["capacity_kl"], "pump": False, "at": None, "overflow_lph": 0.0,
                          "inflow_lph": 0.0, "demand_lph": 0.0, "peak_pump_min": 0.0, "day": None}
                      for t, cfg in C.TANKS.items() if t in self.campus.features}
        self._demand_memo = {}
        # Live IoT readings (ESP32 etc.) override the replayed data while fresh.
        self.live: dict[tuple, dict] = {}
        self.sensor_health: dict[str, dict] = {}
        self._snap, self._snap_t = None, 0.0

    # ---------- trained-model cache (fast start on small cloud machines) ----------
    MODELS_VERSION = 3

    def _models_key(self):
        files = ["buildings.pkl", "rooms.pkl", "water.pkl", "occupancy.pkl", "mapping.json"]
        return [self.MODELS_VERSION] + [(f, (C.CACHE / f).stat().st_size) for f in files]

    def _load_models(self):
        import pickle
        path = C.CACHE / "models.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as fh:
                bundle = pickle.load(fh)
            if bundle.get("key") != self._models_key():
                return False
            self.base, self.iforest, self.forecaster = bundle["base"], bundle["iforest"], bundle["forecaster"]
            self.occupancy, self.leak = bundle["occupancy"], bundle["leak"]
            self.leak.water = self.water
            for b, f in self.forecaster.items():   # forecasters read the live (rescaled) series
                f.s = self.buildings[(b, "elec_kwh")]
            print("Loaded trained models from cache/models.pkl")
            return True
        except Exception as e:  # version mismatch etc.: retrain
            print("models.pkl not usable, retraining:", e)
            return False

    def _save_models(self):
        import pickle
        try:
            leak_water, self.leak.water = self.leak.water, None
            fc_series = {b: f.s for b, f in self.forecaster.items()}
            for f in self.forecaster.values():
                f.s = None
            with open(C.CACHE / "models.pkl", "wb") as fh:
                pickle.dump({"key": self._models_key(), "base": self.base, "iforest": self.iforest,
                             "forecaster": self.forecaster, "occupancy": self.occupancy, "leak": self.leak}, fh)
        except Exception as e:
            print("Could not save models.pkl:", e)
        finally:
            self.leak.water = leak_water
            for b, f in self.forecaster.items():
                f.s = fc_series[b]

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
        fc = self.forecaster[b].forecast(t, 24)
        peak_i = int(np.argmax(fc))
        return {"t": stamps, "kwh": s["elec_kwh"].round(1).tolist(), "baseline": [round(x, 1) for x in base],
                "water_l": s["water_l"].round(0).tolist(),
                "t_fc": [(sim + timedelta(hours=i + 1)).strftime("%H:00") for i in range(len(fc))],
                "forecast": [round(float(x), 1) for x in fc],
                "forecast_kwh_24h": round(float(fc.sum())),
                "forecast_peak_kw": round(float(fc[peak_i]), 1),
                "forecast_peak_at": (sim + timedelta(hours=peak_i + 1)).strftime("%a %H:00"),
                "forecast_mape_pct": round(self.forecaster[b].mape, 1)}

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
            elif kind == "tank_overflow":
                target = target or next(iter(self.tanks), None)
                self.scenarios.append({"kind": "tank_overflow", "target": target, "start": sim})
                if target in self.tanks:  # start near the top so the overflow shows within a minute
                    t = self.tanks[target]
                    t["level_kl"] = max(t["level_kl"], 0.985 * C.TANKS[target]["capacity_kl"])
            elif kind == "night":
                self.clock.jump_to(sim.replace(hour=2, minute=0) + timedelta(days=1 if sim.hour >= 2 else 0))
            elif kind == "morning":
                self.clock.jump_to(sim.replace(hour=9, minute=0) + timedelta(days=1 if sim.hour >= 9 else 0))
            return {"kind": kind, "target": target}

    def clear_demo(self):
        """Remove injected scenarios and close the alerts they raised, so a demo can restart cleanly."""
        with self.lock:
            self.scenarios.clear()
            for b in [b for b, o in self.bin_obs.items() if o["source"] == "demo-scenario"]:
                self.bin_obs.pop(b)
                self.bin_collected[b] = self.clock.now()
            n = 0
            for a in self.alerts.values():
                if a.source == "demo-scenario" and a.status in ("open", "assigned"):
                    a.status, n = "self-cleared", n + 1
                    self.log(a, "self-cleared", "operator", "demo scenario cleared")
                    if a.work_order and self.orders[a.work_order].status != "done":
                        self.orders[a.work_order].status = "cancelled"
                        self.log(self.orders[a.work_order], "cancelled", "operator", "demo scenario cleared")
            self._snap = None
            return n

    # ---------- live IoT ingest ----------
    LIVE_FRESH_S = 120

    def ingest(self, sensor, asset, metric, value):
        """One reading from a real sensor. Building power/water and tank level override the replay while fresh."""
        value = float(value)
        health = self.sensor_health.setdefault(sensor, {"sensor": sensor, "readings": 0, "rejected": 0, "last_error": None,
                                                        "first_seen": time.time()})
        lo, hi = C.SENSOR_RANGES.get(metric, (float("-inf"), float("inf")))
        if not (lo <= value <= hi) or value != value:
            health["rejected"] += 1
            health["last_error"] = f"{metric}={value} outside {lo}-{hi}"
            raise ValueError(f"{metric} reading {value} is outside the plausible range {lo}-{hi}")
        with self.lock:
            if metric == "bin_fill":
                self.observe_bin(asset, max(0.0, min(value, 100.0)), "iot")
            elif metric == "people":
                self.observe_people(asset, int(value))
            elif metric in ("power_kw", "water_lph"):
                if asset not in self.bld_ids:
                    raise KeyError(f"unknown building {asset}")
            elif metric == "tank_level_pct":
                if asset not in self.tanks:
                    raise KeyError(f"unknown tank {asset}")
            else:
                raise KeyError(f"unknown metric {metric}")
            self.live[(asset, metric)] = {"sensor": sensor, "value": value, "t": time.time()}
            health.update(readings=health["readings"] + 1, last=time.time(), asset=asset, metric=metric, value=value)
            return {"ok": True, "asset": asset, "metric": metric, "value": value}

    def _live(self, asset, metric):
        r = self.live.get((asset, metric))
        return r["value"] if r and time.time() - r["t"] < self.LIVE_FRESH_S else None

    def sensors_status(self):
        """Sensor health: a sensor that stops reporting is shown as stale/offline, never as healthy."""
        now = time.time()
        out = []
        for h in self.sensor_health.values():
            age = now - h["last"] if h.get("last") else None
            status = "offline" if age is None or age > C.SENSOR_STALE_S else "stale" if age > C.SENSOR_FRESH_S else "fresh"
            total = h["readings"] + h["rejected"]
            out.append({**{k: v for k, v in h.items() if k != "first_seen"}, "age_s": round(age) if age is not None else None,
                        "status": status, "error_rate": round(h["rejected"] / total, 3) if total else 0.0})
        return sorted(out, key=lambda x: x["sensor"])

    def live_sensors(self):
        now = time.time()
        return [{"asset": a, "metric": m, "sensor": r["sensor"], "value": r["value"], "age_s": round(now - r["t"])}
                for (a, m), r in self.live.items() if now - r["t"] < 3600]

    # ---------- water tanks ----------
    @staticmethod
    def is_peak(hour):
        return any(a <= hour < b for a, b in C.TOD_PEAK_HOURS)

    def _tank_demand(self, tid, when):
        key = (tid, when.replace(minute=0, second=0, microsecond=0))
        if key not in self._demand_memo:
            if len(self._demand_memo) > 2000:
                self._demand_memo.clear()
            row, _ = self._row(self.buildings, when, 2017, "h")
            self._demand_memo[key] = sum(float(row[(b, "water_l")]) for b in C.TANKS[tid]["feeds"] if b in self.bld_ids)
        extra = sum(sc["lph"] for sc in self.scenarios if sc["kind"] == "leak" and sc["target"] in C.TANKS[tid]["feeds"])
        return self._demand_memo[key] + extra

    def _step_tanks(self, sim):
        for tid, t in self.tanks.items():
            cfg = C.TANKS[tid]
            cap = cfg["capacity_kl"]
            if t["at"] is None or sim < t["at"] or sim - t["at"] > timedelta(hours=48):
                t.update(at=sim, day=sim.date(), peak_pump_min=0.0, overflow_lph=0.0)
                t["demand_lph"] = self._tank_demand(tid, sim)
                continue
            stuck = any(sc["kind"] == "tank_overflow" and sc["target"] == tid for sc in self.scenarios)
            while t["at"] < sim:
                dt = min(timedelta(minutes=10), sim - t["at"])
                hrs = dt.total_seconds() / 3600
                demand = self._tank_demand(tid, t["at"])
                pct = 100 * t["level_kl"] / cap
                if stuck:
                    t["pump"] = True           # float switch failed: pump never stops
                elif pct < C.PUMP_ON_PCT:
                    t["pump"] = True
                elif pct > C.PUMP_OFF_PCT:
                    t["pump"] = False
                inflow = cfg["pump_lph"] if t["pump"] else 0.0
                level = t["level_kl"] + (inflow - demand) * hrs / 1000
                overflow = 0.0
                if level > cap:
                    overflow = (level - cap) * 1000 / hrs if hrs else 0.0
                    level = cap
                if t["at"].date() != t["day"]:
                    t["day"], t["peak_pump_min"] = t["at"].date(), 0.0
                if t["pump"] and self.is_peak(t["at"].hour):
                    t["peak_pump_min"] += hrs * 60
                t.update(level_kl=max(level, 0.0), overflow_lph=overflow, inflow_lph=inflow, demand_lph=demand)
                t["at"] += dt
            lv = self._live(tid, "tank_level_pct")
            if lv is not None:
                t["level_kl"] = cap * max(0.0, min(lv, 100.0)) / 100

    def tank_state(self):
        out = {}
        for tid, t in self.tanks.items():
            cfg = C.TANKS[tid]
            lng, lat = self.campus.location(tid)
            pct = 100 * t["level_kl"] / cfg["capacity_kl"]
            out[tid] = {"name": self.campus.features[tid]["properties"]["name"], "level_pct": round(pct),
                        "level_kl": round(t["level_kl"], 1), "capacity_kl": cfg["capacity_kl"], "pump_on": t["pump"],
                        "inflow_lph": round(t["inflow_lph"]), "demand_lph": round(t["demand_lph"]),
                        "overflow_lph": round(t["overflow_lph"]), "peak_pump_min_today": round(t["peak_pump_min"]),
                        "feeds": cfg["feeds"], "live": self._live(tid, "tank_level_pct") is not None,
                        "lng": lng, "lat": lat,
                        "status": "crit" if t["overflow_lph"] > 0 else "warn" if pct < 15 or pct > 98 else "ok"}
        return out

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
        depot = list(self.campus.location("meter-main-inlet"))
        loc = {b: [bins[b]["lng"], bins[b]["lat"]] for b in route}

        def length(order):
            pts = [depot] + [loc[b] for b in order]
            return sum(haversine_m(*a, *b) for a, b in zip(pts, pts[1:]))

        greedy = length(route)
        improved = True
        while improved:  # 2-opt: reverse any segment that shortens the walk
            improved = False
            for i in range(len(route) - 1):
                for k in range(i + 1, len(route)):
                    cand = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
                    if length(cand) < length(route) - 0.5:
                        route, improved = cand, True
        coords = [depot] + [loc[b] for b in route]
        dist = length(route)
        return {"stops": route, "coords": coords, "distance_m": round(dist), "greedy_m": round(greedy),
                "method": "nearest neighbour + 2-opt"}

    # ---------- incident timeline ----------
    def log(self, item, event, by="system", detail=""):
        entry = {"at": self.clock.now().isoformat(timespec="minutes"), "wall": datetime.now().isoformat(timespec="seconds"),
                 "event": event, "by": by, "detail": str(detail)[:300]}
        item.history.append(entry)
        del item.history[:-40]
        return entry

    # ---------- alerts ----------
    def _raise(self, kind, rule, asset, title, detail, severity, source, inr_day, co2_day, evidence, sim):
        key = f"{kind}:{rule}:{asset}"
        existing = next((a for a in self.alerts.values()
                         if f"{a.kind}:{a.rule}:{a.asset}" == key and a.status in ("open", "assigned")), None)
        stamp = sim.isoformat(timespec="minutes")
        if existing:
            existing.updated, existing.detail, existing.evidence = stamp, detail, evidence
            existing.inr_per_day, existing.co2_kg_per_day = round(inr_day), round(co2_day, 1)
            if severity == "crit" and existing.severity != "crit":
                existing.severity = "crit"
                self.log(existing, "escalated", "system", "severity raised to critical")
            return existing
        lng, lat = self.campus.location(asset.split("/")[0])
        a = Alert(id=f"A{next(self._ids):04d}", kind=kind, rule=rule, asset=asset, title=title, detail=detail,
                  severity=severity, source=source, created=stamp, updated=stamp, lng=lng, lat=lat,
                  inr_per_day=round(inr_day), co2_kg_per_day=round(co2_day, 1), evidence=evidence)
        self.alerts[a.id] = a
        self.log(a, "detected", source, detail)
        self.events.append({"kind": kind, "asset": asset, "lng": lng, "lat": lat, "at": stamp, "rule": rule})
        return a

    def _clear(self, kind, rule, asset=None):
        for a in self.alerts.values():
            if a.kind == kind and a.rule == rule and (asset is None or a.asset == asset) \
                    and a.status == "open" and a.source == "sensor-ml":
                a.status = "self-cleared"
                self.log(a, "self-cleared", "system", "condition back to normal")

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
            elif sim.hour >= 6 and s["water_lph"] < wb.expected(ts) * 1.5:
                self._clear("water", "night-flow", b)

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
                        f"Night-time minimum inflow is {excess:.0f} L/h above the 7-night baseline; biggest pressure drop at this zone.",
                        "crit", "demo-scenario" if w["scenario_lph"] else "sensor-ml",
                        kl_day * C.WATER_INR_PER_KL, kl_day * C.WATER_KG_CO2_PER_KL,
                        {"leak_score": round(w["leak_score"], 2), "excess_lph": round(excess)}, sim)
        elif w["leak_score"] < 0.7:
            self._clear("water", "main-leak")

        # Water tanks: overflow (float switch failure) and pumping in peak-tariff hours.
        for tid, t in state["tanks"].items():
            cfg = C.TANKS[tid]
            stuck = any(sc["kind"] == "tank_overflow" and sc["target"] == tid for sc in self.scenarios)
            if t["overflow_lph"] > 0:
                kl_day = t["overflow_lph"] * 24 / 1000
                kwh_day = cfg["pump_kw"] * 24
                self._raise("water", "tank-overflow", tid, f"{t['name']} is overflowing",
                            f"Pump still running at {t['level_pct']}% full: {t['overflow_lph']:,} L/h spilling. "
                            "Likely a failed float switch; switch the pump off and replace the float valve.",
                            "crit", "demo-scenario" if stuck else "sensor-ml",
                            kl_day * C.WATER_INR_PER_KL + kwh_day * C.TARIFF_INR_PER_KWH,
                            kl_day * C.WATER_KG_CO2_PER_KL + kwh_day * C.GRID_KG_CO2_PER_KWH,
                            {"level_pct": t["level_pct"], "overflow_lph": t["overflow_lph"]}, sim)
            elif not stuck:
                self._clear("water", "tank-overflow", tid)
            peak_h = sum(b - a for a, b in C.TOD_PEAK_HOURS)
            if t["pump_on"] and self.is_peak(sim.hour) and t["peak_pump_min_today"] >= 30:
                kwh = cfg["pump_kw"] * t["peak_pump_min_today"] / 60
                self._raise("resource", "pump-peak", tid, f"{t['name']} pump running in peak-tariff hours",
                            f"{t['peak_pump_min_today']} min of pumping in peak windows today ({kwh:.0f} kWh). "
                            f"Schedule refills for 22:00-06:00 and keep a buffer: same water, lower tariff.",
                            "warn", "sensor-ml", kwh * (C.TOD_PEAK_SURCHARGE + C.TOD_NIGHT_REBATE),
                            0.0, {"peak_pump_min": t["peak_pump_min_today"], "peak_hours": peak_h}, sim)
            elif not self.is_peak(sim.hour):
                self._clear("resource", "pump-peak", tid)

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
                live_kw, live_wl = self._live(b, "power_kw"), self._live(b, "water_lph")
                kw = live_kw if live_kw is not None else kw
                wl = live_wl if live_wl is not None else wl
                for sc in self.scenarios:
                    if sc["kind"] == "leak" and sc["target"] == b:
                        wl += sc["lph"]
                st["buildings"][b] = {"name": p["name"], "type": p["type"], "power_kw": round(kw, 1),
                                      "baseline_kw": round(self.base[(b, "elec")].expected(ts), 1),
                                      "water_lph": round(wl), "water_baseline_lph": round(self.base[(b, "water")].expected(ts)),
                                      "live": live_kw is not None or live_wl is not None,
                                      "source": "measured" if live_kw is not None else "replay",
                                      "stale_sensor": live_kw is None and (b, "power_kw") in self.live, "_ts": ts}

            rrow, _ = self._row(self.rooms, sim, 2019, "15min")
            occ_in = [[float(np.nan_to_num(rrow[(r, m)])) for m in ("temp_c", "rh", "lux")] for r in self.room_ids]
            occ = dict(zip(self.room_ids, self.occupancy.probs(occ_in)))
            for r in self.room_ids:
                v = {m: float(np.nan_to_num(rrow[(r, m)])) for m in ("ac_kw", "light_kw", "plug_kw", "temp_c", "rh", "lux")}
                for sc in self.scenarios:
                    if sc["kind"] == "ac_night" and sc["target"] == r:
                        v["ac_kw"] += sc["kw"]
                cam = self.camera_occupancy.get(r)
                people = cam["people"] if cam and (sim - cam["at"]) < timedelta(minutes=30) else None
                v["occ_prob"] = round(occ[r], 2)
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
            self._step_tanks(sim)
            st["tanks"] = self.tank_state()
            h = sim.hour
            st["tariff"] = {"band": "peak" if self.is_peak(h) else "night" if (h >= 22 or h < 6) else "normal",
                            "inr_per_kwh": C.TARIFF_INR_PER_KWH + (C.TOD_PEAK_SURCHARGE if self.is_peak(h) else
                                                                   -C.TOD_NIGHT_REBATE if (h >= 22 or h < 6) else 0)}
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
            st["live_sensors"] = len(self.live_sensors())
            st["generated_at"] = datetime.now().isoformat(timespec="seconds")
            st["data_mode"] = {"meters": "dataset replay", "rooms": "dataset replay", "water_network": "dataset replay",
                               "tanks": "simulated", "bins": "simulated", "money_co2": "estimated", "clock_speed": self.clock.speed}
            return st

    def snapshot_json(self, max_age=None):
        """Serialized state shared by every WebSocket client, recomputed at most once per `max_age` seconds."""
        with self.lock:
            now = time.monotonic()
            max_age = C.WS_INTERVAL_S * 0.8 if max_age is None else max_age
            if self._snap is None or now - self._snap_t >= max_age:
                self._snap, self._snap_t = json.dumps(self.state(), default=str), now
            return self._snap

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
            "tariff_band": st.get("tariff", {}).get("band", "normal"),
        }

    # ---------- work orders ----------
    def create_order(self, alert_id, note="", by="operator"):
        with self.lock:
            a = self.alerts[alert_id]
            if a.status in CLOSED:
                raise ValueError(f"{a.id} is already {a.status}; reopen it first.")
            if a.work_order and self.orders[a.work_order].status != "cancelled":
                return self.orders[a.work_order]   # idempotent: one live order per incident
            sim = self.clock.now()
            o = WorkOrder(id=f"WO{next(self._ids):04d}", alert_id=a.id, team=TEAM[a.kind], title=a.title,
                          created=sim.isoformat(timespec="minutes"),
                          deadline=(sim + timedelta(hours=C.DEADLINE_H[a.kind])).isoformat(timespec="minutes"), note=note)
            self.orders[o.id] = o
            a.work_order, a.status = o.id, "assigned"
            self.log(o, "assigned", by, f"to {o.team}, due {o.deadline.replace('T', ' ')}" + (f": {note}" if note else ""))
            self.log(a, "assigned", by, f"{o.id} to {o.team}, due {o.deadline.replace('T', ' ')}")
            return o

    def set_order_status(self, order_id, status, by="operator", note=""):
        with self.lock:
            o = self.orders[order_id]
            if status not in ORDER_FLOW.get(o.status, set()):
                raise ValueError(f"Can't move {o.id} from {o.status} to {status}.")
            if status == "done":
                raise ValueError("Upload an after-fix photo to close the order; it is verified before closing.")
            a = self.alerts.get(o.alert_id)
            prev, o.status = o.status, status
            self.log(o, status.replace("_", " "), by, note or f"from {prev.replace('_', ' ')}")
            if note:
                o.comments.append({"at": datetime.now().isoformat(timespec="seconds"), "by": by, "text": note[:300]})
            if a:
                if status == "cancelled" and a.status == "assigned":
                    a.status = "open"
                    self.log(a, "order cancelled", by, note or o.id)
                elif status == "reopened":
                    a.status = "assigned"
                    self.log(a, "reopened", by, note or f"{o.id} reopened")
                elif status in ("acknowledged", "in_progress"):
                    self.log(a, f"work {status.replace('_', ' ')}", by, o.id)
            return o

    def comment_order(self, order_id, text, by="operator"):
        with self.lock:
            o = self.orders[order_id]
            o.comments.append({"at": datetime.now().isoformat(timespec="seconds"), "by": by, "text": text[:300]})
            self.log(o, "comment", by, text)
            return o

    def proof_submitted(self, order_id, by="technician"):
        with self.lock:
            o = self.orders[order_id]
            if o.status in ("done", "cancelled"):
                raise ValueError(f"{o.id} is {o.status}.")
            o.status = "awaiting_verification"
            self.log(o, "proof uploaded", by, "after-fix photo sent for AI verification")

    def proof_failed(self, order_id, check):
        with self.lock:
            o = self.orders[order_id]
            o.status = "in_progress"
            self.log(o, "verification failed", "AI", f"issue still visible (model score {check.get('issue_score')})")
            if o.alert_id in self.alerts:
                self.log(self.alerts[o.alert_id], "fix not verified", "AI", f"{o.id}: photo still shows the problem")

    def feedback(self, alert_id, verdict, reason="", by="operator", new_kind=None):
        """Operator label on a detection. False positives and duplicates close the incident."""
        if verdict not in FEEDBACK:
            raise ValueError(f"verdict must be one of {sorted(FEEDBACK)}")
        with self.lock:
            a = self.alerts[alert_id]
            a.feedback = {"verdict": verdict, "reason": reason[:300], "by": by, "at": datetime.now().isoformat(timespec="seconds"),
                          "source": a.source}
            self.log(a, f"labelled {verdict.replace('_', ' ')}", by, reason)
            if verdict in ("false_positive", "duplicate") and a.status not in CLOSED:
                a.status = "rejected"
                if a.work_order and self.orders[a.work_order].status not in ("done", "cancelled"):
                    self.orders[a.work_order].status = "cancelled"
                    self.log(self.orders[a.work_order], "cancelled", by, f"incident rejected as {verdict.replace('_', ' ')}")
            if verdict == "wrong_category" and new_kind in TEAM:
                a.kind = new_kind
                if a.work_order:
                    self.orders[a.work_order].team = TEAM[new_kind]
            return a

    def dismiss(self, alert_id, by="operator", reason=""):
        with self.lock:
            a = self.alerts[alert_id]
            if a.status in CLOSED:
                return a
            a.status = "dismissed"
            self.log(a, "dismissed", by, reason or "dismissed by operator")
            return a

    def feedback_metrics(self):
        """Operator-labelled precision per detection source (only labelled incidents count)."""
        out = {}
        for a in self.alerts.values():
            if not a.feedback:
                continue
            m = out.setdefault(a.feedback.get("source", a.source), {"labelled": 0, "correct": 0, "false_positive": 0, "other": 0})
            m["labelled"] += 1
            v = a.feedback["verdict"]
            m["correct" if v == "correct" else "false_positive" if v == "false_positive" else "other"] += 1
        for m in out.values():
            judged = m["correct"] + m["false_positive"]
            m["precision"] = round(m["correct"] / judged, 2) if judged else None
        return out

    def ledger(self):
        """Impact of fixed incidents. Savings are ESTIMATED from the detected loss rate; measured savings need meters."""
        rows = []
        for o in self.orders.values():
            if o.status != "done" or o.alert_id not in self.alerts:
                continue
            a = self.alerts[o.alert_id]
            closed = next((h["at"] for h in reversed(o.history) if h["event"] in ("done", "verified")), None)
            hours = None
            try:
                hours = round((datetime.fromisoformat(closed) - datetime.fromisoformat(a.created)).total_seconds() / 3600, 1)
            except (TypeError, ValueError):
                pass
            rows.append({"incident": a.id, "order": o.id, "title": a.title, "kind": a.kind, "team": o.team,
                         "detected": a.created, "fixed": closed, "hours_to_fix": hours,
                         "loss_rate_inr_per_day": a.inr_per_day, "co2_kg_per_day": a.co2_kg_per_day,
                         "est_avoided_inr_per_year": round(a.inr_per_day * 365),
                         "est_avoided_co2_t_per_year": round(a.co2_kg_per_day * 365 / 1000, 2),
                         "measured_savings": None, "method": "estimated: detected loss rate x 365 days, assumes the fix holds",
                         "verification": (o.proof or {}).get("issue_score")})
        return sorted(rows, key=lambda r: r["fixed"] or "", reverse=True)

    def complete_order(self, order_id, proof):
        """Close an order once its proof photo has passed verification."""
        with self.lock:
            o = self.orders[order_id]
            a = self.alerts[o.alert_id]
            o.status, o.proof = "done", proof
            a.status = "resolved"
            self.log(o, "verified", "AI", f"after-fix photo passed (model score {proof.get('issue_score')})")
            self.log(o, "done", "AI", "closed after verification")
            self.log(a, "resolved", "AI", f"{o.id} fix verified from photo")
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
            if o.status not in ("done", "cancelled") and not o.overdue and sim > datetime.fromisoformat(o.deadline):
                o.overdue = True   # escalate once
                self.log(o, "overdue", "system", "deadline passed: escalated to the facilities supervisor")
                if o.alert_id in self.alerts:
                    self.log(self.alerts[o.alert_id], "escalated", "system", f"{o.id} overdue, supervisor notified")

    def resolve(self, alert_id, status="resolved", by="system", detail=""):
        with self.lock:
            a = self.alerts[alert_id]
            a.status = status
            self.log(a, status, by, detail)

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
                 solar_kwp=0.0, pickups_per_day=2, smart_pump_schedule=False):
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
        open_leaks = [a for a in self.alerts.values() if a.kind == "water" and a.status in ("open", "assigned")]
        if fix_leaks:
            leak_kl = sum(a.evidence.get("excess_lph", a.evidence.get("lph", 0)) for a in open_leaks) * 24 * 365 / 1000
            leak_kl = leak_kl or float(self.water.leak_m3h.mean()) * 24 * 365 * 0.2
        bins = len(self.campus.of_type("bin"))
        overflow_hours = max(0.0, 6.0 - 2.5 * (pickups_per_day - 1)) * bins * 365

        # Smart pump schedule: same water, pumped 22:00-06:00 instead of in peak windows (Rs only, no kWh).
        pump_inr, pump_peak_kwh = 0.0, 0.0
        water = self.buildings.xs("water_l", axis=1, level=1)
        peak_mask = np.zeros(len(water), dtype=bool)
        for a, b in C.TOD_PEAK_HOURS:
            peak_mask |= (water.index.hour >= a) & (water.index.hour < b)
        for tid in self.tanks:
            cfg = C.TANKS[tid]
            fed = [b for b in cfg["feeds"] if b in water.columns]
            litres = float(water[fed].sum().sum())
            peak_share = float(water[fed][peak_mask].sum().sum()) / litres if litres else 0.33
            pump_peak_kwh += litres / cfg["pump_lph"] * cfg["pump_kw"] * peak_share
        if smart_pump_schedule:
            pump_inr = pump_peak_kwh * (C.TOD_PEAK_SURCHARGE + C.TOD_NIGHT_REBATE)

        inr = {k: v * C.TARIFF_INR_PER_KWH for k, v in saved.items()}
        inr["fix_leaks"] = leak_kl * C.WATER_INR_PER_KL
        inr["smart_pump"] = pump_inr
        n_bld = len(self.bld_ids)
        capex = {
            "ac_setpoint": C.CAPEX["ac_setpoint"] if ac_setpoint_delta > 0 else 0,
            "after_hours": C.CAPEX["after_hours_per_building"] * n_bld * min(max(after_hours_cut, 0), 1),
            "led_retrofit": saved["led_retrofit"] / C.LIGHTING_HOURS_YEAR * 1000 * C.CAPEX["led_per_w_saved"],
            "solar": solar_kwp * C.CAPEX["solar_per_kwp"],
            "fix_leaks": C.CAPEX["leak_fix"] * max(1, len(open_leaks)) if fix_leaks else 0,
            "smart_pump": C.CAPEX["smart_pump"] * len(self.tanks) if smart_pump_schedule else 0,
        }
        opex_year = C.CAPEX["bin_pickup_per_year"] * (pickups_per_day - 2)
        payback = {k: (round(capex[k] / inr[k], 1) if inr[k] > 0 and capex[k] > 0 else (0.0 if inr[k] > 0 else None))
                   for k in capex}
        total_inr = sum(inr.values())
        total_capex = sum(capex.values())
        result = {
            "kwh_saved_year": round(kwh_saved),
            "inr_saved_year": round(total_inr),
            "inr_breakdown": {k: round(v) for k, v in inr.items()},
            "capex_inr": {k: round(v) for k, v in capex.items()},
            "capex_total_inr": round(total_capex),
            "payback_years": payback,
            "payback_total_years": round(total_capex / total_inr, 1) if total_inr > 0 else None,
            "bin_opex_inr_year": round(opex_year),
            "pump_peak_kwh_year": round(pump_peak_kwh),
            "co2_tonnes_saved_year": round((kwh_saved * C.GRID_KG_CO2_PER_KWH + leak_kl * C.WATER_KG_CO2_PER_KL) / 1000, 1),
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
