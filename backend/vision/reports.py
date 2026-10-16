"""Problem reports from students, staff and maintainers: people act as sensors for the twin.

A report has a category (waste, water, energy, other), a location and an optional photo.

With a photo, checks run in order and stop at the first failure:
  1. location is on campus (inside a drawn zone or within 80 m of one)
  2. not a duplicate photo (perceptual hash, Hamming distance <= 6)
  3. AI sees the problem (photo matches the chosen category, or any problem if none chosen)
  4. not a repeat of an open report nearby (image similarity within 60 m) -> adds a confirmation
A verified report becomes a twin alert with source "student".

Without a photo (or when the AI can't confirm it), the report is filed as "unverified":
it still reaches facilities, at lower priority, and is promoted to verified once
3 people report the same category within 60 m.
"""
import itertools
import time

import numpy as np

from twin.geo import haversine_m
from vision.detector import VisionUnavailable, phash, quality_problem

KIND_TITLE = {"waste": "Waste reported", "water": "Water leak reported", "energy": "Energy waste reported",
              "resource": "Problem reported"}
CATEGORY_KIND = {"waste": "waste", "water": "water", "energy": "energy", "other": "resource"}
IMPACT_INR = {"water": 300, "waste": 150, "energy": 200, "resource": 100}
CONFIRMS_TO_VERIFY = 2   # the original reporter + 2 more people
# Green Points (rules from the team's Clean City app): worked out from the reports themselves, so they never drift.
PTS_VERIFIED, PTS_UNVERIFIED, PTS_FIXED, PTS_REJECTED, PTS_CONFIRM, PTS_DUPLICATE = 10, 5, 5, -5, 2, -3
BADGES = [  # key, name, rule, stat, threshold
    ("first", "First report", "Sent a first report", "reports", 1),
    ("regular", "Regular", "5 reports", "reports", 5),
    ("fixer", "Campus Guardian", "3 of their reports got fixed", "fixed", 3),
    ("hero", "Green Hero", "100 Green Points", "points", 100),
]
CLOSED_ALERT = {"resolved": "fixed", "self-cleared": "cleared", "dismissed": "dismissed", "rejected": "rejected as false"}


def display_name(name):
    """Pseudonymous public name: first name + last initial, e.g. 'Asha R.'"""
    parts = [p for p in str(name or "").split() if p]
    if not parts:
        return "Anonymous"
    first = parts[0][:12]
    return f"{first} {parts[-1][0].upper()}." if len(parts) > 1 else first


class ReportDesk:
    def __init__(self, twin, vision):
        self.twin, self.vision = twin, vision
        self.reports = []
        self.points = {}   # only points not tied to a report of your own: confirmations (+2) and repeat photos (-3)
        self._ids = itertools.count(1)

    def _credit(self, reporter, pts):
        if reporter:
            self.points[reporter] = self.points.get(reporter, 0) + pts

    def tally(self):
        """Green Points and badge stats per reporter: +10 verified, +5 unverified, +5 more once fixed,
        -5 (instead) if facilities reject it as false, +2 per confirmation, -3 per already-reported photo."""
        self.sync()
        stats = {}
        for r in self.reports:
            name = r.get("reporter")
            if not name or name == "anonymous":
                continue
            s = stats.setdefault(name, {"points": 0, "reports": 0, "fixed": 0, "rejected": 0})
            if r.get("resolution") == "rejected as false":
                s["rejected"] += 1
                s["points"] += PTS_REJECTED
                continue
            s["reports"] += 1
            s["points"] += PTS_VERIFIED if r.get("verified") else PTS_UNVERIFIED
            if r.get("resolution") == "fixed":
                s["fixed"] += 1
                s["points"] += PTS_FIXED
        for name, extra in self.points.items():
            stats.setdefault(name, {"points": 0, "reports": 0, "fixed": 0, "rejected": 0})["points"] += extra
        for s in stats.values():
            s["badges"] = [{"key": k, "name": n, "rule": rule} for k, n, rule, stat, need in BADGES if s[stat] >= need]
        return stats

    def credits(self, reporter):
        return self.tally().get(reporter, {}).get("points", 0) if reporter else None

    def _zone_name(self, zone):
        return self.twin.campus.features[zone]["properties"]["name"]

    def _asset(self, zone):
        return zone if zone in self.twin.campus.buildings else self.twin.campus.nearest_building(zone)

    def sync(self):
        """Report status follows its linked incident: open -> in progress -> closed."""
        for r in self.reports:
            a = self.twin.alerts.get(r["alert"])
            if a is None:
                r["status"], r["resolution"] = "closed", "incident no longer exists"
            elif a.status in CLOSED_ALERT:
                r["status"], r["resolution"] = "closed", CLOSED_ALERT[a.status]
            else:
                r["status"], r["resolution"] = ("in_progress" if a.status == "assigned" else "open"), None
                r["verified"] = r.get("verified", False) or a.source == "student"

    def _nearby_open(self, kind, lng, lat):
        self.sync()   # only live incidents can collect confirmations
        return next((r for r in reversed(self.reports) if r["status"] in ("open", "in_progress") and r["kind"] == kind
                     and haversine_m(lng, lat, r["lng"], r["lat"]) <= 60), None)

    def _confirm(self, r, reporter, who=""):
        confirmers = r.setdefault("_confirmers", set())
        if who and who in confirmers:   # idempotent: one confirmation (and one credit) per person
            return {"ok": False, "check": "repeat", "report": r["id"],
                    "message": f"You've already reported or confirmed {r['id']}. Thanks, it's being handled."}
        if who:
            confirmers.add(who)
        r["confirmations"] += 1
        self._credit(reporter, PTS_CONFIRM)
        a = self.twin.alerts.get(r["alert"])
        if a:
            with self.twin.lock:
                self.twin.log(a, "confirmed", "reporter", f"{r['confirmations'] + 1} people have reported this")
        msg = f"Already reported nearby as {r['id']}. We added your confirmation."
        if not r["verified"] and r["confirmations"] >= CONFIRMS_TO_VERIFY:
            r["verified"] = True
            a = self.twin.alerts.get(r["alert"])
            if a:
                a.source, a.severity = "student", "crit" if r["kind"] == "water" else "warn"
                a.title = a.title.replace(" (unverified)", "")
                a.evidence["verified_by"] = f"{r['confirmations'] + 1} reporters"
            msg += " Enough people confirmed it, so it's now verified."
        return {"ok": False, "check": "repeat", "message": msg, "report": r["id"]}

    def _file(self, kind, zone, lng, lat, note, reporter, accuracy_m, verified, image_url=None, clip=None, h=None, emb=None, asset=None, who="",
              waste_type=None):
        sim = self.twin.clock.now()
        what = f"{waste_type['name']} reported" if kind == "waste" and waste_type and waste_type["key"] != "mixed" else KIND_TITLE[kind]
        title = f"{what} near {self._zone_name(zone)}" + ("" if verified else " (unverified)")
        detail = note or (clip["top"][0]["label"] if clip else "Reported without a photo")
        evidence = {"image": image_url} if image_url else {}
        if clip:
            evidence["clip"] = clip["top"]
        if kind == "waste" and waste_type:
            evidence["waste_type"] = waste_type
        with self.twin.lock:
            alert = self.twin._raise(kind, f"student-{next(self._ids)}", asset or self._asset(zone), title, detail, "warn",
                                     "student" if verified else "student-unverified",
                                     IMPACT_INR[kind] if verified else IMPACT_INR[kind] // 2, 2.0, evidence, sim)
            alert.lng, alert.lat = lng, lat
            if kind == "waste" and waste_type and waste_type.get("hazardous") and verified:
                alert.severity = "crit"   # syringes or a carcass jump the queue
        rep = {"id": f"R{len(self.reports) + 1:03d}", "alert": alert.id, "kind": kind, "zone": zone, "lng": lng, "lat": lat,
               "note": note, "image": image_url, "status": "open", "confirmations": 0, "clip": clip, "verified": verified,
               "reporter": reporter or "anonymous", "gps_accuracy_m": round(accuracy_m) if accuracy_m else None,
               "created": sim.isoformat(timespec="minutes"), "_hash": h, "_emb": emb, "t": time.time(),
               "_confirmers": {who} if who else set(), "resolution": None}
        self.reports.append(rep)
        if waste_type:
            rep["waste_type"] = waste_type
        return {"ok": True, "verified": verified, "report": self.public(rep), "alert": alert.id,
                "credits": self.credits(reporter), "waste_type": waste_type}

    def submit(self, img, lng, lat, note, image_url, reporter="", accuracy_m=0.0, category="", ip=""):
        who = (reporter.strip().lower() or "") + "|" + ip if ip or reporter else ""
        zone, dist = self.twin.campus.locate(lng, lat)
        if zone is None:
            return {"ok": False, "check": "location", "message": f"That spot is {dist:.0f} m from any campus zone. Reports must be on the DSU campus."}
        chosen = CATEGORY_KIND.get(category)

        if img is None:  # quick report: category + location only
            if not chosen:
                return {"ok": False, "check": "issue", "message": "Pick what kind of problem it is, or add a photo."}
            r = self._nearby_open(chosen, lng, lat)
            if r:
                return self._confirm(r, reporter, who)
            return self._file(chosen, zone, lng, lat, note, reporter, accuracy_m, verified=False, who=who)

        bad = quality_problem(img)
        if bad:
            return {"ok": False, "check": "quality", "message": bad}
        h = phash(img)
        for r in self.reports:
            if r["_hash"] is not None and h - r["_hash"] <= 6:
                self._credit(reporter, PTS_DUPLICATE)
                return {"ok": False, "check": "duplicate", "message": f"This photo was already reported as {r['id']} "
                        f"({PTS_DUPLICATE} Green Points). Take a fresh photo of a new problem."}
        try:
            analysis, emb = self.vision.analyze(img)
        except VisionUnavailable:
            if chosen:  # AI is down: still get it to facilities, unverified
                return self._file(chosen, zone, lng, lat, note, reporter, accuracy_m, False, image_url, h=h, who=who)
            return {"ok": False, "check": "issue", "message": "The photo check is busy. Pick a category and send again."}
        clip = analysis["clip"]
        scores = clip.get("scores", {})
        if chosen in ("waste", "water", "energy") and scores.get(chosen, 0) >= 0.35:
            kind, seen = chosen, True
        elif clip["issue_score"] >= 0.5 and not chosen:
            kind, seen = clip["issue"], True
        elif clip["issue_score"] >= 0.5 and chosen == "resource":
            kind, seen = clip["issue"], True
        else:
            kind, seen = chosen, False
        if not kind:
            return {"ok": False, "check": "issue", "message": "We couldn't see a waste, water or energy problem in this photo. Pick a category, or try a closer shot.", "clip": clip}

        self.sync()
        for r in self.reports:
            if r["status"] in ("open", "in_progress") and r["_emb"] is not None and len(r["_emb"]) == len(emb) \
                    and haversine_m(lng, lat, r["lng"], r["lat"]) <= 60 and float(np.dot(emb, r["_emb"])) >= 0.90:
                return self._confirm(r, reporter, who)

        asset = None
        if kind == "waste" and analysis.get("bin"):
            near_bin = min(self.twin.campus.of_type("bin"), key=lambda b: haversine_m(lng, lat, *self.twin.campus.location(b)))
            if haversine_m(lng, lat, *self.twin.campus.location(near_bin)) < 60:
                asset = near_bin
                self.twin.observe_bin(near_bin, analysis["bin"]["fill"], "student")
        return self._file(kind, zone, lng, lat, note, reporter, accuracy_m, seen, image_url, clip, h, emb, asset, who,
                          analysis.get("waste_type") if kind == "waste" else None)

    def leaderboard(self):
        rows = sorted(self.tally().items(), key=lambda kv: -kv[1]["points"])[:10]
        return [{"name": display_name(n), "credits": s["points"], "reports": s["reports"], "fixed": s["fixed"],
                 "badges": s["badges"]} for n, s in rows]

    @staticmethod
    def public(r):
        """What anyone can see: no exact names, GPS accuracy or internal fields."""
        out = {k: v for k, v in r.items() if not k.startswith("_") and k not in ("t", "gps_accuracy_m")}
        out["reporter"] = display_name(r.get("reporter")) if r.get("reporter") not in (None, "", "anonymous") else "Anonymous"
        return out

    def get(self, rid):
        self.sync()
        r = next((r for r in self.reports if r["id"] == rid), None)
        if not r:
            return None
        a = self.twin.alerts.get(r["alert"])
        return {**self.public(r), "alert_status": a.status if a else None,
                "timeline": [{k: h[k] for k in ("at", "event", "detail")} for h in (a.history if a else [])][-10:]}

    def list(self):
        self.sync()
        return [self.public(r) for r in reversed(self.reports)]
