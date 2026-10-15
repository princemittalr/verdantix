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
from vision.detector import VisionUnavailable, phash

KIND_TITLE = {"waste": "Waste reported", "water": "Water leak reported", "energy": "Energy waste reported",
              "resource": "Problem reported"}
CATEGORY_KIND = {"waste": "waste", "water": "water", "energy": "energy", "other": "resource"}
IMPACT_INR = {"water": 300, "waste": 150, "energy": 200, "resource": 100}
CONFIRMS_TO_VERIFY = 2   # the original reporter + 2 more people


class ReportDesk:
    def __init__(self, twin, vision):
        self.twin, self.vision = twin, vision
        self.reports = []
        self.points = {}   # green credits: 10 verified report, 5 unverified, 2 per confirmation
        self._ids = itertools.count(1)

    def _credit(self, reporter, pts):
        if reporter:
            self.points[reporter] = self.points.get(reporter, 0) + pts

    def _zone_name(self, zone):
        return self.twin.campus.features[zone]["properties"]["name"]

    def _asset(self, zone):
        return zone if zone in self.twin.campus.buildings else self.twin.campus.nearest_building(zone)

    def _nearby_open(self, kind, lng, lat):
        return next((r for r in reversed(self.reports) if r["status"] == "open" and r["kind"] == kind
                     and haversine_m(lng, lat, r["lng"], r["lat"]) <= 60), None)

    def _confirm(self, r, reporter):
        r["confirmations"] += 1
        self._credit(reporter, 2)
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

    def _file(self, kind, zone, lng, lat, note, reporter, accuracy_m, verified, image_url=None, clip=None, h=None, emb=None, asset=None):
        sim = self.twin.clock.now()
        title = f"{KIND_TITLE[kind]} near {self._zone_name(zone)}" + ("" if verified else " (unverified)")
        detail = note or (clip["top"][0]["label"] if clip else "Reported without a photo")
        evidence = {"image": image_url} if image_url else {}
        if clip:
            evidence["clip"] = clip["top"]
        with self.twin.lock:
            alert = self.twin._raise(kind, f"student-{next(self._ids)}", asset or self._asset(zone), title, detail, "warn",
                                     "student" if verified else "student-unverified",
                                     IMPACT_INR[kind] if verified else IMPACT_INR[kind] // 2, 2.0, evidence, sim)
            alert.lng, alert.lat = lng, lat
        rep = {"id": f"R{len(self.reports) + 1:03d}", "alert": alert.id, "kind": kind, "zone": zone, "lng": lng, "lat": lat,
               "note": note, "image": image_url, "status": "open", "confirmations": 0, "clip": clip, "verified": verified,
               "reporter": reporter or "anonymous", "gps_accuracy_m": round(accuracy_m) if accuracy_m else None,
               "created": sim.isoformat(timespec="minutes"), "_hash": h, "_emb": emb, "t": time.time()}
        self.reports.append(rep)
        self._credit(reporter, 10 if verified else 5)
        return {"ok": True, "verified": verified, "report": self.public(rep), "alert": alert.id,
                "credits": self.points.get(reporter, 0) if reporter else None}

    def submit(self, img, lng, lat, note, image_url, reporter="", accuracy_m=0.0, category=""):
        zone, dist = self.twin.campus.locate(lng, lat)
        if zone is None:
            return {"ok": False, "check": "location", "message": f"That spot is {dist:.0f} m from any campus zone. Reports must be on the DSU campus."}
        chosen = CATEGORY_KIND.get(category)

        if img is None:  # quick report: category + location only
            if not chosen:
                return {"ok": False, "check": "issue", "message": "Pick what kind of problem it is, or add a photo."}
            r = self._nearby_open(chosen, lng, lat)
            if r:
                return self._confirm(r, reporter)
            return self._file(chosen, zone, lng, lat, note, reporter, accuracy_m, verified=False)

        h = phash(img)
        for r in self.reports:
            if r["_hash"] is not None and h - r["_hash"] <= 6:
                return {"ok": False, "check": "duplicate", "message": f"This photo was already reported as {r['id']}."}
        try:
            analysis, emb = self.vision.analyze(img)
        except VisionUnavailable:
            if chosen:  # AI is down: still get it to facilities, unverified
                return self._file(chosen, zone, lng, lat, note, reporter, accuracy_m, False, image_url, h=h)
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

        for r in self.reports:
            if r["status"] == "open" and r["_emb"] is not None and haversine_m(lng, lat, r["lng"], r["lat"]) <= 60 \
                    and float(np.dot(emb, r["_emb"])) >= 0.90:
                return self._confirm(r, reporter)

        asset = None
        if kind == "waste" and analysis.get("bin"):
            near_bin = min(self.twin.campus.of_type("bin"), key=lambda b: haversine_m(lng, lat, *self.twin.campus.location(b)))
            if haversine_m(lng, lat, *self.twin.campus.location(near_bin)) < 60:
                asset = near_bin
                self.twin.observe_bin(near_bin, analysis["bin"]["fill"], "student")
        return self._file(kind, zone, lng, lat, note, reporter, accuracy_m, seen, image_url, clip, h, emb, asset)

    def leaderboard(self):
        return [{"name": n, "credits": p} for n, p in sorted(self.points.items(), key=lambda kv: -kv[1])[:10]]

    @staticmethod
    def public(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    def list(self):
        return [self.public(r) for r in reversed(self.reports)]
