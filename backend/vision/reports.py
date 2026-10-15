"""Student photo reports: students act as human sensors for the twin.

Checks run in order and stop at the first failure:
  1. location is on campus (inside a drawn zone or within 80 m of one)
  2. not a duplicate photo (perceptual hash, Hamming distance <= 6)
  3. CLIP sees a real issue (issue score >= 0.5)
  4. not a repeat of an open report (CLIP similarity >= 0.90 within 60 m)
A passing report becomes a twin alert with source "student".
"""
import itertools
import time

import numpy as np

from twin.geo import haversine_m
from vision.detector import VisionUnavailable, phash

KIND_TITLE = {"waste": "Waste reported", "water": "Water leak reported", "energy": "Energy waste reported"}


class ReportDesk:
    def __init__(self, twin, vision):
        self.twin, self.vision = twin, vision
        self.reports = []
        self.points = {}   # green credits: 10 per accepted report, 2 per confirmation
        self._ids = itertools.count(1)

    def submit(self, img, lng, lat, note, image_url, reporter="", accuracy_m=0.0):
        zone, dist = self.twin.campus.locate(lng, lat)
        if zone is None:
            return {"ok": False, "check": "location", "message": f"That spot is {dist:.0f} m from any campus zone. Reports must be on the DSU campus."}
        h = phash(img)
        for r in self.reports:
            if h - r["_hash"] <= 6:
                return {"ok": False, "check": "duplicate", "message": f"This photo was already reported as {r['id']}."}
        try:
            analysis, emb = self.vision.analyze(img)
        except VisionUnavailable as e:
            return {"ok": False, "check": "issue", "message": f"{e} Please try again in a minute."}
        clip = analysis["clip"]
        if clip["issue_score"] < 0.5:
            return {"ok": False, "check": "issue", "message": "We couldn't see a waste, water or energy problem in this photo. Try a closer shot.", "clip": clip}
        for r in self.reports:
            if r["status"] == "open" and haversine_m(lng, lat, r["lng"], r["lat"]) <= 60 and float(np.dot(emb, r["_emb"])) >= 0.90:
                r["confirmations"] += 1
                if reporter:
                    self.points[reporter] = self.points.get(reporter, 0) + 2
                return {"ok": False, "check": "repeat", "message": f"Already reported nearby as {r['id']}. We added your confirmation.", "report": r["id"]}

        kind = clip["issue"]
        asset = zone
        if kind == "waste" and analysis.get("bin"):
            near_bin = min(self.twin.campus.of_type("bin"), key=lambda b: haversine_m(lng, lat, *self.twin.campus.location(b)))
            if haversine_m(lng, lat, *self.twin.campus.location(near_bin)) < 60:
                asset = near_bin
                self.twin.observe_bin(near_bin, analysis["bin"]["fill"], "student")
        if asset == zone and zone not in self.twin.campus.buildings:
            asset = self.twin.campus.nearest_building(zone)
        sim = self.twin.clock.now()
        with self.twin.lock:
            alert = self.twin._raise(kind, f"student-{next(self._ids)}", asset, f"{KIND_TITLE[kind]} near {self.twin.campus.features[zone]['properties']['name']}",
                                     note or clip["top"][0]["label"], "warn", "student",
                                     300 if kind == "water" else 150, 2.0, {"clip": clip["top"], "image": image_url}, sim)
            alert.lng, alert.lat = lng, lat
        rep = {"id": f"R{len(self.reports) + 1:03d}", "alert": alert.id, "kind": kind, "zone": zone, "lng": lng, "lat": lat,
               "note": note, "image": image_url, "status": "open", "confirmations": 0, "clip": clip,
               "reporter": reporter or "anonymous", "gps_accuracy_m": round(accuracy_m) if accuracy_m else None,
               "created": sim.isoformat(timespec="minutes"), "_hash": h, "_emb": emb, "t": time.time()}
        self.reports.append(rep)
        if reporter:
            self.points[reporter] = self.points.get(reporter, 0) + 10
        return {"ok": True, "report": self.public(rep), "alert": alert.id,
                "credits": self.points.get(reporter, 0) if reporter else None}

    def leaderboard(self):
        return [{"name": n, "credits": p} for n, p in sorted(self.points.items(), key=lambda kv: -kv[1])[:10]]

    @staticmethod
    def public(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    def list(self):
        return [self.public(r) for r in reversed(self.reports)]
