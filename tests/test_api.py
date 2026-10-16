"""API regression tests: security, persistence, lifecycles, CCTV dedup, uploads.

Run from the repo root (needs backend/data/cache, which is committed):
    python -m unittest discover -s tests -v
Vision is mocked, so no model downloads or API keys are needed.
"""
import importlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
TMP = tempfile.mkdtemp(prefix="vx-test-")
os.environ.update(VERDANTIX_LITE="1", OPERATOR_TOKEN="test-op", INGEST_TOKEN="test-sensor", GEMINI_API_KEY="",
                  GROQ_API_KEY="", VERDANTIX_DB=str(Path(TMP) / "t.db"), RATE_LIMIT_SCALE="20")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

OP = {"X-Operator": "test-op"}
SENSOR = {"X-Token": "test-sensor"}


def jpeg(seed=0, size=(640, 480)):
    import random
    rnd = random.Random(seed)
    im = Image.new("RGB", (16, 12), (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    for x in range(16):
        for y in range(12):
            im.putpixel((x, y), (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    b = io.BytesIO()
    im.resize(size).save(b, "JPEG")
    return b.getvalue()


def load_app():
    for m in [m for m in sys.modules if m in ("app", "store", "security") or m.startswith(("twin", "vision", "ai."))]:
        sys.modules.pop(m)
    import config
    importlib.reload(config)
    return importlib.import_module("app")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = load_app()
        cls.c = TestClient(cls.A.app)
        cls.scores = []

        def fake(img):
            s = cls.scores.pop(0) if cls.scores else 0.1
            return ({"people": 0, "objects": [], "bin": None, "size": img.size,
                     "clip": {"issue": "waste", "issue_score": s, "scores": {"waste": s, "water": 0.0, "energy": 0.0, "clean": 1 - s},
                              "top": [{"label": "overflowing bin", "p": s}]}}, None)
        cls.A.vision.analyze = fake
        cls.A.vision.verify_fixed = lambda img, kind: {"passed": True, "issue_score": 0.05, "clip": {}, "bin": None}
        lng, lat = cls.A.twin.campus.location("soe-a-block")
        cls.here = {"lng": lng, "lat": lat}


class Security(Base):
    def test_anonymous_cannot_change_operations(self):
        self.assertEqual(self.c.post("/api/clock", json={"speed": 600}).status_code, 401)
        self.assertEqual(self.c.post("/api/scenario/leak").status_code, 401)
        self.assertEqual(self.c.post("/api/scenario/tank_overflow", headers=OP).status_code, 200)
        aid = self.c.post("/api/reports", data={**self.here, "category": "other", "reporter": "Sec"}).json()["alert"]
        self.assertEqual(self.c.post(f"/api/alerts/{aid}/order").status_code, 401)
        self.assertEqual(self.c.post(f"/api/alerts/{aid}/dismiss").status_code, 401)
        self.assertEqual(self.c.post(f"/api/alerts/{aid}/feedback", json={"verdict": "correct"}).status_code, 401)
        self.assertEqual(self.c.post(f"/api/alerts/{aid}/order", headers={"X-Operator": "wrong"}).status_code, 401)
        self.assertEqual(self.c.post(f"/api/alerts/{aid}/order", headers=OP).status_code, 200)

    def test_sensor_ingest_needs_token_and_valid_range(self):
        body = {"sensor": "t1", "asset": "soe-a-block", "metric": "power_kw", "value": 120}
        self.assertEqual(self.c.post("/api/ingest", json=body).status_code, 401)
        self.assertEqual(self.c.post("/api/ingest", json=body, headers=SENSOR).status_code, 200)
        bad = dict(body, value=-5)
        self.assertEqual(self.c.post("/api/ingest", json=bad, headers=SENSOR).status_code, 422)
        s = next(x for x in self.c.get("/api/sensors").json()["sensors"] if x["sensor"] == "t1")
        self.assertEqual(s["status"], "fresh")
        self.assertEqual(s["rejected"], 1)

    def test_uploads_validated(self):
        files = {"file": ("x.jpg", b"not an image", "image/jpeg")}
        r = self.c.post("/api/reports", data={**self.here, "category": "waste"}, files=files)
        self.assertEqual(r.status_code, 400)
        big = b"\xff" * int((self.A.C.MAX_UPLOAD_MB + 2) * 1024 * 1024)
        r = self.c.post("/api/vision/analyze", files={"file": ("b.jpg", big, "image/jpeg")})
        self.assertEqual(r.status_code, 413)
        gif = io.BytesIO(); Image.new("RGB", (64, 64)).save(gif, "GIF")
        r = self.c.post("/api/vision/analyze", files={"file": ("g.gif", gif.getvalue(), "image/gif")})
        self.assertEqual(r.status_code, 415)

    def test_errors_do_not_leak(self):
        r = self.c.get("/api/building/nope/series")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("Traceback", r.text)


class Lifecycle(Base):
    def test_cctv_one_incident_per_problem(self):
        self.scores[:] = [0.8, 0.8, 0.85, 0.9, 0.1, 0.1, 0.1, 0.8]
        actions = [self.c.post("/api/cctv/cam-foodcourt", files={"file": ("f.jpg", jpeg(i), "image/jpeg")}, headers=SENSOR).json()["action"]
                   for i in range(8)]
        self.assertTrue(actions[0].startswith("watching"))
        self.assertTrue(actions[1].startswith("alert"))
        self.assertTrue(actions[2].startswith("updated") and actions[3].startswith("updated"))
        self.assertTrue(actions[6].startswith("cleared"))
        self.assertTrue(actions[7].startswith("reopened"), actions[7])
        cctv = [a for a in self.A.twin.alerts.values() if a.rule == "cctv-waste"]
        self.assertEqual(len(cctv), 1)
        self.assertEqual(self.c.post("/api/cctv/cam-foodcourt", files={"file": ("f.jpg", jpeg(9), "image/jpeg")}).status_code, 401)

    def test_report_confirmations_idempotent_and_closed_reports(self):
        r1 = self.c.post("/api/reports", data={**self.here, "category": "water", "reporter": "Asha Rao"}).json()
        self.assertTrue(r1["ok"]); self.assertFalse(r1["verified"])
        again = self.c.post("/api/reports", data={**self.here, "category": "water", "reporter": "Asha Rao"}).json()
        self.assertIn("already", again["message"])
        credits = self.A.desk.credits("Asha Rao")
        self.c.post("/api/reports", data={**self.here, "category": "water", "reporter": "Asha Rao"})
        self.assertEqual(self.A.desk.credits("Asha Rao"), credits)            # no repeat credit
        board = self.c.get("/api/reports/leaderboard").json()
        self.assertIn("Asha R.", [b["name"] for b in board])                 # pseudonymous
        self.assertNotIn("Asha Rao", str(self.c.get("/api/reports").json()))
        aid = r1["alert"]
        self.c.post(f"/api/alerts/{aid}/feedback", json={"verdict": "false_positive", "reason": "test"}, headers=OP)
        st = self.c.get(f"/api/reports/{r1['report']['id']}").json()
        self.assertEqual(st["status"], "closed")
        r2 = self.c.post("/api/reports", data={**self.here, "category": "water", "reporter": "Ravi"}).json()
        self.assertTrue(r2["ok"])                                             # new report, not a confirmation of a closed one
        self.assertNotEqual(r2["report"]["id"], r1["report"]["id"])

    def test_work_order_transitions(self):
        r = self.c.post("/api/reports", data={**self.here, "category": "energy", "reporter": "T"}).json()
        o = self.c.post(f"/api/alerts/{r['alert']}/order", headers=OP).json()
        self.assertEqual(self.c.post(f"/api/alerts/{r['alert']}/order", headers=OP).json()["id"], o["id"])  # idempotent
        bad = self.c.post(f"/api/orders/{o['id']}/status", json={"status": "done"}, headers=OP)
        self.assertEqual(bad.status_code, 409)
        self.assertEqual(self.c.post(f"/api/orders/{o['id']}/status", json={"status": "acknowledged"}, headers=OP).status_code, 200)
        self.assertEqual(self.c.post(f"/api/orders/{o['id']}/status", json={"status": "in_progress"}, headers=OP).status_code, 200)
        p = self.c.post(f"/api/orders/{o['id']}/proof", files={"file": ("p.jpg", jpeg(3), "image/jpeg")}, headers=OP).json()
        self.assertTrue(p["ok"])
        self.assertEqual(self.A.twin.orders[o["id"]].status, "done")
        events = [h["event"] for h in self.A.twin.alerts[r["alert"]].history]
        self.assertIn("assigned", events); self.assertIn("resolved", events)
        self.assertTrue(any(row["order"] == o["id"] for row in self.c.get("/api/ledger").json()["rows"]))
        log = self.c.get("/api/audit", headers=OP).json()
        self.assertTrue(any(e["action"] == "proof verified" and e["target"] == o["id"] for e in log))

    def test_vision_outage_raises_nothing(self):
        from vision.detector import VisionUnavailable
        before = len(self.A.twin.alerts)
        orig = self.A.vision.analyze
        def down(img):
            raise VisionUnavailable("provider down")
        self.A.vision.analyze = down
        try:
            r = self.c.post("/api/cctv/cam-hostel-mess", files={"file": ("f.jpg", jpeg(5), "image/jpeg")}, headers=SENSOR)
            self.assertEqual(r.status_code, 503)
            rep = self.c.post("/api/reports", data={**self.here, "category": "waste"}, files={"file": ("a.jpg", jpeg(6), "image/jpeg")}).json()
            self.assertTrue(rep["ok"]); self.assertFalse(rep["verified"])   # still reaches facilities, unverified
        finally:
            self.A.vision.analyze = orig
        self.assertEqual(len([a for a in self.A.twin.alerts.values() if a.source == "camera"]) ,
                         len([a for a in list(self.A.twin.alerts.values())[:before] if a.source == "camera"]))

    def test_simulator_labels_savings_as_potential(self):
        r = self.c.post("/api/simulate", json={"ac_setpoint_delta": 1}).json()
        self.assertEqual(r["category"], "potential (modelled)")
        self.assertIn("assumptions", r)


class RateLimit(unittest.TestCase):
    def test_limiter_blocks_floods(self):
        from fastapi import HTTPException
        from security import RateLimiter
        rl = RateLimiter()
        for _ in range(6):
            rl.check("report", "1.2.3.4", 6)
        with self.assertRaises(HTTPException) as e:
            rl.check("report", "1.2.3.4", 6)
        self.assertEqual(e.exception.status_code, 429)
        rl.check("report", "5.6.7.8", 6)   # other clients unaffected


class Persistence(unittest.TestCase):
    def test_records_survive_restart(self):
        A = load_app()
        c = TestClient(A.app)
        lng, lat = A.twin.campus.location("soe-b-block")
        r = c.post("/api/reports", data={"lng": lng, "lat": lat, "category": "waste", "reporter": "Persist Test"}).json()
        o = c.post(f"/api/alerts/{r['alert']}/order", headers=OP).json()
        A.S.flush(A.db, A.twin, A.desk)
        B = load_app()   # a fresh process would do exactly this
        self.assertIn(r["alert"], B.twin.alerts)
        self.assertIn(o["id"], B.twin.orders)
        self.assertTrue(any(x["id"] == r["report"]["id"] for x in B.desk.list()))
        self.assertGreater(B.desk.credits("Persist Test") or 0, 0)
        n = next(B.twin._ids)
        self.assertGreater(n, int(o["id"][2:]))   # ids keep counting, no collisions


if __name__ == "__main__":
    unittest.main()


class CleanCityFeatures(Base):
    """Waste types and team routing, photo quality gate, Green Points rules, hotspot timing (ported from Clean City)."""

    def _fake_type(self, key):
        from vision import waste_types as WT

        def fake(img):
            return ({"people": 0, "objects": [], "bin": None, "size": img.size, "waste_type": WT.info(key, 0.9),
                     "clip": {"issue": "waste", "issue_score": 0.9, "scores": {"waste": 0.9, "water": 0, "energy": 0, "clean": 0.1},
                              "top": [{"label": "used syringes", "p": 0.9}]}}, None)
        self.A.vision.analyze = fake

    def tearDown(self):
        self.A.vision.analyze = type(self).fake_default

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fake_default = cls.A.vision.analyze

    def _report(self, seed, name="Asha Rao", size=(640, 480), img=None):
        lng, lat = self.A.twin.campus.location("soe-food-court") if "soe-food-court" in self.A.twin.campus.features \
            else next(iter(self.A.twin.campus.buildings.values()))["center"]
        return self.c.post("/api/reports", data={"lat": lat, "lng": lng, "reporter": name, "category": "waste"},
                           files={"file": ("r.jpg", img or jpeg(seed, size), "image/jpeg")}).json()

    def test_medical_waste_goes_to_clinic_fast(self):
        self._fake_type("medical")
        r = self._report(901)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["waste_type"]["key"], "medical")
        a = self.A.twin.alerts[r["alert"]]
        self.assertEqual(a.severity, "crit")
        self.assertIn("Biomedical waste", a.title)
        o = self.c.post(f"/api/alerts/{a.id}/order", json={}, headers=OP).json()
        self.assertIn("clinic", o["team"].lower())
        from datetime import datetime
        hours = (datetime.fromisoformat(o["deadline"]) - datetime.fromisoformat(o["created"])).total_seconds() / 3600
        self.assertEqual(hours, 2)

    def test_dark_or_tiny_photo_rejected_before_ai(self):
        calls = []
        self.A.vision.analyze = lambda img: calls.append(1)
        b = io.BytesIO()
        Image.new("RGB", (640, 480), (5, 5, 5)).save(b, "JPEG")
        r = self._report(0, img=b.getvalue())
        self.assertEqual(r["check"], "quality")
        r = self._report(902, size=(200, 150))
        self.assertEqual(r["check"], "quality")
        self.assertEqual(calls, [])   # no AI call spent on unusable photos

    def test_green_points_rules_and_badges(self):
        self._fake_type("plastic")
        name = "Ravi Kumar"
        first = self._report(903, name)
        self.assertEqual(first["credits"], 10)
        again = self._report(903, name)   # same photo again
        self.assertEqual(again["check"], "duplicate")
        board = {b["name"]: b for b in self.c.get("/api/reports/leaderboard").json()}
        self.assertEqual(board["Ravi K."]["credits"], 7)
        self.assertIn("First report", [g["name"] for g in board["Ravi K."]["badges"]])

    def test_hotspot_timing_and_recurrence(self):
        from datetime import datetime, timedelta
        from twin.hotspots import hotspots
        now = datetime(2026, 10, 16, 12, 0)
        ev = [{"kind": "waste", "asset": "bin-x", "lng": 77.45, "lat": 12.66, "at": (now - timedelta(days=d, hours=-1)).replace(hour=14).isoformat()}
              for d in (2, 4, 6, 8, 10)]
        h = hotspots(ev, now, names={"bin-x": "Food court bin"})[0]
        self.assertEqual(h["names"], ["Food court bin"])
        self.assertEqual(h["timing"]["patrol"], "1 PM")
        self.assertEqual(h["recurrence"]["every_days"], 2.0)
        self.assertGreaterEqual(datetime.fromisoformat(h["recurrence"]["next"]), now)


class MvpFeatures(Base):
    """Route simulator, before/after calculator, waste-object corrections, eco advisor."""

    def test_route_simulator_compares_with_one_method(self):
        r = self.c.get("/api/route?pickups=3").json()
        self.assertEqual(len(r["basic"]["stops"]), len(self.A.twin.campus.of_type("bin")))
        self.assertLessEqual(r["smart"]["distance_m"], r["basic"]["distance_m"])
        self.assertEqual(r["rounds_per_year"], 3 * 365)
        self.assertEqual(r["basic"]["coords"][0], r["basic"]["coords"][-1])   # closed tours from the depot
        self.assertIn("haversine", r["method"]["distance"])

    def test_before_after_calculator(self):
        r = self.c.post("/api/simulate", json={"led_retrofit": True, "fix_leaks": False}).json()
        rows = {x["key"]: x for x in r["before_after"]}
        self.assertLess(rows["energy_kwh"]["after"], rows["energy_kwh"]["before"])
        self.assertLess(rows["cost_inr"]["change"], 0)
        self.assertEqual(rows["water_kl"]["change"], 0)

    def test_waste_label_correction(self):
        ok = self.c.post("/api/vision/correct", json={"label": "bottle", "predicted": "plastic", "corrected": "glass"}).json()
        self.assertTrue(ok["ok"])
        self.assertIn("Blue", ok["bin"])
        bad = self.c.post("/api/vision/correct", json={"label": "x", "predicted": "plastic", "corrected": "lava"})
        self.assertEqual(bad.status_code, 400)
        seg = self.A.segregation([{"category": "organic"}, {"category": "organic"}, {"category": "ewaste"}])
        self.assertEqual(seg[0]["category"], "organic")
        self.assertEqual(seg[0]["count"], 2)

    def test_eco_advisor(self):
        a = self.c.get("/api/advisor").json()
        self.assertIn("priorities", a)
        self.assertTrue(a["segregation"])
        self.assertIn("SWM Rules 2016", a["method"])
