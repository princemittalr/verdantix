"""EcoTwin DSU API server.

Run from the repo root:  venv\\Scripts\\python.exe -m uvicorn app:app --app-dir backend --port 8000
Then open http://localhost:8000
"""
import asyncio
import io
import json
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import config as C
from ai import copilot, report
from twin.engine import Twin
from twin.hotspots import hotspots
from vision.detector import Vision
from vision.reports import ReportDesk

app = FastAPI(title="EcoTwin DSU")
twin = Twin()
seeded = twin.backfill(30)
vision = Vision()
desk = ReportDesk(twin, vision)
C.UPLOADS.mkdir(parents=True, exist_ok=True)
last_report = {}


def _img(data: bytes):
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file isn't an image we can read. Use a JPG or PNG.")


def _save(data: bytes, prefix):
    name = f"{prefix}-{uuid.uuid4().hex[:8]}.jpg"
    _img(data).save(C.UPLOADS / name, quality=85)
    return f"/uploads/{name}"


# ---------- twin state ----------
@app.get("/api/campus")
def campus():
    return twin.campus.geojson


@app.get("/api/state")
def state():
    return twin.state()


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            st = await asyncio.to_thread(twin.state)
            await sock.send_text(json.dumps(st, default=str))
            await asyncio.sleep(1.5)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/api/building/{bid}/series")
def building_series(bid: str, hours: int = 24):
    if bid not in twin.bld_ids:
        raise HTTPException(404, f"Unknown building {bid}")
    return twin.building_series(bid, twin.clock.now(), hours)


class ClockIn(BaseModel):
    speed: float | None = None
    jump: str | None = None   # "night" | "morning" | ISO datetime


@app.post("/api/clock")
def clock(c: ClockIn):
    if c.speed is not None:
        twin.clock.set_speed(c.speed)
    if c.jump == "night":
        twin.add_scenario("night")
    elif c.jump == "morning":
        now = twin.clock.now()
        twin.clock.jump_to(now.replace(hour=9, minute=0))
    elif c.jump:
        twin.clock.jump_to(datetime.fromisoformat(c.jump))
    return {"time": twin.clock.now().isoformat(timespec="minutes"), "speed": twin.clock.speed}


@app.post("/api/scenario/{kind}")
def scenario(kind: str, target: str | None = None):
    if kind not in ("leak", "ac_night", "bin_overflow", "night", "clear"):
        raise HTTPException(400, "Scenario must be leak, ac_night, bin_overflow, night or clear")
    if kind == "clear":
        twin.scenarios.clear()
        return {"cleared": True}
    return twin.add_scenario(kind, target)


# ---------- alerts and work orders ----------
class OrderIn(BaseModel):
    note: str = ""


@app.post("/api/alerts/{aid}/order")
def order(aid: str, body: OrderIn | None = None):
    if aid not in twin.alerts:
        raise HTTPException(404, f"Unknown alert {aid}")
    return twin.create_order(aid, body.note if body else "").__dict__


@app.post("/api/alerts/{aid}/dismiss")
def dismiss(aid: str):
    twin.resolve(aid, "self-cleared")
    return {"ok": True}


@app.post("/api/orders/{oid}/proof")
async def proof(oid: str, file: UploadFile = File(...)):
    if oid not in twin.orders:
        raise HTTPException(404, f"Unknown work order {oid}")
    data = await file.read()
    o = twin.orders[oid]
    a = twin.alerts[o.alert_id]
    url = _save(data, "proof")
    if a.kind == "energy":
        check = await asyncio.to_thread(vision.verify_fixed, _img(data), "energy")
    else:
        check = await asyncio.to_thread(vision.verify_fixed, _img(data), a.kind)
    check["image"] = url
    if check["passed"]:
        twin.complete_order(oid, check)
        return {"ok": True, "message": f"Verified. {oid} closed and {a.asset} is back to normal.", "check": check}
    return {"ok": False, "message": "The photo still shows the problem, so the order stays open.", "check": check}


# ---------- vision ----------
@app.post("/api/vision/analyze")
async def analyze(file: UploadFile = File(...), camera: str = Form(""), room: str = Form(""), bin: str = Form("")):
    t0 = time.time()
    img = _img(await file.read())
    result, _ = await asyncio.to_thread(vision.analyze, img)
    events = []
    if room:
        twin.observe_people(room, result["people"])
        events.append(f"{room}: {result['people']} people")
    if bin and result.get("bin"):
        twin.observe_bin(bin, result["bin"]["fill"], "camera")
        events.append(f"{bin}: {result['bin']['label']} ({result['bin']['fill']}%)")
    elif bin and result["clip"]["issue"] == "waste" and result["clip"]["issue_score"] >= 0.6:
        twin.observe_bin(bin, 100, "camera")
        events.append(f"{bin}: CLIP sees overflow ({result['clip']['issue_score']})")
    result["events"] = events
    result["ms"] = round((time.time() - t0) * 1000)
    result["models"] = vision.status
    return result


# ---------- student reports ----------
@app.post("/api/reports")
async def submit_report(file: UploadFile = File(...), lat: float = Form(...), lng: float = Form(...), note: str = Form("")):
    data = await file.read()
    img = _img(data)
    url = _save(data, "report")
    return await asyncio.to_thread(desk.submit, img, lng, lat, note, url)


@app.get("/api/reports")
def list_reports():
    return desk.list()


# ---------- AI ----------
class AskIn(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/api/copilot")
async def ask(q: AskIn):
    return await asyncio.to_thread(copilot.ask, twin, q.question, q.history)


class SimIn(BaseModel):
    ac_setpoint_delta: float = 0
    after_hours_cut: float = 0
    led_retrofit: bool = False
    fix_leaks: bool = False
    solar_kwp: float = 0
    pickups_per_day: int = 2


@app.post("/api/simulate")
def simulate(s: SimIn):
    return twin.simulate(**s.model_dump())


@app.post("/api/report/weekly")
async def weekly():
    r = await asyncio.to_thread(report.build, twin, desk)
    last_report.clear()
    last_report.update(r)
    return r


@app.get("/api/report/weekly.pdf")
def weekly_pdf():
    r = last_report or report.build(twin, desk)
    return Response(report.to_pdf(r), media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=ecotwin-weekly-report.pdf"})


# ---------- analytics ----------
@app.get("/api/route")
def route():
    return twin.pickup_route(twin.clock.now())


@app.get("/api/hotspots")
def get_hotspots():
    return hotspots(twin.events + [{"kind": r["kind"], "asset": r["zone"], "lng": r["lng"], "lat": r["lat"], "at": r["created"]} for r in desk.reports],
                    twin.clock.now())


@app.get("/api/history")
def history(days: int = 7):
    return twin.history_summary(days)


@app.get("/api/about")
def about():
    mapping = json.loads((C.CACHE / "mapping.json").read_text(encoding="utf-8"))
    return {"metrics": twin.metrics, "mapping": mapping, "models": vision.status, "seeded_events": seeded,
            "assumptions": {"grid_kg_co2_per_kwh": C.GRID_KG_CO2_PER_KWH, "tariff_inr_per_kwh": C.TARIFF_INR_PER_KWH,
                            "water_inr_per_kl": C.WATER_INR_PER_KL, "ac_saving_per_deg": C.AC_SAVING_PER_DEG,
                            "solar_kwh_per_kwp_day": C.SOLAR_KWH_PER_KWP_DAY, "deadlines_h": C.DEADLINE_H},
            "llm": {"gemini": bool(C.GEMINI_API_KEY), "openai": bool(C.OPENAI_API_KEY), "groq": bool(C.GROQ_API_KEY)}}


app.mount("/uploads", StaticFiles(directory=C.UPLOADS), name="uploads")
app.mount("/", StaticFiles(directory=C.FRONTEND, html=True), name="frontend")
