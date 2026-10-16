"""Verdantix API server.

Run from the repo root:  venv\\Scripts\\python.exe -m uvicorn app:app --app-dir backend --port 8000
Then open http://localhost:8000

Access (see security.py): viewing and problem reports are public and rate limited; anything that
changes operations (clock, demo scenarios, alerts, work orders, CCTV, sensors) needs a token.
Records persist in SQLite (store.py) and every operator action is written to the audit log.
"""
import asyncio
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import config as C
import store as S
from ai import copilot, local_llm, report
from security import client_ip, is_operator, limit, read_upload, require_maintainer, require_operator, require_sensor, role_of
from twin.engine import CLOSED, FEEDBACK, Twin
from twin.hotspots import hotspots
from collections import Counter, deque

from vision.detector import SEGREGATION, Vision, VisionUnavailable, quality_problem
from vision.reports import ReportDesk

log = logging.getLogger("verdantix")
Image.MAX_IMAGE_PIXELS = C.MAX_IMAGE_PIXELS
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}

@asynccontextmanager
async def lifespan(app_):
    yield
    S.flush(db, twin, desk)   # save everything on a clean shutdown


app = FastAPI(title="Verdantix", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=C.ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-Operator", "X-Token"])

twin = Twin()
vision = Vision()
desk = ReportDesk(twin, vision)
db = S.Store(C.DB_PATH)
restored = S.restore(db, twin, desk)
seeded = twin.backfill(30)
S.start_autosave(db, twin, desk)
C.UPLOADS.mkdir(parents=True, exist_ok=True)
last_report = {}
STARTED = datetime.now().isoformat(timespec="seconds")


@app.middleware("http")
async def body_limit(request: Request, call_next):
    """Reject oversized requests before reading them."""
    size = request.headers.get("content-length")
    if size and size.isdigit() and int(size) > (C.MAX_UPLOAD_MB + 1) * 1024 * 1024:
        return JSONResponse({"detail": f"Request too large. The limit is {C.MAX_UPLOAD_MB:g} MB."}, status_code=413)
    return await call_next(request)


@app.exception_handler(Exception)
async def safe_errors(request: Request, exc: Exception):
    """Never leak stack traces or internal paths to clients."""
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Something went wrong on the server. Please try again."}, status_code=500)


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    fields = ", ".join(".".join(str(x) for x in e["loc"][1:]) for e in exc.errors()) or "request"
    return JSONResponse({"detail": f"Invalid or missing: {fields}."}, status_code=422)


def audit(action, target="", request: Request | None = None, actor="operator", detail=""):
    try:
        db.audit(action, target, actor, client_ip(request) if request else "", detail)
    except Exception:
        log.exception("audit write failed")


def _img(data: bytes):
    MAX_SIDE = 1600   # phone photos can be 50 MP; decode at reduced size to protect a small server's memory
    try:
        im = Image.open(io.BytesIO(data))
        if im.format not in ALLOWED_FORMATS:
            raise HTTPException(415, "Use a JPG, PNG or WebP image.")
        w, h = im.size
        if w * h > C.MAX_IMAGE_PIXELS or w < 32 or h < 32:
            raise HTTPException(400, "That image is too large or too small.")
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        return im
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "That file isn't an image we can read. Use a JPG or PNG.")


def _save(img, prefix):
    name = f"{prefix}-{uuid.uuid4().hex[:10]}.jpg"   # server-chosen name: never trust client filenames
    small = img.copy()
    small.thumbnail((1024, 1024))
    small.save(C.UPLOADS / name, quality=82)
    return f"/uploads/{name}"


def _cleanup_uploads():
    """Retention: delete stored photos older than UPLOAD_RETENTION_DAYS (once at start-up)."""
    cutoff = time.time() - C.UPLOAD_RETENTION_DAYS * 86400
    for f in C.UPLOADS.glob("*.jpg"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)


_cleanup_uploads()


def _get_alert(aid):
    a = twin.alerts.get(aid)
    if not a:
        raise HTTPException(404, f"Unknown alert {aid}")
    return a


def _get_order(oid):
    o = twin.orders.get(oid)
    if not o:
        raise HTTPException(404, f"Unknown work order {oid}")
    return o


# ---------- access ----------
@app.get("/api/whoami")
def whoami(x_operator: str = Header("")):
    r = role_of(x_operator)
    return {"operator": r == "admin", "role": r, "auth_enabled": bool(C.OPERATOR_TOKEN)}


class SignIn(BaseModel):
    passcode: str = Field(max_length=200)


@app.post("/api/signin", dependencies=[Depends(limit("signin"))])
def signin(body: SignIn, request: Request):
    r = role_of(body.passcode)
    audit("signin " + (r if r != "public" else "failed"), r, request, actor="?")
    if r == "public":
        raise HTTPException(401, "Wrong passcode.")
    return {"operator": r == "admin", "role": r}


# ---------- twin state ----------
@app.get("/api/campus")
def campus():
    return twin.campus.geojson


@app.get("/api/state")
def state():
    return json.loads(twin.snapshot_json())


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            # One shared snapshot for every open dashboard, so ten viewers cost the same as one.
            await sock.send_text(await asyncio.to_thread(twin.snapshot_json))
            await asyncio.sleep(C.WS_INTERVAL_S)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/api/building/{bid}/series")
def building_series(bid: str, hours: int = 24):
    if bid not in twin.bld_ids:
        raise HTTPException(404, f"Unknown building {bid}")
    return twin.building_series(bid, twin.clock.now(), max(1, min(hours, 168)))


class ClockIn(BaseModel):
    speed: float | None = Field(None, ge=0, le=3600)
    jump: str | None = None   # "night" | "morning" | ISO datetime


@app.post("/api/clock")
def clock(c: ClockIn, request: Request, who=Depends(require_operator)):
    if c.speed is not None:
        twin.clock.set_speed(c.speed)
    if c.jump == "night":
        twin.add_scenario("night")
    elif c.jump == "morning":
        twin.add_scenario("morning")   # next 9 AM, never backwards
    elif c.jump:
        try:
            twin.clock.jump_to(datetime.fromisoformat(c.jump))
        except ValueError:
            raise HTTPException(400, "jump must be night, morning or an ISO date-time")
    audit("clock", "clock", request, who, c.model_dump_json())
    return {"time": twin.clock.now().isoformat(timespec="minutes"), "speed": twin.clock.speed}


@app.post("/api/scenario/{kind}")
def scenario(kind: str, request: Request, target: str | None = None, who=Depends(require_operator)):
    if kind not in ("leak", "ac_night", "bin_overflow", "tank_overflow", "night", "morning", "clear"):
        raise HTTPException(400, "Scenario must be leak, ac_night, bin_overflow, tank_overflow, night, morning or clear")
    audit(f"scenario {kind}", target or "", request, who)
    if kind == "clear":
        return {"cleared": twin.clear_demo()}
    return twin.add_scenario(kind, target)


# ---------- incidents and work orders ----------
class NoteIn(BaseModel):
    note: str = Field("", max_length=300)


@app.get("/api/alerts/{aid}")
def alert_detail(aid: str):
    a = _get_alert(aid)
    o = twin.orders.get(a.work_order) if a.work_order else None
    from dataclasses import asdict
    return {"alert": asdict(a), "order": asdict(o) if o else None, "audit": db.audit_list(30, aid)}


@app.post("/api/alerts/{aid}/order")
def order(aid: str, request: Request, body: NoteIn | None = None, who=Depends(require_operator)):
    _get_alert(aid)
    try:
        o = twin.create_order(aid, body.note if body else "", by=who)
    except ValueError as e:
        raise HTTPException(409, str(e))
    audit("assign", aid, request, who, o.id)
    return o.__dict__


@app.post("/api/alerts/{aid}/dismiss")
def dismiss(aid: str, request: Request, body: NoteIn | None = None, who=Depends(require_operator)):
    _get_alert(aid)
    a = twin.dismiss(aid, who, body.note if body else "")
    audit("dismiss", aid, request, who, body.note if body else "")
    return {"ok": True, "status": a.status}


class FeedbackIn(BaseModel):
    verdict: str
    reason: str = Field("", max_length=300)
    new_kind: str | None = None


@app.post("/api/alerts/{aid}/feedback")
def feedback(aid: str, body: FeedbackIn, request: Request, who=Depends(require_operator)):
    _get_alert(aid)
    if body.verdict not in FEEDBACK:
        raise HTTPException(400, f"verdict must be one of: {', '.join(sorted(FEEDBACK))}")
    a = twin.feedback(aid, body.verdict, body.reason, who, body.new_kind)
    audit(f"feedback {body.verdict}", aid, request, who, body.reason)
    return {"ok": True, "status": a.status, "feedback": a.feedback}


@app.post("/api/alerts/{aid}/reopen")
def reopen(aid: str, request: Request, body: NoteIn | None = None, who=Depends(require_operator)):
    a = _get_alert(aid)
    if a.status not in CLOSED:
        raise HTTPException(409, f"{aid} is not closed.")
    with twin.lock:
        o = twin.orders.get(a.work_order) if a.work_order else None
        if o and o.status == "done":
            twin.set_order_status(o.id, "reopened", who, body.note if body else "")
        else:
            a.status = "open"
            twin.log(a, "reopened", who, body.note if body else "")
    audit("reopen", aid, request, who)
    return {"ok": True, "status": a.status}


class StatusIn(BaseModel):
    status: str
    note: str = Field("", max_length=300)


@app.post("/api/orders/{oid}/status")
def order_status(oid: str, body: StatusIn, request: Request, who=Depends(require_maintainer)):
    _get_order(oid)
    if who == "maintainer" and body.status not in ("acknowledged", "in_progress"):
        raise HTTPException(403, "Maintainers can acknowledge and start work; cancelling or reopening needs an operator.")
    try:
        o = twin.set_order_status(oid, body.status, who, body.note)
    except ValueError as e:
        raise HTTPException(409, str(e))
    audit(f"order {body.status}", oid, request, who, body.note)
    return o.__dict__


@app.post("/api/orders/{oid}/comment")
def order_comment(oid: str, body: NoteIn, request: Request, who=Depends(require_maintainer)):
    _get_order(oid)
    if not body.note.strip():
        raise HTTPException(400, "Write a comment first.")
    twin.comment_order(oid, body.note.strip(), who)
    audit("order comment", oid, request, who, body.note)
    return {"ok": True}


@app.post("/api/orders/{oid}/proof", dependencies=[Depends(limit("vision"))])
async def proof(oid: str, request: Request, file: UploadFile = File(...), who=Depends(require_maintainer)):
    o = _get_order(oid)
    a = twin.alerts[o.alert_id]
    img = _img(await read_upload(file))
    try:
        twin.proof_submitted(oid, who)
    except ValueError as e:
        raise HTTPException(409, str(e))
    url = _save(img, "proof")
    try:
        check = await asyncio.to_thread(vision.verify_fixed, img, a.kind)
    except VisionUnavailable as e:
        twin.set_order_status(oid, "in_progress", "system", f"verification unavailable: {e}")
        raise HTTPException(503, f"{e} The order stays open; upload the photo again later.")
    check["image"] = url
    audit("proof " + ("verified" if check["passed"] else "rejected"), oid, request, who, f"score {check['issue_score']}")
    if check["passed"]:
        twin.complete_order(oid, check)
        return {"ok": True, "message": f"Verified. {oid} closed and {a.asset} is back to normal.", "check": check}
    twin.proof_failed(oid, check)
    return {"ok": False, "message": "The photo still shows the problem, so the order stays open.", "check": check}


@app.get("/api/ledger")
def ledger():
    return {"rows": twin.ledger(),
            "note": "Estimated, not measured: avoided loss = loss rate at detection x 365 days, assuming the fix holds. "
                    "Measured savings need before/after meter data."}


@app.get("/api/backup")
def backup(request: Request, who=Depends(require_operator)):
    """Operator-only JSON export of every record (alerts, orders, reports, points). Render's free disk is not durable,
    so download this before a redeploy; README explains how to restore it."""
    S.flush(db, twin, desk)
    audit("backup downloaded", "", request, who)
    data = S.snapshot(twin, desk)
    return Response(json.dumps(data, default=str), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=verdantix-backup.json"})


@app.get("/api/audit")
def audit_log(limit_: int = 100, target: str | None = None, who=Depends(require_operator)):
    return db.audit_list(max(1, min(limit_, 500)), target)


# ---------- vision ----------
@app.post("/api/vision/analyze", dependencies=[Depends(limit("vision"))])
async def analyze(file: UploadFile = File(...), room: str = Form(""), bin: str = Form(""), x_operator: str = Header("")):
    t0 = time.time()
    img = _img(await read_upload(file))
    try:
        result, _ = await asyncio.to_thread(vision.analyze, img)
    except VisionUnavailable as e:
        raise HTTPException(503, str(e))
    events = []
    if (room or bin) and not is_operator(x_operator):
        events.append("Sign in as operator to let this camera update rooms or bins.")
        room = bin = ""
    if room in twin.room_ids:
        twin.observe_people(room, result["people"])
        events.append(f"{room}: {result['people']} people")
    if bin in twin.campus.of_type("bin") and result.get("bin"):
        twin.observe_bin(bin, result["bin"]["fill"], "camera")
        events.append(f"{bin}: {result['bin']['label']} ({result['bin']['fill']}%)")
    elif bin in twin.campus.of_type("bin") and result["clip"]["issue"] == "waste" and result["clip"]["issue_score"] >= C.CCTV_RAISE:
        twin.observe_bin(bin, 100, "camera")
        events.append(f"{bin}: overflow seen (model score {result['clip']['issue_score']})")
    result["events"] = events
    result["segregation"] = segregation(result.get("objects", []))
    seen_materials.extend(o["category"] for o in result.get("objects", []) if o.get("category"))
    result["ms"] = round((time.time() - t0) * 1000)
    result["models"] = vision.status
    result["threshold"] = C.CCTV_RAISE
    return result


def segregation(objects):
    """Group detected waste objects by material, with the bin it belongs in and a short tip."""
    counts = Counter(o["category"] for o in objects if o.get("category"))
    return [{"category": c, "count": n, "bin": SEGREGATION[c][0], "tip": SEGREGATION[c][1]} for c, n in counts.most_common()]


seen_materials: deque = deque(maxlen=200)   # recent detections feed the advisor's segregation tips
corrections: Counter = Counter()           # (predicted, corrected) -> times a person fixed the label


class CorrectIn(BaseModel):
    label: str = Field(..., max_length=40)
    predicted: str = Field(..., max_length=20)
    corrected: str = Field(..., max_length=20)


@app.post("/api/vision/correct", dependencies=[Depends(limit("vision"))])
def vision_correct(body: CorrectIn, request: Request):
    """A person fixes a detection's category. Saved to the audit log as a label for tuning the prompts / model later."""
    if body.corrected not in SEGREGATION:
        raise HTTPException(400, f"category must be one of {', '.join(SEGREGATION)}")
    corrections[(body.predicted, body.corrected)] += 1
    seen_materials.append(body.corrected)
    audit("vision correction", body.label, request, "public", f"{body.predicted} -> {body.corrected}")
    total = sum(corrections.values())
    return {"ok": True, "message": f"Saved: {body.label} is {body.corrected}. {total} correction{'s' if total != 1 else ''} collected so far.",
            "bin": SEGREGATION[body.corrected][0], "tip": SEGREGATION[body.corrected][1]}


@app.get("/api/advisor")
def advisor():
    """Eco advisor: today's operational priorities and segregation tips, from rules over the live twin (no LLM, no guessing)."""
    sim = twin.clock.now()
    pri = []
    with twin.lock:
        alerts = [a for a in twin.alerts.values() if a.status in ("open", "assigned")]
        orders = list(twin.orders.values())
    for a in alerts:
        wt = a.evidence.get("waste_type")
        if wt and wt.get("hazardous"):
            pri.append({"level": "crit", "text": f"Send {wt['team']} to {a.title.split(' near ')[-1]} now: possible {wt['name'].lower()} ({a.id}).",
                        "why": f"Safety first: the AI label is a suspicion, so confirm on site before handling. {wt['deadline_h']} h deadline.", "alert": a.id})
    for o in orders:
        if o.overdue and o.status not in ("done", "cancelled"):
            pri.append({"level": "crit", "text": f"{o.id} for {o.team} is overdue: {o.title}.", "why": "Escalated to the supervisor.", "alert": o.alert_id})
    r = twin.pickup_route(sim)
    over = [b for b, s in twin.bin_state(sim).items() if s["fill"] >= 95]
    if r["smart"]["stops"]:
        pri.append({"level": "crit" if over else "warn",
                    "text": f"Run the smart collection round now: {len(r['smart']['stops'])} bins, {fmt_km(r['smart']['distance_m'])}, about {r['smart']['minutes']} min"
                            + (f" ({len(over)} already overflowing)." if over else "."),
                    "why": f"{r['pct_shorter']}% shorter than the fixed round of all {len(r['basic']['stops'])} bins; "
                           f"about ₹{r['saved_per_year']['inr']:,} and {r['saved_per_year']['hours']} crew-hours a year at {r['rounds_per_year'] // 365} rounds a day."})
    for a in alerts:
        if a.kind == "waste" and not a.work_order and not (a.evidence.get("waste_type") or {}).get("hazardous"):
            team = (a.evidence.get("waste_type") or {}).get("team", "Housekeeping")
            pri.append({"level": "warn", "text": f"Assign {a.id} to {team}: {a.title}.", "why": f"Open, about ₹{a.inr_per_day:,.0f} a day at risk.", "alert": a.id})
    for h in hotspots(twin.events, sim, names={k: f["properties"].get("name", k) for k, f in twin.campus.features.items()})[:3]:
        if h["timing"]["patrol"] and h["main_kind"] == "waste":
            pri.append({"level": "info", "text": f"Patrol {', '.join(h['names'][:2])} at {h['timing']['patrol']}, before the usual {h['timing']['peak']} window.",
                        "why": f"{h['events']} waste events in 30 days" + (f", returns about every {h['recurrence']['every_days']} days." if h["recurrence"]["every_days"] else ".")})
    mats = Counter(seen_materials)
    for rep in desk.reports[-50:]:
        wt = rep.get("waste_type") or {}
        mats.update({"food": ["organic"], "plastic": ["plastic"], "ewaste": ["ewaste"], "medical": ["hazardous"]}.get(wt.get("key"), []))
    tips = [{"category": c, "seen": n, "bin": SEGREGATION[c][0], "tip": SEGREGATION[c][1]} for c, n in mats.most_common(4)]
    if not tips:
        tips = [{"category": c, "seen": 0, "bin": SEGREGATION[c][0], "tip": SEGREGATION[c][1]} for c in ("organic", "plastic", "ewaste")]
    return {"priorities": pri[:8], "segregation": tips, "time": sim.isoformat(timespec="minutes"),
            "method": "Rules over the live twin: hazardous waste first, overdue orders, bin fill and the route simulator, open waste incidents, "
                      "hotspot patrol times. Segregation tips follow India's SWM Rules 2016 and what cameras and reports saw recently."}


def fmt_km(m):
    return f"{m / 1000:.1f} km" if m >= 1000 else f"{m} m"


# ---------- CCTV: cameras report problems on their own ----------
cctv_state: dict[str, dict] = {}   # camera -> {hits: {issue: n}, clean: n, last: result, at: time}
CCTV_TEXT = {"waste": "garbage or overflowing bin", "water": "water leak or running tap", "energy": "lights or AC on in an empty room"}


@app.get("/api/cctv")
def cctv_list():
    cams = []
    for cid in twin.campus.of_type("camera"):
        p = twin.campus.features[cid]["properties"]
        st = cctv_state.get(cid, {})
        cams.append({"id": cid, "name": p["name"], "lng": twin.campus.location(cid)[0], "lat": twin.campus.location(cid)[1],
                     "last": st.get("last"), "seen_s": round(time.time() - st["at"]) if st.get("at") else None})
    return {"cameras": cams, "rules": {"raise_score": C.CCTV_RAISE, "clear_score": C.CCTV_CLEAR, "frames_to_open": C.CCTV_MIN_FRAMES,
                                       "clean_frames_to_close": C.CCTV_CLEAN_FRAMES, "reopen_window_s": C.CCTV_REOPEN_S}}


def _cctv_incident(cam, issue):
    """The live incident for this camera + issue, or a recently closed one to reopen (dedup across frames)."""
    best = None
    for a in twin.alerts.values():
        if a.asset == cam and a.rule == f"cctv-{issue}":
            if a.status in ("open", "assigned"):
                return a, False
            closed_at = next((h["wall"] for h in reversed(a.history) if h["event"] == "self-cleared"), None)
            if a.status == "self-cleared" and closed_at and \
                    (datetime.now() - datetime.fromisoformat(closed_at)).total_seconds() < C.CCTV_REOPEN_S:
                best = a
    return best, best is not None


@app.post("/api/cctv/{cam}", dependencies=[Depends(limit("cctv"))])
async def cctv_frame(cam: str, request: Request, file: UploadFile = File(...), who=Depends(require_sensor)):
    """One frame from a CCTV camera (dashboard webcam, video file or scripts/cctv_agent.py).

    One incident per camera and problem: it opens after CCTV_MIN_FRAMES consecutive detections, later frames update
    it (no duplicates), it closes after CCTV_CLEAN_FRAMES clean frames, and reopens if the problem returns within
    CCTV_REOPEN_S. If the vision provider fails, nothing is raised or cleared.
    """
    if cam not in twin.campus.of_type("camera"):
        raise HTTPException(404, f"Unknown camera {cam}. Cameras: {', '.join(twin.campus.of_type('camera'))}")
    img = _img(await read_upload(file))
    bad = quality_problem(img)
    if bad:   # covered lens or a dark night frame: skip the AI call, change nothing
        return {"camera": cam, "issue": "none", "score": 0.0, "threshold": C.CCTV_RAISE, "people": 0,
                "action": "skipped: " + bad.split(".")[0].lower(), "incident": None, "top": [], "objects": [], "size": img.size}
    try:
        result, _ = await asyncio.to_thread(vision.analyze, img)
    except VisionUnavailable as e:
        raise HTTPException(503, f"{e} No incident was raised or cleared.")
    clip = result["clip"]
    name = twin.campus.features[cam]["properties"]["name"]
    lng, lat = twin.campus.location(cam)
    st = cctv_state.setdefault(cam, {"hits": {}, "clean": 0})
    issue, score = clip["issue"], clip["issue_score"]
    if result.get("bin") and result["bin"]["fill"] >= 100:
        issue, score = "waste", max(score, result["bin"]["conf"])
    st.update(at=time.time(), last={"issue": issue, "score": round(score, 3), "people": result["people"],
                                    "waste_type": (result.get("waste_type") or {}).get("name")})
    action, incident = "none", None
    if score >= C.CCTV_RAISE:
        st["clean"] = 0
        st["hits"] = {issue: st["hits"].get(issue, 0) + 1}
        with twin.lock:
            a, reopen_ = _cctv_incident(cam, issue)
            if a is None and st["hits"][issue] < C.CCTV_MIN_FRAMES:
                action = f"watching ({st['hits'][issue]}/{C.CCTV_MIN_FRAMES} frames)"
            elif a is not None:
                if reopen_:
                    a.status = "open"
                    twin.log(a, "reopened", "camera", f"problem seen again (model score {score:.2f})")
                    action = f"reopened {a.id}"
                else:
                    action = f"updated {a.id}"
                ev = a.evidence
                ev["score"], ev["last_seen"] = round(score, 3), datetime.now().isoformat(timespec="seconds")
                ev["frames"] = ev.get("frames", 1) + 1
                if time.time() - ev.get("_last_img", 0) > 60:   # keep one fresh snapshot a minute, not one per frame
                    ev["image"], ev["_last_img"] = _save(img, f"cctv-{cam}"), time.time()
                    twin.log(a, "seen again", "camera", f"model score {score:.2f}, {ev['frames']} frames so far")
                incident = a.id
            else:
                url = _save(img, f"cctv-{cam}")
                wt = result.get("waste_type") if issue == "waste" else None
                seen_text = wt["name"].lower() if wt and wt["key"] != "mixed" else CCTV_TEXT[issue]
                a = twin._raise(issue, f"cctv-{issue}", cam, f"{name} sees {seen_text}",
                                f"Detected by camera AI over {C.CCTV_MIN_FRAMES} frames (model score {score:.2f}, "
                                f"threshold {C.CCTV_RAISE}): " + ", ".join(t["label"] for t in clip["top"][:2]) + ".",
                                "crit" if score >= 0.8 or issue == "water" else "warn", "camera",
                                {"water": 300, "waste": 150, "energy": 200}[issue], 2.0,
                                {"image": url, "_last_img": time.time(), "clip": clip["top"], "score": round(score, 3),
                                 "threshold": C.CCTV_RAISE, "camera": cam, "frames": C.CCTV_MIN_FRAMES,
                                 "model": vision.health["mode"], **({"waste_type": wt} if wt else {})}, twin.clock.now())
                a.lng, a.lat = lng, lat
                if wt and wt.get("hazardous"):   # safety incident: AI suspicion, a person confirms on site
                    a.severity = "crit"
                    a.evidence["needs_human_check"] = True
                action, incident = f"alert {a.id}", a.id
            twin._snap = None
    elif score < C.CCTV_CLEAR:
        st["hits"] = {}
        st["clean"] += 1
        if st["clean"] >= C.CCTV_CLEAN_FRAMES:   # problem gone for a few frames in a row: close it
            with twin.lock:
                for a in twin.alerts.values():
                    if a.asset == cam and a.rule.startswith("cctv-") and a.status == "open":
                        a.status = "self-cleared"
                        twin.log(a, "self-cleared", "camera", f"{C.CCTV_CLEAN_FRAMES} clean frames in a row")
                        action = f"cleared {a.id}"
                twin._snap = None
    else:
        action = "uncertain (between thresholds)"
    return {"camera": cam, "issue": issue, "score": round(score, 3), "threshold": C.CCTV_RAISE, "people": result["people"],
            "action": action, "incident": incident, "top": clip["top"], "objects": result["objects"], "size": result["size"],
            "waste_type": result.get("waste_type")}


# ---------- problem reports (students, staff) ----------
@app.post("/api/reports", dependencies=[Depends(limit("report"))])
async def submit_report(request: Request, lat: float = Form(..., ge=-90, le=90), lng: float = Form(..., ge=-180, le=180),
                        note: str = Form("", max_length=300), reporter: str = Form("", max_length=40),
                        accuracy_m: float = Form(0, ge=0), category: str = Form("", max_length=20),
                        file: UploadFile | None = File(None)):
    img = url = None
    if file is not None and file.filename:
        img = _img(await read_upload(file))
        url = _save(img, "report")
    r = await asyncio.to_thread(desk.submit, img, lng, lat, note.strip(), url, reporter.strip(), accuracy_m,
                                category.strip().lower(), client_ip(request))
    if r.get("ok"):
        audit("report", r["report"]["id"], request, "public", r["alert"])
    return r


@app.get("/api/reports")
def list_reports():
    return desk.list()[:100]


@app.get("/api/reports/leaderboard")
def report_leaderboard():
    return desk.leaderboard()


@app.get("/api/reports/{rid}")
def report_status(rid: str):
    r = desk.get(rid)
    if not r:
        raise HTTPException(404, f"Unknown report {rid}")
    return r


@app.get("/report")
def report_page():
    """Mobile page students open from the QR code: photo + phone GPS -> the twin."""
    return FileResponse(C.FRONTEND / "report.html")


@app.get("/maintainer")
def maintainer_page():
    return FileResponse(C.FRONTEND / "maintainer.html")


# ---------- IoT ingest (ESP32 / MQTT bridge / scripts/fake_sensor.py) ----------
class ReadingIn(BaseModel):
    sensor: str = Field(max_length=64)
    asset: str = Field(max_length=64)
    metric: str        # power_kw | water_lph | bin_fill | people | tank_level_pct
    value: float


@app.post("/api/ingest", dependencies=[Depends(limit("ingest"))])
def ingest(readings: ReadingIn | list[ReadingIn], who=Depends(require_sensor)):
    items = readings if isinstance(readings, list) else [readings]
    if len(items) > 50:
        raise HTTPException(413, "Send at most 50 readings per request.")
    accepted, rejected = [], []
    for r in items:
        try:
            accepted.append(twin.ingest(r.sensor, r.asset, r.metric, r.value))
        except (KeyError, ValueError) as e:
            rejected.append({"sensor": r.sensor, "asset": r.asset, "metric": r.metric, "error": str(e).strip("'")})
    if rejected and not accepted:
        raise HTTPException(422, rejected[0]["error"])
    return {"accepted": accepted, "rejected": rejected}


@app.get("/api/sensors")
def sensors():
    return {"sensors": twin.sensors_status(), "live": twin.live_sensors(),
            "fresh_seconds": C.SENSOR_FRESH_S, "stale_seconds": C.SENSOR_STALE_S, "ranges": C.SENSOR_RANGES,
            "metrics": list(C.SENSOR_RANGES),
            "assets": {"buildings": twin.bld_ids, "bins": twin.campus.of_type("bin"), "rooms": twin.room_ids,
                       "tanks": list(twin.tanks)}}


# ---------- AI ----------
class AskIn(BaseModel):
    question: str = Field(max_length=1000)
    history: list[dict] = Field(default_factory=list, max_length=12)
    lang: str = "en"   # en | kn | hi


@app.post("/api/copilot", dependencies=[Depends(limit("copilot"))])
async def ask(q: AskIn, x_operator: str = Header("")):
    hist = [{"role": h.get("role"), "content": str(h.get("content", ""))[:2000]} for h in q.history
            if h.get("role") in ("user", "assistant")]
    return await asyncio.to_thread(copilot.ask, twin, q.question, hist, q.lang, is_operator(x_operator))


class SimIn(BaseModel):
    ac_setpoint_delta: float = Field(0, ge=0, le=5)
    after_hours_cut: float = Field(0, ge=0, le=1)
    led_retrofit: bool = False
    fix_leaks: bool = False
    solar_kwp: float = Field(0, ge=0, le=5000)
    pickups_per_day: int = Field(2, ge=1, le=6)
    smart_pump_schedule: bool = False


@app.post("/api/simulate", dependencies=[Depends(limit("simulate"))])
def simulate(s: SimIn):
    r = twin.simulate(**s.model_dump())
    r["category"] = "potential (modelled)"
    r["uncertainty"] = {"low_factor": 0.75, "high_factor": 1.25,
                        "why": "replayed open-data meters rescaled to DSU, assumed tariff and shares; not measured at DSU"}
    return r


@app.post("/api/report/weekly", dependencies=[Depends(limit("weekly"))])
async def weekly():
    r = await asyncio.to_thread(report.build, twin, desk)
    last_report.clear()
    last_report.update(r)
    return r


@app.get("/api/report/weekly.pdf", dependencies=[Depends(limit("weekly"))])
def weekly_pdf():
    r = last_report or report.build(twin, desk)
    return Response(report.to_pdf(r), media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=verdantix-weekly-report.pdf"})


# ---------- analytics ----------
@app.get("/api/route")
def route(pickups: int = 2):
    """Collection route simulator: fixed round vs smart round, same cost method (see 'method' in the reply)."""
    return twin.pickup_route(twin.clock.now(), max(1, min(pickups, 6)))


@app.get("/api/hotspots")
def get_hotspots():
    names = {k: f["properties"].get("name", k) for k, f in twin.campus.features.items()}
    return hotspots(twin.events + [{"kind": r["kind"], "asset": r["zone"], "lng": r["lng"], "lat": r["lat"], "at": r["created"]} for r in desk.reports],
                    twin.clock.now(), names=names)


@app.get("/api/history")
def history(days: int = 7):
    h = twin.history_summary(max(1, min(days, 60)))
    h["co2"] = {"kg_per_kwh": C.GRID_KG_CO2_PER_KWH, "scope": "electricity only (Scope 2, location-based)",
                "source": "CEA CO2 Baseline Database, weighted average RE-adjusted, FY 2023-24, India grid",
                "data": "dataset replay rescaled to DSU, not DSU meters"}
    return h


@app.get("/api/health")
def health():
    return {"ok": True, "started": STARTED, "restored": restored, "vision": vision.health,
            "db": str(C.DB_PATH.name), "mode": "lite" if C.LITE else "full"}


@app.get("/api/about")
def about():
    mapping = json.loads((C.CACHE / "mapping.json").read_text(encoding="utf-8"))
    return {"metrics": twin.metrics, "mapping": mapping, "models": vision.status, "vision_health": vision.health,
            "feedback": twin.feedback_metrics(), "seeded_events": seeded, "restored": restored, "started": STARTED,
            "security": {"operator_auth": bool(C.OPERATOR_TOKEN), "sensor_auth": bool(C.INGEST_TOKEN or C.OPERATOR_TOKEN),
                         "production": C.PRODUCTION, "fails_closed": C.PRODUCTION,
                         "misconfigured": C.PRODUCTION and not C.OPERATOR_TOKEN, "client_ip_from": "Cloudflare edge header" if C.TRUST_EDGE_HEADERS else "socket address",
                         "allowed_origins": C.ALLOWED_ORIGINS, "max_upload_mb": C.MAX_UPLOAD_MB},
            "data_sources": {"building energy & water": "dataset replay (BDG2 2017, rescaled to Indian benchmarks)",
                             "rooms": "dataset replay (CU-BEMS 2019)", "water network": "dataset replay (BattLeDIM 2018)",
                             "tanks & bins": "simulated from the replayed demand", "cameras & photos": "AI-inferred",
                             "student reports": "manual reports", "IoT readings": "measured (when sensors post)",
                             "rupees & CO2": "estimated from stated assumptions"},
            "assumptions": {"grid_kg_co2_per_kwh": C.GRID_KG_CO2_PER_KWH, "tariff_inr_per_kwh": C.TARIFF_INR_PER_KWH,
                            "water_inr_per_kl": C.WATER_INR_PER_KL, "ac_saving_per_deg": C.AC_SAVING_PER_DEG,
                            "solar_kwh_per_kwp_day": C.SOLAR_KWH_PER_KWP_DAY, "deadlines_h": C.DEADLINE_H,
                            "tod_peak_hours": C.TOD_PEAK_HOURS, "tod_peak_surcharge_inr": C.TOD_PEAK_SURCHARGE,
                            "tod_night_rebate_inr": C.TOD_NIGHT_REBATE, "capex_inr": C.CAPEX,
                            "tanks": {k: {x: v[x] for x in ("capacity_kl", "pump_kw", "pump_lph")} for k, v in C.TANKS.items()}},
            "llm": {"gemini (free tier)": bool(C.GEMINI_API_KEY), "groq (free tier)": bool(C.GROQ_API_KEY),
                    "local qwen (CPU, offline)": local_llm.available()}}


app.mount("/uploads", StaticFiles(directory=C.UPLOADS), name="uploads")
app.mount("/", StaticFiles(directory=C.FRONTEND, html=True), name="frontend")
