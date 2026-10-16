# Verdantix

**AI-powered digital twin of Dayananda Sagar University's Harohalli campus.**
CodeClash 2.0 · Problem Statement 2: AI-Powered Sustainable Campus.

Verdantix watches campus energy, water and waste using sensor data, computer vision and student photo reports. It runs every problem through one loop: **Detect → Verify → Assign → Prove**. A GenAI copilot turns what the twin sees into ranked actions and weekly sustainability reports, with the impact in rupees and CO₂.

## What it does

| Layer | How |
|---|---|
| **Digital twin** | 3D map of DSU drawn from satellite imagery (buildings, hostels, food court, bins, cameras, water tanks). A simulation clock replays real meter data at 1×, 60× or 600× speed. |
| **Energy** | Hourly meters per building. Anomalies come from a robust hour-of-week z-score, confirmed by an Isolation Forest. Room-level AC and lighting are flagged when cameras see an empty room or when they run after hours. |
| **Water** | Leak detection on the campus inlet using minimum night flow, evaluated on 14 labelled pipe bursts. Pressure drops point to the likely zone. Night-flow alerts per building catch running taps. |
| **Tanks & pumps** | Overhead tank and sump levels simulated from the buildings they feed, with float-switch pumps. Detects tank overflow (failed float valve) and pumping in peak time-of-day tariff hours, the campus's "inefficient resource use" alert. |
| **Forecasting** | Day-ahead electricity forecast per building (gradient-boosted trees, features at least 24 h old), tested on held-out Nov–Dec data against a seasonal-naive baseline. Drawn on the building chart and available to the copilot. |
| **Waste** | Bin fill model with time-to-full forecasting and an optimised pickup route. A fine-tuned YOLO11n classifies bins as Empty, Half or Overflow from camera frames. |
| **Computer vision** | YOLO11n counts people for occupancy. CLIP ViT-B/32 zero-shot covers leaks, running taps, lights on in empty rooms and litter, which have no training data. CLIP also verifies the "after fix" proof photo before a work order can close. |
| **Students as sensors** | A QR code on the dashboard opens a mobile page (`/report`) that uses the phone's GPS and camera. Reports pass ordered checks: on campus, duplicate check (pHash), CLIP issue score, and a repeat check (CLIP similarity within 60 m). Accepted reports earn green credits on a leaderboard. |
| **Live IoT ingest** | `POST /api/ingest` takes readings from real sensors (ESP32 sketch in `hardware/`, simulator in `scripts/fake_sensor.py`). Fresh readings override the replayed data and run through the same detectors. |
| **Hotspots** | DBSCAN over all alert locations (40 m radius, 30-day half-life) suggests permanent fixes. |
| **GenAI copilot** | Gemini free tier, then Groq free tier, through OpenAI-compatible tool calling over live twin data. Without internet, a free local Qwen2.5-1.5B on CPU answers from the same live facts. A rule-based answer is the last fallback. It cites alert IDs and never invents numbers. Voice input and answers in English, Kannada or Hindi. |
| **What-if simulator** | Annual ₹, kWh and CO₂ savings from raising the AC setpoint, cutting after-hours load, an LED retrofit, rooftop solar, fixing leaks, a smart off-peak pump schedule, or changing pickup frequency, computed over a full year of data, with indicative capex and payback per measure. |
| **Reports** | Weekly sustainability report: numbers from the twin, narrative from GenAI, exported as PDF. |
| **Accountability** | Work orders with deadlines per kind (water 4 h, waste 6 h, energy 24 h), an overdue state, a proof-photo close, and green-score grades A–D per block on a leaderboard. |

## ₹0 running cost

Every part of Verdantix is free to run:

| Part | Free option used |
|---|---|
| Vision | YOLO11n and CLIP ViT-B/32 run locally on CPU. No API. |
| ML | scikit-learn models trained locally on open data |
| GenAI | Gemini API free tier, then Groq free tier (no billing on either), then local Qwen2.5-1.5B-Instruct (Apache-2.0) on CPU, then a rule-based answer |
| Map | MapLibre GL (open source) with satellite imagery tiles |
| Hosting | Hugging Face Spaces free CPU tier |

## Data (all free and open, credited)

DSU has no public datasets, so Verdantix maps real open data onto DSU's buildings. It scales each source meter's real load shape to Indian benchmark energy intensities. Live sensors plug into the same API.

| Need | Dataset | License / source |
|---|---|---|
| Building energy and water | Building Data Genome 2 (Miller et al., 2020) | github.com/buds-lab/building-data-genome-project-2 |
| Room AC, lighting and plugs | CU-BEMS (Pipattanasomporn et al., 2020) | Scientific Data 7, 241 |
| Occupancy | UCI Occupancy Detection (Candanedo & Feldheim, 2016) | UCI ML Repository |
| Water network and leaks | BattLeDIM 2020, L-Town (Vrachimis et al.) | KIOS Research / Zenodo |
| Bin fill level | Roboflow `garbage_fill_level` v2 | CC BY 4.0 |
| Overflow and dumping | Roboflow `GarbageOverflow` v1 | Roboflow Universe |
| Pretrained models | YOLO11n (Ultralytics), CLIP ViT-B/32 (OpenAI) | AGPL-3.0 / MIT |
| Impact factors | CEA CO₂ Baseline Database (grid factor), BEE (AC setpoint savings) | Govt. of India |

The **Data & models** panel in the app shows which source meter feeds each DSU building, every assumption, and the measured accuracy of each model.

## Run it

```bash
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env          # add GEMINI_API_KEY / GROQ_API_KEY
venv\Scripts\python.exe scripts\build_campus.py
venv\Scripts\python.exe scripts\prepare_data.py     # needs datasets/ (see above)
venv\Scripts\python.exe scripts\train_bins.py 25 416 # optional: bin model
venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --port 8000
```
Open http://localhost:8000.

## Always-on cloud deployment (free)

- **Frontend:** Vercel, root directory `frontend`. `frontend/config.js` sets `window.VX_API` to the backend URL (no secrets in it).
- **Environment variables (Render):** `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPERATOR_TOKEN`, `INGEST_TOKEN`, `VERDANTIX_LITE=1`, `PYTHON_VERSION=3.12.7`; optional `ALLOWED_ORIGINS`, `VERDANTIX_DB`, `CCTV_*` thresholds. See `.env.example`.
- Hugging Face Spaces was tried first; Docker Spaces now need a paid plan, so it isn't used (`scripts/deploy_hf.py` is kept for reference).
- **Backend:** Render free web service from `render.yaml` in **lite mode**: same twin, models, alerts, copilot and reports, but no PyTorch, so YOLO/CLIP are replaced by Gemini vision for photo checks (fits in 512 MB). Trained models load from `backend/data/cache/models.pkl` (written on first local run), so start-up takes seconds. A free uptime pinger (cron-job.org, every 10 min on `/api/campus`) keeps it awake.
- **Full mode** (YOLO11n + fine-tuned bin model + CLIP on CPU) runs locally, or on any VM with `scripts/setup_vm.sh`.

## Reporter app and CCTV

- **Reporter app** (`/report`): installable PWA (Android "Install app", or package as an APK with PWABuilder from the live URL). Pick a category, add a photo (optional) and GPS. Photo reports are verified by AI instantly; reports without a photo are filed as unverified and become verified when 3 people report the same category within 60 m.
- **CCTV auto-reporting**: `POST /api/cctv/<camera>` with a frame. If the vision model sees garbage, a leak or lights on in an empty room, an alert is raised at that camera's map location; it closes itself after 3 clean frames. Use the Vision tab ("Use as CCTV camera") or `scripts/cctv_agent.py <camera> <webcam|video|rtsp-url>`.

## Waste MVP features

- **Waste vision** (Vision tab): upload or point a camera at waste; every object gets a box, a material (plastic, paper, glass, metal, organic, e-waste, hazardous, textile, other) and a confidence. Anyone can fix a wrong category from a dropdown; corrections are saved to the audit log as labels for tuning. Each material shows its bin under India's SWM Rules 2016 (green wet, blue dry, red domestic hazardous, e-waste drop point).
- **Collection route simulator** (What-if tab, `GET /api/route?pickups=2`): a basic fixed round that empties every bin vs a smart round that only visits bins at 60%+ or full within 3 h (nearest neighbour + 2-opt). Both are closed tours from the depot, costed with one documented method: haversine distance x 1.3 path factor, 6 km/h e-cart, 3 min per bin, ₹4/km + ₹300/h crew, 0.08 kWh/km x CEA grid factor. Shows distance, time, ₹ and CO₂ per round and per year, which bins were skipped and why, and both routes on the map.
- **Before-and-after calculator** (What-if tab): the campus over a full replayed year as it runs today vs with the chosen changes, for energy, bill, CO₂, water and bin overflow-hours.
- **Eco advisor** (Copilot tab, `GET /api/advisor`): today's priorities from rules over the live twin (hazardous waste first, overdue orders, the smart route, unassigned waste incidents, hotspot patrol times) and segregation tips for what cameras and reports saw recently. No LLM, so no made-up numbers.

## Built on our Clean City app

These parts come from [Clean City](https://github.com/pandeylakshya207-max/clean-city-app), the team's earlier street-garbage reporting app for Bengaluru, adapted from city wards to one campus:

- **Waste type and the right team**: the AI tells mixed, plastic, food/wet, construction, garden, bulky, e-waste, biomedical waste and dead animals apart. Each type has its own campus team and deadline (biomedical 2 h, carcass 4 h, food 3 h, debris 48 h); hazardous types are raised as critical.
- **Photo quality gate before any AI call**: too small, too dark or blank photos are rejected instantly (also skips dark CCTV frames, which saves API quota).
- **Green Points and badges**: +10 verified report, +5 unverified, +5 more when it gets fixed, +2 per confirmation, -3 for re-sending an already-reported photo, -5 if facilities reject it as false. Points are recomputed from the reports, so they can't drift.
- **Hotspot time patterns and recurrence**: for every DBSCAN hotspot, the 3-hour window when the problem usually appears, the busiest weekday, a patrol time one hour earlier, how often it comes back and when it is likely next (a simple average of past gaps, not a trained forecast).
- **Offline outbox**: a report made with no signal is kept on the phone (IndexedDB) and sent automatically when the phone is back online, with the time it was taken.

## Real sensors

```bash
venv\Scripts\python.exe scripts\fake_sensor.py               # simulated ESP32 nodes -> local twin
curl -X POST localhost:8000/api/ingest -H "Content-Type: application/json" \
     -d '{"sensor":"esp32-01","asset":"bin-soe-1","metric":"bin_fill","value":92}'
```
Metrics: `power_kw`, `water_lph` (buildings), `bin_fill` (bins), `people` (rooms), `tank_level_pct` (tanks). `GET /api/sensors` lists live sensors and valid asset ids. Set `INGEST_TOKEN` in `.env` to require an `X-Token` header.

## Demo

Sign in with **Operator sign-in** (the demo passcode is given to judges with the submission), then press **▶ Guided demo** in the header: it walks through night leak → work order → tank overflow → empty-room AC → bin route → copilot → what-if → student QR, with captions (→ / Esc to skip / stop).

## Security, records and trust

- **Fails closed in production.** On Render (or with `VERDANTIX_ENV=production`) a missing `OPERATOR_TOKEN` disables every privileged action (503) and a missing sensor token disables ingestion; the public dashboard still works. `/api/about` reports `security.misconfigured`.
- **Rate limits can't be dodged with a forged X-Forwarded-For**: behind Render's Cloudflare edge the client IP comes from `CF-Connecting-IP`.

| Area | What's in place |
|---|---|
| **Access** | Public: view the twin, report problems, copilot, what-if. **Operator** (passcode `OPERATOR_TOKEN`, sent as `X-Operator`): clock and demo controls, assign / dismiss / label / reopen incidents, work orders, CCTV from the dashboard. **Sensors** (`INGEST_TOKEN` as `X-Token`): `/api/ingest`, `/api/cctv`. The copilot can only create work orders for signed-in operators. No secrets live in frontend code. |
| **Limits** | Per-device rate limits (reports 6/min, vision 30/min, copilot 12/min, weekly report 4/min), 8 MB upload cap, JPEG/PNG/WebP only, pixel-count limit, server-chosen filenames, 30-day photo retention, one or two concurrent inferences, generic error messages, CORS limited to `ALLOWED_ORIGINS`. |
| **Records** | Incidents, work orders, reports, confirmations and credits are saved to SQLite (`backend/data/state/verdantix.db`) every few seconds and restored on start-up; demo injections are closed on restart. Every operator action and report is written to an append-only **audit log** (`GET /api/audit`, operator only). Backup: copy the `.db` file. |
| **Incident lifecycle** | Every incident has a timeline (detected, escalated, assigned, acknowledged, in progress, proof uploaded, verified or failed, resolved, reopened, labelled). Work orders: assigned → acknowledged → in progress → awaiting verification → done (only via a verified after-fix photo), with comments, cancel, reopen and overdue escalation. Operators label detections Correct / False alarm / Duplicate / Needs inspection; precision per source is shown in Data & models. |
| **Reports** | Status follows the linked incident (open, in progress, closed with the reason). One confirmation and one credit per person; closed incidents can't be confirmed. Public names are pseudonymous (first name + initial). Reporters see their reports' status in the app. |
| **CCTV** | One incident per camera and problem: opens after 2 consecutive detections over the score threshold, later frames update it (snapshot at most once a minute), closes after 3 clean frames, reopens if the problem returns within 15 minutes. If the vision provider fails, nothing is raised or cleared. Thresholds are environment settings. |
| **Provenance** | Every incident, building, tank and bin is labelled *Dataset replay*, *Measured*, *AI-inferred*, *Manual report*, *Simulated* or *Estimated*; a stale sensor is shown as stale, never as healthy. The dashboard shows when its data last updated. |
| **Savings** | What-if results are labelled **potential (modelled)** with a ±25% range, the baseline and projection, and every assumption; scenarios can be compared and exported to CSV. The **impact ledger** lists fixed incidents with the *estimated* avoided loss; measured savings need before/after meter data. |
| **Models** | Scores are shown as uncalibrated model scores with their thresholds, not as probabilities. `/api/health` and Data & models show the active vision mode (full: YOLO11n + CLIP locally; lite: cloud vision API), provider health, the last success and the last error; a failing provider is backed off for a minute. |

### Tests

```bash
python -m unittest discover -s tests -v
```
Covers: anonymous users can't run operator actions; sensor auth and range checks; invalid, oversized and wrong-format uploads; no leaked errors; repeated CCTV frames make one incident that closes and reopens; idempotent confirmations and credits; closed reports can't be confirmed; work-order transition rules; audit entries; vision outages raise nothing; savings labelled as potential; rate limiting; records survive a restart. CI runs them on every push (`.github/workflows/ci.yml`).

## Architecture

```
 cameras / webcam / demo video ──YOLO11n + CLIP──┐
 student photo reports ──checks + CLIP───────────┤
 replayed meters (BDG2, CU-BEMS, BattLeDIM, UCI) ├─> FastAPI twin engine ──WebSocket──> 3D dashboard (MapLibre)
 ESP32 sensors ──POST /api/ingest───────────────┤    anomaly models · forecasts · tanks · work orders · simulator
                                                 └─> GenAI copilot (tool calls) ──> actions · weekly PDF report
```

## Honest limits
- **Not DSU's meters.** Building, room and water data are real open datasets from other campuses, rescaled to DSU's floor areas; the UI labels them as dataset replay. Tanks and bins are simulated from that demand. Rupees and CO₂ are estimates from stated assumptions.
- **Vision is not validated on DSU footage.** Bin YOLO and occupancy numbers are on held-out public data; CLIP and the cloud vision path have no DSU evaluation set yet. Operator labels (correct / false alarm) are collected so precision can be measured in use.
- **Leak detection** uses one inlet signal (7 of 14 bursts caught in 3 days, about 54% of flagged nights real); the best published methods use all 33 pressure sensors.
- **Savings are potential, not measured.** Measured savings need before/after meter readings at DSU.
- **Persistence on Render's free tier:** the SQLite file survives restarts of the process but not a redeploy or instance replacement (the free disk is not persistent). Before a redeploy, an operator can download every record from `GET /api/backup` (operator only). For durable records use a Render persistent disk (paid) mounted at `VERDANTIX_DB`, the VM setup (`scripts/setup_vm.sh`), or a hosted Postgres.
- **AI photo checks are not inspections.** A work order closes when the after-fix photo passes the AI check; the timeline and impact ledger say "AI photo check (not a physical inspection)". Hazardous waste (syringes, carcasses) is raised as critical but marked "confirm on site": the AI label is a suspicion, never a verdict.
- **Penalties are human-reviewed.** The only automatic deduction is -3 for re-sending a photo whose perceptual hash matches an earlier report; the -5 for a false report applies only after facilities reject it.
- **Route simulator** uses nearest neighbour + 2-opt: a good heuristic, not a proven optimum. Costs are assumed constants (config.ROUTE), identical for both routes.
- **Accounts:** one shared operator passcode and one sensor token, not individual user accounts.
- Demo scenarios (leak, AC left on, bin overflow, tank overflow) are injections, labelled "Simulated demo", and closed on restart.
- The name Verdantix is also used by an existing research firm; check before any commercial use.

Built during CodeClash 2.0 (15–16 Oct 2026). Ideas carried over from the team's earlier Clean City project. All code is new.
