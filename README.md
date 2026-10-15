# EcoTwin DSU

**AI-powered digital twin of Dayananda Sagar University's Harohalli campus.**
CodeClash 2.0 · Problem Statement 2: AI-Powered Sustainable Campus.

EcoTwin watches campus energy, water and waste using sensor data, computer vision and student photo reports. It runs every problem through one loop: **Detect → Verify → Assign → Prove**. A GenAI copilot turns what the twin sees into ranked actions and weekly sustainability reports, with the impact in rupees and CO₂.

## What it does

| Layer | How |
|---|---|
| **Digital twin** | 3D map of DSU drawn from satellite imagery (buildings, hostels, food court, bins, cameras, water tanks). A simulation clock replays real meter data at 1×, 60× or 600× speed. |
| **Energy** | Hourly meters per building. Anomalies come from a robust hour-of-week z-score, confirmed by an Isolation Forest. Room-level AC and lighting are flagged when cameras see an empty room or when they run after hours. |
| **Water** | Leak detection on the campus inlet using minimum night flow, evaluated on 14 labelled pipe bursts. Pressure drops point to the likely zone. Night-flow alerts per building catch running taps. |
| **Waste** | Bin fill model with time-to-full forecasting and an optimised pickup route. A fine-tuned YOLO11n classifies bins as Empty, Half or Overflow from camera frames. |
| **Computer vision** | YOLO11n counts people for occupancy. CLIP ViT-B/32 zero-shot covers leaks, running taps, lights on in empty rooms and litter, which have no training data. CLIP also verifies the "after fix" proof photo before a work order can close. |
| **Students as sensors** | Students submit photo reports, which pass ordered checks: on campus, duplicate check (pHash), CLIP issue score, and a repeat check (CLIP similarity within 60 m). |
| **Hotspots** | DBSCAN over all alert locations (40 m radius, 30-day half-life) suggests permanent fixes. |
| **GenAI copilot** | Gemini free tier, then Groq free tier, through OpenAI-compatible tool calling over live twin data. Without internet, a free local Qwen2.5-1.5B on CPU answers from the same live facts. A rule-based answer is the last fallback. It cites alert IDs and never invents numbers. |
| **What-if simulator** | Annual ₹, kWh and CO₂ savings from raising the AC setpoint, cutting after-hours load, an LED retrofit, rooftop solar, fixing leaks, or changing pickup frequency, computed over a full year of data. |
| **Reports** | Weekly sustainability report: numbers from the twin, narrative from GenAI, exported as PDF. |
| **Accountability** | Work orders with deadlines per kind (water 4 h, waste 6 h, energy 24 h), an overdue state, a proof-photo close, and green-score grades A–D per block on a leaderboard. |

## ₹0 running cost

Every part of EcoTwin is free to run:

| Part | Free option used |
|---|---|
| Vision | YOLO11n and CLIP ViT-B/32 run locally on CPU. No API. |
| ML | scikit-learn models trained locally on open data |
| GenAI | Gemini API free tier, then Groq free tier (no billing on either), then local Qwen2.5-1.5B-Instruct (Apache-2.0) on CPU, then a rule-based answer |
| Map | MapLibre GL (open source) with satellite imagery tiles |
| Hosting | Hugging Face Spaces free CPU tier |

## Data (all free and open, credited)

DSU has no public datasets, so EcoTwin maps real open data onto DSU's buildings. It scales each source meter's real load shape to Indian benchmark energy intensities. Live sensors plug into the same API.

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

## Architecture

```
 cameras / webcam / demo video ──YOLO11n + CLIP──┐
 student photo reports ──checks + CLIP───────────┤
 replayed meters (BDG2, CU-BEMS, BattLeDIM, UCI) ├─> FastAPI twin engine ──WebSocket──> 3D dashboard (MapLibre)
 (future: ESP32 → MQTT → same ingest)            │    anomaly models · alerts · work orders · simulator
                                                 └─> GenAI copilot (tool calls) ──> actions · weekly PDF report
```

## Honest limits
- The meter data is real but comes from other campuses, rescaled to DSU's footprint. It is not DSU's own consumption.
- The leak detector uses one inlet signal. BattLeDIM's best methods use all 33 pressure sensors.
- The occupancy model is accurate on UCI data (see the app), but alerts come only from camera evidence or after-hours rules.
- Demo scenarios (leak, AC left on, bin overflow) are injections, and the UI labels them as such.

Built during CodeClash 2.0 (15–16 Oct 2026). Ideas carried over from the team's earlier Clean City project. All code is new.
