"""Shared settings and the impact constants every rupee / CO2 figure comes from."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "backend" / "data"
CACHE = DATA / "cache"
UPLOADS = ROOT / "backend" / "uploads"
FRONTEND = ROOT / "frontend"
WEIGHTS = ROOT / "backend" / "weights"


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Free local fallback model (runs on CPU, no key, works offline).
LOCAL_LLM = os.getenv("LOCAL_LLM", "Qwen/Qwen2.5-1.5B-Instruct")

# Impact constants (editable; shown on the "Assumptions" panel).
GRID_KG_CO2_PER_KWH = 0.71   # CEA CO2 Baseline Database, all-India grid (approx.)
TARIFF_INR_PER_KWH = 8.0     # assumed HT educational tariff, Karnataka
WATER_INR_PER_KL = 60.0      # assumed blended BWSSB / tanker cost
# Collection route simulator: one documented cost method, applied identically to both routes.
ROUTE = {
    "road_factor": 1.3,        # straight-line (haversine) distance x 1.3 for paths around buildings (assumed)
    "speed_kmh": 6.0,          # electric garbage cart on campus paths (assumed)
    "stop_min": 3.0,           # minutes to empty one bin (assumed)
    "cart_inr_per_km": 4.0,    # e-cart energy + wear (assumed)
    "crew_inr_per_h": 300.0,   # 2-person housekeeping crew (assumed)
    "cart_kwh_per_km": 0.08,   # e-cart consumption (assumed); CO2 uses the CEA grid factor
    "collect_fill_pct": 60,    # smart route: collect bins at or above this fill...
    "collect_eta_h": 3,        # ...or that will be full within this many hours
}
WATER_KG_CO2_PER_KL = 0.30   # pumping + treatment energy (approx.)
WASTE_KG_CO2_PER_KG = 0.50   # mixed waste to landfill (approx.)
AC_SAVING_PER_DEG = 0.06     # BEE: ~6% AC energy saved per 1 degC setpoint raise
SOLAR_KWH_PER_KWP_DAY = 4.2  # Bengaluru average yield

# Annual energy intensity targets (kWh per m2 of floor area) used to scale the
# source meters to Indian campus levels (ECBC / BEE benchmark ranges).
EUI_KWH_M2_YEAR = {"academic": 70, "hospital": 180, "hostel": 45, "canteen": 200, "parking": 8}

# Facilities response deadlines (hours) per alert kind.
DEADLINE_H = {"water": 4, "waste": 6, "energy": 24, "resource": 24}

# Campus clock runs on Indian Standard Time even when the server is in UTC (e.g. Hugging Face).
IST_OFFSET_H = 5.5

# Time-of-day tariff (assumed, modelled on KERC HT ToD): surcharge in peak windows, rebate at night.
TOD_PEAK_HOURS = [(6, 10), (18, 22)]
TOD_PEAK_SURCHARGE = 1.0     # Rs/kWh extra in peak windows
TOD_NIGHT_REBATE = 1.0       # Rs/kWh less from 22:00 to 06:00

# Water tanks: capacity, pump and which buildings each tank feeds (float switch: on < 35 %, off > 95 %).
TANKS = {
    "tank-soe-overhead": {"capacity_kl": 60, "pump_kw": 7.5, "pump_lph": 30000,
                          "feeds": ["soe-a-block", "soe-b-block", "soe-food-court"]},
    "tank-hostel-sump": {"capacity_kl": 120, "pump_kw": 11.0, "pump_lph": 45000,
                         "feeds": ["boys-hostel", "girls-hostel", "hostel-mess"]},
}
PUMP_ON_PCT, PUMP_OFF_PCT = 35, 95

# Indicative capital costs for the What-if payback (assumptions, shown in the UI).
CAPEX = {
    "ac_setpoint": 0,                 # policy + thermostat lock, no hardware
    "after_hours_per_building": 150000,  # smart relays / BMS schedules per building, scaled by the cut
    "led_per_w_saved": 25,            # Rs per W of lighting load removed (LED tube vs T8)
    "solar_per_kwp": 45000,           # rooftop solar, institutional, Rs per kWp
    "leak_fix": 40000,                # plumbing repair + a sub-meter
    "smart_pump": 30000,              # timer / level controller per pump
    "bin_pickup_per_year": 180000,    # one extra daily pickup round, staff time per year (opex)
}
LIGHTING_HOURS_YEAR = 3000

# Cloud "lite" mode: no PyTorch (YOLO/CLIP); photos are checked by Gemini vision instead.
# Turns on automatically when torch isn't installed, or with VERDANTIX_LITE=1.
import importlib.util as _ilu
LITE = os.getenv("VERDANTIX_LITE", "") == "1" or _ilu.find_spec("torch") is None
WS_INTERVAL_S = float(os.getenv("WS_INTERVAL", "3" if LITE else "1.5"))
VISION_MODEL = os.getenv("VISION_MODEL", GEMINI_MODEL)

# Access control (see backend/security.py). Set both on any public deployment.
OPERATOR_TOKEN = os.getenv("OPERATOR_TOKEN", "")   # facilities staff passcode (X-Operator header)
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")       # sensors / CCTV agents (X-Token header); falls back to OPERATOR_TOKEN
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "https://verdantix-frontend.vercel.app,http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
MAX_IMAGE_PIXELS = 60_000_000          # reject decompression bombs (a 12 MP photo is 12 000 000)
UPLOAD_RETENTION_DAYS = int(os.getenv("UPLOAD_RETENTION_DAYS", "30"))

# Persistence: alerts, work orders, reports, credits and the audit log live in SQLite.
DB_PATH = Path(os.getenv("VERDANTIX_DB", str(DATA / "state" / "verdantix.db")))

# CCTV incident rules (see app.cctv_frame)
CCTV_RAISE = float(os.getenv("CCTV_RAISE", "0.6"))       # model score needed to count a frame as a detection
CCTV_CLEAR = float(os.getenv("CCTV_CLEAR", "0.35"))      # score below which a frame counts as clean
CCTV_MIN_FRAMES = int(os.getenv("CCTV_MIN_FRAMES", "2"))  # consecutive detections before an incident opens
CCTV_CLEAN_FRAMES = int(os.getenv("CCTV_CLEAN_FRAMES", "3"))  # consecutive clean frames before it closes
CCTV_REOPEN_S = int(os.getenv("CCTV_REOPEN_S", "900"))   # same problem again within 15 min reopens the old incident

# Plausible ranges for sensor readings; anything outside is rejected and counted as a sensor error.
SENSOR_RANGES = {"power_kw": (0, 5000), "water_lph": (0, 200000), "bin_fill": (0, 100), "people": (0, 500), "tank_level_pct": (0, 100)}
SENSOR_FRESH_S, SENSOR_STALE_S = 120, 900
