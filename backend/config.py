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

# Optional shared secret for the IoT ingest API (set in .env for a public deployment).
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
