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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

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
