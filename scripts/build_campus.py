"""Normalise the hand-drawn DSU GeoJSON into the twin's campus model.

geojson.io stored each feature as {"<Name>": "<type>"}; this rewrites it to
{"id", "name", "type", ...} and adds the sensor points the twin needs
(cameras, water tanks, water meter, extra bins).

Usage: python scripts/build_campus.py
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "dsu_campus.geojson"
OUT = ROOT / "backend" / "data" / "campus.geojson"

TYPE_FIX = {"PARKING": "parking", "MESS": "canteen"}
NICE_NAME = {
    "SOE B BLOCK": "SOE B Block",
    "CDSIMER BLOCK A": "CDSIMER Block A",
    "CDSIMER": "CDSIMER Hospital",
    "CRICKET GROUND": "Cricket Ground",
    "FOOTBALL GROUND": "Football Ground",
    "BASKETBALL COURT": "Basketball Court",
    "BOYS": "Boys Hostel",
    "GIRLS": "Girls Hostel",
    "MESS": "Hostel Mess",
    "PARKING": "Hostel Parking",
    "SOE FOOD COURT": "SOE Food Court",
}
# Floors drive the 3D extrusion height and per-floor drill-down.
FLOORS = {"academic": 5, "hospital": 6, "hostel": 6, "canteen": 1, "parking": 1, "sports": 0}

# Extra sensor points placed near the drawn buildings (lng, lat).
EXTRA_POINTS = [
    ("Bin-FoodCourt-1", "bin", 77.45178, 12.66255),
    ("Bin-FoodCourt-2", "bin", 77.45160, 12.66232),
    ("Bin-Hostel-Mess", "bin", 77.44600, 12.65775),
    ("Bin-Boys-Hostel", "bin", 77.44745, 12.65745),
    ("Bin-CDSIMER", "bin", 77.44985, 12.66090),
    ("Cam-SOE-A-Entrance", "camera", 77.45015, 12.66205),
    ("Cam-FoodCourt", "camera", 77.45170, 12.66245),
    ("Cam-Hostel-Mess", "camera", 77.44610, 12.65785),
    ("Tank-SOE-Overhead", "water_tank", 77.45060, 12.66180),
    ("Tank-Hostel-Sump", "water_tank", 77.44660, 12.65740),
    ("Meter-Main-Inlet", "water_meter", 77.44880, 12.65930),
]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def polygon_area_m2(ring):
    """Shoelace area on an equirectangular projection (fine at campus scale)."""
    lat0 = math.radians(sum(p[1] for p in ring) / len(ring))
    kx, ky = 111320 * math.cos(lat0), 110540
    pts = [(x * kx, y * ky) for x, y in ring]
    a = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))
    return abs(a) / 2


def centroid(ring):
    return [sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)]


def main():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    out = []
    for f in src["features"]:
        props = {k: v for k, v in f["properties"].items() if not k.startswith("@")}
        raw_name, ftype = next(iter(props.items()))
        ftype = TYPE_FIX.get(raw_name, ftype)
        name = NICE_NAME.get(raw_name, raw_name)
        p = {"id": slug(name), "name": name, "type": ftype}
        if f["geometry"]["type"] == "Polygon":
            ring = f["geometry"]["coordinates"][0]
            p["area_m2"] = round(polygon_area_m2(ring))
            p["floors"] = FLOORS.get(ftype, 1)
            p["center"] = centroid(ring)
        out.append({"type": "Feature", "properties": p, "geometry": f["geometry"]})

    for name, ftype, lng, lat in EXTRA_POINTS:
        out.append({
            "type": "Feature",
            "properties": {"id": slug(name), "name": name, "type": ftype},
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": out}, indent=1), encoding="utf-8")
    kinds = {}
    for f in out:
        kinds[f["properties"]["type"]] = kinds.get(f["properties"]["type"], 0) + 1
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} features {kinds}")


if __name__ == "__main__":
    main()
