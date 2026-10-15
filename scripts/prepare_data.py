"""Turn the raw open datasets into compact replay tables for the DSU twin.

Inputs  (datasets/, not committed):
  bdg2/archive.zip           Building Data Genome 2: hourly building meters
  cubems/2019Floor{2,3,4}    CU-BEMS: 1-min AC / lighting / plug load per zone
  occupancy/*.zip            UCI occupancy detection
  battledim/                 L-Town SCADA 2018 + leak labels
Outputs (backend/data/cache/):
  buildings.pkl   hourly elec_kwh / water_l per DSU building for one year
  rooms.pkl       15-min AC / light / plug kW, temp, RH, lux per SOE-A room
  water.pkl       5-min campus inlet flow + pressures + leak label
  mapping.json    which source series feeds which DSU asset (shown in the UI)

Usage: python scripts/prepare_data.py
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets"
OUT = ROOT / "backend" / "data" / "cache"
CAMPUS = ROOT / "backend" / "data" / "campus.geojson"

# Source building usage preferred for each DSU building type.
USAGE_PREF = {
    "academic": ["Education"],
    "hospital": ["Healthcare", "Education"],
    "hostel": ["Lodging/residential", "Education"],
    "canteen": ["Food sales and service", "Retail", "Education"],
    "parking": ["Parking", "Other"],
}
LITRES_PER_M2_DAY = {"academic": 4.0, "hospital": 12.0, "hostel": 9.0, "canteen": 15.0, "parking": 0.3}


def load_campus():
    g = json.loads(CAMPUS.read_text(encoding="utf-8"))
    return [f["properties"] for f in g["features"] if f["geometry"]["type"] == "Polygon" and f["properties"]["type"] in USAGE_PREF]


def prepare_buildings(mapping):
    z = zipfile.ZipFile(DS / "bdg2" / "archive.zip")
    meta = pd.read_csv(z.open("metadata.csv"))
    elec = pd.read_csv(z.open("electricity_cleaned.csv"), index_col="timestamp", parse_dates=True)
    water = pd.read_csv(z.open("water_cleaned.csv"), index_col="timestamp", parse_dates=True)
    elec, water = elec.loc["2017"], water.loc["2017"]

    def usable(cols, frame):
        ok = frame[cols].notna().mean() > 0.95
        return [c for c in cols if ok.get(c, False)]

    used = set()
    frames = {}
    for b in load_campus():
        gross = b["area_m2"] * max(b["floors"], 1)
        cand = meta[meta.primaryspaceusage.isin(USAGE_PREF[b["type"]]) & (meta.electricity == "Yes")]
        cand = cand[cand.building_id.isin(usable(list(cand.building_id), elec))]
        with_water = cand[cand.building_id.isin(usable([c for c in cand.building_id if c in water], water))]
        pool = with_water if len(with_water) else cand
        pool = pool[~pool.building_id.isin(used)]
        # Closest floor area to the DSU building keeps the load shape realistic.
        src = pool.iloc[(pool.sqm - gross).abs().argsort()].iloc[0]
        used.add(src.building_id)
        scale = gross / src.sqm

        e = elec[src.building_id].interpolate(limit=6).bfill() * scale
        if src.building_id in water.columns:
            w = water[src.building_id].interpolate(limit=6).bfill()
            w_shape = w / w.mean()
        else:
            w_shape = (e / e.mean()) ** 0.7
        litres_h = LITRES_PER_M2_DAY[b["type"]] * gross / 24
        frames[(b["id"], "elec_kwh")] = e.round(2)
        frames[(b["id"], "water_l")] = (w_shape * litres_h).round(1)
        mapping["buildings"][b["id"]] = {
            "source": src.building_id, "source_sqm": float(src.sqm), "dsu_gross_m2": gross,
            "scale": round(scale, 3), "water_source": "metered" if src.building_id in water.columns else "derived",
        }
        print(f"  {b['name']:<18} <- {src.building_id} (x{scale:.2f})")

    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_pickle(OUT / "buildings.pkl")
    print(f"buildings.pkl {df.shape}")


def prepare_rooms(mapping):
    frames = []
    for floor in (2, 3, 4):
        z = zipfile.ZipFile(DS / "cubems" / f"2019Floor{floor}.csv.zip")
        name = next(n for n in z.namelist() if n.endswith(".csv") and "MACOSX" not in n)
        df = pd.read_csv(z.open(name), parse_dates=["Date"], index_col="Date")
        df = df.apply(pd.to_numeric, errors="coerce").resample("15min").mean()
        zones = sorted({c.split("_")[0] for c in df.columns if c.startswith("z")})
        for zone in zones:
            cols = [c for c in df.columns if c.startswith(zone + "_")]
            ac = df[[c for c in cols if "AC" in c]].sum(axis=1, min_count=1)
            light = df[[c for c in cols if "Light" in c]].sum(axis=1, min_count=1)
            plug = df[[c for c in cols if "Plug" in c]].sum(axis=1, min_count=1)
            temp = df[[c for c in cols if "degC" in c]].mean(axis=1)
            rh = df[[c for c in cols if "RH" in c]].mean(axis=1)
            lux = df[[c for c in cols if "lux" in c]].mean(axis=1)
            room = f"soe-a-block/F{floor}-{zone.upper()}"
            part = pd.DataFrame({"ac_kw": ac, "light_kw": light, "plug_kw": plug, "temp_c": temp, "rh": rh, "lux": lux})
            part.columns = pd.MultiIndex.from_product([[room], part.columns])
            frames.append(part)
            mapping["rooms"][room] = {"source": f"CU-BEMS 2019 Floor{floor} {zone}"}
    rooms = pd.concat(frames, axis=1).interpolate(limit=8).round(3)
    rooms.to_pickle(OUT / "rooms.pkl")
    print(f"rooms.pkl {rooms.shape} ({len(frames)} rooms)")


def prepare_occupancy(mapping):
    z = zipfile.ZipFile(DS / "occupancy" / "occupancy+detection.zip")
    parts = []
    for n in z.namelist():
        if n.endswith(".txt"):
            parts.append(pd.read_csv(z.open(n)).assign(split=Path(n).stem))
    occ = pd.concat(parts, ignore_index=True)
    occ.to_pickle(OUT / "occupancy.pkl")
    mapping["occupancy"] = {"source": "UCI Occupancy Detection", "rows": len(occ), "splits": sorted(occ.split.unique().tolist())}
    print(f"occupancy.pkl {occ.shape}")


def prepare_water(mapping):
    xl = pd.ExcelFile(DS / "battledim" / "2018_SCADA.xlsx")
    sheets = {s.strip().lower(): s for s in xl.sheet_names}
    print("  SCADA sheets:", xl.sheet_names)

    def read(key):
        s = next(v for k, v in sheets.items() if key in k)
        df = pd.read_excel(xl, s)
        ts = df.columns[0]
        return df.set_index(pd.to_datetime(df[ts])).drop(columns=[ts])

    flows, pressures = read("flow"), read("pressure")
    leaks = pd.read_csv(DS / "battledim" / "2018_Leakages.csv", sep=";", decimal=",")
    ts = leaks.columns[0]
    leaks = leaks.set_index(pd.to_datetime(leaks[ts], format="ISO8601")).drop(columns=[ts]).apply(pd.to_numeric, errors="coerce")

    water = pd.DataFrame(index=flows.index)
    # L-Town flows are m3/h; campus inlet uses total inflow scaled to a campus-size demand.
    inflow = flows.sum(axis=1)
    water["inlet_lph"] = (inflow / inflow.mean() * 9000).round(1)
    for i, col in enumerate(pressures.columns[:6]):
        water[f"p{i + 1}_m"] = pressures[col].round(3)
    leak_total = leaks.reindex(water.index).fillna(0).sum(axis=1)
    water["leak_m3h"] = leak_total.round(4)
    water["leak_label"] = (leak_total > 0.5).astype(int)
    water.to_pickle(OUT / "water.pkl")
    onsets = {c: str(leaks[c][leaks[c] > 0].index.min()) for c in leaks.columns if (leaks[c] > 0).any()}
    mapping["water"] = {
        "leak_onsets": onsets,
        "source": "BattLeDIM 2018 L-Town SCADA", "pressure_sensors": list(pressures.columns[:6]),
        "leak_rows": int(water.leak_label.sum()), "rows": len(water),
    }
    print(f"water.pkl {water.shape}, leak-labelled rows {int(water.leak_label.sum())}")


def main():
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "mapping.json"
    mapping = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"buildings": {}, "rooms": {}}
    steps = [prepare_buildings, prepare_rooms, prepare_occupancy, prepare_water]
    only = sys.argv[1:]  # e.g. "water" to rerun one step
    for step in steps:
        if only and not any(o in step.__name__ for o in only):
            continue
        print(f"== {step.__name__}")
        step(mapping)
        (OUT / "mapping.json").write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
