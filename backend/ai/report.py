"""Weekly sustainability report: numbers from the twin, narrative from GenAI, PDF out."""
import io
import json
from datetime import datetime

import config as C
from ai.llm import LLMUnavailable, chat

PROMPT = """Write the weekly sustainability report for DSU Harohalli campus facilities from this JSON.
Sections (markdown headings): Executive summary (3 sentences), What went well, Top issues (cite alert IDs),
Recommended actions (table: action | owner team | deadline | Rs saved per year | CO2 saved), Student engagement.
Use only numbers in the JSON. Rupees as Rs. Keep it under 450 words.

DATA:
"""


def build(twin, reports_desk):
    sim = twin.clock.now()
    hist = twin.history_summary(7)
    st = twin.state()
    alerts = [a for a in st["alerts"] if a["status"] in ("open", "assigned", "resolved")]
    sim_plan = twin.simulate(ac_setpoint_delta=1, after_hours_cut=0.5, led_retrofit=True, fix_leaks=True, solar_kwp=0)
    data = {
        "week_ending": sim.strftime("%d %b %Y"),
        "totals": {"energy_kwh": hist["total_kwh"], "cost_inr": round(hist["total_kwh"] * C.TARIFF_INR_PER_KWH),
                   "co2_kg": round(hist["total_kwh"] * C.GRID_KG_CO2_PER_KWH), "water_kl": hist["total_water_kl"]},
        "buildings": hist["buildings"], "green_score": st["kpis"]["green_score"],
        "alerts": [{k: a[k] for k in ("id", "kind", "title", "severity", "inr_per_day", "status", "source")} for a in alerts[:15]],
        "work_orders": [{k: o[k] for k in ("id", "team", "status", "deadline")} for o in st["orders"][:10]],
        "student_reports": len(reports_desk.reports),
        "interventions_if_applied": sim_plan,
    }
    try:
        provider, msg = chat([{"role": "user", "content": PROMPT + json.dumps(data, default=str)}], max_tokens=1500)
        narrative = msg.get("content") or ""
    except LLMUnavailable:
        provider = "offline"
        worst = sorted(hist["buildings"].values(), key=lambda b: b["vs_expected_pct"], reverse=True)[:3]
        narrative = "## Executive summary\n" + (
            f"The campus used {data['totals']['energy_kwh']:,} kWh (Rs {data['totals']['cost_inr']:,}, {data['totals']['co2_kg']:,} kg CO2) "
            f"and {data['totals']['water_kl']} kL of water this week. Green score: {data['green_score']}/100.\n\n## Top issues\n"
            + "\n".join(f"- {b['name']}: {b['vs_expected_pct']:+}% vs expected, {b['night_kwh']} kWh at night" for b in worst)
            + f"\n\n## Recommended actions\n- Raise AC setpoint 1 degC, cut after-hours load 50%, LED retrofit, fix leaks: "
              f"Rs {sim_plan['inr_saved_year']:,}/year, {sim_plan['co2_t_saved_year']} t CO2/year.")
    return {"data": data, "narrative": narrative, "provider": provider, "generated": datetime.now().isoformat(timespec="minutes")}


def to_pdf(report):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="EcoTwin DSU weekly sustainability report")
    ss = getSampleStyleSheet()
    d = report["data"]
    story = [Paragraph("EcoTwin DSU - Weekly Sustainability Report", ss["Title"]),
             Paragraph(f"Week ending {d['week_ending']} - Harohalli campus - green score {d['green_score']}/100", ss["Normal"]),
             Spacer(1, 10)]
    t = d["totals"]
    kpi = Table([["Energy", "Cost", "CO2", "Water"],
                 [f"{t['energy_kwh']:,} kWh", f"Rs {t['cost_inr']:,}", f"{t['co2_kg']:,} kg", f"{t['water_kl']} kL"]])
    kpi.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f7a5c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                             ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [kpi, Spacer(1, 12)]
    rows = [["Building", "kWh", "vs expected", "Night kWh", "Water kL"]]
    for b in d["buildings"].values():
        rows.append([b["name"], f"{b['kwh']:,}", f"{b['vs_expected_pct']:+}%", f"{b['night_kwh']:,}", b["water_kl"]])
    bt = Table(rows)
    bt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dff1ea"))]))
    story += [bt, Spacer(1, 12)]
    for line in report["narrative"].splitlines():
        line = line.strip()
        if not line or set(line) <= {"|", "-", " ", ":"}:
            continue
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace("**", "")
        if line.startswith("#"):
            story.append(Paragraph(safe.lstrip("# "), ss["Heading2"]))
        elif line.startswith("|"):
            story.append(Paragraph(" - ".join(c.strip() for c in safe.strip("|").split("|")), ss["Normal"]))
        else:
            story.append(Paragraph(safe.lstrip("-* "), ss["Normal"]))
    story += [Spacer(1, 12), Paragraph(
        "Data: open datasets (BDG2, CU-BEMS, BattLeDIM, UCI Occupancy) mapped onto DSU buildings; "
        f"grid factor {C.GRID_KG_CO2_PER_KWH} kg CO2/kWh (CEA), tariff Rs {C.TARIFF_INR_PER_KWH}/kWh (assumed). "
        f"Narrative generated by {report['provider']}.", ss["Italic"])]
    doc.build(story)
    return buf.getvalue()
