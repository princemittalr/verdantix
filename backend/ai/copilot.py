"""Verdantix copilot: answers facilities questions from live twin data.

The model never invents numbers: every figure comes from a tool result, and
alerts are cited by ID. Without a reachable LLM, a rule-based answer is
built from the same tools so the demo never breaks.
"""
import json

from ai import local_llm
from ai.llm import PROVIDERS, LLMUnavailable, run_tools

SYSTEM = """You are Verdantix, the sustainability copilot for Dayananda Sagar University's Harohalli campus digital twin.
Facilities managers ask you what is wasting energy, water or resources and what to do about it.
Rules:
- Use the tools to get facts. Never invent numbers; quote figures exactly as tools return them.
- Cite alert IDs (e.g. A0012) and building names for every claim.
- Rank actions by rupee impact per day, and give each one an owner team and a deadline.
- Money in Indian rupees (Rs), CO2 in kg or tonnes. Be concise: short bullet points.
- If asked to create a work order, call create_work_order.
- Electricity has a time-of-day tariff (peak 06-10 and 18-22 cost more, 22-06 cheaper); suggest shifting flexible loads such as water pumping.
- Water tanks have float-switch pumps; an overflowing tank wastes water and pump energy.
- The data is real open datasets (BDG2, CU-BEMS, BattLeDIM, UCI) mapped onto DSU's buildings; say so if asked about data.
- Be evidence-driven: for each claim give the alert ID or building, the time (twin clock time from campus_overview),
  the source (dataset replay / measured sensor / camera AI / student report / simulated demo), and for savings say
  they are modelled potential savings with stated assumptions, not measured. Never claim work was completed unless
  a work order shows status done.
- Model scores are uncalibrated scores, not probabilities: write "model score 0.72", not "72% sure".
- End with one line "Assumptions & limits:" naming the main assumption or uncertainty."""

LANG = {
    "kn": "Reply in Kannada (ಕನ್ನಡ script). Keep alert IDs, numbers, units and building names exactly as given.",
    "hi": "Reply in Hindi (Devanagari script). Keep alert IDs, numbers, units and building names exactly as given.",
}

TOOLS = [
    {"type": "function", "function": {"name": "campus_overview", "description": "Campus KPIs now: power, today's energy, cost, CO2, water, green score, open alerts.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_alerts", "description": "Open alerts with impact. Optionally filter by kind.", "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["energy", "water", "waste", "resource", "all"]}}}}},
    {"type": "function", "function": {"name": "building_status", "description": "Live status, score and last 24h energy for one building.", "parameters": {"type": "object", "properties": {"building": {"type": "string", "description": "Building id or name, e.g. soe-a-block or 'Boys Hostel'"}}, "required": ["building"]}}},
    {"type": "function", "function": {"name": "simulate", "description": "What-if: annual savings from interventions.", "parameters": {"type": "object", "properties": {
        "ac_setpoint_delta": {"type": "number", "description": "Raise AC setpoint by N degC"},
        "after_hours_cut": {"type": "number", "description": "Fraction 0-1 of after-hours excess load removed"},
        "led_retrofit": {"type": "boolean"}, "fix_leaks": {"type": "boolean"},
        "solar_kwp": {"type": "number"}, "pickups_per_day": {"type": "integer"},
        "smart_pump_schedule": {"type": "boolean", "description": "Pump water tanks off-peak (22-06) instead of in peak hours"}}}}},
    {"type": "function", "function": {"name": "water_tanks", "description": "Water tank levels, pump state, overflow and peak-hour pumping today, plus the current tariff band.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "energy_forecast", "description": "Day-ahead (next 24 h) electricity forecast for one building: total kWh and expected peak.", "parameters": {"type": "object", "properties": {"building": {"type": "string"}}, "required": ["building"]}}},
    {"type": "function", "function": {"name": "bins_and_route", "description": "Bin fill levels, time-to-full forecast and optimised pickup route.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "create_work_order", "description": "Assign an alert to the facilities team with a deadline.", "parameters": {"type": "object", "properties": {"alert_id": {"type": "string"}, "note": {"type": "string"}}, "required": ["alert_id"]}}},
]


def make_handlers(twin, operator=False):
    def campus_overview():
        st = twin.state()
        return {"time": st["time"], **st["kpis"],
                "buildings": {b: {"name": s["name"], "score": s["score"], "grade": s["grade"], "status": s["status"]} for b, s in st["buildings"].items()}}

    def list_alerts(kind="all"):
        st = twin.state()
        al = [a for a in st["alerts"] if a["status"] in ("open", "assigned") and (kind in (None, "all") or a["kind"] == kind)]
        al.sort(key=lambda a: a["inr_per_day"], reverse=True)
        return [{k: a[k] for k in ("id", "kind", "title", "detail", "severity", "source", "inr_per_day", "co2_kg_per_day", "status", "work_order", "asset", "created")} for a in al]

    def building_status(building):
        st = twin.state()
        key = building.lower().strip()
        bid = next((b for b, s in st["buildings"].items() if key in (b, s["name"].lower()) or key in s["name"].lower()), None)
        if not bid:
            return {"error": f"unknown building {building}", "known": [s["name"] for s in st["buildings"].values()]}
        series = twin.building_series(bid, twin.clock.now())
        return {"id": bid, **st["buildings"][bid], "last_24h_kwh": round(sum(series["kwh"])),
                "baseline_24h_kwh": round(sum(series["baseline"]))}

    def simulate(**kw):
        r = twin.simulate(**kw)
        return {k: v for k, v in r.items() if k != "assumptions"}

    def water_tanks():
        st = twin.state()
        return {"tariff": st["tariff"], "tanks": {k: {x: v[x] for x in ("name", "level_pct", "pump_on", "overflow_lph",
                                                                            "peak_pump_min_today", "demand_lph", "feeds")}
                                                    for k, v in st["tanks"].items()}}

    def energy_forecast(building):
        st = twin.state()
        key = building.lower().strip()
        bid = next((b for b, s in st["buildings"].items() if key in (b, s["name"].lower()) or key in s["name"].lower()), None)
        if not bid:
            return {"error": f"unknown building {building}"}
        s = twin.building_series(bid, twin.clock.now())
        return {"building": st["buildings"][bid]["name"], "next_24h_kwh": s["forecast_kwh_24h"],
                "peak_kw": s["forecast_peak_kw"], "peak_at": s["forecast_peak_at"],
                "model_test_mape_pct": s["forecast_mape_pct"]}

    def bins_and_route():
        sim = twin.clock.now()
        r = twin.pickup_route(sim)
        return {"bins": twin.bin_state(sim), "route": {k: r[k] for k in ("stops", "distance_m", "saved_per_round", "saved_per_year",
                                                                          "pct_shorter", "skipped", "method")}}

    def create_work_order(alert_id, note=""):
        if not operator:
            return {"error": "Only signed-in facilities operators can create work orders. Ask an operator to assign it."}
        if alert_id not in twin.alerts:
            return {"error": f"unknown alert {alert_id}"}
        try:
            o = twin.create_order(alert_id, note, by="copilot (operator)")
        except ValueError as e:
            return {"error": str(e)}
        return {"work_order": o.id, "team": o.team, "deadline": o.deadline, "title": o.title}

    return {"campus_overview": campus_overview, "list_alerts": list_alerts, "building_status": building_status,
            "simulate": simulate, "bins_and_route": bins_and_route, "create_work_order": create_work_order,
            "water_tanks": water_tanks, "energy_forecast": energy_forecast}


def fallback_answer(handlers, question):
    """Deterministic answer from the same tools when no LLM is reachable."""
    ov = handlers["campus_overview"]()
    alerts = handlers["list_alerts"]()
    lines = [f"Campus green score {ov['green_score']}/100, {ov['open_alerts']} open alerts, "
             f"estimated waste Rs {ov['waste_inr_per_day']}/day."]
    if alerts:
        lines.append("Fix first (highest rupee impact):")
        for a in alerts[:5]:
            lines.append(f"- {a['id']} {a['title']}: Rs {a['inr_per_day']}/day, {a['co2_kg_per_day']} kg CO2/day ({a['severity']})")
    else:
        lines.append("No open issues right now.")
    lines.append("(Offline mode: rule-based answer from live twin data.)")
    return "\n".join(lines)


def ask(twin, question, history=None, lang="en", operator=False):
    handlers = make_handlers(twin, operator)
    system = SYSTEM + ("\n- " + LANG[lang] if lang in LANG else "")
    base = [{"role": "system", "content": system}] + (history or [])[-6:] + [{"role": "user", "content": question}]
    errors = []
    # Free providers in order: Gemini (free tier), Groq (free tier), local Qwen on CPU, then rule-based.
    for first in [name for name, _, key, _ in PROVIDERS if key()]:
        try:
            provider, text, log = run_tools(list(base), TOOLS, handlers, provider=first)
            return {"answer": text, "provider": provider, "tools": log}
        except LLMUnavailable as e:
            errors.append(str(e))
    if local_llm.available():
        try:
            facts, used = grounded_facts(handlers, question)
            text = local_llm.generate(system, f"Live twin data (JSON):\n{json.dumps(facts, default=str)[:6000]}\n\nQuestion: {question}")
            return {"answer": text, "provider": "local-qwen", "tools": [{"tool": t, "args": {}} for t in used], "error": " | ".join(errors)}
        except Exception as e:
            errors.append(f"local: {e}")
    return {"answer": fallback_answer(handlers, question), "provider": "offline", "tools": [], "error": " | ".join(errors)}


def grounded_facts(handlers, question):
    """Pick the tool results a question needs, so a small local model can answer from them."""
    q = question.lower()
    facts, used = {"overview": handlers["campus_overview"](), "alerts": handlers["list_alerts"]()[:8]}, ["campus_overview", "list_alerts"]
    if any(w in q for w in ("save", "saving", "what if", "what-if", "led", "solar", "setpoint", "ac ")):
        facts["simulation"] = handlers["simulate"](ac_setpoint_delta=1, after_hours_cut=0.5, led_retrofit=True, fix_leaks=True)
        used.append("simulate")
    if any(w in q for w in ("tank", "pump", "tariff", "peak", "overflow")):
        facts["tanks"] = handlers["water_tanks"]()
        used.append("water_tanks")
    if any(w in q for w in ("bin", "route", "pickup", "waste", "garbage")):
        r = handlers["bins_and_route"]()
        facts["bins"] = {k: {"fill": v["fill"], "eta_full_h": v["eta_full_h"]} for k, v in r["bins"].items()}
        facts["route"] = {"stops": r["route"]["stops"], "distance_m": r["route"]["distance_m"]}
        used.append("bins_and_route")
    return facts, used
