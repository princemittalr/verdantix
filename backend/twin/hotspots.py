"""Recurring-problem hotspots: DBSCAN over every alert location, plus when each one usually happens.

Events are weighted by recency (30-day half-life). Clusters with more weight get higher priority and a
suggested permanent fix. Adapted from the team's Clean City app (hotspots.py), scaled from city streets
to a campus (40 m radius instead of 200 m):
- time pattern: the 3-hour window of the day when the problem usually appears, the busiest weekday, and a
  patrol time one hour before that window (the problem starts before somebody notices it);
- recurrence: the average gap between separate occurrences and, from the last one, when it is likely next.
  This is a simple average, not a trained forecasting model, and it is only shown with enough history.
"""
from datetime import datetime, timedelta

import numpy as np
from sklearn.cluster import DBSCAN

EARTH_M = 6371000
FIX = {
    "waste": "Add a second bin or an extra midday pickup here.",
    "water": "Schedule a plumbing inspection of this line; consider a sub-meter.",
    "energy": "Install occupancy sensors / auto-off timers for this block.",
    "resource": "Review usage schedule with the block coordinator.",
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WINDOW_H = 3
MIN_TIMING = 4          # events needed before a time pattern is claimed
MIN_PEAK_SHARE = 0.4    # the window must hold at least 40% of events
MIN_DAY_SHARE = 0.3
SAME_OCCURRENCE_H = 6   # events closer than this are the same occurrence, not a recurrence
MIN_OCCURRENCES = 3


def hour_label(h):
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def timing(times):
    hours, days = [0] * 24, [0] * 7
    for t in times:
        hours[t.hour] += 1
        days[t.weekday()] += 1
    out = {"hours": hours, "peak": None, "share": None, "patrol": None, "busy_day": None}
    if len(times) < MIN_TIMING:
        return out
    start = max(range(24), key=lambda s: sum(hours[(s + k) % 24] for k in range(WINDOW_H)))
    for _ in range(WINDOW_H - 1):   # tighten: start the window at its first busy hour
        if hours[start % 24]:
            break
        start = (start + 1) % 24
    share = sum(hours[(start + k) % 24] for k in range(WINDOW_H)) / len(times)
    if share >= MIN_PEAK_SHARE:
        out.update(peak=f"{hour_label(start)} to {hour_label((start + WINDOW_H) % 24)}", share=round(share, 2),
                   patrol=hour_label((start - 1) % 24))
    top = max(range(7), key=lambda d: days[d])
    if days[top] >= 2 and days[top] / len(times) >= MIN_DAY_SHARE:
        out["busy_day"] = DAY_NAMES[top]
    return out


def recurrence(times, now):
    occ = []
    for t in sorted(times):
        if not occ or (t - occ[-1]).total_seconds() / 3600 >= SAME_OCCURRENCE_H:
            occ.append(t)
    if len(occ) < MIN_OCCURRENCES:
        return {"occurrences": len(occ), "every_days": None, "next": None}
    gap = sum((b - a).total_seconds() for a, b in zip(occ, occ[1:])) / (len(occ) - 1) / 86400
    nxt = occ[-1] + timedelta(days=gap)
    while nxt < now:      # already overdue by the average: the next one is due any time
        nxt += timedelta(days=gap)
    nxt = nxt.replace(minute=0, second=0, microsecond=0)
    return {"occurrences": len(occ), "every_days": round(gap, 1), "next": nxt.isoformat(timespec="minutes")}


def hotspots(events, now, eps_m=40, min_samples=3, names=None):
    if len(events) < min_samples:
        return []
    X = np.radians([[e["lat"], e["lng"]] for e in events])
    labels = DBSCAN(eps=eps_m / EARTH_M, min_samples=min_samples, metric="haversine").fit(X).labels_
    out = []
    for lab in set(labels) - {-1}:
        members = [e for e, l in zip(events, labels) if l == lab]
        times = [datetime.fromisoformat(e["at"]) for e in members]
        weight = sum(0.5 ** ((now - t).total_seconds() / 86400 / 30) for t in times)
        kinds = {}
        for e in members:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        main = max(kinds, key=kinds.get)
        assets = sorted({e["asset"] for e in members})
        tm, rec = timing(times), recurrence(times, now)
        plan = FIX[main]
        if tm["patrol"]:
            plan = f"Check here around {tm['patrol']}{' on ' + tm['busy_day'] + 's' if tm['busy_day'] else ''}, before the usual {tm['peak']} window. " + plan
        out.append({
            "lng": float(np.mean([e["lng"] for e in members])), "lat": float(np.mean([e["lat"] for e in members])),
            "events": len(members), "weight": round(weight, 1), "kinds": kinds, "main_kind": main,
            "assets": assets, "names": [(names or {}).get(a, a) for a in assets],
            "priority": "high" if weight >= 6 else "medium" if weight >= 3 else "low",
            "suggestion": plan, "timing": tm, "recurrence": rec,
            "last": max(times).isoformat(timespec="minutes"),
        })
    return sorted(out, key=lambda h: h["weight"], reverse=True)
