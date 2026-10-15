"""Recurring-problem hotspots: DBSCAN over every alert location.

Events are weighted by recency (30-day half-life). Clusters with more weight
and more repeats get higher priority and a suggested permanent fix.
"""
import math
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN

EARTH_M = 6371000
FIX = {
    "waste": "Add a second bin or an extra midday pickup here.",
    "water": "Schedule a plumbing inspection of this line; consider a sub-meter.",
    "energy": "Install occupancy sensors / auto-off timers for this block.",
    "resource": "Review usage schedule with the block coordinator.",
}


def hotspots(events, now, eps_m=40, min_samples=3):
    if len(events) < min_samples:
        return []
    X = np.radians([[e["lat"], e["lng"]] for e in events])
    labels = DBSCAN(eps=eps_m / EARTH_M, min_samples=min_samples, metric="haversine").fit(X).labels_
    out = []
    for lab in set(labels) - {-1}:
        members = [e for e, l in zip(events, labels) if l == lab]
        weight = sum(0.5 ** ((now - datetime.fromisoformat(e["at"])).total_seconds() / 86400 / 30) for e in members)
        kinds = {}
        for e in members:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        main = max(kinds, key=kinds.get)
        out.append({
            "lng": float(np.mean([e["lng"] for e in members])), "lat": float(np.mean([e["lat"] for e in members])),
            "events": len(members), "weight": round(weight, 1), "kinds": kinds, "main_kind": main,
            "assets": sorted({e["asset"] for e in members}),
            "priority": "high" if weight >= 6 else "medium" if weight >= 3 else "low",
            "suggestion": FIX[main],
        })
    return sorted(out, key=lambda h: h["weight"], reverse=True)
