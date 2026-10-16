"""What kind of waste is in a photo, which campus team should handle it, and how fast.

Adapted from the team's Clean City app (garbage_types.py), retargeted from BBMP street crews to DSU campus teams.
Full mode: CLIP zero-shot over the descriptions below. Lite mode: the cloud vision model names the type directly.
Hazardous types get shorter deadlines and their own team, so a syringe pile never waits behind a full bin.
"""

# key: (name, team, deadline in hours, CLIP descriptions)
TYPES = {
    "mixed": ("Mixed waste", "Housekeeping", 6, [
        "a photo of mixed garbage bags dumped on the ground",
        "a photo of an overflowing garbage bin with mixed trash"]),
    "plastic": ("Plastic and dry waste", "Housekeeping (dry waste)", 6, [
        "a photo of plastic bottles, wrappers and paper cups littered on the ground",
        "a photo of a heap of plastic and paper waste"]),
    "food": ("Food and wet waste", "Mess and housekeeping (wet waste)", 3, [
        "a photo of leftover food and wet kitchen waste spilling from a bin",
        "a photo of rotting food waste on the ground"]),
    "construction": ("Construction debris", "Estate civil works", 48, [
        "a photo of construction debris, broken bricks and concrete rubble",
        "a photo of a pile of sand, stones and building rubble"]),
    "garden": ("Garden waste", "Gardening team", 24, [
        "a photo of cut branches, dry leaves and garden waste in a pile",
        "a photo of a heap of fallen leaves and trimmed plants"]),
    "bulky": ("Furniture and bulky items", "Estate stores (bulky pickup)", 48, [
        "a photo of broken chairs, desks or an old mattress dumped outside",
        "a photo of large discarded furniture on a corridor or footpath"]),
    "ewaste": ("E-waste", "IT stores (authorised e-waste recycler)", 48, [
        "a photo of discarded electronics, old computers, keyboards and cables",
        "a photo of broken monitors, circuit boards and batteries thrown away"]),
    "medical": ("Biomedical waste", "Campus clinic (biomedical waste, handle with care)", 2, [
        "a photo of used syringes, masks, gloves and medicine strips thrown away",
        "a photo of hospital or lab biomedical waste lying on the ground"]),
    "dead_animal": ("Dead animal", "Estate office (carcass removal)", 4, [
        "a photo of a dead animal lying on the ground"]),
}
DEFAULT = "mixed"
MIN_CONFIDENCE = 0.35   # CLIP: below this share the type falls back to mixed waste
HAZARDOUS = {"medical", "dead_animal"}


def info(key, conf=None):
    key = key if key in TYPES else DEFAULT
    name, team, hours, _ = TYPES[key]
    out = {"key": key, "name": name, "team": team, "deadline_h": hours, "hazardous": key in HAZARDOUS}
    if conf is not None:
        out["conf"] = round(float(conf), 2)
    return out


def prompts():
    return [(k, p) for k, (_, _, _, ps) in TYPES.items() for p in ps]
