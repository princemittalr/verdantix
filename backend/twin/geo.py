"""Small geometry helpers for the campus model (no GIS dependencies)."""
import json
import math

from config import DATA


def haversine_m(lng1, lat1, lng2, lat2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def point_in_ring(lng, lat, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lng < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi:
            inside = not inside
        j = i
    return inside


class Campus:
    def __init__(self):
        self.geojson = json.loads((DATA / "campus.geojson").read_text(encoding="utf-8"))
        self.features = {f["properties"]["id"]: f for f in self.geojson["features"]}
        self.buildings = {k: f["properties"] for k, f in self.features.items()
                          if f["geometry"]["type"] == "Polygon" and f["properties"]["type"] not in ("sports",)}
        self.points = {k: f for k, f in self.features.items() if f["geometry"]["type"] == "Point"}

    def location(self, fid):
        f = self.features[fid]
        if f["geometry"]["type"] == "Point":
            return f["geometry"]["coordinates"]
        return f["properties"]["center"]

    def of_type(self, t):
        return [k for k, f in self.features.items() if f["properties"]["type"] == t]

    def locate(self, lng, lat):
        """Polygon containing the point, else the nearest feature within 80 m."""
        for k, f in self.features.items():
            if f["geometry"]["type"] == "Polygon" and point_in_ring(lng, lat, f["geometry"]["coordinates"][0]):
                return k, 0.0
        best = min(self.features, key=lambda k: haversine_m(lng, lat, *self.location(k)))
        d = haversine_m(lng, lat, *self.location(best))
        return (best, d) if d <= 80 else (None, d)

    def nearest_building(self, fid):
        lng, lat = self.location(fid)
        return min(self.buildings, key=lambda k: haversine_m(lng, lat, *self.location(k)))
