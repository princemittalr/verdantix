"""Pretend to be ESP32 field sensors and stream readings into the twin's ingest API.

    venv\\Scripts\\python.exe scripts\\fake_sensor.py                      # local server
    venv\\Scripts\\python.exe scripts\\fake_sensor.py https://you-verdantix.hf.space

Streams three sensors every 3 s (standard library only, no installs):
  - an ultrasonic bin-level sensor on Bin-FoodCourt-2 that fills up until it overflows
  - a CT-clamp energy meter on SOE B Block that spikes after 30 s (the twin flags it)
  - a float/ultrasonic level sensor on the hostel sump tank
The dashboard shows an "IoT live" badge and LIVE tags; readings override the replay while fresh.
Real hardware uses exactly the same JSON: see hardware/esp32_sensor/esp32_sensor.ino.
"""
import json
import math
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
TOKEN = sys.argv[2] if len(sys.argv) > 2 else ""


def send(readings):
    req = urllib.request.Request(f"{BASE}/api/ingest", data=json.dumps(readings).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def expected_kw(building):
    try:
        with urllib.request.urlopen(f"{BASE}/api/state", timeout=20) as r:
            return float(json.loads(r.read())["buildings"][building]["baseline_kw"]) or 200.0
    except Exception:
        return 200.0


def main():
    print(f"Streaming to {BASE}/api/ingest  (Ctrl+C to stop)")
    t0, fill = time.time(), 40.0
    expected = expected_kw("soe-b-block")
    while True:
        t = time.time() - t0
        fill = min(fill + 3.5, 100.0)
        if int(t) % 30 < 3:
            expected = expected_kw("soe-b-block")   # follow the twin's clock
        base_kw = expected * (1 + 0.03 * math.sin(t / 7))
        power = base_kw * (3.0 if t > 30 else 1.0)      # someone leaves the chillers on
        tank = 55 + 30 * math.sin(t / 40)
        readings = [
            {"sensor": "esp32-bin-fc2", "asset": "bin-foodcourt-2", "metric": "bin_fill", "value": round(fill, 1)},
            {"sensor": "esp32-ct-soeb", "asset": "soe-b-block", "metric": "power_kw", "value": round(power, 1)},
            {"sensor": "esp32-lvl-sump", "asset": "tank-hostel-sump", "metric": "tank_level_pct", "value": round(tank, 1)},
        ]
        try:
            send(readings)
            print(f"t={t:5.0f}s  bin {fill:5.1f}%  SOE-B {power:6.1f} kW  sump {tank:4.1f}%")
        except Exception as e:
            print("send failed:", e)
        if fill >= 100:
            fill = 100.0
        time.sleep(3)


if __name__ == "__main__":
    main()
