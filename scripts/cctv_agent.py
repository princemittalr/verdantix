"""CCTV agent: watches a real camera feed and lets the twin report problems on its own.

    venv\\Scripts\\python.exe scripts\\cctv_agent.py cam-foodcourt 0                      # laptop webcam
    venv\\Scripts\\python.exe scripts\\cctv_agent.py cam-foodcourt demo.mp4               # a recorded video
    venv\\Scripts\\python.exe scripts\\cctv_agent.py cam-soe-a-entrance rtsp://user:pass@192.168.1.20:554/stream1
    ... add  --server https://verdantix-api.onrender.com  to send to the cloud twin (default: http://localhost:8000)

Every few seconds it grabs a frame, shrinks it and sends it to POST /api/cctv/<camera>.
The server runs the vision models; when it sees garbage, a leak or lights on in an empty room it
raises an alert at that camera's map location, and closes it once the scene is clean for 3 frames.
Camera ids come from backend/data/campus.geojson (GET /api/cctv lists them). Most campus IP
cameras (Hikvision, CP Plus, Dahua) expose an RTSP URL like the one above.
"""
import argparse
import json
import time
import urllib.request
import uuid

import cv2


def post_frame(server, cam, jpeg):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"frame.jpg\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + jpeg + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{server}/api/cctv/{cam}", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camera", help="camera id on the twin, e.g. cam-foodcourt")
    ap.add_argument("source", help="0 for webcam, a video file, or an rtsp:// / http:// stream URL")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between analysed frames (default 5)")
    a = ap.parse_args()
    server = a.server.rstrip("/")
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Couldn't open {a.source}")
    is_file = isinstance(src, str) and not src.lower().startswith(("rtsp://", "http://", "https://"))
    print(f"Watching {a.source} as {a.camera}, sending a frame every {a.every:g}s to {server}  (Ctrl+C to stop)")
    last = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_file:  # loop recorded videos
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("Stream dropped, reconnecting in 3 s"); time.sleep(3); cap = cv2.VideoCapture(src); continue
        if time.time() - last < a.every:
            if is_file:
                time.sleep(1 / 30)
            continue
        last = time.time()
        h, w = frame.shape[:2]
        k = min(1.0, 960 / max(h, w))
        if k < 1:
            frame = cv2.resize(frame, (int(w * k), int(h * k)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            r = post_frame(server, a.camera, buf.tobytes())
            seen = f"{r['issue']} {round(r['score'] * 100)}%" if r["score"] >= 0.6 else "clear"
            print(f"{time.strftime('%H:%M:%S')}  {seen:<16} people={r['people']:<3} {'' if r['action'] == 'none' else '-> ' + r['action']}")
        except Exception as e:
            print("send failed:", e)


if __name__ == "__main__":
    main()
