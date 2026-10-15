"""Computer vision for the twin.

- YOLO11n (COCO) counts people for room occupancy.
- A YOLO11n fine-tuned on bin fill levels (Empty / Half / Overflow) when
  backend/weights/bins.pt exists.
- CLIP ViT-B/32 zero-shot covers issues with no training data: water leaks,
  running taps, lights on in empty rooms, litter. It also double-checks YOLO
  and verifies "after fixing" proof photos.
"""
import base64
import io
import json
import re
import threading
import time

import imagehash
import numpy as np
from PIL import Image

import config as C
from config import WEIGHTS


class VisionUnavailable(Exception):
    """The photo could not be checked (e.g. cloud vision API down or rate-limited)."""


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
LITE_PROMPT = """You are the camera sensor of a university campus sustainability digital twin.
Look at the photo and reply with ONLY this JSON (no markdown):
{"waste": 0-1, "water": 0-1, "energy": 0-1, "clean": 0-1, "people": <int>, "bin": "empty"|"half"|"overflow"|"none", "labels": ["<short description>", "..."]}
waste = overflowing bin, litter or dumped garbage. water = leaking pipe, puddle, tap left running.
energy = lights, fans, AC or projector switched on in an empty room. clean = none of these problems.
The four scores are probabilities that sum to 1. people = number of people visible. bin = fill level of the main
garbage bin if one is visible, else "none". labels = up to 3 short descriptions of what you see, most relevant first."""

ISSUE_PROMPTS = {
    "waste": ["a photo of an overflowing garbage bin", "a photo of litter and trash scattered on the ground",
              "a photo of a pile of garbage dumped on the roadside"],
    "water": ["a photo of water leaking from a pipe", "a photo of a large puddle of water on the floor",
              "a photo of a water tap left running"],
    "energy": ["a photo of an empty classroom with the lights on", "a photo of an empty room with lights and fans switched on",
               "a photo of a projector switched on in an empty room"],
    "clean": ["a photo of a clean empty corridor", "a photo of a classroom with lights off", "a photo of an empty garbage bin",
              "a photo of students sitting in a classroom", "a photo of a tidy campus lawn", "a photo of a dry clean floor",
              "a photo of a closed water tap", "a photo of a person", "a photo of an office desk"],
}
FILL_FROM_CLASS = {"empty": 10, "half": 55, "overflow": 100, "garbage_overflow": 100}


class Vision:
    def __init__(self):
        self._lock = threading.Lock()
        self._yolo = self._bins = self._clip = None
        self.lite = C.LITE
        self._last_call = 0.0
        if self.lite:
            self.status = {"yolo": "off (cloud lite mode)", "bins": "off (cloud lite mode)",
                           "clip": f"{C.VISION_MODEL} vision (cloud lite mode)"}
        else:
            self.status = {"yolo": "not loaded", "bins": "not loaded", "clip": "not loaded"}

    # ----- cloud lite mode: one Gemini vision call replaces YOLO + CLIP -----
    @staticmethod
    def _cheap_embedding(img):
        """Tiny colour-layout vector for the 'same problem reported nearby' check (stands in for CLIP)."""
        a = np.asarray(img.convert("RGB").resize((12, 12)), dtype=np.float32).ravel()
        a -= a.mean()
        n = np.linalg.norm(a)
        return a / n if n else a

    def _gemini(self, img):
        import httpx
        if not C.GEMINI_API_KEY:
            raise VisionUnavailable("Photo check unavailable: no GEMINI_API_KEY on the server.")
        small = img.convert("RGB").copy()
        small.thumbnail((768, 768))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=85)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"model": C.VISION_MODEL, "temperature": 0, "max_tokens": 1500,
                "messages": [{"role": "user", "content": [{"type": "text", "text": LITE_PROMPT},
                                                          {"type": "image_url", "image_url": {"url": url}}]}]}
        with self._lock:  # free tier: keep >= 4 s between calls
            wait = 4 - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                for attempt in range(2):
                    r = httpx.post(GEMINI_URL, json=body, timeout=40,
                                   headers={"Authorization": f"Bearer {C.GEMINI_API_KEY}"})
                    if r.status_code not in (429, 503) or attempt:
                        break
                    time.sleep(3)
            except Exception as e:
                raise VisionUnavailable(f"Photo check unavailable ({type(e).__name__}).")
            finally:
                self._last_call = time.time()
        if r.status_code != 200:
            raise VisionUnavailable(f"Photo check unavailable (vision API {r.status_code}).")
        text = r.json()["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        scores = {k: max(float(d.get(k, 0) or 0), 0.0) for k in ("waste", "water", "energy", "clean")}
        total = sum(scores.values())
        if not total:
            raise VisionUnavailable("Photo check returned no answer.")
        scores = {k: v / total for k, v in scores.items()}
        issue = max(("waste", "water", "energy"), key=scores.get)
        labels = [str(x)[:60] for x in (d.get("labels") or [])][:3] or [issue]
        clip = {"scores": {k: round(v, 3) for k, v in scores.items()}, "issue": issue,
                "issue_score": round(scores[issue], 3),
                "top": [{"label": l, "p": round(scores[issue] if i == 0 else 0, 3)} for i, l in enumerate(labels)]}
        b = str(d.get("bin", "none")).lower()
        bin_ = {"label": b.capitalize(), "conf": 0.8, "fill": FILL_FROM_CLASS[b]} if b in ("empty", "half", "overflow") else None
        try:
            people = int(d.get("people") or 0)
        except (TypeError, ValueError):
            people = 0
        return {"people": people, "objects": [], "bin": bin_, "clip": clip, "size": small.size}

    # Models load lazily so the API starts instantly.
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(str(WEIGHTS / "yolo11n.pt"))
            self.status["yolo"] = "yolo11n (COCO)"
        return self._yolo

    def bins(self):
        if self._bins is None and (WEIGHTS / "bins.pt").exists():
            from ultralytics import YOLO
            self._bins = YOLO(str(WEIGHTS / "bins.pt"))
            self.status["bins"] = "yolo11n fine-tuned: Empty/Half/Overflow"
        return self._bins

    def clip(self):
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.labels = [(cat, p) for cat, ps in ISSUE_PROMPTS.items() for p in ps]
            with torch.no_grad():
                t = proc(text=[p for _, p in self.labels], return_tensors="pt", padding=True)
                # Project the pooled outputs explicitly; works across transformers 4.x and 5.x.
                txt = model.text_projection(model.text_model(**t).pooler_output)
                self._txt = txt / txt.norm(dim=-1, keepdim=True)
            self._clip = (model, proc)
            self.status["clip"] = "openai/clip-vit-base-patch32 (zero-shot)"
        return self._clip

    def embed(self, img):
        import torch
        model, proc = self.clip()
        with torch.no_grad():
            px = proc(images=img, return_tensors="pt")["pixel_values"]
            f = model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
        return (f / f.norm(dim=-1, keepdim=True))[0]

    def classify(self, img):
        """Zero-shot issue scores per category, plus the image embedding."""
        with self._lock:
            emb = self.embed(img)
            logits = 100.0 * emb @ self._txt.T
            probs = logits.softmax(dim=-1).numpy()
        cats = {}
        for (cat, _), p in zip(self.labels, probs):
            cats[cat] = cats.get(cat, 0.0) + float(p)
        top = sorted(((float(p), prompt) for (_, prompt), p in zip(self.labels, probs)), reverse=True)[:3]
        issue = max((c for c in cats if c != "clean"), key=cats.get)
        return {"scores": {k: round(v, 3) for k, v in cats.items()}, "issue": issue, "issue_score": round(cats[issue], 3),
                "top": [{"label": l.replace("a photo of ", ""), "p": round(p, 3)} for p, l in top]}, emb.numpy()

    def detect(self, img):
        out = {"people": 0, "objects": [], "bin": None}
        with self._lock:
            res = self.yolo().predict(np.array(img), imgsz=640, conf=0.35, verbose=False)[0]
            for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                name = res.names[int(cls)]
                out["objects"].append({"label": name, "conf": round(conf, 2), "box": [round(v) for v in box]})
                out["people"] += name == "person"
            m = self.bins()
            if m is not None:
                r = m.predict(np.array(img), imgsz=640, conf=0.3, verbose=False)[0]
                best = None
                for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    name = r.names[int(cls)]
                    out["objects"].append({"label": f"bin:{name}", "conf": round(conf, 2), "box": [round(v) for v in box]})
                    if best is None or conf > best["conf"]:
                        best = {"label": name, "conf": round(conf, 2), "fill": FILL_FROM_CLASS.get(name.lower(), 50)}
                out["bin"] = best
        return out

    def analyze(self, img):
        if self.lite:
            out = self._gemini(img)
            return out, self._cheap_embedding(img)
        img = img.convert("RGB")
        img.thumbnail((960, 960))
        det = self.detect(img)
        clip, emb = self.classify(img)
        return {**det, "clip": clip, "size": img.size}, emb

    def verify_fixed(self, img, kind):
        """Proof photo passes when the original issue is no longer visible."""
        img = img.convert("RGB")
        kind = kind if kind in ISSUE_PROMPTS else "waste"
        if self.lite:
            r = self._gemini(img)
            score = r["clip"]["scores"].get(kind, 0)
            overflow = r["bin"] and r["bin"]["fill"] >= 100 and kind == "waste"
            return {"passed": score < 0.35 and not overflow, "issue_score": round(score, 3), "clip": r["clip"], "bin": r["bin"]}
        clip, _ = self.classify(img)
        score = clip["scores"].get(kind, 0)
        det = self.detect(img) if kind == "waste" else {"bin": None}
        overflow = det["bin"] and det["bin"]["fill"] >= 100
        return {"passed": score < 0.35 and not overflow, "issue_score": round(score, 3), "clip": clip, "bin": det["bin"]}


def phash(img):
    return imagehash.phash(img.convert("RGB"))
