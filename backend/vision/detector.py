"""Computer vision for the twin.

- YOLO11n (COCO) counts people for room occupancy.
- A YOLO11n fine-tuned on bin fill levels (Empty / Half / Overflow) when
  backend/weights/bins.pt exists.
- CLIP ViT-B/32 zero-shot covers issues with no training data: water leaks,
  running taps, lights on in empty rooms, litter. It also double-checks YOLO
  and verifies "after fixing" proof photos.
"""
import threading

import imagehash
import numpy as np
from PIL import Image

from config import WEIGHTS

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
        self.status = {"yolo": "not loaded", "bins": "not loaded", "clip": "not loaded"}

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
        img = img.convert("RGB")
        img.thumbnail((960, 960))
        det = self.detect(img)
        clip, emb = self.classify(img)
        return {**det, "clip": clip, "size": img.size}, emb

    def verify_fixed(self, img, kind):
        """Proof photo passes when the original issue is no longer visible."""
        img = img.convert("RGB")
        clip, _ = self.classify(img)
        score = clip["scores"].get(kind if kind in ISSUE_PROMPTS else "waste", 0)
        det = self.detect(img) if kind == "waste" else {"bin": None}
        overflow = det["bin"] and det["bin"]["fill"] >= 100
        return {"passed": score < 0.35 and not overflow, "issue_score": round(score, 3), "clip": clip, "bin": det["bin"]}


def phash(img):
    return imagehash.phash(img.convert("RGB"))
