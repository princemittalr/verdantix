"""Fine-tune YOLO11n on bin fill levels (Empty / Half / Overflow).

Dataset: Roboflow 'garbage_fill_level' v2 (CC BY 4.0), 622 images.
Writes backend/weights/bins.pt and backend/data/cache/vision_metrics.json.

Usage: python scripts/train_bins.py [epochs] [imgsz]
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "datasets" / "vision" / "Garbage_Fill_Level.v2i.yolov8.zip"
DATA = ROOT / "datasets" / "vision" / "fill_level"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 416


def main():
    if not (DATA / "train").exists():
        zipfile.ZipFile(ZIP).extractall(DATA)
    yaml = DATA / "verdantix.yaml"
    yaml.write_text(
        f"path: {DATA.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 3\nnames: ['Empty', 'Half', 'Overflow']\n", encoding="utf-8")

    model = YOLO(str(ROOT / "backend" / "weights" / "yolo11n.pt"))
    model.train(data=str(yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=16, device="cpu", workers=2,
                project=str(ROOT / "runs"), name="bins", exist_ok=True, patience=8, plots=True, verbose=False)
    best = ROOT / "runs" / "bins" / "weights" / "best.pt"
    shutil.copy(best, ROOT / "backend" / "weights" / "bins.pt")

    m = YOLO(str(best)).val(data=str(yaml), split="test", imgsz=IMGSZ, device="cpu", verbose=False)
    metrics = {"model": "yolo11n fine-tuned", "dataset": "Roboflow garbage_fill_level v2 (CC BY 4.0)",
               "epochs": EPOCHS, "imgsz": IMGSZ, "test_map50": round(float(m.box.map50), 3),
               "test_map50_95": round(float(m.box.map), 3),
               "per_class_map50": {n: round(float(v), 3) for n, v in zip(m.names.values(), m.box.ap50)}}
    (ROOT / "backend" / "data" / "cache" / "vision_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(metrics)


if __name__ == "__main__":
    main()
