from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

model = YOLO(
    str(ROOT / "models" / "bdd100k-yolo11n.pt")
)

metrics = model.val(
    data=str(ROOT / "datasets" / "bdd100k-road" / "road.yaml"),
    device="cpu",
    imgsz=640,
    plots=True,
    project=str(ROOT / "results"),
    name="model-evaluation",
)

print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")
