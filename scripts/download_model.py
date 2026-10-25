"""Download BDD100K pretrained YOLO11n."""

from pathlib import Path
from shutil import copy2

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "bdd100k-yolo11n.pt"

REPO = "dronefreak/bdd100k-yolo11n"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if not DEST.exists():
        source = hf_hub_download(
            repo_id=REPO,
            filename="best.pt",
        )
        copy2(source, DEST)

    model = YOLO(str(DEST))

    print(f"Model saved to: {DEST}")
    print("Model classes:")
    for class_id, name in model.names.items():
        print(f"  {class_id}: {name}")


if __name__ == "__main__":
    main()
