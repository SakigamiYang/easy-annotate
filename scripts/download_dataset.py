"""Download BDD100K validation samples with human annotations."""

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "bdd100k-road"

REPO_ID = "dronefreak/BDD100K"
SPLIT = "val"
SHARD = "shard_000"

IMAGE_PREFIX = f"data/images_100k/100k/{SPLIT}/{SHARD}"
LABEL_PREFIX = f"data/labels/100k/{SPLIT}/{SHARD}"

CLASS_NAMES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
]

CLASS_TO_ID = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}


def list_images():
    """List images from the verified HF directory."""
    api = HfApi()
    images = []

    print(f"Listing {IMAGE_PREFIX}...")

    for entry in api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo=IMAGE_PREFIX,
            recursive=False,
    ):
        if entry.path.lower().endswith(
                (".jpg", ".jpeg", ".png")
        ):
            images.append(entry.path)

    if not images:
        raise RuntimeError("No images found")

    print(f"Found {len(images)} images")
    return sorted(images)


def download_file(remote_path):
    """Download one file, using the local HF cache."""
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote_path,
        )
    )


def convert_annotations(data, width, height):
    """Convert BDD100K human boxes to YOLO format."""
    lines = []

    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            category = obj.get("category")
            box = obj.get("box2d")

            if category not in CLASS_TO_ID or not box:
                continue

            x1 = max(0.0, min(width, float(box["x1"])))
            y1 = max(0.0, min(height, float(box["y1"])))
            x2 = max(0.0, min(width, float(box["x2"])))
            y2 = max(0.0, min(height, float(box["y2"])))

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / (2 * width)
            cy = (y1 + y2) / (2 * height)
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height

            class_id = CLASS_TO_ID[category]

            lines.append(
                f"{class_id} "
                f"{cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )

    return lines


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")

    image_dir = DATASET_DIR / "images" / "val"
    label_dir = DATASET_DIR / "labels" / "val"
    raw_dir = DATASET_DIR / "raw"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    images = list_images()

    if len(images) < args.count:
        raise RuntimeError(
            f"Only {len(images)} images available"
        )

    rng = random.Random(args.seed)
    rng.shuffle(images)

    selected = []
    total_boxes = 0

    for image_path in images:
        if len(selected) >= args.count:
            break

        name = Path(image_path).name
        stem = Path(name).stem

        label_path = f"{LABEL_PREFIX}/{stem}.json"

        try:
            # Original human annotations
            source_json = download_file(label_path)
            annotation = json.loads(
                source_json.read_text(encoding="utf-8")
            )

            # Original image
            source_image = download_file(image_path)

            with Image.open(source_image) as image:
                width, height = image.size
                image.verify()

            # Convert human labels to YOLO TXT
            lines = convert_annotations(
                annotation, width, height
            )

        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue

        shutil.copy2(source_image, image_dir / name)
        shutil.copy2(source_json, raw_dir / f"{stem}.json")

        (label_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

        sample = {
            "id": len(selected),
            "image": name,
            "label": f"{stem}.txt",
            "raw_label": f"{stem}.json",
            "boxes": len(lines),
        }

        selected.append(sample)
        total_boxes += len(lines)

        print(
            f"[{len(selected)}/{args.count}] "
            f"{name}: {len(lines)} boxes"
        )

    if len(selected) != args.count:
        raise RuntimeError(
            f"Only downloaded {len(selected)} "
            f"of {args.count} samples"
        )

    # Remove files not in the current selection.
    keep = {item["image"] for item in selected}
    stems = {Path(name).stem for name in keep}

    for path in image_dir.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()

    for path in label_dir.glob("*.txt"):
        if path.stem not in stems:
            path.unlink()

    for path in raw_dir.glob("*.json"):
        if path.stem not in stems:
            path.unlink()

    # Ultralytics evaluation configuration
    config = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/val",
        "val": "images/val",
        "names": dict(enumerate(CLASS_NAMES)),
    }

    with (DATASET_DIR / "road.yaml").open(
            "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    # Keep compatibility with show_annotation.py
    manifest = {
        "source": REPO_ID,
        "split": SPLIT,
        "seed": args.seed,
        "images": [s["image"] for s in selected],
        "samples": selected,
    }

    (DATASET_DIR / "samples.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nDownload complete")
    print(f"Images: {len(selected)}")
    print(f"Human bounding boxes: {total_boxes}")
    print(f"Dataset: {DATASET_DIR}")


if __name__ == "__main__":
    main()
