"""Display a BDD100K image and its annotations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "datasets" / "bdd100k-road"
MODEL = ROOT / "models" / "bdd100k-yolo11n.pt"

NAMES = [
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


def ground_truth(path, width, height):
    boxes = []

    for line in path.read_text(
            encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        cls, xc, yc, bw, bh = map(
            float, line.split()
        )

        x = (xc - bw / 2) * width
        y = (yc - bh / 2) * height

        boxes.append(
            (
                int(cls),
                x,
                y,
                bw * width,
                bh * height,
                None,
            )
        )

    return boxes


def plot_boxes(ax, img, boxes, title, edge):
    ax.imshow(img)

    for cls, x, y, w, h, conf in boxes:
        rect = Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor=edge,
            linewidth=1.6,
        )

        ax.add_patch(rect)

        label = NAMES[cls]

        if conf is not None:
            label += f" {conf:.2f}"

        ax.text(
            x,
            max(0, y - 4),
            label,
            color="white",
            fontsize=7,
            bbox={
                "facecolor": edge,
                "alpha": 0.8,
                "pad": 1,
                "edgecolor": "none",
            },
        )

    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "id",
        type=int,
        help="Image ID in samples.json",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare human labels and AI predictions",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--save",
        type=Path,
        help="Save the visualization",
    )

    args = parser.parse_args()

    manifest = json.loads(
        (DATA / "samples.json").read_text(
            encoding="utf-8"
        )
    )

    images = manifest["images"]

    if not 0 <= args.id < len(images):
        parser.error(
            f"Image ID must be 0 to {len(images) - 1}"
        )

    name = images[args.id]

    image_path = DATA / "images" / "val" / name
    label_path = (
            DATA / "labels" / "val"
            / f"{Path(name).stem}.txt"
    )

    img = Image.open(image_path).convert("RGB")

    gt = ground_truth(
        label_path,
        *img.size,
    )

    columns = 2 if args.compare else 1

    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(16 if args.compare else 10, 7),
    )

    ax_gt = axes[0] if args.compare else axes

    plot_boxes(
        ax_gt,
        img,
        gt,
        f"ID {args.id}: Human Ground Truth",
        "limegreen",
    )

    if args.compare:
        if not MODEL.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL}"
            )

        model = YOLO(str(MODEL))

        result = model.predict(
            source=img,
            conf=args.conf,
            device="cpu",
            verbose=False,
        )[0]

        predictions = []

        for box in result.boxes:
            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            predictions.append(
                (
                    int(box.cls.item()),
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    float(box.conf.item()),
                )
            )

        plot_boxes(
            axes[1],
            img,
            predictions,
            f"YOLO Predictions ({len(predictions)})",
            "orangered",
        )

    fig.tight_layout()

    if args.save:
        args.save.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            args.save,
            dpi=160,
        )

        print(f"Saved: {args.save}")
    else:
        plt.show()

    plt.close(fig)

    print(f"Image ID: {args.id}")
    print(f"Filename: {name}")
    print(f"Human annotations: {len(gt)}")


if __name__ == "__main__":
    main()
