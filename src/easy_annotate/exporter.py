"""Export annotations in YOLO format."""

import io
import zipfile

import yaml
from PIL import Image

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


def export_yolo(
        image: Image.Image,
        annotations: list[dict],
        filename: str,
) -> bytes:
    """Export image and annotations as a YOLO ZIP dataset."""
    width, height = image.size

    lines = []

    for box in annotations:
        class_id = CLASS_NAMES.index(box["class"])

        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        cx = (x1 + x2) / (2 * width)
        cy = (y1 + y2) / (2 * height)
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} "
            f"{bw:.6f} {bh:.6f}"
        )

    stem = filename.rsplit(".", 1)[0]

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")

    config = {
        "path": ".",
        "train": "images",
        "val": "images",
        "names": dict(enumerate(CLASS_NAMES)),
    }

    output = io.BytesIO()

    with zipfile.ZipFile(
            output, "w", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            f"images/{stem}.png",
            image_buffer.getvalue(),
        )

        archive.writestr(
            f"labels/{stem}.txt",
            "\n".join(lines) + ("\n" if lines else ""),
        )

        archive.writestr(
            "data.yaml",
            yaml.safe_dump(config, sort_keys=False),
        )

    return output.getvalue()
