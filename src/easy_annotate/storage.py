"""Local annotation persistence."""

import json
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
ANNOTATION_DIR = DATA_DIR / "annotations"


def save_annotations(
        image_id: str,
        filename: str,
        image_bytes: bytes,
        image_size: tuple[int, int],
        canvas_size: tuple[int, int],
        drawing: dict,
        annotations: list[dict],
) -> Path:
    """Save image and annotation state locally."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    image_path = IMAGE_DIR / f"{image_id}.png"
    annotation_path = ANNOTATION_DIR / f"{image_id}.json"

    # Store the original image bytes separately in app.py
    # as normalized PNG bytes.
    image_path.write_bytes(image_bytes)

    payload = {
        "image_id": image_id,
        "filename": filename,
        "width": image_size[0],
        "height": image_size[1],
        "canvas_width": canvas_size[0],
        "canvas_height": canvas_size[1],
        "annotations": annotations,
        "drawing": drawing,
    }

    annotation_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Saved {} annotations to {}",
        len(annotations),
        annotation_path,
    )

    return annotation_path


def load_annotations(image_id: str) -> dict | None:
    """Load saved annotations for an image."""
    path = ANNOTATION_DIR / f"{image_id}.json"

    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Loaded saved annotations: {}", path)
    return payload
