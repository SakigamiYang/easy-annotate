"""YOLO object detection for EasyAnnotate."""

from collections import Counter
from pathlib import Path
from time import perf_counter

import streamlit as st
from PIL import Image
from loguru import logger
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "bdd100k-yolo11n.pt"


@st.cache_resource
def load_model():
    """Load the pretrained model once."""
    logger.info("Loading model: {}", MODEL_PATH)

    if not MODEL_PATH.is_file():
        logger.error("Model file not found: {}", MODEL_PATH)
        raise FileNotFoundError(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    logger.info("Model loaded: classes={}", model.names)
    return model


def detect_objects(
        image: Image.Image,
        confidence: float = 0.25,
) -> list[dict]:
    """Run object detection and return bounding boxes."""
    logger.info(
        "Starting inference: image_size={}, conf={}",
        image.size,
        confidence,
    )

    start = perf_counter()
    model = load_model()

    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        device="cpu",
        verbose=False,
    )[0]

    boxes = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls.item())

        boxes.append({
            "class": model.names[class_id],
            "confidence": round(float(box.conf.item()), 4),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    logger.info(
        "Inference completed: count={}, duration={:.3f}s",
        len(boxes),
        perf_counter() - start,
    )

    logger.info(
        "Detection classes: {}",
        dict(Counter(box["class"] for box in boxes)),
    )

    return boxes
