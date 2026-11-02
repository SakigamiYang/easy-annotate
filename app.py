"""EasyAnnotate: AI-assisted image annotation."""

import hashlib
import io
from copy import deepcopy

import streamlit as st
from PIL import Image
from loguru import logger
from streamlit_drawable_canvas import (
    boxes_to_drawing,
    st_canvas,
)

from easy_annotate.detector import detect_objects
from easy_annotate.exporter import export_yolo
from easy_annotate.storage import (
    load_annotations,
    save_annotations,
)

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

MAX_CANVAS_WIDTH = 900

st.set_page_config(
    page_title="EasyAnnotate",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state():
    defaults = {
        "image_id": None,
        "drawing": {"objects": []},
        "canvas_initial": {"objects": []},
        "canvas_version": 0,
        "pending_canvas_update": False,
        "mode": "Manual",
        "prediction_count": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)


def reset_canvas():
    """Remount Canvas after programmatic changes."""
    st.session_state.canvas_initial = deepcopy(
        st.session_state.drawing
    )
    st.session_state.canvas_version += 1
    st.session_state.pending_canvas_update = True


def canvas_objects(drawing):
    """Get labeled rectangles from Fabric canvas JSON."""
    return [
        obj
        for obj in drawing.get("objects", [])
        if obj.get("type", "").lower()
           in {"labeledrect", "rect"}
    ]


def canvas_signature(drawing):
    """Identify canvas contents during initialization."""
    return [
        (
            obj.get("type"),
            obj.get("label"),
            round(float(obj.get("left", 0)), 2),
            round(float(obj.get("top", 0)), 2),
            round(float(obj.get("width", 0)), 2),
            round(float(obj.get("height", 0)), 2),
        )
        for obj in canvas_objects(drawing)
    ]


def get_annotations(drawing, sx, sy, image_size):
    """Convert canvas objects to original image coordinates."""
    image_width, image_height = image_size
    annotations = []

    for obj in canvas_objects(drawing):
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        width = (
                float(obj.get("width", 0))
                * float(obj.get("scaleX", 1))
        )
        height = (
                float(obj.get("height", 0))
                * float(obj.get("scaleY", 1))
        )

        x1 = max(0, min(min(left, left + width) * sx, image_width))
        y1 = max(0, min(min(top, top + height) * sy, image_height))
        x2 = max(0, min(max(left, left + width) * sx, image_width))
        y2 = max(0, min(max(top, top + height) * sy, image_height))

        if x2 <= x1 or y2 <= y1:
            continue

        label = obj.get("label", "car")

        annotations.append({
            "class": label if label in CLASS_NAMES else "car",
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
        })

    return annotations


def predictions_to_drawing(predictions, sx, sy):
    """Convert YOLO predictions to labeled Canvas objects."""
    boxes = []

    for prediction in predictions:
        boxes.append({
            "label": prediction["class"],
            "left": prediction["x1"] / sx,
            "top": prediction["y1"] / sy,
            "width": (
                             prediction["x2"] - prediction["x1"]
                     ) / sx,
            "height": (
                              prediction["y2"] - prediction["y1"]
                      ) / sy,
        })

    return boxes_to_drawing(
        boxes,
        stroke_color="#00FF00",
        stroke_width=2,
        fill_color="rgba(0, 255, 0, 0.08)",
    )


def main():
    init_state()
    st.title("EasyAnnotate")

    with st.sidebar:
        st.header("Image")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
        )

    if uploaded_file is None:
        st.info("Upload an image to start annotating.")
        return

    image_bytes = uploaded_file.getvalue()
    image_id = hashlib.sha256(image_bytes).hexdigest()[:16]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    width, height = image.size
    scale = min(MAX_CANVAS_WIDTH / width, 1.0)
    canvas_width = max(1, round(width * scale))
    canvas_height = max(1, round(height * scale))

    sx = width / canvas_width
    sy = height / canvas_height

    if st.session_state.image_id != image_id:
        st.session_state.image_id = image_id
        st.session_state.drawing = {"objects": []}
        st.session_state.mode = "Manual"
        st.session_state.prediction_count = None

        saved = load_annotations(image_id)

        if saved:
            st.session_state.drawing = saved["drawing"]
            logger.info("Restored annotations for {}", image_id)

        reset_canvas()

    with st.sidebar:
        st.caption(uploaded_file.name)
        st.caption(f"{width} × {height}")

        st.divider()
        st.header("Annotation")

        selected_class = st.selectbox(
            "Class for new boxes",
            options=CLASS_NAMES,
            index=2,
        )

        st.divider()
        st.header("AI Assistance")

        if st.button(
                "Auto Annotate",
                type="primary",
                use_container_width=True,
        ):
            logger.info("Auto annotation started: {}", uploaded_file.name)

            try:
                with st.spinner("Detecting objects..."):
                    predictions = detect_objects(image)

                st.session_state.drawing = predictions_to_drawing(
                    predictions, sx, sy
                )
                st.session_state.prediction_count = len(predictions)
                st.session_state.mode = "AI-Assisted"

                logger.info(
                    "Auto annotation finished: {} boxes",
                    len(predictions),
                )

                reset_canvas()
                st.rerun()

            except Exception:
                logger.exception("Auto annotation failed")
                st.error("Detection failed. Check terminal logs.")

        if st.session_state.prediction_count is not None:
            st.caption(
                f"AI generated "
                f"{st.session_state.prediction_count} boxes."
            )

    left_col, right_col = st.columns([3, 1], gap="large")

    with left_col:
        st.subheader("Image Annotation")

        result = st_canvas(
            background_image=image,
            background_image_fit="stretch",
            drawing_mode="labeled_rect",
            label=selected_class,
            stroke_width=2,
            stroke_color="#00FF00",
            fill_color="rgba(0, 255, 0, 0.08)",
            width=canvas_width,
            height=canvas_height,
            update_streamlit=True,
            return_image_data=False,
            initial_drawing=st.session_state.canvas_initial,
            key=(
                f"canvas_{image_id}_"
                f"{st.session_state.canvas_version}"
            ),
        )

    canvas_data = result.json_data

    if st.session_state.pending_canvas_update:
        if (
                canvas_data is not None
                and canvas_signature(canvas_data)
                == canvas_signature(st.session_state.canvas_initial)
        ):
            st.session_state.drawing = deepcopy(canvas_data)
            st.session_state.pending_canvas_update = False
            logger.debug("Canvas initialized successfully")

    elif canvas_data is not None:
        st.session_state.drawing = deepcopy(canvas_data)

    objects = canvas_objects(st.session_state.drawing)
    annotations = get_annotations(
        st.session_state.drawing,
        sx,
        sy,
        image.size,
    )

    with right_col:
        st.subheader("Annotations")
        st.caption(f"Mode: {st.session_state.mode}")
        st.metric("Total Boxes", len(annotations))

        with st.container(height=450, border=False):
            for index, obj in enumerate(objects):
                label = obj.get("label", "car")
                if label not in CLASS_NAMES:
                    label = "car"

                with st.container(border=True):
                    st.markdown(f"**Box {index + 1}**")

                    new_label = st.selectbox(
                        "Class",
                        options=CLASS_NAMES,
                        index=CLASS_NAMES.index(label),
                        key=(
                            f"label_{image_id}_"
                            f"{st.session_state.canvas_version}_"
                            f"{index}"
                        ),
                    )

                    if new_label != label:
                        obj["label"] = new_label
                        reset_canvas()
                        st.rerun()

                    if st.button(
                            "Delete",
                            key=f"delete_{image_id}_{index}",
                            use_container_width=True,
                    ):
                        st.session_state.drawing["objects"].remove(obj)
                        logger.info("Deleted box {}", index + 1)
                        reset_canvas()
                        st.rerun()

                    if index < len(annotations):
                        box = annotations[index]
                        st.caption(
                            f"({box['x1']:.0f}, {box['y1']:.0f}) → "
                            f"({box['x2']:.0f}, {box['y2']:.0f})"
                        )

        if st.button(
                "Clear All",
                use_container_width=True,
        ):
            st.session_state.drawing = {"objects": []}
            st.session_state.prediction_count = None
            st.session_state.mode = "Manual"
            reset_canvas()
            st.rerun()

        if st.button(
                "Save Annotations",
                type="primary",
                use_container_width=True,
        ):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")

            path = save_annotations(
                image_id=image_id,
                filename=uploaded_file.name,
                image_bytes=buffer.getvalue(),
                image_size=image.size,
                canvas_size=(canvas_width, canvas_height),
                drawing=st.session_state.drawing,
                annotations=annotations,
            )

            st.success(f"Saved {len(annotations)} boxes")
            logger.info("Annotations saved: {}", path)

        st.download_button(
            "Export ZIP",
            data=export_yolo(
                image=image,
                annotations=annotations,
                filename=uploaded_file.name,
            ),
            file_name="annotations.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if annotations:
        with st.expander("Annotation Data"):
            st.dataframe(
                annotations,
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
