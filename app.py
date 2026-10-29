"""EasyAnnotate: lightweight AI-assisted image annotation."""

import hashlib
import io
from copy import deepcopy

import streamlit as st
from loguru import logger
from PIL import Image
from easy_annotate.detector import detect_objects
from streamlit_drawable_canvas import st_canvas

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


def initialize_state():
    """Initialize application state."""
    defaults = {
        "image_id": None,
        "canvas_version": 0,
        "drawing": {"objects": []},
        "mode": "Manual",
        "prediction_count": None,
        "pending_canvas_update": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)


def make_drawing(boxes, scale_x, scale_y):
    """Convert image-space boxes into editable canvas objects."""
    objects = []

    for box in boxes:
        x1 = box["x1"] / scale_x
        y1 = box["y1"] / scale_y
        x2 = box["x2"] / scale_x
        y2 = box["y2"] / scale_y

        objects.append({
            "type": "LabeledRect",
            "label": box["class"],
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "originX": "left",
            "originY": "top",
            "scaleX": 1,
            "scaleY": 1,
            "angle": 0,
            "stroke": "#00FF00",
            "strokeWidth": 2,
            "fill": "rgba(0, 255, 0, 0.08)",
            "lockRotation": True,
            "strokeUniform": True,
        })

    return {"objects": objects}


def extract_annotations(drawing, scale_x, scale_y, image_size):
    """Extract labeled rectangles in original image coordinates."""
    if not drawing:
        return []

    image_width, image_height = image_size
    annotations = []

    for obj in drawing.get("objects", []):
        if obj.get("type", "").lower() not in {
            "labeledrect",
            "rect",
        }:
            continue

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

        x1 = min(left, left + width) * scale_x
        y1 = min(top, top + height) * scale_y
        x2 = max(left, left + width) * scale_x
        y2 = max(top, top + height) * scale_y

        x1 = max(0, min(x1, image_width))
        x2 = max(0, min(x2, image_width))
        y1 = max(0, min(y1, image_height))
        y2 = max(0, min(y2, image_height))

        if x2 <= x1 or y2 <= y1:
            continue

        annotations.append({
            "class": obj.get("label") or "car",
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
        })

    return annotations


def main():
    initialize_state()

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

    if st.session_state.image_id != image_id:
        st.session_state.image_id = image_id
        st.session_state.drawing = {"objects": []}
        st.session_state.canvas_version += 1
        st.session_state.mode = "Manual"
        st.session_state.prediction_count = None

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    original_width, original_height = image.size

    scale = min(MAX_CANVAS_WIDTH / original_width, 1.0)
    canvas_width = max(1, round(original_width * scale))
    canvas_height = max(1, round(original_height * scale))

    scale_x = original_width / canvas_width
    scale_y = original_height / canvas_height

    with st.sidebar:
        st.caption(f"Filename: {uploaded_file.name}")
        st.caption(
            f"Resolution: {original_width} × {original_height}"
        )

        st.divider()
        st.header("Annotation")

        selected_class = st.selectbox(
            "Class for new boxes",
            options=CLASS_NAMES,
            index=2,
            accept_new_options=False,
            filter_mode=None,
        )

        st.divider()
        st.header("AI Assistance")

        if st.button(
                "Auto Annotate",
                type="primary",
                use_container_width=True,
        ):
            logger.info("Auto Annotate clicked: image={}", uploaded_file.name)

            try:
                with st.spinner("Detecting objects..."):
                    predictions = detect_objects(image)

                logger.info(
                    "Predictions received: count={}",
                    len(predictions),
                )

                drawing = make_drawing(
                    predictions,
                    scale_x,
                    scale_y,
                )

                logger.info(
                    "Canvas drawing generated: objects={}",
                    len(drawing["objects"]),
                )

                st.session_state.drawing = drawing
                st.session_state.prediction_count = len(predictions)
                st.session_state.mode = "AI-Assisted"
                st.session_state.canvas_version += 1

                # Keep track of pending AI annotations.
                st.session_state.pending_canvas_update = True

                logger.info(
                    "Session state updated: canvas_version={}",
                    st.session_state.canvas_version,
                )

                st.rerun()

            except Exception:
                logger.exception("Auto annotation failed")
                st.error("Auto annotation failed. Check terminal logs.")

        if st.session_state.prediction_count is not None:
            st.caption(
                f"AI generated "
                f"{st.session_state.prediction_count} boxes."
            )

    left_col, right_col = st.columns(
        [3, 1],
        gap="large",
    )

    with left_col:
        st.subheader("Image Annotation")

        logger.debug(
            "Rendering canvas: key_version={}, stored_objects={}",
            st.session_state.canvas_version,
            len(st.session_state.drawing.get("objects", [])),
        )

        canvas_result = st_canvas(
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
            initial_drawing=st.session_state.drawing,
            key=(
                f"canvas_{image_id}_"
                f"{st.session_state.canvas_version}"
            ),
        )

    canvas_data = canvas_result.json_data

    if st.session_state.get("pending_canvas_update", False):
        expected = len(st.session_state.drawing.get("objects", []))
        actual = (
            len(canvas_data.get("objects", []))
            if canvas_data is not None
            else None
        )

        logger.info(
            "Waiting for canvas initialization: expected={}, actual={}",
            expected,
            actual,
        )

        if (
                canvas_data is not None
                and len(canvas_data.get("objects", [])) == expected
        ):
            st.session_state.pending_canvas_update = False
            st.session_state.drawing = canvas_data
            logger.info("Canvas initialization completed")

    elif canvas_data is not None:
        st.session_state.drawing = canvas_data

        logger.debug(
            "Canvas updated: objects={}",
            len(canvas_data.get("objects", [])),
        )

    annotations = extract_annotations(
        st.session_state.drawing,
        scale_x,
        scale_y,
        image.size,
    )

    with right_col:
        st.subheader("Annotations")

        st.caption(f"Mode: {st.session_state.mode}")
        st.metric("Total Boxes", len(annotations))

        # Scrollable annotation list
        with st.container(height=450, border=False):
            for index, annotation in enumerate(annotations):
                with st.container(border=True):
                    st.write(
                        f"**{index + 1}. {annotation['class']}**"
                    )

                    st.caption(
                        f"({annotation['x1']:.0f}, "
                        f"{annotation['y1']:.0f}) → "
                        f"({annotation['x2']:.0f}, "
                        f"{annotation['y2']:.0f})"
                    )

        if st.button(
                "Clear All",
                use_container_width=True,
        ):
            st.session_state.drawing = {"objects": []}
            st.session_state.prediction_count = None
            st.session_state.mode = "Manual"
            st.session_state.canvas_version += 1
            st.rerun()

    if annotations:
        with st.expander("Annotation Data"):
            st.dataframe(
                annotations,
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
