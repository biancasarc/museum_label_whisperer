from pathlib import Path
import random

import cv2
import streamlit as st
from ultralytics import YOLO

from backend.folder_picker import browse_input

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: **{current_proj}**")

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj
RUNS_DIRECTORY = PROJECT_ROOT / "runs" / "detect"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "04_cropping_result"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_latest_best_model() -> Path | None:
    checkpoints = list(RUNS_DIRECTORY.glob("**/weights/best.pt")) if RUNS_DIRECTORY.is_dir() else []
    return max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None


best_model = find_latest_best_model()
saved_source = st.session_state.get("prediction_source_directory")

st.title("Step 5 — Crop")
st.write("Cut out every object the model is confident about, saving each as its own image.")

if best_model is None:
    st.warning("No trained `best.pt` model was found. Complete Step 3 first.")
    st.caption(f"Looked in: `{RUNS_DIRECTORY}`")
    manual = browse_input(
        "…or choose a best.pt file yourself",
        state_key="manual_best_pt",
        prompt="Choose a trained model (best.pt)",
        extensions=(".pt",),
        is_file=True,
    )
    if manual and Path(manual).expanduser().is_file():
        best_model = Path(manual).expanduser().resolve()
    else:
        st.stop()

with st.container(border=True):
    st.caption(f"Best model: `{best_model.relative_to(PROJECT_ROOT)}`")
    st.caption(f"Crops will be saved to: `{OUTPUT_DIRECTORY.relative_to(PROJECT_ROOT)}`")

    # Outside the form: a form only accepts its own submit button, so Browse
    # cannot live inside one.
    source_directory = browse_input(
        "Folder of images to crop",
        state_key="crop_source_directory",
        default=saved_source,
        prompt="Choose the folder of images to crop",
        help="Defaults to the folder you chose in Step 1.",
    )

    with st.form("crop_form"):
        image_limit = st.number_input(
            "How many images to do (0 = all of them)",
            min_value=0,
            value=0,
            step=1,
        )
        buffer = st.number_input("Extra space around each cut-out (pixels)", min_value=0, value=10, step=1)
        confidence_threshold = st.number_input(
            "Minimum confidence (0 to 1)",
            min_value=0.0,
            max_value=1.0,
            value=0.90,
            step=0.01,
            format="%.2f",
        )
        start_cropping = st.form_submit_button(
            "Cut out objects",
            type="primary",
            icon=":material/content_cut:",
        )

if start_cropping:
    source_path = Path(source_directory).expanduser()
    if not source_path.is_dir():
        st.error(f"Image directory not found: {source_path}")
        st.stop()

    image_paths = [
        path for path in source_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    random.shuffle(image_paths)
    if image_limit:
        image_paths = image_paths[:image_limit]

    if not image_paths:
        st.error("No images found in that folder.")
        st.stop()

    st.session_state["prediction_source_directory"] = str(source_path.resolve())
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(best_model))
    progress = st.progress(0)
    status = st.empty()
    saved_crops = 0
    skipped_images = 0

    try:
        for image_number, image_path in enumerate(image_paths, start=1):
            status.write(f"Processing {image_number} of {len(image_paths)}: `{image_path.name}`")
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_images += 1
                progress.progress(image_number / len(image_paths))
                continue

            # Predict on the very array we crop from. cv2.imread applies the EXIF
            # rotation tag; predicting on the file path would not, and the boxes
            # would land on the wrong part of a rotated photo.
            result = model.predict(source=image, imgsz=1024, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                progress.progress(image_number / len(image_paths))
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            boxes = boxes[confidences >= confidence_threshold]

            for crop_number, (x1, y1, x2, y2) in enumerate(boxes, start=1):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                x1 = max(0, x1 - buffer)
                y1 = max(0, y1 - buffer)
                x2 = min(image.shape[1], x2 + buffer)
                y2 = min(image.shape[0], y2 + buffer)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]
                output_path = OUTPUT_DIRECTORY / f"{image_path.stem}_label_{crop_number:02d}.png"
                if cv2.imwrite(str(output_path), crop):
                    saved_crops += 1

            progress.progress(image_number / len(image_paths))
    except Exception as error:
        st.error(f"Cropping failed: {error}")
    else:
        status.empty()
        st.success(f"Created {saved_crops} crop(s) from {len(image_paths)} image(s).")
        if skipped_images:
            st.warning(f"Skipped {skipped_images} unreadable image(s).")
        st.caption(f"Crops saved to: {OUTPUT_DIRECTORY}")
