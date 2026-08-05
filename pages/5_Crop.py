from pathlib import Path
import random

import cv2
import streamlit as st
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = PROJECT_ROOT / "runs" / "detect"
DEFAULT_SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "original"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "cropping_result"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_latest_best_model() -> Path | None:
    checkpoints = list(RUNS_DIRECTORY.glob("*/weights/best.pt"))
    return max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None


best_model = find_latest_best_model()
saved_source = st.session_state.get("prediction_source_directory")
source_default = saved_source or str(DEFAULT_SOURCE_DIRECTORY)

st.title("Step 5 - Crop detected labels")
st.write("Create one cropped image for every high-confidence label detected by the best model.")

if best_model is None:
    st.warning("No trained `best.pt` model was found. Complete Step 3 first.")
    st.stop()

with st.container(border=True):
    st.caption(f"Best model: `{best_model.relative_to(PROJECT_ROOT)}`")
    st.caption(f"Crops will be saved to: `{OUTPUT_DIRECTORY.relative_to(PROJECT_ROOT)}`")

    with st.form("crop_form"):
        source_directory = st.text_input(
            "Directory of images to crop",
            value=source_default,
            help="Defaults to the directory entered in Step 1.",
        )
        image_limit = st.number_input(
            "Number of random images (0 processes all images)",
            min_value=0,
            value=0,
            step=1,
        )
        buffer = st.number_input("Crop buffer (pixels)", min_value=0, value=10, step=1)
        confidence_threshold = st.number_input(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.90,
            step=0.01,
            format="%.2f",
        )
        start_cropping = st.form_submit_button(
            "Create crops",
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
        st.error("No supported images were found in the selected directory.")
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

            result = model.predict(source=str(image_path), imgsz=1024, verbose=False)[0]
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
