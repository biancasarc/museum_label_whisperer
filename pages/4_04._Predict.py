from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

from backend.folder_picker import browse_input

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: **{current_proj}**")

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj
RUNS_DIRECTORY = PROJECT_ROOT / "runs" / "detect"

PREVIEW_DIRECTORY = PROJECT_ROOT / "data" / "03_prediction_preview"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_latest_best_model() -> Path | None:
    """Return the most recently created YOLO best.pt checkpoint."""
    checkpoints = list(RUNS_DIRECTORY.glob("**/weights/best.pt")) if RUNS_DIRECTORY.is_dir() else []
    return max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None


best_model = find_latest_best_model()
saved_source = st.session_state.get("prediction_source_directory")


st.title("Step 4 — Predict")
st.write("See what the trained model finds, so you can check it works before cutting anything out.")

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

    # Outside the form: a form only accepts its own submit button, so Browse
    # cannot live inside one.
    source_directory = browse_input(
        "Folder of images to search",
        state_key="predict_source_directory",
        default=saved_source,
        prompt="Choose the folder of images to search",
        help="Defaults to the folder you chose in Step 1. You can change it here.",
    )

    # No form left to submit: the only field moved out so Browse could sit
    # beside it, so a plain button is all that is needed.
    start_prediction = st.button(
        "Find objects",
        type="primary",
        icon=":material/play_arrow:",
    )

if start_prediction:
    source_path = Path(source_directory).expanduser()
    if not source_path.is_dir():
        st.error(f"Image directory not found: {source_path}")
        st.stop()

    st.session_state["prediction_source_directory"] = str(source_path.resolve())

    image_paths = sorted(
        p for p in source_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        st.error("No images found in that folder.")
        st.stop()

    PREVIEW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0)
    detections = 0
    try:
        model = YOLO(str(best_model))
        for number, image_path in enumerate(image_paths, start=1):
            # cv2.imread applies the EXIF rotation tag, exactly like Steps 2, 3 and 5.
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            result = model.predict(source=image, imgsz=1024, verbose=False)[0]
            detections += len(result.boxes) if result.boxes is not None else 0
            cv2.imwrite(str(PREVIEW_DIRECTORY / f"{image_path.stem}_pred.jpg"), result.plot())
            progress.progress(number / len(image_paths))
    except Exception as error:
        st.error(f"Prediction failed: {error}")
    else:
        st.success(f"Prediction completed: {detections} label(s) found in {len(image_paths)} image(s).")
        st.caption(f"Annotated preview images saved to: {PREVIEW_DIRECTORY}")
