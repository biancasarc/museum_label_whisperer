import contextlib
import logging
import os
import traceback
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from ultralytics.utils import LOGGER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml"
RUNS_DIRECTORY = PROJECT_ROOT / "runs" / "detect"
TRAIN_LOG = PROJECT_ROOT / "runs" / "train_log.txt"

st.title("Step 3 - Train YOLO model")
st.write("Train a YOLOv8 small detection model using the exported dataset.")

with st.container(border=True):
    st.caption(f"Dataset configuration: `{DATASET_YAML.relative_to(PROJECT_ROOT)}`")
    st.caption("Model: `yolov8s.pt` · Image size: 640 · Workers: 4 ")
    st.caption(f"Results will be saved to: `{RUNS_DIRECTORY.relative_to(PROJECT_ROOT)}/<run name>/weights/best.pt`")

    # let the user choose epochs (recommend at least 50)
    epochs = st.number_input(
        "Epochs",
        min_value=1,
        value=50,
        step=1,
        help="Number of training epochs. Recommended: at least 50 for good results.",
    )
    if epochs < 50:
        st.warning("Recommendation: at least 50 epochs for better convergence.")

    start_training = st.button(
        "Start training",
        type="primary",
        icon=":material/play_arrow:",
    )

ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "annotations.json"
if DATASET_YAML.is_file() and ANNOTATIONS_FILE.is_file() \
        and ANNOTATIONS_FILE.stat().st_mtime > DATASET_YAML.stat().st_mtime:
    st.warning(
        "The annotations were changed after the training dataset was built. "
        "Go back to **Step 2** and click **Build training dataset** before training, "
        "otherwise the model learns from outdated boxes."
    )

if start_training:
    if not DATASET_YAML.is_file():
        st.error(f"Dataset configuration not found: {DATASET_YAML}")
        st.stop()
    if ANNOTATIONS_FILE.is_file() and ANNOTATIONS_FILE.stat().st_mtime > DATASET_YAML.stat().st_mtime:
        st.error("Training dataset is out of date — rebuild it in Step 2 first.")
        st.stop()

    RUNS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT_ROOT)  # train.txt / val.txt paths are relative to the project root
    log_handler = logging.FileHandler(TRAIN_LOG, mode="w", encoding="utf-8")
    LOGGER.addHandler(log_handler)  # ultralytics logs (epochs, metrics, errors) go to the log file
    try:
        with st.spinner("Training YOLO model… This may take a while (minutes to hours on a laptop CPU)."), \
                open(TRAIN_LOG.with_suffix(".stdout.txt"), "w") as out, contextlib.redirect_stdout(out):
            model = YOLO("yolov8s.pt")
            results = model.train(
                data=str(DATASET_YAML),
                epochs=int(epochs),
                imgsz=640,
                workers=4,
                project=str(RUNS_DIRECTORY),
                name="1.1",
                exist_ok=True,
            )
    except Exception as error:
        LOGGER.removeHandler(log_handler)
        st.error(f"Training failed: {error}")
        st.code(traceback.format_exc())
        if TRAIN_LOG.is_file():
            st.caption(f"Full training log: `{TRAIN_LOG}`")
            st.code(TRAIN_LOG.read_text()[-4000:])
    else:
        LOGGER.removeHandler(log_handler)
        best = Path(results.save_dir) / "weights" / "best.pt"
        if best.is_file():
            st.success("Training completed successfully.")
            st.caption(f"Best model saved to: `{best}`")
        else:
            st.warning(f"Training finished but no best.pt was written in {results.save_dir}. See log: `{TRAIN_LOG}`")
