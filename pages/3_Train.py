from pathlib import Path

import streamlit as st
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml"

st.title("Step 3 - Train YOLO model")
st.write("Train a YOLOv8 small detection model using the exported dataset.")

with st.container(border=True):
    st.caption(f"Dataset configuration: `{DATASET_YAML.relative_to(PROJECT_ROOT)}`")
    st.caption("Model: `yolov8s.pt` · Epochs: 20 · Image size: 640 · Workers: 4 ")

    start_training = st.button(
        "Start training",
        type="primary",
        icon=":material/play_arrow:",
    )

if start_training:
    if not DATASET_YAML.is_file():
        st.error(f"Dataset configuration not found: {DATASET_YAML}")
        st.stop()

    try:
        with st.spinner("Training YOLO model… This may take a while."):
            model = YOLO("yolov8s.pt")
            results = model.train(
                data=str(DATASET_YAML),
                epochs=20,
                imgsz=640,
                workers=4,
                name="1.1",
            )
    except Exception as error:
        st.error(f"Training failed: {error}")
    else:
        st.success("Training completed successfully.")
        st.caption(f"Results saved to: {results.save_dir}")
