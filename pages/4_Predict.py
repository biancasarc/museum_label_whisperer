from pathlib import Path

import streamlit as st
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = PROJECT_ROOT / "runs" / "detect"
DEFAULT_SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "original"


def find_latest_best_model() -> Path | None:
    """Return the most recently created YOLO best.pt checkpoint."""
    checkpoints = list(RUNS_DIRECTORY.glob("*/weights/best.pt"))
    return max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None


best_model = find_latest_best_model()
saved_source = st.session_state.get("prediction_source_directory")
source_default = saved_source or str(DEFAULT_SOURCE_DIRECTORY)

st.title("Step 4 - Predict with the best model")
st.write("Run the best checkpoint from the latest YOLO training run on your image directory.")

if best_model is None:
    st.warning("No trained `best.pt` model was found. Complete Step 3 first.")
    st.stop()

with st.container(border=True):
    st.caption(f"Best model: `{best_model.relative_to(PROJECT_ROOT)}`")

    with st.form("prediction_form"):
        source_directory = st.text_input(
            "Directory of images to predict",
            value=source_default,
            help="This is the directory entered in Step 1. You can change it here if needed.",
        )
        start_prediction = st.form_submit_button(
            "Run prediction",
            type="primary",
            icon=":material/play_arrow:",
        )

if start_prediction:
    source_path = Path(source_directory).expanduser()
    if not source_path.is_dir():
        st.error(f"Image directory not found: {source_path}")
        st.stop()

    st.session_state["prediction_source_directory"] = str(source_path.resolve())

    try:
        with st.spinner("Running predictions…"):
            model = YOLO(str(best_model))
            results = model.predict(source=str(source_path), save=True)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
    else:
        output_directory = results[0].save_dir if results else None
        st.success("Prediction completed successfully.")
        if output_directory:
            st.caption(f"Annotated images saved to: {output_directory}")
