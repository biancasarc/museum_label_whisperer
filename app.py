import base64
import os
import random
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Museum Label Whisperer",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_FOLDER  = PROJECT_ROOT / "projects"
os.makedirs(DATA_FOLDER, exist_ok=True)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_files(directory: Path) -> bool:
    """Return True if the directory exists and contains at least one file."""
    return directory.is_dir() and any(directory.iterdir())

def image_to_b64(path: Path) -> str:
    """Convert an image file to a base64 data URI for st.dataframe's ImageColumn."""
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# ---------------------------------------------------------------------------
# Current project banner
# ---------------------------------------------------------------------------

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: **{current_proj}**")

# ---------------------------------------------------------------------------
# Project manager
# ---------------------------------------------------------------------------

st.markdown("## Project manager")

projects = [p.name for p in DATA_FOLDER.iterdir() if p.is_dir()]

col_input, col_btn, col_or, col_select = st.columns([3, 1, 1, 4], vertical_alignment="bottom")

with col_input:
    new_proj = st.text_input("Create a new project:")

with col_btn:
    if st.button("Create") and new_proj:
        if new_proj in projects:
            st.warning("A project with that name already exists.")
        else:
            os.makedirs(DATA_FOLDER / new_proj, exist_ok=True)
            projects.append(new_proj)
            st.session_state["current_project"] = new_proj
            st.session_state["_created_msg"] = new_proj
            st.rerun()

with col_or:
    st.markdown("**OR**")

with col_select:
    if projects:
        active = st.session_state.get("current_project")
        proj_index = projects.index(active) if active in projects else 0
        selected = st.selectbox("Choose an existing project:", options=projects, index=proj_index)
        if selected != active:
            st.session_state["current_project"] = selected
            st.rerun()
    else:
        st.caption("No projects yet — create one above.")

# Show "just created" confirmation once, then clear it
if msg := st.session_state.pop("_created_msg", None):
    st.success(f"Project **{msg}** has been created.")

# ---------------------------------------------------------------------------
# Project dashboard
# ---------------------------------------------------------------------------


st.markdown("#### Project overview")

if not projects:
    st.info("No projects yet. Create one above to get started.")
else:
    rows = []
    for proj in projects:
        root       = DATA_FOLDER / proj
        upload_dir = root / "data" / "01_train_val_subset"

        # Pick a random preview image if any have been uploaded
        images = [
            f for f in upload_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXT
        ] if upload_dir.is_dir() else []
        preview = image_to_b64(random.choice(images)) if images else None

        # Check whether each pipeline step has been completed
        trained = (
            any((root / "runs" / "detect").glob("**/weights/best.pt"))
            if (root / "runs" / "detect").is_dir() else False
        )

        # Helper: return "✅ Done" when a step is complete, empty string otherwise
        def done(flag: bool) -> str:
            return "✅ Done" if flag else ""

        rows.append({
            "Project":  proj,
            "Preview":  preview,
            "Upload":   done(has_files(upload_dir)),
            "Annotate": done((root / "data" / "02_yolo_dataset" / "data.yaml").is_file()),
            "Train":    done(trained),
            "Predict":  done(has_files(root / "data" / "03_prediction_preview")),
            "Crop":     done(has_files(root / "data" / "04_cropping_result")),
            "OCR":      done(has_files(root / "data" / "ocr_results")),
        })


    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Preview": st.column_config.ImageColumn("Preview", width="small"),
        },
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

st.divider()
st.title("Museum Label Whisperer")
st.markdown("""
This app transcribes museum specimen labels into organised, machine-readable metadata.

## How it works

### Label cropping
1. **Upload** — import a subset of specimen images for training
2. **Annotate** — draw bounding boxes around each label
3. **Train** — train a YOLOv8 detection model on your annotations
4. **Predict** — run the trained model on your full image collection
5. **Crop** — extract each detected label as an individual image

### OCR
Transcribe the cropped label images into structured Darwin Core metadata fields.

### Quality checking
Review a subset of OCR results — images with low confidence scores, conflicting reads,
or a random sample — and correct any errors before export.
""")
