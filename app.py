import base64
import os
import random
from pathlib import Path
import shutil

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
if current_proj != "No project selected":
    st.info(f"Current project: **{current_proj}**")

# ---------------------------------------------------------------------------
# Project manager
# ---------------------------------------------------------------------------

st.markdown("## Project manager")

projects = [p.name for p in DATA_FOLDER.iterdir() if p.is_dir()]

col_input, col_btn, col_or, col_select = st.columns([3, 1, 1, 4], vertical_alignment="bottom")

with col_input:
    new_proj = st.text_input("Create a new project. Give it a name:")
    if " " in new_proj:
        st.warning("Names cannot contain spaces — use underscores (_) instead.")
        

with col_btn:
    if st.button("Create") and new_proj and " " not in new_proj:
        if new_proj in projects:
            st.warning("A project with that name already exists.")
        else:
            os.makedirs(DATA_FOLDER / new_proj, exist_ok=True)
            projects.append(new_proj)
            st.session_state["current_project"] = new_proj
            st.session_state["_created_msg"] = new_proj
            st.rerun()

if projects != []:
    with col_or:
        st.markdown("**OR**")

    with col_select:
        if projects:
            active = st.session_state.get("current_project")
            proj_index = projects.index(active) if active in projects else 0
            selected = st.selectbox("Choose an existing project:", options=projects, index=None)
            if selected != active and selected !=None:
                st.session_state["current_project"] = selected
                st.rerun()
        else:
            st.caption("No projects yet — create one above.")

    # Show "just created" confirmation once, then clear it
    if msg := st.session_state.pop("_created_msg", None):
        st.success(f"Project **{msg}** has been created. Continue with Upload.")

# ---------------------------------------------------------------------------
# Project dashboard
# ---------------------------------------------------------------------------


st.markdown("#### Projects overview")

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
            "OCR":      done(has_files(root / "data" / "06_structured_output")),
        })


    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Preview": st.column_config.ImageColumn("Preview", width="small"),
        },
        hide_index=True,
        use_container_width=True,
    )

col1, col2, col3 = st.columns([3,1,5], vertical_alignment= "center")

if projects:
    with col1:
        deleted_proj= st.selectbox("Delete a project:", options=projects, index = None)
        if deleted_proj != None:
            confirm = st.checkbox(f"I want to permanently delete '{deleted_proj}'")
            with col2:    
                if st.button("Delete", type="primary", disabled=not confirm):
                    shutil.rmtree(DATA_FOLDER / deleted_proj)
                    if st.session_state.get("current_project") == deleted_proj:
                        st.session_state.pop("current_project", None)
                    st.rerun()

    

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

st.divider()
st.title("Museum Label Whisperer")
st.markdown("""
This app finds the things you care about inside a set of images, cuts them out,
and turns any text in them into a table you can work with.

You teach it what to look for by drawing boxes on a handful of images first, so it
can learn to find anything that looks reasonably consistent — specimen labels, pages
of a book, signs, forms, plant tags.

## How it works

### Finding and cutting out
1. **Upload** — bring in a handful of images to learn from
2. **Annotate** — draw a box around each object you want it to find
3. **Train** — the app learns what to look for from your boxes
4. **Predict** — see what it finds, so you can check it got it right
5. **Crop** — cut out everything it found, one image per object

### Reading the text
Turn each cut-out image into text, then sort that text into columns of your choice —
these can follow an existing standard (GBIF Darwin Core, for example), or whatever
suits your own records.

### Checking
Go through the results alongside the original images, and correct anything that came
out wrong before you export.
""")
