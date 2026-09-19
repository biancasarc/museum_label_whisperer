import os
import random
import shutil
from pathlib import Path

import streamlit as st

from backend.folder_picker import PickerUnavailable, pick_folder

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: **{current_proj}**")

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj
DATA_FOLDER = PROJECT_ROOT / "data" / "01_train_val_subset"

if current_proj != "No project selected":
    os.makedirs(DATA_FOLDER, exist_ok=True)

st.title("Step 1 — Upload images")

col1, col2=st.columns(2)

with col1:
    st.markdown("""
    Choose how many images to start with. The app learns from the boxes you draw on
    them, so a small, well-chosen set goes a long way. Use more images if:
    * The objects you want to find are complex or detailed
    * They vary a lot between images in colour, shape or position
    * Use fewer if they always look much the same
    """)

with col2:
    st.image("learning_curve.png")

# The Browse button writes the chosen path here; the text box reads it back, so
# typing a path by hand still works exactly as before.
CHOSEN_FOLDER = "upload_image_dir"

col_path, col_browse = st.columns([5, 1])

with col_path:
    image_dir = st.text_input(
        "Folder containing your images (your originals are never changed)",
        value=st.session_state.get(CHOSEN_FOLDER, ""),
        placeholder="/full/path/to/your/images",
    )

with col_browse:
    st.markdown('<div style="height: 7mm;"></div>', unsafe_allow_html=True)
    if st.button("Browse…", width="stretch"):
        try:
            chosen = pick_folder("Choose the folder with your images")
        except PickerUnavailable as error:
            st.warning(f"Could not open a folder window: {error}. Type the path instead.")
        else:
            if chosen:
                st.session_state[CHOSEN_FOLDER] = chosen
                st.rerun()



n_images = st.number_input(
    "How many images to bring in (the app keeps 20% back to check its own work)",
    min_value=1,
    value=20,
    step=1
)

if st.button("Import images"):

    if not os.path.isdir(image_dir):
        st.error("That folder does not exist.")
        st.stop()

    # Keep the original directory available as the prediction source in Step 4.
    st.session_state["prediction_source_directory"] = str(
        Path(image_dir).expanduser().resolve()
    )

    # Find all supported image files
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    image_files = [
        f for f in Path(image_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if len(image_files) == 0:
        st.error("No images found in that folder.")
        st.stop()

    if n_images > len(image_files):
        st.warning(
            f"Only {len(image_files)} images found. Importing all of them."
        )
        n_images = len(image_files)

    selected = random.sample(image_files, n_images)


    progress = st.progress(0)

    for i, img in enumerate(selected, start=1):
        shutil.copy2(img, DATA_FOLDER / img.name)
        progress.progress(i / n_images)

    st.success(f"Imported {n_images} random images!")
    st.info("Continue to Step 2.")

