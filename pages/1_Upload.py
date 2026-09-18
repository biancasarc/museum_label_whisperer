import os
import random
import shutil
from pathlib import Path

import streamlit as st

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: **{current_proj}**")

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj
DATA_FOLDER = PROJECT_ROOT / "data" / "01_train_val_subset"

if current_proj != "No project selected":
    os.makedirs(DATA_FOLDER, exist_ok=True)

st.title("Step 1 - Upload Images")

col1, col2=st.columns(2)

with col1:
    st.markdown("""
    Decide on a number of images you wish to annotate to train the label detection model. There
    are a few characteristics to consider before deciding on the size of this subset:
    * Are the labels very complex?
    * Do the labels vary between images in color, shape, position? 
    * Do the labels follow a pattern?
    """)

with col2:
    st.image("learning_curve.png")

image_dir = st.text_input(
    "Add the path of the directory with the images you'd like to be cropped (raw files will not be modified)",
    placeholder="/full/path/to/specimen_images"
)



n_images = st.number_input(
    "Add the total number of images for training and validation (20% of these will be used for validation)",
    min_value=1,
    value=20,
    step=1
)

if st.button("Import images"):

    if not os.path.isdir(image_dir):
        st.error("Directory does not exist.")
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
        st.error("No supported image files found.")
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

