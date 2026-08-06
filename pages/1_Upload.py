import os
import random
import shutil
from pathlib import Path

import streamlit as st


DATA_FOLDER = "data/original"

os.makedirs(DATA_FOLDER, exist_ok=True)

st.title("Step 1 - Upload Images")

image_dir = st.text_input(
    "Add the path of the directory with the images you'd like to be cropped (raw files will not be modified)",
    placeholder="/full/path/to/specimen_images",
    key="upload_image_directory",
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

    os.makedirs(DATA_FOLDER, exist_ok=True)

    progress = st.progress(0)

    for i, img in enumerate(selected, start=1):
        shutil.copy2(img, Path(DATA_FOLDER) / img.name)
        progress.progress(i / n_images)

    st.success(f"Imported {n_images} random images!")
    st.info("Continue to Step 2.")
