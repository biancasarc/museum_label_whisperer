import os
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Museum Label Whisperer",
 #   page_icon="🏷️",
    layout="wide"
)

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: {current_proj}")

st.markdown("# Project manager")

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_FOLDER = PROJECT_ROOT / "projects"

os.makedirs(DATA_FOLDER, exist_ok=True)

new_proj = st.text_input("Create a new project:")

if st.button("Create"):
    st.session_state["current_project"] = new_proj
    os.makedirs(PROJECT_ROOT / "projects"/ new_proj)
    st.rerun()

if new_proj:
    st.write(f"New project {new_proj} created.")

    

st.title("Museum Label Whisperer")

st.markdown("""
This is an app created for transcribing museum label specimens into organised metadata that is fully customisable, readable by both humans and machines.


## This is how it works:

### Label cropping
1. Upload a subset of images for model training
2. Annotate the images
3. Train YOLO model
4. Predict 
5. Crop labels

### OCR
...in the process of being integrated...

### Checking correctness
Manually checking a subset of the OCR results. This subset can be:
* Images that got different reads between two separate OCR pharses
* Images with a low confidence score (unclear writing)
* Or a random subset of images 

... besides a recommended workflow, the user can choose how much they want to check.


""")