import os
from pathlib import Path
import shutil

import streamlit as st

st.set_page_config(
    page_title="Museum Label Whisperer",
 #   page_icon="🏷️",
    layout="wide"
)

current_proj = st.session_state.get("current_project", "No project selected")

st.info(f"Current project: **{current_proj}**")


st.markdown("# Project manager")

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_FOLDER = PROJECT_ROOT / "projects"
os.makedirs(DATA_FOLDER, exist_ok=True)

projects = [p.name for p in DATA_FOLDER.iterdir() if p.is_dir()]


col1, col2, col3, col4 = st.columns([3,1,1,4], vertical_alignment="bottom")

with col1:
    new_proj = st.text_input("Create a new project:")


with col2:
    if st.button("Create") and new_proj:
        if new_proj in projects:
            st.info("Project name already exists. Choose a different name.")

        else:
            
            projects.append(new_proj)
            os.makedirs(PROJECT_ROOT / "projects"/ new_proj, exist_ok = True)
            st.session_state["current_project"] = new_proj
            st.session_state["show_created_msg"] = new_proj  # store the name
            st.rerun()

if msg := st.session_state.pop("show_created_msg", None):
    st.success(f"Project **{msg}** has been created.")

with col3:
    st.markdown("**OR**")


with col4:
    if projects:
        current_proj = st.session_state.get("current_project")
        proj_index = projects.index(current_proj) if current_proj in projects else 0
        selected_proj = st.selectbox("Choose an existing project:", options=projects, index=proj_index)
        if selected_proj != current_proj:
            st.session_state["current_project"] = selected_proj
            st.rerun()
    else:
        st.info("No projects yet — create one above.")


st.divider()
st.markdown("### Delete a project")

if projects:
    proj_to_delete = st.selectbox("Select project to delete:", options=projects, key="delete_selectbox")
    if st.button("🗑️ Delete", type="secondary"):
        shutil.rmtree(DATA_FOLDER / proj_to_delete)
        # if we deleted the active project, clear it from session state
        if st.session_state.get("current_project") == proj_to_delete:
            st.session_state.pop("current_project", None)
        st.rerun()


# if st.button("Create") and new_proj:
#     st.session_state["current_project"] = new_proj
#     os.makedirs(PROJECT_ROOT / "projects"/ new_proj)
#     st.write(f"New project {new_proj} created.")
#     proj_index = projects.index(new_proj)
#     st.rerun()

# projects = [p.name for p in DATA_FOLDER.iterdir() if p.is_dir()]

# if new_proj:

# else:
#     proj_index=0


# selected_proj = st.selectbox("Choose an existing project:",
#             options = projects,
#             index = proj_index)

# if selected_proj != st.session_state.get("current_project"):
#     st.session_state["current_project"] = selected_proj
#     st.rerun()




    

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