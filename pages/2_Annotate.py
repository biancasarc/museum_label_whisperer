import streamlit as st

from backend.cvat_manager import (
    create_task,
    export_dataset,
    open_browser,
    processing_yolo_dir,
    set_credentials,
)


st.title("Step 2 - Annotate")

if "task_id" not in st.session_state:
    st.session_state.task_id = None

if "task_name" not in st.session_state:
    st.session_state.task_name = ""

if "cvat_username" not in st.session_state:
    st.session_state.cvat_username = ""

if "cvat_password" not in st.session_state:
    st.session_state.cvat_password = ""

# CVAT credentials inputs
st.subheader("CVAT Credentials")
st.session_state.cvat_username = st.text_input(
    "CVAT username", value=st.session_state.cvat_username
)
st.session_state.cvat_password = st.text_input(
    "CVAT password", value=st.session_state.cvat_password, type="password"
)

st.session_state.task_name = st.text_input(
    "Write the name of the CVAT task",
    value=st.session_state.task_name,
    placeholder="e.g. label-detection-task",
)

if st.button("Create CVAT Task"):
    if not st.session_state.task_name.strip():
        st.error("Please enter a task name before creating the CVAT task.")
    else:
        with st.spinner("Creating CVAT task..."):
            # apply provided credentials before creating the task
            set_credentials(st.session_state.cvat_username, st.session_state.cvat_password)
            task_id = create_task(task_name=st.session_state.task_name)

        st.session_state.task_id = task_id
        st.success(f"Created CVAT task: {st.session_state.task_name}")

if st.session_state.task_id:
    st.info(f"Current CVAT task: {st.session_state.task_name}")

col_btn, col_help = st.columns([1, 3])
with col_btn:
    if st.button("Open CVAT", help="Open the CVAT web UI for the current task"):
        open_browser()

with col_help:
    with st.expander("How to annotate (click to expand)"):
        st.markdown(
            """
- Click **OPEN** on the task, then open the **Job #**.
- Choose **Draw a rectangle** or press **n** on keyboard.
- Draw a rectangle around each label.
- Click **Save** when finished. 
- CVAT can be closed now.
"""
        )

if st.button("Export YOLO Dataset"):
    if not st.session_state.task_id:
        st.error("Create a CVAT task before exporting annotations.")
    else:
        with st.spinner("Exporting annotations..."):
            # ensure credentials are set before exporting
            set_credentials(st.session_state.cvat_username, st.session_state.cvat_password)
            output_directory = export_dataset(st.session_state.task_id)

        st.success(f"Dataset exported: {output_directory}")
        output_directory = processing_yolo_dir(output_directory)

