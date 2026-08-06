import streamlit as st

from backend.cvat_manager import create_task, export_dataset, open_browser, processing_yolo_dir


st.title("Step 2 - Annotate")

if "task_id" not in st.session_state:
    st.session_state.task_id = None

if "task_name" not in st.session_state:
    st.session_state.task_name = ""

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
            task_id = create_task(task_name=st.session_state.task_name)

        st.session_state.task_id = task_id
        st.success(f"Created CVAT task: {st.session_state.task_name}")

if st.session_state.task_id:
    st.info(f"Current CVAT task: {st.session_state.task_name}")

if st.button("Open CVAT"):
    open_browser()

if st.button("Export YOLO Dataset"):
    if not st.session_state.task_id:
        st.error("Create a CVAT task before exporting annotations.")
    else:
        with st.spinner("Exporting annotations..."):
            output_directory = export_dataset(st.session_state.task_id)

        st.success(f"Dataset exported: {output_directory}")
        output_directory = processing_yolo_dir(output_directory)

