import streamlit as st

st.set_page_config(
    page_title="Museum Label Whisperer",
 #   page_icon="🏷️",
    layout="wide"
)

st.title("Museum Label Whisperer")

st.markdown("""
This app guides you through the entire workflow.

### Workflow

1. Upload images
2. Annotate them in CVAT
3. Train YOLO
4. Predict
5. Crop labels


Choose a page from the sidebar.
""")