import streamlit as st

st.set_page_config(
    page_title="Museum Label Reader",
 #   page_icon="🏷️",
    layout="wide"
)

st.title("Museum Label Reader")

st.markdown("""
This app guides you through the entire workflow.

### Workflow

1. Upload images
2. Annotate them in CVAT
3. Train YOLO
4.  bla bla bla more stuff later


Choose a page from the sidebar.
""")