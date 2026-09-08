import streamlit as st

st.set_page_config(
    page_title="Museum Label Whisperer",
 #   page_icon="🏷️",
    layout="wide"
)

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