import streamlit as st
import pandas as pd
from pathlib import Path


IMAGE_FOLDER = Path("data/original")
CSV_FILE = Path("results/data.csv")

df = pd.read_csv(CSV_FILE)


# -----------------------------
# Get images
# -----------------------------

image_files = sorted([
    f for f in IMAGE_FOLDER.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
])


# -----------------------------
# Keep track of current image
# -----------------------------

if "image_index" not in st.session_state:
    st.session_state.image_index = 0


# -----------------------------
# Current image
# -----------------------------

current_image = image_files[st.session_state.image_index]

st.title("Specimen review")

st.write(
    f"Image {st.session_state.image_index + 1} "
    f"of {len(image_files)}"
)


# -----------------------------
# Find corresponding CSV row
# -----------------------------

matching_rows = df[
    df["Specimen.image"] == current_image.name
]


# -----------------------------
# Display image + editable row
# -----------------------------

col3, col4 = st.columns(2)
col1, col2 = st.columns(2)

with col1:
    st.image( current_image, 
             caption=current_image.name, 
             width="stretch" )




with col2:

    if len(matching_rows) > 0:

        # Dropdown with all CSV columns
        selected_column = st.selectbox(
            "Select information to review:",
            df.columns[1:]
        )

        # Show only the selected column
        edited_row = st.data_editor(
            matching_rows[["Specimen.image", selected_column]],
            num_rows="fixed",
            use_container_width=True,
            disabled=["Specimen.image"]
        )

    else:
        st.warning("No matching row found in the CSV.")
        edited_row = None




# -----------------------------
# Save changes + go to next
# -----------------------------

with col3:
    if st.session_state.image_index > 1:
        if st.button("Previous image", use_container_width=True):
            if edited_row is not None:

                # Find the original row
                row_index = df[
                    df["Specimen.image"] == current_image.name
                ].index

                # Save the edited value for the selected column
                df.loc[row_index, selected_column] = (
                    edited_row[selected_column].values
                )

                # Save CSV
                df.to_csv(CSV_FILE, index=False)

                st.session_state.image_index -= 1
                st.rerun()

with col4:
    if st.session_state.image_index < len(image_files) - 1:
        if st.button("Next image", use_container_width=True):

            if edited_row is not None:

                # Find the original row
                row_index = df[
                    df["Specimen.image"] == current_image.name
                ].index

                # Save the edited value for the selected column
                df.loc[row_index, selected_column] = (
                    edited_row[selected_column].values
                )

                # Save CSV
                df.to_csv(CSV_FILE, index=False)

                st.session_state.image_index += 1
                st.rerun()
    else:
        st.success("You reached the last image.")



