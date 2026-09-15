import streamlit as st
import shutil
import pandas as pd
from pathlib import Path

from backend.images import load_rgb, make_display_image

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: {current_proj}")

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj
RAW_CSV = PROJECT_ROOT / "results" / "data.csv"
CSV_FILE = PROJECT_ROOT / "results" / "modified_data.csv"

def reset_widget_value(widget_key, original_value):
    st.session_state[widget_key] = original_value


if not RAW_CSV.exists():
    st.warning(f"No OCR results found at `{RAW_CSV.relative_to(PROJECT_ROOT)}`. Run the OCR step first.")
    st.stop()

if not CSV_FILE.exists():
    shutil.copyfile(RAW_CSV, CSV_FILE)

df = pd.read_csv(CSV_FILE)
raw_df = pd.read_csv(RAW_CSV)

saved_source = st.session_state.get("prediction_source_directory")

image_files = sorted([
    f for f in saved_source.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]]) # a list with all the images


if "image_index" not in st.session_state:
    st.session_state.image_index = 0  #keeping track of the current image


current_image = image_files[st.session_state.image_index]

st.title("OCR Checking")

st.write(
    f"Image {st.session_state.image_index + 1} "
    f"of {len(image_files)}")

#matching the current image with the row
matching_rows = df[df["Specimen.image"] == current_image.name] 


col3, col4 = st.columns(2) 
col1, col2 = st.columns(2)

with col1:
    # load_rgb applies the EXIF rotation tag, so photos taken with a rotated
    # camera appear upright (st.image on the raw file would show them flipped)
    display_img, _ = make_display_image(load_rgb(current_image), 1600)
    st.image(display_img, caption=current_image.name, width="stretch")



# Option 1: appearing as one excel row
# with col2:

#     if len(matching_rows) > 0:

#         # Select multiple CSV columns
#         selected_column = st.multiselect(
#             "Select information to review:",
#             df.columns[1:]
#         )

#         if selected_column:
#             edited_row = st.data_editor(
#                 matching_rows[selected_column],
#                 num_rows="fixed",
#                 use_container_width=True,
#                 disabled=["Specimen.image"]
#             )
#         else:
#             st.info("Select at least one column to review.")
#             edited_row = None

#     else:
#         st.warning("No matching row found in the CSV.")
#         edited_row = None


# Option 2: arranged in different rows

with col2:
    if len(matching_rows) > 0:

        selected_column = st.multiselect(
            "Select column for checking:", df.columns[1:])

        if selected_column:

            edited_data = {} # dictionary to store edited values

            for index, row in matching_rows.iterrows():

                edited_data[index] = {}

                for column in selected_column:

                    # Current value from the edited CSV
                    value = row[column]

                    # Find the corresponding original value from raw CSV
                    image_name = row["Specimen.image"]

                    original_row = raw_df[
                        raw_df["Specimen.image"] == image_name
                    ]

                    original_value = original_row.iloc[0][column]

                    # Create input and reset button next to each other
                    input_col, reset_col = st.columns([12, 1])

                    with input_col:
                        edited_data[index][column] = st.text_input(
                            f"{column}:",
                            value=str(value) if pd.notna(value) else "",
                            key=f"{index}_{column}"
                        )

                    with reset_col:
                        st.markdown(
                            '<div style="height: 7mm;"></div>',
                            unsafe_allow_html=True,
                        )
                        st.button(
                            "↺",
                            key=f"reset_{index}_{column}",
                            help="Reset to original OCR value",
                            on_click=reset_widget_value,
                            args=(
                                f"{index}_{column}",
                                str(original_value)
                                if pd.notna(original_value)
                                else "",
                            ),
                        )


            # for index, row in matching_rows.iterrows():

            #     edited_data[index] = {}

            #     for column in selected_column:

            #         value = row[column]

            #         edited_data[index][column] = st.text_input(
            #             f"{column}:",
            #             value=str(value) if pd.notna(value) else "",
            #             key=f"{index}_{column}"
            #         )

            # Convert edited values back into a DataFrame
            edited_row = matching_rows.copy()

            for index in edited_data:
                for column in edited_data[index]:
                    edited_row.loc[index, column] = edited_data[index][column]

        else:
            st.info("Select at least one column to review.")
            edited_row = None

    else:
        st.warning("No matching row found in the CSV.")
        edited_row = None



# -----------------------------
# Save changes + go to next
# -----------------------------

with col3:
    if st.session_state.image_index > 0:
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



