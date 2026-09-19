import streamlit as st
import base64
import re
import shutil
from io import BytesIO

import pandas as pd
from pathlib import Path

from backend.folder_picker import browse_input
from backend.images import load_rgb, make_display_image

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: **{current_proj}**")

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj
RAW_CSV = PROJECT_ROOT / "data" / "06_structured_output" / "structured_dwc_metadata.csv"
CSV_FILE = PROJECT_ROOT / "data" / "07_quality_checking" / "corrected_metadata.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DISPLAY_MAX_SIDE = 2400  # decoded once per image; zooming only scales it in the browser


def reset_widget_value(widget_key, original_value):
    st.session_state[widget_key] = original_value


def specimen_id(crop_name: str) -> str:
    """`DSC_0127_label_03.png` -> `DSC_0127`, matching how Step 6 groups cut-outs."""
    match = re.match(r"(.+)_label_\d+", Path(crop_name).stem)
    return match.group(1) if match else Path(crop_name).stem


@st.cache_resource(show_spinner="Loading image…", max_entries=8)
def image_data_uri(path_str: str, mtime: float) -> str:
    """The image, EXIF-corrected and downscaled once, as a JPEG data URI.

    Cached on path + modification time so dragging the zoom slider re-renders in
    the browser instead of decoding a 40-megapixel photograph again.
    """
    display, _ = make_display_image(load_rgb(Path(path_str)), DISPLAY_MAX_SIDE)
    buffer = BytesIO()
    display.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def zoomable_image(path_str: str, mtime: float, zoom: float) -> str:
    """Image in a box that scrolls once zoomed wider than the column."""
    return (
        '<div style="overflow:auto; max-height:75vh;'
        ' border:1px solid rgba(128,128,128,0.35); border-radius:0.5rem;">'
        f'<img src="{image_data_uri(path_str, mtime)}"'
        f' style="width:{zoom * 100:.0f}%; max-width:none; display:block;"/>'
        "</div>"
    )


if not RAW_CSV.exists():
    st.warning(f"No OCR results found at `{RAW_CSV.relative_to(PROJECT_ROOT)}`. Run the OCR step first.")
    st.stop()

if not CSV_FILE.exists():
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RAW_CSV, CSV_FILE)

df = pd.read_csv(CSV_FILE, dtype=str)
raw_df = pd.read_csv(RAW_CSV, dtype=str)

# Falls back to the folder Step 1 recorded, and is always shown so it can be
# corrected rather than only appearing when nothing is set.
saved_source = browse_input(
    "Folder containing the original images",
    state_key="quality_check_image_dir",
    default=st.session_state.get("prediction_source_directory") or "",
    prompt="Choose the folder with the original images",
    placeholder="/full/path/to/your/images",
)

if not saved_source:
    st.warning("Choose the folder holding your original images, or complete Step 1 first.")
    st.stop()

source_path = Path(saved_source).expanduser()
if not source_path.is_dir():
    st.error(f"That folder does not exist: {source_path}")
    st.stop()

# Show every image Step 5 cut something out of — not every image in the folder.
# Walking the whole folder buries you in blank screens, and hides the case worth
# seeing: an image that was cropped but whose text never came through Step 6.
CROPS_DIR = PROJECT_ROOT / "data" / "04_cropping_result"

crop_counts: dict[str, int] = {}
if CROPS_DIR.is_dir():
    for crop in CROPS_DIR.iterdir():
        if crop.is_file() and crop.suffix.lower() in IMAGE_SUFFIXES:
            key = specimen_id(crop.name)
            crop_counts[key] = crop_counts.get(key, 0) + 1

if not crop_counts:
    st.warning(
        f"No cut-out images found in `{CROPS_DIR.relative_to(PROJECT_ROOT)}`. "
        "Complete Step 5 first."
    )
    st.stop()

image_files = sorted(
    f for f in source_path.iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES and f.stem in crop_counts
)

if not image_files:
    st.warning(
        f"None of the images in `{source_path}` match the cut-outs in "
        f"`{CROPS_DIR.relative_to(PROJECT_ROOT)}`. Is this the folder Step 5 used?"
    )
    st.stop()

# Which images made it all the way through to text, and which did not.
have_text = set(df["Specimen.image"].astype(str).str.rsplit(".", n=1).str[0])


if "image_index" not in st.session_state:
    st.session_state.image_index = 0  #keeping track of the current image


# The list changes as you re-run earlier steps, so never trust a stale index.
st.session_state.image_index = min(st.session_state.image_index, len(image_files) - 1)
current_image = image_files[st.session_state.image_index]

st.title("Step 7 — Check the results")

without_text = [f for f in image_files if f.stem not in have_text]
st.write(
    f"Image {st.session_state.image_index + 1} of {len(image_files)} with cut-outs"
    + (f" · {len(without_text)} have no text yet" if without_text else "")
)


def jump_label(position: int) -> str:
    image = image_files[position]
    mark = "✅" if image.stem in have_text else "⚠️"
    return f"{mark} {position + 1}. {image.name} · {crop_counts[image.stem]} cut-out(s)"


jump = st.selectbox(
    "Jump to image",
    options=list(range(len(image_files))),
    index=st.session_state.image_index,
    format_func=jump_label,
)
if jump != st.session_state.image_index:
    st.session_state.image_index = jump
    st.rerun()

#matching the current image with the row
matching_rows = df[df["Specimen.image"].str.rsplit(".", n=1).str[0] == current_image.stem]


col3, col4 = st.columns(2) 
col1, col2 = st.columns(2)

with col1:
    # A fixed key means the zoom level survives into the next image instead of
    # snapping back to 1x every time you move on.
    zoom = st.slider(
        "Zoom", min_value=1.0, max_value=6.0, value=1.0, step=0.25,
        format="%.2fx", key="qc_zoom",
        help="Cut-outs can be small. Zoom in, then scroll inside the image to read them.",
    )
    st.markdown(
        zoomable_image(str(current_image), current_image.stat().st_mtime, zoom),
        unsafe_allow_html=True,
    )
    st.caption(f"{current_image.name} · {crop_counts[current_image.stem]} cut-out(s)")



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
#             st.info("Choose at least one column to check.")
#             edited_row = None

#     else:
#         st.warning("No saved text found for this image.")
#         edited_row = None


# Option 2: arranged in different rows

with col2:
    if len(matching_rows) > 0:

        selected_column = st.multiselect(
            "Which columns do you want to check?", df.columns[1:])

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
                            help="Put the original text back",
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
            st.info("Choose at least one column to check.")
            edited_row = None

    else:
        st.warning(
            f"This image has {crop_counts[current_image.stem]} cut-out(s), but no text "
            "came through Step 6. Re-run Step 6, or check the cut-outs are readable."
        )
        edited_row = None



# -----------------------------
# Save changes + go to next
# -----------------------------

with col3:
    if st.session_state.image_index > 0:
        if st.button("◀ Previous image", use_container_width=True):
            if edited_row is not None:

                # Find the original row
                row_index = matching_rows.index

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
        if st.button("Next image ▶", use_container_width=True):

            if edited_row is not None:

                # Find the original row
                row_index = matching_rows.index

                # Save the edited value for the selected column
                df.loc[row_index, selected_column] = (
                    edited_row[selected_column].values
                )

                # Save CSV
                df.to_csv(CSV_FILE, index=False)

                st.session_state.image_index += 1
                st.rerun()
    else:
        st.success("That was the last image.")



