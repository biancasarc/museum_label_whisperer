"""Step 2 - Annotate labels directly in the app (no CVAT / Docker needed).

Boxes are drawn with the ``streamlit-image-annotation`` component, stored per
image in ``data/annotations.json`` (original-image pixel coordinates) and
turned into a YOLO dataset with the "Build training dataset" button.
"""

from pathlib import Path

import streamlit as st
from streamlit_image_annotation import detection

from backend.dataset import (
    build_yolo_dataset,
    list_images,
    load_annotations,
    render_yolo_labels,
    save_annotations,
)
from backend.images import exif_orientation, load_rgb, make_display_image, raw_to_upright_box

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: {current_proj}")

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj
ORIGINAL_DIR = PROJECT_ROOT / "data" / "01_train_val_subset"
ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "annotations.json"
DATASET_DIR = PROJECT_ROOT / "data" / "02_yolo_dataset"

LABEL_LIST = ["label"]  # single class, id 0
img_size = 700  # px; longest side of the image shown in the browser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOADER_VERSION = 2  # bump when load_rgb changes so cached (possibly rotated) images are dropped


@st.cache_resource(show_spinner="Loading image…", max_entries=40)
def get_display_image(path_str: str, mtime: float, max_side: int, version: int = LOADER_VERSION):
    img = load_rgb(Path(path_str))
    display, scale = make_display_image(img, max_side)
    return display, scale, img.size  # img.size == (width, height)


def get_state():
    """Initialise session state on first visit."""
    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations(ANNOTATIONS_FILE)
    if "annotate_index" not in st.session_state:
        st.session_state.annotate_index = 0
    if "annotate_rev" not in st.session_state:
        # per-image revision counter: bumping it gives the component a fresh
        # key so it forgets a stale value after we change boxes in Python
        st.session_state.annotate_rev = {}


def persist():
    save_annotations(ANNOTATIONS_FILE, st.session_state.annotations)


def set_boxes(name: str, boxes: list, size: tuple[int, int]):
    """Store *boxes* (original pixels) for image *name* and write to disk."""
    st.session_state.annotations[name] = {
        "width": int(size[0]),
        "height": int(size[1]),
        "boxes": [[float(v) for v in b] for b in boxes],
        "frame": "upright",  # coordinates refer to the EXIF-corrected (upright) image
    }
    persist()


def migrate_legacy_entries(annotations: dict, images: list[Path]) -> int:
    """Boxes saved by the first version of this page were drawn on the raw
    pixel grid (photos with an EXIF rotation tag appeared upside down).  Move
    them into the upright frame once, so they match what is trained on now."""
    by_name = {p.name: p for p in images}
    changed = 0
    for name, entry in annotations.items():
        if entry.get("frame") == "upright" or name not in by_name:
            continue
        orientation = exif_orientation(by_name[name])
        w, h = int(entry.get("width", 0)), int(entry.get("height", 0))
        if orientation != 1 and w and h:
            entry["boxes"] = [raw_to_upright_box(b, w, h, orientation) for b in entry.get("boxes", [])]
            if orientation >= 5:
                entry["width"], entry["height"] = h, w
        entry["frame"] = "upright"
        changed += 1
    return changed


def bump_rev(name: str):
    st.session_state.annotate_rev[name] = st.session_state.annotate_rev.get(name, 0) + 1


def go(delta: int, n: int):
    st.session_state.annotate_index = max(0, min(n - 1, st.session_state.annotate_index + delta))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Step 2 - Annotate")
st.write(
    "Draw a rectangle around **every label** in each photo. Your boxes are saved "
    "as you go, so you can close the app and continue later."
)

get_state()
images = list_images(ORIGINAL_DIR)
n_images = len(images)

if n_images == 0:
    st.warning(
        f"No images found in `{ORIGINAL_DIR.relative_to(PROJECT_ROOT)}`. "
        "Go back to Step 1 and import some images first."
    )
    st.stop()

# Drop annotations for images that are no longer present (e.g. after the user
# cleared data/original and imported a new sample).
present = {p.name for p in images}
stale = [k for k in st.session_state.annotations if k not in present]
if stale:
    for k in stale:
        del st.session_state.annotations[k]
    persist()

annotations = st.session_state.annotations
if migrate_legacy_entries(annotations, images):
    persist()
index = min(st.session_state.annotate_index, n_images - 1)
st.session_state.annotate_index = index
current = images[index]
name = current.name

done = sum(1 for p in images if p.name in annotations)
remaining = n_images - done

# --- progress ---------------------------------------------------------------
st.progress(done / n_images, text=f"Image {index + 1} of {n_images} · {remaining} remaining to annotate")


# --- annotator --------------------------------------------------------------

with st.expander("How to annotate", expanded=done == 0):
    st.markdown(
        """
- **Draw** a box: click and drag on an empty part of the image.
- **Move / resize** a box: click it to select, then drag it or its corner handles.
- **Delete** a box: switch **Mode** (right of the image) from *Transform* to *Del*, click the box,
  then switch back to *Transform* to keep drawing.
- When the image is done, click **Complete** under the image (or press **Space**).
  This saves the boxes — the status line below the image turns green.
- Then click **Next ▶**. Photos with no labels: just click **Complete** without drawing.
"""
    )

size_slider, null,nav_prev, nav_next = st.columns(4)


### Image slider needs more testing, doesnt work properly yet
# with size_slider:
#     img_size= st.slider(
#         "Change image display side:",
#         min_value = 300,
#         max_value = 1500,
#         value=700,
#         step=50,
#         width= "stretch"
#     )


try:
    display_img, scale, original_size = get_display_image(str(current), current.stat().st_mtime, img_size, LOADER_VERSION)
except Exception as error:
    st.error(f"Could not open `{name}`: {error}")
    st.stop()

entry = annotations.get(name)
saved_boxes = entry["boxes"] if entry else []
# original pixels -> display pixels for the component
display_boxes = [[v / scale for v in box] for box in saved_boxes]




# --- navigation -------------------------------------------------------------

with nav_prev:
    st.button(
        "◀ Previous", width="stretch", disabled=index == 0,
        on_click=go, args=(-1, n_images),
    )
with nav_next:
    st.button(
        "Next ▶", width="stretch", disabled=index >= n_images - 1,
        on_click=go, args=(1, n_images),
    )


rev = st.session_state.annotate_rev.get(name, 0)
result = detection(
    image_path=display_img,
    label_list=LABEL_LIST,
    bboxes=display_boxes,
    labels=[0] * len(display_boxes),
    width=display_img.size[0],
    height=display_img.size[1],
    line_width=4,
    use_space=True,
    key=f"annot::{name}::{rev}",
)

if result is not None:
    # The component only sends a value when the user clicks "Complete" /
    # presses Space. Convert display pixels back to original pixels.
    new_boxes = [[round(v * scale, 2) for v in item["bbox"]] for item in result]
    # tolerant comparison so float round-tripping cannot trigger endless reruns
    if entry is None or [[round(float(v), 2) for v in b] for b in saved_boxes] != new_boxes:
        set_boxes(name, new_boxes, original_size)
    # Auto-advance: Complete saves and moves to the next image automatically.
    # On the last image, just refresh the UI so the saved status updates.
    if index < n_images - 1:
        st.session_state.annotate_index = index + 1
    st.rerun()


status_col, act1, act2 = st.columns([3, 1, 1])
with status_col:
    if entry is None:
        st.warning("Not saved yet — draw the boxes, then click **Complete**.")
    else:
        st.success(f"Saved: {len(saved_boxes)} box(es) on this image "
                   f"({original_size[0]}x{original_size[1]} px original).")
with act1:
    if st.button("No labels here", width="stretch",
                 help="Save this image with zero boxes (it becomes a background example)."):
        set_boxes(name, [], original_size)
        bump_rev(name)
        st.rerun()
with act2:
    if st.button("Clear boxes", width="stretch", disabled=not saved_boxes,
                 help="Remove all saved boxes on this image."):
        set_boxes(name, [], original_size)
        bump_rev(name)
        st.rerun()

# --- build dataset ----------------------------------------------------------
st.divider()
st.subheader("Build training dataset")
st.caption(
    f"Copies the images to `{(DATASET_DIR / 'images').relative_to(PROJECT_ROOT)}` and writes YOLO "
    f"label files to `{(DATASET_DIR / 'labels').relative_to(PROJECT_ROOT)}` "
    "(80 % train / 20 % validation). Your originals are not modified."
)

unannotated = [p.name for p in images if p.name not in annotations]
build_anyway = True
if unannotated:
    st.warning(
        f"{len(unannotated)} image(s) have not been annotated yet: "
        + ", ".join(unannotated[:5]) + (" …" if len(unannotated) > 5 else "")
    )
    build_anyway = st.checkbox(
        "Build anyway (unannotated images are treated as having no labels)", value=False
    )

total_boxes = sum(len(e.get("boxes", [])) for e in annotations.values())
if total_boxes == 0:
    st.info("Draw at least one box before building the dataset.")

if st.button(
    "Build training dataset", type="primary", icon=":material/dataset:",
    disabled=not build_anyway or total_boxes == 0,
):
    with st.spinner("Writing YOLO dataset…"):
        try:
            out = build_yolo_dataset(ORIGINAL_DIR, annotations, DATASET_DIR)
        except Exception as error:
            st.error(f"Building the dataset failed: {error}")
        else:
            n_train = len((out / "train.txt").read_text().splitlines())
            n_val = len((out / "val.txt").read_text().splitlines())
            st.success(
                f"Dataset ready in `{out.relative_to(PROJECT_ROOT)}`: "
                f"{n_train} training and {n_val} validation images, {total_boxes} boxes."
            )
            st.info("Continue to Step 3 to train the model.")
            st.session_state["dataset_built"] = True

# --- verify what the model will actually see --------------------------------
if (DATASET_DIR / "train.txt").is_file():
    with st.expander("Check the training labels (what YOLO will see)", expanded=st.session_state.pop("dataset_built", False)):
        st.caption(
            "These are the files in `data/yolo_dataset`, read exactly the way the trainer reads them. "
            "Every red box should surround a label. If not, fix the boxes above and rebuild."
        )
        built = sorted((DATASET_DIR / "images").glob("*/*.*"))
        if built:
            pick = st.selectbox("Training image", options=built, format_func=lambda p: f"{p.parent.name}/{p.name}")
            label_file = DATASET_DIR / "labels" / pick.parent.name / f"{pick.stem}.txt"
            try:
                st.image(render_yolo_labels(pick, label_file), caption=f"{pick.name} · {label_file.name}")
            except Exception as error:
                st.error(str(error))
