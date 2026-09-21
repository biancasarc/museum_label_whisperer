"""Dataset helpers for Museum Label Whisperer.

This module replaces the CVAT-based ``backend/cvat_manager.py``.  It holds:

* the annotation store used by ``pages/2_02._Annotate_images.py`` (a small JSON file with
  one entry per image, boxes kept in *original image pixel* coordinates),
* ``build_yolo_dataset`` which turns those annotations into the YOLO folder
  layout that ``processing_yolo_dir`` expects, and
* ``processing_yolo_dir`` itself, moved here unchanged from ``cvat_manager.py``.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_ID = 0  # single class: "label"


# ---------------------------------------------------------------------------
# Image listing
# ---------------------------------------------------------------------------

def list_images(folder: Path) -> list[Path]:
    """Return the supported image files in *folder*, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Annotation store
# ---------------------------------------------------------------------------
#
# Format of the JSON file (one entry per image *name*):
#
#   {
#     "IMG_0001.tif": {"width": 6000, "height": 4000,
#                      "boxes": [[x, y, w, h], ...]},   # pixels, top-left origin
#     ...
#   }
#
# An image with an entry but an empty "boxes" list has been reviewed and has
# no labels on it; an image with no entry at all has not been annotated yet.


def load_annotations(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_annotations(path: Path, annotations: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(annotations, fh, indent=1)
    os.replace(tmp, path)  # atomic on every OS Streamlit runs on


# ---------------------------------------------------------------------------
# YOLO conversion
# ---------------------------------------------------------------------------

def boxes_to_yolo_lines(boxes: list, width: int, height: int) -> list[str]:
    """Convert ``[x, y, w, h]`` pixel boxes into YOLO label lines.

    Boxes are clipped to the image; degenerate boxes (zero area after clipping)
    are dropped.
    """
    lines = []
    for box in boxes:
        x, y, w, h = (float(v) for v in box)
        x1, y1 = max(0.0, min(x, x + w)), max(0.0, min(y, y + h))
        x2, y2 = min(float(width), max(x, x + w)), min(float(height), max(y, y + h))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        xc = (x1 + x2) / 2 / width
        yc = (y1 + y2) / 2 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def copy_training_image(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* for training.

    Plain 8-bit RGB files without an EXIF rotation are copied byte-for-byte.
    Everything else is re-saved under the same name so that the file's raw
    pixel grid *is* the upright image the boxes were drawn on:

    * photos with an EXIF orientation tag are physically rotated (ultralytics
      reads raw pixels and ignores the tag, so the tag must be baked in);
    * 16-bit or single-channel files (scanned TIFFs) become 8-bit RGB, because
      ultralytics' mosaic augmentation cannot mix them with ordinary RGB.

    The original in ``data/original`` is never modified.
    """
    from PIL import Image

    from backend.images import exif_orientation

    try:
        with Image.open(src) as im:
            needs_convert = im.mode != "RGB" or exif_orientation(src) != 1
    except Exception:
        needs_convert = True  # let the robust loader deal with it

    if not needs_convert:
        shutil.copy2(src, dst)
        return

    from backend.images import load_rgb

    rgb = load_rgb(src)
    if dst.suffix.lower() in (".jpg", ".jpeg"):
        rgb.save(dst, quality=95)
    elif dst.suffix.lower() in (".tif", ".tiff"):
        rgb.save(dst, compression="tiff_deflate")
    else:
        rgb.save(dst)


def build_yolo_dataset(original_dir: Path, annotations: dict, output_dir: Path) -> Path:
    """Write ``images/train`` + ``labels/train`` from *annotations* and finish
    the dataset with :func:`processing_yolo_dir`.

    * Images are **copied** (never moved) out of *original_dir*.
    * Every image gets a ``.txt`` file, empty when it has no boxes.
    * Any previous content of *output_dir* is removed first so that a rebuild
      does not mix in files from an earlier run.
    """
    original_dir = Path(original_dir)
    output_dir = Path(output_dir).expanduser().resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_train = output_dir / "images" / "train"
    labels_train = output_dir / "labels" / "train"
    images_train.mkdir(parents=True)
    labels_train.mkdir(parents=True)

    for image_path in list_images(original_dir):
        copy_training_image(image_path, images_train / image_path.name)
        entry = annotations.get(image_path.name, {})
        lines = boxes_to_yolo_lines(
            entry.get("boxes", []),
            int(entry.get("width", 0) or 0),
            int(entry.get("height", 0) or 0),
        ) if entry.get("width") and entry.get("height") else []
        with open(labels_train / f"{image_path.stem}.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))

    return processing_yolo_dir(output_dir)


def render_yolo_labels(image_path: Path, label_path: Path, max_side: int = 900):
    """Draw the YOLO boxes from *label_path* onto *image_path* exactly as
    ultralytics will read them (raw pixels, no EXIF handling) and return a
    downscaled Pillow image for a visual check."""
    import cv2
    from PIL import Image, ImageDraw

    arr = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if arr is None:
        raise ValueError(f"Could not read {image_path}")
    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    W, H = img.size
    draw = ImageDraw.Draw(img)
    lw = max(2, int(max(W, H) / 400))
    if Path(label_path).is_file():
        for line in Path(label_path).read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            x1, y1 = (xc - bw / 2) * W, (yc - bh / 2) * H
            x2, y2 = (xc + bw / 2) * W, (yc + bh / 2) * H
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=lw)
    scale = max(W, H) / max_side
    if scale > 1:
        img = img.resize((int(W / scale), int(H / scale)))
    return img


# ---------------------------------------------------------------------------
# Moved verbatim from backend/cvat_manager.py
# ---------------------------------------------------------------------------

def processing_yolo_dir(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    repo_root = output_dir.parents[1]
    os.makedirs(output_dir/"images"/"val", exist_ok=True)
    os.makedirs(output_dir/"labels"/"val", exist_ok=True)

    # create the YAML with repo-root path, so train/val can use data/yolo_dataset/ paths
    with open(output_dir / "data.yaml", "w") as d:  #overwriting the data.yaml
        d.write(f"train: train.txt\nval: val.txt\nnc: 1\nnames:\n  0: label\npath: {output_dir}\n")

    #selecting 20% of training img and moving them to val
    images_path =Path(output_dir/"images"/"train").expanduser().resolve()
    images = [p for p in images_path.iterdir() if p.is_file()]
    train_count=len(images)
    val_count = int(round(len(images) * 0.2))
    if val_count == 0:
        val_count = 1
    if val_count == train_count:
        val_count = train_count - 1

    val_images = random.sample(images, val_count)

    for img in val_images:
        shutil.move(img, output_dir / "images" / "val" / img.name)

        label_file = output_dir / "labels" / "train" / f"{img.stem}.txt"
        if label_file.exists():
            shutil.move(label_file, output_dir / "labels" / "val" / label_file.name)
        else:
            print(f"Warning: label not found for image {img.name}")

    # remove stale dataset caches before regenerating split lists
    for cache_file in (output_dir / "labels").glob("*.cache"):
        if cache_file.exists():
            cache_file.unlink()

    # creating the val.txt
    val_path = Path(output_dir / "images" / "val").expanduser().resolve()

    # write only actual image files (skip dotfiles & folders)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    with open(output_dir / "val.txt", "w") as v:
        for image in sorted(val_path.iterdir()):
            if not image.is_file():
                continue
            if image.name.startswith("."):
                continue
            if image.suffix.lower() not in valid_exts:
                continue
            v.write(image.relative_to(repo_root).as_posix() + "\n")

    # regenerate train.txt from remaining images in images/train
    train_path = Path(output_dir / "images" / "train").expanduser().resolve()
    with open(output_dir / "train.txt", "w") as t:
        for image in sorted(train_path.iterdir()):
            if not image.is_file():
                continue
            if image.name.startswith("."):
                continue
            if image.suffix.lower() not in valid_exts:
                continue
            t.write(image.relative_to(repo_root).as_posix() + "\n")

    return output_dir
