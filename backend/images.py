"""Image loading helpers shared by the annotation page.

Specimen photos are often large TIFFs (sometimes 16-bit).  The annotator
component only needs an 8-bit RGB PNG/JPEG-like image for display, so we
load with Pillow, fall back to OpenCV for TIFF flavours Pillow cannot read,
and downscale a *copy* for the browser.  Boxes are always stored in the
coordinates of the original file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Large museum scans can exceed Pillow's default decompression-bomb limit.
Image.MAX_IMAGE_PIXELS = None

EXIF_ORIENTATION_TAG = 274


def exif_orientation(path: Path) -> int:
    """Return the EXIF orientation tag (1-8) of *path*, 1 when absent/unreadable."""
    try:
        with Image.open(path) as im:
            value = im.getexif().get(EXIF_ORIENTATION_TAG, 1)
        return int(value) if value in range(1, 9) else 1
    except Exception:
        return 1


def raw_to_upright_box(box, raw_width: int, raw_height: int, orientation: int):
    """Map an ``[x, y, w, h]`` box drawn on the *raw* (un-rotated) pixel grid to
    the *upright* frame produced by ``ImageOps.exif_transpose``.

    Orientations 5-8 rotate by 90 degrees, so width/height are swapped.
    """
    x, y, w, h = (float(v) for v in box)
    W, H = float(raw_width), float(raw_height)
    return {
        1: [x, y, w, h],
        2: [W - x - w, y, w, h],
        3: [W - x - w, H - y - h, w, h],
        4: [x, H - y - h, w, h],
        5: [y, x, h, w],
        6: [H - y - h, x, h, w],
        7: [H - y - h, W - x - w, h, w],
        8: [y, W - x - w, h, w],
    }.get(int(orientation), [x, y, w, h])


def load_rgb(path: Path) -> Image.Image:
    """Open *path* and return an 8-bit RGB Pillow image at full resolution,
    **upright** (EXIF orientation applied).

    Everything downstream works in this upright frame: boxes are stored in
    it, training images are re-saved in it (see ``dataset.copy_training_image``)
    and prediction/cropping load images the same way.  This avoids the classic
    mismatch where one library honours the EXIF rotation tag and another
    does not.
    """
    path = Path(path)
    try:
        img = Image.open(path)
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(img, dtype=np.float32)
            arr = _to_uint8(arr)
            return Image.fromarray(arr).convert("RGB")
        return img.convert("RGB")
    except Exception:
        pass  # fall through to OpenCV

    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise ValueError(f"Could not read image: {path}")
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr.astype(np.float32))
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def make_display_image(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Return a downscaled copy of *img* whose longest side is <= *max_side*
    together with ``scale = original / display`` (>= 1).

    Multiply display-space coordinates by ``scale`` to get original pixels.
    """
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img.copy(), 1.0
    scale = longest / max_side
    new_size = (max(1, round(w / scale)), max(1, round(h / scale)))
    display = img.resize(new_size, Image.Resampling.BILINEAR)
    # Recompute from the actual integer size so the round trip is exact.
    return display, w / display.size[0]
