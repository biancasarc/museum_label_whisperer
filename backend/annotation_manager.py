import os

from PIL import Image


IMAGE_FOLDER = "data/images"
LABEL_FOLDER = "data/labels"

os.makedirs(LABEL_FOLDER, exist_ok=True)


def get_images():

    images = sorted([
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    return images


def save_yolo_label(image_name, box):

    image = Image.open(
        os.path.join(IMAGE_FOLDER, image_name)
    )

    img_w, img_h = image.size

    left = box["left"]
    top = box["top"]
    width = box["width"]
    height = box["height"]

    x_center = (left + width / 2) / img_w
    y_center = (top + height / 2) / img_h

    width /= img_w
    height /= img_h

    label_path = os.path.join(
        LABEL_FOLDER,
        image_name.rsplit(".", 1)[0] + ".txt"
    )

    with open(label_path, "w") as f:

        f.write(
            f"0 {x_center} {y_center} {width} {height}\n"
        )