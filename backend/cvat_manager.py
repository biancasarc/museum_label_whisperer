import os
import shutil
from pathlib import Path
import random
import zipfile

from cvat_sdk import make_client


HOST = "localhost"
PORT = 8080



def set_credentials(username: str, password: str):
    """Set CVAT credentials at runtime (used by Streamlit UI)."""
    global USERNAME, PASSWORD
    USERNAME = username
    PASSWORD = password

IMAGE_FOLDER = "data/original"


def connect():
    client = make_client(HOST, port=PORT)
    client.login((USERNAME, PASSWORD))
    return client


def create_task(task_name):
    client = connect()
    task = client.tasks.create_from_data(
        spec={
            "name": task_name,
            "labels": [{"name": "label", "type": "rectangle"}],
        },
        resources=[
            os.path.join(IMAGE_FOLDER, image_name)
            for image_name in os.listdir(IMAGE_FOLDER)
        ],
    )
    client.logout()
    return task.id


def open_browser():
    import webbrowser

    webbrowser.open("http://localhost:8080")


def export_dataset(task_id, output_dir="data/yolo_dataset"):
    client = connect()
    task = client.tasks.retrieve(task_id)
    zip_file = "data/temp_export.zip"
    task.export_dataset(
        format_name="Ultralytics YOLO Detection 1.0",
        filename=zip_file,
    )
    client.logout()

    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(zip_file)

    return output_dir

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