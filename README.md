# Museum Label Whisperer

A small Streamlit app that helps museum staff crop specimen labels out of
photographs. You import a sample of photos, draw boxes around the labels
directly in the app, train a YOLOv8 detector on those boxes, and then use the
trained model to find and crop the labels in the rest of your images.

No Docker, no external annotation tool — everything runs in the app.

# Installation

## 1. Install Python

You need Python 3.10 or newer. Download it from https://www.python.org/downloads/
(on Windows, tick **"Add python.exe to PATH"** in the installer).

## 2. Get the Museum Label Whisperer code

Either download the ZIP from GitHub (green **Code** button → **Download ZIP**)
and unzip it, or clone the repository:

```bash
git clone https://github.com/biancasarc/museum_label_whisperer
cd museum_label_whisperer
```

## 3. Create a Python environment (recommended)

It is strongly recommended to install the Python dependencies in a virtual environment.

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

## 4. Run the app

Make sure you are inside the `museum_label_whisperer` directory, then start the app:

```bash
streamlit run app.py
```

Your browser opens automatically. Follow the steps in the sidebar in order.

---

# Workflow

| Step | Page | What happens |
|------|------|--------------|
| 1 | Upload | Copies a random sample (default 20) of your photos into `data/original`. Your originals are never modified. |
| 2 | Annotate | Draw a box around every label, image by image. Boxes are saved automatically in `data/annotations.json`. Click **Build training dataset** when done. |
| 3 | Train | Trains a YOLOv8 model on the dataset from Step 2 (at least 50 epochs recommended). |
| 4 | Predict | Runs the trained model on your full image folder. |
| 5 | Crop | Saves one cropped image per detected label. |

## Annotating (Step 2)

- **Draw** a box: click and drag on the image.
- **Move / resize**: click a box to select it, then drag it or its corner handles.
- **Delete**: switch *Mode* (right of the image) to **Del**, click the box, then switch back to *Transform*.
- Click **Complete** (or press Space) to save the boxes for that image, then **Next**.
- Photos without any label: click **Complete** without drawing (or the *No labels here* button).
- Large photos are shown downscaled, but boxes are stored in the original image's pixel coordinates.
- You can close the app and come back later — your progress is kept.

---

# Important notes

⚠️ **Please read!**

- The application is designed to run through the steps in order. **Skipping steps or proceeding before a process has finished successfully may cause errors.**
  Make sure each button is clicked in sequence and that each process completes successfully before moving to the next step.

- After each run, intermediate files will be stored in:

```text
museum_label_whisperer/data
```

  Before starting a new run, **delete all files inside the `data` directory** to avoid conflicts with previous runs.

- The results (cropped labels) are stored here:

```text
museum_label_whisperer/data/cropping_result
```
