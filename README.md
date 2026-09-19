# Museum Label Whisperer

A Streamlit app that finds the labels in specimen photographs, trains a YOLO
model to detect them, and crops each label out into its own image.

---

# Installation

There are two ways to install. Pick the one that matches you:

| | Who it's for | What you need |
|---|---|---|
| **[Option A — Easy install](#option-a--easy-install-no-coding)** | Anyone. No terminal, no commands. | Just a mouse |
| **[Option B — Terminal install](#option-b--terminal-install)** | People comfortable with a command line | Git and Python |

Both options install the same app and work identically once running.

---

## Option A — Easy install (no coding)

> ⚠️ **Not available yet.** The launcher files described below are still being
> prepared. For now, please use **Option B**, or ask for help setting it up.

Once ready, this is all it will take:

**1. Install Python (one time only)**

Download Python **3.10 or newer** from https://www.python.org/downloads/ and run
the installer.

- On **Windows**, tick the box that says **"Add Python to PATH"** on the first
  screen of the installer. This matters.
- On **macOS**, just click through the installer.

**2. Download the app**

Go to https://github.com/biancasarc/museum_label_whisperer, click the green
**Code** button, then **Download ZIP**. Unzip it somewhere you'll find again,
like your Desktop or Documents.

**3. Install it — double-click once**

Open the unzipped folder and double-click:

- **macOS:** `Install Museum Label Whisperer.command`
- **Windows:** `Install Museum Label Whisperer.bat`

A black window opens and installs everything. This takes several minutes and
downloads around 2 GB, so stay on a good connection. When it says
**"Installation complete"**, close the window.

You only do this once.

> **macOS:** the first time, macOS may refuse to open the file and say it is
> from an unidentified developer. **Right-click** the file, choose **Open**,
> then click **Open** in the dialog. You only need to do this the first time.

**4. Start the app — double-click any time**

Double-click:

- **macOS:** `Start Museum Label Whisperer.command`
- **Windows:** `Start Museum Label Whisperer.bat`

The app opens in your web browser. A black window stays open in the background —
**leave it open** while you work; closing it stops the app.

When you're finished, close the browser tab and then close the black window.

---

## Option B — Terminal install

### 1. Install Python

Python **3.10 or newer** is required. Check what you have:

```bash
python --version
```

If you need it, download Python from https://www.python.org/downloads/

### 2. Clone the repository

```bash
git clone https://github.com/biancasarc/museum_label_whisperer
cd museum_label_whisperer
```

### 3. Create a Python environment (recommended)

It is strongly recommended to install the Python dependencies in a virtual environment.

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

### 4. Run the app

Make sure you are inside the `museum_label_whisperer` directory, then:

```bash
streamlit run app.py
```

The app opens in your browser. Choose a step from the sidebar.

---

# Workflow

Work through the steps **in order**, using the sidebar.

## Step 1 — Upload images

Paste the full path to the folder holding your specimen photographs and choose
how many to import. That many images are picked at random and **copied** into
`data/original`. Your originals are never modified.

Import enough images to train on — around 20–50 is a reasonable start. 20 % of
them are held back automatically for validation.

## Step 2 — Annotate

Draw a rectangle around **every label** in each photo, directly in the app.

- **Draw** a box: click and drag on an empty part of the image.
- **Move / resize** a box: click it to select, then drag it or its corner handles.
- **Delete** a box: switch **Mode** (right of the image) from *Transform* to *Del*,
  click the box, then switch back to *Transform*.
- Click **Complete** to save the annotations for each image.

Your boxes are saved to `data/annotations.json` as you go, so you can close the
app and continue later.

When every image is done, click **Build training dataset**. This writes the YOLO
dataset to `data/yolo_dataset` (80 % training / 20 % validation).

Then open **Check the training labels** and page through a few images. Every red
box should sit around a label. If they don't, fix the boxes above and rebuild.

> If you change any boxes afterwards, **click Build training dataset again** —
> otherwise Step 3 trains on the old boxes. The app warns you when this happens.

## Step 3 — Train the YOLO model

Choose the number of epochs (at least 50 is recommended; more epochs means a
better model but a longer wait) and click **Start training**.

Training can take minutes to hours depending on your computer and the number of
images. The trained model is saved to:

```text
runs/detect/1.1/weights/best.pt
```

A full training log is written to `runs/train_log.txt`.

## Step 4 — Predict

Runs the trained model over a folder of images so you can see what it detects.
Annotated preview images are written to `data/prediction_preview` — open a few
and check the boxes look right before cropping.

## Step 5 — Crop detected labels

Creates one cropped image per detected label. You can set:

- **Number of random images** — 0 processes every image in the folder.
- **Crop buffer** — extra pixels kept around each label.
- **Minimum confidence** — only detections above this score are cropped
  (0.90 by default; lower it if too few labels are being cropped).

Crops are saved to:

```text
data/cropping_result
```

## Step 6 — OCR checking

Review and correct the transcribed text against the photograph, one image at a
time. Use ↺ next to any field to restore the original OCR value.

This step reads `results/data.csv` (the OCR output) and writes your corrections
to `results/modified_data.csv`, leaving the original untouched.

---

# Important notes

⚠️ **Please read!**

- The app is designed to run through the steps **in order**. Skipping a step, or
  moving on before a process has finished, may cause errors. Wait for each step
  to report success before continuing.

- **Rotated photos.** Photographs carrying an EXIF rotation tag are handled
  consistently at every step, so boxes always line up with what you see on
  screen. Nothing is required from you.

- The results of every step are in the data directory of each project
