# Museum Label Whisperer

An app that finds the things you care about inside a set of images, cuts them out,
and turns any text in them into a table you can work with.

You teach it what to look for by drawing boxes on a handful of images first, so it
can learn to find anything that looks reasonably consistent — specimen labels, pages
of a book, signs, forms, plant tags.

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

It is strongly recommended to install the dependencies in a virtual environment.

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

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

### 4. Run the app

Make sure you are inside the `museum_label_whisperer` directory, then:

```bash
python -m streamlit run app.py
```

The app opens in your browser. Choose a step from the sidebar.

> **Use `python -m streamlit`, not plain `streamlit`.** If you have Anaconda or
> Miniconda installed, a bare `streamlit` command often starts a *different*
> Python that has none of these packages, and the app fails with
> `ModuleNotFoundError`. Writing it this way always uses the environment you
> just activated.
>
> If your prompt shows `(base)` — with or without `(.venv)` — run
> `conda deactivate` first, then `source .venv/bin/activate`. The prompt should
> read `(.venv)` on its own.

---

# What you need before you start

- **A folder of images.** They stay where they are; the app only ever copies them.
- **An OpenAI API key**, but only for Step 6 (reading the text). Steps 1–5 work
  without one. Get a key at https://platform.openai.com/api-keys — note that
  reading text is charged per image by OpenAI, so it costs a small amount of
  money to run.

---

# Projects

The front page is where you create, choose and delete projects. Everything a
project produces is kept in its own folder under `projects/<name>/data/`, so
separate projects never overwrite each other's results.

**Create or choose a project before you start Step 1.** The name shown at the top
of every page is the project your work is being filed under.

> Project names cannot contain spaces — use underscores instead.
>
> If you begin a step without choosing a project, the app files the work under a
> folder literally called `No project selected`. Nothing is lost, but it is not
> filed under a real project: pick one on the front page and run the step again.

---

# Workflow

Work through the steps **in order**, using the sidebar. Each step needs the one
before it to have finished.

## Step 1 — Upload images

Paste the full path to the folder holding your images and choose how many to bring
in. That many are picked at random and **copied** into `data/01_train_val_subset`.
Your originals are never changed.

Around 20–50 is a reasonable start. 20% are held back automatically so the app can
check its own work. Use more if the objects you want to find are complex, or vary a
lot between images.

## Step 2 — Annotate

Draw a box around **every object** you want the app to find, directly in the app.

- **Draw** a box: click and drag on an empty part of the image.
- **Move or resize** a box: click it to select, then drag it or its corner handles.
- **Delete** a box: switch **Mode** (right of the image) from *Transform* to *Del*,
  click the box, then switch back to *Transform*.
- Click **Complete** to save that image. The status line turns green.
- For an image with nothing to find, click **Complete** without drawing, or use
  **Nothing to find here**. These are useful — they teach the app what to ignore.

Your boxes are saved to `data/annotations.json` as you go, so you can close the app
and come back later.

When every image is done, click **Build training set**. This writes the training
files to `data/02_yolo_dataset` (80% to learn from, 20% kept back for checking).

Then open **Check your boxes** and page through a few images. Every red box should
sit around an object. If not, fix the boxes above and build again.

> If you change any boxes afterwards, **click Build training set again** —
> otherwise Step 3 learns from the old boxes. The app warns you when this happens.

## Step 3 — Train the model

Choose the number of training rounds — at least 50 is recommended; more rounds
means better results but a longer wait — then click **Start training**.

This can take minutes to hours depending on your computer and how many images you
used. The trained model is saved to:

```text
runs/detect/1.1/weights/best.pt
```

A full log is written to `runs/train_log.txt`. If training fails, that file usually
says why.

## Step 4 — Predict

Runs the trained model over a folder of images so you can see what it finds, before
you cut anything out. Preview images with the boxes drawn on are written to
`data/03_prediction_preview` — open a few and check they look right.

## Step 5 — Crop

Cuts out one image per object found. You can set:

- **How many images to do** — 0 does all of them.
- **Extra space around each cut-out** — pixels kept around the edge.
- **Minimum confidence** — only objects the model is at least this sure about are
  cut out (0.90 by default; lower it if too few are coming through).

Cut-outs are saved to `data/04_cropping_result`.

## Step 6 — Read the text

Needs an **OpenAI API key**, pasted into the settings box at the top. It is used
only while the app is open and is never saved to your computer. You will need to
paste it again each time you restart.

This step has two parts, which you can run separately:

**Part 1 — Read the text.** Sends each cut-out image off to be read and saves the
result to `data/05_ocr_results`. You can edit the reading instructions and save
them; they are kept in `data/ocr_prompt.txt`.

**Part 2 — Sort into columns.** Groups the cut-outs back together by the image they
came from, then uses five editable prompts to sort the text into columns of your
choice — these can follow an existing standard (GBIF Darwin Core, for example), or
whatever suits your own records. You can add your own prompts too. Results go to
`data/06_structured_output`.

Both parts skip anything already done, so you can stop and resume.

## Step 7 — Check the results

Go through the results one image at a time, with the original image beside them, and
correct anything that came out wrong. Use ↺ next to any field to put the original
text back.

Reads `data/06_structured_output/structured_dwc_metadata.csv` and writes your
corrections to `data/07_quality_checking/corrected_metadata.csv`, leaving the
original untouched.

---

# Where everything is saved

Everything lives under the folder for the project you have chosen:

```text
projects/<your project>/
  data/01_train_val_subset     images brought in at Step 1
  data/annotations.json        the boxes you drew at Step 2
  data/02_yolo_dataset         the training set built at Step 2
  data/03_prediction_preview   previews from Step 4
  data/04_cropping_result      cut-out images from Step 5
  data/05_ocr_results          text read at Step 6, part 1
  data/06_structured_output    columns sorted at Step 6, part 2
  data/07_quality_checking     your corrections from Step 7
  runs/detect/<name>/weights/  the trained model from Step 3
  runs/train_log.txt           training log
```

---

# Important notes

⚠️ **Please read!**

- Run the steps **in order**, and wait for each one to report success before moving
  on. Skipping ahead, or starting a step before the last one finished, causes errors.

- **Starting fresh.** To begin again with a different set of images, create a new
  project on the front page. The old project's results stay where they are. To
  redo a project in place instead, delete its `data/` and `runs/` folders.

- **Rotated images.** Images carrying an EXIF rotation tag are handled consistently
  at every step, so boxes always line up with what you see on screen. Nothing is
  required from you.
