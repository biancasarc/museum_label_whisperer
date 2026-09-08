from openai import OpenAI
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO

import base64
import json
import csv
import re
import time

# =========================================================
# SETTINGS
# =========================================================

UPSCALE_FACTOR = 1.9
NUM_SPECIMENS  = 800

# =========================================================
# PATHS
# =========================================================

IMAGE_FOLDER = Path(
    "/Users/christinapanagaki/Documents/museum.labels.project/2026/final_run/full-pipeline-other-collections/florida-museum-extras/low-res/low-res-label-crops"
)

OUTPUT_DIR = Path(
    "/Users/christinapanagaki/Documents/museum.labels.project/2026/final_run/full-pipeline-other-collections/florida-museum-extras/raw-ocr-low-res"
)

PROMPT_PATH = Path("/Users/christinapanagaki/Documents/museum.labels.project/2026/final_run/ocr/ocr_prompt.txt")
API_KEY_PATH = Path("/Users/christinapanagaki/Documents/museum.labels.project/2026/api.key")

CSV_PATH = OUTPUT_DIR / "raw-label-ocr-florida-leftovers-low-res.csv"
RAW_RESPONSE_DIR = OUTPUT_DIR / "raw_responses_low_res"

# Live-updated token usage snapshot (overwritten after every image,
# so token spend is never lost even if the run is force-stopped)
TOKEN_STATS_JSON = OUTPUT_DIR / "token_usage_progress.json"

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_RESPONSE_DIR.mkdir(exist_ok=True)

# =========================================================
# START TIMER
# =========================================================

start_time = time.time()

# =========================================================
# LOAD API KEY
# =========================================================

with open(API_KEY_PATH, "r", encoding="utf-8") as f:
    api_key = f.read().strip()

# =========================================================
# LOAD SYSTEM PROMPT
# =========================================================

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    system_prompt = f.read()

# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(api_key=api_key)

# =========================================================
# TOKEN TRACKING
# =========================================================

total_input_tokens  = 0
total_cached_tokens = 0
total_output_tokens = 0
total_tokens        = 0

def save_token_stats(images_done, images_total, elapsed_seconds):
    """
    Overwrite TOKEN_STATS_JSON with current cumulative token usage.
    Called after every image (and on interrupt/finish) so nothing
    is lost if the run is stopped early.
    """
    snapshot = {
        "images_done": images_done,
        "images_total": images_total,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "totals": {
            "input": total_input_tokens,
            "cached": total_cached_tokens,
            "output": total_output_tokens,
            "total": total_tokens
        }
    }

    with open(TOKEN_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

# =========================================================
# CSV FIELDS
# =========================================================

csv_fields = [
    "Specimen.image",
    "Image.name",
    "Segments"
]

# =========================================================
# HELPERS
# =========================================================

def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    # Parse only the first complete JSON object, ignoring any trailing
    # extra content -- more robust than slicing to the last "}".
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON object from response: {e}")

    return obj


def get_specimen_id(filename):
    match = re.match(
        r"(.+?)_d_cropped_label_\d+",
        Path(filename).stem
    )
    return match.group(1) if match else Path(filename).stem


def get_original_image_name(filename):
    specimen_id = get_specimen_id(filename)
    return f"{specimen_id}.png"

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")

    # UPSCALE
    new_size = (
        int(img.width  * UPSCALE_FACTOR),
        int(img.height * UPSCALE_FACTOR)
    )
    img = img.resize(new_size, Image.LANCZOS)

    # AUTOCONTRAST
    img = ImageOps.autocontrast(img, cutoff=0.5)

    # SAVE TO MEMORY
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode("utf-8")

# =========================================================
# LOAD IMAGES
# =========================================================

if not IMAGE_FOLDER.exists():
    raise FileNotFoundError(
        f"Image folder not found:\n{IMAGE_FOLDER}"
    )

image_paths = sorted(
    list(IMAGE_FOLDER.glob("*.png"))  +
    list(IMAGE_FOLDER.glob("*.PNG"))  +
    list(IMAGE_FOLDER.glob("*.jpg"))  +
    list(IMAGE_FOLDER.glob("*.JPG"))  +
    list(IMAGE_FOLDER.glob("*.jpeg")) +
    list(IMAGE_FOLDER.glob("*.JPEG"))
)

if len(image_paths) == 0:
    raise FileNotFoundError(
        f"No images found in:\n{IMAGE_FOLDER}"
    )

# =========================================================
# LIMIT TO FIRST N SPECIMENS
# =========================================================

selected_specimens   = set()
filtered_image_paths = []

for path in image_paths:
    specimen_id = get_specimen_id(path.name)

    if specimen_id not in selected_specimens:
        if len(selected_specimens) >= NUM_SPECIMENS:
            continue
        selected_specimens.add(specimen_id)

    filtered_image_paths.append(path)

image_paths = filtered_image_paths

print(f"Selected {len(selected_specimens)} specimens")
print(f"Processing {len(image_paths)} cropped labels")

# =========================================================
# RESUME: figure out what's already been done
# =========================================================

already_done_names = set()  # Image.name values already written to the CSV

if CSV_PATH.exists():
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for existing_row in reader:
            name = existing_row.get("Image.name")
            if name:
                already_done_names.add(name)

skipped_already_done = [p for p in image_paths if p.name in already_done_names]
remaining_image_paths = [p for p in image_paths if p.name not in already_done_names]

if skipped_already_done:
    print(f"Resuming: {len(skipped_already_done)} labels already in {CSV_PATH.name}, skipping those")
print(f"Remaining to process this run: {len(remaining_image_paths)}")

image_paths = remaining_image_paths

# =========================================================
# CREATE / RESUME CSV
# =========================================================

if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

# =========================================================
# OCR FUNCTION
# =========================================================

def process_image(image_path, index, total, reuse_saved_response=True):
    global total_input_tokens
    global total_cached_tokens
    global total_output_tokens
    global total_tokens

    raw_response_path = RAW_RESPONSE_DIR / f"{image_path.stem}.txt"

    # If we already have a saved raw response from a previous (possibly
    # interrupted) run, reuse it instead of paying for the API call again.
    if reuse_saved_response and raw_response_path.exists():
        print(f"\n[{index}/{total}] Reusing saved response: {image_path.name} (no API call)")
        response_text = raw_response_path.read_text(encoding="utf-8")
        row = extract_json(response_text)

        required_keys = ["Segments", "Image.name"]
        missing_keys  = [k for k in required_keys if k not in row]
        extra_keys    = [k for k in row if k not in required_keys]
        if missing_keys or extra_keys:
            raise ValueError(
                f"JSON schema mismatch in saved response — "
                f"missing={missing_keys} extra={extra_keys}"
            )

        row["Specimen.image"] = get_original_image_name(image_path.name)
        row["Image.name"]     = image_path.name
        return row

    print(f"\n[{index}/{total}] Processing: {image_path.name}")

    image_base64  = preprocess_image(image_path)
    image_data_url = f"data:image/png;base64,{image_base64}"

    response = client.responses.create(
        model="gpt-5.1",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Perform diplomatic archival OCR "
                            "transcription of this specimen "
                            "label image. "
                            "Return only the JSON object."
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                        "detail": "high"
                    }
                ]
            }
        ]
    )

    # TOKEN TRACKING
    if response.usage:
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        tokens_used   = response.usage.total_tokens
        cached_tokens = 0

        if hasattr(response.usage, "input_tokens_details"):
            details = response.usage.input_tokens_details
            if hasattr(details, "cached_tokens"):
                cached_tokens = details.cached_tokens

        total_input_tokens  += input_tokens
        total_cached_tokens += cached_tokens
        total_output_tokens += output_tokens
        total_tokens        += tokens_used

        print(
            f"  [{index}/{total}] input={input_tokens} | "
            f"output={output_tokens} | "
            f"cached={cached_tokens} | "
            f"total={tokens_used}"
        )

    # SAVE RAW RESPONSE
    with open(raw_response_path, "w", encoding="utf-8") as f:
        f.write(response.output_text)

    # EXTRACT JSON
    row = extract_json(response.output_text)

    # VALIDATE JSON KEYS
    required_keys = ["Segments", "Image.name"]
    missing_keys  = [k for k in required_keys if k not in row]
    extra_keys    = [k for k in row if k not in required_keys]

    if missing_keys or extra_keys:
        raise ValueError(
            f"JSON schema mismatch — "
            f"missing={missing_keys} "
            f"extra={extra_keys}"
        )

    # ADD CSV FIELDS
    row["Specimen.image"] = get_original_image_name(image_path.name)
    row["Image.name"]     = image_path.name

    print(f"  [{index}/{total}] Done: {image_path.name}")
    return row

# =========================================================
# RUN OCR
# =========================================================

failed = []
total_images = len(image_paths)
interrupted = False
images_done_this_session = 0

try:

    for i, image_path in enumerate(image_paths, start=1):

        elapsed_so_far = time.time() - start_time
        if images_done_this_session > 0:
            avg_per_image = elapsed_so_far / images_done_this_session
            remaining = total_images - images_done_this_session
            eta_seconds = avg_per_image * remaining
            eta_min, eta_sec = divmod(eta_seconds, 60)
            print(
                f"\n--- [{i}/{total_images}] elapsed "
                f"{int(elapsed_so_far // 60)}m{int(elapsed_so_far % 60)}s "
                f"| ETA {int(eta_min)}m{int(eta_sec)}s ---"
            )

        try:
            row = process_image(image_path, i, total_images)

            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow(row)

        except Exception as e:
            print(f"\n[{i}/{total_images}] FAILED: {image_path.name}")
            print(e)
            failed.append(image_path.name)

        images_done_this_session += 1

        # Persist token usage after every image, success or failure,
        # so a force-stop never loses the running token tally.
        save_token_stats(
            images_done=len(skipped_already_done) + images_done_this_session,
            images_total=len(skipped_already_done) + total_images,
            elapsed_seconds=time.time() - start_time
        )

        if i % 25 == 0:
            print(f"--- checkpoint: {i}/{total_images} images processed, "
                  f"{len(failed)} failed so far ---")

except KeyboardInterrupt:
    interrupted = True
    print("\n\n*** RUN INTERRUPTED BY USER (Ctrl+C) ***")
    print(f"Completed {images_done_this_session}/{total_images} images this session "
          f"before stopping.")
    print(f"CSV is up to date through the last finished image:\n  {CSV_PATH}")
    save_token_stats(
        images_done=len(skipped_already_done) + images_done_this_session,
        images_total=len(skipped_already_done) + total_images,
        elapsed_seconds=time.time() - start_time
    )
    print(f"Token usage snapshot saved to:\n  {TOKEN_STATS_JSON}")

# =========================================================
# SUMMARY
# =========================================================

total_elapsed = time.time() - start_time
minutes, seconds = divmod(total_elapsed, 60)

print("\n==============================")
print("INTERRUPTED — PARTIAL RUN" if interrupted else "FINISHED")
print("==============================")
print(
    f"Processed this session: {images_done_this_session - len(failed)} labels"
)
print(f"Failed this session: {len(failed)}")
print(
    f"Total done overall: "
    f"{len(skipped_already_done) + images_done_this_session}/"
    f"{len(skipped_already_done) + total_images} labels "
    f"from {len(selected_specimens)} specimens"
)

if failed:
    for name in failed:
        print(f"  - {name}")

print(f"Time: {int(minutes)}m {seconds:.1f}s")

print("\n==============================")
print("TOKEN USAGE")
print("==============================")
print(f"Input:   {total_input_tokens:,}")
print(f"Cached:  {total_cached_tokens:,}")
print(f"Output:  {total_output_tokens:,}")
print(f"Total:   {total_tokens:,}")
print(f"\n(This same data is saved after every image to:\n {TOKEN_STATS_JSON})")
