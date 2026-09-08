"""
Page 6 — OCR

Sends cropped label images to an OpenAI vision model and saves the
transcribed text as a CSV.  The system prompt is fully editable in the UI
and persisted to data/ocr_prompt.txt between sessions.
"""

from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
from openai import OpenAI

import base64
import csv
import json
import re
import time

import streamlit as st

# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT       = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR  = PROJECT_ROOT / "data" / "cropping_result"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ocr_results"
PROMPT_FILE        = PROJECT_ROOT / "data" / "ocr_prompt.txt"

# =========================================================
# DEFAULT PROMPT
# (shown when no saved prompt file exists yet)
# =========================================================

DEFAULT_PROMPT = """\
You are an expert in diplomatic archival transcription of natural history specimen labels.

Your task is to transcribe the text on the label exactly as written, preserving original
spelling, capitalisation and punctuation. Do not correct errors or add information that is
not visible on the label.

Return ONLY a valid JSON object with the following structure:
{
  "Image.name": "<filename of the image>",
  "Segments": [
    {
      "segment_text": "<exact text of one line or block on the label>",
      "segment_type": "<one of: taxon, locality, collector, date, collection_number, other>"
    }
  ]
}
"""

# =========================================================
# HELPERS
# =========================================================

def load_prompt() -> str:
    """Return the saved prompt, or the hard-coded default if no file exists yet."""
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8")
    return DEFAULT_PROMPT


def save_prompt(text: str) -> None:
    """Write the current prompt text to disk so it persists across sessions."""
    PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(text, encoding="utf-8")


def extract_json(text: str) -> dict:
    """
    Pull the first complete JSON object out of a model response string.
    Handles markdown code fences (```json ... ```) automatically.
    """
    text = text.strip()

    # Strip markdown code fences if the model wrapped its JSON in one
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    # Use raw_decode so trailing text after the object doesn't cause an error
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


def preprocess_image(image_path: Path, upscale_factor: float) -> str:
    """
    Open an image, upscale it and sharpen contrast, then return the result
    as a base64-encoded PNG string ready for the API.
    """
    img = Image.open(image_path).convert("RGB")
    new_size = (int(img.width * upscale_factor), int(img.height * upscale_factor))
    img = img.resize(new_size, Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=0.5)  # boost contrast without clipping much

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_specimen_id(filename: str) -> str:
    """
    Strip the crop-number suffix from a label filename to recover the
    original specimen ID.  Expects names like: specimen123_label_01.png
    """
    match = re.match(r"(.+?)_label_\d+", Path(filename).stem)
    return match.group(1) if match else Path(filename).stem


# =========================================================
# PAGE LAYOUT
# =========================================================

st.title("Step 6 — OCR")
st.write(
    "Send cropped label images to a vision model and save the transcribed "
    "text as a CSV file."
)

# ------------------------------------------------------------------
# SECTION 1 · Configuration
# ------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Configuration")

    col1, col2 = st.columns(2)
    with col1:
        input_dir_str = st.text_input(
            "Input folder (cropped labels)",
            value=str(DEFAULT_INPUT_DIR),
            help="The folder produced by Step 5 — Crop.",
        )
    with col2:
        output_dir_str = st.text_input(
            "Output folder",
            value=str(DEFAULT_OUTPUT_DIR),
            help="Where the CSV and raw model responses will be written.",
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        model_name = st.text_input(
            "Model",
            value="gpt-4o",
            help="Any OpenAI vision model (e.g. gpt-4o, gpt-4o-mini, gpt-5.1).",
        )
    with col4:
        upscale_factor = st.number_input(
            "Upscale factor",
            min_value=1.0,
            max_value=5.0,
            value=1.9,
            step=0.1,
            help="Images are upscaled before being sent to improve OCR accuracy.",
        )
    with col5:
        num_specimens = st.number_input(
            "Max specimens (0 = all)",
            min_value=0,
            value=0,
            step=1,
        )

    # Store API key in session state so it survives page navigation
    api_key = st.text_input(
        "OpenAI API key",
        value=st.session_state.get("openai_api_key", ""),
        type="password",
        help="Used only in this session — never saved to disk.",
    )
    if api_key:
        st.session_state["openai_api_key"] = api_key

# ------------------------------------------------------------------
# SECTION 2 · Editable system prompt
# ------------------------------------------------------------------
with st.container(border=True):
    st.subheader("OCR prompt")
    st.caption(
        "This system prompt is sent to the model for every image. "
        "Edit it to change which fields are extracted or how the JSON output is structured. "
        "Changes are applied immediately when you run OCR — save to keep them for next time."
    )

    prompt_text = st.text_area(
        "System prompt",
        value=load_prompt(),
        height=320,
        label_visibility="collapsed",
    )

    col_save, col_reset, _ = st.columns([1.2, 1.4, 5])
    with col_save:
        if st.button("💾 Save prompt", use_container_width=True):
            save_prompt(prompt_text)
            st.success(f"Saved → `{PROMPT_FILE.relative_to(PROJECT_ROOT)}`")
    with col_reset:
        if st.button("↩️ Reset to default", use_container_width=True):
            save_prompt(DEFAULT_PROMPT)
            st.rerun()

# ------------------------------------------------------------------
# SECTION 3 · Run
# ------------------------------------------------------------------
st.divider()
run_ocr = st.button("▶ Run OCR", type="primary", icon=":material/text_snippet:")

if run_ocr:

    # --- Validate inputs ---
    if not api_key:
        st.error("Please enter your OpenAI API key above.")
        st.stop()

    input_dir = Path(input_dir_str).expanduser()
    if not input_dir.is_dir():
        st.error(f"Input folder not found: `{input_dir}`")
        st.stop()

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    if not image_paths:
        st.error("No images found in the input folder.")
        st.stop()

    # --- Apply specimen limit ---
    if num_specimens > 0:
        seen: set[str] = set()
        filtered = []
        for p in image_paths:
            sid = get_specimen_id(p.name)
            if sid not in seen:
                if len(seen) >= num_specimens:
                    continue
                seen.add(sid)
            filtered.append(p)
        image_paths = filtered
        st.info(f"Limited to {len(seen)} specimens → {len(image_paths)} label image(s).")

    # --- Set up output directories ---
    output_dir = Path(output_dir_str).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_responses"   # one .txt per image with the raw model output
    raw_dir.mkdir(exist_ok=True)

    csv_path   = output_dir / "ocr_results.csv"
    csv_fields = ["Specimen.image", "Image.name", "Segments"]

    # --- Resume: skip images already written to the CSV ---
    already_done: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Image.name"):
                    already_done.add(row["Image.name"])

    remaining = [p for p in image_paths if p.name not in already_done]

    if already_done:
        st.info(
            f"Resuming: {len(already_done)} image(s) already done, "
            f"{len(remaining)} remaining."
        )

    if not remaining:
        st.success("All images have already been processed — nothing to do!")
        st.stop()

    # Write CSV header if this is a fresh run
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # --- OCR loop ---
    client = OpenAI(api_key=api_key)

    # Running token totals
    total_input = total_output = total_all = 0

    progress_bar  = st.progress(0)
    status_text   = st.empty()
    token_caption = st.empty()
    failed        = []
    start_time    = time.time()

    for i, image_path in enumerate(remaining, start=1):
        status_text.write(f"Processing {i} / {len(remaining)}: `{image_path.name}`")
        raw_path = raw_dir / f"{image_path.stem}.txt"

        try:
            # Reuse a saved raw response if we crashed mid-run last time —
            # avoids paying for the same API call twice.
            if raw_path.exists():
                response_text = raw_path.read_text(encoding="utf-8")

            else:
                image_b64 = preprocess_image(image_path, upscale_factor)
                data_url  = f"data:image/png;base64,{image_b64}"

                response = client.responses.create(
                    model=model_name,
                    instructions=prompt_text,          # <-- editable prompt
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "Perform diplomatic archival OCR transcription "
                                        "of this specimen label image. "
                                        "Return only the JSON object."
                                    ),
                                },
                                {
                                    "type": "input_image",
                                    "image_url": data_url,
                                    "detail": "high",
                                },
                            ],
                        }
                    ],
                )

                # Accumulate token usage so the user can track cost
                if response.usage:
                    total_input  += response.usage.input_tokens
                    total_output += response.usage.output_tokens
                    total_all    += response.usage.total_tokens

                response_text = response.output_text
                # Save the raw response immediately — if we crash, it can be reused
                raw_path.write_text(response_text, encoding="utf-8")

            # Parse the JSON the model returned
            row = extract_json(response_text)
            row["Specimen.image"] = f"{get_specimen_id(image_path.name)}.png"
            row["Image.name"]     = image_path.name

            # Append one row to the CSV (safe to interrupt — completed rows are kept)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        except Exception as e:
            failed.append(image_path.name)
            st.warning(f"Failed: `{image_path.name}` — {e}")

        # Update progress indicators
        progress_bar.progress(i / len(remaining))
        elapsed = time.time() - start_time
        token_caption.caption(
            f"Tokens — input: {total_input:,} | output: {total_output:,} | "
            f"total: {total_all:,}  ·  elapsed: {elapsed:.0f}s"
        )

    # --- Final status ---
    status_text.empty()
    n_done = len(remaining) - len(failed)

    if failed:
        st.warning(
            f"Finished with {len(failed)} failure(s). "
            f"Successfully processed {n_done} image(s)."
        )
    else:
        st.success(
            f"Done! Processed {n_done} image(s). "
            f"Results saved to `{csv_path.relative_to(PROJECT_ROOT)}`."
        )

    st.caption(f"Raw model responses saved to: `{raw_dir.relative_to(PROJECT_ROOT)}`")
