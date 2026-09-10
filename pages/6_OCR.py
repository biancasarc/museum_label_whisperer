"""
Page 6 — OCR & Data Structuring

Two-stage pipeline on one page:

  Part 1 — OCR
    Sends each cropped label image to a vision model and saves one row per
    label crop (Specimen.image, Image.name, Segments) to a CSV.

  Part 2 — Structuring
    Groups label crops by specimen, builds a combined transcription, then
    runs five specialist prompts to extract Darwin Core metadata fields into
    a final structured CSV.

Run them in order, or re-run either stage independently (both support
resume — already-processed images / specimens are skipped automatically).
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

# Part 1 — OCR
DEFAULT_IMAGE_DIR  = PROJECT_ROOT / "data" / "cropping_result"
DEFAULT_OCR_DIR    = PROJECT_ROOT / "data" / "ocr_results"
OCR_PROMPT_FILE    = PROJECT_ROOT / "data" / "ocr_prompt.txt"

# Part 2 — Structuring
DEFAULT_STRUCT_DIR = PROJECT_ROOT / "data" / "structured"
PROMPT_DIR         = PROJECT_ROOT / "separate-prompts"

# Mapping of internal key → prompt file on disk
STRUCT_PROMPT_FILES = {
    "taxonomy":         PROMPT_DIR / "species-prompt.txt",
    "collection_event": PROMPT_DIR / "collection-event-prompt.txt",
    "geography":        PROMPT_DIR / "geography-prompt.txt",
    "identifiers":      PROMPT_DIR / "collection-identifiers-prompt.txt",
    "remarks":          PROMPT_DIR / "remarks-etc-prompt.txt",
}

# Darwin Core fields written to the final structured CSV
OUTPUT_FIELDS = [
    "Specimen.image",
    "scientificName",
    "eventDate",
    "recordedBy",
    "basisOfRecord",
    "country",
    "countryCode",
    "stateProvince",
    "county",
    "locality",
    "institutionCode",
    "collectionCode",
    "catalogNumber",
    "occurrenceID",
    "accessionNumber",
    "occurrenceRemarks",
    "Raw.transcription",
]

# Default OCR prompt shown when no saved file exists yet
DEFAULT_OCR_PROMPT = """\
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
# SHARED HELPERS
# =========================================================

def load_text_file(path: Path, default: str = "") -> str:
    """Return file contents if the file exists, otherwise return `default`."""
    return path.read_text(encoding="utf-8") if path.exists() else default


def save_text_file(path: Path, text: str) -> None:
    """Write text to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_json(text: str) -> dict:
    """
    Extract the first complete JSON object from a model response string.
    Strips markdown code fences automatically if the model wrapped its JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$",              "", text)
        text = text.strip()

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")

    return json.loads(text[start:end])


def get_specimen_id(filename: str) -> str:
    """
    Recover the bare specimen ID from a cropped-label filename.
    Handles both naming conventions produced by the YOLO pipeline:
      MGCL1202187_d_cropped_label_03.png  →  MGCL1202187
      MGCL1202187_label_03.png            →  MGCL1202187
    """
    for pattern in (r"(.+?)_d_cropped_label_\d+", r"(.+?)_label_\d+"):
        m = re.match(pattern, Path(filename).stem)
        if m:
            return m.group(1)
    return Path(filename).stem


# =========================================================
# PART 1 HELPERS
# =========================================================

def preprocess_image(image_path: Path, upscale_factor: float) -> str:
    """
    Open a label-crop image, upscale it, apply autocontrast to sharpen
    faded text, and return the result as a base64-encoded PNG string.
    """
    img = Image.open(image_path).convert("RGB")
    new_size = (int(img.width * upscale_factor), int(img.height * upscale_factor))
    img = img.resize(new_size, Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=0.5)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =========================================================
# PART 2 HELPERS
# =========================================================

def segments_to_text(segments_json: str) -> str:
    """
    Convert the Segments JSON string from the OCR CSV into a plain text
    transcription by joining all segment_text values with newlines.
    This is the input fed to each structuring prompt.
    """
    try:
        segments = json.loads(segments_json)
        return "\n".join(
            s.get("segment_text", "")
            for s in segments
            if s.get("segment_text")
        )
    except (json.JSONDecodeError, TypeError):
        # Fallback: if Segments is somehow already plain text, return as-is
        return str(segments_json)


def aggregate_by_specimen(ocr_csv: Path) -> dict[str, str]:
    """
    Read the OCR results CSV (one row per label crop) and group by specimen.
    Multiple crops from the same specimen are joined with a blank line.
    Returns {Specimen.image: combined_transcription_text}.
    """
    buckets: dict[str, list[str]] = {}

    with open(ocr_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            specimen = row.get("Specimen.image", "").strip()
            segments = row.get("Segments",       "").strip()
            if not specimen or not segments:
                continue
            buckets.setdefault(specimen, []).append(segments_to_text(segments))

    return {
        specimen: "\n\n".join(parts)
        for specimen, parts in buckets.items()
    }


def call_structuring_prompt(
    client:        OpenAI,
    model:         str,
    prompt_key:    str,
    system_prompt: str,
    transcription: str,
    token_totals:  dict,
) -> str:
    """
    Call the API with one specialist structuring prompt and the specimen's
    plain-text transcription.  Accumulates token usage into `token_totals`.
    Returns the raw response text.
    """
    response = client.responses.create(
        model=model,
        instructions=system_prompt,   # specialist system prompt
        input=[{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    "Extract structured metadata from this museum specimen "
                    "OCR transcription.\n\n"
                    f"{transcription}"
                ),
            }],
        }],
    )

    if response.usage:
        token_totals[prompt_key]["input"]  += response.usage.input_tokens
        token_totals[prompt_key]["output"] += response.usage.output_tokens
        token_totals[prompt_key]["total"]  += response.usage.total_tokens

    return response.output_text


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Step 6 — OCR & Data Structuring")
st.markdown(
    "**Part 1** — OCR: transcribe each cropped label image into raw text.  \n"
    "**Part 2** — Structuring: extract Darwin Core metadata fields using five "
    "specialist prompts.  \n"
    "Run them in order, or re-run either independently."
)

# =========================================================
# SHARED CONFIGURATION
# =========================================================

with st.container(border=True):
    st.subheader("Shared configuration")
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.text_input(
            "Model",
            value="gpt-5.1",
            help="Used for both OCR and structuring (e.g. gpt-4o, gpt-5.1).",
        )
    with col2:
        api_key = st.text_input(
            "OpenAI API key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            help="Used only in this session — never saved to disk.",
        )
        if api_key:
            st.session_state["openai_api_key"] = api_key

# =========================================================
# PART 1 — OCR
# =========================================================

st.divider()
st.header("Part 1 — OCR")
st.write("Send each cropped label image to the vision model and save the raw transcription.")

with st.container(border=True):
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        image_dir_str = st.text_input(
            "Input folder (cropped labels)",
            value=str(DEFAULT_IMAGE_DIR),
            help="Folder produced by Step 5 — Crop.",
        )
    with col2:
        ocr_out_str = st.text_input(
            "OCR output folder",
            value=str(DEFAULT_OCR_DIR),
        )

    col3, col4 = st.columns(2)
    with col3:
        upscale_factor = st.number_input(
            "Upscale factor",
            min_value=1.0, max_value=5.0, value=1.9, step=0.1,
            help="Images are upscaled before sending to improve OCR accuracy.",
        )
    with col4:
        num_specimens = st.number_input(
            "Max specimens (0 = all)", min_value=0, value=0, step=1,
            key="ocr_max_specimens",
        )

with st.container(border=True):
    st.subheader("OCR prompt")
    st.caption(
        "Sent as the system instruction for every image. "
        "Save to keep changes between sessions."
    )

    ocr_prompt = st.text_area(
        "ocr_prompt",
        value=load_text_file(OCR_PROMPT_FILE, DEFAULT_OCR_PROMPT),
        height=280,
        label_visibility="collapsed",
        key="ocr_prompt_area",
    )

    c1, c2, _ = st.columns([1.2, 1.4, 5])
    with c1:
        if st.button("💾 Save", key="save_ocr_prompt", use_container_width=True):
            save_text_file(OCR_PROMPT_FILE, ocr_prompt)
            st.success(f"Saved → `{OCR_PROMPT_FILE.relative_to(PROJECT_ROOT)}`")
    with c2:
        if st.button("↩️ Reset to default", key="reset_ocr_prompt", use_container_width=True):
            save_text_file(OCR_PROMPT_FILE, DEFAULT_OCR_PROMPT)
            # Clear the widget's session-state value so the text area reloads from file
            st.session_state.pop("ocr_prompt_area", None)
            st.rerun()

run_ocr = st.button(
    "▶ Run OCR", type="primary", key="run_ocr_btn",
    icon=":material/image:",
)

if run_ocr:
    if not api_key:
        st.error("Please enter your OpenAI API key in the shared configuration above.")
        st.stop()

    image_dir = Path(image_dir_str).expanduser()
    if not image_dir.is_dir():
        st.error(f"Input folder not found: `{image_dir}`")
        st.stop()

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if not image_paths:
        st.error("No images found in the input folder.")
        st.stop()

    # Apply per-specimen limit
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

    ocr_out = Path(ocr_out_str).expanduser()
    ocr_out.mkdir(parents=True, exist_ok=True)
    raw_dir  = ocr_out / "raw_responses"   # one .txt per image with the raw model output
    raw_dir.mkdir(exist_ok=True)
    csv_path   = ocr_out / "ocr_results.csv"
    csv_fields = ["Specimen.image", "Image.name", "Segments"]

    # Resume: skip images already in the output CSV
    already_done: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Image.name"):
                    already_done.add(row["Image.name"])

    remaining = [p for p in image_paths if p.name not in already_done]
    if already_done:
        st.info(f"Resuming: {len(already_done)} image(s) already done, {len(remaining)} remaining.")
    if not remaining:
        st.success("All images already processed — nothing to do.")
        st.stop()

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    client = OpenAI(api_key=api_key)
    total_in = total_out = total_all = 0
    progress   = st.progress(0)
    status     = st.empty()
    tok_info   = st.empty()
    ocr_failed: list[str] = []
    t0 = time.time()

    for i, image_path in enumerate(remaining, start=1):
        status.write(f"OCR {i}/{len(remaining)}: `{image_path.name}`")
        raw_path = raw_dir / f"{image_path.stem}.txt"

        try:
            # Reuse a saved raw response to avoid a duplicate API call on resume
            if raw_path.exists():
                response_text = raw_path.read_text(encoding="utf-8")
            else:
                b64      = preprocess_image(image_path, upscale_factor)
                data_url = f"data:image/png;base64,{b64}"

                response = client.responses.create(
                    model=model_name,
                    instructions=ocr_prompt,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text",
                             "text": ("Perform diplomatic archival OCR transcription "
                                      "of this specimen label image. "
                                      "Return only the JSON object.")},
                            {"type": "input_image",
                             "image_url": data_url,
                             "detail": "high"},
                        ],
                    }],
                )

                if response.usage:
                    total_in  += response.usage.input_tokens
                    total_out += response.usage.output_tokens
                    total_all += response.usage.total_tokens

                response_text = response.output_text
                # Save immediately so a crash doesn't lose this call
                raw_path.write_text(response_text, encoding="utf-8")

            row = extract_json(response_text)
            row["Specimen.image"] = f"{get_specimen_id(image_path.name)}.png"
            row["Image.name"]     = image_path.name

            # Serialize Segments as a JSON string — prevents ASCII codec errors
            # when label text contains non-ASCII characters (e.g. em dashes).
            if "Segments" in row:
                row["Segments"] = json.dumps(row["Segments"], ensure_ascii=False)

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        except Exception as e:
            ocr_failed.append(image_path.name)
            st.warning(f"OCR failed: `{image_path.name}` — {e}")

        progress.progress(i / len(remaining))
        tok_info.caption(
            f"Tokens — input: {total_in:,} | output: {total_out:,} | "
            f"total: {total_all:,}  ·  elapsed: {time.time()-t0:.0f}s"
        )

    status.empty()
    n_ok = len(remaining) - len(ocr_failed)
    if ocr_failed:
        st.warning(f"OCR finished with {len(ocr_failed)} failure(s). {n_ok} image(s) succeeded.")
    else:
        st.success(
            f"OCR done! {n_ok} image(s) processed → "
            f"`{csv_path.relative_to(PROJECT_ROOT)}`"
        )

# =========================================================
# PART 2 — DATA STRUCTURING
# =========================================================

st.divider()
st.header("Part 2 — Data Structuring")
st.write(
    "Reads the OCR CSV, groups label crops by specimen, then runs five specialist "
    "prompts to extract Darwin Core metadata fields."
)

with st.container(border=True):
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        struct_in_str = st.text_input(
            "Input CSV (OCR results)",
            value=str(DEFAULT_OCR_DIR / "ocr_results.csv"),
            help="The CSV produced by Part 1.",
        )
    with col2:
        struct_out_str = st.text_input(
            "Structuring output folder",
            value=str(DEFAULT_STRUCT_DIR),
        )

    max_specimens = st.number_input(
        "Max specimens (0 = all)", min_value=0, value=0, step=1,
        key="struct_max_specimens",
    )

# --- Five editable structuring prompts ---
with st.container(border=True):
    st.subheader("Structuring prompts")
    st.caption(
        "One specialist prompt per metadata category. Each is called separately "
        "with the specimen's combined transcription text as the user message. "
        "The `{{OCR_TEXT}}` placeholder at the end of each file is for reference — "
        "the transcription is injected automatically, not substituted into the prompt."
    )

    # Human-readable label for each prompt category
    PROMPT_LABELS = {
        "taxonomy":         "Taxonomy — scientificName",
        "collection_event": "Collection event — date, collector, basis of record",
        "geography":        "Geography — country → locality",
        "identifiers":      "Identifiers — institution, collection, catalog, occurrence",
        "remarks":          "Remarks — provenance and miscellaneous notes",
    }

    # Collect the (possibly edited) prompt texts from the UI
    edited_prompts: dict[str, str] = {}

    for key, label in PROMPT_LABELS.items():
        filepath = STRUCT_PROMPT_FILES[key]
        with st.expander(label):
            text = st.text_area(
                label,
                value=load_text_file(filepath),
                height=280,
                label_visibility="collapsed",
                key=f"struct_prompt_{key}",
            )
            edited_prompts[key] = text

            c1, c2, _ = st.columns([1, 1.5, 5])
            with c1:
                if st.button("💾 Save", key=f"save_struct_{key}", use_container_width=True):
                    save_text_file(filepath, text)
                    st.success(f"Saved → `{filepath.relative_to(PROJECT_ROOT)}`")
            with c2:
                if st.button("↩️ Reload from file", key=f"reset_struct_{key}", use_container_width=True):
                    # Clear the widget's session-state so the text area re-reads the file
                    st.session_state.pop(f"struct_prompt_{key}", None)
                    st.rerun()

run_struct = st.button(
    "▶ Run Structuring", type="primary", key="run_struct_btn",
    icon=":material/table:",
)

if run_struct:
    if not api_key:
        st.error("Please enter your OpenAI API key in the shared configuration above.")
        st.stop()

    struct_in = Path(struct_in_str).expanduser()
    if not struct_in.exists():
        st.error(f"Input CSV not found: `{struct_in}`. Run Part 1 first.")
        st.stop()

    # Group OCR results by specimen and build plain-text transcriptions
    with st.spinner("Grouping label crops by specimen…"):
        specimen_transcriptions = aggregate_by_specimen(struct_in)

    if not specimen_transcriptions:
        st.error("No specimen data found in the OCR results CSV.")
        st.stop()

    specimens = list(specimen_transcriptions.items())  # [(specimen_image, transcription)]

    if max_specimens > 0:
        specimens = specimens[:max_specimens]
        st.info(f"Limited to first {max_specimens} specimen(s).")

    # Set up output directories (one sub-folder per prompt category for raw responses)
    struct_out = Path(struct_out_str).expanduser()
    struct_out.mkdir(parents=True, exist_ok=True)
    raw_struct_dir = struct_out / "raw_responses"
    for key in PROMPT_LABELS:
        (raw_struct_dir / key).mkdir(parents=True, exist_ok=True)

    struct_csv = struct_out / "structured_dwc_metadata.csv"

    # Resume: skip specimens already in the output CSV
    already_structured: set[str] = set()
    if struct_csv.exists():
        with open(struct_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Specimen.image"):
                    already_structured.add(row["Specimen.image"])

    remaining_specimens = [
        (s, t) for s, t in specimens if s not in already_structured
    ]

    if already_structured:
        st.info(
            f"Resuming: {len(already_structured)} specimen(s) already done, "
            f"{len(remaining_specimens)} remaining."
        )
    if not remaining_specimens:
        st.success("All specimens already structured — nothing to do.")
        st.stop()

    if not struct_csv.exists():
        with open(struct_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=OUTPUT_FIELDS).writeheader()

    client = OpenAI(api_key=api_key)

    # Token counts per prompt category
    token_totals = {key: {"input": 0, "output": 0, "total": 0} for key in PROMPT_LABELS}

    progress2      = st.progress(0)
    status2        = st.empty()
    tok_info2      = st.empty()
    struct_failed: list[str] = []
    t0 = time.time()

    for i, (specimen_image, transcription) in enumerate(remaining_specimens, start=1):
        status2.write(f"Structuring {i}/{len(remaining_specimens)}: `{specimen_image}`")

        combined: dict[str, str] = {
            "Specimen.image":    specimen_image,
            "Raw.transcription": transcription,
        }

        try:
            for key in PROMPT_LABELS:
                system_prompt  = edited_prompts[key]
                raw_resp_path  = raw_struct_dir / key / f"{Path(specimen_image).stem}.txt"

                # Reuse a cached raw response if available
                if raw_resp_path.exists():
                    raw_text = raw_resp_path.read_text(encoding="utf-8")
                else:
                    raw_text = call_structuring_prompt(
                        client, model_name, key,
                        system_prompt, transcription, token_totals,
                    )
                    raw_resp_path.write_text(raw_text, encoding="utf-8")

                # Merge only the known output fields from this prompt's JSON response
                parsed = extract_json(raw_text)
                for field in OUTPUT_FIELDS:
                    if field in parsed and parsed[field] is not None:
                        combined[field] = str(parsed[field])

            # Fill any fields not covered by any prompt with an empty string
            for field in OUTPUT_FIELDS:
                combined.setdefault(field, "")

            # Append the row — safe to interrupt, completed rows are already saved
            with open(struct_csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=OUTPUT_FIELDS).writerow(combined)

        except Exception as e:
            struct_failed.append(specimen_image)
            st.warning(f"Structuring failed: `{specimen_image}` — {e}")

        progress2.progress(i / len(remaining_specimens))
        grand_in    = sum(v["input"]  for v in token_totals.values())
        grand_out   = sum(v["output"] for v in token_totals.values())
        grand_total = sum(v["total"]  for v in token_totals.values())
        tok_info2.caption(
            f"Tokens — input: {grand_in:,} | output: {grand_out:,} | "
            f"total: {grand_total:,}  ·  elapsed: {time.time()-t0:.0f}s"
        )

    status2.empty()
    n_ok2 = len(remaining_specimens) - len(struct_failed)

    if struct_failed:
        st.warning(
            f"Structuring finished with {len(struct_failed)} failure(s). "
            f"{n_ok2} specimen(s) succeeded."
        )
    else:
        st.success(
            f"Structuring done! {n_ok2} specimen(s) → "
            f"`{struct_csv.relative_to(PROJECT_ROOT)}`"
        )
