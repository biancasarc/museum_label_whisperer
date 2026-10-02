"""
Page 9 — Second-pass OCR

Re-sends specific cropped label images to the vision model for columns
where the first OCR pass may have been imprecise.

Filtering logic:
  - Skips rows already manually checked in Step 7.
  - Skips rows where the selected column is empty (nothing to re-read).
  - Uses ocr_results.csv to find *which* label(s) hold the relevant
    information, so only those targeted crops are sent — not all labels
    for the specimen.

Reuse pattern: raw API responses are saved to disk immediately after each
call, so a re-run parses from disk instead of re-paying.
"""

import base64
import csv
import json
import re
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageOps

# =========================================================
# PAGE SETUP
# =========================================================

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: **{current_proj}**")

st.title("Step 9 — Second-pass OCR")
st.markdown(
    "Re-send specific label images for columns where the first read may be wrong. "
    "Only rows with a value that haven't been manually checked are re-processed, "
    "and only the label crops that actually contain the relevant information are sent."
)

# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj

CORRECTED_CSV    = PROJECT_ROOT / "data" / "07_quality_checking" / "corrected_metadata.csv"
RAW_OCR_CSV      = PROJECT_ROOT / "data" / "06_structured_output" / "structured_dwc_metadata.csv"
OCR_RESULTS_CSV  = PROJECT_ROOT / "data" / "05_ocr_results" / "ocr_results.csv"
CROPS_DIR        = PROJECT_ROOT / "data" / "04_cropping_result"

SECOND_PARSE_DIR = PROJECT_ROOT / "data" / "08_second_parse"
RAW_RESPONSE_DIR = SECOND_PARSE_DIR / "raw_responses"
OUTPUT_CSV       = SECOND_PARSE_DIR / "second_parse_results.csv"
PROMPTS_DIR      = SECOND_PARSE_DIR / "prompts"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# =========================================================
# SEGMENT TYPE MAPPING
#
# Maps Darwin Core column names to OCR segment types.
# Labels whose Segments list contains any of these types are
# prioritised; if none match, all labels for the specimen are used.
# =========================================================

COLUMN_TO_SEGMENT_TYPES: dict[str, list[str]] = {
    "scientificName":   ["taxon"],
    "eventDate":        ["date"],
    "recordedBy":       ["collector"],
    "locality":         ["locality"],
    "stateProvince":    ["locality"],
    "county":           ["locality"],
    "country":          ["locality"],
    "countryCode":      ["locality"],
    "catalogNumber":    ["collection_number"],
    "accessionNumber":  ["collection_number"],
    "occurrenceID":     ["collection_number"],
    "collectionCode":   ["collection_number"],
    "institutionCode":  ["collection_number"],
    "occurrenceRemarks":["other"],
    "basisOfRecord":    ["other"],
}

# =========================================================
# GUARD — need the corrected (or raw) CSV to exist
# =========================================================

input_csv = CORRECTED_CSV if CORRECTED_CSV.exists() else (
    RAW_OCR_CSV if RAW_OCR_CSV.exists() else None
)

if input_csv is None:
    st.warning("No OCR results found. Run Step 6 first, then optionally Step 7.")
    st.stop()

# =========================================================
# HELPERS
# =========================================================

def preprocess_image(image_path: Path, upscale_factor: float) -> str:
    """Upscale, autocontrast, and base64-encode a crop image as PNG."""
    img = Image.open(image_path).convert("RGB")
    new_size = (int(img.width * upscale_factor), int(img.height * upscale_factor))
    img = img.resize(new_size, Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=0.5)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_json(text: str) -> dict:
    """Extract the first complete JSON object from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text, start)
    return obj


def default_prompt_for(column: str) -> str:
    return (
        f"You are an expert in natural history museum specimens.\n\n"
        f"Look carefully at the label image(s) provided and extract the value "
        f"for the field **{column}**.\n\n"
        f"Return ONLY a valid JSON object in this exact format:\n"
        f'{{\n  "{column}": "<extracted value, or null if not present or unreadable>"\n}}\n\n'
        f"Do not add any explanation or extra text — only the JSON object."
    )


def load_prompt(column: str) -> str:
    path = PROMPTS_DIR / f"{column}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else default_prompt_for(column)


def save_prompt(column: str, text: str) -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROMPTS_DIR / f"{column}.txt").write_text(text, encoding="utf-8")


def labels_for_column(
    specimen_name: str,
    column: str,
    ocr_df: pd.DataFrame,
) -> list[Path]:
    """
    Return the crop image paths most relevant for re-parsing `column`.

    Strategy:
      1. Find all rows in ocr_results for this specimen.
      2. Keep rows whose Segments JSON contains at least one segment of
         the target type(s) for this column.
      3. If none match (or no mapping exists), fall back to ALL crops.
    """
    specimen_rows = ocr_df[ocr_df["Specimen.image"] == specimen_name]
    target_types  = COLUMN_TO_SEGMENT_TYPES.get(column)  # None = no mapping

    matched_image_names: list[str] = []

    for _, row in specimen_rows.iterrows():
        image_name   = row["Image.name"]
        segments_raw = row.get("Segments", "")

        # No type mapping for this column — include all labels
        if target_types is None:
            matched_image_names.append(image_name)
            continue

        try:
            segments       = json.loads(segments_raw)
            types_in_label = {s.get("segment_type", "") for s in segments}
            if types_in_label & set(target_types):
                matched_image_names.append(image_name)
        except (json.JSONDecodeError, TypeError):
            # Malformed segment — include to be safe
            matched_image_names.append(image_name)

    # Resolve filenames to actual paths in the crops directory
    found = [p for name in matched_image_names if (p := CROPS_DIR / name).exists()]

    # Fallback: all crops for this specimen
    if not found and CROPS_DIR.is_dir():
        stem  = Path(specimen_name).stem
        found = sorted(
            p for p in CROPS_DIR.iterdir()
            if p.suffix.lower() in IMAGE_EXT
            and re.match(rf"^{re.escape(stem)}_label_\d+", p.stem)
        )

    return found

# =========================================================
# LOAD SOURCE DATA
# =========================================================

df     = pd.read_csv(input_csv, dtype=str).fillna("")
ocr_df = pd.read_csv(OCR_RESULTS_CSV, dtype=str).fillna("") if OCR_RESULTS_CSV.exists() else pd.DataFrame()

source_label = (
    "corrected metadata (Step 7)"
    if input_csv == CORRECTED_CSV
    else "raw OCR output (Step 6)"
)
st.caption(f"Source: `{input_csv.relative_to(PROJECT_ROOT)}` ({source_label})")

if ocr_df.empty:
    st.warning(
        f"OCR results file not found at `{OCR_RESULTS_CSV.relative_to(PROJECT_ROOT)}`. "
        "Label-level targeting will fall back to all crops per specimen."
    )

# Columns eligible for second parse
SKIP_COLS      = {"Specimen.image", "Manually checked", "Raw.transcription"}
PARSEABLE_COLS = [c for c in df.columns if c not in SKIP_COLS]

if not PARSEABLE_COLS:
    st.warning("No parseable columns found in the metadata CSV.")
    st.stop()

# =========================================================
# CONFIGURATION
# =========================================================

with st.container(border=True):
    st.subheader("Configuration")
    col_a, col_b = st.columns(2)
    with col_a:
        model_name = st.text_input(
            "Model", value="gpt-5.1",
            help="Vision model to use (e.g. gpt-4o, gpt-5.1).",
            key="sp_model",
        )
    with col_b:
        api_key = st.text_input(
            "OpenAI API key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            help="Used only while the app is open — never saved to disk.",
            key="sp_api_key",
        )
        if api_key:
            st.session_state["openai_api_key"] = api_key

    col_c, col_d = st.columns(2)
    with col_c:
        upscale_factor = st.number_input(
            "Enlarge images by",
            min_value=1.0, max_value=5.0, value=1.9, step=0.1,
            key="sp_upscale",
        )
    with col_d:
        reuse_responses = st.checkbox(
            "Re-use saved responses (skip API calls for already-processed images)",
            value=True, key="sp_reuse",
        )

    skip_checked = st.checkbox(
        "Skip rows already manually checked in Step 7",
        value=True, key="sp_skip_checked",
        help="Uncheck to re-parse everything, including rows you already reviewed.",
    )

# =========================================================
# COLUMN SELECTION
# =========================================================

st.divider()
st.subheader("Columns to re-parse")

selected_columns = st.multiselect(
    "Choose one or more columns:",
    options=PARSEABLE_COLS,
    default=[],
    key="sp_columns",
)

# =========================================================
# PREVIEW — what will actually be processed
# =========================================================

if selected_columns:
    st.divider()
    st.subheader("What will be processed")

    preview_rows = []
    for col in selected_columns:
        mask = df[col].str.strip() != ""
        if skip_checked:
            mask &= df.get("Manually checked", pd.Series(["no"] * len(df))) != "yes"
        eligible = df[mask]
        target_types = COLUMN_TO_SEGMENT_TYPES.get(col, [])
        preview_rows.append({
            "Column": col,
            "Rows to re-parse": len(eligible),
            "Label type(s) targeted": ", ".join(target_types) if target_types else "all labels (no mapping)",
        })

    st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

    # Per-column prompt editors
    st.divider()
    st.subheader("Reading instructions (one prompt per column)")
    st.caption("Edit and save the prompt each column uses.")

    prompts: dict[str, str] = {}
    for col in selected_columns:
        with st.expander(f"Prompt — {col}", expanded=False):
            prompt_text = st.text_area(
                f"Instructions for {col}",
                value=load_prompt(col),
                height=200,
                key=f"sp_prompt_{col}",
                label_visibility="collapsed",
            )
            prompts[col] = prompt_text
            c1, c2, _ = st.columns([1.2, 1.8, 5])
            with c1:
                if st.button("💾 Save", key=f"save_prompt_{col}", use_container_width=True):
                    save_prompt(col, prompt_text)
                    st.success("Saved.")
            with c2:
                if st.button("↩️ Reset to default", key=f"reset_prompt_{col}", use_container_width=True):
                    default = default_prompt_for(col)
                    save_prompt(col, default)
                    st.session_state[f"sp_prompt_{col}"] = default
                    st.rerun()
else:
    prompts = {}

# =========================================================
# LOAD EXISTING RESULTS
# =========================================================

SECOND_PARSE_DIR.mkdir(parents=True, exist_ok=True)
RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

existing_results: dict[str, dict] = {}
if OUTPUT_CSV.exists():
    sp_df = pd.read_csv(OUTPUT_CSV, dtype=str)
    for _, row in sp_df.iterrows():
        existing_results[row["Specimen.image"]] = row.to_dict()

# =========================================================
# RUN BUTTON
# =========================================================

st.divider()

if not selected_columns:
    st.info("Select at least one column above to get started.")
    st.stop()

run_btn = st.button(
    "▶ Run second-pass OCR", type="primary", key="sp_run",
    icon=":material/image:",
)

if not run_btn:
    st.stop()

# =========================================================
# VALIDATE
# =========================================================

if not api_key:
    st.error("Enter your OpenAI API key above before running.")
    st.stop()

# =========================================================
# BUILD WORK LIST
# =========================================================

work_items: list[tuple[str, str]] = []   # (specimen_name, column)

for col in selected_columns:
    mask = df[col].str.strip() != ""
    if skip_checked:
        mask &= df.get("Manually checked", pd.Series(["no"] * len(df))) != "yes"
    eligible = df[mask]
    for specimen_name in eligible["Specimen.image"]:
        output_key = f"{col}_second_parse"
        if (
            specimen_name in existing_results
            and output_key in existing_results[specimen_name]
        ):
            continue   # already done in a previous run
        work_items.append((specimen_name, col))

if not work_items:
    st.success("Nothing to do — all eligible rows already have second-parse results.")
    st.stop()

# =========================================================
# RUN
# =========================================================

client = OpenAI(api_key=api_key)

total_input_tokens  = 0
total_output_tokens = 0
total_tokens        = 0
api_calls           = 0
reused_from_disk    = 0
failed: list[tuple[str, str]] = []

progress_bar = st.progress(0.0, text="Starting…")
status_box   = st.empty()
token_box    = st.empty()
start_time   = time.time()


def write_results() -> None:
    """Persist current results to the output CSV."""
    if not existing_results:
        return
    all_sp_cols = sorted({
        k for row in existing_results.values() for k in row if k != "Specimen.image"
    })
    fieldnames = ["Specimen.image"] + all_sp_cols
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for name in sorted(existing_results):
            writer.writerow(existing_results[name])


interrupted = False

try:
    for idx, (specimen_name, col) in enumerate(work_items):
        progress_bar.progress(idx / len(work_items), text=f"{idx}/{len(work_items)}")

        output_key = f"{col}_second_parse"
        safe_col   = re.sub(r"[^\w\-]", "_", col)
        safe_spec  = re.sub(r"[^\w\-]", "_", Path(specimen_name).stem)
        raw_path   = RAW_RESPONSE_DIR / f"{safe_spec}__{safe_col}.txt"

        if specimen_name not in existing_results:
            existing_results[specimen_name] = {"Specimen.image": specimen_name}

        try:
            raw_text = None

            if reuse_responses and raw_path.exists():
                raw_text = raw_path.read_text(encoding="utf-8")
                reused_from_disk += 1
                status_box.write(f"♻ {specimen_name} / **{col}** — re-parsed from disk")

            else:
                crops = labels_for_column(specimen_name, col, ocr_df)

                if not crops:
                    status_box.write(f"⚠ {specimen_name} / **{col}** — no crops found, skipping")
                    failed.append((f"{specimen_name} / {col}", "No crop images found"))
                    existing_results[specimen_name][output_key] = ""
                    continue

                image_blocks = [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{preprocess_image(crop, upscale_factor)}",
                        "detail": "high",
                    }
                    for crop in crops
                ]

                prompt_text = prompts.get(col, load_prompt(col))

                response = client.responses.create(
                    model=model_name,
                    instructions=prompt_text,
                    input=[{"role": "user", "content": image_blocks}],
                )

                api_calls += 1

                if response.usage:
                    total_input_tokens  += response.usage.input_tokens
                    total_output_tokens += response.usage.output_tokens
                    total_tokens        += response.usage.total_tokens

                # Save raw response before parsing
                raw_path.write_text(response.output_text, encoding="utf-8")
                raw_text = response.output_text

                status_box.write(
                    f"✅ {specimen_name} / **{col}** — "
                    f"{len(crops)} label(s) sent | "
                    f"tokens: {response.usage.total_tokens if response.usage else '?'}"
                )

            # Parse value from JSON response
            parsed = extract_json(raw_text)
            value  = None
            for key in [col, col.replace(".", "_"), col.lower()]:
                if key in parsed:
                    v = parsed[key]
                    value = (
                        "" if (v is None or str(v).strip().lower() in ("null", "none", ""))
                        else str(v).strip()
                    )
                    break

            if value is None:
                for v in parsed.values():
                    if v is not None and str(v).strip().lower() not in ("null", "none", ""):
                        value = str(v).strip()
                        break
                value = value or ""

            existing_results[specimen_name][output_key] = value

        except Exception as e:
            status_box.write(f"❌ {specimen_name} / **{col}**: {type(e).__name__}: {e}")
            failed.append((f"{specimen_name} / {col}", f"{type(e).__name__}: {e}"))
            existing_results[specimen_name][output_key] = ""

        if (idx + 1) % 10 == 0:
            write_results()

        token_box.write(
            f"**Token usage so far** — "
            f"input: {total_input_tokens:,} | output: {total_output_tokens:,} | "
            f"total: {total_tokens:,} | API calls: {api_calls} | reused: {reused_from_disk}"
        )

except KeyboardInterrupt:
    interrupted = True
    status_box.warning("Run interrupted.")

write_results()
progress_bar.progress(1.0, text="Done")

# =========================================================
# SUMMARY
# =========================================================

elapsed       = time.time() - start_time
minutes, secs = divmod(elapsed, 60)

st.divider()
st.subheader("Run complete" if not interrupted else "Run interrupted")

col_l, col_r = st.columns(2)
with col_l:
    st.metric("Items processed", len(work_items) - len(failed))
    st.metric("New API calls", api_calls)
    st.metric("Re-used from disk", reused_from_disk)
with col_r:
    st.metric("Total tokens used", f"{total_tokens:,}")
    st.metric("Failures", len(failed))
    st.metric("Time", f"{int(minutes)}m {secs:.0f}s")

if OUTPUT_CSV.exists():
    st.success(f"Results saved → `{OUTPUT_CSV.relative_to(PROJECT_ROOT)}`")

if failed:
    with st.expander(f"{len(failed)} failure(s)"):
        for name, why in failed:
            st.write(f"- **{name}**: {why}")

# =========================================================
# PREVIEW
# =========================================================

if OUTPUT_CSV.exists():
    st.divider()
    st.subheader("Preview")
    st.dataframe(pd.read_csv(OUTPUT_CSV, dtype=str), use_container_width=True, hide_index=True)
