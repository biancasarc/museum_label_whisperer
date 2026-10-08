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

SECOND_PARSE_DIR  = PROJECT_ROOT / "data" / "08_second_parse"
RAW_RESPONSE_DIR  = SECOND_PARSE_DIR / "raw_responses"
OUTPUT_CSV        = SECOND_PARSE_DIR / "second_parse_results.csv"
PROMPTS_DIR       = SECOND_PARSE_DIR / "prompts"

# Per-column prompt files created in the separate-prompts/per-column folder
COLUMN_PROMPT_DIR = Path(__file__).resolve().parents[1] / "separate-prompts" / "per-column"

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


def extract_rules(column: str) -> str:
    """
    Pull just the rules section out of a per-column prompt file.
    Strips the intro sentence and JSON schema, keeps everything up to
    (but not including) the 'OCR input:' line.
    Falls back to a one-line generic rule if no file exists.
    """
    path = COLUMN_PROMPT_DIR / f"{column}.txt"
    if not path.exists():
        return f"* Extract the {column} field. Return null if absent or unreadable."
    text = path.read_text(encoding="utf-8")
    # Drop 'OCR input:' and everything after it
    text = text.split("OCR input:")[0].strip()
    # Drop everything up to and including the closing } of the JSON schema block
    if "}" in text:
        text = text[text.rfind("}") + 1:].strip()
    return text


def default_prompt_for(columns: list[str]) -> str:
    """
    Assemble a combined extraction prompt from the individual per-column
    prompt files. Each column's rules section is included under its own
    header so the model knows exactly what to look for.
    """
    fields_json = "\n".join(f'  "{c}": ""' for c in columns)
    sections = "\n\n".join(
        f"=== {c} ===\n{extract_rules(c)}" for c in columns
    )
    return (
        "You are an expert in natural history museum specimens.\n\n"
        "Look carefully at the label image(s) provided and extract the following fields.\n\n"
        "Return ONLY a valid JSON object with exactly these keys "
        "(use null if a field is not present or unreadable):\n"
        f"{{\n{fields_json}\n}}\n\n"
        f"{sections}"
    )


def _prompt_key(columns: list[str]) -> str:
    """Stable filename key for a given set of columns."""
    return "+".join(sorted(columns))


def load_prompt(columns: list[str]) -> str:
    path = PROMPTS_DIR / f"{_prompt_key(columns)}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else default_prompt_for(columns)


def save_prompt(columns: list[str], text: str) -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROMPTS_DIR / f"{_prompt_key(columns)}.txt").write_text(text, encoding="utf-8")


def labels_for_columns(
    specimen_name: str,
    columns: list[str],
    ocr_df: pd.DataFrame,
) -> list[Path]:
    """
    Return the crop image paths most relevant for re-parsing `columns`.

    Strategy:
      1. Collect the union of target segment types across all selected columns.
      2. Keep crop labels whose Segments JSON contains at least one of those types.
      3. If none match (or no mapping exists), fall back to ALL crops.
    """
    specimen_rows = ocr_df[ocr_df["Specimen.image"] == specimen_name]

    # Union of segment types across all selected columns
    target_types: set[str] = set()
    for col in columns:
        types = COLUMN_TO_SEGMENT_TYPES.get(col)
        if types:
            target_types.update(types)

    matched_image_names: list[str] = []

    for _, row in specimen_rows.iterrows():
        image_name   = row["Image.name"]
        segments_raw = row.get("Segments", "")

        # No type mapping for any selected column — include all labels
        if not target_types:
            matched_image_names.append(image_name)
            continue

        try:
            segments       = json.loads(segments_raw)
            types_in_label = {s.get("segment_type", "") for s in segments}
            if types_in_label & target_types:
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

    # Single prompt editor covering all selected columns
    st.divider()
    st.subheader("Reading instructions")
    st.caption("One prompt is sent per specimen, extracting all selected columns in a single call.")

    prompt_key = _prompt_key(selected_columns)
    with st.expander("Edit prompt", expanded=False):
        active_prompt = st.text_area(
            "Prompt",
            value=load_prompt(selected_columns),
            height=220,
            key=f"sp_prompt_{prompt_key}",
            label_visibility="collapsed",
        )
        c1, c2, _ = st.columns([1.2, 1.8, 5])
        with c1:
            if st.button("💾 Save", key=f"save_prompt_{prompt_key}", use_container_width=True):
                save_prompt(selected_columns, active_prompt)
                st.success("Saved.")
        with c2:
            if st.button("↩️ Reset to default", key=f"reset_prompt_{prompt_key}", use_container_width=True):
                default = default_prompt_for(selected_columns)
                save_prompt(selected_columns, default)
                st.session_state[f"sp_prompt_{prompt_key}"] = default
                st.rerun()
else:
    active_prompt = ""

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

work_items: list[str] = []   # specimen names
output_keys = [f"{col}_second_parse" for col in selected_columns]

# Eligible: non-empty in at least one selected column, optionally not checked
base_mask = pd.Series([False] * len(df))
for col in selected_columns:
    base_mask |= df[col].str.strip() != ""
if skip_checked:
    base_mask &= df.get("Manually checked", pd.Series(["no"] * len(df))) != "yes"
eligible = df[base_mask]

for specimen_name in eligible["Specimen.image"]:
    if (
        specimen_name in existing_results
        and all(k in existing_results[specimen_name] for k in output_keys)
    ):
        continue   # all output keys already present for this specimen
    work_items.append(specimen_name)

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

safe_cols_key = re.sub(r"[^\w\-]", "_", _prompt_key(selected_columns))

try:
    for idx, specimen_name in enumerate(work_items):
        progress_bar.progress(idx / len(work_items), text=f"{idx}/{len(work_items)}")

        safe_spec = re.sub(r"[^\w\-]", "_", Path(specimen_name).stem)
        raw_path  = RAW_RESPONSE_DIR / f"{safe_spec}__{safe_cols_key}.txt"

        if specimen_name not in existing_results:
            existing_results[specimen_name] = {"Specimen.image": specimen_name}

        try:
            raw_text = None

            if reuse_responses and raw_path.exists():
                raw_text = raw_path.read_text(encoding="utf-8")
                reused_from_disk += 1
                status_box.write(f"♻ {specimen_name} — re-parsed from disk")

            else:
                crops = labels_for_columns(specimen_name, selected_columns, ocr_df)

                if not crops:
                    status_box.write(f"⚠ {specimen_name} — no crops found, skipping")
                    failed.append((specimen_name, "No crop images found"))
                    for key in output_keys:
                        existing_results[specimen_name][key] = ""
                    continue

                image_blocks = [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{preprocess_image(crop, upscale_factor)}",
                        "detail": "high",
                    }
                    for crop in crops
                ]

                response = client.responses.create(
                    model=model_name,
                    instructions=active_prompt,
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
                    f"✅ {specimen_name} — "
                    f"{len(crops)} label(s) sent | "
                    f"tokens: {response.usage.total_tokens if response.usage else '?'}"
                )

            # Parse JSON — extract one value per selected column
            parsed = extract_json(raw_text)
            for col in selected_columns:
                output_key = f"{col}_second_parse"
                v = parsed.get(col) or parsed.get(col.lower()) or parsed.get(col.replace(".", "_"))
                value = (
                    "" if (v is None or str(v).strip().lower() in ("null", "none", ""))
                    else str(v).strip()
                )
                existing_results[specimen_name][output_key] = value

        except Exception as e:
            status_box.write(f"❌ {specimen_name}: {type(e).__name__}: {e}")
            failed.append((specimen_name, f"{type(e).__name__}: {e}"))
            for key in output_keys:
                existing_results[specimen_name][key] = ""

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
    st.metric("Specimens processed", len(work_items) - len(failed))
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

# =========================================================
# COMPARISON WITH CORRECTED METADATA
# =========================================================

if OUTPUT_CSV.exists() and CORRECTED_CSV.exists():
    st.divider()
    st.subheader("Comparison with corrected metadata")
    st.caption(
        "Compares second-parse values against the ground-truth values in "
        "`corrected_metadata.csv`. Both sides are lowercased and stripped for comparison."
    )

    sp_df   = pd.read_csv(OUTPUT_CSV,    dtype=str).fillna("")
    corr_df = pd.read_csv(CORRECTED_CSV, dtype=str).fillna("")

    # Only compare columns that exist in both files
    comparable_cols = [
        c for c in selected_columns
        if f"{c}_second_parse" in sp_df.columns and c in corr_df.columns
    ]

    if not comparable_cols:
        st.info(
            "No overlap between selected columns and corrected_metadata.csv. "
            "Make sure the corrected CSV has matching column names."
        )
    else:
        merged = sp_df.merge(
            corr_df[["Specimen.image"] + comparable_cols],
            on="Specimen.image",
            how="inner",
            suffixes=("_sp", "_corr"),
        )

        if merged.empty:
            st.warning("No matching Specimen.image entries between the two files.")
        else:
            summary_rows = []
            detail_frames = {}

            for col in comparable_cols:
                sp_col   = f"{col}_second_parse"
                corr_col = col  # from corrected CSV (renamed to col_corr after merge if clash)

                # After merge with suffixes, corrected column might be renamed
                if corr_col not in merged.columns and f"{col}_corr" in merged.columns:
                    corr_col = f"{col}_corr"

                # Normalise: lowercase + strip for comparison
                sp_vals   = merged[sp_col].str.lower().str.strip()
                corr_vals = merged[corr_col].str.lower().str.strip()

                agree    = (sp_vals == corr_vals).sum()
                disagree = (sp_vals != corr_vals).sum()
                total    = len(merged)

                summary_rows.append({
                    "Column":       col,
                    "Agree":        agree,
                    "Disagree":     disagree,
                    "Total":        total,
                    "Agreement %":  f"{100 * agree / total:.1f}%" if total else "—",
                })

                # Build detail table for disagreements only
                mask = sp_vals != corr_vals
                detail = merged.loc[mask, ["Specimen.image", sp_col, corr_col]].copy()
                detail.columns = ["Specimen", "Second-parse value", "Corrected value"]
                detail_frames[col] = detail

            # Summary table
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            # Per-column expanders showing disagreements
            for col in comparable_cols:
                detail = detail_frames[col]
                n_dis  = len(detail)
                if n_dis == 0:
                    continue
                with st.expander(f"{col} — {n_dis} disagreement(s)"):
                    st.dataframe(detail, hide_index=True, use_container_width=True)

elif OUTPUT_CSV.exists() and not CORRECTED_CSV.exists():
    st.info("No corrected_metadata.csv found — run Step 7 first to enable comparison.")
