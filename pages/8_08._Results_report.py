"""
Step 8 — Results report

Compares the corrected text from Step 7 against what the reading step first
produced, so you can see which columns needed the most fixing. Only images
marked "Manually checked" in Step 7 count towards the correction rate — an
unchecked row is just an untouched copy of the reading step's output, and
would make every column look error-free.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

current_proj = st.session_state.get("current_project", "No project selected")
st.info(f"Current project: **{current_proj}**")

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "projects" / current_proj
RAW_CSV = PROJECT_ROOT / "data" / "06_structured_output" / "structured_dwc_metadata.csv"
CORRECTED_CSV = PROJECT_ROOT / "data" / "07_quality_checking" / "corrected_metadata.csv"

ID_COLUMN = "Specimen.image"

st.title("Step 8 — Results report")
st.write(
    "See which columns the reading step (OCR) struggled with most, based on how "
    "often they had to be corrected in Step 7. Columns with a high correction "
    "rate are good candidates for a re-read with a different prompt."
)

if not RAW_CSV.exists():
    st.warning(f"No results found at `{RAW_CSV.relative_to(PROJECT_ROOT)}`. Complete Step 6 first.")
    st.stop()

if not CORRECTED_CSV.exists():
    st.warning(f"No corrections found at `{CORRECTED_CSV.relative_to(PROJECT_ROOT)}`. Complete Step 7 first.")
    st.stop()

raw_df = pd.read_csv(RAW_CSV, dtype=str).fillna("")
corrected_df = pd.read_csv(CORRECTED_CSV, dtype=str).fillna("")

if "Manually checked" not in corrected_df.columns:
    corrected_df["Manually checked"] = "no"

FIELDS = [
    c for c in raw_df.columns
    if c != ID_COLUMN and c in corrected_df.columns
]

merged = raw_df.merge(corrected_df, on=ID_COLUMN, suffixes=(".read", ".corrected"))
checked = merged[merged["Manually checked"] == "yes"]

total = len(raw_df)
checked_count = len(checked)

st.write(f"{total} images went through the reading step. {checked_count} have been manually checked in Step 7.")

if checked_count == 0:
    st.warning(
        "No images are marked \"Manually checked\" in Step 7 yet, so there's nothing "
        "to compare corrections against. The \"Filled by reading step\" column below "
        "still works without any checking."
    )

rows = []
for field in FIELDS:
    read_col = f"{field}.read" if f"{field}.read" in merged.columns else field
    corrected_col = f"{field}.corrected" if f"{field}.corrected" in merged.columns else field

    fill_rate = (raw_df[field].str.strip() != "").mean() * 100 if total else 0.0

    if checked_count:
        corrections = int((checked[read_col].str.strip() != checked[corrected_col].str.strip()).sum())
        correction_rate = corrections / checked_count * 100
    else:
        corrections = None
        correction_rate = None

    rows.append({
        "Column": field,
        "Filled by reading step": fill_rate,
        "Corrected (of checked images)": correction_rate,
        "Times corrected": corrections,
    })

report_df = pd.DataFrame(rows)

if checked_count:
    report_df = report_df.sort_values("Corrected (of checked images)", ascending=False)
    st.bar_chart(report_df.set_index("Column")["Corrected (of checked images)"])

display_df = report_df.copy()
display_df["Filled by reading step"] = display_df["Filled by reading step"].map(lambda v: f"{v:.0f}%")
display_df["Corrected (of checked images)"] = display_df["Corrected (of checked images)"].map(
    lambda v: f"{v:.0f}%" if pd.notna(v) else "—"
)
display_df["Times corrected"] = display_df["Times corrected"].map(lambda v: v if pd.notna(v) else "—")

st.dataframe(display_df, hide_index=True, use_container_width=True)

st.caption(
    "\"Manually checked\" is set per image in Step 7, not per column — a checked "
    "image may still have had only some of its columns reviewed there, which the "
    "correction rate can't tell apart from a column that was reviewed and found correct."
)
