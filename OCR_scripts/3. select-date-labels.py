#!/usr/bin/env python3
"""
Copy image files listed in Image.name for every row that has a non-empty
eventDate value, from the source images folder into the destination folder.

Usage:
    python3 copy_eventdate_images.py

Edit CSV_PATH, SOURCE_DIR, DEST_DIR below if your paths change.
"""

import csv
import shutil
from pathlib import Path

# ---- Configuration -------------------------------------------------------
# NOTE: must point at the STRUCTURED checkpoint CSV (the one with an
# eventDate column), not the raw OCR CSV. raw-label-ocr-florida-leftovers.csv
# only has Specimen.image / Image.name / Segments -- no eventDate at all --
# so pointing CSV_PATH there means every row silently counts as "no date"
# and nothing ever gets copied.
CSV_PATH = "/Users/christinapanagaki/Documents/museum.labels.project/2026/final_run/full-pipeline-other-collections/florida-museum-extras/pre-concatenation-seperate-prompts-data-structuring/structured_dwc_metadata_checkpoint_florida_leftovers.csv"
SOURCE_DIR = "/Users/christinapanagaki/Documents/museum.labels.project/2026/final_run/full-pipeline-other-collections/florida-museum-extras/cropped-labels"
DEST_DIR = "/Users/christinapanagaki/Documents/museum.labels.project/2026/final_run/full-pipeline-other-collections/florida-museum-extras/dates-2nd-parse/only-date-labels"
# ---------------------------------------------------------------------------


def main():
    csv_path = Path(CSV_PATH)
    source_dir = Path(SOURCE_DIR)
    dest_dir = Path(DEST_DIR)

    dest_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source image folder not found: {source_dir}")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    if "eventDate" not in header:
        raise KeyError(
            f"'{csv_path.name}' has no 'eventDate' column (columns found: "
            f"{header}). You're likely pointed at the raw OCR CSV instead "
            f"of the structured checkpoint CSV -- check CSV_PATH."
        )

    copied = 0
    missing = []
    skipped_no_date = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_date = (row.get("eventDate") or "").strip()
            image_name = (row.get("Image.name") or "").strip()

            if not event_date:
                skipped_no_date += 1
                continue
            if not image_name:
                continue

            src_file = source_dir / image_name
            if not src_file.exists():
                missing.append(image_name)
                continue

            dest_file = dest_dir / image_name
            shutil.copy2(src_file, dest_file)
            copied += 1

    log_path = dest_dir / "_copy_log.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Copied {copied} image(s) to: {dest_dir}\n")
        log.write(f"Rows skipped (no eventDate): {skipped_no_date}\n")
        if missing:
            log.write(f"\n{len(missing)} image(s) listed in CSV but not found in source folder:\n")
            for m in missing:
                log.write(f"  - {m}\n")


if __name__ == "__main__":
    main()