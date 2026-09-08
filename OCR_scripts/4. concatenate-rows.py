from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_CSV = Path("/Users/christinapanagaki/Documents/museum.labels.project/2026/final_run/full-pipeline-other-collections/florida-museum-extras/pre-concatenation-seperate-prompts-data-structuring/structured_dwc_metadata_checkpoint_florida_leftovers_low_res.csv")
OUTPUT_CSV = Path("concatenated-florida-leftovers-low-res.csv")

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------
# Create specimen prefix
# ---------------------------------------------------------

df["image_prefix"] = (
    df["Image.name"]
    .str.replace(r"_label_\d+.*$", "", regex=True)
)

# Preserve the individual crop filenames under a non-colliding name
# before grouping.
df["label_crop_filenames"] = df["Image.name"]
df = df.drop(columns=["Image.name", "Specimen.image"])

# ---------------------------------------------------------
# Concatenation function
# ---------------------------------------------------------

def combine(series):

    values = []

    for value in series:

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value == "":
            continue

        if value not in values:
            values.append(value)

    if not values:
        return None

    return " | ".join(values)

# ---------------------------------------------------------
# Group by specimen
# ---------------------------------------------------------

result = (
    df
    .groupby("image_prefix", as_index=False)
    .agg(combine)
)

# Keep the prefix as Image.name
result.rename(columns={"image_prefix": "Image.name"}, inplace=True)

# Put the identifier and crop-filename list up front for readability.
front = ["Image.name", "label_crop_filenames"]
result = result[front + [c for c in result.columns if c not in front]]

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

result.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(result)} rows to {OUTPUT_CSV}")
