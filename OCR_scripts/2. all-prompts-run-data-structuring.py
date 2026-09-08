from openai import OpenAI
from pathlib import Path

import csv
import json
import re
import shutil
import time

# =========================================================
# PROCESSING CONTROL
# =========================================================

# None = process all rows
# Integer = process only first N rows

MAX_ROWS = 804

# =========================================================
# BASE PATHS
# =========================================================

BASE_DIR = Path(
    "/Users/christinapanagaki/Documents/museum.labels.project/2026/final_run/full-pipeline-other-collections/florida-museum-extras/seperate-prompts-ocr"
)

INPUT_CSV = BASE_DIR / "specimen_level_transcriptions.csv"
OUTPUT_CSV = BASE_DIR / "structured_dwc_metadata.csv"
CHECKPOINT_CSV = BASE_DIR / "structured_dwc_metadata_checkpoint.csv"
RAW_RESPONSE_DIR = BASE_DIR / "raw_prompt_responses"
PROMPT_DIR = BASE_DIR / "seperate-prompts"

# Live-updated token usage snapshot (overwritten after every specimen,
# so token spend is never lost even if the run is force-stopped)
TOKEN_STATS_JSON = BASE_DIR / "token_usage_progress.json"

API_KEY_PATH = Path(
    "/Users/christinapanagaki/Documents/"
    "museum.labels.project/2026/api.key"
)

# =========================================================
# PROMPT FILES
# =========================================================

PROMPTS = {
    "taxonomy":         PROMPT_DIR / "species-prompt.txt",
    "collection_event": PROMPT_DIR / "collection-event-prompt.txt",
    "geography":        PROMPT_DIR / "geography-prompt.txt",
    "identifiers":      PROMPT_DIR / "collection-identifiers-prompt.txt",
    "remarks":          PROMPT_DIR / "remarks-etc-prompt.txt"
}

# =========================================================
# OUTPUT SCHEMA
# =========================================================

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
    "Raw.transcription"
]

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

for prompt_name in PROMPTS:
    (RAW_RESPONSE_DIR / prompt_name).mkdir(parents=True, exist_ok=True)

# =========================================================
# START TIMER
# =========================================================

start_time = time.time()

# =========================================================
# LOAD API KEY
# =========================================================

api_key = API_KEY_PATH.read_text(encoding="utf-8").strip()

# =========================================================
# OPENAI CLIENT
# =========================================================

client = OpenAI(api_key=api_key)

# =========================================================
# LOAD PROMPTS
# =========================================================

loaded_prompts = {}

for prompt_name, prompt_path in PROMPTS.items():
    loaded_prompts[prompt_name] = prompt_path.read_text(encoding="utf-8")

print(f"Loaded {len(loaded_prompts)} prompts")

# =========================================================
# TOKEN TRACKING
# =========================================================

token_stats = {
    prompt_name: {"input": 0, "cached": 0, "output": 0, "total": 0}
    for prompt_name in PROMPTS
}

def save_token_stats(specimens_done, specimens_total, elapsed_seconds):
    """
    Overwrite TOKEN_STATS_JSON with the current cumulative token usage.
    Called after every specimen (and on interrupt/finish) so nothing
    is lost if the run is stopped early.
    """
    grand_input = sum(s["input"] for s in token_stats.values())
    grand_cached = sum(s["cached"] for s in token_stats.values())
    grand_output = sum(s["output"] for s in token_stats.values())
    grand_total = sum(s["total"] for s in token_stats.values())

    snapshot = {
        "specimens_done": specimens_done,
        "specimens_total": specimens_total,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "per_prompt": token_stats,
        "grand_total": {
            "input": grand_input,
            "cached": grand_cached,
            "output": grand_output,
            "total": grand_total
        }
    }

    with open(TOKEN_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

# =========================================================
# JSON EXTRACTION
# =========================================================

def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end])

# =========================================================
# LOAD INPUT CSV
# =========================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found:\n{INPUT_CSV}")

with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# =========================================================
# OPTIONAL ROW LIMIT
# =========================================================

if MAX_ROWS is not None:
    rows = rows[:MAX_ROWS]

print(f"Loaded {len(rows)} rows")
print(f"Processing limit: {MAX_ROWS if MAX_ROWS else 'ALL'}")

# =========================================================
# LOAD CHECKPOINTS
# =========================================================

processed_images = set()

if CHECKPOINT_CSV.exists():
    with open(CHECKPOINT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_images.add(row["Specimen.image"])

print(f"Already processed: {len(processed_images)}")

# =========================================================
# CREATE CHECKPOINT FILE
# =========================================================

if not CHECKPOINT_CSV.exists():
    with open(CHECKPOINT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

# =========================================================
# GPT CALL FUNCTION
# =========================================================

def run_prompt(prompt_name, system_prompt, transcription):

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
                            "Extract structured metadata "
                            "from this museum specimen OCR "
                            "transcription.\n\n"
                            f"{transcription}"
                        )
                    }
                ]
            }
        ]
    )

    if response.usage:
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        total_tokens = usage.total_tokens
        cached_tokens = 0

        if hasattr(usage, "input_tokens_details"):
            details = usage.input_tokens_details
            if hasattr(details, "cached_tokens"):
                cached_tokens = details.cached_tokens

        token_stats[prompt_name]["input"]  += input_tokens
        token_stats[prompt_name]["output"] += output_tokens
        token_stats[prompt_name]["cached"] += cached_tokens
        token_stats[prompt_name]["total"]  += total_tokens

        print(
            f"    input={input_tokens} | "
            f"output={output_tokens} | "
            f"cached={cached_tokens} | "
            f"total={total_tokens}"
        )

    return response.output_text

# =========================================================
# PROCESS ROWS
# =========================================================

failed = []
total_rows = len(rows)
rows_to_process = [r for r in rows if r["Specimen.image"] not in processed_images]
already_skipped = total_rows - len(rows_to_process)
done_count = 0
interrupted = False

print(f"\n{len(rows_to_process)} specimens remaining "
      f"({already_skipped} already done, {total_rows} total)")

try:

    for row in rows_to_process:

        specimen_image = row["Specimen.image"]
        raw_transcription = row["Raw.transcription"]

        done_count += 1
        elapsed_so_far = time.time() - start_time
        avg_per_specimen = elapsed_so_far / done_count
        remaining = len(rows_to_process) - done_count
        eta_seconds = avg_per_specimen * remaining
        eta_min, eta_sec = divmod(eta_seconds, 60)

        print(f"\n=================================")
        print(f"[{done_count}/{len(rows_to_process)}] "
              f"(overall {already_skipped + done_count}/{total_rows}) "
              f"Processing: {specimen_image}  "
              f"| elapsed {int(elapsed_so_far // 60)}m{int(elapsed_so_far % 60)}s "
              f"| ETA {int(eta_min)}m{int(eta_sec)}s")

        combined = {
            "Specimen.image": specimen_image,
            "Raw.transcription": raw_transcription
        }

        try:

            for prompt_name, prompt_text in loaded_prompts.items():

                print(f"  Running prompt: {prompt_name}")

                raw_response = run_prompt(
                    prompt_name,
                    prompt_text,
                    raw_transcription
                )

                # Save raw response
                raw_response_path = (
                    RAW_RESPONSE_DIR /
                    prompt_name /
                    f"{Path(specimen_image).stem}.txt"
                )

                with open(raw_response_path, "w", encoding="utf-8") as f:
                    f.write(raw_response)

                # Parse and merge — only keep known output fields
                parsed = extract_json(raw_response)
                for field in OUTPUT_FIELDS:
                    if field in parsed:
                        combined[field] = parsed[field]

            # Ensure all output fields exist
            for field in OUTPUT_FIELDS:
                if field not in combined:
                    combined[field] = ""

            # Save checkpoint immediately
            with open(CHECKPOINT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
                writer.writerow(combined)
                f.flush()

            print(f"  Finished: {specimen_image}")

        except Exception as e:
            print(f"\nFAILED: {specimen_image}")
            print(e)
            failed.append(specimen_image)

        # Persist token usage after every specimen, success or failure,
        # so a force-stop never loses the running token tally.
        save_token_stats(
            specimens_done=already_skipped + done_count,
            specimens_total=total_rows,
            elapsed_seconds=time.time() - start_time
        )

except KeyboardInterrupt:
    interrupted = True
    print("\n\n*** RUN INTERRUPTED BY USER (Ctrl+C) ***")
    print(f"Completed {done_count}/{len(rows_to_process)} specimens this session "
          f"before stopping.")
    print(f"Checkpoint file is up to date through the last finished specimen:")
    print(f"  {CHECKPOINT_CSV}")
    save_token_stats(
        specimens_done=already_skipped + done_count,
        specimens_total=total_rows,
        elapsed_seconds=time.time() - start_time
    )
    print(f"Token usage snapshot saved to:")
    print(f"  {TOKEN_STATS_JSON}")

# =========================================================
# COPY FINAL OUTPUT
# =========================================================

shutil.copy(CHECKPOINT_CSV, OUTPUT_CSV)

# =========================================================
# SUMMARY
# =========================================================

elapsed = time.time() - start_time
minutes, seconds = divmod(elapsed, 60)

print("\n=================================")
print("INTERRUPTED — PARTIAL RUN" if interrupted else "FINISHED")
print("=================================")
print(f"Processed this session: {done_count - len(failed)}")
print(f"Failed this session:    {len(failed)}")
print(f"Total done overall:     {already_skipped + done_count}/{total_rows}")

if failed:
    print("\nFailed specimens:")
    for specimen in failed:
        print(f"  - {specimen}")

print(f"\nElapsed time: {int(minutes)}m {seconds:.1f}s")

# =========================================================
# TOKEN SUMMARY
# =========================================================

print("\n=================================")
print("TOKEN USAGE")
print("=================================")

grand_input = grand_cached = grand_output = grand_total = 0

for prompt_name, stats in token_stats.items():
    print(f"\n{prompt_name}")
    print(f"  input:   {stats['input']:,}")
    print(f"  cached:  {stats['cached']:,}")
    print(f"  output:  {stats['output']:,}")
    print(f"  total:   {stats['total']:,}")
    grand_input  += stats["input"]
    grand_cached += stats["cached"]
    grand_output += stats["output"]
    grand_total  += stats["total"]

print("\n=================================")
print("GRAND TOTAL")
print("=================================")
print(f"Input:   {grand_input:,}")
print(f"Cached:  {grand_cached:,}")
print(f"Output:  {grand_output:,}")
print(f"Total:   {grand_total:,}")

print(f"\n(This same data is saved after every specimen to:\n {TOKEN_STATS_JSON})")
