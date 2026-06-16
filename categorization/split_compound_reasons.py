"""
LLM-based compound reason splitting.

Sends all distinct reasons to gpt-4.1-mini to decide whether each should be
further split (e.g. on "and", "/", "or") and into what parts.

Input:  original_data/reason_index.json  (list where list[idx] = reason_text)
Output: original_data/compound_split_map.json  ({reason_idx: [split_parts]})
        original_data/compound_split_log.jsonl  (batch-level resume log)
"""

import json
import os
import time
from openai import OpenAI

# --- Configuration ---
BASE_DIR = "original_data"
INPUT_FILE = os.path.join(BASE_DIR, "reason_index.json")
SPLIT_MAP_FILE = os.path.join(BASE_DIR, "compound_split_map.json")
LOG_FILE = os.path.join(BASE_DIR, "compound_split_log.jsonl")

MODEL = "gpt-4.1-mini"
BATCH_SIZE = 20
MAX_RETRIES = 5
INITIAL_BACKOFF = 30
SLEEP_SEC = 0.1
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """You are processing radiology reason-for-visit text.

Your task is to split the text into parts ONLY when they represent
independent clinical information.

Each part may correspond to different types of information such as:
- demographic information (e.g., age, sex),
- medical history or prior conditions,
- symptoms or physical findings,
- measurements or observed values,
- procedures or treatments.

Splitting principles:
- Use ONLY words that already appear in the original text.
- Do NOT rephrase, normalize, or add any information.
- Keep a phrase intact if it describes ONE integrated clinical situation,
  even if it is long or complex.
- Do NOT split cause–effect, temporal, or procedural chains
  (e.g., "after X due to Y", "status post X for Y").
- Split only when the parts could stand alone as separate clinical facts
  without losing meaning.

Examples:

Input:
"assessment of the patient after massive blood transfusion due to splenic rupture"

Output:
["assessment of the patient after massive blood transfusion due to splenic rupture"]

Input:
"history of hiv with lower extremity swelling and o2 saturation of 94%"

Output:
[
  "history of hiv",
  "lower extremity swelling",
  "o2 saturation of 94%"
]

Now process the following text and return ONLY a JSON array of strings.
Do not add, remove, or modify words.
"""


def call_llm(client, batch):
    """Send a batch of reasons to the LLM. Returns parsed results array."""
    user_payload = json.dumps(
        [{"reason_idx": idx, "reason_text": text} for idx, text in batch],
        ensure_ascii=False
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "results" in parsed:
                return parsed["results"]
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  Rate limited (attempt {attempt+1}/{MAX_RETRIES}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"  Error: {err_str[:100]}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def load_progress():
    """Load existing progress for resume."""
    split_map = {}
    processed_batch_ids = set()

    if os.path.exists(SPLIT_MAP_FILE):
        with open(SPLIT_MAP_FILE) as f:
            split_map = json.load(f)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    processed_batch_ids.add(entry["batch_id"])

    return split_map, processed_batch_ids


def save_split_map(split_map):
    with open(SPLIT_MAP_FILE, "w") as f:
        json.dump(split_map, f, indent=2, ensure_ascii=False)


def log_batch(batch_id, results):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({"batch_id": batch_id, "results": results}, ensure_ascii=False) + "\n")


def main():
    # 1. Load all reasons
    print(f"Loading reasons from {INPUT_FILE}...")
    with open(INPUT_FILE) as f:
        reason_index = json.load(f)
    print(f"  {len(reason_index)} distinct reasons")

    # 2. Create (idx, text) pairs for all reasons
    all_items = list(enumerate(reason_index))

    # 3. Create batches
    batches = [all_items[i:i+BATCH_SIZE] for i in range(0, len(all_items), BATCH_SIZE)]
    print(f"  {len(batches)} batches of {BATCH_SIZE}")

    # 4. Load progress
    split_map, processed_batch_ids = load_progress()
    remaining = sum(1 for i in range(len(batches)) if f"batch_{i:06d}" not in processed_batch_ids)
    print(f"  {len(processed_batch_ids)} batches already done, {remaining} remaining")

    if remaining == 0:
        print("All batches already processed!")
        print(f"Split map has {len(split_map)} entries")
        return

    # 5. Initialize client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 6. Process batches
    split_count = len(split_map)
    for batch_idx, batch in enumerate(batches):
        batch_id = f"batch_{batch_idx:06d}"
        if batch_id in processed_batch_ids:
            continue

        print(f"[{batch_idx+1}/{len(batches)}] Processing reasons {batch[0][0]}-{batch[-1][0]}...", end="", flush=True)

        try:
            results = call_llm(client, batch)
        except Exception as e:
            print(f" FAILED: {e}")
            continue

        # Update split_map: only store reasons that need splitting
        batch_splits = 0
        for result in results:
            ridx = result.get("reason_idx")
            if ridx is not None and result.get("split"):
                parts = result.get("parts", [])
                if len(parts) > 1:
                    split_map[str(ridx)] = parts
                    batch_splits += 1

        split_count += batch_splits
        print(f" {batch_splits} splits (total: {split_count})")

        # Log and save
        log_batch(batch_id, results)
        save_split_map(split_map)
        processed_batch_ids.add(batch_id)

        time.sleep(SLEEP_SEC)

    # 7. Summary
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Total reasons: {len(reason_index)}")
    print(f"  Reasons split: {len(split_map)}")
    print(f"  Reasons kept:  {len(reason_index) - len(split_map)}")
    print(f"  Split map saved to {SPLIT_MAP_FILE}")


if __name__ == "__main__":
    main()
