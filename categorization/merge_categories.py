"""
Merge existing fine-grained categories into broader, more reusable categories.
This script takes an existing taxonomy and consolidates similar categories.
"""

import json
import os
import time
from openai import OpenAI

# --- Configuration ---
INPUT_TAXONOMY = "sweep_results_all_normalized_reasons/nn15_nc16_mcs100_ms50/categorize_result/category_taxonomy.json"
OUTPUT_DIR = "sweep_results_all_normalized_reasons/nn15_nc16_mcs100_ms50/categorize_result_merged"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_taxonomy.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "merge_log.jsonl")

MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# How many categories to process per LLM call
BATCH_SIZE = 20

SYSTEM_PROMPT = """You are consolidating a fine-grained taxonomy into a BROAD, REUSABLE category system.

You will receive a batch of existing categories. Your task:
1. Identify categories that are semantically similar or overlapping
2. Merge them into broader, more general categories
3. Preserve distinct semantic dimensions (don't merge unrelated concepts)

Guidelines:
- Use high-level clinical domain names
- Create inclusive, abstract definitions that cover a range of related conditions
- If categories represent fundamentally different concepts, keep them separate
- Prioritize reducing redundancy and over-specificity

Return valid JSON only. No extra text.

Schema:
{
  "merged_categories": [
    {
      "new_category_name": "...",
      "definition": "broad, inclusive definition",
      "source_category_ids": [1, 5, 12],  // IDs of categories being merged
      "rationale": "why these categories were merged"
    }
  ],
  "unchanged_categories": [7, 9, 15],  // IDs that should remain separate
  "summary": "brief summary of consolidation strategy"
}
"""


def call_llm(client, categories_batch):
    """Send a batch of categories to GPT for consolidation."""

    # Format categories for the prompt
    cat_list = []
    for cat in categories_batch:
        cat_list.append({
            "category_id": cat["category_id"],
            "category_name": cat["category_name"],
            "definition": cat["definition"],
            "member_count": len(cat["members"])
        })

    user_prompt = f"""Please consolidate these {len(categories_batch)} categories:

{json.dumps(cat_list, indent=2)}

Merge similar/overlapping categories into broader ones. Return the consolidation plan."""

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (2 ** attempt)
                print(f"  Rate limited (attempt {attempt + 1}/5). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after 5 retries")


def apply_merges(original_categories, merge_plan):
    """Apply the merge plan to create new consolidated categories."""

    merged = []
    id_map = {}  # old_id -> new_category object

    # Create merged categories
    for merge_spec in merge_plan.get("merged_categories", []):
        new_cat = {
            "category_id": len(merged),
            "category_name": merge_spec["new_category_name"],
            "definition": merge_spec["definition"],
            "members": [],
            "merged_from": merge_spec["source_category_ids"],
            "rationale": merge_spec.get("rationale", "")
        }

        # Collect all members from source categories
        for old_id in merge_spec["source_category_ids"]:
            for cat in original_categories:
                if cat["category_id"] == old_id:
                    new_cat["members"].extend(cat["members"])
                    id_map[old_id] = new_cat
                    break

        merged.append(new_cat)

    # Add unchanged categories
    for old_id in merge_plan.get("unchanged_categories", []):
        for cat in original_categories:
            if cat["category_id"] == old_id:
                new_cat = {
                    "category_id": len(merged),
                    "category_name": cat["category_name"],
                    "definition": cat["definition"],
                    "members": cat["members"],
                    "merged_from": [old_id],
                    "rationale": "kept separate"
                }
                merged.append(new_cat)
                id_map[old_id] = new_cat
                break

    return merged, id_map


def iterative_merge(categories, num_rounds=3, start_round=1):
    """Perform multiple rounds of merging to progressively consolidate."""

    client = OpenAI(api_key=OPENAI_API_KEY)
    current_cats = categories

    for round_num in range(start_round, start_round + num_rounds):
        print(f"\n{'='*80}")
        print(f"MERGE ROUND {round_num}/{num_rounds}")
        print(f"Current categories: {len(current_cats)}")
        print(f"{'='*80}")

        if len(current_cats) <= 10:
            print("Fewer than 10 categories remaining, stopping early.")
            break

        # Process in batches
        all_merged = []

        for i in range(0, len(current_cats), BATCH_SIZE):
            batch = current_cats[i:i + BATCH_SIZE]
            print(f"\nProcessing batch {i//BATCH_SIZE + 1} ({len(batch)} categories)...")

            merge_plan = call_llm(client, batch)
            merged_cats, _ = apply_merges(batch, merge_plan)

            all_merged.extend(merged_cats)

            print(f"  Consolidated {len(batch)} -> {len(merged_cats)} categories")
            time.sleep(1)  # Rate limiting

        # Log this round
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({
                "round": round_num,
                "input_count": len(current_cats),
                "output_count": len(all_merged),
                "reduction": len(current_cats) - len(all_merged)
            }) + "\n")

        current_cats = all_merged

        # Save intermediate results
        with open(OUTPUT_FILE.replace(".json", f"_round{round_num}.json"), "w") as f:
            json.dump(current_cats, f, indent=2, ensure_ascii=False)

        print(f"\nRound {round_num} complete: {len(current_cats)} categories remaining")

    return current_cats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for existing merge progress
    resume_file = None
    for i in range(10, 0, -1):
        candidate = OUTPUT_FILE.replace(".json", f"_round{i}.json")
        if os.path.exists(candidate):
            resume_file = candidate
            break

    # Load taxonomy (resume from latest round if available)
    if resume_file:
        print(f"Found existing merge progress: {resume_file}")
        print(f"Loading from {resume_file} to continue merging...")
        with open(resume_file) as f:
            original_cats = json.load(f)
        # Determine starting round number
        start_round = int(resume_file.split("_round")[-1].replace(".json", "")) + 1
    else:
        print(f"Loading taxonomy from {INPUT_TAXONOMY}")
        with open(INPUT_TAXONOMY) as f:
            original_cats = json.load(f)
        start_round = 1

    print(f"Original categories: {len(original_cats)}")
    print(f"Total members: {sum(len(c['members']) for c in original_cats)}")

    # Show category size distribution
    sizes = [len(c["members"]) for c in original_cats]
    sizes.sort()
    print(f"\nCategory size distribution:")
    print(f"  Min: {min(sizes)}, Median: {sizes[len(sizes)//2]}, Max: {max(sizes)}")
    print(f"  Categories with 1-5 members: {sum(1 for s in sizes if s <= 5)}")
    print(f"  Categories with 6-20 members: {sum(1 for s in sizes if 6 <= s <= 20)}")
    print(f"  Categories with 20+ members: {sum(1 for s in sizes if s > 20)}")

    # Perform iterative merging
    num_rounds = int(input("\nHow many additional merge rounds? [default: 3]: ") or "3")

    final_cats = iterative_merge(original_cats, num_rounds=num_rounds, start_round=start_round)

    # Save final taxonomy
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_cats, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Original categories: {len(original_cats)}")
    print(f"Final categories: {len(final_cats)}")
    print(f"Reduction: {len(original_cats) - len(final_cats)} ({100*(len(original_cats)-len(final_cats))/len(original_cats):.1f}%)")
    print(f"\nMerged taxonomy saved to: {OUTPUT_FILE}")

    # Show top 20 final categories
    final_cats.sort(key=lambda c: len(c["members"]), reverse=True)
    print(f"\nTop 20 categories by member count:")
    for i, cat in enumerate(final_cats[:20], 1):
        print(f"  {i}. {cat['category_name']}: {len(cat['members'])} members")
        if "merged_from" in cat and len(cat["merged_from"]) > 1:
            print(f"     (merged from {len(cat['merged_from'])} categories)")


if __name__ == "__main__":
    main()
