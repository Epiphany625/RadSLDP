import json
import os
import time
from openai import OpenAI

# --- Configuration ---
SUMMARIES_FILE = "sweep_result_clean_split/nn15_nc16_mcs100_ms50/cluster_summaries.json"
OUTPUT_DIR = "sweep_result_clean_split/nn15_nc16_mcs100_ms50/categorize_result_5.2"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "category_taxonomy.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "categorize_log.jsonl")

MODEL = "gpt-5.2"
MAX_RETRIES = 5
INITIAL_BACKOFF = 30
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """You are categorizing radiology reason-for-visit cluster summaries into a BROAD, REUSABLE taxonomy.

Each cluster summary represents ONE group of related reasons. Assign it to exactly ONE category based on its primary meaning. If the boundary is vague, choose the category that best captures the dominant clinical concept.

For each NEW REASON TEXT:
1) Decide whether the information is image-inferable from a chest X-ray alone.
   - Output: "yes" or "no"
   - Provide a short justification (1 sentence).

2) Assign the reason to exactly ONE category:
   - Assign it to an existing category, OR
   - Fix (broaden) an existing category definition to include it, OR
   - Create a new BROAD, GENERAL category if no existing category can be reasonably expanded.

Critical Guidelines:
- ALWAYS prefer assign > fix > create. Only create when no existing category can be fixed.
- When creating, make categories BROAD and GENERAL enough for many similar future cases.
- Definitions should be inclusive and abstract.
- Do NOT rename categories.
- A fix may only GENERALIZE a definition, never narrow it.
- Use concise category names and definitions.
- Create as FEW categories as possible while maintaining meaningful distinctions.

Return valid JSON only. No extra text.
Schema:
{
  "reason": {
    "text": "...",
    "image_inferable": "yes" | "no",
    "why": "one sentence justification"
  },

  "categorization": {
    "action": "assign" | "create" | "fix",

    "category_id": <int>,              // for assign or fix

    "new_category": {                  // for create
      "category_name": "...",
      "definition": "..."
    },

    "updated_definition": "...",       // for fix

    "why": "brief justification"
  }
}
"""


def build_user_prompt(categories, item_text, item_id):
    """Build the user prompt with current categories and the new item."""
    if categories:
        cat_list = json.dumps(
            [{"category_id": c["category_id"], "category_name": c["category_name"],
              "definition": c["definition"]}
             for c in categories],
            indent=2
        )
    else:
        cat_list = "[]  (no categories yet — you must create the first one)"

    return f"""CURRENT CATEGORIES:
{cat_list}

NEW ITEM:
cluster_id: {item_id}
reason_text: "{item_text}"

Decide: assign to an existing category, or create a new one."""


def call_llm(client, user_prompt):
    """Call GPT-5.2 with retry on rate limit errors."""
    for attempt in range(MAX_RETRIES):
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
            text = response.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"  Rate limited (attempt {attempt + 1}/{MAX_RETRIES}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def load_progress():
    """Load existing progress to allow resuming."""
    categories = []
    processed_ids = set()

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            categories = json.load(f)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                entry = json.loads(line)
                processed_ids.add(str(entry["cluster_id"]))

    return categories, processed_ids


def save_categories(categories):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)


def log_decision(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load cluster summaries
    with open(SUMMARIES_FILE, "r") as f:
        summaries = json.load(f)

    # Build {cluster_id: reason_text} from summaries
    representatives = {}
    for cid, entry in summaries.items():
        text = entry.get("summary") or entry.get("representative", "")
        representatives[cid] = text

    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    categories, processed_ids = load_progress()

    total = len(representatives)
    print(f"Model: {MODEL}")
    print(f"Total cluster summaries: {total}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Existing categories: {len(categories)}")
    for cluster_id, reason_text in representatives.items():
        if str(cluster_id) in processed_ids:
            continue

        print(f"\n[{len(processed_ids) + 1}/{total}] cluster {cluster_id}: \"{reason_text}\"")

        prompt = build_user_prompt(categories, reason_text, cluster_id)
        response = call_llm(client, prompt)

        decision = response.get("categorization", response)
        action = decision["action"]
        member = {"cluster_id": cluster_id, "text": reason_text}

        if action == "assign":
            cat_id = decision["category_id"]
            for cat in categories:
                if cat["category_id"] == cat_id:
                    cat["members"].append(member)
                    print(f"  -> Assigned to: {cat['category_name']} (id={cat_id})")
                    break
            else:
                print(f"  WARNING: category_id {cat_id} not found, skipping")

        elif action == "fix":
            cat_id = decision["category_id"]
            for cat in categories:
                if cat["category_id"] == cat_id:
                    old_def = cat["definition"]
                    cat["definition"] = decision["updated_definition"]
                    cat["members"].append(member)
                    print(f"  -> Fixed category: {cat['category_name']} (id={cat_id})")
                    print(f"     old def: {old_def}")
                    print(f"     new def: {cat['definition']}")
                    break
            else:
                print(f"  WARNING: category_id {cat_id} not found for fix, skipping")

        elif action == "create":
            new_cat = decision["new_category"]
            cat_id = len(categories)
            categories.append({
                "category_id": cat_id,
                "category_name": new_cat["category_name"],
                "definition": new_cat["definition"],
                "members": [member]
            })
            print(f"  -> NEW category: {new_cat['category_name']} (id={cat_id})")

        else:
            print(f"  WARNING: unknown action '{action}', skipping")

        # Log and save after every cluster
        log_decision({
            "cluster_id": cluster_id,
            "reason_text": reason_text,
            "reason_info": response.get("reason"),
            "categorization": decision,
        })
        save_categories(categories)
        processed_ids.add(str(cluster_id))
    # Final summary
    print(f"\n{'='*60}")
    print(f"DONE. {len(categories)} categories created.")
    for cat in categories:
        print(f"  [{cat['category_id']}] {cat['category_name']}: {len(cat['members'])} members")
    print(f"\nTaxonomy saved to: {OUTPUT_FILE}")
    print(f"Decision log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
