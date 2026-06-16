"""
Generate final output files from the clustering + categorization pipeline.

Uses noise_assignments.json to include former noise points in the output.

Outputs:
1. id_to_categorized_reasons.json
   For each dataset:id:
   {
     "dataset_id": "mimic:12345",
     "reasons": [
       {
         "reason_idx": 42,
         "reason_text": "original full normalized reason text",
         "parts": [
           {
             "part_text": "chest pain",
             "cluster_id": 7,
             "cluster_summary": "...",
             "final_category": "Clinical Symptoms and Functional Status",
             "image_inferable": "yes",
             "from_noise": false
           },
           ...
         ]
       },
       ...
     ]
   }

2. id_to_final_categories.json  (flat summary)
   {dataset:id -> {image_inferable: bool, final_categories: [str, ...]}}
"""

import json
import os
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = os.environ.get("DATA_DIR", "original_data")
SWEEP_DIR = os.environ.get("SWEEP_DIR", "sweep_result_clean_split")
RUN_NAME = os.environ.get("RUN_NAME", "nn15_nc16_mcs100_ms50")
CAT_NAME = os.environ.get("CAT_NAME", "categorize_result_5.2")

RUN_DIR = os.path.join(SWEEP_DIR, RUN_NAME)
CAT_DIR = os.path.join(RUN_DIR, CAT_NAME)

# Input files
CLUSTERS_FILE = os.path.join(RUN_DIR, "clusters.json")
NOISE_ASSIGNMENTS_FILE = os.path.join(RUN_DIR, "noise_assignments.json")
SUMMARIES_FILE = os.path.join(RUN_DIR, "cluster_summaries.json")
TAXONOMY_FILE = os.path.join(CAT_DIR, "category_taxonomy.json")
MAPPING_FILE = os.path.join(CAT_DIR, "category_mapping.json")
LOG_FILE = os.path.join(CAT_DIR, "categorize_log.jsonl")
REASON_INDEX_FILE = os.path.join(DATA_DIR, "reason_index.json")
ID_TO_REASONS_FILE = os.path.join(DATA_DIR, "id_to_reasons.json")
COMPOUND_SPLIT_MAP_FILE = os.path.join(DATA_DIR, "compound_split_map.json")

# Output files
DETAILED_OUTPUT = os.path.join(CAT_DIR, "id_to_categorized_reasons.json")
SUMMARY_OUTPUT = os.path.join(CAT_DIR, "id_to_final_categories.json")

WORD_COUNT_THRESHOLD = 6

# Additional augmented files to integrate (datasets not in the original pipeline)
# These already have reason_parts from augment_training_data.py (nearest-centroid)
AUGMENTED_EXTRAS = [
    {
        "file": f"{DATA_DIR}/augmented/rexgradient_test_augmented.json",
        "prefix": "rex_test",
        "id_fn": lambda e: e["id"],
    },
]

# Manual mappings for old categories not in category_mapping.json
EXTRA_MAPPINGS = {
    "Cardiac size/mediastinal contour evaluation": "Cardiovascular and Hemodynamic Conditions",
    "General intrathoracic pathology evaluation": "Thoracic Pathology Evaluation",
}


def main():
    # =================================================================
    # 1. Load all data
    # =================================================================
    print("Loading data...")
    with open(CLUSTERS_FILE) as f:
        clusters = json.load(f)
    with open(NOISE_ASSIGNMENTS_FILE) as f:
        noise_assignments = json.load(f)
    with open(SUMMARIES_FILE) as f:
        summaries = json.load(f)
    with open(TAXONOMY_FILE) as f:
        taxonomy = json.load(f)
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)
    with open(LOG_FILE) as f:
        log_entries = [json.loads(line) for line in f if line.strip()]
    with open(REASON_INDEX_FILE) as f:
        reason_index = json.load(f)
    with open(ID_TO_REASONS_FILE) as f:
        id_to_reasons = json.load(f)
    with open(COMPOUND_SPLIT_MAP_FILE) as f:
        compound_split_map = json.load(f)

    # =================================================================
    # 2. Build lookups
    # =================================================================
    print("Building lookups...")

    # old_cat_name -> final_cat_name
    old_name_to_final = {}
    for final_name, info in mapping.items():
        for orig_name in info["original_categories"]:
            old_name_to_final[orig_name] = final_name
    old_name_to_final.update(EXTRA_MAPPINGS)

    # cluster_id -> final_cat (via taxonomy -> mapping)
    cluster_to_final = {}
    for cat in taxonomy:
        final = old_name_to_final.get(cat["category_name"], cat["category_name"])
        for member in cat["members"]:
            cluster_to_final[str(member["cluster_id"])] = final

    # cluster_id -> image_inferable
    cluster_to_inferable = {}
    for entry in log_entries:
        cid = str(entry["cluster_id"])
        ri = entry.get("reason_info") or {}
        cluster_to_inferable[cid] = ri.get("image_inferable", "unknown")

    # cluster_id -> summary
    cluster_to_summary = {}
    for cid, info in summaries.items():
        cluster_to_summary[cid] = info.get("summary", "")

    # part_text -> cluster_id (from non-noise clusters)
    part_to_cluster = {}
    for cid, parts in clusters.items():
        if cid == "-1":
            continue
        for part in parts:
            part_to_cluster[part["part_text"]] = cid

    # Merge noise assignments: part_text -> cluster_id
    noise_part_set = set()
    for part_text, info in noise_assignments.items():
        assigned_cid = str(info["assigned_cluster"])
        if part_text not in part_to_cluster:
            part_to_cluster[part_text] = assigned_cid
            noise_part_set.add(part_text)

    print(f"  {len(cluster_to_final)} clusters with final category")
    print(f"  {len(part_to_cluster)} total parts mapped to clusters "
          f"({len(noise_part_set)} from noise)")

    # =================================================================
    # 3. Build reason_idx -> list of parts
    # =================================================================
    print("Building reason_idx -> parts mapping...")

    def get_parts_for_reason(ridx):
        """Get the atomic parts for a reason_idx."""
        text = reason_index[ridx]
        wc = len(text.split())
        idx_str = str(ridx)
        if wc >= WORD_COUNT_THRESHOLD and idx_str in compound_split_map:
            parts = [p.strip() for p in compound_split_map[idx_str] if p.strip()]
            return parts if parts else [text]
        return [text]

    # =================================================================
    # 4. Generate detailed output (per dataset:id)
    # =================================================================
    print(f"Generating {DETAILED_OUTPUT}...")
    id_to_categorized = {}
    id_to_summary = {}

    for dataset_id, reason_idxs in id_to_reasons.items():
        reasons_out = []
        all_final_cats = set()
        any_inferable = False

        for ridx in reason_idxs:
            reason_text = reason_index[ridx]
            parts_text = get_parts_for_reason(ridx)
            parts_out = []

            for pt in parts_text:
                cid = part_to_cluster.get(pt)
                from_noise = pt in noise_part_set

                if cid is not None:
                    final_cat = cluster_to_final.get(cid, "")
                    inferable = cluster_to_inferable.get(cid, "unknown")
                    summary = cluster_to_summary.get(cid, "")

                    if final_cat:
                        all_final_cats.add(final_cat)
                    if inferable == "yes":
                        any_inferable = True
                else:
                    final_cat = ""
                    inferable = "unknown"
                    summary = ""
                    from_noise = False

                parts_out.append({
                    "part_text": pt,
                    "cluster_id": int(cid) if cid is not None else None,
                    "cluster_summary": summary,
                    "final_category": final_cat,
                    "image_inferable": inferable,
                    "from_noise": from_noise,
                })

            reasons_out.append({
                "reason_idx": ridx,
                "reason_text": reason_text,
                "parts": parts_out,
            })

        id_to_categorized[dataset_id] = {"reasons": reasons_out}
        id_to_summary[dataset_id] = {
            "image_inferable": any_inferable,
            "final_categories": sorted(all_final_cats),
        }

    # =================================================================
    # 4b. Integrate augmented extras (e.g. Rex test)
    # =================================================================
    for extra in AUGMENTED_EXTRAS:
        if not os.path.exists(extra["file"]):
            print(f"\nSkipping augmented extra (not found yet): {extra['file']}")
            continue
        print(f"\nIntegrating augmented extra: {extra['file']}...")
        with open(extra["file"]) as f:
            extra_data = json.load(f)

        added = 0
        for entry in extra_data:
            reason_parts = entry.get("reason_parts")
            if not reason_parts:
                continue

            dataset_id = f"{extra['prefix']}:{extra['id_fn'](entry)}"
            all_final_cats = set()
            any_inferable = False

            reasons_out = []
            for rp in reason_parts:
                parts_out = []
                for part in rp.get("parts", []):
                    fc = part.get("final_category", "")
                    inf = part.get("image_inferable", "unknown")
                    if fc:
                        all_final_cats.add(fc)
                    if inf == "yes":
                        any_inferable = True
                    parts_out.append({
                        "part_text": part.get("part_text", ""),
                        "cluster_id": part.get("cluster_id"),
                        "cluster_summary": part.get("cluster_summary", ""),
                        "final_category": fc,
                        "image_inferable": inf,
                        "from_noise": part.get("from_noise", False),
                    })
                reasons_out.append({
                    "reason_idx": rp.get("reason_idx"),
                    "reason_text": rp.get("reason_text", ""),
                    "parts": parts_out,
                })

            id_to_categorized[dataset_id] = {"reasons": reasons_out}
            id_to_summary[dataset_id] = {
                "image_inferable": any_inferable,
                "final_categories": sorted(all_final_cats),
            }
            added += 1

        print(f"  Added {added} entries with prefix '{extra['prefix']}'")

    with open(DETAILED_OUTPUT, "w") as f:
        json.dump(id_to_categorized, f, ensure_ascii=False)

    # =================================================================
    # 5. Generate summary output
    # =================================================================
    print(f"Generating {SUMMARY_OUTPUT}...")
    with open(SUMMARY_OUTPUT, "w") as f:
        json.dump(id_to_summary, f, ensure_ascii=False)

    # =================================================================
    # 6. Stats
    # =================================================================
    total_ids = len(id_to_summary)
    ids_with_cats = sum(1 for v in id_to_summary.values() if v["final_categories"])
    ids_inferable = sum(1 for v in id_to_summary.values() if v["image_inferable"])

    cat_counts = defaultdict(int)
    for v in id_to_summary.values():
        for c in v["final_categories"]:
            cat_counts[c] += 1

    print(f"\n{'='*60}")
    print("OUTPUT SUMMARY")
    print(f"{'='*60}")
    print(f"\nFile 1: {DETAILED_OUTPUT}")
    print(f"  {total_ids} dataset:id entries")
    print(f"\nFile 2: {SUMMARY_OUTPUT}")
    print(f"  {total_ids} dataset:id entries")
    print(f"  {ids_with_cats} ({ids_with_cats/total_ids*100:.1f}%) have >= 1 final category")
    print(f"  {ids_inferable} ({ids_inferable/total_ids*100:.1f}%) have >= 1 image-inferable part")
    print(f"  {total_ids - ids_with_cats} ({(total_ids-ids_with_cats)/total_ids*100:.1f}%) uncategorized")
    print(f"\n  Final category distribution:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cnt:6d} ({cnt/total_ids*100:5.1f}%)  {cat}")


if __name__ == "__main__":
    main()
