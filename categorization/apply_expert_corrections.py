"""
Apply the medical-physicist corrections to the categorized reasons produced by
``generate_final_outputs.py`` and emit the public-facing NVCC taxonomy file
``RadSLDP_nvcc_taxonomy.json`` consumed by ``training/hf_finetune.py``.

Of the 347 GPT-annotated cluster summaries, five were re-labeled by a certified
medical physicist (see paper §2.2). This script flips ``image_inferable`` from
``"no"`` to ``"yes"`` on every reason whose ``cluster_summary`` falls in
``TARGET_SUMMARIES`` and writes the corrected file.

Usage:
    python3 apply_expert_corrections.py            # uses default paths
    INPUT=... OUTPUT=... python3 apply_expert_corrections.py
"""

import json
import os
import sys

# Default INPUT matches the output location of generate_final_outputs.py
# (Step 8) under the env-var defaults documented in README.md. Run from the
# working directory that contains original_data/ and sweep_result_clean_split/.
SWEEP_DIR = os.environ.get("SWEEP_DIR", "sweep_result_clean_split")
RUN_NAME = os.environ.get("RUN_NAME", "nn15_nc16_mcs100_ms50")
CAT_NAME = os.environ.get("CAT_NAME", "categorize_result_5.2")
INPUT = os.environ.get(
    "INPUT",
    os.path.join(SWEEP_DIR, RUN_NAME, CAT_NAME, "id_to_categorized_reasons.json"),
)

# Default OUTPUT writes alongside this script (categorization/) so that
# training/hf_finetune.py can read ../categorization/RadSLDP_nvcc_taxonomy.json.
OUTPUT = os.environ.get(
    "OUTPUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "RadSLDP_nvcc_taxonomy.json"),
)

# Cluster summaries the physicist flipped from image_inferable=no to yes.
TARGET_SUMMARIES = {
    "Antibiotic treatment status or response",
    "Chest tube clamping status or trial",
    "Evaluation for acute cardiopulmonary abnormalities or changes",
    "Evaluation for acute or chronic intrathoracic pathology or abnormalities",
    "Evaluation for sarcoidosis or granulomatous disease",
}


def main():
    if not os.path.exists(INPUT):
        sys.exit(
            f"ERROR: input file not found: {INPUT}\n"
            "Run generate_final_outputs.py first, or set INPUT=<path>."
        )

    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_modified = 0
    for patient_data in data.values():
        for reason in patient_data.get("reasons", []):
            for part in reason.get("parts", []):
                if part.get("cluster_summary") in TARGET_SUMMARIES:
                    if part.get("image_inferable") != "yes":
                        part["image_inferable"] = "yes"
                        n_modified += 1

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Corrected {n_modified} entries across {len(TARGET_SUMMARIES)} cluster summaries.")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
