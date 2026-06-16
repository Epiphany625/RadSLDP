"""
Build cluster_details.json (the input to summarize_clustering.py / Step 4).

This file was originally produced ad-hoc and the generating script was not kept.
It is fully reconstructable from two files written by
encode_and_clustering_clean_split.py (Step 3):

  - clusters.json               {cluster_id: [{part_text, reason_idxs, x, y}, ...]}
  - cluster_representatives.json {cluster_id: representative_part_text}

For each non-noise cluster it emits:
  {
    "<cluster_id>": {
      "representative": <rep text from cluster_representatives.json>,
      "samples":        [up to N_SAMPLES member part_texts, excluding the rep],
      "cluster_size":   <number of members in the cluster>
    },
    ...
  }

The sample set is a random draw of the cluster members (the representative is
excluded from the pool, matching the original artifact in which the rep never
appeared in `samples`). A fixed seed makes the draw reproducible; the exact
membership of the sample list does not affect downstream summarization quality.

Verified against the released artifact
(sweep_result_clean_split/nn15_nc16_mcs100_ms50/cluster_details.json):
representative, cluster_size match exactly for all 349 clusters and every
sample is a genuine cluster member.

Usage:
    python categorization/generate_cluster_details.py
    python categorization/generate_cluster_details.py <run_dir>

Config (env vars, same convention as the other scripts):
    SWEEP_DIR  default "sweep_result_clean_split"
    RUN_NAME   default "nn15_nc16_mcs100_ms50"
    N_SAMPLES  default 30
    SEED       default 42
"""

import json
import os
import random
import sys


def main():
    sweep_dir = os.environ.get("SWEEP_DIR", "sweep_result_clean_split")
    run_name = os.environ.get("RUN_NAME", "nn15_nc16_mcs100_ms50")
    n_samples = int(os.environ.get("N_SAMPLES", "30"))
    seed = int(os.environ.get("SEED", "42"))

    # Optional positional override: path to the run directory
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(sweep_dir, run_name)

    clusters_path = os.path.join(run_dir, "clusters.json")
    reps_path = os.path.join(run_dir, "cluster_representatives.json")
    out_path = os.path.join(run_dir, "cluster_details.json")

    print(f"Loading {clusters_path}...")
    with open(clusters_path) as f:
        clusters = json.load(f)
    print(f"Loading {reps_path}...")
    with open(reps_path) as f:
        reps = json.load(f)

    rng = random.Random(seed)

    details = {}
    for cid, members in clusters.items():
        if cid == "-1":  # noise cluster is not summarized
            continue
        member_texts = [p["part_text"] for p in members]
        rep_text = reps.get(cid, reps.get(str(cid), ""))

        others = [t for t in member_texts if t != rep_text]
        k = min(n_samples, len(others))
        samples = rng.sample(others, k)

        details[cid] = {
            "representative": rep_text,
            "samples": samples,
            "cluster_size": len(member_texts),
        }

    print(f"Writing {out_path} ({len(details)} clusters)...")
    with open(out_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print("Done.")


if __name__ == "__main__":
    main()
