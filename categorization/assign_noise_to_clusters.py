"""
Assign noise points (cluster -1) to the nearest non-noise cluster.

Strategy: For each noise point, compute cosine similarity to every cluster
centroid in the original 768-dim embedding space, then assign to the most
similar cluster.

Input:
  - sweep_result_clean_split/embeddings.npy
  - sweep_result_clean_split/<run>/clusters.json
  - sweep_result_clean_split/part_to_reason_idxs.json (for part ordering)

Output:
  - sweep_result_clean_split/<run>/clusters_full.json  (all points assigned)
  - sweep_result_clean_split/<run>/noise_assignments.json  (noise -> assigned cluster)
"""

import json
import os
import re
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

# =============================================================================
# Configuration
# =============================================================================
SWEEP_DIR = os.environ.get("SWEEP_DIR", "sweep_result_clean_split")
RUN = os.environ.get("RUN_NAME", "nn15_nc16_mcs100_ms50")
RUN_DIR = os.path.join(SWEEP_DIR, RUN)

EMBEDDINGS_FILE = os.path.join(SWEEP_DIR, "embeddings.npy")
CLUSTERS_FILE = os.path.join(RUN_DIR, "clusters.json")
SUMMARIES_FILE = os.path.join(RUN_DIR, "cluster_summaries.json")
PART_MAP_FILE = os.path.join(SWEEP_DIR, "part_to_reason_idxs.json")

OUTPUT_CLUSTERS = os.path.join(RUN_DIR, "clusters_full.json")
OUTPUT_ASSIGNMENTS = os.path.join(RUN_DIR, "noise_assignments.json")


def main():
    # 1. Load embeddings and part ordering
    print("Loading embeddings...")
    X = np.load(EMBEDDINGS_FILE)
    print(f"  Shape: {X.shape}")

    print("Loading part_to_reason_idxs (for part ordering)...")
    with open(PART_MAP_FILE) as f:
        part_to_reason_idxs = json.load(f)
    unique_parts = list(part_to_reason_idxs.keys())
    part_to_idx = {p: i for i, p in enumerate(unique_parts)}
    print(f"  {len(unique_parts)} parts")

    # 2. Load cluster assignments and summaries
    print("Loading clusters...")
    with open(CLUSTERS_FILE) as f:
        clusters = json.load(f)

    print("Loading cluster summaries...")
    with open(SUMMARIES_FILE) as f:
        summaries = json.load(f)
    cid_to_summary = {cid: info.get("summary", "") for cid, info in summaries.items()}

    noise_parts = clusters.get("-1", [])
    non_noise_cids = [cid for cid in clusters if cid != "-1"]
    print(f"  Non-noise clusters: {len(non_noise_cids)}")
    print(f"  Noise points: {len(noise_parts)}")

    if not noise_parts:
        print("No noise points to assign!")
        return

    # 3. Compute cluster centroids in embedding space
    print("Computing cluster centroids...")
    centroids = {}
    for cid in non_noise_cids:
        emb_indices = [part_to_idx[p["part_text"]] for p in clusters[cid]
                       if p["part_text"] in part_to_idx]
        if emb_indices:
            centroids[cid] = X[emb_indices].mean(axis=0)

    centroid_cids = list(centroids.keys())
    centroid_matrix = np.array([centroids[cid] for cid in centroid_cids])
    print(f"  {len(centroid_cids)} centroids computed")

    # 4. Assign each noise point to nearest centroid
    print("Assigning noise points...")

    # For similarity computation, replace placeholder patterns with actual words
    # so that e.g. "in a ___f" is encoded as "in a female". The recorded text
    # stays unchanged.
    PLACEHOLDER_SUBS = [
        (re.compile(r'___f\b'), 'female'),
        (re.compile(r'___m\b'), 'male'),
        (re.compile(r'___'), ''),          # other ___ placeholders: just remove
    ]

    def remap_text(text):
        out = text
        for pat, repl in PLACEHOLDER_SUBS:
            out = pat.sub(repl, out)
        return re.sub(r'\s+', ' ', out).strip()

    # Split noise parts into those that can use existing embeddings vs need re-encoding
    need_reencode = []  # (index_in_noise_parts, remapped_text)
    use_existing = []   # (index_in_noise_parts, embedding_idx)

    for idx, p in enumerate(noise_parts):
        pt = p["part_text"]
        remapped = remap_text(pt)
        if remapped != pt:
            # Text changed after substitution -> needs re-encoding
            need_reencode.append((idx, remapped))
        elif pt in part_to_idx:
            use_existing.append((idx, part_to_idx[pt]))

    # Re-encode remapped texts
    reencoded_embeddings = None
    if need_reencode:
        print(f"  Re-encoding {len(need_reencode)} noise parts with placeholder substitutions...")
        model = SentenceTransformer(MODEL_NAME)
        texts_to_encode = [t for _, t in need_reencode]
        reencoded_embeddings = model.encode(texts_to_encode, show_progress_bar=True,
                                            batch_size=512, normalize_embeddings=False)
        del model

    # Build combined noise embeddings in original order
    # Map: position_in_combined -> index_in_noise_parts
    combined_order = []
    combined_embeddings = []

    reencode_map = {idx: i for i, (idx, _) in enumerate(need_reencode)}
    existing_map = {idx: emb_idx for idx, emb_idx in use_existing}

    for idx in range(len(noise_parts)):
        if idx in reencode_map:
            combined_order.append(idx)
            combined_embeddings.append(reencoded_embeddings[reencode_map[idx]])
        elif idx in existing_map:
            combined_order.append(idx)
            combined_embeddings.append(X[existing_map[idx]])

    noise_embeddings = np.array(combined_embeddings) if combined_embeddings else np.empty((0, X.shape[1]))

    # Cosine similarity: (N_noise, N_clusters)
    sims = cosine_similarity(noise_embeddings, centroid_matrix)
    best_cluster_indices = sims.argmax(axis=1)
    best_sims = sims[np.arange(len(sims)), best_cluster_indices]

    # 5. Build output
    noise_assignments = {}
    clusters_full = {cid: list(parts) for cid, parts in clusters.items() if cid != "-1"}

    for pos, noise_idx in enumerate(combined_order):
        part = noise_parts[noise_idx]
        pt = part["part_text"]
        assigned_cid = centroid_cids[best_cluster_indices[pos]]
        sim_score = float(best_sims[pos])

        # Add to the assigned cluster (text stays as original)
        clusters_full.setdefault(assigned_cid, []).append({
            "part_text": pt,
            "reason_idxs": part["reason_idxs"],
            "x": part["x"],
            "y": part["y"],
            "from_noise": True,
            "similarity": round(sim_score, 4),
        })

        noise_assignments[pt] = {
            "assigned_cluster": int(assigned_cid),
            "cluster_summary": cid_to_summary.get(assigned_cid, ""),
            "similarity": round(sim_score, 4),
        }

    # Sort by cluster id
    clusters_full_sorted = {str(k): clusters_full[str(k)]
                            for k in sorted(int(c) for c in clusters_full)}

    # 6. Save
    print(f"Saving {OUTPUT_CLUSTERS}...")
    with open(OUTPUT_CLUSTERS, "w") as f:
        json.dump(clusters_full_sorted, f, ensure_ascii=False)

    print(f"Saving {OUTPUT_ASSIGNMENTS}...")
    with open(OUTPUT_ASSIGNMENTS, "w") as f:
        json.dump(noise_assignments, f, ensure_ascii=False)

    # 7. Stats
    assigned_sims = best_sims
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Noise points assigned: {len(noise_assignments)}")
    print(f"  Similarity to assigned centroid:")
    print(f"    Min:    {assigned_sims.min():.4f}")
    print(f"    25th:   {np.percentile(assigned_sims, 25):.4f}")
    print(f"    Median: {np.median(assigned_sims):.4f}")
    print(f"    75th:   {np.percentile(assigned_sims, 75):.4f}")
    print(f"    Max:    {assigned_sims.max():.4f}")

    # Cluster size comparison
    orig_sizes = {cid: len(parts) for cid, parts in clusters.items() if cid != "-1"}
    new_sizes = {cid: len(parts) for cid, parts in clusters_full_sorted.items()}
    total_orig = sum(orig_sizes.values())
    total_new = sum(new_sizes.values())
    print(f"\n  Original non-noise: {total_orig}")
    print(f"  After assignment:   {total_new}")
    print(f"  Increase:           {total_new - total_orig} ({(total_new-total_orig)/total_orig*100:.1f}%)")

    # Distribution of noise assignments across clusters
    assign_counts = defaultdict(int)
    for info in noise_assignments.values():
        assign_counts[info["assigned_cluster"]] += 1
    top_receivers = sorted(assign_counts.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 clusters receiving noise points:")
    for cid, cnt in top_receivers:
        print(f"    Cluster {cid}: +{cnt} (was {orig_sizes.get(str(cid), 0)}, now {new_sizes.get(str(cid), 0)})")


if __name__ == "__main__":
    main()
