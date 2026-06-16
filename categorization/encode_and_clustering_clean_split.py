"""
Encode and cluster the clean-split reason parts.

Pipeline:
1. Load reason_index.json and compound_split_map.json
2. Build atomic parts:
   - word_count < 6 → use the original reason text
   - word_count >= 6 → use split parts from compound_split_map
     (or original text if not in the map)
3. Deduplicate: many reasons split to the same part.
   Save part_to_reason_idxs.json: {part_text -> [reason_idx, ...]}
   Only cluster each unique part once.
4. Encode with SentenceTransformer, run UMAP + HDBSCAN sweep
5. Save results to sweep_result_clean_split/

Outputs per sweep run (in sweep_result_clean_split/nn{}_nc{}_mcs{}_ms{}/):
  - clusters.json   (JSONL: {part_text, reason_idxs, x, y, cluster})
  - cluster_representatives.json
  - metrics.json

Global outputs (in sweep_result_clean_split/):
  - part_to_reason_idxs.json
  - param_sweep_results.json
"""

import json
import os
import itertools
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = "original_data"
REASON_INDEX_FILE = os.path.join(BASE_DIR, "reason_index.json")
SPLIT_MAP_FILE = os.path.join(BASE_DIR, "compound_split_map.json")

SWEEP_DIR = "sweep_result_clean_split"
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

WORD_COUNT_THRESHOLD = 6  # reasons with >= this many words were sent for splitting

UMAP_N_NEIGHBORS = [10, 15, 30]
UMAP_N_COMPONENTS = [5, 8, 16]
HDBSCAN_MIN_CLUSTER_SIZE = [30, 50, 100]
HDBSCAN_MIN_SAMPLES = [10, 25, 50]

# =============================================================================
# 1. Load data
# =============================================================================
print("Loading reason_index.json...")
with open(REASON_INDEX_FILE) as f:
    reason_index = json.load(f)
print(f"  {len(reason_index)} reasons")

print("Loading compound_split_map.json...")
with open(SPLIT_MAP_FILE) as f:
    compound_split_map = json.load(f)
print(f"  {len(compound_split_map)} reasons were split")

# =============================================================================
# 2. Build atomic parts and deduplicate
# =============================================================================
print("\nBuilding atomic parts...")
# part_text -> set of source reason_idxs
part_to_reason_idxs = defaultdict(set)

for idx, text in enumerate(reason_index):
    wc = len(text.split())
    idx_str = str(idx)
    if wc >= WORD_COUNT_THRESHOLD and idx_str in compound_split_map:
        # Use the split parts
        for part in compound_split_map[idx_str]:
            part = part.strip()
            if part:
                part_to_reason_idxs[part].add(idx)
    else:
        # Use the original text (short reasons or not in split map)
        part_to_reason_idxs[text].add(idx)

# Convert sets to sorted lists for JSON serialization
part_to_reason_idxs_list = {
    part: sorted(idxs) for part, idxs in part_to_reason_idxs.items()
}

unique_parts = list(part_to_reason_idxs_list.keys())
print(f"  Total unique parts for clustering: {len(unique_parts)}")
multi_source = sum(1 for v in part_to_reason_idxs_list.values() if len(v) > 1)
print(f"  Parts shared by >1 source reason: {multi_source}")

# Save part_to_reason_idxs mapping
os.makedirs(SWEEP_DIR, exist_ok=True)
part_map_path = os.path.join(SWEEP_DIR, "part_to_reason_idxs.json")
print(f"Saving {part_map_path}...")
with open(part_map_path, "w") as f:
    json.dump(part_to_reason_idxs_list, f, ensure_ascii=False)

# =============================================================================
# 3. Encode
# =============================================================================
print(f"\nEncoding {len(unique_parts)} parts with {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
X = model.encode(unique_parts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
print(f"  Embedding shape: {X.shape}")

# Save embeddings for reuse
emb_path = os.path.join(SWEEP_DIR, "embeddings.npy")
np.save(emb_path, X)
print(f"  Saved embeddings to {emb_path}")

# =============================================================================
# 4. Detect GPU
# =============================================================================
USE_GPU = False
try:
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    import cudf
    USE_GPU = True
    print("\nGPU acceleration available (cuml)")
except ImportError:
    import umap
    import hdbscan
    print("\ncuml not available, using CPU (umap + hdbscan)")


# =============================================================================
# 5. Metrics
# =============================================================================
def compute_metrics(X_orig, Z, labels, clusterer):
    metrics = {}

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    metrics["n_clusters"] = n_clusters
    metrics["noise_ratio"] = float((labels == -1).mean())

    cluster_sizes = Counter(l for l in labels if l != -1)
    if cluster_sizes:
        sizes = list(cluster_sizes.values())
        metrics["cluster_size_min"] = min(sizes)
        metrics["cluster_size_median"] = float(np.median(sizes))
        metrics["cluster_size_max"] = max(sizes)
    else:
        metrics["cluster_size_min"] = 0
        metrics["cluster_size_median"] = 0
        metrics["cluster_size_max"] = 0

    if hasattr(clusterer, "cluster_persistence_"):
        persistence = clusterer.cluster_persistence_
        metrics["mean_persistence"] = float(np.mean(persistence))
        metrics["min_persistence"] = float(np.min(persistence))
    else:
        metrics["mean_persistence"] = None
        metrics["min_persistence"] = None

    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        metrics["silhouette"] = float(
            silhouette_score(Z[mask], labels[mask], sample_size=min(10000, mask.sum()))
        )
    else:
        metrics["silhouette"] = None

    coherence_scores = []
    for c in unique_labels:
        if c == -1:
            continue
        cidx = np.where(labels == c)[0]
        if len(cidx) < 2:
            continue
        cluster_embs = X_orig[cidx]
        sim_matrix = cosine_similarity(cluster_embs)
        n = sim_matrix.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        mean_sim = sim_matrix[triu_idx].mean()
        coherence_scores.append(mean_sim)
    if coherence_scores:
        metrics["mean_cosine_coherence"] = float(np.mean(coherence_scores))
        metrics["std_cosine_coherence"] = float(np.std(coherence_scores))
        metrics["min_cosine_coherence"] = float(np.min(coherence_scores))
    else:
        metrics["mean_cosine_coherence"] = None
        metrics["std_cosine_coherence"] = None
        metrics["min_cosine_coherence"] = None

    return metrics


# =============================================================================
# 6. Parameter sweep
# =============================================================================
# Build reason_idxs array aligned with unique_parts for output
reason_idxs_per_part = [part_to_reason_idxs_list[p] for p in unique_parts]

all_results = []
param_grid = list(itertools.product(
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES
))
print(f"\nTotal parameter combinations: {len(param_grid)}")

for i, (nn, nc, mcs, ms) in tqdm(enumerate(param_grid), total=len(param_grid), desc="Parameter sweep"):
    run_dir = os.path.join(SWEEP_DIR, f"nn{nn}_nc{nc}_mcs{mcs}_ms{ms}")

    # Skip if already done
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            result = json.load(f)
        all_results.append(result)
        continue

    print(f"\n[{i+1}/{len(param_grid)}] "
          f"UMAP(n_neighbors={nn}, n_components={nc}) | "
          f"HDBSCAN(min_cluster_size={mcs}, min_samples={ms})")

    # --- UMAP ---
    if USE_GPU:
        reducer = cuUMAP(
            n_neighbors=nn, n_components=nc,
            min_dist=0.0, metric="cosine", random_state=42
        )
    else:
        reducer = umap.UMAP(
            n_neighbors=nn, n_components=nc,
            min_dist=0.0, metric="cosine", random_state=42
        )
    Z = reducer.fit_transform(X)
    if hasattr(Z, 'to_numpy'):
        Z = Z.to_numpy()

    # --- HDBSCAN ---
    if USE_GPU:
        clusterer = cuHDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            metric="euclidean", cluster_selection_method="eom"
        )
        Z_gpu = cudf.DataFrame(Z)
        labels = clusterer.fit_predict(Z_gpu)
        if hasattr(labels, 'to_numpy'):
            labels = labels.to_numpy()
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            metric="euclidean", cluster_selection_method="eom"
        )
        labels = clusterer.fit_predict(Z)

    # --- Metrics ---
    metrics = compute_metrics(X, Z, labels, clusterer)
    params = {
        "umap_n_neighbors": nn,
        "umap_n_components": nc,
        "hdbscan_min_cluster_size": mcs,
        "hdbscan_min_samples": ms,
    }
    result = {**params, **metrics}
    all_results.append(result)

    print(f"  n_clusters={metrics['n_clusters']}, "
          f"noise={metrics['noise_ratio']:.3f}, "
          f"silhouette={metrics['silhouette']}, "
          f"coherence={metrics['mean_cosine_coherence']}")

    # --- Save per-run outputs ---
    os.makedirs(run_dir, exist_ok=True)

    # clusters.json: {cluster_id: [{part_text, reason_idxs, x, y}, ...]}
    clusters_by_id = defaultdict(list)
    for j in range(len(unique_parts)):
        clusters_by_id[int(labels[j])].append({
            "part_text": unique_parts[j],
            "reason_idxs": reason_idxs_per_part[j],
            "x": float(Z[j, 0]),
            "y": float(Z[j, 1]),
        })
    # Sort by cluster id; noise (-1) comes first
    clusters_sorted = {str(k): clusters_by_id[k] for k in sorted(clusters_by_id)}
    clusters_path = os.path.join(run_dir, "clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(clusters_sorted, f, ensure_ascii=False)

    # Cluster representatives
    df = pd.DataFrame({
        "text": unique_parts,
        "x": Z[:, 0],
        "y": Z[:, 1],
        "cluster": labels
    })
    representatives = {}
    for c, rows in df[df.cluster != -1].groupby("cluster"):
        pts = rows[["x", "y"]].values
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        representatives[int(c)] = rows.iloc[np.argmin(dists)]["text"]

    with open(os.path.join(run_dir, "cluster_representatives.json"), "w") as f:
        json.dump(representatives, f, indent=4, ensure_ascii=False)

    # Metrics
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved to {run_dir}/")

# =============================================================================
# 7. Summary table
# =============================================================================
print("\n" + "=" * 120)
print("PARAMETER SWEEP RESULTS")
print("=" * 120)
header = (f"{'nn':>4} {'nc':>4} {'mcs':>5} {'ms':>4} | "
          f"{'#clust':>6} {'noise%':>7} {'sil':>7} "
          f"{'cos_coh':>8} {'cos_std':>8} {'persist':>8} | "
          f"{'sz_min':>6} {'sz_med':>7} {'sz_max':>6}")
print(header)
print("-" * 120)
for r in all_results:
    sil = f"{r['silhouette']:.4f}" if r['silhouette'] is not None else "   N/A"
    coh = f"{r['mean_cosine_coherence']:.4f}" if r['mean_cosine_coherence'] is not None else "   N/A"
    coh_std = f"{r['std_cosine_coherence']:.4f}" if r['std_cosine_coherence'] is not None else "   N/A"
    per = f"{r['mean_persistence']:.4f}" if r['mean_persistence'] is not None else "   N/A"
    print(f"{r['umap_n_neighbors']:>4} {r['umap_n_components']:>4} "
          f"{r['hdbscan_min_cluster_size']:>5} {r['hdbscan_min_samples']:>4} | "
          f"{r['n_clusters']:>6} {r['noise_ratio']:>7.3f} {sil:>7} "
          f"{coh:>8} {coh_std:>8} {per:>8} | "
          f"{r['cluster_size_min']:>6} {r['cluster_size_median']:>7.0f} {r['cluster_size_max']:>6}")

# Save full sweep summary
sweep_results_path = os.path.join(SWEEP_DIR, "param_sweep_results.json")
with open(sweep_results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSweep summary saved to {sweep_results_path}")

# =============================================================================
# 8. Best parameters
# =============================================================================
candidates = [r for r in all_results if r["silhouette"] is not None]
if candidates:
    best = max(candidates, key=lambda r: r["silhouette"])
    print(f"\n{'=' * 120}")
    print("BEST PARAMETERS (highest silhouette score)")
    print(f"{'=' * 120}")
    print(f"  UMAP:    n_neighbors={best['umap_n_neighbors']}, "
          f"n_components={best['umap_n_components']}")
    print(f"  HDBSCAN: min_cluster_size={best['hdbscan_min_cluster_size']}, "
          f"min_samples={best['hdbscan_min_samples']}")
    print(f"  ---")
    print(f"  n_clusters         = {best['n_clusters']}")
    print(f"  noise_ratio        = {best['noise_ratio']:.4f}")
    print(f"  silhouette         = {best['silhouette']:.4f}")
    coh = f"{best['mean_cosine_coherence']:.4f}" if best['mean_cosine_coherence'] is not None else "N/A"
    per = f"{best['mean_persistence']:.4f}" if best['mean_persistence'] is not None else "N/A"
    print(f"  cosine_coherence   = {coh}")
    print(f"  mean_persistence   = {per}")
    print(f"  cluster_size       = min={best['cluster_size_min']}, "
          f"median={best['cluster_size_median']:.0f}, max={best['cluster_size_max']}")

candidates_both = [r for r in all_results
                   if r["silhouette"] is not None and r["mean_persistence"] is not None]
if candidates_both:
    sil_vals = [r["silhouette"] for r in candidates_both]
    per_vals = [r["mean_persistence"] for r in candidates_both]
    sil_min, sil_max = min(sil_vals), max(sil_vals)
    per_min, per_max = min(per_vals), max(per_vals)

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    for r in candidates_both:
        r["_combined"] = (norm(r["silhouette"], sil_min, sil_max) +
                          norm(r["mean_persistence"], per_min, per_max)) / 2.0

    best_c = max(candidates_both, key=lambda r: r["_combined"])
    print(f"\n{'=' * 120}")
    print("BEST PARAMETERS (highest avg of normalized persistence + silhouette)")
    print(f"{'=' * 120}")
    print(f"  UMAP:    n_neighbors={best_c['umap_n_neighbors']}, "
          f"n_components={best_c['umap_n_components']}")
    print(f"  HDBSCAN: min_cluster_size={best_c['hdbscan_min_cluster_size']}, "
          f"min_samples={best_c['hdbscan_min_samples']}")
    print(f"  combined_score     = {best_c['_combined']:.4f}")
    print(f"  silhouette         = {best_c['silhouette']:.4f}")
    print(f"  mean_persistence   = {best_c['mean_persistence']:.4f}")
    print(f"  n_clusters         = {best_c['n_clusters']}")
    print(f"  noise_ratio        = {best_c['noise_ratio']:.4f}")
