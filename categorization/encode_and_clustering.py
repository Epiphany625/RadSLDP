import json
import re
import numpy as np
import pandas as pd
# import umap
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class ReasonNormalizer:
    """Normalizes reason text to handle variations"""

    def __init__(self):
        # Abbreviation mappings
        self.abbreviations = {
            r'\bpna\b': 'pneumonia',
            r'\bptx\b': 'pneumothorax',
            r'\bsob\b': 'shortness of breath',
            r'\bdoe\b': 'dyspnea on exertion',
            r'\bchf\b': 'congestive heart failure',
            r'\bams\b': 'altered mental status',
            r'\br/o\b': 'rule out',
            r'\beval\b': 'evaluate',
            r'\bs/p\b': 'status post',
            r'\bng\b': 'nasogastric',
            r'\bet\b': 'endotracheal',
            r'\bicu\b': 'intensive care unit',
            r'\buti\b': 'urinary tract infection',
            r'\bcxr\b': 'chest x-ray',
            r'\bcopd\b': 'chronic obstructive pulmonary disease',
            r'\bmi\b': 'myocardial infarction',
            r'\bcad\b': 'coronary artery disease',
            r'\bcp\b': 'chest pain'
        }

    def normalize(self, reason: str) -> str:
        """Apply all normalization steps to a reason string"""
        if not reason:
            return ""

        # Convert to lowercase
        normalized = reason.lower()

        # Remove "History:" prefix
        normalized = re.sub(r'^history:\s*', '', normalized)

        # add a ; after the following patterns to separate them from the rest of the text
        patterns = [
        r'___-year-old\s+(male|female|man|woman|m|f)',
        r'___\s*(year old|y\.?o\.?)\s+(male|female|man|woman|m|f)',
        r'\d+-year-old\s+(male|female|man|woman)',
        r'\d+\s*-\s*year[- ]?old\s+(male|female|man|woman|m|f)',
        r'\d+\s*(year[- ]?old|y\.?o\.?)\s+(male|female|man|woman|m|f)',
        r'\d+\s*years\s*(male|female|man|woman|m|f)',
        r'\d+\s*y\/?o\s*(male|female|man|woman|m|f)',
        r'\b\d+\s*-\s*year[- ]?old\b',
        r'___f|___m\b',
        ]
        for p in patterns:
            normalized = re.sub(p, r'\g<0>;', normalized, flags=re.IGNORECASE)

        # Expand abbreviations
        for abbrev, full in self.abbreviations.items():
            normalized = re.sub(abbrev, full, normalized)

        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove common separators and trailing punctuation
        normalized = re.sub(r'\s*//\s*', ', ', normalized)
        normalized = re.sub(r'[\.;]+$', '', normalized)

        # Clean up extra spaces around punctuation
        normalized = re.sub(r'\s*,\s*', ', ', normalized)
        normalized = normalized.strip()

        return normalized
#get all the reason field from the merged results data
with open("batch_outputs/merged_results.json", "r") as f:
    merged_results = json.load(f)

rex_reasons = []
for entry in merged_results.values():
    for part in entry.get("normalized_parts", []):
        text = part.get("text", "")
        if text:
            rex_reasons.append(text)
print("Distinct rex reasons and their counts:")
normalizer = ReasonNormalizer()
# split them by "." and ",", and normalize them using the ReasonNormalizer
normalized_reasons = []
for reason in tqdm(rex_reasons, desc="Normalizing reasons"):
    normalized_reason = normalizer.normalize(reason.strip())
    split_reasons = re.split(r'[.;,]', normalized_reason)
    for r in split_reasons:
        if r.strip() != "":
            normalized_reasons.append(r.strip())
# count the occurrences of each normalized reason
rex_reason_counts = Counter(normalized_reasons)
# for reason, count in rex_reason_counts.most_common():
#     print(f"{reason}: {count}")

#load the chexpert plus normalized reasons and add to the normalized reasons list
with open("chexpert_plus_reason_counts.json", "r") as f:
    chexplus_reason_counts = json.load(f)
for reason, count in chexplus_reason_counts:
    normalized_reasons.append(reason)

#load rex distinct normalized reasons and their counts, and add to the normalized reasons list
with open("rex_distinct_normalized_reasons.json", "r") as f:
    rex_distinct_reasons_dict = json.load(f)
for reason, count in rex_distinct_reasons_dict:
    normalized_reasons.append(reason)

#dump them in to a json file and ordered by count
with open("all_normalized_reasons.json", "w") as f:
    json.dump(Counter(normalized_reasons).most_common(), f, indent=4)

normalized_reasons = list(set(normalized_reasons))
print(f"Total distinct normalized reasons: {len(normalized_reasons)}")
# Load SentenceTransformer model (same base model as before)
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
model = SentenceTransformer(MODEL_NAME)

# Embed the distinct normalized reasons
# rex_distinct_reasons = list(rex_reason_counts.keys())
# rex_reason_embeddings = model.encode(
#     rex_distinct_reasons,
#     batch_size=32,
#     show_progress_bar=True,
#     convert_to_numpy=True
# )
# print("Rex reason embeddings shape:", rex_reason_embeddings.shape)
X = model.encode(normalized_reasons, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import os

# =============================================================================
# Parameter grid
# =============================================================================
UMAP_N_NEIGHBORS = [10, 15, 30]
UMAP_N_COMPONENTS = [5, 8, 16]
HDBSCAN_MIN_CLUSTER_SIZE = [30, 50, 100]
HDBSCAN_MIN_SAMPLES = [10, 25, 50]

SWEEP_DIR = "sweep_results_all_normalized_reasons"
os.makedirs(SWEEP_DIR, exist_ok=True)

# Detect GPU availability once
USE_GPU = False
try:
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    import cudf
    USE_GPU = True
    print("GPU acceleration available (cuml)")
except ImportError:
    import umap
    import hdbscan
    print("cuml not available, using CPU (umap + hdbscan)")


def compute_metrics(X_orig, Z, labels, clusterer):
    """Compute clustering quality metrics.

    Args:
        X_orig: original high-dim embeddings (N, D)
        Z: UMAP-reduced embeddings (N, d)
        labels: cluster labels (N,), -1 = noise
        clusterer: fitted HDBSCAN object
    Returns:
        dict of metric name -> value
    """
    metrics = {}

    # --- 1. Number of clusters ---
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    metrics["n_clusters"] = n_clusters

    # --- 2. Noise ratio ---
    metrics["noise_ratio"] = float((labels == -1).mean())

    # --- 3. Cluster size statistics ---
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

    # --- 4. HDBSCAN cluster persistence (CPU hdbscan only) ---
    if hasattr(clusterer, "cluster_persistence_"):
        persistence = clusterer.cluster_persistence_
        metrics["mean_persistence"] = float(np.mean(persistence))
        metrics["min_persistence"] = float(np.min(persistence))
    else:
        metrics["mean_persistence"] = None
        metrics["min_persistence"] = None

    # --- 5. Silhouette score (on UMAP space, excluding noise) ---
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        metrics["silhouette"] = float(
            silhouette_score(Z[mask], labels[mask], sample_size=min(10000, mask.sum()))
        )
    else:
        metrics["silhouette"] = None

    # --- 6. Intra-cluster cosine coherence (on original embeddings) ---
    coherence_scores = []
    for c in unique_labels:
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        cluster_embs = X_orig[idx]
        sim_matrix = cosine_similarity(cluster_embs)
        # mean of upper triangle (excluding diagonal)
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
# Parameter sweep
# =============================================================================
all_results = []

param_grid = list(itertools.product(
    UMAP_N_NEIGHBORS, UMAP_N_COMPONENTS,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES
))
print(f"\nTotal parameter combinations: {len(param_grid)}")

for i, (nn, nc, mcs, ms) in tqdm(enumerate(param_grid), total=len(param_grid), desc="Parameter sweep"):
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
        Z = Z.to_numpy()  # cuml returns cupy/cudf

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

    # --- Save per-run outputs to subdirectory ---
    run_dir = os.path.join(SWEEP_DIR, f"nn{nn}_nc{nc}_mcs{mcs}_ms{ms}")
    os.makedirs(run_dir, exist_ok=True)

    # Cluster assignments
    df = pd.DataFrame({
        "text": normalized_reasons,
        "x": Z[:, 0],
        "y": Z[:, 1],
        "cluster": labels
    })
    df = df.sort_values(by=["cluster"])
    df.to_json(os.path.join(run_dir, "clusters.json"), orient="records", lines=True)

    # Cluster representatives (closest to cluster center in first 2 UMAP dims)
    representatives = {}
    for c, rows in df[df.cluster != -1].groupby("cluster"):
        pts = rows[["x", "y"]].values
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        representatives[c] = rows.iloc[np.argmin(dists)]["text"]

    with open(os.path.join(run_dir, "cluster_representatives.json"), "w") as f:
        json.dump(representatives, f, indent=4)

    # Metrics for this run
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved to {run_dir}/")

# =============================================================================
# Summary table
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
with open("param_sweep_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSweep summary saved to param_sweep_results.json")
print(f"Per-run outputs saved under {SWEEP_DIR}/")

# =============================================================================
# Best parameters by silhouette score
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
    run_dir = os.path.join(SWEEP_DIR, f"nn{best['umap_n_neighbors']}_nc{best['umap_n_components']}"
              f"_mcs{best['hdbscan_min_cluster_size']}_ms{best['hdbscan_min_samples']}")
    print(f"  results dir        = {run_dir}/")

# =============================================================================
# Best parameters by average of persistence and silhouette score
# =============================================================================
candidates_both = [r for r in all_results
                   if r["silhouette"] is not None and r["mean_persistence"] is not None]
if candidates_both:
    # Min-max normalize both metrics across all candidates so they contribute equally
    sil_vals = [r["silhouette"] for r in candidates_both]
    per_vals = [r["mean_persistence"] for r in candidates_both]
    sil_min, sil_max = min(sil_vals), max(sil_vals)
    per_min, per_max = min(per_vals), max(per_vals)

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    for r in candidates_both:
        r["_norm_sil"] = norm(r["silhouette"], sil_min, sil_max)
        r["_norm_per"] = norm(r["mean_persistence"], per_min, per_max)
        r["_combined"] = (r["_norm_sil"] + r["_norm_per"]) / 2.0

    best_combined = max(candidates_both, key=lambda r: r["_combined"])

    print(f"\n{'=' * 120}")
    print("BEST PARAMETERS (highest avg of normalized persistence + silhouette)")
    print(f"{'=' * 120}")
    print(f"  UMAP:    n_neighbors={best_combined['umap_n_neighbors']}, "
          f"n_components={best_combined['umap_n_components']}")
    print(f"  HDBSCAN: min_cluster_size={best_combined['hdbscan_min_cluster_size']}, "
          f"min_samples={best_combined['hdbscan_min_samples']}")
    print(f"  ---")
    print(f"  combined_score     = {best_combined['_combined']:.4f}")
    print(f"  silhouette         = {best_combined['silhouette']:.4f}  (normalized: {best_combined['_norm_sil']:.4f})")
    print(f"  mean_persistence   = {best_combined['mean_persistence']:.4f}  (normalized: {best_combined['_norm_per']:.4f})")
    print(f"  n_clusters         = {best_combined['n_clusters']}")
    print(f"  noise_ratio        = {best_combined['noise_ratio']:.4f}")
    coh = f"{best_combined['mean_cosine_coherence']:.4f}" if best_combined['mean_cosine_coherence'] is not None else "N/A"
    print(f"  cosine_coherence   = {coh}")
    print(f"  cluster_size       = min={best_combined['cluster_size_min']}, "
          f"median={best_combined['cluster_size_median']:.0f}, max={best_combined['cluster_size_max']}")
    run_dir = os.path.join(SWEEP_DIR, f"nn{best_combined['umap_n_neighbors']}_nc{best_combined['umap_n_components']}"
              f"_mcs{best_combined['hdbscan_min_cluster_size']}_ms{best_combined['hdbscan_min_samples']}")
    print(f"  results dir        = {run_dir}/")
