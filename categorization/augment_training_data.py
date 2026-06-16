"""
Augment original training data files with categorized reason parts.

For each sample in the 6 source files, adds:
- reason_parts: [{reason_text, parts: [{part_text, cluster_id, cluster_summary,
                   final_category, image_inferable, from_noise}]}]
- final_categories: [str, ...] (unique final categories)
- has_image_inferable: bool

For parts already seen in training, uses the pre-computed cluster mapping.
For new parts, encodes them with SentenceTransformer and assigns to the
nearest cluster centroid via cosine similarity.

Input:
  - 6 original dataset files in original_data/
  - id_to_categorized_reasons.json (pre-computed for MIMIC train, CheXpert+, Rex train)
  - Lookup data for on-the-fly computation (for MIMIC dev/test, Rex test)

Output (in original_data/augmented/):
  1. chat_train_MIMIC_CXR_augmented.json
  2. chat_dev_MIMIC_CXR_augmented.json
  3. chat_test_MIMIC_CXR_augmented.json
  4. df_chexpert_plus_augmented.json
  5. rexgradient_train_augmented.json
  6. rexgradient_test_augmented.json
"""

import json
import os
import re
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = os.environ.get("DATA_DIR", "original_data")
SWEEP_DIR = os.environ.get("SWEEP_DIR", "sweep_result_clean_split")
RUN_NAME = os.environ.get("RUN_NAME", "nn15_nc16_mcs100_ms50")
CAT_NAME = os.environ.get("CAT_NAME", "categorize_result_5.2")

ORIGINAL_DATA = DATA_DIR
RUN_DIR = os.path.join(SWEEP_DIR, RUN_NAME)
CAT_DIR = os.path.join(RUN_DIR, CAT_NAME)

# Pre-computed categorization
ID_TO_CATEGORIZED_FILE = f"{CAT_DIR}/id_to_categorized_reasons.json"

# Lookup files for on-the-fly computation
REASON_INDEX_FILE = f"{ORIGINAL_DATA}/reason_index.json"
COMPOUND_SPLIT_MAP_FILE = f"{ORIGINAL_DATA}/compound_split_map.json"
CLUSTERS_FILE = f"{RUN_DIR}/clusters.json"
NOISE_ASSIGNMENTS_FILE = f"{RUN_DIR}/noise_assignments.json"
SUMMARIES_FILE = f"{RUN_DIR}/cluster_summaries.json"
TAXONOMY_FILE = f"{CAT_DIR}/category_taxonomy.json"
MAPPING_FILE = f"{CAT_DIR}/category_mapping.json"
LOG_FILE = f"{CAT_DIR}/categorize_log.jsonl"
EMBEDDINGS_FILE = f"{SWEEP_DIR}/embeddings.npy"
PART_MAP_FILE = f"{SWEEP_DIR}/part_to_reason_idxs.json"

OUTPUT_DIR = f"{ORIGINAL_DATA}/augmented"

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

WORD_COUNT_THRESHOLD = 6

EXTRA_MAPPINGS = {
    "Cardiac size/mediastinal contour evaluation": "Cardiovascular and Hemodynamic Conditions",
    "General intrathoracic pathology evaluation": "Thoracic Pathology Evaluation",
}


# =============================================================================
# ReasonNormalizer (same as build_reason_index.py)
# =============================================================================
class ReasonNormalizer:
    def __init__(self):
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
        if not reason:
            return ""
        normalized = reason.lower()
        normalized = re.sub(r'^history:\s*', '', normalized)
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
        for abbrev, full in self.abbreviations.items():
            normalized = re.sub(abbrev, full, normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\s*//\s*', ', ', normalized)
        normalized = re.sub(r'[\.;]+$', '', normalized)
        normalized = re.sub(r'\s*,\s*', ', ', normalized)
        normalized = normalized.strip()
        return normalized


def split_reason(text):
    parts = re.split(r'[.;,]', text)
    return [p.strip() for p in parts if p.strip()]


# =============================================================================
# Part-level lookup (for on-the-fly categorization of dev/test)
# =============================================================================
class PartLookup:
    """Look up categorization info for any part_text.

    For parts already seen in training clusters/noise, uses the existing
    mapping. For unknown parts, encodes them with SentenceTransformer and
    assigns to the nearest cluster centroid via cosine similarity.
    """

    def __init__(self):
        from sklearn.metrics.pairwise import cosine_similarity
        self._cosine_similarity = cosine_similarity

        print("Building part lookup tables...")

        # Load clusters
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

        # Build reason_text -> [split_parts] from compound_split_map
        with open(REASON_INDEX_FILE) as f:
            reason_index = json.load(f)
        with open(COMPOUND_SPLIT_MAP_FILE) as f:
            raw_split_map = json.load(f)

        self.text_split_map = {}
        for idx_str, parts in raw_split_map.items():
            idx = int(idx_str)
            if idx < len(reason_index):
                self.text_split_map[reason_index[idx]] = parts

        # old_cat_name -> final_cat_name
        old_name_to_final = {}
        for final_name, info in mapping.items():
            for orig_name in info["original_categories"]:
                old_name_to_final[orig_name] = final_name
        old_name_to_final.update(EXTRA_MAPPINGS)

        # cluster_id -> final_cat
        self.cluster_to_final = {}
        for cat in taxonomy:
            final = old_name_to_final.get(cat["category_name"], cat["category_name"])
            for member in cat["members"]:
                self.cluster_to_final[str(member["cluster_id"])] = final

        # cluster_id -> image_inferable
        self.cluster_to_inferable = {}
        for entry in log_entries:
            cid = str(entry["cluster_id"])
            ri = entry.get("reason_info") or {}
            self.cluster_to_inferable[cid] = ri.get("image_inferable", "unknown")

        # cluster_id -> summary
        self.cluster_to_summary = {}
        for cid, info in summaries.items():
            self.cluster_to_summary[cid] = info.get("summary", "")

        # part_text -> cluster_id
        self.part_to_cluster = {}
        self.noise_part_set = set()
        for cid, parts in clusters.items():
            if cid == "-1":
                continue
            for part in parts:
                self.part_to_cluster[part["part_text"]] = cid

        for part_text, info in noise_assignments.items():
            assigned_cid = str(info["assigned_cluster"])
            if part_text not in self.part_to_cluster:
                self.part_to_cluster[part_text] = assigned_cid
                self.noise_part_set.add(part_text)

        print(f"  {len(self.part_to_cluster)} parts mapped")
        print(f"  {len(self.text_split_map)} compound splits loaded")

        # Build cluster centroids for nearest-centroid assignment of new parts
        print("  Building cluster centroids for new-part assignment...")
        X = np.load(EMBEDDINGS_FILE)
        with open(PART_MAP_FILE) as f:
            part_to_reason_idxs = json.load(f)
        unique_parts = list(part_to_reason_idxs.keys())
        part_to_emb_idx = {p: i for i, p in enumerate(unique_parts)}

        non_noise_cids = [cid for cid in clusters if cid != "-1"]
        self.centroids = {}
        for cid in non_noise_cids:
            emb_indices = [part_to_emb_idx[p["part_text"]]
                           for p in clusters[cid]
                           if p["part_text"] in part_to_emb_idx]
            if emb_indices:
                self.centroids[cid] = X[emb_indices].mean(axis=0)

        self.centroid_cids = list(self.centroids.keys())
        self.centroid_matrix = np.array([self.centroids[c] for c in self.centroid_cids])
        print(f"  {len(self.centroid_cids)} centroids built")

        # Lazy-loaded encoder for new parts
        self._encoder = None
        self._new_part_cache = {}  # cache: part_text -> cid

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print("  Loading SentenceTransformer for new-part encoding...")
            self._encoder = SentenceTransformer(MODEL_NAME)
        return self._encoder

    def _assign_nearest_cluster(self, pt):
        """Encode a new part and assign to nearest cluster centroid."""
        if pt in self._new_part_cache:
            return self._new_part_cache[pt]
        encoder = self._get_encoder()
        emb = encoder.encode([pt], normalize_embeddings=False)
        sims = self._cosine_similarity(emb, self.centroid_matrix)
        best_idx = sims[0].argmax()
        cid = self.centroid_cids[best_idx]
        self._new_part_cache[pt] = cid
        return cid

    def get_parts_for_text(self, text):
        """Get atomic parts for a reason text."""
        wc = len(text.split())
        if wc >= WORD_COUNT_THRESHOLD and text in self.text_split_map:
            parts = [p.strip() for p in self.text_split_map[text] if p.strip()]
            return parts if parts else [text]
        return [text]

    def categorize_part(self, pt):
        """Get categorization info for a single part_text."""
        cid = self.part_to_cluster.get(pt)
        from_noise = pt in self.noise_part_set

        if cid is not None:
            return {
                "part_text": pt,
                "cluster_id": int(cid),
                "cluster_summary": self.cluster_to_summary.get(cid, ""),
                "final_category": self.cluster_to_final.get(cid, ""),
                "image_inferable": self.cluster_to_inferable.get(cid, "unknown"),
                "from_noise": from_noise,
            }
        else:
            # New part: encode and assign to nearest centroid
            cid = self._assign_nearest_cluster(pt)
            self.part_to_cluster[pt] = cid
            self.noise_part_set.add(pt)
            return {
                "part_text": pt,
                "cluster_id": int(cid),
                "cluster_summary": self.cluster_to_summary.get(cid, ""),
                "final_category": self.cluster_to_final.get(cid, ""),
                "image_inferable": self.cluster_to_inferable.get(cid, "unknown"),
                "from_noise": True,
            }

    def categorize_reason(self, raw_reason, normalizer):
        """Categorize a raw reason string into parts with categories."""
        normalized = normalizer.normalize(raw_reason)
        split_parts = split_reason(normalized)
        if not split_parts:
            return [], [], False

        reasons_out = []
        all_cats = set()
        any_inferable = False

        for reason_text in split_parts:
            atomic_parts = self.get_parts_for_text(reason_text)
            parts_out = []
            for pt in atomic_parts:
                info = self.categorize_part(pt)
                parts_out.append(info)
                if info["final_category"]:
                    all_cats.add(info["final_category"])
                if info["image_inferable"] == "yes":
                    any_inferable = True

            reasons_out.append({
                "reason_text": reason_text,
                "parts": parts_out,
            })

        return reasons_out, sorted(all_cats), any_inferable


def extract_summary(categorized_reasons):
    """Extract final_categories and has_image_inferable from categorized reasons."""
    all_cats = set()
    any_inferable = False
    for reason in categorized_reasons:
        for part in reason["parts"]:
            if part["final_category"]:
                all_cats.add(part["final_category"])
            if part["image_inferable"] == "yes":
                any_inferable = True
    return sorted(all_cats), any_inferable


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load pre-computed categorization
    print("Loading pre-computed categorization...")
    with open(ID_TO_CATEGORIZED_FILE) as f:
        id_to_categorized = json.load(f)
    print(f"  {len(id_to_categorized)} entries")

    # Build part lookup for on-the-fly computation
    lookup = PartLookup()
    normalizer = ReasonNormalizer()

    # =========================================================================
    # Process each source file
    # =========================================================================
    sources = [
        {
            "name": "MIMIC train",
            "file": f"{ORIGINAL_DATA}/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json",
            "output": f"{OUTPUT_DIR}/chat_train_MIMIC_CXR_augmented.json",
            "key_fn": lambda e: f"mimic:{e['id']}",
            "reason_fn": lambda e: e.get("reason") or "",
        },
        {
            "name": "MIMIC dev",
            "file": f"{ORIGINAL_DATA}/chat_dev_MIMIC_CXR_all_gpt4extract_rulebased_v1.json",
            "output": f"{OUTPUT_DIR}/chat_dev_MIMIC_CXR_augmented.json",
            "key_fn": lambda e: f"mimic:{e['id']}",
            "reason_fn": lambda e: e.get("reason") or "",
        },
        {
            "name": "MIMIC test",
            "file": f"{ORIGINAL_DATA}/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json",
            "output": f"{OUTPUT_DIR}/chat_test_MIMIC_CXR_augmented.json",
            "key_fn": lambda e: f"mimic:{e['id']}",
            "reason_fn": lambda e: e.get("reason") or "",
        },
        {
            "name": "CheXpert+",
            "file": f"{ORIGINAL_DATA}/df_chexpert_plus_240401.json",
            "output": f"{OUTPUT_DIR}/df_chexpert_plus_augmented.json",
            "key_fn": lambda e: f"chexpert:{e['path_to_image']}",
            "reason_fn": lambda e: ". ".join(filter(None, [
                (e.get("section_clinical_history") or "").strip(),
                (e.get("section_history") or "").strip(),
            ])).strip(),
        },
        {
            "name": "Rex train",
            "file": f"{ORIGINAL_DATA}/rexgradient_train.json",
            "output": f"{OUTPUT_DIR}/rexgradient_train_augmented.json",
            "key_fn": lambda e: f"rex:{e['id']}",
            "reason_fn": lambda e: e.get("indication") or "",
        },
        {
            "name": "Rex test",
            "file": f"{ORIGINAL_DATA}/rexgradient_test.json",
            "output": f"{OUTPUT_DIR}/rexgradient_test_augmented.json",
            "key_fn": lambda e: f"rex:{e['id']}",
            "reason_fn": lambda e: e.get("indication") or "",
        },
    ]

    total_stats = {"total": 0, "precomputed": 0, "computed": 0, "no_reason": 0}

    for src in sources:
        print(f"\n{'='*60}")
        print(f"Processing {src['name']}...")
        print(f"{'='*60}")

        with open(src["file"]) as f:
            data = json.load(f)
        print(f"  {len(data)} samples")

        precomputed = 0
        computed = 0
        no_reason = 0

        for entry in tqdm(data, desc=src["name"]):
            key = src["key_fn"](entry)
            raw_reason = src["reason_fn"](entry)

            if not raw_reason:
                entry["reason_parts"] = []
                entry["final_categories"] = []
                entry["has_image_inferable"] = False
                no_reason += 1
                continue

            # Try pre-computed first
            if key in id_to_categorized:
                cat_info = id_to_categorized[key]
                entry["reason_parts"] = cat_info["reasons"]
                cats, inferable = extract_summary(cat_info["reasons"])
                entry["final_categories"] = cats
                entry["has_image_inferable"] = inferable
                precomputed += 1
            else:
                # Compute on-the-fly
                reasons_out, cats, inferable = lookup.categorize_reason(
                    raw_reason, normalizer
                )
                entry["reason_parts"] = reasons_out
                entry["final_categories"] = cats
                entry["has_image_inferable"] = inferable
                computed += 1

        print(f"  Pre-computed: {precomputed}")
        print(f"  Computed on-the-fly: {computed}")
        print(f"  No reason: {no_reason}")

        total_stats["total"] += len(data)
        total_stats["precomputed"] += precomputed
        total_stats["computed"] += computed
        total_stats["no_reason"] += no_reason

        # Category distribution
        cat_counts = defaultdict(int)
        inferable_count = 0
        for entry in data:
            for c in entry["final_categories"]:
                cat_counts[c] += 1
            if entry["has_image_inferable"]:
                inferable_count += 1

        print(f"  Samples with image-inferable parts: {inferable_count} "
              f"({inferable_count/len(data)*100:.1f}%)")
        print(f"  Category distribution:")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {cnt:6d} ({cnt/len(data)*100:5.1f}%)  {cat}")

        # Save
        print(f"  Saving {src['output']}...")
        with open(src["output"], "w") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  Done.")

    # Final summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples: {total_stats['total']}")
    print(f"  Pre-computed: {total_stats['precomputed']}")
    print(f"  Computed on-the-fly: {total_stats['computed']}")
    print(f"  No reason: {total_stats['no_reason']}")
    print(f"\nAll augmented files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
