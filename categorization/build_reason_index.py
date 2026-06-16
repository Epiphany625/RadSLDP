"""
Build reason indices from the 3 original datasets.

Steps:
1. Load MIMIC-CXR, CheXpert+, Rex original data
2. Normalize reasons with ReasonNormalizer
3. Split by ";.,"
4. Collect all distinct reasons
5. Sort reasons by word count (descending: longest to shortest)
6. Assign reason_idx based on sorted order

Outputs (all in original_data/):
- reason_index.json:              list where list[idx] = reason_text (idx 0 = longest reason)
- id_to_reasons.json:             {dataset:id -> [reason_idx, ...]}
- reason_to_ids.json:             {reason_idx -> [{dataset, id}, ...]}
- distinct_reasons_by_length.json: [[reason_idx, text, word_count], ...] sorted by word count desc
- distinct_reasons_by_count.json:  [[reason_idx, text, count], ...] sorted by count desc
"""

import json
import os
import re
from collections import Counter, defaultdict
from tqdm import tqdm
from pathlib import Path


class ReasonNormalizer:
    """Normalizes reason text to handle variations"""

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
    """Split a normalized reason string by ;., into non-empty parts."""
    parts = re.split(r'[.;,]', text)
    return [p.strip() for p in parts if p.strip()]


def apply_compound_splits(parts, split_map):
    """Apply LLM-derived compound splitting to each part."""
    result = []
    for part in parts:
        if part in split_map:
            result.extend(split_map[part])
        else:
            result.append(part)
    return result


def main():
    base = Path(os.environ.get("DATA_DIR", "original_data"))
    normalizer = ReasonNormalizer()

    # Load compound split map if available (text-keyed version)
    compound_split_map = {}
    split_map_path = base / "compound_split_map.json"
    if split_map_path.exists():
        with open(split_map_path) as f:
            raw_map = json.load(f)
        # Load reason_index to convert idx keys to text keys
        reason_index_path = base / "reason_index.json"
        if reason_index_path.exists():
            with open(reason_index_path) as f:
                prev_index = json.load(f)
            for idx_str, parts in raw_map.items():
                idx = int(idx_str)
                if idx < len(prev_index):
                    compound_split_map[prev_index[idx]] = parts
        print(f"Loaded compound split map: {len(compound_split_map)} entries")

    # =========================================================================
    # PASS 1: Collect text-based mappings
    # =========================================================================
    # {dataset:id -> [reason_text, ...]}
    id_to_reasons_text = {}
    # {reason_text -> [{dataset, id}]}
    reason_to_ids_text = defaultdict(list)
    # Count appearances
    reason_counter_text = Counter()

    # =========================================================================
    # 1. MIMIC-CXR
    # =========================================================================
    print("Loading MIMIC-CXR...")
    with open(base / "chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json") as f:
        mimic = json.load(f)

    for entry in tqdm(mimic, desc="MIMIC-CXR"):
        raw = entry.get("reason")
        if not raw:
            continue
        sid = entry["id"]
        key = f"mimic:{sid}"

        normalized = normalizer.normalize(raw)
        parts = split_reason(normalized)
        parts = apply_compound_splits(parts, compound_split_map)

        if not parts:
            continue

        unique_parts = list(dict.fromkeys(parts))
        id_to_reasons_text[key] = unique_parts

        for part in unique_parts:
            reason_to_ids_text[part].append({"dataset": "mimic", "id": sid})
            reason_counter_text[part] += 1

    del mimic
    print(f"  MIMIC-CXR done: {len(id_to_reasons_text)} entries so far")

    # =========================================================================
    # 2. CheXpert+
    # =========================================================================
    print("Loading CheXpert+...")
    with open(base / "df_chexpert_plus_240401.json") as f:
        chex = json.load(f)

    for entry in tqdm(chex, desc="CheXpert+"):
        clinical = (entry.get("section_clinical_history") or "").strip()
        history = (entry.get("section_history") or "").strip()
        # Concatenate both fields if present; use whichever is available
        raw = ". ".join(filter(None, [clinical, history])).strip()
        if not raw:
            continue
        sid = entry["path_to_image"]
        key = f"chexpert:{sid}"

        normalized = normalizer.normalize(raw)
        parts = split_reason(normalized)
        parts = apply_compound_splits(parts, compound_split_map)

        if not parts:
            continue

        unique_parts = list(dict.fromkeys(parts))
        id_to_reasons_text[key] = unique_parts

        for part in unique_parts:
            reason_to_ids_text[part].append({"dataset": "chexpert", "id": sid})
            reason_counter_text[part] += 1

    del chex
    print(f"  CheXpert+ done: {len(id_to_reasons_text)} entries so far")

    # =========================================================================
    # 3. Rex
    # =========================================================================
    print("Loading Rex...")
    with open(base / "rexgradient_train.json") as f:
        rex = json.load(f)

    for entry in tqdm(rex, desc="Rex"):
        raw = entry.get("indication")
        if not raw:
            continue
        sid = entry["id"]
        key = f"rex:{sid}"

        normalized = normalizer.normalize(raw)
        parts = split_reason(normalized)
        parts = apply_compound_splits(parts, compound_split_map)

        if not parts:
            continue

        unique_parts = list(dict.fromkeys(parts))
        id_to_reasons_text[key] = unique_parts

        for part in unique_parts:
            reason_to_ids_text[part].append({"dataset": "rex", "id": sid})
            reason_counter_text[part] += 1

    del rex
    print(f"  Rex done: {len(id_to_reasons_text)} entries so far")

    # =========================================================================
    # PASS 2: Sort reasons by word count (descending) and assign idx
    # =========================================================================
    print("\nSorting reasons by word count (longest first)...")
    # Sort by word count descending, then alphabetically for ties
    sorted_reasons = sorted(
        reason_counter_text.keys(),
        key=lambda r: (-len(r.split()), r)
    )

    # Build reason_index (list where idx -> text) and reason_to_idx (text -> idx)
    reason_index = sorted_reasons
    reason_to_idx = {text: idx for idx, text in enumerate(reason_index)}

    print(f"  {len(reason_index)} distinct reasons")
    print(f"  Longest: {len(reason_index[0].split())} words")
    print(f"  Shortest: {len(reason_index[-1].split())} words")

    # =========================================================================
    # PASS 3: Convert text-based mappings to idx-based
    # =========================================================================
    print("\nConverting to idx-based mappings...")
    id_to_reasons = {}
    reason_to_ids = defaultdict(list)
    reason_counter = Counter()

    for key, text_list in id_to_reasons_text.items():
        idx_list = [reason_to_idx[text] for text in text_list]
        id_to_reasons[key] = idx_list

    for text, id_list in reason_to_ids_text.items():
        idx = reason_to_idx[text]
        reason_to_ids[idx] = id_list

    for text, count in reason_counter_text.items():
        idx = reason_to_idx[text]
        reason_counter[idx] = count

    # =========================================================================
    # Save outputs
    # =========================================================================
    out_dir = base
    print(f"\nTotal entries with reasons: {len(id_to_reasons)}")
    print(f"Total distinct reason parts: {len(reason_index)}")
    print(f"Total reason-id links: {sum(reason_counter.values())}")

    # 0. reason_index (canonical idx <-> text mapping, sorted by word count desc)
    print("Saving reason_index.json...")
    with open(out_dir / "reason_index.json", "w") as f:
        json.dump(reason_index, f, ensure_ascii=False)

    # 1. id_to_reasons (stores idx only)
    print("Saving id_to_reasons.json...")
    with open(out_dir / "id_to_reasons.json", "w") as f:
        json.dump(id_to_reasons, f, ensure_ascii=False)

    # 2. reason_to_ids (keyed by str(idx) for JSON compat)
    print("Saving reason_to_ids.json...")
    r2ids = {str(idx): ids for idx, ids in reason_to_ids.items()}
    with open(out_dir / "reason_to_ids.json", "w") as f:
        json.dump(r2ids, f, ensure_ascii=False)

    # 3. Distinct reasons sorted by word count (descending)
    print("Saving distinct_reasons_by_length.json...")
    by_length = [[idx, text, len(text.split())] for idx, text in enumerate(reason_index)]
    with open(out_dir / "distinct_reasons_by_length.json", "w") as f:
        json.dump(by_length, f, ensure_ascii=False, indent=2)

    # 4. Distinct reasons sorted by count (descending)
    print("Saving distinct_reasons_by_count.json...")
    by_count = [[idx, reason_index[idx], count] for idx, count in reason_counter.most_common()]
    with open(out_dir / "distinct_reasons_by_count.json", "w") as f:
        json.dump(by_count, f, ensure_ascii=False, indent=2)

    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Per-dataset stats
    ds_counts = Counter()
    for key in id_to_reasons:
        ds = key.split(":")[0]
        ds_counts[ds] += 1
    for ds, cnt in ds_counts.most_common():
        print(f"  {ds}: {cnt} entries with reasons")

    print(f"\n  Distinct reasons: {len(reason_index)}")
    print(f"  Top 10 most common:")
    for idx, count in reason_counter.most_common(10):
        print(f"    ({count:5d}) [idx={idx}] {reason_index[idx][:80]}")

    # Word count distribution
    word_counts = [len(r.split()) for r in reason_index]
    word_counts_sorted = sorted(word_counts, reverse=True)
    n = len(word_counts_sorted)
    mean_wc = sum(word_counts) / n
    median_wc = word_counts_sorted[n // 2]
    p25 = word_counts_sorted[n // 4]
    p75 = word_counts_sorted[3 * n // 4]
    p90 = word_counts_sorted[int(n * 0.9)]
    p95 = word_counts_sorted[int(n * 0.95)]
    p99 = word_counts_sorted[int(n * 0.99)]

    print(f"\n  Reason length distribution (word count):")
    print(f"    Max:    {word_counts_sorted[0]}")
    print(f"    99th:   {p99}")
    print(f"    95th:   {p95}")
    print(f"    90th:   {p90}")
    print(f"    75th:   {p75}")
    print(f"    Median: {median_wc}")
    print(f"    Mean:   {mean_wc:.1f}")
    print(f"    25th:   {p25}")
    print(f"    Min:    {word_counts_sorted[-1]}")

    # Histogram buckets
    buckets = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 50), (51, None)]
    print(f"\n  Word count histogram:")
    for lo, hi in buckets:
        if hi is None:
            cnt = sum(1 for w in word_counts if w >= lo)
            label = f"    {lo}+:"
        elif lo == hi:
            cnt = sum(1 for w in word_counts if w == lo)
            label = f"    {lo}:"
        else:
            cnt = sum(1 for w in word_counts if lo <= w <= hi)
            label = f"    {lo}-{hi}:"
        pct = cnt / n * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:>10s} {cnt:7d} ({pct:5.1f}%) {bar}")

    print(f"\n  Longest reasons (5 examples):")
    for idx, text, wc in by_length[:5]:
        print(f"    [idx={idx}] ({wc:3d} words) {text[:100]}...")

    print(f"\n  Shortest reasons (5 examples):")
    for idx, text, wc in by_length[-5:]:
        print(f"    [idx={idx}] ({wc:3d} words) {text}")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
