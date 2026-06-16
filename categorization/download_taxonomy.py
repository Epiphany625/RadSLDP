"""
Fetch the expert-validated RadSLDP NVCC taxonomy from Hugging Face into this
directory so that ``training/hf_finetune.py`` can load it.

By default the file is placed at
``categorization/RadSLDP_nvcc_taxonomy.json`` — the same path that
``apply_expert_corrections.py`` writes to and that ``hf_finetune.py`` reads
from. Use this script when you want to reproduce the paper's runs without
regenerating the taxonomy from scratch (Steps 1-10 in ``README.md``).

Usage:
    python3 download_taxonomy.py
    OUTPUT_DIR=/path/to/dir python3 download_taxonomy.py
"""

import os

from huggingface_hub import hf_hub_download

REPO_ID = "Konghao/RadSLDP-NVCC-taxonomy"
FILENAME = "RadSLDP_nvcc_taxonomy.json"

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
    )
    print(f"Downloaded {FILENAME} to {path}")


if __name__ == "__main__":
    main()
