# ReXGradient

Scripts for preparing the ReXGradient-160K chest X-ray dataset.

## Prerequisites

You must first obtain access rights to the ReXGradient dataset
(`rajpurkarlab/ReXGradient-160K` on Hugging Face) before downloading.

## Layout

- **`download/`** — downloads the ReXGradient dataset (images + metadata) and decompresses it.
  - `download.py` — fetches metadata and the `deid_png.part*` shards from Hugging Face.
  - `decompress.py` — decompresses the combined `deid_png.tar` (zstd).
  - `download_rex.sh` — end-to-end driver: sets up the conda env, downloads, combines, decompresses, and extracts images.
- **`processing/`** — cleans and processes the dataset into a MIMIC-compatible format.
  - `process.py` — parses the raw metadata into the MIMIC-compatible version.
  - `process_data.sh` — runs the processing step.

## Usage

```bash
# 1. Download, decompress, and extract images
cd download
bash download_rex.sh

# 2. Process into MIMIC-compatible format
cd ../processing
bash process_data.sh
```

## Expected output

After downloading and processing, you should have:

- `datasets/ReXGradient/download/deid_png/` — chest images.
- `datasets/ReXGradient/download/ReXGradient/metadata/` — raw and processed metadata.
