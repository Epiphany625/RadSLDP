# CheXpert Plus

Scripts for preparing the CheXpert Plus chest X-ray dataset
(Chambon et al., 2024) into the RadSLDP training format.

## Access

CheXpert Plus is distributed only through the Stanford AIMI portal and requires
registration and a signed data-use agreement before download:

1. Create an account at <https://stanfordaimi.azurewebsites.net/datasets>.
2. Request access to **CheXpert Plus** and accept the Stanford University School
   of Medicine Research Data Use Agreement.
3. Stanford AIMI then issues per-user signed download URLs that expire after a
   short window.

## Layout

- **`download/download.sh`** — wget driver. Paste the signed Stanford AIMI URLs
  into the `URLS=( ... )` array, then run.
- **`processing/preprocess.py`** — parses `df_chexpert_plus_240401.csv` into the
  RadSLDP per-sample JSON format (`indication`, `findings`, demographics,
  `conversations`). Override `CHEX_CSV` and `CHEX_OUT_DIR` env vars to customize
  paths.
- **`processing/preprocess.sh`** — one-line driver around `preprocess.py`.

## Required files (from the portal)

| File                                | Used by           |
|-------------------------------------|-------------------|
| `df_chexpert_plus_240401.csv`       | `preprocess.py`   |
| `PNG.zip` (frontal + lateral views) | training scripts  |
| `chexbert_labels.zip`               | optional          |
| `radgraph-XL-annotations.zip`       | optional          |

The DICOM tarballs are also distributed by Stanford AIMI but are not consumed
by this repository.

## Usage

```bash
# 1. Paste signed URLs into URLS=() in download.sh, then:
cd download
bash download.sh

# 2. Convert the CSV into RadSLDP-format JSON / JSONL:
cd ../processing
bash preprocess.sh
```

## Expected output

After processing you should have:

- `datasets/CheXpertPlus/download/chexpertplus/PNG/` — chest X-ray PNGs.
- `datasets/CheXpertPlus/download/chexpertplus/df_chexpert_plus_240401.csv` — source metadata CSV.
- `datasets/CheXpertPlus/processing/chexpert_plus_train.json` (and `.jsonl`)  — training split.
- `datasets/CheXpertPlus/processing/chexpert_plus_valid.json` (and `.jsonl`)  — validation split (used as the held-out test split in this work).

Point `--train_image_folder` / `--train_dataset` in
`training/run_training_chex.sh` (and the matching prediction / evaluation
scripts) at the directories above.
