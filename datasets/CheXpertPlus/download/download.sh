#!/usr/bin/env bash
# CheXpert Plus is distributed through Stanford AIMI's portal (no public
# mirror). You must register an account, accept the data-use agreement at
#     https://stanfordaimi.azurewebsites.net/datasets
# and request access to "CheXpert Plus". The portal then issues per-user signed
# download URLs that expire after a short window.
#
# Once you have the signed URLs, paste them into the URLS array below and run
# this script. The expected files are:
#   df_chexpert_plus_240401.csv      (~400 MB; the metadata CSV consumed by ../processing/preprocess.py)
#   PNG.zip                          (~440 GB; PNG images, frontal + lateral views)
#   chexbert_labels.zip              (~ 6 MB; CheXbert labels per study, optional)
#   radgraph-XL-annotations.zip      (~ 75 MB; RadGraph annotations, optional)
#
# DICOM source images are also distributed but are not used by this repository.
set -euo pipefail

OUT_DIR="${OUT_DIR:-./chexpertplus}"
mkdir -p "${OUT_DIR}"

URLS=(
  # "https://<stanford-aimi-signed-url>/df_chexpert_plus_240401.csv"
  # "https://<stanford-aimi-signed-url>/PNG.zip"
  # "https://<stanford-aimi-signed-url>/chexbert_labels.zip"
  # "https://<stanford-aimi-signed-url>/radgraph-XL-annotations.zip"
)

if [ "${#URLS[@]}" -eq 0 ]; then
  echo "ERROR: URLS array is empty."
  echo "Register at https://stanfordaimi.azurewebsites.net/datasets, request"
  echo "access to CheXpert Plus, and paste the signed URLs into this script."
  exit 1
fi

for url in "${URLS[@]}"; do
  echo "Downloading: ${url}"
  wget -c -P "${OUT_DIR}" "${url}"
done

# Decompress PNG archive (skip if already done)
if [ -f "${OUT_DIR}/PNG.zip" ] && [ ! -d "${OUT_DIR}/PNG" ]; then
  echo "Unzipping PNG.zip ..."
  unzip -q "${OUT_DIR}/PNG.zip" -d "${OUT_DIR}"
fi

echo "Done. Files placed under ${OUT_DIR}"
