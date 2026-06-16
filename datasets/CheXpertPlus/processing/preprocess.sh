#!/usr/bin/env bash
# Process the CheXpert Plus metadata CSV into RadSLDP-format JSON / JSONL.
# Inputs: df_chexpert_plus_240401.csv (from ../download/)
# Outputs: chexpert_plus_{train,valid}.{json,jsonl}
set -euo pipefail

export CHEX_CSV="${CHEX_CSV:-../download/chexpertplus/df_chexpert_plus_240401.csv}"
export CHEX_OUT_DIR="${CHEX_OUT_DIR:-.}"

python3 preprocess.py
