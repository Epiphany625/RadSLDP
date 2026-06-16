#!/usr/bin/env bash
# Compute BLEU-1/4, ROUGE-L, CheXbert Micro-F1, and F1-RadGraph for the 12
# ReXGradient prediction files produced by ../prediction/run_prediction_rex.sh.
set -euo pipefail

MODELS=(
  llava-hf/llava-1.5-7b-hf
  Qwen/Qwen2-VL-7B-Instruct
  Qwen/Qwen2-VL-2B-Instruct
)

PRED_BASE="${PRED_BASE:-../prediction/predictions}"
EVAL_BASE="${EVAL_BASE:-./evaluations}"
EPSILON=8

REGIMES=(
  "nodp"
  "dp_${EPSILON}"
  "ldp_${EPSILON}"
  "sldp_${EPSILON}"
)

mkdir -p "${EVAL_BASE}"
for MODEL in "${MODELS[@]}"; do
  NAME="$(basename "$MODEL")"
  for TAG in "${REGIMES[@]}"; do
    PRED_FILE="${PRED_BASE}/${NAME}-rex-${TAG}"
    EVAL_DIR="${EVAL_BASE}/${NAME}-rex-${TAG}"
    echo "===================================================="
    echo "EVAL REX | ${TAG} | ${MODEL}"
    echo "Predictions: ${PRED_FILE}"
    echo "Output dir:  ${EVAL_DIR}"
    echo "===================================================="
    mkdir -p "${EVAL_DIR}"
    python3 evaluation.py \
      --results_file "${PRED_FILE}" \
      --output_dir "${EVAL_DIR}"
  done
done
