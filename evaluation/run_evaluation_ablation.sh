#!/usr/bin/env bash
# Compute utility metrics for the REX ablation prediction files produced by
# ../prediction/run_prediction_ablation.sh: 3 models x {ldp, sldp} x
# {eps=4, 8, inf}.
set -euo pipefail

MODELS=(
  llava-hf/llava-1.5-7b-hf
  Qwen/Qwen2-VL-7B-Instruct
  Qwen/Qwen2-VL-2B-Instruct
)

PRED_BASE="${PRED_BASE:-../prediction/predictions}"
EVAL_BASE="${EVAL_BASE:-./evaluations}"

METHODS=( "ldp" "sldp" )
EPS_LABELS=( 4 8 inf )

mkdir -p "${EVAL_BASE}"
for MODEL in "${MODELS[@]}"; do
  NAME="$(basename "$MODEL")"
  for METHOD in "${METHODS[@]}"; do
    for EPS in "${EPS_LABELS[@]}"; do
      PRED_FILE="${PRED_BASE}/${NAME}-rex-${METHOD}_${EPS}_ablation"
      EVAL_DIR="${EVAL_BASE}/${NAME}-rex-${METHOD}_${EPS}_ablation"
      echo "===================================================="
      echo "EVAL REX ABLATION | ${METHOD} | eps=${EPS} | ${MODEL}"
      echo "Predictions: ${PRED_FILE}"
      echo "Output dir:  ${EVAL_DIR}"
      echo "===================================================="
      mkdir -p "${EVAL_DIR}"
      python3 evaluation.py \
        --results_file "${PRED_FILE}" \
        --output_dir "${EVAL_DIR}"
    done
  done
done
