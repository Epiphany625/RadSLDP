#!/usr/bin/env bash
# Generate predictions on the ReXGradient test split for all ablation
# checkpoints (3 models x {LDP, RadSLDP} x {eps=4, 8, inf}) produced by
# ../training/train_rex_ablation.sh.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODELS=(
  llava-hf/llava-1.5-7b-hf
  Qwen/Qwen2-VL-7B-Instruct
  Qwen/Qwen2-VL-2B-Instruct
)

DATA_ROOT="../datasets/ReXGradient/download/ReXGradient"
TEST_JSON="${DATA_ROOT}/metadata/rexgradient_test.json"
CKPT_ROOT="${CKPT_ROOT:-../checkpoints}"
PRED_BASE="${PRED_BASE:-./predictions}"

METHODS=( "ldp" "sldp" )
EPS_LABELS=( 4 8 inf )

mkdir -p "${PRED_BASE}"
for MODEL in "${MODELS[@]}"; do
  NAME="$(basename "$MODEL")"
  for METHOD in "${METHODS[@]}"; do
    for EPS in "${EPS_LABELS[@]}"; do
      CKPT="${CKPT_ROOT}/${NAME}-rex-${METHOD}_${EPS}_ablation"
      PRED_FILE="${PRED_BASE}/${NAME}-rex-${METHOD}_${EPS}_ablation"
      echo "===================================================="
      echo "PREDICT REX ABLATION | ${METHOD} | eps=${EPS} | ${MODEL}"
      echo "Checkpoint: ${CKPT}"
      echo "Prediction file: ${PRED_FILE}"
      echo "===================================================="
      python3 prediction.py \
        --model_id "${MODEL}" \
        --finetuned_model_path "${CKPT}" \
        --test_image_folder "${DATA_ROOT}" \
        --test_dataset "${TEST_JSON}" \
        --prediction_file "${PRED_FILE}" \
        --use_lora
    done
  done
done
