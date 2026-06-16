#!/usr/bin/env bash
# Generate predictions on the MIMIC-CXR test split for all 12 trained
# checkpoints (3 models x 4 regimes) produced by ../training/run_training_mimic.sh.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODELS=(
  llava-hf/llava-1.5-7b-hf
  Qwen/Qwen2-VL-7B-Instruct
  Qwen/Qwen2-VL-2B-Instruct
)

DATA_ROOT="../datasets/MIMIC-CXR/download/mimic-cxr-jpg/2.1.0"
TEST_JSON="../datasets/MIMIC-CXR/processing/chat_test_MIMIC_CXR_parsed_rule-based.json"
CKPT_ROOT="${CKPT_ROOT:-../checkpoints}"
PRED_BASE="${PRED_BASE:-./predictions}"
EPSILON=8

REGIMES=(
  "nodp"
  "dp_${EPSILON}"
  "ldp_${EPSILON}"
  "sldp_${EPSILON}"
)

mkdir -p "${PRED_BASE}"
for MODEL in "${MODELS[@]}"; do
  NAME="$(basename "$MODEL")"
  for TAG in "${REGIMES[@]}"; do
    CKPT="${CKPT_ROOT}/${NAME}-mimic-${TAG}_final"
    PRED_FILE="${PRED_BASE}/${NAME}-mimic-${TAG}"
    echo "===================================================="
    echo "PREDICT MIMIC | ${TAG} | ${MODEL}"
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
