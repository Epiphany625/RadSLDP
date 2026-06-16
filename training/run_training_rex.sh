#!/usr/bin/env bash
# Train all 4 regimes (Non-private, DP-SGD, LDP, RadSLDP) for all 3 models on
# ReXGradient at per-token epsilon=8. Reproduces the ReXGradient column of
# Table 2 in the paper.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}

MODELS=(
  llava-hf/llava-1.5-7b-hf
  Qwen/Qwen2-VL-7B-Instruct
  Qwen/Qwen2-VL-2B-Instruct
)

DATA_ROOT="../datasets/ReXGradient/download/ReXGradient"
TRAIN_JSON="${DATA_ROOT}/metadata/rexgradient_train.json"
DELTA=2.8643446e-05            # 1 / n_train
OUT_ROOT="${OUT_ROOT:-../checkpoints}"
EPSILON=8

COMMON=(
  --use_lora
  --gradient_accumulation_steps 1
  --num_train_epochs 2
  --max_length 4096
  --train_image_folder "${DATA_ROOT}"
  --train_dataset "${TRAIN_JSON}"
)

run() {
  local model="$1" tag="$2" extra="$3" bs="$4"
  local name; name="$(basename "$model")"
  local outdir="${OUT_ROOT}/${name}-rex-${tag}_final"
  echo "=============================================="
  echo "REX | ${tag} | ${model} -> ${outdir}"
  echo "=============================================="
  # shellcheck disable=SC2086
  torchrun --nproc_per_node=${NPROC} hf_finetune.py \
    --model_id "${model}" \
    --output_dir "${outdir}" \
    --per_device_train_batch_size "${bs}" \
    --target_epsilon "${EPSILON}" \
    --target_delta "${DELTA}" \
    "${COMMON[@]}" \
    ${extra}
}

# Non-private (no privacy flags)
for MODEL in "${MODELS[@]}"; do run "${MODEL}" "nodp"           ""                                6; done
# DP-SGD baseline
for MODEL in "${MODELS[@]}"; do run "${MODEL}" "dp_${EPSILON}"  "--use_dp"                        1; done
# Uniform LDP
for MODEL in "${MODELS[@]}"; do run "${MODEL}" "ldp_${EPSILON}" "--use_ldp"                       6; done
# RadSLDP (selective LDP)
for MODEL in "${MODELS[@]}"; do run "${MODEL}" "sldp_${EPSILON}" "--use_ldp --selective_ldp"      6; done
