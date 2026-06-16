#!/usr/bin/env bash
# Privacy-budget ablation on ReXGradient (Table 3 in the paper).
# Sweeps per-token epsilon in {4, 8, inf} across all 3 models for LDP and
# RadSLDP. epsilon=inf is implemented via --no_noise (clip only, no Gaussian
# noise). epsilon=8 overlaps with run_training_rex.sh and is re-trained here so
# the ablation is self-contained.
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
DELTA=2.8643446e-05
OUT_ROOT="${OUT_ROOT:-../checkpoints}"

COMMON=(
  --use_lora
  --gradient_accumulation_steps 1
  --num_train_epochs 2
  --max_length 4096
  --per_device_train_batch_size 6
  --train_image_folder "${DATA_ROOT}"
  --train_dataset "${TRAIN_JSON}"
  --target_delta "${DELTA}"
)

# eps_label -> ( epsilon_value, extra_noise_flag )
declare -A EPS_VAL=(  [4]=4   [8]=8   [inf]=8 )
declare -A EPS_FLAG=( [4]=""  [8]=""  [inf]="--no_noise" )
EPS_LABELS=( 4 8 inf )

run() {
  local model="$1" method_tag="$2" method_flags="$3" eps_label="$4"
  local name; name="$(basename "$model")"
  local outdir="${OUT_ROOT}/${name}-rex-${method_tag}_${eps_label}_ablation"
  local eps_val="${EPS_VAL[$eps_label]}"
  local eps_flag="${EPS_FLAG[$eps_label]}"
  echo "=============================================="
  echo "REX ABLATION | ${method_tag} | eps=${eps_label} | ${model}"
  echo "Output dir: ${outdir}"
  echo "=============================================="
  # shellcheck disable=SC2086
  torchrun --nproc_per_node=${NPROC} hf_finetune.py \
    --model_id "${model}" \
    --output_dir "${outdir}" \
    --target_epsilon "${eps_val}" \
    "${COMMON[@]}" \
    ${method_flags} ${eps_flag}
}

for MODEL in "${MODELS[@]}"; do
  for EPS in "${EPS_LABELS[@]}"; do
    run "${MODEL}" "ldp"  "--use_ldp"                  "${EPS}"
    run "${MODEL}" "sldp" "--use_ldp --selective_ldp"  "${EPS}"
  done
done
