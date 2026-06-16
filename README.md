# RadSLDP: Selective Local Differential Privacy for Radiology Vision-Language Models

This repository contains the official implementation of **RadSLDP**, presented
at **MICCAI 2026**.

RadSLDP is a privacy-preserving fine-tuning mechanism for medical vision-language
models (VLMs) in the *untrusted external trainer* setting, where a hospital must
outsource fine-tuning of large radiology report-generation models to a cloud
provider it does not trust with raw patient data. Standard DP-SGD assumes a
trusted trainer and is unsuitable; uniform Local Differential Privacy (LDP)
protects every text token but destroys clinical utility. RadSLDP instead
perturbs **only the embeddings of Non-Visual Clinical Context (NVCC) tokens**
(demographics, medical history, indications) — identified via an
expert-validated 347-cluster taxonomy derived from 141,129 clinical
indications — while leaving image-inferable tokens and the image embeddings
unmodified. Each protected token embedding is clipped to a fixed ℓ₂ norm and
perturbed with calibrated Gaussian noise, giving a formal per-token
(ε, δ)-LDP guarantee independent of the downstream model. At per-token ε = 8,
RadSLDP matches or outperforms uniform LDP in 94% of comparisons across three
chest X-ray datasets and three VLM architectures, with BLEU-4 gains up to 77%
and CheXbert Micro-F1 gains up to 108% on ReXGradient.

## NVCC Taxonomy (released artifact)

The expert-validated **NVCC taxonomy** — 347 cluster summaries derived from
141,129 unique clinical indications across ReXGradient-160K, CheXpert+, and
MIMIC-CXR, each tagged `image_inferable: yes|no` after physicist review — is
one of the paper's two core contributions and is **required** by every
training run.

- **Hugging Face Datasets:** https://huggingface.co/datasets/Konghao/RadSLDP-NVCC-taxonomy
- **License:** CC-BY-4.0
- **File:** `RadSLDP_nvcc_taxonomy.json` (~640 MB)

Fetch it once after cloning the repo:

```bash
python3 categorization/download_taxonomy.py
# -> categorization/RadSLDP_nvcc_taxonomy.json
```

That single command is the only data prerequisite for training; everything
under `training/` loads the file from this path automatically.

If you instead want to **rebuild** the taxonomy from raw clinical indications
(e.g., to extend it to a new dataset), follow Steps 1–10 in
`categorization/README.md` and use `apply_expert_corrections.py` to produce the
same file from scratch.

## Repository layout

```
RadSLDP/
├── categorization/    NVCC taxonomy pipeline. Splits, clusters, LLM-annotates,
│                      and physicist-corrects the 141,129 clinical indications
│                      from ReXGradient / CheXpert+ / MIMIC-CXR into the 347
│                      image-inferable / non-inferable cluster summaries.
│                      Outputs RadSLDP_nvcc_taxonomy.json — the public artifact
│                      loaded by training.
│
├── datasets/          Per-dataset download + preprocessing scripts for
│                      ReXGradient, CheXpert+, and MIMIC-CXR. All three are
│                      access-controlled (PhysioNet credential, Stanford AIMI
│                      DUA, Hugging Face access request); see each subfolder's
│                      README.
│
├── training/          Fine-tuning entry point hf_finetune.py and per-dataset
│                      drivers run_training_{rex,chex,mimic}.sh. Each driver
│                      trains all 3 models under all 4 regimes
│                      (Non-private / DP-SGD / LDP / RadSLDP) at per-token
│                      ε = 8. train_rex_ablation.sh sweeps ε ∈ {4, 8, ∞} on
│                      ReXGradient for Table 3 of the paper.
│
├── prediction/        Inference driver prediction.py and per-dataset shells
│                      that generate report predictions on the test split for
│                      every trained checkpoint.
│
├── evaluation/        evaluation.py computes BLEU-1/4, ROUGE-L, CheXbert
│                      Micro-F1, and F1-RadGraph using the bundled rrg_eval/
│                      package.
│
├── environment.yml    Conda environment (PyTorch + transformers + trl +
│                      opacus + peft + huggingface_hub).
│
└── README.md          This file.
```

## Quickstart

Reproduce one row of Table 2 — RadSLDP on ReXGradient with Qwen2-VL-2B at
per-token ε = 8.

```bash
# 1. Environment
conda env create -f environment.yml
conda activate vlm

# 2. Fetch the NVCC taxonomy (see the "NVCC Taxonomy" section above)
python3 categorization/download_taxonomy.py

# 3. Download ReXGradient (requires Hugging Face access)
cd datasets/ReXGradient/download && bash download_rex.sh
cd ../processing && bash process_data.sh
cd ../../..

# 4. Train: all 4 regimes (No-DP, DP-SGD, LDP, RadSLDP) x 3 models on REX
cd training
bash run_training_rex.sh
# Checkpoints land in ../checkpoints/<model-name>-rex-<regime>_final/

# 5. Predict on the test split for every trained checkpoint
cd ../prediction
bash run_prediction_rex.sh
# Predictions land in ./predictions/<model-name>-rex-<regime>

# 6. Evaluate (BLEU / ROUGE / CheXbert / RadGraph)
cd ../evaluation
bash run_evaluation_rex.sh
# Per-run metrics land in ./evaluations/<model-name>-rex-<regime>/
```

Substitute `chex` or `mimic` for `rex` to reproduce the other two columns of
Table 2. Substitute `ablation` for the ε-sweep (Table 3, ReXGradient only).

### Running a single configuration

If you only want one specific run rather than the full sweep, call the trainer
directly:

```bash
cd training
torchrun --nproc_per_node=8 hf_finetune.py \
    --model_id Qwen/Qwen2-VL-2B-Instruct \
    --use_lora \
    --train_image_folder ../datasets/ReXGradient/download/ReXGradient \
    --train_dataset ../datasets/ReXGradient/download/ReXGradient/metadata/rexgradient_train.json \
    --output_dir ../checkpoints/Qwen2-VL-2B-Instruct-rex-sldp_8_final \
    --target_epsilon 8 \
    --target_delta 2.8643446e-05 \
    --max_length 4096 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --use_ldp \
    --selective_ldp
```

Privacy regimes are selected by combining these flags:

| Regime           | Flags                                  |
|------------------|----------------------------------------|
| Non-private      | (no privacy flags)                     |
| DP-SGD           | `--use_dp`                             |
| Uniform LDP      | `--use_ldp`                            |
| **RadSLDP**      | `--use_ldp --selective_ldp`            |
| Clip only (ε=∞)  | `--use_ldp --selective_ldp --no_noise` |

## Citation

If you use this code or the **RadSLDP NVCC taxonomy** in your work, please cite:

```bibtex
@inproceedings{zhao2026radsldp,
  title     = {RadSLDP: Selective Local Differential Privacy for Radiology Vision-Language Models},
  author    = {Zhao, Konghao and Xu, Xinyang and Xu, Runhui and Natsuaki, Yutaka and Xiong, Wenjie and Karimireddy, Sai Praneeth and Liu, Ruishan},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
}
```
