# from datasets import load_dataset
import re
import math
import os
import torch
from trl import SFTConfig, SFTTrainer
import gc
import time 
from transformers import AutoProcessor
from datasets import Dataset
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
import torch.nn as nn


import random
import numpy as np
from transformers import set_seed

from PIL import Image

from peft import LoraConfig

from utils import clear_memory, load_model, load_data, find_modules, UnifiedDataCollator
from arg_parse import parse_args
from tqdm import tqdm
import json
import shortuuid

from peft.tuners.lora import LoraLayer
from peft import PeftModel

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.optimizers import DPOptimizer

import os
import torch


def unwrap_model(m):
    """
    Unwrap only training wrappers, but DO NOT unwrap PEFT.
    This keeps PeftModel intact so save_model() writes adapter_config.json.
    """
    seen = set()
    while True:
        mid = id(m)
        if mid in seen:
            break
        seen.add(mid)

        # unwrap DDP/DPDDP
        if hasattr(m, "module") and m.module is not None:
            m = m.module
            continue

        # unwrap Opacus GradSampleModule
        if hasattr(m, "_module") and m._module is not None:
            m = m._module
            continue

        # STOP if it's a PEFT model
        if isinstance(m, PeftModel):
            break

        break
    return m

class BF16SafeDPOptimizer(DPOptimizer):
    """
    Version-agnostic bf16 fix:
    - compute per-sample grad norms ourselves
    - compute clip factors ourselves
    - do clipping accumulation in fp32
    - add noise in fp32, write p.grad in param dtype
    """

    def _get_per_sample_norms(self):
        # returns (B,) fp32
        per_sample_sq = None
        B = None

        for p in self.params:
            gs = getattr(p, "grad_sample", None)
            if gs is None:
                continue

            # grad_sample can be Tensor or list[Tensor]
            if isinstance(gs, list):
                gs_list = gs
            else:
                gs_list = [gs]

            for g in gs_list:
                if g is None:
                    continue
                if B is None:
                    B = g.shape[0]

                g32 = g.detach().to(torch.float32)
                flat = g32.view(g32.shape[0], -1)  # (B, *)
                sq = (flat * flat).sum(dim=1)      # (B,)

                per_sample_sq = sq if per_sample_sq is None else (per_sample_sq + sq)

        if per_sample_sq is None:
            return None  # no grads
        return per_sample_sq.sqrt()  # (B,)

    def clip_and_accumulate(self):
        norms = self._get_per_sample_norms()
        if norms is None:
            return

        # clip_factor: (B,) fp32
        clip_factor = (self.max_grad_norm / (norms + 1e-6)).clamp(max=1.0).to(torch.float32)

        for p in self.params:
            gs = getattr(p, "grad_sample", None)
            if gs is None:
                continue

            if isinstance(gs, list):
                gs_list = gs
            else:
                gs_list = [gs]

            summed = None
            for g in gs_list:
                if g is None:
                    continue
                g32 = g.detach().to(torch.float32)
                part = torch.einsum("i,i...", clip_factor, g32)  # fp32-safe
                summed = part if summed is None else (summed + part)

            if summed is None:
                continue

            if getattr(p, "summed_grad", None) is None:
                p.summed_grad = summed
            else:
                p.summed_grad += summed

    def add_noise(self):
        for p in self.params:
            if getattr(p, "summed_grad", None) is None:
                continue

            noise = torch.normal(
                mean=0.0,
                std=float(self.noise_multiplier * self.max_grad_norm),
                size=p.summed_grad.shape,
                device=p.summed_grad.device,
                dtype=torch.float32,
            )
            noisy = p.summed_grad + noise
            p.grad = noisy.view_as(p).to(dtype=p.dtype)


class SelectiveEmbeddingNoise:
    """Add noise to embeddings ONLY where sensitive_mask=True (B,L)."""
    def __init__(self, processor, noise_std=0.05, clip_norm=None, no_noise=False):
        self.processor = processor
        self.noise_std = noise_std
        self.clip_norm = clip_norm
        self.no_noise = no_noise
        self.sensitive_mask = None  # (B,L) bool set by trainer

    def __call__(self, module, inputs, output):
        # embedding forward hook: inputs[0]=input_ids (B,L), output=(B,L,D)
        if not module.training:
            return output
        if self.sensitive_mask is None:
            return output
        if self.sensitive_mask.shape[:2] != output.shape[:2]:
            return output

        mask = self.sensitive_mask
        idx = mask[0].nonzero(as_tuple=True)[0]
        # print("NUM MASKED TOKENS:", idx.numel())

        # if idx.numel() > 0:
        #     idx10 = idx[:]   # 👈 only first 10
        #     token_ids = inputs[0][0][idx10]
        #     print("TEXT BEING NOISED (first 10 tokens):")
        #     print(self.processor.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True))
        # else:
        #     print("TEXT BEING NOISED: (none)")

        if not mask.any():
            return output

        if not mask.any():
            return output

        out = output

        if self.clip_norm is not None:
            out_fp32 = out.float()
            sel = out_fp32[mask]
            norms = sel.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
            scale = (self.clip_norm / norms).clamp_max(1.0)
            out_fp32[mask] = sel * scale
            out = out_fp32.to(dtype=out.dtype)

        if not self.no_noise:
            noise = torch.randn((mask.sum().item(), out.shape[-1]),
                                device=out.device, dtype=out.dtype) * self.noise_std
            out = out.clone()
            out[mask] = out[mask] + noise
        else:
            out = out.clone() 

        return out

def get_base_model_for_embeddings(model):
    if hasattr(model, "module"):
        model = model.module

    if hasattr(model, "get_input_embeddings"):
        return model

    if hasattr(model, "base_model") and hasattr(model.base_model, "get_input_embeddings"):
        return model.base_model

    return model  

def setup_ldp_noise_insertion(model, processor, noise_std=0.05, clip_norm=None, no_noise=False):
    noise_module = SelectiveEmbeddingNoise(processor, noise_std=noise_std, clip_norm=clip_norm, no_noise=no_noise)

    base = get_base_model_for_embeddings(model)
    embed_layer = base.get_input_embeddings()
    if embed_layer is None:
        print("Warning: get_input_embeddings() returned None")
        return None, None

    handle = embed_layer.register_forward_hook(noise_module)
    return noise_module, handle

class LDPSFTTrainer(SFTTrainer):
    def __init__(self, *args, ldp_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ldp_config = ldp_config
        self.noise_module = None
        self.noise_handle = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        # 1) Register hook once, on the REAL wrapped model (DDP-safe)
        if self.ldp_config is not None and self.noise_module is None:
            self.noise_module, self.noise_handle = setup_ldp_noise_insertion(
                model,                  # <-- IMPORTANT: use the arg, not self.model
                self.processing_class,  # <-- your processor
                noise_std=self.ldp_config.get("noise_std"),
                clip_norm=self.ldp_config.get("clip_norm"),
                no_noise=self.ldp_config.get("no_noise", False)
            )

        sample_ids = inputs.pop("sample_id", None)  # <-- NEW: remove before forward()

        if self.noise_module is not None and "input_ids" in inputs:
            if self.ldp_config.get("selective"):
                # Selective LDP - ONLY tokens corresponding to reason_text from your JSON (+ abbrev variants)
                self.noise_module.sensitive_mask = build_policy_mask_from_ids(
                    processor=self.processing_class,
                    input_ids=inputs["input_ids"],
                    sample_ids=sample_ids,
                    id_to_cat=ID_TO_CAT,
                    abbrev_map=ABBREV_MAP,
                )
            else:
                self.noise_module.sensitive_mask = build_user_input_mask(self.processing_class, inputs["input_ids"])

        return super().training_step(model, inputs, num_items_in_batch)

class MatchTypeDPOptimizer(DPOptimizer):
    """DPOptimizer that handles bf16 gradients"""
    def add_noise(self):
        for p in self.params:
            if p.grad is not None and p.summed_grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=p.summed_grad.shape,
                    device=p.summed_grad.device,
                    dtype=p.summed_grad.dtype,  # Match dtype
                )
                noisy_grad = (p.summed_grad + noise).view_as(p)
                p.grad = noisy_grad.to(p.dtype) 

class DPSFTTrainer(SFTTrainer):
    def __init__(self, *args, dp_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_config = dp_config
        self.privacy_engine = None

    def _cast_module_bf16_(self, m: torch.nn.Module):
    # cast parameters
        for p in m.parameters(recurse=True):
            if p is not None and p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        # cast buffers (layernorm running stats, etc.)
        for name, b in m.named_buffers(recurse=True):
            if b is not None and b.dtype != torch.bfloat16:
                setattr(m, name, b.to(torch.bfloat16))

    def _setup_dp(self):
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        self.model = self.model.to("cuda")

        train_dataloader = self.get_train_dataloader()

        self.privacy_engine = PrivacyEngine()

        out = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=train_dataloader,
            target_epsilon=self.dp_config['target_epsilon'],
            target_delta=self.dp_config['target_delta'],
            epochs=self.args.num_train_epochs,
            max_grad_norm=self.dp_config['max_grad_norm'],
            poisson_sampling=True,
        )

        if len(out) == 3:
            self.model, self.optimizer, self._dp_train_dataloader = out
        elif len(out) == 4:
            self.model, self.optimizer, self._dp_train_dataloader, self.privacy_engine = out
        self.model = DPDDP(self.model)
        
        self.optimizer.__class__ = BF16SafeDPOptimizer

    def train(self, *args, **kwargs):
        if self.dp_config is not None:
            if self.optimizer is None:
                self.create_optimizer()
            self._setup_dp()
        return super().train(*args, **kwargs)
    
    def get_privacy_spent(self):
        if self.privacy_engine is None:
            return None
        return self.privacy_engine.get_epsilon(self.dp_config['target_delta'])
    
    def save_model(self, output_dir=None, _internal_call=False):
        if self.dp_config is not None:
            orig = self.model
            try:
                self.model = unwrap_model(self.model)
                return super().save_model(output_dir, _internal_call)
            finally:
                self.model = orig
        return super().save_model(output_dir, _internal_call)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Clear grad_sample before each step for Opacus
        if self.dp_config is not None:
            self.optimizer.zero_grad()
            
            # Fix dtype mismatch - cast pixel_values to model dtype
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                inputs['pixel_values'] = inputs['pixel_values'].to(dtype=torch.bfloat16)
            if 'pixel_values_videos' in inputs and inputs['pixel_values_videos'] is not None:
                inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(dtype=torch.bfloat16)
                
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, *args, **kwargs):
        if self.dp_config is not None and self.privacy_engine is not None:
            eps = self.privacy_engine.get_epsilon(self.dp_config['target_delta'])
            logs['epsilon'] = eps
        super().log(logs, *args, **kwargs)
    
    def _save_checkpoint(self, model, trial):
        if self.dp_config is not None:
            orig = self.model
            try:
                self.model = unwrap_model(self.model)
                return super()._save_checkpoint(unwrap_model(model), trial)
            finally:
                self.model = orig
        return super()._save_checkpoint(model, trial)

    def create_model_card(self, **kwargs):
        if self.dp_config is not None:
            orig = self.model
            try:
                base = unwrap_model(self.model)
                if not hasattr(base, "config"):
                    if self.is_world_process_zero():
                        print("[WARN] Skipping model card: unwrapped model still has no `.config`.")
                    return
                self.model = base
                return super().create_model_card(**kwargs)
            finally:
                self.model = orig
        return super().create_model_card(**kwargs)

def build_full_mask(input_ids: torch.Tensor):
    """Mask ALL tokens for pure LDP"""
    B, L = input_ids.shape
    return torch.ones((B, L), dtype=torch.bool, device=input_ids.device)

def build_user_input_mask(processor, input_ids: torch.Tensor):
    """Mask USER input only (not ASSISTANT response) for pure LDP"""
    B, L = input_ids.shape
    device = input_ids.device
    mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    
    ids_cpu = input_ids.detach().cpu()
    
    # Try multiple marker formats (different models use different formats)
    end_markers = ["ASSISTANT:", "assistant:", "assistant\n", "assistant", "<|assistant|>"]
    start_markers = ["USER:", "user:", "user\n", "user", "<|user|>"]
    
    for b in range(B):
        # Find USER start
        user_start = 0
        found_user = False
        for marker in start_markers:
            if found_user:
                break
            marker_ids = processor.tokenizer.encode(marker, add_special_tokens=False)
            if len(marker_ids) == 0:
                continue
            marker_tensor = torch.tensor(marker_ids, dtype=ids_cpu.dtype)
            for i in range(L - len(marker_ids) + 1):
                if torch.equal(ids_cpu[b, i:i+len(marker_ids)], marker_tensor):
                    user_start = i + len(marker_ids)
                    found_user = True
                    break
        
        # Find ASSISTANT start
        assistant_start = L
        found_assistant = False
        for marker in end_markers:
            if found_assistant:
                break
            marker_ids = processor.tokenizer.encode(marker, add_special_tokens=False)
            if len(marker_ids) == 0:
                continue
            marker_tensor = torch.tensor(marker_ids, dtype=ids_cpu.dtype)
            for i in range(user_start, L - len(marker_ids) + 1):
                if torch.equal(ids_cpu[b, i:i+len(marker_ids)], marker_tensor):
                    assistant_start = i
                    found_assistant = True
                    break
        
        # Mask between user and assistant
        mask[b, user_start:assistant_start] = True
    
    return mask

def build_policy_mask_from_ids(processor, input_ids: torch.Tensor, sample_ids, id_to_cat: dict, abbrev_map: dict):
    """
    Build (B,L) sensitive mask by:
      1) locating the raw indication span text between markers in decoded text
      2) doing exact text matching (word-boundary) on that raw span
      3) for each text match, encode the exact matched substring (incl. trailing punctuation) and
         find it in token space to mark the mask.

    Also masks demographics tokens:
      - male/female
      - numbers (all digit runs)
      - race: White, Asian, Black, Pacific Islander, Native American
      - ethnicity: Hispanic/Latino, Non-Hispanic/Non-Latino
    (Exact match for each phrase; NOT masking Unknown / Patient Refused / Other)
    """
    B, L = input_ids.shape
    device = input_ids.device
    mask = torch.zeros((B, L), dtype=torch.bool, device=device)

    if sample_ids is None:
        return mask

    # normalize ids list length
    if not isinstance(sample_ids, (list, tuple)):
        sample_ids = [sample_ids] * B
    sample_ids = [None if x is None else str(x) for x in sample_ids]

    ids_cpu = input_ids.detach().cpu()

    # token boundaries for indication span (token-space)
    start_marker = "given the following indications:"
    end_marker = "ASSISTANT:"
    start_ids = processor.tokenizer.encode(start_marker, add_special_tokens=False)
    end_ids = processor.tokenizer.encode(end_marker, add_special_tokens=False)
    start_tensor = torch.tensor(start_ids, dtype=ids_cpu.dtype)
    end_tensor = torch.tensor(end_ids, dtype=ids_cpu.dtype)

    # cache tokenized sequences for substrings we want to mark
    cache = {}  # exact_substring -> list[tensor token_ids]

    def norm(s: str) -> str:
        return " ".join(str(s).lower().split())

    def resolve_pid(sid: str):
        if sid is None:
            return None
        if sid in id_to_cat:
            return sid
        rex = f"rex:{sid}"
        if rex in id_to_cat:
            return rex
        mimic = f"mimic:{sid}"
        if mimic in id_to_cat:
            return mimic
        chexpert = f"chexpert:{sid}".replace(".png", ".jpg")
        if chexpert in id_to_cat:
            return chexpert
        return None

    def gen_variants(reason_text: str):
        """canonical + abbreviation substitution variants (lowercased)."""
        base = norm(reason_text)
        variants = {base}
        for canon, abbrs in (abbrev_map or {}).items():
            canon_n = norm(canon)
            if canon_n in base:
                for abbr in abbrs:
                    variants.add(base.replace(canon_n, norm(abbr)))
        return list(variants)

    def get_token_seqs(exact_substring: str):
        """
        Tokenize *exact* substring (preserving casing/punct) and also " "+substring,
        so we can match either tokenization with/without leading space token.
        """
        if exact_substring not in cache:
            seqs = []
            for v in (exact_substring, " " + exact_substring):
                enc = processor.tokenizer.encode(v, add_special_tokens=False)
                if enc:
                    seqs.append(torch.tensor(enc, device=device, dtype=input_ids.dtype))
            cache[exact_substring] = seqs
        return cache[exact_substring]

    # demographics / protected tokens (text-level exact)
    DEMO_PHRASES = [
        # sex
        "male", "female",
        # ethnicity
        "hispanic/latino", "non-hispanic/non-latino",
        # race
        "white", "asian", "black", "pacific islander", "native american",
    ]
    # compile regex for word-boundary-ish matching on lowercased span
    def boundary_pat(term_lower: str):
        return re.compile(rf"(?<![a-z0-9]){re.escape(term_lower)}(?![a-z0-9])")

    DEMO_PATS = [boundary_pat(t) for t in DEMO_PHRASES]
    NUM_PAT = re.compile(r"(?<![0-9])([0-9]+)(?![0-9])")  # digit runs, e.g. 65, 2017

    # punctuation we allow to “attach” to a matched token span
    PUNCT_AFTER = set(". , ; :".split())

    for b in range(B):
        pid = resolve_pid(sample_ids[b])
        if pid is None:
            continue  # training patient id not in file => skip

        # find indication span in token indices (token-space)
        search_start = 0
        if start_ids:
            for i in range(L - len(start_ids) + 1):
                if torch.equal(ids_cpu[b, i:i+len(start_ids)], start_tensor):
                    search_start = i + len(start_ids)
                    break

        search_end = L
        if end_ids:
            for i in range(search_start, L - len(end_ids) + 1):
                if torch.equal(ids_cpu[b, i:i+len(end_ids)], end_tensor):
                    search_end = i
                    break

        if search_end <= search_start:
            continue

        # === RAW SPAN TEXT (this is what you wanted matching on) ===
        span_text_raw = processor.tokenizer.decode(
            ids_cpu[b, search_start:search_end].tolist(),
            skip_special_tokens=False
        )
        span_low = span_text_raw.lower()

        ids_b = input_ids[b]

        # helper: mark occurrences found in RAW span text by converting them to token matches
        def mark_text_match(char_s: int, char_e: int):
            # extend with immediate trailing punctuation (.,;:) repeated
            end = char_e
            while end < len(span_text_raw) and span_text_raw[end] in PUNCT_AFTER:
                end += 1

            exact_sub = span_text_raw[char_s:end]  # preserves casing/punct
            if not exact_sub.strip():
                return

            # tokenize exact substring and find in token space inside [search_start, search_end)
            for seq in get_token_seqs(exact_sub):
                n = seq.numel()
                if n == 0 or (search_end - search_start) < n:
                    continue
                for ti in range(search_start, search_end - n + 1):
                    if torch.equal(ids_b[ti:ti+n], seq):
                        mask[b, ti:ti+n] = True

        # 1) demographics: numbers + fixed phrases
        for m in NUM_PAT.finditer(span_low):
            mark_text_match(m.start(1), m.end(1))

        for pat in DEMO_PATS:
            for m in pat.finditer(span_low):
                mark_text_match(m.start(), m.end())

        # 2) JSON-driven sensitive parts (image_inferable == "no")
        reasons = id_to_cat[pid].get("reasons", []) or []
        for r in reasons:
            parts = r.get("parts", []) or []
            if not parts:
                continue

            for p in parts:
                if norm(p.get("image_inferable", "")) != "no":
                    continue  # only sensitive parts

                rt = (p.get("part_text", "") or r.get("reason_text", "") or "").strip()
                if not rt:
                    continue

                # generate canonical + abbrev variants (lower)
                for v in gen_variants(rt):
                    if not v:
                        continue
                    # exact word-boundary match on RAW span (lowercased)
                    pat = boundary_pat(v)
                    for m in pat.finditer(span_low):
                        mark_text_match(m.start(), m.end())

    return mask

def compute_noise_std(clip_norm, target_epsilon, target_delta):
    """Compute required noise_std for target epsilon"""
    sensitivity = 2 * clip_norm
    noise_std = (sensitivity / target_epsilon) * math.sqrt(2 * math.log(1.25 / target_delta))
    return noise_std

def main():
    args = parse_args()
    seed = args.seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    model, processor = load_model(args.model_id)
    lora_modules, full_training_modules = find_modules(model, args.model_id)

    train_dataset = load_data(args.train_image_folder, args.train_dataset)
        
    if args.use_lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=32, #32
            bias="none",
            target_modules=lora_modules,
            modules_to_save=full_training_modules,
            task_type="CAUSAL_LM",
        )
    
    dp_config = None
    if args.use_dp:
        dp_config = {
            'target_epsilon': args.target_epsilon,
            'target_delta': args.target_delta,
            'max_grad_norm': args.max_grad_norm,
        }

    ldp_config = None
    if args.use_ldp:
        noise_std = compute_noise_std(args.clip_norm, args.target_epsilon, args.target_delta)
        print(f"LDP: ε={args.target_epsilon}, δ={args.target_delta:.2e}, computed noise_std={noise_std:.4f}")
        ldp_config = {
            "noise_std": noise_std,  # map multiplier -> std
            "clip_norm": args.clip_norm, 
            "selective": args.selective_ldp,
            "no_noise": args.no_noise,
        }

    use_gradient_checkpointing = False if args.use_dp else True

    max_steps = -1
    if args.use_dp:  # poisson_sampling=True in Opacus
        world_size = int(os.environ.get("WORLD_SIZE", "1"))  # torchrun sets this
        N = len(train_dataset)  # 34912
        denom = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        steps_per_epoch = math.ceil(N / denom)
        max_steps = steps_per_epoch * args.num_train_epochs
        
    training_args = SFTConfig(
        output_dir=args.output_dir, 
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        
        logging_steps=50,
        logging_strategy="steps",
        logging_dir=os.path.join(args.output_dir, "logs"),

        eval_strategy="no",        
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        seed=seed,
        gradient_checkpointing=use_gradient_checkpointing,
        ddp_find_unused_parameters = False,

        max_steps=max_steps,
        # max_steps=20,
        save_strategy="no",
    )

    if args.use_dp:
        TrainerClass = DPSFTTrainer
    elif args.use_ldp:
        TrainerClass = LDPSFTTrainer
    else:
        TrainerClass = SFTTrainer

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": processor,
    }

    # LoRA
    if args.use_lora:
        trainer_kwargs["peft_config"] = peft_config

    trainer_kwargs["data_collator"] =  UnifiedDataCollator(processor, args.max_length)
    # ONLY pass dp_config to DPSFTTrainer
    if args.use_dp:
        trainer_kwargs["dp_config"] = dp_config

    if args.use_ldp:
        trainer_kwargs["ldp_config"] = ldp_config


    trainer = TrainerClass(**trainer_kwargs)

    trainer.train()
    
    if args.use_dp:
        print(f"Final epsilon: {trainer.get_privacy_spent():.4f}")
    
    trainer.save_model(training_args.output_dir)
    print("Model saved to: " + training_args.output_dir)

if __name__ == "__main__":
    # Parse first so --help / argument errors short-circuit before the
    # ~640 MB taxonomy file is read from disk.
    args = parse_args()

    ID_TO_CAT_JSON = "../categorization/RadSLDP_nvcc_taxonomy.json"
    ABBREV_MAP = {
        "chronic obstructive pulmonary disease": ["copd"],
        "congestive heart failure": ["chf"],
        "shortness of breath": ["sob"],
        "myocardial infarction": ["mi"],
        "shortness of breath chest pain": ["cp"],
        "nasogastric":["ng"],
        "rule out":["r/o"],
        "endotracheal":["et"],
        "chest pain":["cp"],
        "status post":["s/p"],
        "pneumothorax":["ptx"],
        "chest x-ray":["cxr"],
        "altered mental status": ["ams"],
        "urinary tract infection": ["uti"],
        "pneumonia": ["pna"],
        "evaluate": ["eval"],
        "rule out congestive heart failure": ["R/O Chf"],    
        "coronary artery disease": ["cad"],    
        "rule out pneumonia": ["R/O Pna"],
        "intensive care unit": ["icu"],
        "chronic obstructive pulmonary disease": ["copd"],
        "evaluate for pneumonia": ["eval for pna", "eval pneumonia"],
        "evaluate for congestive heart failure": ["eval for chf"]
    }

    if not os.path.exists(ID_TO_CAT_JSON):
        raise SystemExit(
            f"NVCC taxonomy not found at {ID_TO_CAT_JSON}\n"
            "Fetch it from Hugging Face with:\n"
            "    python3 ../categorization/download_taxonomy.py\n"
            "or regenerate it via categorization/apply_expert_corrections.py."
        )
    with open(ID_TO_CAT_JSON, "r") as f:
        ID_TO_CAT = json.load(f)
    main()