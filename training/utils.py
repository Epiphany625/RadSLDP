from PIL import Image
import torch
from trl import SFTConfig, SFTTrainer
import gc
import time 
from transformers import LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration
from transformers import AutoProcessor

import json
import os

# run an inference
# from qwen_vl_utils import process_vision_info
from peft import LoraConfig
from tqdm import tqdm
import shortuuid

from PIL import Image

class UnifiedDataCollator:
    """
    - Uses processor.apply_chat_template (official template) for EACH model
    - Computes labels so loss is ONLY on assistant tokens
    - Supports multimodal (images)
    """
    def __init__(self, processor, max_length):
        self.MAX_LEN = max_length
        self.processor = processor
        # ensure pad token exists
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _load_images(self, image_paths):
        loaded = []
        for p in image_paths or []:
            if isinstance(p, str):
                loaded.append(Image.open(p).convert("RGB"))
            elif isinstance(p, Image.Image):
                loaded.append(p)
            else:
                loaded.append(p)  # already tensor/processed
        return loaded

    @torch.no_grad()
    def __call__(self, features):
        # Build per-sample tensors then pad manually (works across processors)
        all_input_ids, all_attn, all_labels = [], [], []
        extra = {}

        sample_ids = []
        for f in features:
            sample_ids.append(f.get("id") or f["messages"][0].get("sample_id"))
            msgs = f["messages"]
            images = self._load_images(f.get("images", []))

            # 1) full conversation (user + assistant) -> what we train on
            text_full = self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )

            # 2) prompt only (user) -> boundary where assistant begins
            # Use add_generation_prompt=True so template includes the "assistant prefix"
            text_prompt = self.processor.apply_chat_template(
                msgs[:1],
                tokenize=False,
                add_generation_prompt=True,
            )

            enc_full = self.processor(
                text=text_full,
                images=images if images else None,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.MAX_LEN
            )
            enc_prompt = self.processor(
                text=text_prompt,
                images=images if images else None,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.MAX_LEN
            )

            input_ids = enc_full["input_ids"].squeeze(0)
            attn = enc_full["attention_mask"].squeeze(0)

            # assistant tokens start at prompt length (in tokens)
            prompt_len = enc_prompt["input_ids"].squeeze(0).shape[0]

            labels = input_ids.clone()
            labels[:prompt_len] = -100  # mask prompt/user tokens
            # also mask padding later

            all_input_ids.append(input_ids)
            all_attn.append(attn)
            all_labels.append(labels)

            # carry other processor keys (pixel_values, image_sizes, etc.)
            for k, v in enc_full.items():
                if k in ("input_ids", "attention_mask"):
                    continue
                extra.setdefault(k, []).append(v)

        # Pad to max len (LEFT padding like you set)
        max_len = max(x.shape[0] for x in all_input_ids)
        pad_id = self.processor.tokenizer.pad_token_id

        def left_pad_1d(x, pad_value):
            pad_len = max_len - x.shape[0]
            if pad_len <= 0:
                return x
            return torch.cat([torch.full((pad_len,), pad_value, dtype=x.dtype), x], dim=0)

        batch_input_ids = torch.stack([left_pad_1d(x, pad_id) for x in all_input_ids], dim=0)
        batch_attn = torch.stack([left_pad_1d(x, 0) for x in all_attn], dim=0)
        batch_labels = torch.stack([left_pad_1d(x, -100) for x in all_labels], dim=0)

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attn,
            "labels": batch_labels,
            "sample_id": sample_ids
        }

        # Merge extra keys
        for k, vs in extra.items():
            # try cat along batch dim if tensor-like
            try:
                if isinstance(vs[0], torch.Tensor):
                    batch[k] = torch.cat(vs, dim=0)
                else:
                    batch[k] = vs
            except Exception:
                batch[k] = vs

        return batch

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)
    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def load_model(MODEL_ID, cache_dir=None):
    if "qwen2-vl" in MODEL_ID.lower():
        print("loading qwen model")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            # device_map="auto", #if you use torchrun multiple GPU, comment this line out
            dtype=torch.bfloat16,
            cache_dir = cache_dir,
            attn_implementation="sdpa"
        )
    elif "llava" in MODEL_ID.lower():
        print("loading llava model")
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            # device_map="auto", #if you use torchrun multiple GPU, comment this line out
            dtype=torch.bfloat16,
            cache_dir = cache_dir,
            attn_implementation="sdpa"
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    if "qwen2-vl" in MODEL_ID.lower():
        processor.image_processor.min_pixels = 256 * 28 * 28
        processor.image_processor.max_pixels = 512 * 28 * 28
        
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.truncation_side = "right"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor

def find_modules(model, model_name):
    if "llava" in model_name.lower():
        lora_targets = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        }
        full_training_targets = {"linear_1", "linear_2"}
        skip_name = "vision_tower"
    
    if "qwen" in model_name.lower():
        lora_targets = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        }
        full_training_targets = {"linear_fc1", "linear_fc2"}
        skip_name = "visual.blocks"

    lora_modules = []
    full_training_modules = []

    for name, module in model.named_modules():
        if skip_name in name:
            continue

        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in lora_targets:
                lora_modules.append(name)
            if suffix in full_training_targets:
                full_training_modules.append(name)


    return lora_modules, full_training_modules

def resolve_path(root_dir: str, path: str) -> str:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root_dir, path)

def parse_question_format(line: dict) -> tuple:
    """
    Parse question from different formats.
    Supports:
    1. Simple format: {"text": "...", "answer": "..."}
    2. Conversations format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

    Returns: (question_id, image_file, question_text, ground_truth)
    """
    # Get question ID
    question_id = line.get("question_id", line.get("id", None))

    # Get image file
    # image_file = line.get("image", None).replace("..", ".").replace(".jpg", ".png")
    image_file = line.get("image", None).replace("..", ".")


    # Parse question and answer based on format
    if "conversations" in line:
        # Conversations format
        conversations = line["conversations"]

        # Extract question from human turn
        question_text = None
        ground_truth = None

        for conv in conversations:
            if conv.get("from") == "human":
                # Remove <image> token if present
                question_text = conv.get("value", "").replace("<image>", "").strip()
            elif conv.get("from") == "gpt":
                ground_truth = conv.get("value", "").strip()

        if question_text is None:
            raise ValueError("No 'human' conversation found")

    else:
        # Simple format
        question_text = line.get("text", None)
        ground_truth = line.get("answer", line.get("ground_truth", None))

        if question_text is None:
            raise ValueError("No 'text' field found")

    return question_id, image_file, question_text, ground_truth


def load_data(IMAGE_FOLDER, QUESTION_FILE_PATH):
    res = []
    # load question data
    try:
        # Try loading as JSON array first
        with open(QUESTION_FILE_PATH, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                questions = json.loads(content)
            else:
                # Load as JSONL
                questions = [json.loads(line) for line in content.split('\n') if line.strip()]
    except json.JSONDecodeError:
        # Fallback to line-by-line JSONL parsing
        questions = [json.loads(q) for q in open(QUESTION_FILE_PATH, "r") if q.strip()]
    print(f"Loaded {len(questions)} questions from {QUESTION_FILE_PATH}")

    for line in tqdm(questions, desc="Loading Data"):
        idx, image_file, prompt, ground_truth = parse_question_format(line)
        if idx is None or image_file is None or prompt is None or ground_truth is None:
            print("error parsing. Skipping this sample... ")
            continue
        # print(idx, image_file, prompt, ground_truth)

        image_file_path = resolve_path(IMAGE_FOLDER, image_file)
        assert(image_file_path is not None)

        messages =  {
            "id": idx, # custom ID field that I defined for later processing
            "prompt": prompt, # define another prompt field for easier prompt retrieval during evaluation
            "images": [image_file_path],
            "ground_truth": ground_truth, 
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file_path,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                    "sample_id": idx
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": ground_truth
                        }
                    ],
                },
            ]
        }

        res.append(messages)

    return res
