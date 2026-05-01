"""
generate_predictions.py
-----------------------
Generate LLM predictions using Qwen3-4B (4-bit quantised, CUDA 12.4).

Option B: runs both thinking / no_thinking modes for every dataset
as a built-in ablation study.

Outputs (one per dataset × mode):
  data/processed/{dataset}_{mode}_predictions.jsonl

Each line:
  {
    "id":           str,
    "question":     str,
    "ground_truth": str | list[str],
    "prediction":   str,          ← clean final answer
    "raw_output":   str,          ← full decoded text (with <think> if present)
    "thinking_mode": "thinking" | "no_thinking",
    "dataset":      str,
    "task_type":    str
  }

Usage:
  python scripts/generate_predictions.py
  python scripts/generate_predictions.py --datasets sciq simpleqa
  python scripts/generate_predictions.py --thinking_modes no_thinking
"""

import argparse
import json
import os
import re
import sys

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# torch <2.5 is missing set_submodule, which transformers >=5.x needs for BnB quantisation
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
        if not target:
            raise ValueError("Cannot set the module itself using set_submodule.")
        atoms = target.split(".")
        mod = self
        for atom in atoms[:-1]:
            mod = getattr(mod, atom)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Model loading ────────────────────────────────────────────────────────────

def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    lcfg = cfg["llm"]
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(lcfg["bnb_4bit_compute_dtype"], torch.float16)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=lcfg["bnb_use_double_quant"],
        bnb_4bit_quant_type=lcfg["bnb_4bit_quant_type"],
    )


def load_model_and_tokenizer(cfg: dict):
    lcfg = cfg["llm"]
    print(f"Loading tokenizer:  {lcfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        lcfg["model_name"], trust_remote_code=True
    )

    print(f"Loading model (4-bit NF4, CUDA 12.4): {lcfg['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        lcfg["model_name"],
        quantization_config=build_bnb_config(cfg),
        device_map=lcfg["device_map"],
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")
    return model, tokenizer


# ── Prompt construction ──────────────────────────────────────────────────────

def build_prompt(question: str, task_type: str) -> str:
    if task_type == "short_form":
        return (
            "Answer the following question concisely. "
            "Provide only a short phrase or name — do not explain.\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return (
        "Answer the following question accurately and concisely.\n"
        f"Question: {question}\n"
        "Answer:"
    )


# ── Generation & output parsing ──────────────────────────────────────────────

def _strip_im_tokens(text: str) -> str:
    """Remove Qwen special IM tokens like <|im_end|>, <|endoftext|>, etc."""
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


def extract_answer(raw_output: str, thinking_mode: str) -> str:
    """
    For thinking mode: return text after the closing </think> tag.
    For no_thinking mode: return the raw output as-is (already cleaned).
    Falls back to the full output if no </think> tag is found.
    """
    if thinking_mode == "thinking":
        match = re.search(r"</think>(.*)", raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()
    return raw_output.strip()


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    enable_thinking: bool,
    max_new_tokens: int,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens; keep special tokens so we can
    # parse <think>...</think> manually.
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)
    return generated


# ── Inference loop ───────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    samples: list,
    thinking_mode: str,
    cfg: dict,
    out_path: str,
) -> None:
    enable_thinking = thinking_mode == "thinking"
    lcfg = cfg["llm"]
    max_new_tokens = (
        lcfg["thinking_max_new_tokens"] if enable_thinking else lcfg["max_new_tokens"]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc=f"[{thinking_mode}] Generating"):
            prompt = build_prompt(sample["question"], sample["task_type"])
            raw = generate_one(model, tokenizer, prompt, enable_thinking, max_new_tokens)
            raw_clean = _strip_im_tokens(raw)
            prediction = extract_answer(raw_clean, thinking_mode)

            record = {
                "id": sample["id"],
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "prediction": prediction,
                "raw_output": raw_clean,
                "thinking_mode": thinking_mode,
                "dataset": sample["dataset"],
                "task_type": sample["task_type"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Data loading ─────────────────────────────────────────────────────────────

NAME_TO_FILE = {
    "sciq": "sciq_data.jsonl",
    "simpleqa": "simpleqa_data.jsonl",
    "natural_questions": "natural_questions_data.jsonl",
    "truthfulqa": "truthfulqa_data.jsonl",
}


def load_dataset_samples(processed_dir: str, dataset_names: list) -> dict:
    """Returns dict mapping dataset_name → list of sample dicts."""
    all_samples = {}
    for name in dataset_names:
        path = os.path.join(processed_dir, NAME_TO_FILE[name])
        if not os.path.exists(path):
            print(
                f"[ERROR] {path} not found. Run scripts/prepare_data.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        all_samples[name] = samples
        print(f"  Loaded {len(samples):>4} samples from {name}")
    return all_samples


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate predictions with Qwen3-4B")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sciq", "simpleqa", "natural_questions", "truthfulqa"],
        choices=list(NAME_TO_FILE.keys()),
    )
    parser.add_argument(
        "--thinking_modes",
        nargs="+",
        default=None,  # defaults to config value
        choices=["thinking", "no_thinking"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = cfg["paths"]["processed_data"]
    os.makedirs(processed_dir, exist_ok=True)

    thinking_modes = args.thinking_modes or cfg["llm"]["thinking_modes"]

    print("Loading datasets …")
    dataset_samples = load_dataset_samples(processed_dir, args.datasets)

    model, tokenizer = load_model_and_tokenizer(cfg)

    for mode in thinking_modes:
        for dataset_name in args.datasets:
            samples = dataset_samples[dataset_name]
            out_path = os.path.join(
                processed_dir, f"{dataset_name}_{mode}_predictions.jsonl"
            )
            if os.path.exists(out_path):
                print(
                    f"[skip] {out_path} already exists "
                    "(delete the file to regenerate)."
                )
                continue
            print(f"\n── [{mode}] {dataset_name} ({len(samples)} samples) ──")
            run_inference(model, tokenizer, samples, mode, cfg, out_path)
            print(f"Saved → {out_path}")

    print("\nDone. Run scripts/encode_embeddings.py next.")


if __name__ == "__main__":
    main()
