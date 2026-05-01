"""
prepare_data.py
---------------
Download and pre-process QA datasets into a unified JSONL format.

Output file per dataset:  data/processed/{dataset}_data.jsonl
Each line:
  {
    "id":           str,
    "question":     str,
    "ground_truth": str | list[str],
    "dataset":      str,
    "task_type":    "short_form" | "long_form",
    ...optional extra fields...
  }

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --datasets sciq simpleqa
"""

import argparse
import json
import os
import random
import sys

import yaml
from datasets import load_dataset
from tqdm import tqdm

SEED = 42


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample(ds, sample_size: int, seed: int = SEED):
    """Randomly sub-sample a HuggingFace dataset if needed."""
    if sample_size <= 0 or sample_size >= len(ds):
        return ds
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return ds.select(indices[:sample_size])


# ── Per-dataset processors ───────────────────────────────────────────────────

def process_sciq(cfg: dict, out_dir: str) -> None:
    dcfg = cfg["datasets"]["sciq"]
    ds = load_dataset(dcfg["hf_id"], split=dcfg["split"])
    ds = _sample(ds, dcfg["sample_size"])

    out_path = os.path.join(out_dir, "sciq_data.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(ds, desc="SciQ")):
            record = {
                "id": f"sciq_{i}",
                "question": item["question"].strip(),
                "ground_truth": item["correct_answer"].strip(),
                "dataset": "sciq",
                "task_type": "short_form",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[SciQ]     {len(ds):>4} samples → {out_path}")


def process_simpleqa(cfg: dict, out_dir: str) -> None:
    dcfg = cfg["datasets"]["simpleqa"]
    ds = load_dataset(dcfg["hf_id"], split=dcfg["split"])
    ds = _sample(ds, dcfg["sample_size"])

    out_path = os.path.join(out_dir, "simpleqa_data.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(ds, desc="SimpleQA")):
            record = {
                "id": f"simpleqa_{i}",
                "question": item["problem"].strip(),
                "ground_truth": item["answer"].strip(),
                "dataset": "simpleqa",
                "task_type": "short_form",
                "metadata": item.get("metadata", {}),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[SimpleQA] {len(ds):>4} samples → {out_path}")


def process_natural_questions(cfg: dict, out_dir: str) -> None:
    """
    Uses nq_open (simplified NQ).
    Fields: question (str), answer (list[str]).
    Filters out samples that have no answers.
    """
    dcfg = cfg["datasets"]["natural_questions"]
    ds = load_dataset(dcfg["hf_id"], split=dcfg["split"])
    # Filter samples with at least one non-empty answer
    ds = ds.filter(lambda x: len(x["answer"]) > 0 and any(a.strip() for a in x["answer"]))
    ds = _sample(ds, dcfg["sample_size"])

    out_path = os.path.join(out_dir, "natural_questions_data.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(ds, desc="NQ")):
            record = {
                "id": f"nq_{i}",
                "question": item["question"].strip(),
                "ground_truth": [a.strip() for a in item["answer"] if a.strip()],
                "dataset": "natural_questions",
                "task_type": "long_form",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[NQ]       {len(ds):>4} samples → {out_path}")


def process_truthfulqa(cfg: dict, out_dir: str) -> None:
    dcfg = cfg["datasets"]["truthfulqa"]
    ds = load_dataset(dcfg["hf_id"], dcfg["hf_config"], split=dcfg["split"])
    # TruthfulQA: use all samples (no sub-sampling by default)
    if dcfg["sample_size"] > 0:
        ds = _sample(ds, dcfg["sample_size"])

    out_path = os.path.join(out_dir, "truthfulqa_data.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(ds, desc="TruthfulQA")):
            best = item.get("best_answer", "").strip()
            correct_list = [a.strip() for a in item.get("correct_answers", []) if a.strip()]
            # Ensure best_answer is first in the list without duplication
            if best and best not in correct_list:
                correct_list = [best] + correct_list
            elif not correct_list and best:
                correct_list = [best]

            record = {
                "id": f"tqa_{i}",
                "question": item["question"].strip(),
                "ground_truth": correct_list,
                "best_answer": best,
                "dataset": "truthfulqa",
                "task_type": "long_form",
                "category": item.get("category", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[TruthfulQA] {len(ds):>4} samples → {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

PROCESSORS = {
    "sciq": process_sciq,
    "simpleqa": process_simpleqa,
    "natural_questions": process_natural_questions,
    "truthfulqa": process_truthfulqa,
}


def main():
    parser = argparse.ArgumentParser(description="Prepare QA datasets")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PROCESSORS.keys()),
        choices=list(PROCESSORS.keys()),
        help="Which datasets to prepare (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["paths"]["processed_data"]
    os.makedirs(out_dir, exist_ok=True)

    for name in args.datasets:
        print(f"\n── Processing {name} ──")
        PROCESSORS[name](cfg, out_dir)

    print("\nDone. Run scripts/generate_predictions.py next.")


if __name__ == "__main__":
    main()
