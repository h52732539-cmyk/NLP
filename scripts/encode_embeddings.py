"""
encode_embeddings.py
--------------------
Encode prediction strings and ground-truth strings into dense embedding
vectors using three pretrained sentence-embedding models.

Handles per-model instruction prefixes (e5-base requires "passage: " prefix
for symmetric comparison; BGE and MiniLM need none).

Outputs (per dataset × thinking_mode × embedding_model):
  data/processed/{dataset}_{mode}_{emb_name}_pred.npy   shape: (N, D)
  data/processed/{dataset}_{mode}_{emb_name}_gt.npy     shape: (N, D)
  data/processed/{dataset}_{mode}_{emb_name}_ids.json   list of sample ids

Usage:
  python scripts/encode_embeddings.py
  python scripts/encode_embeddings.py --datasets sciq simpleqa --batch_size 128
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_predictions(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def gt_to_str(ground_truth) -> str:
    """Convert ground_truth (str or list[str]) to a single string."""
    if isinstance(ground_truth, list):
        return ground_truth[0].strip() if ground_truth else ""
    return str(ground_truth).strip()


def apply_prefix(texts: list, prefix: str) -> list:
    if not prefix:
        return texts
    return [prefix + t for t in texts]


# ── Encoding ─────────────────────────────────────────────────────────────────

def encode_texts(
    model: SentenceTransformer,
    texts: list,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )


# ── Entry point ──────────────────────────────────────────────────────────────

DATASET_FILES = {
    "sciq": "sciq",
    "simpleqa": "simpleqa",
    "natural_questions": "natural_questions",
    "truthfulqa": "truthfulqa",
}


def main():
    parser = argparse.ArgumentParser(description="Encode texts into embeddings")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_FILES.keys()),
        choices=list(DATASET_FILES.keys()),
    )
    parser.add_argument(
        "--thinking_modes",
        nargs="+",
        default=None,  # defaults to config value
        choices=["thinking", "no_thinking"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = cfg["paths"]["processed_data"]
    thinking_modes = args.thinking_modes or cfg["llm"]["thinking_modes"]

    for emb_cfg in cfg["embedding_models"]:
        emb_name = emb_cfg["name"]
        print(f"\n══ Embedding model: {emb_cfg['hf_id']} ({emb_name}) ══")
        model = SentenceTransformer(emb_cfg["hf_id"])

        for dataset_name in args.datasets:
            for mode in thinking_modes:
                pred_file = os.path.join(
                    processed_dir, f"{dataset_name}_{mode}_predictions.jsonl"
                )
                if not os.path.exists(pred_file):
                    print(f"  [skip] {pred_file} not found (run generate_predictions.py first)")
                    continue

                out_prefix = os.path.join(
                    processed_dir, f"{dataset_name}_{mode}_{emb_name}"
                )
                if os.path.exists(out_prefix + "_pred.npy"):
                    print(f"  [skip] {out_prefix}_pred.npy already exists")
                    continue

                records = load_predictions(pred_file)
                if not records:
                    print(f"  [skip] {pred_file} is empty")
                    continue

                predictions = [r["prediction"] for r in records]
                ground_truths = [gt_to_str(r["ground_truth"]) for r in records]
                ids = [r["id"] for r in records]

                pred_texts = apply_prefix(predictions, emb_cfg["prefix_passage"])
                gt_texts = apply_prefix(ground_truths, emb_cfg["prefix_passage"])

                print(f"  ── {dataset_name}/{mode}/{emb_name}  ({len(records)} samples)")

                print("    Encoding predictions …")
                pred_embs = encode_texts(
                    model, pred_texts, args.batch_size, emb_cfg["normalize"]
                )

                print("    Encoding ground truths …")
                gt_embs = encode_texts(
                    model, gt_texts, args.batch_size, emb_cfg["normalize"]
                )

                np.save(out_prefix + "_pred.npy", pred_embs)
                np.save(out_prefix + "_gt.npy", gt_embs)
                with open(out_prefix + "_ids.json", "w", encoding="utf-8") as f:
                    json.dump(ids, f)

                print(
                    f"    Saved: {out_prefix}_{{pred,gt}}.npy  "
                    f"shape={pred_embs.shape}  dim={pred_embs.shape[1]}"
                )

    print("\nDone. Run scripts/similarity_analysis.py next.")


if __name__ == "__main__":
    main()
