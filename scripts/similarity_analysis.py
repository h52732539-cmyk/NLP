"""
similarity_analysis.py
----------------------
Compute cosine similarity between prediction and ground-truth embeddings,
merge with correctness labels, and analyse whether similarity serves as a
reliable proxy for factual correctness.

Outputs:
  results/{dataset}_{mode}_{emb_name}_similarity.csv  ← per-sample scores
  results/{dataset}_{mode}_{emb_name}_roc.csv         ← ROC curve data
  results/summary_auc.csv                             ← AUC + threshold table
  results/summary_stats.csv                           ← group statistics

Usage:
  python scripts/similarity_analysis.py
  python scripts/similarity_analysis.py --datasets sciq --thinking_modes no_thinking
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, roc_curve

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.correctness import batch_evaluate


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Similarity ───────────────────────────────────────────────────────────────

def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity between two matrices.
    If embeddings are already L2-normalised, this reduces to a dot product.
    """
    norms_a = np.linalg.norm(a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True)
    a_normed = a / np.clip(norms_a, 1e-9, None)
    b_normed = b / np.clip(norms_b, 1e-9, None)
    return np.einsum("ij,ij->i", a_normed, b_normed)


# ── Threshold search ─────────────────────────────────────────────────────────

def find_optimal_threshold(
    scores: np.ndarray, labels: np.ndarray, n_steps: int = 200
) -> dict:
    """
    Grid search for the threshold that maximises Youden's J statistic
    (sensitivity + specificity − 1), and report F1 at that threshold.
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    best_j, best_thresh, best_f1 = -np.inf, float(thresholds[0]), 0.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0

        if j > best_j:
            best_j = j
            best_thresh = float(t)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            best_f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    return {
        "threshold": best_thresh,
        "youden_j": float(best_j),
        "f1_at_threshold": float(best_f1),
    }


# ── Group statistics ─────────────────────────────────────────────────────────

def compute_group_stats(similarities: np.ndarray, labels: np.ndarray) -> dict:
    correct = similarities[labels == 1]
    incorrect = similarities[labels == 0]

    def _stats(arr):
        if len(arr) == 0:
            return dict(n=0, mean=float("nan"), std=float("nan"),
                        median=float("nan"), q25=float("nan"), q75=float("nan"))
        return dict(
            n=len(arr),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            median=float(np.median(arr)),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
        )

    result = {}
    for k, v in _stats(correct).items():
        result[f"correct_{k}"] = v
    for k, v in _stats(incorrect).items():
        result[f"incorrect_{k}"] = v

    # Welch's t-test
    if len(correct) >= 2 and len(incorrect) >= 2:
        t_stat, p_val = scipy_stats.ttest_ind(correct, incorrect, equal_var=False)
        result["t_stat"] = float(t_stat)
        result["p_value"] = float(p_val)
    else:
        result["t_stat"] = float("nan")
        result["p_value"] = float("nan")

    return result


# ── Core analysis ────────────────────────────────────────────────────────────

def analyse_one(
    records: list,
    pred_embs: np.ndarray,
    gt_embs: np.ndarray,
    cfg: dict,
) -> tuple:
    """
    Returns (DataFrame of per-sample results, np.array similarities, np.array labels).
    """
    f1_thresh = cfg["correctness"]["long_form_f1_threshold"]
    evaluated = batch_evaluate(records, long_form_f1_threshold=f1_thresh)
    similarities = cosine_similarity_batch(pred_embs, gt_embs)
    labels = np.array([int(r["is_correct"]) for r in evaluated])

    rows = [
        {
            "id": r["id"],
            "question": r["question"],
            "prediction": r["prediction"],
            "ground_truth": str(r["ground_truth"]),
            "dataset": r["dataset"],
            "task_type": r["task_type"],
            "thinking_mode": r["thinking_mode"],
            "is_correct": int(r["is_correct"]),
            "correctness_score": r["correctness_score"],
            "similarity": float(similarities[i]),
        }
        for i, r in enumerate(evaluated)
    ]
    return pd.DataFrame(rows), similarities, labels


# ── Entry point ──────────────────────────────────────────────────────────────

DATASET_KEYS = ["sciq", "simpleqa", "natural_questions", "truthfulqa"]


def main():
    parser = argparse.ArgumentParser(description="Similarity analysis")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--datasets", nargs="+", default=DATASET_KEYS,
                        choices=DATASET_KEYS)
    parser.add_argument("--thinking_modes", nargs="+", default=None,
                        choices=["thinking", "no_thinking"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = cfg["paths"]["processed_data"]
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    thinking_modes = args.thinking_modes or cfg["llm"]["thinking_modes"]
    n_steps = cfg["similarity"]["threshold_n_steps"]

    auc_rows, stats_rows = [], []

    for dataset_name in args.datasets:
        for mode in thinking_modes:
            pred_file = os.path.join(
                processed_dir, f"{dataset_name}_{mode}_predictions.jsonl"
            )
            if not os.path.exists(pred_file):
                print(f"[skip] {pred_file} not found")
                continue

            records = []
            with open(pred_file, encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))

            for emb_cfg in cfg["embedding_models"]:
                emb_name = emb_cfg["name"]
                pfx = os.path.join(processed_dir, f"{dataset_name}_{mode}_{emb_name}")

                if not os.path.exists(pfx + "_pred.npy"):
                    print(f"  [skip] embeddings not found for {dataset_name}/{mode}/{emb_name}")
                    continue

                pred_embs = np.load(pfx + "_pred.npy")
                gt_embs = np.load(pfx + "_gt.npy")

                df, similarities, labels = analyse_one(records, pred_embs, gt_embs, cfg)

                # ── per-sample CSV ──────────────────────────────────────────
                csv_out = os.path.join(
                    results_dir, f"{dataset_name}_{mode}_{emb_name}_similarity.csv"
                )
                df.to_csv(csv_out, index=False)

                # ── ROC curve ───────────────────────────────────────────────
                n_pos = int(labels.sum())
                n_neg = int((labels == 0).sum())

                if n_pos > 0 and n_neg > 0:
                    auc = float(roc_auc_score(labels, similarities))
                    fpr, tpr, _ = roc_curve(labels, similarities)
                    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
                        os.path.join(
                            results_dir,
                            f"{dataset_name}_{mode}_{emb_name}_roc.csv",
                        ),
                        index=False,
                    )
                else:
                    auc = float("nan")

                # ── Threshold search ────────────────────────────────────────
                opt = find_optimal_threshold(similarities, labels, n_steps)

                # ── Group stats ─────────────────────────────────────────────
                gstats = compute_group_stats(similarities, labels)

                auc_rows.append(
                    {
                        "dataset": dataset_name,
                        "thinking_mode": mode,
                        "emb_model": emb_name,
                        "auc": auc,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                        **opt,
                    }
                )
                stats_rows.append(
                    {
                        "dataset": dataset_name,
                        "thinking_mode": mode,
                        "emb_model": emb_name,
                        **gstats,
                    }
                )

                print(
                    f"  [{dataset_name}/{mode}/{emb_name}]  "
                    f"AUC={auc:.4f}  "
                    f"correct_mean={gstats.get('correct_mean', float('nan')):.4f}  "
                    f"incorrect_mean={gstats.get('incorrect_mean', float('nan')):.4f}  "
                    f"p={gstats.get('p_value', float('nan')):.3e}"
                )

    # ── Summary tables ───────────────────────────────────────────────────────
    pd.DataFrame(auc_rows).to_csv(
        os.path.join(results_dir, "summary_auc.csv"), index=False
    )
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(results_dir, "summary_stats.csv"), index=False
    )
    print("\nSummary tables saved to results/")


if __name__ == "__main__":
    main()
