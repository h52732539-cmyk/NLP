"""
failure_analysis.py
-------------------
Identify and categorise failure cases where cosine-similarity-based
correctness prediction disagrees with the ground-truth label.

Failure types:
  False Positive (FP): high similarity → predicted correct, but actually wrong
  False Negative (FN): low  similarity → predicted wrong,   but actually correct

Failure causes (heuristic):
  - lexical_variation         : prediction and GT differ in surface form but
                                 share the same semantic meaning
  - semantic_ambiguity        : question allows multiple valid answers; the
                                 model chose a plausible but non-matching one
  - long_form_reasoning       : multi-sentence answer; simple cosine fails to
                                 capture reasoning complexity
  - other

Outputs:
  results/failure_cases_{dataset}_{mode}.csv   ← top-N FP + FN cases
  results/failure_summary.csv                  ← aggregated counts

NOTE on improvement direction
-------------------------------
The analysis here motivates moving from Euclidean cosine similarity to a
hyperbolic-space distance metric (Poincaré Ball), following:
  [1] He et al. HELM (arXiv:2505.24722, 2025)
  [2] Patil et al. Hierarchical Mamba (arXiv:2505.18973, 2025)
Implementation deferred until current experiments are complete.

Usage:
  python scripts/failure_analysis.py
  python scripts/failure_analysis.py --top_n 30 --datasets sciq simpleqa
"""

import argparse
import os
import re

import pandas as pd
import yaml


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Failure cause heuristics ─────────────────────────────────────────────────

def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.sub(r"[^\w\s]", "", a.lower()).split())
    tokens_b = set(re.sub(r"[^\w\s]", "", b.lower()).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def estimate_failure_cause(row: pd.Series) -> str:
    pred = str(row.get("prediction", "")).strip()
    gt = str(row.get("ground_truth", "")).strip()
    task_type = str(row.get("task_type", ""))
    question = str(row.get("question", "")).lower()

    pred_len = len(pred.split())
    gt_len = len(gt.split())

    # Long-form: structural complexity is the dominant cause
    if task_type == "long_form" or gt_len > 15:
        return "long_form_reasoning"

    overlap = _token_overlap(pred, gt)

    # Short answers with near-zero lexical overlap → likely phrasing variation
    if pred_len <= 8 and gt_len <= 8 and overlap < 0.15:
        return "lexical_variation"

    # Wh-questions about entities with short expected answers → ambiguity
    ambig_patterns = [
        r"\bwho\b", r"\bwhich\b", r"\bwhat (kind|type|sort)\b",
        r"\bname (the|a|an)\b",
    ]
    if gt_len <= 5 and any(re.search(p, question) for p in ambig_patterns):
        return "semantic_ambiguity"

    return "other"


# ── Failure extraction ───────────────────────────────────────────────────────

def extract_failures(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Use the optimal AUC threshold (median split if not available) to label
    FP / FN cases, then return the most extreme examples.
    """
    if df.empty:
        return df

    df = df.copy()
    threshold = df["similarity"].median()
    df["sim_predicted_correct"] = (df["similarity"] >= threshold).astype(int)

    fp_mask = (df["sim_predicted_correct"] == 1) & (df["is_correct"] == 0)
    fn_mask = (df["sim_predicted_correct"] == 0) & (df["is_correct"] == 1)

    df["failure_type"] = "none"
    df.loc[fp_mask, "failure_type"] = "false_positive"
    df.loc[fn_mask, "failure_type"] = "false_negative"

    failures = df[df["failure_type"] != "none"].copy()
    if failures.empty:
        return failures

    failures["failure_cause"] = failures.apply(estimate_failure_cause, axis=1)

    # Most extreme: FP sorted by ↓ similarity (most confusing high-sim wrong),
    #               FN sorted by ↑ similarity (most confusing low-sim correct)
    fp_top = (
        failures[failures["failure_type"] == "false_positive"]
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )
    fn_top = (
        failures[failures["failure_type"] == "false_negative"]
        .sort_values("similarity", ascending=True)
        .head(top_n)
    )
    return pd.concat([fp_top, fn_top], ignore_index=True)


# ── Entry point ──────────────────────────────────────────────────────────────

DATASET_KEYS = ["sciq", "simpleqa", "natural_questions", "truthfulqa"]


def main():
    parser = argparse.ArgumentParser(description="Failure case analysis")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--datasets", nargs="+", default=DATASET_KEYS,
                        choices=DATASET_KEYS)
    parser.add_argument("--thinking_modes", nargs="+", default=None,
                        choices=["thinking", "no_thinking"])
    parser.add_argument(
        "--top_n", type=int, default=50,
        help="Max failure cases to extract per type (FP or FN) per dataset/mode",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    thinking_modes = args.thinking_modes or cfg["llm"]["thinking_modes"]
    primary_emb = cfg["embedding_models"][0]["name"]   # bge-base for failure analysis

    summary_rows = []

    for dataset_name in args.datasets:
        for mode in thinking_modes:
            csv_path = os.path.join(
                results_dir, f"{dataset_name}_{mode}_{primary_emb}_similarity.csv"
            )
            if not os.path.exists(csv_path):
                print(
                    f"[skip] {csv_path} not found "
                    "(run scripts/similarity_analysis.py first)"
                )
                continue

            df = pd.read_csv(csv_path)
            failures = extract_failures(df, top_n=args.top_n)

            out_path = os.path.join(
                results_dir, f"failure_cases_{dataset_name}_{mode}.csv"
            )
            failures.to_csv(out_path, index=False)
            print(f"Saved {len(failures):>4} failure cases → {out_path}")

            # Aggregate for summary
            if not failures.empty:
                tc = failures["failure_type"].value_counts().to_dict()
                cc = failures["failure_cause"].value_counts().to_dict()
            else:
                tc, cc = {}, {}

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "thinking_mode": mode,
                    "emb_model": primary_emb,
                    "n_fp": tc.get("false_positive", 0),
                    "n_fn": tc.get("false_negative", 0),
                    "n_lexical_variation": cc.get("lexical_variation", 0),
                    "n_semantic_ambiguity": cc.get("semantic_ambiguity", 0),
                    "n_long_form_reasoning": cc.get("long_form_reasoning", 0),
                    "n_other": cc.get("other", 0),
                    # FP stats
                    "fp_sim_mean": (
                        failures[failures["failure_type"] == "false_positive"][
                            "similarity"
                        ].mean()
                        if not failures.empty
                        else float("nan")
                    ),
                    "fn_sim_mean": (
                        failures[failures["failure_type"] == "false_negative"][
                            "similarity"
                        ].mean()
                        if not failures.empty
                        else float("nan")
                    ),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(results_dir, "failure_summary.csv"), index=False
    )
    print("\nFailure summary → results/failure_summary.csv")
    print(
        "\n[Note] Improvement direction: Hyperbolic embedding distance (Poincaré Ball)\n"
        "  See: He et al. HELM (arXiv:2505.24722) and Patil et al. (arXiv:2505.18973).\n"
        "  Implementation pending current experiment results."
    )


if __name__ == "__main__":
    main()
