"""
Re-evaluate existing similarity result CSVs with HHEM and write augmented files.

By default, outputs are written to `results/hhem_backfill/` to avoid touching the
original experiment artifacts.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.evaluation.hhem import attach_hhem_scores
from src.evaluation.hhem import load_hhem_model


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def main():
    parser = argparse.ArgumentParser(description="Backfill HHEM scores for old result CSVs")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(ROOT_DIR, "results"),
    )
    parser.add_argument(
        "--pattern",
        default="_similarity.csv",
        help="Only process files whose names end with this suffix.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "results", "hhem_backfill"),
    )
    parser.add_argument(
        "--hhem-model-path",
        default="/hpc2hdd/home/yuxuanzhao/xuhaodong/NLI Project v2/models/HHEM-2.1-Open",
    )
    parser.add_argument("--hhem-threshold", type=float, default=0.5)
    parser.add_argument("--hhem-batch-size", type=int, default=32)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the original CSVs instead of writing into output-dir.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_hhem_model(args.hhem_model_path)

    filenames = sorted(
        name
        for name in os.listdir(args.results_dir)
        if name.endswith(args.pattern) and "_with_hhem" not in name
    )

    summary_rows = []
    for filename in filenames:
        input_path = os.path.join(args.results_dir, filename)
        df = pd.read_csv(input_path)
        required = {"prediction", "ground_truth"}
        if not required.issubset(df.columns):
            print(f"[skip] missing required columns in {filename}")
            continue

        records = df.to_dict(orient="records")
        enriched = attach_hhem_scores(
            records,
            model=model,
            threshold=args.hhem_threshold,
            batch_size=args.hhem_batch_size,
        )
        out_df = pd.DataFrame(enriched)

        if args.in_place:
            output_path = input_path
        else:
            stem, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output_dir, f"{stem}_with_hhem{ext}")
        out_df.to_csv(output_path, index=False)

        summary = {
            "file": filename,
            "output_path": output_path,
            "n_samples": len(out_df),
            "hhem_mean_score": float(out_df["hhem_score"].mean()),
            "hhem_positive_rate": float(out_df["hhem_is_consistent"].mean()),
        }
        if "is_correct" in out_df.columns:
            correctness = out_df["is_correct"].astype(int).to_numpy()
            hhem_labels = out_df["hhem_is_consistent"].astype(int).to_numpy()
            summary["correctness_hhem_agreement"] = float(np.mean(correctness == hhem_labels))
        if {"similarity", "hhem_is_consistent"}.issubset(out_df.columns):
            similarity = out_df["similarity"].astype(float).to_numpy()
            hhem_labels = out_df["hhem_is_consistent"].astype(int).to_numpy()
            summary["similarity_auc_vs_hhem"] = safe_auc(hhem_labels, similarity)
        summary_rows.append(summary)
        print(f"[done] {filename} -> {output_path}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "hhem_backfill_summary.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
