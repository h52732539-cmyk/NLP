"""
Summarize the two added experiment outputs:

1. NLP/scripts/evaluate_ta_baseline_compatibility.py artifacts under
   data/processed/ta_baseline_compatibility/.
2. NLI Project v2/run_all_datasets.sh artifacts under
   ../NLI Project v2/results/.

The script is intentionally read-only with respect to experiment artifacts. It writes
compact CSV/Markdown summaries under results/added_experiments/.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


NLP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = NLP_ROOT.parent
NLI_ROOT = WORKSPACE_ROOT / "NLI Project v2"
OUT_DIR = NLP_ROOT / "results" / "added_experiments"


def normalize_answer(text: Any) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_pickle_stream(path: Path) -> list[Any]:
    chunks = []
    with path.open("rb") as f:
        while True:
            try:
                chunks.append(pickle.load(f))
            except EOFError:
                return chunks


def summarize_ta_compatibility() -> list[dict[str, Any]]:
    processed_dir = NLP_ROOT / "data" / "processed" / "ta_baseline_compatibility"
    rows: list[dict[str, Any]] = []
    for path in sorted(processed_dir.glob("*_no_thinking_predictions.jsonl")):
        dataset = path.name.removesuffix("_no_thinking_predictions.jsonl")
        records = load_jsonl(path)
        exact_flags = []
        contains_flags = []
        correct_flags = []
        pred_word_counts = []

        for record in records:
            pred = normalize_answer(record.get("prediction", ""))
            gt = normalize_answer(record.get("ground_truth", ""))
            exact = bool(gt) and pred == gt
            contains = bool(gt) and (gt in pred or pred in gt)
            exact_flags.append(exact)
            contains_flags.append(contains)
            correct_flags.append(exact or contains)
            pred_word_counts.append(len(str(record.get("prediction", "")).split()))

        rows.append(
            {
                "dataset": dataset,
                "mode": "no_thinking",
                "n_predictions": len(records),
                "exact_rate": float(np.mean(exact_flags)) if records else np.nan,
                "contains_rate": float(np.mean(contains_flags)) if records else np.nan,
                "correct_rate": float(np.mean(correct_flags)) if records else np.nan,
                "prediction_words_mean": float(np.mean(pred_word_counts)) if records else np.nan,
                "prediction_words_median": float(np.median(pred_word_counts)) if records else np.nan,
                "prediction_words_max": int(max(pred_word_counts)) if records else 0,
                "note": "Generated predictions exist; embedding/HHEM summary CSV was not present.",
            }
        )
    return rows


def _enable_cpu_unpickle_for_torch_tensors() -> bool:
    try:
        import torch
    except Exception:
        return False

    torch.storage._load_from_bytes = lambda b: torch.load(  # type: ignore[attr-defined]
        io.BytesIO(b),
        map_location="cpu",
        weights_only=False,
    )
    return True


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def summarize_nli_pipeline() -> list[dict[str, Any]]:
    _enable_cpu_unpickle_for_torch_tensors()
    rows: list[dict[str, Any]] = []
    results_dir = NLI_ROOT / "results"

    for dataset_dir in sorted(path for path in results_dir.glob("*") if path.is_dir()):
        dataset = dataset_dir.name
        pred_path = dataset_dir / "prediction.pkl"
        hhem_path = dataset_dir / "correctness.json"
        emb_path = dataset_dir / "embeddings.pkl"

        pred_count = 0
        pred_lens: list[int] = []
        if pred_path.exists():
            chunks = load_pickle_stream(pred_path)
            records = [item for chunk in chunks for item in chunk]
            pred_count = len(records)
            pred_lens = [int(item[0].get("prediction_length", 0)) for item in records]

        hhem_scores = np.array([], dtype=np.float32)
        if hhem_path.exists():
            hhem_scores = np.asarray(json.loads(hhem_path.read_text()), dtype=np.float32)

        cosine_scores = np.array([], dtype=np.float32)
        if emb_path.exists():
            embeddings = pickle.loads(emb_path.read_bytes())
            pred_embeddings = embeddings["pred_embeddings"]
            true_embeddings = embeddings["true_embeddings"]
            scores = []
            for pred_emb, true_emb in zip(pred_embeddings, true_embeddings):
                a = _tensor_to_numpy(pred_emb)
                b = _tensor_to_numpy(true_emb)
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                scores.append(float(np.dot(a, b) / denom) if denom else np.nan)
            cosine_scores = np.asarray(scores, dtype=np.float32)

        rows.append(
            {
                "dataset": dataset,
                "n_predictions": pred_count,
                "has_hhem": bool(hhem_path.exists()),
                "has_embeddings": bool(emb_path.exists()),
                "prediction_tokens_mean": float(np.mean(pred_lens)) if pred_lens else np.nan,
                "prediction_tokens_median": float(np.median(pred_lens)) if pred_lens else np.nan,
                "prediction_tokens_max": int(max(pred_lens)) if pred_lens else 0,
                "hhem_n": int(len(hhem_scores)),
                "hhem_mean": float(np.mean(hhem_scores)) if len(hhem_scores) else np.nan,
                "hhem_median": float(np.median(hhem_scores)) if len(hhem_scores) else np.nan,
                "hhem_positive_rate_at_0_5": float(np.mean(hhem_scores >= 0.5))
                if len(hhem_scores)
                else np.nan,
                "hhem_unique_count": int(len(np.unique(np.round(hhem_scores, 6))))
                if len(hhem_scores)
                else 0,
                "mpnet_cosine_n": int(len(cosine_scores)),
                "mpnet_cosine_mean": float(np.mean(cosine_scores)) if len(cosine_scores) else np.nan,
                "mpnet_cosine_median": float(np.median(cosine_scores)) if len(cosine_scores) else np.nan,
                "mpnet_cosine_min": float(np.min(cosine_scores)) if len(cosine_scores) else np.nan,
                "mpnet_cosine_max": float(np.max(cosine_scores)) if len(cosine_scores) else np.nan,
                "note": "HHEM is non-discriminative if unique_count=1.",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(ta_rows: list[dict[str, Any]], nli_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Added Experiment Summary",
        "",
        "## TA Baseline Compatibility Predictions",
        "",
        "| Dataset | Mode | n | Exact | Contains/Correct | Mean words | Max words |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ta_rows:
        lines.append(
            "| {dataset} | {mode} | {n_predictions} | {exact_rate:.3f} | "
            "{correct_rate:.3f} | {prediction_words_mean:.2f} | {prediction_words_max} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Original TA Pipeline Outputs",
            "",
            "| Dataset | n pred | HHEM n | HHEM mean | HHEM >=0.5 | HHEM unique | MPNet cos mean | MPNet cos median | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in nli_rows:
        status = []
        if row["has_hhem"]:
            status.append("HHEM")
        if row["has_embeddings"]:
            status.append("embeddings")
        if not status:
            status.append("prediction only")
        lines.append(
            "| {dataset} | {n_predictions} | {hhem_n} | {hhem_mean} | {hhem_pos} | "
            "{hhem_unique_count} | {cos_mean} | {cos_median} | {status} |".format(
                dataset=row["dataset"],
                n_predictions=row["n_predictions"],
                hhem_n=row["hhem_n"],
                hhem_mean=fmt(row["hhem_mean"], 4),
                hhem_pos=fmt(row["hhem_positive_rate_at_0_5"], 3),
                hhem_unique_count=row["hhem_unique_count"],
                cos_mean=fmt(row["mpnet_cosine_mean"], 3),
                cos_median=fmt(row["mpnet_cosine_median"], 3),
                status=", ".join(status),
            )
        )

    lines.extend(
        [
            "",
            "Key note: the original TA pipeline HHEM outputs for completed datasets are constant at 0.5021, "
            "so thresholding at 0.5 marks every sample positive and cannot support ROC/AUC analysis.",
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ta_rows = summarize_ta_compatibility()
    nli_rows = summarize_nli_pipeline()
    write_csv(OUT_DIR / "ta_compatibility_prediction_summary.csv", ta_rows)
    write_csv(OUT_DIR / "ta_original_pipeline_summary.csv", nli_rows)
    write_markdown(ta_rows, nli_rows)
    print(f"Wrote summaries to {OUT_DIR}")


if __name__ == "__main__":
    main()
