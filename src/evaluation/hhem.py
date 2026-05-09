"""
Utilities for scoring prediction correctness with HHEM.

HHEM is an asymmetric consistency model:
  premise    -> reference / supporting answer
  hypothesis -> model prediction
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedModel


if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    @property
    def all_tied_weights_keys(self):
        stored = getattr(self, "_all_tied_weights_keys", None)
        if stored is not None:
            return stored
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        return {key: None for key in keys}

    @all_tied_weights_keys.setter
    def all_tied_weights_keys(self, value):
        object.__setattr__(self, "_all_tied_weights_keys", value)

    PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys


@dataclass
class HHEMRecordScore:
    score: float
    label: bool
    best_reference: str


def normalize_ground_truth_list(ground_truth) -> list[str]:
    """
    Convert str / list / CSV-stringified list ground truth into a clean list.
    """
    if isinstance(ground_truth, list):
        values = ground_truth
    elif isinstance(ground_truth, str):
        stripped = ground_truth.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                values = parsed if isinstance(parsed, list) else [ground_truth]
            except (SyntaxError, ValueError):
                values = [ground_truth]
        else:
            values = [ground_truth]
    elif ground_truth is None:
        values = []
    else:
        values = [str(ground_truth)]

    return [str(v).strip() for v in values if str(v).strip()]


def load_hhem_model(model_path: str, device: str | None = None):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model


def _batched_predict(model, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        batch_scores = model.predict(batch_pairs)
        if isinstance(batch_scores, torch.Tensor):
            scores.extend(float(x) for x in batch_scores.detach().cpu().tolist())
        else:
            scores.extend(float(x) for x in batch_scores)
    return scores


def score_predictions_with_hhem(
    model,
    predictions: Sequence[str],
    ground_truths: Sequence,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> list[HHEMRecordScore]:
    """
    Score each prediction against one or more references and keep the max score.
    """
    flattened_pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int, list[str]]] = []

    for prediction, ground_truth in zip(predictions, ground_truths):
        references = normalize_ground_truth_list(ground_truth)
        if not references:
            spans.append((-1, -1, []))
            continue
        start = len(flattened_pairs)
        flattened_pairs.extend((reference, str(prediction)) for reference in references)
        end = len(flattened_pairs)
        spans.append((start, end, references))

    flattened_scores = _batched_predict(model, flattened_pairs, batch_size) if flattened_pairs else []

    results: list[HHEMRecordScore] = []
    for start, end, references in spans:
        if start < 0:
            results.append(HHEMRecordScore(score=0.0, label=False, best_reference=""))
            continue
        candidate_scores = flattened_scores[start:end]
        best_idx = max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
        best_score = float(candidate_scores[best_idx])
        results.append(
            HHEMRecordScore(
                score=best_score,
                label=best_score >= threshold,
                best_reference=references[best_idx],
            )
        )
    return results


def attach_hhem_scores(
    records: Iterable[dict],
    model,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> list[dict]:
    records = list(records)
    scores = score_predictions_with_hhem(
        model=model,
        predictions=[record.get("prediction", "") for record in records],
        ground_truths=[record.get("ground_truth", "") for record in records],
        threshold=threshold,
        batch_size=batch_size,
    )

    merged = []
    for record, score in zip(records, scores):
        merged.append(
            {
                **record,
                "hhem_score": score.score,
                "hhem_is_consistent": int(score.label),
                "hhem_best_reference": score.best_reference,
                "hhem_threshold": threshold,
            }
        )
    return merged
