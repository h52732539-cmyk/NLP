"""
correctness.py
--------------
Correctness evaluation functions for QA predictions.

Supports:
  - Short-form QA: Exact Match (normalised) + substring containment fallback
  - Long-form QA:  Token-level F1 with configurable threshold for binary label
"""

import re
import string
from collections import Counter
from typing import Union


# ── Text normalisation ──────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, remove articles/punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


# ── Short-form metrics ──────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_match(prediction: str, ground_truth: str) -> bool:
    """True if the normalised GT string is a substring of the normalised prediction."""
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    return bool(norm_gt) and norm_gt in norm_pred


# ── Long-form metric ────────────────────────────────────────────────────────

def get_tokens(s: str) -> list:
    return normalize_answer(s).split()


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and a single ground-truth string."""
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Unified correctness function ────────────────────────────────────────────

def compute_correctness(
    prediction: str,
    ground_truth: Union[str, list],
    task_type: str,
    dataset: str,
    long_form_f1_threshold: float = 0.3,
) -> dict:
    """
    Evaluate whether a prediction is correct.

    Parameters
    ----------
    prediction            : model-generated answer string
    ground_truth          : reference answer (str) or list of acceptable answers
    task_type             : "short_form" | "long_form"
    dataset               : dataset name (for logging)
    long_form_f1_threshold: minimum token-F1 to be counted as correct

    Returns
    -------
    dict with keys:
        is_correct (bool)  – binary correctness label
        score      (float) – raw metric value used for the decision
        method     (str)   – description of the method used
    """
    # Normalise ground_truth to a non-empty list
    if isinstance(ground_truth, list):
        gt_list = [g for g in ground_truth if g and str(g).strip()]
    else:
        gt_list = [ground_truth] if ground_truth and str(ground_truth).strip() else []

    if not gt_list:
        return {"is_correct": False, "score": 0.0, "method": "no_gt"}

    if task_type == "short_form":
        em_scores = [float(exact_match(prediction, gt)) for gt in gt_list]
        cont_scores = [float(contains_match(prediction, gt)) for gt in gt_list]
        score = max(max(em_scores), max(cont_scores))
        is_correct = score > 0.0
        method = "exact_match_or_contains"
    else:
        f1_scores = [token_f1(prediction, gt) for gt in gt_list]
        score = max(f1_scores)
        is_correct = score >= long_form_f1_threshold
        method = f"token_f1_threshold={long_form_f1_threshold}"

    return {"is_correct": is_correct, "score": float(score), "method": method}


# ── Batch helper ────────────────────────────────────────────────────────────

def batch_evaluate(records: list, long_form_f1_threshold: float = 0.3) -> list:
    """
    Evaluate correctness for a list of prediction records.

    Each record is expected to have:
        prediction, ground_truth, task_type, dataset

    Returns the same list with three extra keys added per record:
        is_correct, correctness_score, correctness_method
    """
    results = []
    for r in records:
        result = compute_correctness(
            prediction=r.get("prediction", ""),
            ground_truth=r.get("ground_truth", ""),
            task_type=r.get("task_type", "short_form"),
            dataset=r.get("dataset", ""),
            long_form_f1_threshold=long_form_f1_threshold,
        )
        results.append(
            {
                **r,
                "is_correct": result["is_correct"],
                "correctness_score": result["score"],
                "correctness_method": result["method"],
            }
        )
    return results
