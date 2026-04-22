from __future__ import annotations

import math
from itertools import combinations
from random import Random
from typing import Callable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, roc_auc_score


def safe_spearman(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    if len(labels) < 2:
        return None
    correlation, _ = spearmanr(labels, scores)
    if math.isnan(correlation):
        return None
    return float(correlation)


def safe_pearson(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    if len(labels) < 2:
        return None
    correlation, _ = pearsonr(labels, scores)
    if math.isnan(correlation):
        return None
    return float(correlation)


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def best_f1(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    best_value = 0.0
    for threshold in sorted(set(scores)):
        predictions = [1 if score >= threshold else 0 for score in scores]
        best_value = max(best_value, float(f1_score(labels, predictions)))
    return best_value


def pairwise_ranking_accuracy(
    labels: Sequence[float],
    scores: Sequence[float],
    sample_limit: int = 50000,
    seed: int = 42,
) -> float | None:
    indexed_pairs = [
        (left, right)
        for left, right in combinations(range(len(labels)), 2)
        if labels[left] != labels[right]
    ]
    if not indexed_pairs:
        return None

    if len(indexed_pairs) > sample_limit:
        rng = Random(seed)
        indexed_pairs = rng.sample(indexed_pairs, sample_limit)

    correct = 0
    for left, right in indexed_pairs:
        label_order = labels[left] > labels[right]
        score_order = scores[left] > scores[right]
        if label_order == score_order:
            correct += 1
    return correct / len(indexed_pairs)


def bootstrap_interval(
    labels: Sequence[float],
    scores: Sequence[float],
    metric_fn: Callable[[Sequence[float], Sequence[float]], float | None],
    iterations: int = 500,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    if len(labels) < 2:
        return None, None

    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    estimates: list[float] = []

    for _ in range(iterations):
        indices = rng.integers(0, len(labels_array), size=len(labels_array))
        estimate = metric_fn(labels_array[indices], scores_array[indices])
        if estimate is not None and not math.isnan(estimate):
            estimates.append(float(estimate))

    if not estimates:
        return None, None

    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )