from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def exact_match(candidate: str, reference: str) -> float:
    return float(normalize_text(candidate) == normalize_text(reference))


def token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = normalize_text(candidate).split()
    reference_tokens = normalize_text(reference).split()
    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0

    candidate_counter = Counter(candidate_tokens)
    reference_counter = Counter(reference_tokens)
    overlap = sum((candidate_counter & reference_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(candidate: str, reference: str) -> float:
    candidate_tokens = normalize_text(candidate).split()
    reference_tokens = normalize_text(reference).split()
    if not candidate_tokens or not reference_tokens:
        return 0.0

    lcs = longest_common_subsequence(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    numerator = float(np.dot(vector_a, vector_b))
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def linear_cka(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    centered_x = matrix_x - matrix_x.mean(axis=0, keepdims=True)
    centered_y = matrix_y - matrix_y.mean(axis=0, keepdims=True)

    cross_covariance = centered_x.T @ centered_y
    self_covariance_x = centered_x.T @ centered_x
    self_covariance_y = centered_y.T @ centered_y

    numerator = float(np.linalg.norm(cross_covariance, ord="fro") ** 2)
    denominator = float(
        np.linalg.norm(self_covariance_x, ord="fro")
        * np.linalg.norm(self_covariance_y, ord="fro")
    )
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_bertscore(candidates: Sequence[str], references: Sequence[str]) -> list[float]:
    from bert_score import score

    _, _, f1 = score(cands=list(candidates), refs=list(references), lang="en", verbose=False)
    return [float(value) for value in f1]


def compute_simcse_cosine(
    candidates: Sequence[str],
    references: Sequence[str],
    model_name: str = "princeton-nlp/sup-simcse-roberta-base",
) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    candidate_embeddings = model.encode(list(candidates), normalize_embeddings=True)
    reference_embeddings = model.encode(list(references), normalize_embeddings=True)
    scores = np.sum(candidate_embeddings * reference_embeddings, axis=1)
    return [float(value) for value in scores]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def longest_common_subsequence(source: Sequence[str], target: Sequence[str]) -> int:
    rows = len(source) + 1
    cols = len(target) + 1
    table = [[0] * cols for _ in range(rows)]
    for row in range(1, rows):
        for col in range(1, cols):
            if source[row - 1] == target[col - 1]:
                table[row][col] = table[row - 1][col - 1] + 1
            else:
                table[row][col] = max(table[row - 1][col], table[row][col - 1])
    return table[-1][-1]