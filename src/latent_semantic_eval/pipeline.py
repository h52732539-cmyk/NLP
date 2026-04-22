from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from latent_semantic_eval.analysis import (
    best_f1,
    binary_auc,
    bootstrap_interval,
    pairwise_ranking_accuracy,
    safe_pearson,
    safe_spearman,
)
from latent_semantic_eval.config import ExperimentConfig, dump_resolved_config, load_experiment_config
from latent_semantic_eval.datasets import load_records
from latent_semantic_eval.generation import QwenGenerator
from latent_semantic_eval.io_utils import ensure_directory, write_json, write_jsonl
from latent_semantic_eval.metrics import (
    compute_bertscore,
    compute_simcse_cosine,
    cosine_similarity,
    exact_match,
    linear_cka,
    rouge_l,
    token_f1,
)
from latent_semantic_eval.representations import QwenRepresentationExtractor


def run_experiment(config_path: str | Path) -> Path:
    config = load_experiment_config(config_path)
    _set_seed(config.seed)

    output_dir = config.resolve_path(config.output_dir)
    ensure_directory(output_dir)
    write_json(output_dir / "resolved_config.json", dump_resolved_config(config))

    extractor: QwenRepresentationExtractor | None = None
    generator: QwenGenerator | None = None
    summary_frames: list[pd.DataFrame] = []

    for recipe in config.datasets:
        dataset_dir = output_dir / recipe.name
        ensure_directory(dataset_dir)

        record_scores_path = dataset_dir / "record_scores.csv"
        metric_summary_path = dataset_dir / "metric_summary.csv"

        if record_scores_path.exists() and metric_summary_path.exists():
            tqdm.write(f"[cache] Loading existing results for {recipe.name}, skipping evaluation.")
            summary_frame = pd.read_csv(metric_summary_path)
            summary_frames.append(summary_frame)
            continue

        records = load_records(config, recipe, prefer_processed=True)
        if recipe.generate_candidate:
            if generator is None:
                generator = QwenGenerator(config.model)
            _populate_missing_candidates(records, generator)

        if extractor is None:
            extractor = QwenRepresentationExtractor(config.model)

        row_frame, summary_frame = _evaluate_dataset(config, recipe.name, records, extractor)

        row_frame.to_csv(record_scores_path, index=False)
        write_jsonl(dataset_dir / "record_scores.jsonl", row_frame.to_dict(orient="records"))
        summary_frame.to_csv(metric_summary_path, index=False)
        write_json(dataset_dir / "metric_summary.json", summary_frame.to_dict(orient="records"))
        summary_frames.append(summary_frame)

    combined_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    combined_summary.to_csv(output_dir / "combined_metric_summary.csv", index=False)
    write_json(output_dir / "combined_metric_summary.json", combined_summary.to_dict(orient="records"))
    return output_dir


def _evaluate_dataset(
    config: ExperimentConfig,
    dataset_name: str,
    records,
    extractor: QwenRepresentationExtractor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    latent_vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"candidate": [], "reference": []})
    resolved_layers = extractor.resolve_layers(config.representation.layers)

    for record in tqdm(records, desc=f"Evaluating {dataset_name}"):
        row = {
            "dataset_name": dataset_name,
            "record_id": record.record_id,
            "prompt": record.prompt,
            "candidate": record.candidate,
            "reference": record.reference,
            "human_score": record.human_score,
            "binary_label": record.binary_label,
        }

        if "exact_match" in config.metrics.lexical:
            row["exact_match"] = exact_match(record.candidate, record.reference)
        if "token_f1" in config.metrics.lexical:
            row["token_f1"] = token_f1(record.candidate, record.reference)
        if "rouge_l" in config.metrics.lexical:
            row["rouge_l"] = rouge_l(record.candidate, record.reference)

        candidate_bundle = extractor.extract_bundle(record.prompt, record.candidate)
        reference_bundle = extractor.extract_bundle(record.prompt, record.reference)

        for pooling in config.representation.poolings:
            for use_prompt_residual in config.representation.use_prompt_residual:
                for layer in resolved_layers:
                    metric_key = (
                        f"latent__pool={pooling}"
                        f"__residual={int(use_prompt_residual)}"
                        f"__layer={layer}"
                        f"__cosine"
                    )
                    candidate_vector = extractor.pool_vector(candidate_bundle, layer, pooling, use_prompt_residual)
                    reference_vector = extractor.pool_vector(reference_bundle, layer, pooling, use_prompt_residual)
                    row[metric_key] = cosine_similarity(candidate_vector, reference_vector)
                    latent_vectors[metric_key]["candidate"].append(candidate_vector)
                    latent_vectors[metric_key]["reference"].append(reference_vector)

        rows.append(row)

    row_frame = pd.DataFrame(rows)

    if "bertscore_f1" in config.metrics.semantic:
        row_frame["bertscore_f1"] = compute_bertscore(row_frame["candidate"].tolist(), row_frame["reference"].tolist())
    if "simcse_cosine" in config.metrics.semantic:
        row_frame["simcse_cosine"] = compute_simcse_cosine(
            row_frame["candidate"].tolist(),
            row_frame["reference"].tolist(),
        )

    summary_frame = _summarize_metrics(config, dataset_name, row_frame, latent_vectors)
    return row_frame, summary_frame


def _summarize_metrics(
    config: ExperimentConfig,
    dataset_name: str,
    row_frame: pd.DataFrame,
    latent_vectors: dict[str, dict[str, list[np.ndarray]]],
) -> pd.DataFrame:
    summary_rows: list[dict] = []
    excluded_columns = {
        "dataset_name",
        "record_id",
        "prompt",
        "candidate",
        "reference",
        "human_score",
        "binary_label",
    }

    metric_columns = [column for column in row_frame.columns if column not in excluded_columns]
    for metric_name in metric_columns:
        summary_row = {"dataset_name": dataset_name, "metric_name": metric_name}

        scored_frame = row_frame[[metric_name, "human_score"]].dropna()
        if not scored_frame.empty:
            labels = scored_frame["human_score"].tolist()
            scores = scored_frame[metric_name].tolist()
            summary_row["spearman"] = safe_spearman(labels, scores)
            summary_row["pearson"] = safe_pearson(labels, scores)
            summary_row["pairwise_accuracy"] = pairwise_ranking_accuracy(labels, scores, seed=config.seed)
            ci_low, ci_high = bootstrap_interval(labels, scores, safe_spearman, seed=config.seed)
            summary_row["spearman_ci_low"] = ci_low
            summary_row["spearman_ci_high"] = ci_high

        binary_frame = row_frame[[metric_name, "binary_label"]].dropna()
        if not binary_frame.empty:
            labels = [int(value) for value in binary_frame["binary_label"].tolist()]
            scores = binary_frame[metric_name].tolist()
            summary_row["auc"] = binary_auc(labels, scores)
            summary_row["best_f1"] = best_f1(labels, scores)

        if config.metrics.compute_dataset_level_cka and metric_name in latent_vectors:
            candidate_matrix = np.vstack(latent_vectors[metric_name]["candidate"])
            reference_matrix = np.vstack(latent_vectors[metric_name]["reference"])
            summary_row["dataset_level_cka"] = linear_cka(candidate_matrix, reference_matrix)

        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def _populate_missing_candidates(records, generator: QwenGenerator) -> None:
    missing_indices = [index for index, record in enumerate(records) if not record.candidate.strip()]
    if not missing_indices:
        return

    prompts = [records[index].prompt for index in missing_indices]
    generations = generator.generate(prompts)
    for index, generation in zip(missing_indices, generations):
        records[index].candidate = generation.text
        if generation.reasoning:
            records[index].metadata["reasoning"] = generation.reasoning


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass