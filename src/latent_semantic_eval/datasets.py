from __future__ import annotations

from pathlib import Path
from typing import Any

from latent_semantic_eval.config import DatasetRecipe, ExperimentConfig
from latent_semantic_eval.io_utils import ensure_directory, read_jsonl, write_jsonl
from latent_semantic_eval.schemas import EvaluationRecord


def prepare_datasets(config: ExperimentConfig) -> list[Path]:
    processed_dir = config.resolve_path("data/processed")
    ensure_directory(processed_dir)
    output_paths: list[Path] = []
    for recipe in config.datasets:
        records = load_records(config, recipe, prefer_processed=False)
        output_path = processed_dir / f"{recipe.name}.jsonl"
        write_jsonl(output_path, [record.to_dict() for record in records])
        output_paths.append(output_path)
    return output_paths


def load_records(
    config: ExperimentConfig,
    recipe: DatasetRecipe,
    prefer_processed: bool = True,
) -> list[EvaluationRecord]:
    processed_path = config.resolve_path(f"data/processed/{recipe.name}.jsonl")
    if prefer_processed and processed_path.exists():
        return [EvaluationRecord.from_dict(row) for row in read_jsonl(processed_path)]

    if recipe.source == "jsonl":
        raw_path = config.resolve_path(recipe.path)
        rows = read_jsonl(raw_path)
    elif recipe.source == "hf":
        rows = _load_from_hugging_face(config, recipe)
    else:
        raise ValueError(f"Unsupported dataset source: {recipe.source}")

    records = [_to_record(recipe, row, index) for index, row in enumerate(rows)]
    if recipe.limit is not None:
        records = records[: recipe.limit]
    return records


def _load_from_hugging_face(config: ExperimentConfig, recipe: DatasetRecipe) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        path=recipe.path,
        name=recipe.config_name,
        split=recipe.split,
        cache_dir=str(config.resolve_path(config.cache_dir)),
    )
    return [dict(row) for row in dataset]


def _to_record(recipe: DatasetRecipe, row: dict[str, Any], index: int) -> EvaluationRecord:
    record_id = _field_as_str(row, recipe.id_field) if recipe.id_field else f"{recipe.name}-{index:06d}"
    return EvaluationRecord(
        record_id=record_id,
        prompt=_field_as_str(row, recipe.prompt_field),
        candidate=_field_as_str(row, recipe.candidate_field),
        reference=_field_as_str(row, recipe.reference_field),
        human_score=_field_as_float(row, recipe.human_score_field),
        binary_label=_field_as_int(row, recipe.binary_label_field),
        metadata={
            "dataset_name": recipe.name,
            "task_type": recipe.task_type,
            "source": recipe.source,
            "split": recipe.split,
        },
    )


def _field_as_str(row: dict[str, Any], field_name: str | None) -> str:
    if not field_name:
        return ""
    value = row.get(field_name, "")
    if value is None:
        return ""
    return str(value)


def _field_as_float(row: dict[str, Any], field_name: str | None) -> float | None:
    if not field_name:
        return None
    value = row.get(field_name)
    if value is None or value == "":
        return None
    return float(value)


def _field_as_int(row: dict[str, Any], field_name: str | None) -> int | None:
    if not field_name:
        return None
    value = row.get(field_name)
    if value is None or value == "":
        return None
    return int(value)