from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    model_name_or_path: str
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = None
    max_input_length: int = 4096
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = False
    trust_remote_code: bool = True


@dataclass(slots=True)
class RepresentationConfig:
    layers: list[int]
    poolings: list[str]
    use_prompt_residual: list[bool]


@dataclass(slots=True)
class MetricsConfig:
    lexical: list[str]
    semantic: list[str]
    latent: list[str]
    compute_dataset_level_cka: bool = True


@dataclass(slots=True)
class DatasetRecipe:
    name: str
    source: str
    path: str
    config_name: str | None = None
    split: str | None = None
    prompt_field: str | None = None
    candidate_field: str = "candidate"
    reference_field: str = "reference"
    human_score_field: str | None = None
    binary_label_field: str | None = None
    id_field: str | None = None
    task_type: str = "generation"
    limit: int | None = None
    generate_candidate: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    output_dir: str
    cache_dir: str
    model: ModelConfig
    representation: RepresentationConfig
    metrics: MetricsConfig
    datasets: list[DatasetRecipe]
    config_path: Path
    project_root: Path

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    model_payload = payload.get("model", {})
    representation_payload = payload.get("representation", {})
    metrics_payload = payload.get("metrics", {})

    datasets = [DatasetRecipe(**dataset_payload) for dataset_payload in payload.get("datasets", [])]

    return ExperimentConfig(
        experiment_name=str(payload["experiment_name"]),
        seed=int(payload.get("seed", 42)),
        output_dir=str(payload.get("output_dir", "results/default_run")),
        cache_dir=str(payload.get("cache_dir", "data/cache")),
        model=ModelConfig(**model_payload),
        representation=RepresentationConfig(
            layers=[int(value) for value in representation_payload.get("layers", [-1])],
            poolings=[str(value) for value in representation_payload.get("poolings", ["mean"])],
            use_prompt_residual=[bool(value) for value in representation_payload.get("use_prompt_residual", [False])],
        ),
        metrics=MetricsConfig(
            lexical=[str(value) for value in metrics_payload.get("lexical", [])],
            semantic=[str(value) for value in metrics_payload.get("semantic", [])],
            latent=[str(value) for value in metrics_payload.get("latent", ["cosine"])],
            compute_dataset_level_cka=bool(metrics_payload.get("compute_dataset_level_cka", True)),
        ),
        datasets=datasets,
        config_path=config_path,
        project_root=project_root,
    )


def dump_resolved_config(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "output_dir": str(config.resolve_path(config.output_dir)),
        "cache_dir": str(config.resolve_path(config.cache_dir)),
        "model": asdict(config.model),
        "representation": asdict(config.representation),
        "metrics": asdict(config.metrics),
        "datasets": [asdict(dataset) for dataset in config.datasets],
        "config_path": str(config.config_path),
        "project_root": str(config.project_root),
    }