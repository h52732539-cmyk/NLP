from __future__ import annotations

import argparse

from latent_semantic_eval.datasets import prepare_datasets
from latent_semantic_eval.pipeline import run_experiment
from latent_semantic_eval.config import load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latent semantic evaluation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Standardize configured datasets")
    prepare_parser.add_argument("--config", required=True, help="Path to the experiment YAML config")

    run_parser = subparsers.add_parser("run", help="Run the full evaluation pipeline")
    run_parser.add_argument("--config", required=True, help="Path to the experiment YAML config")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        config = load_experiment_config(args.config)
        prepare_datasets(config)
        return

    if args.command == "run":
        run_experiment(args.config)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()