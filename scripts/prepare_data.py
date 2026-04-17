from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent_semantic_eval.config import load_experiment_config
from latent_semantic_eval.datasets import prepare_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for latent semantic evaluation")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare_datasets(load_experiment_config(arguments.config))