from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local SICK-R and SummEval JSONL files from MTEB mirrors")
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "data" / "cache"),
        help="Datasets cache directory",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Output directory for local JSONL files",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sickr_rows(cache_dir: str) -> list[dict]:
    dataset = load_dataset("mteb/sickr-sts", split="test", cache_dir=cache_dir)
    rows: list[dict] = []
    for index, row in enumerate(dataset):
        rows.append(
            {
                "id": f"sickr-{index:05d}",
                "sentence1": str(row["sentence1"]),
                "sentence2": str(row["sentence2"]),
                "relatedness_score": float(row["score"]),
            }
        )
    return rows


def build_summeval_rows(cache_dir: str) -> list[dict]:
    dataset = load_dataset("mteb/summeval", split="test", cache_dir=cache_dir)
    rows: list[dict] = []

    for row in dataset:
        references = [summary.strip() for summary in row["human_summaries"] if str(summary).strip()]
        if not references:
            raise ValueError(f"SummEval row {row['id']} has no human summaries")

        reference_text = references[0]
        prompt_text = f"Summarize the following article:\n\n{str(row['text']).strip()}"
        machine_summaries = [str(summary).strip() for summary in row["machine_summaries"]]
        consistency_scores = row["consistency"]

        if not isinstance(consistency_scores, list):
            consistency_scores = [consistency_scores] * len(machine_summaries)

        if len(machine_summaries) != len(consistency_scores):
            raise ValueError(
                f"SummEval row {row['id']} has {len(machine_summaries)} machine summaries but {len(consistency_scores)} consistency scores"
            )

        for summary_index, (candidate_text, consistency_score) in enumerate(zip(machine_summaries, consistency_scores)):
            if not candidate_text:
                continue

            rows.append(
                {
                    "id": f"summeval-{row['id']}-{summary_index:02d}",
                    "prompt": prompt_text,
                    "candidate": candidate_text,
                    "reference": reference_text,
                    "consistency_score": float(consistency_score),
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()

    sickr_rows = build_sickr_rows(args.cache_dir)
    write_jsonl(raw_dir / "sickr_test.jsonl", sickr_rows)

    summeval_rows = build_summeval_rows(args.cache_dir)
    write_jsonl(raw_dir / "summeval.jsonl", summeval_rows)

    print(f"Wrote {len(sickr_rows)} rows to {raw_dir / 'sickr_test.jsonl'}")
    print(f"Wrote {len(summeval_rows)} rows to {raw_dir / 'summeval.jsonl'}")


if __name__ == "__main__":
    main()