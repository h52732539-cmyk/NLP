"""
Run the original NLP evaluation pipeline on the TA baseline datasets stored in
`NLI Project v2/processed_data`, and add HHEM-based correctness signals.

Outputs are isolated from the existing benchmark artifacts:
  data/processed/ta_baseline_compatibility/
  results/ta_baseline_compatibility/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
        if not target:
            raise ValueError("Cannot set the module itself using set_submodule.")
        atoms = target.split(".")
        mod = self
        for atom in atoms[:-1]:
            mod = getattr(mod, atom)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from src.evaluation.correctness import batch_evaluate
from src.evaluation.hhem import attach_hhem_scores, load_hhem_model


SOURCE_DATASETS = {
    "sciq": {
        "source_subdir": "sciq",
        "task_type": "short_form",
    },
    "simple_questions_wiki": {
        "source_subdir": "simple_questions_wiki",
        "task_type": "short_form",
    },
    "nq": {
        "source_subdir": "nq",
        "task_type": "long_form",
    },
    "truthfulQA": {
        "source_subdir": "truthfulQA",
        "task_type": "long_form",
    },
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    lcfg = cfg["llm"]
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(lcfg["bnb_4bit_compute_dtype"], torch.float16)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=lcfg["bnb_use_double_quant"],
        bnb_4bit_quant_type=lcfg["bnb_4bit_quant_type"],
    )


def load_model_and_tokenizer(cfg: dict):
    lcfg = cfg["llm"]
    tokenizer = AutoTokenizer.from_pretrained(
        lcfg["model_name"],
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        lcfg["model_name"],
        quantization_config=build_bnb_config(cfg),
        device_map=lcfg["device_map"],
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def build_prompt(question: str, task_type: str) -> str:
    if task_type == "short_form":
        return (
            "Answer the following question concisely. "
            "Provide only a short phrase or name — do not explain.\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return (
        "Answer the following question accurately and concisely.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _strip_im_tokens(text: str) -> str:
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


def extract_answer(raw_output: str, thinking_mode: str) -> str:
    if thinking_mode == "thinking":
        match = re.search(r"</think>(.*)", raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()
    return raw_output.strip()


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, enable_thinking: bool, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)


def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_ta_dataset(source_root: str, dataset_name: str, out_dir: str) -> str:
    meta = SOURCE_DATASETS[dataset_name]
    source_path = os.path.join(source_root, meta["source_subdir"], "merged_fb.json")
    out_path = os.path.join(out_dir, f"{dataset_name}_data.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    with open(source_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            records.append(
                {
                    "id": f"{dataset_name}_{idx}",
                    "question": str(item["question"]).strip(),
                    "ground_truth": str(item["correct_answer"]).strip(),
                    "dataset": dataset_name,
                    "task_type": meta["task_type"],
                }
            )
    write_jsonl(out_path, records)
    return out_path


def gt_to_str(ground_truth) -> str:
    if isinstance(ground_truth, list):
        return ground_truth[0].strip() if ground_truth else ""
    return str(ground_truth).strip()


def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norms_a = np.linalg.norm(a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True)
    a_normed = a / np.clip(norms_a, 1e-9, None)
    b_normed = b / np.clip(norms_b, 1e-9, None)
    return np.einsum("ij,ij->i", a_normed, b_normed)


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray, n_steps: int = 200) -> dict:
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    best_j, best_thresh, best_f1 = -np.inf, float(thresholds[0]), 0.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0

        if j > best_j:
            best_j = j
            best_thresh = float(t)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            best_f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    return {
        "threshold": best_thresh,
        "youden_j": float(best_j),
        "f1_at_threshold": float(best_f1),
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def write_roc_csv(path: str, labels: np.ndarray, scores: np.ndarray) -> None:
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TA baseline datasets with HHEM support")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument(
        "--source-root",
        default="/hpc2hdd/home/yuxuanzhao/xuhaodong/NLI Project v2/processed_data",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sciq", "simple_questions_wiki"],
        choices=list(SOURCE_DATASETS.keys()),
    )
    parser.add_argument(
        "--thinking_modes",
        nargs="+",
        default=None,
        choices=["thinking", "no_thinking"],
    )
    parser.add_argument(
        "--processed-dir",
        default=os.path.join(ROOT_DIR, "data", "processed", "ta_baseline_compatibility"),
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(ROOT_DIR, "results", "ta_baseline_compatibility"),
    )
    parser.add_argument(
        "--hhem-model-path",
        default="/hpc2hdd/home/yuxuanzhao/xuhaodong/NLI Project v2/models/HHEM-2.1-Open",
    )
    parser.add_argument("--hhem-threshold", type=float, default=0.5)
    parser.add_argument("--hhem-batch-size", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-hhem", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    thinking_modes = args.thinking_modes or cfg["llm"]["thinking_modes"]

    print("Preparing TA baseline datasets ...")
    dataset_files = {
        dataset_name: prepare_ta_dataset(args.source_root, dataset_name, args.processed_dir)
        for dataset_name in args.datasets
    }

    if not args.skip_generation:
        print("Loading LLM for prediction generation ...")
        model, tokenizer = load_model_and_tokenizer(cfg)
        for dataset_name, dataset_file in dataset_files.items():
            samples = read_jsonl(dataset_file)
            for thinking_mode in thinking_modes:
                out_path = os.path.join(
                    args.processed_dir,
                    f"{dataset_name}_{thinking_mode}_predictions.jsonl",
                )
                if os.path.exists(out_path):
                    print(f"[skip] predictions already exist: {out_path}")
                    continue
                enable_thinking = thinking_mode == "thinking"
                max_new_tokens = (
                    cfg["llm"]["thinking_max_new_tokens"]
                    if enable_thinking
                    else cfg["llm"]["max_new_tokens"]
                )
                generated_records = []
                for sample in tqdm(samples, desc=f"{dataset_name}/{thinking_mode}"):
                    prompt = build_prompt(sample["question"], sample["task_type"])
                    raw = generate_one(model, tokenizer, prompt, enable_thinking, max_new_tokens)
                    raw_clean = _strip_im_tokens(raw)
                    generated_records.append(
                        {
                            **sample,
                            "prediction": extract_answer(raw_clean, thinking_mode),
                            "raw_output": raw_clean,
                            "thinking_mode": thinking_mode,
                        }
                    )
                write_jsonl(out_path, generated_records)

    if not args.skip_embeddings:
        for emb_cfg in cfg["embedding_models"]:
            emb_name = emb_cfg["name"]
            print(f"Loading embedding model: {emb_cfg['hf_id']}")
            emb_model = SentenceTransformer(emb_cfg["hf_id"])
            for dataset_name in args.datasets:
                for thinking_mode in thinking_modes:
                    pred_path = os.path.join(
                        args.processed_dir,
                        f"{dataset_name}_{thinking_mode}_predictions.jsonl",
                    )
                    if not os.path.exists(pred_path):
                        print(f"[skip] missing predictions: {pred_path}")
                        continue
                    records = read_jsonl(pred_path)
                    out_prefix = os.path.join(
                        args.processed_dir,
                        f"{dataset_name}_{thinking_mode}_{emb_name}",
                    )
                    if os.path.exists(out_prefix + "_pred.npy"):
                        print(f"[skip] embeddings already exist: {out_prefix}_pred.npy")
                        continue
                    pred_texts = [r["prediction"] for r in records]
                    gt_texts = [gt_to_str(r["ground_truth"]) for r in records]
                    if emb_cfg["prefix_passage"]:
                        pred_texts = [emb_cfg["prefix_passage"] + text for text in pred_texts]
                        gt_texts = [emb_cfg["prefix_passage"] + text for text in gt_texts]
                    pred_embs = emb_model.encode(
                        pred_texts,
                        batch_size=args.embedding_batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=emb_cfg["normalize"],
                        convert_to_numpy=True,
                    )
                    gt_embs = emb_model.encode(
                        gt_texts,
                        batch_size=args.embedding_batch_size,
                        show_progress_bar=True,
                        normalize_embeddings=emb_cfg["normalize"],
                        convert_to_numpy=True,
                    )
                    np.save(out_prefix + "_pred.npy", pred_embs)
                    np.save(out_prefix + "_gt.npy", gt_embs)

    hhem_model = None
    if not args.skip_hhem:
        print("Loading HHEM model ...")
        hhem_model = load_hhem_model(args.hhem_model_path)

    summary_rows = []

    for dataset_name in args.datasets:
        for thinking_mode in thinking_modes:
            pred_path = os.path.join(
                args.processed_dir,
                f"{dataset_name}_{thinking_mode}_predictions.jsonl",
            )
            if not os.path.exists(pred_path):
                print(f"[skip] missing predictions for analysis: {pred_path}")
                continue

            records = read_jsonl(pred_path)
            evaluated = batch_evaluate(
                records,
                long_form_f1_threshold=cfg["correctness"]["long_form_f1_threshold"],
            )
            if hhem_model is not None:
                evaluated = attach_hhem_scores(
                    evaluated,
                    model=hhem_model,
                    threshold=args.hhem_threshold,
                    batch_size=args.hhem_batch_size,
                )

            correctness_labels = np.array([int(r["is_correct"]) for r in evaluated], dtype=int)
            hhem_labels = np.array(
                [int(r.get("hhem_is_consistent", 0)) for r in evaluated],
                dtype=int,
            )

            for emb_cfg in cfg["embedding_models"]:
                emb_name = emb_cfg["name"]
                out_prefix = os.path.join(
                    args.processed_dir,
                    f"{dataset_name}_{thinking_mode}_{emb_name}",
                )
                pred_emb_path = out_prefix + "_pred.npy"
                gt_emb_path = out_prefix + "_gt.npy"
                if not (os.path.exists(pred_emb_path) and os.path.exists(gt_emb_path)):
                    print(f"[skip] missing embeddings for {dataset_name}/{thinking_mode}/{emb_name}")
                    continue

                pred_embs = np.load(pred_emb_path)
                gt_embs = np.load(gt_emb_path)
                similarities = cosine_similarity_batch(pred_embs, gt_embs)

                rows = []
                for idx, record in enumerate(evaluated):
                    rows.append(
                        {
                            "id": record["id"],
                            "question": record["question"],
                            "prediction": record["prediction"],
                            "ground_truth": str(record["ground_truth"]),
                            "dataset": record["dataset"],
                            "task_type": record["task_type"],
                            "thinking_mode": record["thinking_mode"],
                            "is_correct": int(record["is_correct"]),
                            "correctness_score": record["correctness_score"],
                            "correctness_method": record["correctness_method"],
                            "similarity": float(similarities[idx]),
                            "hhem_score": float(record.get("hhem_score", np.nan)),
                            "hhem_is_consistent": int(record.get("hhem_is_consistent", 0))
                            if "hhem_is_consistent" in record
                            else np.nan,
                            "hhem_best_reference": record.get("hhem_best_reference", ""),
                        }
                    )
                df = pd.DataFrame(rows)
                csv_path = os.path.join(
                    args.results_dir,
                    f"{dataset_name}_{thinking_mode}_{emb_name}_similarity.csv",
                )
                df.to_csv(csv_path, index=False)

                similarity_auc_correctness = safe_auc(correctness_labels, similarities)
                write_roc_csv(
                    os.path.join(
                        args.results_dir,
                        f"{dataset_name}_{thinking_mode}_{emb_name}_roc_correctness.csv",
                    ),
                    correctness_labels,
                    similarities,
                )

                row = {
                    "dataset": dataset_name,
                    "thinking_mode": thinking_mode,
                    "emb_model": emb_name,
                    "n_samples": len(df),
                    "correctness_positive_rate": float(np.mean(correctness_labels)),
                    "similarity_auc_vs_correctness": similarity_auc_correctness,
                    **{
                        f"correctness_{k}": v
                        for k, v in find_optimal_threshold(
                            similarities,
                            correctness_labels,
                            cfg["similarity"]["threshold_n_steps"],
                        ).items()
                    },
                }

                if hhem_model is not None:
                    row["hhem_positive_rate"] = float(np.mean(hhem_labels))
                    row["correctness_hhem_agreement"] = float(
                        np.mean(correctness_labels == hhem_labels)
                    )
                    row["similarity_auc_vs_hhem"] = safe_auc(hhem_labels, similarities)
                    write_roc_csv(
                        os.path.join(
                            args.results_dir,
                            f"{dataset_name}_{thinking_mode}_{emb_name}_roc_hhem.csv",
                        ),
                        hhem_labels,
                        similarities,
                    )
                summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.results_dir, "summary_auc.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
