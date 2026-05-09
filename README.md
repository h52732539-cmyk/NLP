# Semantic Similarity Measurement in Latent Space for LLM Prediction Evaluation

**Course:** AIAA 4051 — Final Research Project  
**Model:** Qwen3-4B (4-bit NF4, CUDA 12.4 / NVIDIA driver 12040)

---

## Research Overview

This project investigates whether cosine similarity in embedding space can serve
as a reliable proxy for factual correctness of LLM-generated answers.
Two QA task types are compared:

| Type | Datasets |
|---|---|
| Short-form | SciQ · SimpleQA |
| Long-form | Natural Questions (nq_open) · TruthfulQA |

**Ablation (Option B):** every dataset is evaluated under both
`no_thinking` (concise answers) and `thinking` (chain-of-thought) generation modes.

**Embedding models compared:** BGE-base-en-v1.5 · all-MiniLM-L6-v2 · e5-base-v2

---

## Directory Structure

```
NLP/
├── configs/
│   └── benchmark.yaml          # all hyperparameters & paths
├── data/
│   ├── raw/                    # manually downloaded / cached files
│   └── processed/              # auto-generated JSONL + .npy files
├── scripts/
│   ├── prepare_data.py         # Step 1: download & unify datasets
│   ├── generate_predictions.py # Step 2: Qwen3-4B inference (both modes)
│   ├── encode_embeddings.py    # Step 3: embed predictions & ground truths
│   ├── similarity_analysis.py  # Step 4: cosine sim → AUC / threshold
│   └── failure_analysis.py     # Step 5: FP/FN case extraction
├── src/
│   └── evaluation/
│       └── correctness.py      # EM / token-F1 correctness functions
├── notebooks/
│   └── empirical_study.ipynb   # visualisations & summary tables
├── results/                    # CSVs + figures (auto-created)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Install PyTorch with CUDA 12.4

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Verify GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Step-by-step Usage

Run all commands from the project root directory (`NLP/`).

### Step 1 — Download & pre-process datasets

```bash
python scripts/prepare_data.py
```

Downloads SciQ, SimpleQA, NQ (nq_open), TruthfulQA from HuggingFace.  
Outputs `data/processed/{dataset}_data.jsonl` files.

To process specific datasets only:
```bash
python scripts/prepare_data.py --datasets sciq simpleqa
```

---

### Step 2 — Generate predictions (≈ 10–15 GPU hours)

```bash
python scripts/generate_predictions.py
```

Runs Qwen3-4B with 4-bit quantisation in both `no_thinking` and `thinking` modes.  
Outputs `data/processed/{dataset}_{mode}_predictions.jsonl`.

To run one mode only:
```bash
python scripts/generate_predictions.py --thinking_modes no_thinking
```

---

### Step 3 — Encode embeddings

```bash
python scripts/encode_embeddings.py
```

Encodes all prediction / ground-truth pairs with three embedding models.  
Outputs `data/processed/{dataset}_{mode}_{emb}_pred.npy` and `_gt.npy`.

Increase batch size on large-memory GPUs:
```bash
python scripts/encode_embeddings.py --batch_size 128
```

---

### Step 4 — Similarity analysis

```bash
python scripts/similarity_analysis.py
```

Computes cosine similarities, runs ROC / AUC analysis, and finds the optimal
correctness threshold via Youden's J statistic.  
Outputs `results/summary_auc.csv` and per-sample CSVs.

---

### Step 5 — Failure case analysis

```bash
python scripts/failure_analysis.py
```

Extracts False Positive / False Negative examples and categorises failure causes.  
Outputs `results/failure_cases_{dataset}_{mode}.csv` and `results/failure_summary.csv`.

---

### Step 6 — Visualisations

Open `notebooks/empirical_study.ipynb` in Jupyter and run all cells.  
Figures are saved to `results/figures/` as PDF.

---

## Configuration

All settings are controlled by `configs/benchmark.yaml`.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `llm.model_name` | `Qwen/Qwen3-4B` | HuggingFace model ID |
| `llm.bnb_4bit_quant_type` | `nf4` | BitsAndBytes quant type |
| `llm.max_new_tokens` | `256` | Max tokens for no_thinking |
| `llm.thinking_max_new_tokens` | `2048` | Max tokens for thinking mode |
| `llm.thinking_modes` | `[no_thinking, thinking]` | Ablation modes |
| `correctness.long_form_f1_threshold` | `0.3` | Token-F1 threshold for long-form |
| `similarity.threshold_n_steps` | `200` | Grid steps for Youden-J search |

---

## Correctness Evaluation

| Task type | Method |
|---|---|
| Short-form (SciQ, SimpleQA) | Normalised Exact Match + substring containment |
| Long-form (NQ) | Token-level F1 ≥ 0.3 |
| Long-form (TruthfulQA) | Match against any correct answer in `correct_answers` list |

---

## Expected GPU Budget

| Step | Estimated time |
|---|---|
| Qwen3-4B inference (×2 modes, ~2,300 samples) | 10–15 h |
| Embedding encoding (3 models) | 2–5 h |
| Total | **< 20 h** (well within 50 h budget) |

---

## Part 4 — Improvement Direction

Failure analysis (Step 5) provides motivation for a hyperbolic-geometry
embedding approach. The proposed improvement — mapping Euclidean embeddings to
a **Poincaré Ball** and computing hyperbolic distance — is aligned with:

- He et al. *HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts* (arXiv:2505.24722, 2025)  
- Patil et al. *Hierarchical Mamba Meets Hyperbolic Geometry* (arXiv:2505.18973, 2025)

Implementation decision pending current experiment results.

---

## TA Baseline Compatibility

To evaluate the TA baseline datasets already prepared under:

```bash
/hpc2hdd/home/yuxuanzhao/xuhaodong/NLI Project v2/processed_data
```

run:

```bash
python scripts/evaluate_ta_baseline_compatibility.py
```

This script:

- reads all four datasets from `merged_fb.json`
- runs the original prediction → embedding → similarity workflow
- adds HHEM-2.1-Open scores and labels for each sample
- writes outputs to isolated directories so old benchmark artifacts are not overwritten

Default output locations:

```bash
data/processed/ta_baseline_compatibility/
results/ta_baseline_compatibility/
```

Datasets supported:

- `sciq`
- `simple_questions_wiki`
- `nq`
- `truthfulQA`

Default run set:

- `sciq`
- `simple_questions_wiki`

Useful options:

```bash
python scripts/evaluate_ta_baseline_compatibility.py --skip-generation
python scripts/evaluate_ta_baseline_compatibility.py --thinking_modes no_thinking
python scripts/evaluate_ta_baseline_compatibility.py --hhem-threshold 0.5
python scripts/evaluate_ta_baseline_compatibility.py --datasets sciq simple_questions_wiki
```

Per-sample result CSVs include both the original correctness fields and:

- `hhem_score`
- `hhem_is_consistent`
- `hhem_best_reference`

The summary file `results/ta_baseline_compatibility/summary_auc.csv` also reports:

- similarity AUC vs. original correctness labels
- similarity AUC vs. HHEM labels
- agreement between original correctness labels and HHEM labels

---

## HHEM Correctness Label Comparison

The local HHEM model is expected at:

```bash
/hpc2hdd/home/yuxuanzhao/xuhaodong/NLI Project v2/models/HHEM-2.1-Open
```

To backfill HHEM scores for existing experiment CSVs in `results/`, run:

```bash
python scripts/recompute_hhem_for_results.py
```

By default this script:

- scans existing `*_similarity.csv` files
- recomputes HHEM scores from each row's `ground_truth` and `prediction`
- writes augmented CSVs to `results/hhem_backfill/`
- keeps the original experiment files unchanged

Useful options:

```bash
python scripts/recompute_hhem_for_results.py --pattern '_similarity.csv'
python scripts/recompute_hhem_for_results.py --hhem-threshold 0.5
python scripts/recompute_hhem_for_results.py --in-place
```

The backfill summary is saved to:

```bash
results/hhem_backfill/hhem_backfill_summary.csv
```
