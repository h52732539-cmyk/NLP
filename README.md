# Semantic Similarity Measurement in Latent Space for LLM Prediction Evaluation

This repository contains a reproducible research scaffold for studying whether latent-space similarity derived from Qwen3-4B can evaluate model predictions more faithfully than lexical overlap metrics. The implementation is designed for a course research project and focuses on English benchmarks first.

The codebase supports three layers of evaluation:

1. Semantic calibration on sentence-pair benchmarks such as STS-B and SICK-R.
2. Robustness checks on high lexical overlap data such as PAWS.
3. Prediction evaluation on generation datasets such as SummEval, with optional self-generated candidates from Qwen3-4B.

The main research question is: which hidden layers, pooling strategies, and prompt-conditioned representations from Qwen3-4B best reflect semantic correctness?

## Research scope

The default implementation follows these design choices:

1. Qwen3-4B is used as the main latent representation extractor.
2. The primary latent metric is cosine similarity between pooled hidden-state vectors.
3. Dataset-level linear CKA is reported as an auxiliary representational similarity signal.
4. Prompt-conditioned residual representations are supported by subtracting a prompt-only encoding from the prompt-plus-answer encoding.
5. The main comparison baselines are Exact Match, token F1, ROUGE-L, BERTScore, and SimCSE cosine.

## Repository layout

```text
.
|- configs/
|  |- benchmark.yaml
|- data/
|  |- raw/
|  |- processed/
|- results/
|- scripts/
|  |- prepare_data.py
|  |- run_experiment.py
|- src/
|  |- latent_semantic_eval/
|     |- analysis.py
|     |- cli.py
|     |- config.py
|     |- datasets.py
|     |- generation.py
|     |- io_utils.py
|     |- metrics.py
|     |- modeling.py
|     |- pipeline.py
|     |- representations.py
|     |- schemas.py
|- pyproject.toml
|- requirements.txt
```

## Environment requirements

The code was written to be run on a Linux server or workstation with a real Python environment. It was only checked for syntax in the current workspace and has not been executed end to end locally.

Minimum software requirements:

1. Python 3.10 or 3.11.
2. CUDA 12.x with a PyTorch build compatible with your driver.
3. pip 23.0 or newer.
4. git if you plan to clone and version results.

Recommended hardware for Qwen3-4B latent extraction:

1. One GPU with at least 48 GB VRAM.
2. 16 CPU cores or more for dataset preparation and result aggregation.
3. 64 GB system RAM or more.
4. 100 GB or more free disk space for model weights, caches, processed data, and results.

Optional but recommended:

1. flash-attn for faster attention on supported GPUs.
2. A Hugging Face access token if the server requires authenticated model downloads.

## Installation

Use one of the following installation paths on the server.

Option A: create and activate a fresh environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Option B: reuse an existing conda or system Python environment and skip environment creation.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you want Flash Attention and your GPU/toolchain support it, install it separately after PyTorch is available.

## Required datasets

The benchmark config mixes Hugging Face datasets and local JSONL files.

Included as Hugging Face sources in the default config:

1. STS-B validation split via glue/stsb.
2. PAWS validation split via paws/labeled_final.

Expected as local JSONL files in data/raw:

1. data/raw/sickr_test.jsonl
2. data/raw/summeval.jsonl

See data/raw/README.md for the exact JSONL schema expected by the loaders.

## Model access

The default model is Qwen/Qwen3-4B.

Before running on a server, verify the following:

1. The server can download the model from Hugging Face.
2. Your Hugging Face account has permission if gated access applies.
3. The installed transformers version supports Qwen3 hidden-state extraction and chat templating.

If authentication is required:

```bash
huggingface-cli login
```

## Configuration

The main experiment configuration is stored in configs/benchmark.yaml.

Important fields to review before running:

1. model.model_name_or_path
2. model.torch_dtype
3. model.max_input_length
4. representation.layers
5. representation.poolings
6. representation.use_prompt_residual
7. dataset paths for SICK-R and SummEval

The local dataset format is intentionally explicit so that you can replace it with your own annotated data later without changing core code.

## Reproduction workflow

### Step 1: Prepare datasets

Standardize all configured datasets into a shared JSONL format:

```bash
python scripts/prepare_data.py --config configs/benchmark.yaml
```

This writes standardized files into data/processed.

### Step 2: Run the full evaluation pipeline

```bash
python scripts/run_experiment.py --config configs/benchmark.yaml
```

The pipeline will:

1. Load or standardize the configured datasets.
2. Optionally generate missing candidates with Qwen3-4B.
3. Extract hidden-state representations for candidates and references.
4. Compute lexical, semantic, and latent-space metrics.
5. Save row-level results and dataset summaries to results.

### Step 3: Inspect outputs

Expected outputs include:

1. Per-dataset row-level scores in CSV and JSONL form.
2. Per-dataset metric summaries with correlations and binary classification statistics.
3. A combined summary file aggregating all datasets.
4. A copy of the resolved experiment config in the result directory.

### Optional syntax-only validation

Before launching GPU jobs on the server, you can run a syntax-only verification pass:

```bash
python -m compileall src scripts
```

This checks Python syntax without downloading models or executing the full pipeline.

## Local JSONL schema

Each JSONL line should contain one evaluation record. The loader accepts any superset of the following keys.

Example for SICK-R style data:

```json
{"id": "sickr-0001", "sentence1": "A dog is running.", "sentence2": "An animal runs outdoors.", "relatedness_score": 4.3}
```

Example for SummEval style data:

```json
{"id": "summeval-0001", "prompt": "Summarize the following article: ...", "candidate": "System summary text.", "reference": "Reference summary text.", "consistency_score": 4.5}
```

If you want the pipeline to let Qwen3-4B generate candidates automatically, leave the configured candidate field empty in your local file and set generate_candidate: true for that dataset in the config.

## Main outputs and how to read them

For each dataset, the project saves:

1. record_scores.csv: one row per example with all metric values.
2. record_scores.jsonl: the same information in JSONL form.
3. metric_summary.csv: per-metric aggregate statistics.
4. metric_summary.json: the same summary in JSON form.

Key columns in metric_summary.csv:

1. spearman and pearson for human-scored datasets.
2. pairwise_accuracy for ranking consistency.
3. auc and best_f1 for binary datasets such as PAWS.
4. dataset_level_cka for latent representation families.
5. spearman_ci_low and spearman_ci_high for bootstrap intervals.

## Suggested first experiment

To keep the first run manageable, use the default benchmark config and focus interpretation on:

1. Which Qwen3-4B layers perform best on STS-B.
2. Whether latent cosine is more robust than lexical overlap on PAWS.
3. Whether prompt residual representations help on generation-style evaluation.
4. Whether BERTScore remains a stronger baseline than latent similarity on your first pass.

## Implementation notes

1. Hidden states are extracted with output_hidden_states=True.
2. Latent cosine is computed example by example.
3. Linear CKA is computed at the dataset level over the matrix of candidate and reference embeddings.
4. Prompt-conditioned residuals are computed by subtracting a prompt-only pooled vector from a prompt-plus-answer pooled vector.
5. The project is designed to be readable and editable for research, not to be a fully optimized production system.

## Known limitations

1. The current implementation prioritizes clarity and reproducibility over maximal throughput.
2. SummEval and SICK-R are assumed to be prepared locally in the expected JSONL format.
3. The pipeline does not include notebook visualizations yet; it focuses on reproducible batch runs.
4. The code has only been checked for syntax in this workspace, not executed end to end on GPU hardware.

## Recommended next extensions

1. Add layer-wise heatmap plotting scripts.
2. Add thinking vs no-thinking ablation using Qwen3 generation outputs.
3. Add an NLI-based semantic consistency baseline.
4. Add a lightweight learned combiner on top of latent and baseline features.