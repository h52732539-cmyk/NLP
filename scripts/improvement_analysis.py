"""
improvement_analysis.py
-----------------------
Evaluate two improved similarity metrics against the cosine baseline:

  Method A — Poincaré Ball (hyperbolic embedding)
    Map pre-computed Euclidean embeddings to the Poincaré ball via the
    exponential map at the origin, then replace cosine similarity with
    poincaré distance (converted to a similarity score).
    Pure numpy, no geoopt required.

  Method B — MaxSim (sentence-level maximum cosine similarity)
    Split each LLM prediction into sentences, encode them with the same
    SentenceTransformer, and take the maximum cosine similarity to the
    ground-truth embedding instead of a single average embedding.

Outputs (all in results/):
  poincare_{ds}_{mode}_{emb}_similarity.csv
  maxsim_{ds}_{mode}_{emb}_similarity.csv
  improvement_summary.csv      ← baseline / poincare / maxsim AUC comparison

Usage:
  python scripts/improvement_analysis.py
  python scripts/improvement_analysis.py --datasets truthfulqa natural_questions
  python scripts/improvement_analysis.py --methods poincare
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
PROCESSED    = os.path.join(ROOT_DIR, "data", "processed")
RESULTS      = os.path.join(ROOT_DIR, "results")
CONFIG_PATH  = os.path.join(ROOT_DIR, "configs", "benchmark.yaml")

DATASETS  = ["sciq", "simpleqa", "natural_questions", "truthfulqa"]
MODES     = ["no_thinking", "thinking"]
EMB_NAMES = {
    "bge-base": "BAAI/bge-base-en-v1.5",
    "minilm":   "sentence-transformers/all-MiniLM-L6-v2",
    "e5-base":  "intfloat/e5-base-v2",
}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", a / np.clip(na, 1e-9, None),
                                  b / np.clip(nb, 1e-9, None))


def find_optimal_threshold(scores, labels, n_steps=200):
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    best_j, best_t, best_f1 = -np.inf, thresholds[0], 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j    = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)
            prec   = tp / (tp + fp) if (tp + fp) else 0.0
            best_f1 = 2*prec*sens/(prec+sens) if (prec+sens) else 0.0
    return {"threshold": best_t, "youden_j": float(best_j), "f1_at_threshold": float(best_f1)}


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ── Method A: Poincaré Ball ───────────────────────────────────────────────────

def exp_map_origin(v: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """
    Exponential map at the origin of the Poincaré ball.
    Maps Euclidean vector v → point on the Poincaré ball.
    scale controls how 'spread out' points are (avoid boundary collapse).
    """
    v_scaled = v * scale
    norms = np.linalg.norm(v_scaled, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    tanh_norms = np.tanh(norms)                   # tanh(||v||) ∈ (0, 1)
    return tanh_norms * v_scaled / norms           # ∈ Poincaré ball


def mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Möbius addition: (-x) ⊕ y in the Poincaré ball."""
    neg_x  = -x
    xy     = np.sum(neg_x * y, axis=1, keepdims=True)
    nx2    = np.sum(neg_x ** 2, axis=1, keepdims=True)
    ny2    = np.sum(y   ** 2, axis=1, keepdims=True)
    num    = (1 + 2 * xy + ny2) * neg_x + (1 - nx2) * y
    denom  = 1 + 2 * xy + nx2 * ny2
    return num / np.clip(denom, 1e-9, None)


def poincare_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Poincaré ball geodesic distance: d(x,y) = 2 arctanh(||(-x)⊕y||)
    """
    diff = mobius_add(x, y)
    norms = np.linalg.norm(diff, axis=1)
    norms = np.clip(norms, 0.0, 1.0 - 1e-6)      # must be < 1 for arctanh
    return 2.0 * np.arctanh(norms)


def poincare_similarity(pred_embs: np.ndarray, gt_embs: np.ndarray,
                         scale: float = 0.1) -> np.ndarray:
    """Convert Poincaré distance to similarity score in [0,1]."""
    p = exp_map_origin(pred_embs, scale)
    g = exp_map_origin(gt_embs,  scale)
    dist = poincare_distance(p, g)
    return np.exp(-dist)                           # ∈ (0, 1], higher = more similar


def grid_search_scale(pred_embs, gt_embs, labels, scales=(0.05, 0.1, 0.2)):
    """Pick the scale that maximises AUC on this split."""
    best_auc, best_scale = -np.inf, scales[0]
    for s in scales:
        sims = poincare_similarity(pred_embs, gt_embs, scale=s)
        a = safe_auc(labels, sims)
        if not np.isnan(a) and a > best_auc:
            best_auc, best_scale = a, s
    return best_scale


# ── Method B: MaxSim ─────────────────────────────────────────────────────────

def split_sentences(text: str):
    """Simple sentence splitter using regex (no nltk download required)."""
    text = str(text).strip()
    # Split on sentence-ending punctuation followed by whitespace or EOS
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 5]
    if not sents:
        sents = [text]
    return sents


def compute_maxsim(similarity_csv: str, gt_embs: np.ndarray, model,
                   batch_size: int = 64) -> np.ndarray:
    """
    For each prediction, split into sentences, encode, and take the max
    cosine similarity to the corresponding ground-truth embedding.
    Falls back to a single-sentence cosine when the split gives only one sentence.
    """
    df = pd.read_csv(similarity_csv)
    predictions = df["prediction"].tolist()
    n = len(predictions)
    max_sims = np.zeros(n, dtype=np.float32)

    for i, pred in enumerate(predictions):
        sents = split_sentences(pred)
        # Encode all sentences for this sample
        sent_embs = model.encode(sents, batch_size=batch_size,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
        # gt embedding for this sample
        gt = gt_embs[i:i+1]           # (1, D)
        gt_norm = gt / np.clip(np.linalg.norm(gt, axis=1, keepdims=True), 1e-9, None)
        sims = np.dot(sent_embs, gt_norm.T).flatten()
        max_sims[i] = float(np.max(sims))

    return max_sims


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_poincare(datasets, modes, emb_keys):
    """Run Poincaré Ball analysis. Returns list of result dicts."""
    records = []
    for ds in datasets:
        for mode in modes:
            for emb in emb_keys:
                pred_path = os.path.join(PROCESSED, f"{ds}_{mode}_{emb}_pred.npy")
                gt_path   = os.path.join(PROCESSED, f"{ds}_{mode}_{emb}_gt.npy")
                sim_csv   = os.path.join(RESULTS, f"{ds}_{mode}_{emb}_similarity.csv")

                if not (os.path.exists(pred_path) and os.path.exists(gt_path)
                        and os.path.exists(sim_csv)):
                    print(f"  [skip] missing files for {ds}/{mode}/{emb}")
                    continue

                pred_embs = np.load(pred_path).astype(np.float32)
                gt_embs   = np.load(gt_path).astype(np.float32)
                base_df   = pd.read_csv(sim_csv)
                labels    = base_df["is_correct"].values.astype(int)

                print(f"  Poincaré  {ds:20s} {mode:15s} {emb}  ", end="", flush=True)

                # Grid-search scale
                scale = grid_search_scale(pred_embs, gt_embs, labels)
                sims  = poincare_similarity(pred_embs, gt_embs, scale)

                auc = safe_auc(labels, sims)
                thr = find_optimal_threshold(sims, labels)
                print(f"scale={scale:.2f}  AUC={auc:.4f}")

                # Save per-sample CSV
                out_df = base_df.copy()
                out_df["poincare_scale"] = scale
                out_df["similarity"]     = sims
                out_path = os.path.join(RESULTS, f"poincare_{ds}_{mode}_{emb}_similarity.csv")
                out_df.to_csv(out_path, index=False)

                records.append({
                    "dataset": ds, "thinking_mode": mode, "emb_model": emb,
                    "method": "poincare",
                    "auc": auc,
                    "threshold": thr["threshold"],
                    "youden_j": thr["youden_j"],
                    "f1_at_threshold": thr["f1_at_threshold"],
                    "poincare_scale": scale,
                    "n_pos": int(labels.sum()),
                    "n_neg": int((1 - labels).sum()),
                })
    return records


def run_maxsim(datasets, modes, emb_keys, batch_size=64):
    """Run MaxSim analysis. Returns list of result dicts."""
    from sentence_transformers import SentenceTransformer

    records = []
    # Cache loaded models to avoid re-loading for each (ds, mode)
    model_cache = {}

    for ds in datasets:
        for mode in modes:
            for emb in emb_keys:
                gt_path  = os.path.join(PROCESSED, f"{ds}_{mode}_{emb}_gt.npy")
                sim_csv  = os.path.join(RESULTS,   f"{ds}_{mode}_{emb}_similarity.csv")

                if not (os.path.exists(gt_path) and os.path.exists(sim_csv)):
                    print(f"  [skip] missing files for {ds}/{mode}/{emb}")
                    continue

                gt_embs = np.load(gt_path).astype(np.float32)
                base_df = pd.read_csv(sim_csv)
                labels  = base_df["is_correct"].values.astype(int)

                print(f"  MaxSim   {ds:20s} {mode:15s} {emb}  ", end="", flush=True)

                # Load or reuse model
                if emb not in model_cache:
                    model_name = EMB_NAMES[emb]
                    print(f"\n    Loading {model_name} ...", end="", flush=True)
                    model_cache[emb] = SentenceTransformer(model_name)
                    print(" done")
                model = model_cache[emb]

                sims = compute_maxsim(sim_csv, gt_embs, model, batch_size=batch_size)

                auc = safe_auc(labels, sims)
                thr = find_optimal_threshold(sims, labels)
                print(f"AUC={auc:.4f}")

                # Save per-sample CSV
                out_df = base_df.copy()
                out_df["similarity"] = sims
                out_path = os.path.join(RESULTS, f"maxsim_{ds}_{mode}_{emb}_similarity.csv")
                out_df.to_csv(out_path, index=False)

                records.append({
                    "dataset": ds, "thinking_mode": mode, "emb_model": emb,
                    "method": "maxsim",
                    "auc": auc,
                    "threshold": thr["threshold"],
                    "youden_j": thr["youden_j"],
                    "f1_at_threshold": thr["f1_at_threshold"],
                    "poincare_scale": None,
                    "n_pos": int(labels.sum()),
                    "n_neg": int((1 - labels).sum()),
                })
    return records


def build_summary(new_records: list):
    """Merge baseline AUC with new method results."""
    baseline_path = os.path.join(RESULTS, "summary_auc.csv")
    if os.path.exists(baseline_path):
        base = pd.read_csv(baseline_path)[
            ["dataset", "thinking_mode", "emb_model", "auc",
             "threshold", "youden_j", "f1_at_threshold", "n_pos", "n_neg"]
        ].copy()
        base["method"] = "baseline"
        base["poincare_scale"] = None
    else:
        base = pd.DataFrame()

    new_df = pd.DataFrame(new_records) if new_records else pd.DataFrame()
    combined = pd.concat([base, new_df], ignore_index=True)
    combined = combined.sort_values(["dataset", "thinking_mode", "emb_model", "method"])

    out_path = os.path.join(RESULTS, "improvement_summary.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSummary saved → {out_path}")
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Improvement analysis: Poincaré + MaxSim")
    parser.add_argument("--datasets",      nargs="+", default=DATASETS)
    parser.add_argument("--thinking_modes", nargs="+", default=MODES)
    parser.add_argument("--emb_models",    nargs="+", default=list(EMB_NAMES.keys()))
    parser.add_argument("--methods",       nargs="+", default=["poincare", "maxsim"],
                        help="Which methods to run: poincare maxsim")
    parser.add_argument("--batch_size",    type=int,  default=64)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    all_records = []

    if "poincare" in args.methods:
        print("\n=== Method A: Poincaré Ball ===")
        all_records += run_poincare(args.datasets, args.thinking_modes, args.emb_models)

    if "maxsim" in args.methods:
        print("\n=== Method B: MaxSim (sentence-level max cosine) ===")
        all_records += run_maxsim(args.datasets, args.thinking_modes,
                                  args.emb_models, args.batch_size)

    if all_records:
        summary = build_summary(all_records)
        print("\n=== Improvement vs Baseline (ΔAUC) ===")
        pivot = summary.pivot_table(
            index=["dataset", "thinking_mode", "emb_model"],
            columns="method", values="auc"
        )
        if "baseline" in pivot.columns:
            for m in ["poincare", "maxsim"]:
                if m in pivot.columns:
                    pivot[f"Δ{m}"] = pivot[m] - pivot["baseline"]
        print(pivot.to_string())
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
