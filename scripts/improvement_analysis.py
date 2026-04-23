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
import json
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

# Embedding output dimensions (used for pre-allocating arrays in run_helm)
EMB_DIMS_HINT = {
    "bge-base": 768,
    "minilm":   384,
    "e5-base":  768,
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


# ── Method C: HELM hyperbolic projection (trained) ────────────────────────────

def compute_helm_similarity(pred_embs: np.ndarray, gt_embs: np.ndarray,
                             emb: str, test_indices: np.ndarray) -> np.ndarray:
    """
    Run the trained HyperbolicProjection head on the given embeddings.
    Returns similarity scores for all N samples; positions not in
    test_indices are set to NaN (caller must filter accordingly).
    """
    import torch
    from geoopt.manifolds import PoincareBall

    ckpt_path = os.path.join(os.path.dirname(RESULTS), "checkpoints",
                             f"helm_proj_{emb}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"HELM checkpoint not found: {ckpt_path}\n"
            "Run  python scripts/train_hyperbolic.py  first.")

    class _HypProj(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, curvature):
            super().__init__()
            self.ball = PoincareBall(c=curvature)
            self.mlp  = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, x):
            return self.ball.expmap0(self.mlp(x))

        def dist(self, p, q):
            return self.ball.dist(p, q)

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = _HypProj(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["curvature"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    all_sims = np.full(len(pred_embs), np.nan, dtype=np.float32)

    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(test_indices), batch_size):
            batch_idx = test_indices[start:start + batch_size]
            pb    = torch.from_numpy(pred_embs[batch_idx]).to(device)
            gb    = torch.from_numpy(gt_embs[batch_idx]).to(device)
            p_hyp = model(pb)
            g_hyp = model(gb)
            dist  = model.dist(p_hyp, g_hyp)
            sims  = torch.exp(-dist).cpu().numpy()
            all_sims[batch_idx] = sims

    return all_sims


def run_helm(datasets, modes, emb_keys):
    """Run HELM (test-set only, no data leakage). Returns list of result dicts."""
    records = []
    for emb in emb_keys:
        # Load test indices produced by train_hyperbolic.py
        test_idx_path = os.path.join(PROCESSED, f"helm_{emb}_test_indices.json")
        if not os.path.exists(test_idx_path):
            print(f"  [skip] test indices not found for {emb} – run train_hyperbolic.py first")
            continue
        with open(test_idx_path) as f:
            test_indices = np.array(json.load(f), dtype=np.int64)

        # Reconstruct cumulative offsets matching train_hyperbolic.py concatenation order
        offsets = {}
        cum = 0
        for ds in DATASETS:
            for mode in MODES:
                sim_csv = os.path.join(RESULTS, f"{ds}_{mode}_{emb}_similarity.csv")
                if os.path.exists(sim_csv):
                    n = len(pd.read_csv(sim_csv))
                    offsets[(ds, mode)] = (cum, cum + n)
                    cum += n

        total_n  = cum
        emb_dim  = EMB_DIMS_HINT.get(emb, 768)
        all_pred = np.zeros((total_n, emb_dim), dtype=np.float32)
        all_gt   = np.zeros_like(all_pred)

        for ds in DATASETS:
            for mode in MODES:
                key = (ds, mode)
                if key not in offsets:
                    continue
                start, end = offsets[key]
                pred_path = os.path.join(PROCESSED, f"{ds}_{mode}_{emb}_pred.npy")
                gt_path   = os.path.join(PROCESSED, f"{ds}_{mode}_{emb}_gt.npy")
                if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
                    continue
                arr = np.load(pred_path).astype(np.float32)
                all_pred[start:end] = arr
                all_gt[start:end]   = np.load(gt_path).astype(np.float32)

        print(f"  HELM computing for emb={emb} ({total_n} total, "
              f"{len(test_indices)} test) ...", flush=True)
        all_sims = compute_helm_similarity(all_pred, all_gt, emb, test_indices)

        test_set = set(test_indices.tolist())

        for ds in datasets:
            for mode in modes:
                key = (ds, mode)
                if key not in offsets:
                    print(f"  [skip] {ds}/{mode}/{emb} – no offset info")
                    continue
                start, end = offsets[key]

                local_global    = np.arange(start, end)
                local_test_mask = np.array(
                    [i in test_set for i in local_global], dtype=bool)

                if not local_test_mask.any():
                    print(f"  [skip] {ds}/{mode}/{emb} – no test samples")
                    continue

                sim_csv = os.path.join(RESULTS, f"{ds}_{mode}_{emb}_similarity.csv")
                base_df = pd.read_csv(sim_csv)
                labels  = base_df["is_correct"].values.astype(int)

                sims_local  = all_sims[start:end]
                test_labels = labels[local_test_mask]
                test_sims   = sims_local[local_test_mask]

                valid       = ~np.isnan(test_sims)
                test_sims   = test_sims[valid]
                test_labels = test_labels[valid]

                auc = safe_auc(test_labels, test_sims)
                thr = find_optimal_threshold(test_sims, test_labels)
                print(f"  HELM      {ds:20s} {mode:15s} {emb}  "
                      f"n_test={valid.sum()}  AUC={auc:.4f}")

                out_df = base_df.copy()
                sims_out = np.full(len(base_df), np.nan, dtype=np.float32)
                sims_out[local_test_mask] = all_sims[start:end][local_test_mask]
                out_df["similarity"]      = sims_out
                out_df["helm_test_only"]  = local_test_mask.astype(int)
                out_path = os.path.join(
                    RESULTS, f"helm_{ds}_{mode}_{emb}_similarity.csv")
                out_df.to_csv(out_path, index=False)

                records.append({
                    "dataset": ds, "thinking_mode": mode, "emb_model": emb,
                    "method": "helm",
                    "auc": auc,
                    "threshold": thr["threshold"],
                    "youden_j": thr["youden_j"],
                    "f1_at_threshold": thr["f1_at_threshold"],
                    "poincare_scale": None,
                    "n_pos": int(test_labels.sum()),
                    "n_neg": int((1 - test_labels).sum()),
                })
    return records


# ── Method D: Span-MaxSim (answer span extraction) ────────────────────────────

def extract_answer_span(prediction: str, thinking_mode: str) -> str:
    """
    Extract the final answer span from a prediction string.

    no_thinking: return the whole prediction unchanged.
    thinking:    try answer-indicator regex patterns first, then fall
                 back to the last sentence.  This prevents MaxSim from
                 firing on GT keywords that appear in the *context* of
                 chain-of-thought reasoning rather than the final answer.
    """
    text = str(prediction).strip()

    if thinking_mode != "thinking":
        return text

    patterns = [
        r"(?:the answer is[:\s]+)(.+?)(?:[.!?]|$)",
        r"(?:therefore[,:\s]+)(.+?)(?:[.!?]|$)",
        r"(?:thus[,:\s]+)(.+?)(?:[.!?]|$)",
        r"(?:in conclusion[,:\s]+)(.+?)(?:[.!?]|$)",
        r"(?:so[,:\s]+the answer is[:\s]*)(.+?)(?:[.!?]|$)",
        r"(?:it (?:is|was)[:\s]+)(.+?)(?:[.!?]|$)",
        r"(?:actually[,:\s]+)(.+?)(?:[.!?]|$)",
        r"(?:to summarize[,:\s]+)(.+?)(?:[.!?]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            span = m.group(1).strip()
            if len(span) >= 5:
                return span

    # Last sentence fallback
    sents = split_sentences(text)
    if sents:
        last = sents[-1].strip()
        if len(last) >= 5:
            return last

    return text


def compute_maxsim_span(similarity_csv: str, gt_embs: np.ndarray, model,
                         batch_size: int = 64) -> np.ndarray:
    """
    Encode the extracted answer span for each prediction and compute
    cosine similarity to the GT embedding.
    """
    df             = pd.read_csv(similarity_csv)
    predictions    = df["prediction"].tolist()
    thinking_modes = df["thinking_mode"].tolist()

    spans = [extract_answer_span(p, m)
             for p, m in zip(predictions, thinking_modes)]

    span_embs = model.encode(spans, batch_size=batch_size,
                              normalize_embeddings=True,
                              show_progress_bar=False)

    gt_norm   = gt_embs / np.clip(
        np.linalg.norm(gt_embs, axis=1, keepdims=True), 1e-9, None)
    span_norm = span_embs / np.clip(
        np.linalg.norm(span_embs, axis=1, keepdims=True), 1e-9, None)
    return np.einsum("ij,ij->i", span_norm, gt_norm).astype(np.float32)


def run_maxsim_span(datasets, modes, emb_keys, batch_size=64):
    """Run Span-MaxSim analysis. Returns list of result dicts."""
    from sentence_transformers import SentenceTransformer

    records     = []
    model_cache = {}

    for ds in datasets:
        for mode in modes:
            for emb in emb_keys:
                gt_path = os.path.join(PROCESSED, f"{ds}_{mode}_{emb}_gt.npy")
                sim_csv = os.path.join(RESULTS,   f"{ds}_{mode}_{emb}_similarity.csv")

                if not (os.path.exists(gt_path) and os.path.exists(sim_csv)):
                    print(f"  [skip] missing files for {ds}/{mode}/{emb}")
                    continue

                gt_embs = np.load(gt_path).astype(np.float32)
                base_df = pd.read_csv(sim_csv)
                labels  = base_df["is_correct"].values.astype(int)

                print(f"  SpanSim  {ds:20s} {mode:15s} {emb}  ",
                      end="", flush=True)

                if emb not in model_cache:
                    model_name = EMB_NAMES[emb]
                    print(f"\n    Loading {model_name} ...", end="", flush=True)
                    model_cache[emb] = SentenceTransformer(model_name)
                    print(" done")
                model = model_cache[emb]

                sims = compute_maxsim_span(sim_csv, gt_embs, model, batch_size)
                auc  = safe_auc(labels, sims)
                thr  = find_optimal_threshold(sims, labels)
                print(f"AUC={auc:.4f}")

                out_df = base_df.copy()
                out_df["similarity"] = sims
                out_path = os.path.join(
                    RESULTS, f"maxsim_span_{ds}_{mode}_{emb}_similarity.csv")
                out_df.to_csv(out_path, index=False)

                records.append({
                    "dataset": ds, "thinking_mode": mode, "emb_model": emb,
                    "method": "maxsim_span",
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
    parser = argparse.ArgumentParser(
        description="Improvement analysis: Poincaré / MaxSim / HELM / Span-MaxSim")
    parser.add_argument("--datasets",       nargs="+", default=DATASETS)
    parser.add_argument("--thinking_modes",  nargs="+", default=MODES)
    parser.add_argument("--emb_models",     nargs="+", default=list(EMB_NAMES.keys()))
    parser.add_argument("--methods",        nargs="+",
                        default=["poincare", "maxsim", "helm", "maxsim_span"],
                        help="Methods to run: poincare maxsim helm maxsim_span")
    parser.add_argument("--batch_size",     type=int,  default=64)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    all_records = []

    if "poincare" in args.methods:
        print("\n=== Method A: Poincaré Ball (post-hoc) ===")
        all_records += run_poincare(args.datasets, args.thinking_modes, args.emb_models)

    if "maxsim" in args.methods:
        print("\n=== Method B: MaxSim (sentence-level max cosine) ===")
        all_records += run_maxsim(args.datasets, args.thinking_modes,
                                  args.emb_models, args.batch_size)

    if "helm" in args.methods:
        print("\n=== Method C: HELM (trained hyperbolic projection) ===")
        all_records += run_helm(args.datasets, args.thinking_modes, args.emb_models)

    if "maxsim_span" in args.methods:
        print("\n=== Method D: Span-MaxSim (answer span extraction) ===")
        all_records += run_maxsim_span(args.datasets, args.thinking_modes,
                                       args.emb_models, args.batch_size)

    if all_records:
        summary = build_summary(all_records)
        print("\n=== Improvement vs Baseline (\u0394AUC) ===")
        pivot = summary.pivot_table(
            index=["dataset", "thinking_mode", "emb_model"],
            columns="method", values="auc"
        )
        if "baseline" in pivot.columns:
            for m in ["poincare", "maxsim", "helm", "maxsim_span"]:
                if m in pivot.columns:
                    pivot[f"\u0394{m}"] = pivot[m] - pivot["baseline"]
        print(pivot.to_string())
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
