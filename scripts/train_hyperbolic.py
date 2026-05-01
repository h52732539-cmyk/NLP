"""
train_hyperbolic.py
-------------------
HELM-inspired training: learn a hyperbolic projection head on top of
frozen pre-computed embeddings, replacing the post-hoc exp_map approach.

Architecture (per embedding model):
    frozen embedding  (768 or 384 dim)
        → Linear(D, 256) → ReLU → Linear(256, 256)
        → geoopt PoincareBall.expmap0()     ← map to Poincaré ball

Loss:
    sim = exp(-poincare_dist(pred_hyp, gt_hyp))
    BCE(sim, is_correct.float())

Optimiser:
    geoopt.optim.RiemannianAdam (Riemannian gradient on manifold)
    lr=1e-3, 30 epochs, batch_size=64

Outputs:
    checkpoints/helm_proj_{emb}.pt          ← best-val-AUC checkpoint
    data/processed/helm_{emb}_test_indices.json   ← held-out 20% indices

Usage:
    python scripts/train_hyperbolic.py
    python scripts/train_hyperbolic.py --emb_models bge-base
    python scripts/train_hyperbolic.py --emb_models bge-base minilm e5-base \\
        --epochs 50 --lr 0.0005
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# geoopt: Riemannian optimisation on manifolds
import geoopt
from geoopt.manifolds import PoincareBall

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
PROCESSED   = os.path.join(ROOT_DIR, "data", "processed")
RESULTS     = os.path.join(ROOT_DIR, "results")
CKPT_DIR    = os.path.join(ROOT_DIR, "checkpoints")

DATASETS  = ["sciq", "simpleqa", "natural_questions", "truthfulqa"]
MODES     = ["no_thinking", "thinking"]
EMB_DIMS  = {
    "bge-base": 768,
    "minilm":   384,
    "e5-base":  768,
}

# ── Model ─────────────────────────────────────────────────────────────────────

class HyperbolicProjection(nn.Module):
    """
    MLP projection head that maps Euclidean embeddings into the Poincaré ball.

    The final Linear layer outputs an unconstrained vector; we then apply
    geoopt's expmap0 to project onto the ball (radius=1 by default).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 ball_curvature: float = 1.0):
        super().__init__()
        self.ball = PoincareBall(c=ball_curvature)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns points on the Poincaré ball (manifold-aware tensor)."""
        h = self.mlp(x)
        # expmap0 maps tangent vector at origin → point on manifold
        return self.ball.expmap0(h)

    def poincare_dist(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Geodesic distance on the Poincaré ball."""
        return self.ball.dist(p, q)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_triples(emb: str):
    """
    Load all (pred_emb, gt_emb, is_correct) triples from pre-computed .npy
    files across all datasets and thinking modes.

    Returns
    -------
    pred_embs : np.ndarray  shape (N, D)
    gt_embs   : np.ndarray  shape (N, D)
    labels    : np.ndarray  shape (N,)   int {0, 1}
    """
    import pandas as pd

    all_pred, all_gt, all_labels = [], [], []

    for ds in DATASETS:
        for mode in MODES:
            pred_path = os.path.join(PROCESSED, f"{ds}_{mode}_{emb}_pred.npy")
            gt_path   = os.path.join(PROCESSED, f"{ds}_{mode}_{emb}_gt.npy")
            sim_csv   = os.path.join(RESULTS,   f"{ds}_{mode}_{emb}_similarity.csv")

            if not (os.path.exists(pred_path) and os.path.exists(gt_path)
                    and os.path.exists(sim_csv)):
                continue

            pred_embs = np.load(pred_path).astype(np.float32)
            gt_embs   = np.load(gt_path).astype(np.float32)
            labels    = pd.read_csv(sim_csv)["is_correct"].values.astype(np.int32)

            if len(pred_embs) != len(labels):
                print(f"  [warn] length mismatch {ds}/{mode}/{emb}, skipping")
                continue

            all_pred.append(pred_embs)
            all_gt.append(gt_embs)
            all_labels.append(labels)

    if not all_pred:
        raise RuntimeError(f"No data found for embedding model '{emb}'.")

    return (np.concatenate(all_pred,  axis=0),
            np.concatenate(all_gt,    axis=0),
            np.concatenate(all_labels, axis=0))


# ── Training ──────────────────────────────────────────────────────────────────

def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def train_one_model(emb: str, args) -> dict:
    """Train a HyperbolicProjection head for one embedding model."""

    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{emb}] device={device}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"[{emb}] Loading triples...", flush=True)
    pred_np, gt_np, labels_np = load_all_triples(emb)
    N = len(labels_np)
    print(f"[{emb}] Total samples: {N}  "
          f"(pos={int(labels_np.sum())}, neg={N - int(labels_np.sum())})")

    # ── 80/20 split ────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    split = int(0.8 * N)
    train_idx, test_idx = idx[:split], idx[split:]

    # Save test indices for use in improvement_analysis.py
    test_idx_path = os.path.join(PROCESSED, f"helm_{emb}_test_indices.json")
    with open(test_idx_path, "w") as f:
        json.dump(test_idx.tolist(), f)
    print(f"[{emb}] Test indices saved → {test_idx_path}")

    # Further split train into train/val (90/10 of train set)
    split_val = int(0.9 * split)
    train_idx_t = idx[:split_val]
    val_idx     = idx[split_val:split]

    def make_tensors(idxs):
        return (
            torch.from_numpy(pred_np[idxs]).to(device),
            torch.from_numpy(gt_np[idxs]).to(device),
            torch.from_numpy(labels_np[idxs].astype(np.float32)).to(device),
        )

    pred_tr, gt_tr, lab_tr = make_tensors(train_idx_t)
    pred_val, gt_val, lab_val = make_tensors(val_idx)

    train_ds = TensorDataset(pred_tr, gt_tr, lab_tr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # ── Model ──────────────────────────────────────────────────────────────
    input_dim = EMB_DIMS.get(emb, pred_np.shape[1])
    model = HyperbolicProjection(input_dim, hidden_dim=args.hidden_dim,
                                  ball_curvature=args.curvature).to(device)

    # RiemannianAdam respects the manifold structure of the output layer's
    # weights; standard Adam is used for the MLP parameters since they live
    # in Euclidean space.
    opt = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.lr,
                                       stabilize=10)
    criterion = nn.BCELoss()

    best_val_auc  = -np.inf
    best_ckpt_path = os.path.join(CKPT_DIR, f"helm_proj_{emb}.pt")
    history = {"train_loss": [], "val_auc": []}

    print(f"[{emb}] Training for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for pb, gb, lb in train_loader:
            opt.zero_grad()
            p_hyp = model(pb)   # points on Poincaré ball
            g_hyp = model(gb)
            dist  = model.poincare_dist(p_hyp, g_hyp)   # (B,)
            sim   = torch.exp(-dist)                      # ∈ (0, 1]
            loss  = criterion(sim, lb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(lb)

        avg_loss = epoch_loss / len(train_idx_t)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            p_hyp = model(pred_val)
            g_hyp = model(gt_val)
            dist  = model.poincare_dist(p_hyp, g_hyp)
            sims  = torch.exp(-dist).cpu().numpy()
        val_auc = safe_auc(lab_val.cpu().numpy(), sims)

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  val_auc={val_auc:.4f}")

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "state_dict": model.state_dict(),
                "input_dim":  input_dim,
                "hidden_dim": args.hidden_dim,
                "curvature":  args.curvature,
                "emb":        emb,
                "best_val_auc": best_val_auc,
            }, best_ckpt_path)

    print(f"[{emb}] Best val AUC = {best_val_auc:.4f}  "
          f"checkpoint → {best_ckpt_path}")
    return {"emb": emb, "best_val_auc": best_val_auc,
            "history": history, "ckpt": best_ckpt_path}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train HELM-inspired hyperbolic projection heads")
    parser.add_argument("--emb_models", nargs="+",
                        default=list(EMB_DIMS.keys()),
                        choices=list(EMB_DIMS.keys()))
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--curvature",  type=float, default=1.0,
                        help="Curvature c of the Poincaré ball (default 1.0)")
    args = parser.parse_args()

    print("=" * 60)
    print("HELM Hyperbolic Projection Training")
    print(f"  embedding models : {args.emb_models}")
    print(f"  epochs           : {args.epochs}")
    print(f"  lr               : {args.lr}")
    print(f"  hidden_dim       : {args.hidden_dim}")
    print(f"  curvature        : {args.curvature}")
    print("=" * 60)

    results = []
    for emb in args.emb_models:
        try:
            res = train_one_model(emb, args)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {emb}: {e}")

    print("\n=== Training Summary ===")
    for r in results:
        print(f"  {r['emb']:12s}  best_val_AUC={r['best_val_auc']:.4f}  "
              f"ckpt={r['ckpt']}")


if __name__ == "__main__":
    main()
