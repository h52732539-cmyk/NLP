#!/usr/bin/env bash
# rerun_sciq_no_thinking.sh
# 重跑 sciq/no_thinking 实验（该实验仅跑了 102/500 条样本）
# 使用 conda 环境 nlp，在计算节点（gpu2-3）上运行

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROCESSED_DIR="$PROJECT_DIR/data/processed"
RESULTS_DIR="$PROJECT_DIR/results"

echo "========================================"
echo "  NLP rerun: sciq / no_thinking"
echo "  Project dir: $PROJECT_DIR"
echo "========================================"

# ── Step 0: 删除不完整文件 ─────────────────────────────────────────────────
echo ""
echo "[Step 0] 删除不完整的 sciq/no_thinking 文件 ..."

STALE_FILES=(
    "$PROCESSED_DIR/sciq_no_thinking_predictions.jsonl"
    "$PROCESSED_DIR/sciq_no_thinking_bge-base_pred.npy"
    "$PROCESSED_DIR/sciq_no_thinking_bge-base_gt.npy"
    "$PROCESSED_DIR/sciq_no_thinking_bge-base_ids.json"
    "$PROCESSED_DIR/sciq_no_thinking_minilm_pred.npy"
    "$PROCESSED_DIR/sciq_no_thinking_minilm_gt.npy"
    "$PROCESSED_DIR/sciq_no_thinking_minilm_ids.json"
    "$PROCESSED_DIR/sciq_no_thinking_e5-base_pred.npy"
    "$PROCESSED_DIR/sciq_no_thinking_e5-base_gt.npy"
    "$PROCESSED_DIR/sciq_no_thinking_e5-base_ids.json"
)

for f in "${STALE_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        rm "$f"
        echo "  Deleted: $f"
    else
        echo "  Not found (skip): $f"
    fi
done

cd "$PROJECT_DIR"

# ── Step 1: 重跑推理 ───────────────────────────────────────────────────────
echo ""
echo "[Step 1] 重跑 sciq/no_thinking 推理 ..."
python scripts/generate_predictions.py \
    --datasets sciq \
    --thinking_modes no_thinking

echo ""
echo "[Step 1] 验证行数 ..."
LINES=$(wc -l < "$PROCESSED_DIR/sciq_no_thinking_predictions.jsonl")
echo "  sciq_no_thinking_predictions.jsonl: $LINES 行 (期望 500)"
if [[ "$LINES" -ne 500 ]]; then
    echo "  [ERROR] 行数不符，请检查推理步骤！"
    exit 1
fi

# ── Step 2: 编码 embeddings ────────────────────────────────────────────────
echo ""
echo "[Step 2] 编码 sciq/no_thinking embeddings ..."
python scripts/encode_embeddings.py \
    --datasets sciq \
    --thinking_modes no_thinking

# ── Step 3: 全量 similarity analysis（更新 summary_auc.csv）──────────────
echo ""
echo "[Step 3] 全量 similarity analysis（更新 summary_auc.csv）..."
python scripts/similarity_analysis.py

# ── Step 4: 全量 failure analysis（更新 failure_summary.csv）────────────
echo ""
echo "[Step 4] 全量 failure analysis（更新 failure_summary.csv）..."
python scripts/failure_analysis.py

echo ""
echo "========================================"
echo "  全部完成！"
echo "========================================"
