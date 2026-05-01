#!/usr/bin/env bash
# rerun_improvements_and_viz.sh
# 重跑改进方法分析和可视化
# 前提：sciq/no_thinking 推理+embedding 已完成（由 rerun_sciq_no_thinking.sh 完成）
# 使用 conda 环境 nlp，在计算节点（gpu2-3）上运行

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROCESSED_DIR="$PROJECT_DIR/data/processed"
RESULTS_DIR="$PROJECT_DIR/results"
CKPT_DIR="$PROJECT_DIR/checkpoints"
LOGS_DIR="$PROJECT_DIR/logs"

echo "========================================"
echo "  NLP improvements + viz rerun"
echo "  Project dir: $PROJECT_DIR"
echo "========================================"

cd "$PROJECT_DIR"

# ── Step 1: 删除 sciq/no_thinking 的旧改进 CSV ───────────────────────────────
echo ""
echo "[Step 1] 删除 sciq/no_thinking 的旧改进输出 CSV ..."
for method in poincare maxsim maxsim_span helm; do
    for emb in bge-base minilm e5-base; do
        f="$RESULTS_DIR/${method}_sciq_no_thinking_${emb}_similarity.csv"
        if [[ -f "$f" ]]; then
            rm "$f"
            echo "  Deleted: $f"
        fi
    done
done
# 同时删除旧的 improvement_summary.csv（将由全量重跑重建）
if [[ -f "$RESULTS_DIR/improvement_summary.csv" ]]; then
    rm "$RESULTS_DIR/improvement_summary.csv"
    echo "  Deleted: $RESULTS_DIR/improvement_summary.csv"
fi

# ── Step 2: 运行 Method A/B/D（poincare / maxsim / maxsim_span）全量重建 ──────
echo ""
echo "[Step 2] 运行 improvement_analysis.py（Method A/B/D）..."
python scripts/improvement_analysis.py --methods poincare maxsim maxsim_span

echo ""
echo "[Step 2] 验证 improvement_summary.csv ..."
ROWS=$(tail -n +2 "$RESULTS_DIR/improvement_summary.csv" | wc -l)
echo "  improvement_summary.csv: $ROWS 行"

# ── Step 3: 重训练 HELM（因为 sciq/no_thinking embedding 已更新）─────────────
echo ""
echo "[Step 3] 重训练 HELM 投影头（train_hyperbolic.py）..."
# 删除旧 checkpoint（强制重训练）
for emb in bge-base minilm e5-base; do
    ckpt="$CKPT_DIR/helm_proj_${emb}.pt"
    idx="$PROCESSED_DIR/helm_${emb}_test_indices.json"
    [[ -f "$ckpt" ]] && rm "$ckpt" && echo "  Deleted: $ckpt"
    [[ -f "$idx" ]]  && rm "$idx"  && echo "  Deleted: $idx"
done
python scripts/train_hyperbolic.py

# ── Step 4: 运行 Method C（HELM）全量 ─────────────────────────────────────────
echo ""
echo "[Step 4] 运行 improvement_analysis.py（Method C: HELM）..."
# 删除旧 helm similarity CSVs（所有数据集，因模型已重训）
find "$RESULTS_DIR" -name "helm_*_similarity.csv" -delete
echo "  旧 helm similarity CSV 已全部删除"
python scripts/improvement_analysis.py --methods helm

# ── Step 5: 重新执行 empirical_study.ipynb ────────────────────────────────────
echo ""
echo "[Step 5] 重新执行 empirical_study.ipynb ..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=600 \
    --output empirical_study_executed.ipynb \
    --output-dir notebooks \
    notebooks/empirical_study.ipynb
echo "  Saved → notebooks/empirical_study_executed.ipynb"

# ── Step 6: 重新执行 improvement_study.ipynb ─────────────────────────────────
echo ""
echo "[Step 6] 重新执行 improvement_study.ipynb ..."
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=600 \
    --output improvement_study_executed.ipynb \
    --output-dir notebooks \
    notebooks/improvement_study.ipynb
echo "  Saved → notebooks/improvement_study_executed.ipynb"

echo ""
echo "========================================"
echo "  全部完成！"
echo "========================================"
echo ""
echo "生成的文件："
echo "  results/improvement_summary.csv"
echo "  results/figures/*.pdf"
echo "  notebooks/empirical_study_executed.ipynb"
echo "  notebooks/improvement_study_executed.ipynb"
