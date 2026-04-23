## 结果分析报告

### 1. 核心指标汇总 (AUC)

| Dataset | Task Type | no_thinking ||||  thinking |||
|---|---|---|---|---|---|---|---|---|
| | | BGE | MiniLM | E5 | BGE | MiniLM | E5 |
| **SciQ** | Short-form | 0.954 | **0.972** | 0.962 | 0.902 | 0.934 | 0.890 |
| **SimpleQA** | Short-form | **0.976** | 0.933 | 0.931 | 0.934 | 0.879 | 0.886 |
| **NQ** | Long-form | 0.905 | 0.894 | **0.934** | 0.880 | 0.869 | **0.899** |
| **TruthfulQA** | Long-form | 0.743 | 0.723 | 0.784 | 0.801 | 0.763 | **0.813** |

---

### 2. 核心发现

#### 2.1 Short-form vs Long-form：差异显著

Cosine similarity 在短问答任务上作为正确性代理的效果远优于长文本任务：
- SciQ / SimpleQA AUC 集中在 **0.88–0.98**
- TruthfulQA AUC 仅为 **0.72–0.81**
- Natural Questions 居中 **0.87–0.93**

**原因**：短答案嵌入向量语义高度集中，正确答案与预测之间方向一致性强；长文本答案包含更多冗余内容，与标准答案的 cosine 距离噪声大（TruthfulQA 案例显示预测"Fortune cookies originated in China..."与正确答案列表的相似度达 0.82 却被标记为错误）。

#### 2.2 Thinking 模式的双向影响（消融实验 Option B）

| Dataset | no_thinking 最佳 AUC | thinking 最佳 AUC | 趋势 |
|---|---|---|---|
| SciQ | 0.972 (MiniLM) | 0.934 (MiniLM) | **↓ 下降** |
| SimpleQA | 0.976 (BGE) | 0.934 (BGE) | **↓ 下降** |
| NQ | 0.934 (E5) | 0.899 (E5) | ↓ 轻微下降 |
| TruthfulQA | 0.784 (E5) | **0.813** (E5) | **↑ 上升** |

- 对于**短问答**：CoT 推理引入冗余词汇，使预测文本更长，与简短 ground truth 的 embedding 距离反而增大，AUC 下降约 3–4 个点。
- 对于 **TruthfulQA**：CoT 推理抑制了模型常见的"自信错误"（如 "veins appear blue because blood is blue"），thinking 模式下 Accuracy 虽略降（35.0% → 28.2%），但正确答案与错误答案之间的 cosine 差距更清晰（bge 均值差异：0.113 → 0.130），AUC 反而提升。

#### 2.3 嵌入模型对比

- **E5-base-v2** 最稳健：在 4 个数据集 × 2 模式共 8 个组合中，5 次位列最优或次优，尤其在长文本任务上表现突出（NQ: 0.934，TruthfulQA: 0.813）。
- **MiniLM-L6-v2** 在 SciQ 上最强（0.972），但长文本任务上最弱（TruthfulQA: 0.723），极化明显。
- **BGE-base-en-v1.5** 在 SimpleQA 上表现最佳（0.976），整体均衡但略逊于 E5。

**推荐**：若需要单一嵌入模型，E5-base-v2 是跨任务最佳选择。

#### 2.4 统计显著性：全部通过

所有 24 个 (dataset, mode, model) 组合的独立样本 t-test p 值均 $\ll 0.001$（最大为 $1.4 \times 10^{-4}$），确认 cosine similarity 对正确/错误答案的区分在统计上是可靠的。

各数据集的 **cosine 均值差异**（BGE，no_thinking）：

$$\Delta\bar{s} = \bar{s}_{\text{correct}} - \bar{s}_{\text{incorrect}}$$

| Dataset | $\bar{s}_{\text{correct}}$ | $\bar{s}_{\text{incorrect}}$ | $\Delta\bar{s}$ |
|---|---|---|---|
| SciQ | 0.930 | 0.697 | **+0.233** |
| SimpleQA | 0.925 | 0.554 | **+0.371** |
| NQ | 0.742 | 0.516 | **+0.226** |
| TruthfulQA | 0.791 | 0.678 | +0.113 |

#### 2.5 准确率：Thinking 模式并不总是提升正确性

| Dataset | no_thinking Accuracy | thinking Accuracy |
|---|---|---|
| SciQ | 56.9% (58/102) | 60.8% (304/500) |
| SimpleQA | 3.2% (16/500) | 3.2% (16/500) |
| **NQ** | 9.3% (28/300) | **4.0%** (12/300) |
| **TruthfulQA** | 35.0% (286/817) | **28.2%** (230/817) |

对于长文本任务，CoT 推理反而降低了 Qwen3-4B 的正确率，这与部分文献中关于 4B 小模型在复杂推理上"过度思考"的观察一致。

> **注意**：SciQ no_thinking 只运行了 102 条样本（vs thinking 的 500 条），可能是由于实验配置差异，在解读 SciQ no_thinking AUC 时需注意样本量偏小的影响。

---

### 3. 错误分析

| Dataset | FP | FN | 主要失败类别 |
|---|---|---|---|
| SciQ (no_thinking) | 3 | 10 | 语义歧义 (5)、其他 (6) |
| SciQ (thinking) | 16 | 50 | 语义歧义 (21)、其他 (39) |
| SimpleQA (no_thinking) | 50 | 0 | 词汇变体 (22)、语义歧义 (11) |
| NQ (no_thinking) | 50 | 1 | 长文本推理 (51) |
| TruthfulQA (no_thinking) | 50 | 50 | 长文本推理 (100) |

**主要规律：**
- **SimpleQA 几乎无 FN（0–1 个）**：当模型答对时，相似度极高（bge 均值 0.925），几乎不会被漏判；但 FP 严重（50 个），根本原因是 SimpleQA 的 ground truth 仅约 3.2% 为正确，大量错误答案的嵌入向量仍与问题主题相近，产生高相似度假阳性（FP 均值 ~0.759）。
- **NQ / TruthfulQA 的主导失败是"长文本推理"**：长答案包含大量铺垫内容，即使核心事实正确，embedding 也被稀释；即使核心事实错误，答案中的相关背景也会拉高相似度。
- **FN 相似度均值低于阈值说明真正失败**：SciQ no_thinking 的 FN 均值 0.824 仍高于 0.7，显示这些案例确为模型难以区分的边界样本，而非明显错误。

---

### 4. 最优配置推荐

| 场景 | 推荐配置 | AUC |
|---|---|---|
| Short-form 最优 | SimpleQA + no_thinking + BGE | **0.976** |
| Long-form 最优 | TruthfulQA + thinking + E5 | **0.813** |
| 跨任务最鲁棒 | E5-base-v2 + no_thinking | 平均 ~0.903 |

---

### 5. Part 4 Improvement Direction 的支撑

失败分析直接支持 Poincaré Ball 的动机：
1. TruthfulQA 的 FP 均值高达 **0.863**（bge no_thinking），这意味着错误答案与正确答案在欧式嵌入空间几乎无法分离；层次化几何（双曲空间）能更好捕捉答案之间的语义层次关系（如"Fortune cookies originated in China" 与 "... in the U.S." 的矛盾关系）。
2. 长文本推理失败 100+ 例均集中在 TruthfulQA，提示欧式 cosine 度量在捕捉"事实否定"语义时存在根本局限，这正是 HELM (arXiv:2505.24722) 所解决的问题。