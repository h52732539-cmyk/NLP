# Latent Semantic Evaluation — 管线流程详解

## 概览

本项目的核心目标是：**利用大语言模型（Qwen3-4B）的中间隐层表示（Latent Representations），对 candidate 文本与 reference 文本之间的语义相似度进行多维度评估**，并与传统词法指标、语义指标进行对比分析。

整体管线由以下五个阶段串联构成：

```
YAML 配置加载
     │
     ▼
数据集加载 / 缓存读取
     │
     ▼
（可选）候选文本生成
     │
     ▼
逐记录评估（词法 + 语义 + 隐层特征）
     │
     ▼
汇总统计 + 结果持久化
```

---

## 入口：`run_experiment`（`pipeline.py`）

```
scripts/run_experiment.py --config configs/benchmark.yaml
    └─► latent_semantic_eval.pipeline.run_experiment(config_path)
```

### 步骤 1：加载与初始化

| 操作 | 代码位置 | 说明 |
|------|----------|------|
| 解析 YAML 配置 | `config.py::load_experiment_config` | 将 YAML 反序列化为强类型 dataclass |
| 固定随机种子 | `pipeline.py::_set_seed` | 同时设置 `random`、`numpy`、`torch` 的种子 |
| 创建输出目录 | `io_utils.py::ensure_directory` | 按 `output_dir` 字段自动创建 |
| 保存已解析配置 | `output_dir/resolved_config.json` | 用于实验复现追溯 |

---

### 步骤 2：数据集循环（含缓存机制）

对配置中每个 `datasets` 条目（`DatasetRecipe`）依次执行：

#### 2a. 缓存命中检测（跳过重复评估）

```
output_dir/<dataset_name>/record_scores.csv
output_dir/<dataset_name>/metric_summary.csv
```

若以上两个文件均已存在，则：
- 直接读取 `metric_summary.csv`，跳过所有模型推理与评估
- 打印 `[cache] Loading existing results for <name>...`
- **`QwenRepresentationExtractor` 在此情况下不会被加载**，节省显存与时间

> 若需强制重新评估某个 dataset，删除对应子目录即可：
> ```bash
> rm -rf results/<experiment_name>/<dataset_name>/
> ```

#### 2b. 数据加载（`datasets.py::load_records`）

支持两种数据源：

| `source` 值 | 加载方式 | 说明 |
|-------------|----------|------|
| `hf` | `datasets.load_dataset(...)` | 从 HuggingFace Hub 拉取，使用 `cache_dir` 缓存 |
| `jsonl` | 本地文件读取 | 相对于项目根目录解析路径 |

**二级缓存（预处理缓存）**：加载后会优先检查 `data/processed/<name>.jsonl`，若存在则直接使用，无需重新转换字段。

每条记录被标准化为 `EvaluationRecord`（`schemas.py`），包含字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `record_id` | `str` | 记录唯一标识 |
| `prompt` | `str` | 输入提示（部分任务为空） |
| `candidate` | `str` | 待评估文本（可由模型生成） |
| `reference` | `str` | 参考文本（人工答案） |
| `human_score` | `float \| None` | 回归任务的人工评分 |
| `binary_label` | `int \| None` | 二分类任务的标签（0/1） |
| `metadata` | `dict` | 数据集名、任务类型、来源等辅助信息 |

#### 2c. 候选文本生成（可选）

当 `DatasetRecipe.generate_candidate = true` 时，对 `candidate` 为空的记录，使用 `QwenGenerator` 批量调用模型生成候选文本。

**`QwenGenerator`（`generation.py`）**：
1. 将 prompt 包装为 Chat Template 格式（支持 `enable_thinking` 模式）
2. 批量 `model.generate`，解码输出
3. 若输出含 `<think>...</think>` 块，通过 `split_reasoning` 自动分离推理链与答案

---

### 步骤 3：逐记录评估（`_evaluate_dataset`）

**懒加载**：`QwenRepresentationExtractor` 仅在首次真正需要时初始化，避免缓存命中时白白加载模型。

对每条 `EvaluationRecord`，依次计算以下三类指标：

#### 3a. 词法指标（Lexical Metrics）

在 `metrics.py` 中实现，逐记录计算：

| 指标 | 方法 | 说明 |
|------|------|------|
| `exact_match` | 规范化后字符串比较 | 完全匹配得 1，否则 0 |
| `token_f1` | Token 级 Precision/Recall 调和平均 | 使用词频计数求交集 |
| `rouge_l` | 最长公共子序列（LCS）F1 | Token 级 LCS |

规范化操作（`normalize_text`）：小写 + 去标点。

#### 3b. 隐层表示提取与相似度计算

这是本项目的核心步骤，由 `QwenRepresentationExtractor`（`representations.py`）执行：

**前向传播**（`extract_bundle`）：
```
prompt + candidate → Qwen3 前向传播 → 全层 hidden_states（L+1 层，每层 shape=[T, D]）
prompt only        → Qwen3 前向传播 → prompt_hidden_states（用于 residual 计算）
```

- 输入格式：`"Prompt:\n{prompt}\n\nAnswer:\n{text}"`（无 prompt 时直接使用 text）
- 使用 `output_hidden_states=True`，获取每一层的 token-level 激活值
- 截断至 `max_input_length`（默认 4096）

**向量池化**（`pool_vector`）：

对指定 `layer` 的 token 矩阵按 `pooling` 策略降维为单一向量：

| `pooling` | 说明 |
|-----------|------|
| `mean` | 对所有 token 向量取均值 |
| `last_token` | 取最后一个 token 的向量 |

**Prompt Residual**（`use_prompt_residual`）：

| `use_prompt_residual` | 向量计算方式 |
|-----------------------|-------------|
| `false` | 仅对 answer 部分的 token 池化（从 `answer_start` 截取） |
| `true` | `pool(combined) - pool(prompt_only)` 消除 prompt 影响 |

**组合展开**：每条记录产生 `len(poolings) × len(use_prompt_residual) × len(layers)` 个隐层特征对，每对计算余弦相似度：

$$\text{cosine}(\mathbf{v}_c, \mathbf{v}_r) = \frac{\mathbf{v}_c \cdot \mathbf{v}_r}{\|\mathbf{v}_c\| \cdot \|\mathbf{v}_r\|}$$

指标命名格式：
```
latent__pool={pooling}__residual={0|1}__layer={layer_idx}__cosine
```

例如（使用 `benchmark.yaml`）：`layers=[-1,-4,-8,-16]`、`poolings=[mean, last_token]`、`use_prompt_residual=[false, true]`，共产生 $2 \times 2 \times 4 = 16$ 个隐层指标。

#### 3c. 语义指标（Semantic Metrics）

在所有记录处理完毕后，对完整的 candidate/reference 列表批量计算：

| 指标 | 模型 | 说明 |
|------|------|------|
| `bertscore_f1` | `bert-score` 库 | 基于 BERT token 级对齐的 F1 |
| `simcse_cosine` | `sup-simcse-roberta-base` | SimCSE 句向量余弦相似度 |

---

### 步骤 4：汇总统计（`_summarize_metrics`）

对每个指标列，依据记录中存在的标注类型，分别计算以下统计量：

#### 回归任务（有 `human_score`）

| 统计量 | 方法 | 说明 |
|--------|------|------|
| `spearman` | Spearman 相关系数 | 排名相关性 |
| `pearson` | Pearson 相关系数 | 线性相关性 |
| `pairwise_accuracy` | 配对排序准确率 | 最多采样 50000 对，种子固定 |
| `spearman_ci_low/high` | Bootstrap 置信区间（500次，95% CI） | 评估稳定性 |

#### 二分类任务（有 `binary_label`）

| 统计量 | 方法 | 说明 |
|--------|------|------|
| `auc` | ROC AUC | 需至少两类标签 |
| `best_f1` | 最优阈值 F1 | 扫描所有分数值作为阈值 |

#### 数据集级 CKA（若 `compute_dataset_level_cka = true`）

对隐层指标，计算 candidate 向量矩阵与 reference 向量矩阵之间的线性 CKA（Centered Kernel Alignment）：

$$\text{CKA}(X, Y) = \frac{\|X^\top Y\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

其中 $X, Y$ 为中心化后的向量矩阵（每列减去均值）。

---

### 步骤 5：结果持久化

#### 每个 dataset 子目录（`output_dir/<dataset_name>/`）

| 文件 | 内容 |
|------|------|
| `record_scores.csv` | 每条记录的所有原始指标分数（行=记录，列=指标） |
| `record_scores.jsonl` | 同上，JSONL 格式 |
| `metric_summary.csv` | 每个指标的汇总统计量（Spearman/AUC/CKA 等） |
| `metric_summary.json` | 同上，JSON 格式 |

#### 顶层汇总（`output_dir/`）

| 文件 | 内容 |
|------|------|
| `resolved_config.json` | 实验运行时的完整配置（含所有默认值） |
| `combined_metric_summary.csv` | 所有 dataset 的 metric_summary 合并表 |
| `combined_metric_summary.json` | 同上，JSON 格式 |

---

## 模块依赖关系

```
pipeline.py
├── config.py          ← YAML 解析，ExperimentConfig / DatasetRecipe 等 dataclass
├── datasets.py        ← EvaluationRecord 加载（HF / JSONL / 预处理缓存）
├── schemas.py         ← EvaluationRecord 数据结构
├── generation.py      ← QwenGenerator（可选候选文本生成）
├── representations.py ← QwenRepresentationExtractor（隐层特征提取与池化）
│   └── modeling.py    ← load_tokenizer_and_model（模型加载）
├── metrics.py         ← 词法指标 + BERTScore + SimCSE + 余弦相似度 + CKA
├── analysis.py        ← Spearman/Pearson/AUC/Best-F1/Pairwise Acc/Bootstrap CI
└── io_utils.py        ← ensure_directory / write_json / write_jsonl / read_jsonl
```

---

## 配置字段速查（`benchmark.yaml`）

```yaml
experiment_name: <实验名>
seed: 42                    # 全局随机种子
output_dir: results/<name>  # 输出根目录（相对项目根）
cache_dir: data/cache        # HF 数据集缓存目录

model:
  model_name_or_path: Qwen/Qwen3-4B
  torch_dtype: bfloat16
  device_map: auto
  attn_implementation: flash_attention_2
  max_input_length: 4096
  max_new_tokens: 256
  temperature: 0.0
  enable_thinking: false

representation:
  layers: [-1, -4, -8, -16]           # 负数表示从最后一层倒数
  poolings: [mean, last_token]
  use_prompt_residual: [false, true]

metrics:
  lexical: [exact_match, token_f1, rouge_l]
  semantic: [bertscore_f1, simcse_cosine]
  latent: [cosine]
  compute_dataset_level_cka: true

datasets:
  - name: stsb_validation              # 回归任务（human_score）
    source: hf
    task_type: regression

  - name: paws_validation              # 二分类任务（binary_label）
    source: hf
    task_type: binary

  - name: sickr_local                  # 本地 JSONL，回归任务
    source: jsonl

  - name: summeval_local               # 生成评估任务
    source: jsonl
    generate_candidate: false
```

---

## 缓存策略总结

| 层级 | 缓存路径 | 命中条件 | 跳过内容 |
|------|----------|----------|----------|
| 预处理缓存 | `data/processed/<name>.jsonl` | 文件存在 | HF 数据集下载与字段转换 |
| 评估结果缓存 | `output_dir/<name>/record_scores.csv` + `metric_summary.csv` | 两者均存在 | 模型加载、全部推理、指标计算 |
| HF 数据集缓存 | `data/cache/` | HF 自动管理 | 重复下载 |
