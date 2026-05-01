# Semantic Similarity Measurement in Latent Space for LLM Prediction Evaluation

**Course:** AIAA 4051 — Final Research Project  
**Authors:** Xuhaodong, Yuxuanzhao  
**Affiliation:** The Hong Kong University of Science and Technology (Guangzhou)  
**Date:** April 2026

---

## Abstract

Evaluating Large Language Model (LLM) predictions remains challenging, particularly in the presence of diverse valid answers and hallucinations. This paper investigates whether cosine similarity in embedding space can serve as a reliable proxy for factual correctness across four benchmark datasets spanning short-form QA (SciQ, SimpleQA) and long-form QA (Natural Questions, TruthfulQA). Using Qwen3-4B under both direct-answer (*no-thinking*) and chain-of-thought (*thinking*) modes, we encode predictions and ground-truth answers with three pretrained embedding models (BGE-base-en-v1.5, all-MiniLM-L6-v2, E5-base-v2) and measure AUC as the primary discriminability metric.

Our main findings are: (1) cosine similarity is a strong correctness indicator for short-form QA (**AUC up to 0.976**) but degrades substantially for long-form QA (TruthfulQA AUC 0.72–0.81); (2) chain-of-thought reasoning *hurts* discriminability for short answers but *improves* it for TruthfulQA; (3) E5-base-v2 is the most robust embedding model across tasks. A trained hyperbolic projection (HELM-style) achieves the largest consistent gains on TruthfulQA (+7.1%–+12.3% AUC).

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [Methodology](#3-methodology)
4. [Experimental Setup](#4-experimental-setup)
5. [Results](#5-results)
   - [5.1 Main Results](#51-main-results)
   - [5.2 Short-form vs. Long-form QA](#52-short-form-vs-long-form-qa)
   - [5.3 Embedding Model Comparison](#53-embedding-model-comparison)
   - [5.4 Chain-of-Thought Ablation](#54-chain-of-thought-ablation)
6. [Failure Analysis](#6-failure-analysis)
7. [Improvement Methods](#7-improvement-methods)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Introduction

Large Language Models (LLMs) have achieved remarkable performance on a wide array of question-answering (QA) benchmarks. However, assessing the *correctness* of generated responses at scale is itself a non-trivial problem: exact-match metrics miss semantically equivalent paraphrases, while task-specific rule sets are difficult to generalise. An appealing alternative is to measure the **semantic similarity** between a model prediction and the ground-truth answer in a shared embedding space, using cosine similarity as a lightweight, model-agnostic proxy for correctness.

This approach raises two fundamental questions:
1. Is embedding similarity genuinely discriminative between correct and incorrect predictions—and to what degree?
2. Does the answer depend on the *type* of QA task (concise factual answers vs. open-ended explanations)?

We address both questions with a controlled study spanning four datasets and three embedding models, and extend the analysis with an ablation on LLM generation mode (chain-of-thought reasoning vs. direct answers).

**Contributions:**
- A systematic benchmark comparing cosine-similarity discriminability across 2 short-form and 2 long-form QA datasets
- An ablation study on the effect of chain-of-thought reasoning on embedding similarity
- A categorised failure analysis attributing errors to lexical variation, semantic ambiguity, and long-form reasoning complexity
- Evaluation of four improvement strategies, including a trained hyperbolic projection achieving consistent gains on TruthfulQA

---

## 2. Related Work

**Embedding-based evaluation metrics.** BERTScore uses contextual token-level similarities from BERT to evaluate text generation. Sentence-BERT produces fixed-size sentence vectors well-suited for semantic textual similarity (STS) benchmarks. E5 and BGE establish state-of-the-art results on MTEB, providing powerful general-purpose embeddings. However, none of these studies directly benchmark embedding similarity as a *correctness classifier* on diverse QA tasks.

**Hallucination in LLMs.** Prior work characterises hallucination as generation of plausible but factually unsupported content. RAGTruth provides a corpus of RAG hallucinations, revealing that cosine similarity alone conflates faithful and unfaithful responses. Our work explores this limitation systematically on non-RAG QA.

**Chain-of-thought prompting.** Wei et al. (2022) showed that chain-of-thought (CoT) prompting substantially improves multi-step reasoning. Recent work has also noted that CoT can introduce verbose, off-topic text for smaller models, potentially distorting embedding representations. We test this hypothesis empirically using Qwen3-4B.

**Hyperbolic embeddings for NLP.** Poincaré embeddings and HELM (He et al., 2025) demonstrate that hyperbolic geometry captures hierarchical semantic structure more faithfully than Euclidean cosine similarity. We evaluate a HELM-inspired trained projection head as an improvement strategy.

---

## 3. Methodology

### 3.1 Datasets

| Dataset | Type | Size | Description |
|---|---|---|---|
| **SciQ** | Short-form | 500 | Multiple-choice science questions, short ground-truth answers |
| **SimpleQA** | Short-form | 500 | Single-hop factual questions with one short answer string |
| **Natural Questions (NQ)** | Long-form | 300 | Open-domain questions with answer spans from Wikipedia |
| **TruthfulQA** | Long-form | 817 | Adversarially curated questions targeting common misconceptions |

### 3.2 LLM Inference

All predictions are generated by **Qwen3-4B** loaded at 4-bit NF4 quantisation on a single NVIDIA GPU. Two modes are evaluated:

- **`no_thinking`**: Direct answer generation (temperature 0). Responses are typically one short sentence or phrase.
- **`thinking`**: Chain-of-thought reasoning enabled via system prompt. Responses contain explicit reasoning steps before a final answer.

Ground-truth correctness labels use a hybrid exact-match / contains-match criterion:
$$\text{is\_correct} = \max_i \bigl[\texttt{em}(p, g_i) + \texttt{contains}(p, g_i)\bigr] > 0$$

### 3.3 Embedding Models

Three pretrained sentence encoders:
| Model | Dimensions | Parameters |
|---|---|---|
| BGE-base-en-v1.5 | 768 | 109M |
| all-MiniLM-L6-v2 | 384 | 22M |
| E5-base-v2 | 768 | 109M |

### 3.4 Similarity Computation and Evaluation

Cosine similarity between prediction embedding $\mathbf{e}_p$ and ground-truth embedding $\mathbf{e}_g$:
$$s = \frac{\mathbf{e}_p \cdot \mathbf{e}_g}{\|\mathbf{e}_p\|\,\|\mathbf{e}_g\|}$$

Primary metric: **AUC** (Area Under ROC Curve). Threshold selected by maximising Youden's J statistic. Statistical significance assessed by two-sample t-tests.

---

## 4. Experimental Setup

### Correctness Rates

| Dataset | Type | no_thinking | thinking | n |
|---|---|---|---|---|
| SciQ | Short | 52.2% (261/500) | 60.8% (304/500) | 500 |
| SimpleQA | Short | 3.2% (16/500) | 3.2% (16/500) | 500 |
| NQ | Long | 9.3% (28/300) | 4.0% (12/300) | 300 |
| TruthfulQA | Long | 35.0% (286/817) | 28.2% (230/817) | 817 |

### Statistical Validation

All 24 (dataset, mode, model) configurations yield $p \ll 0.001$ in two-sample t-tests, confirming statistically significant separation of correct vs. incorrect similarity distributions.

**Cosine similarity statistics (BGE, no_thinking):**

| Dataset | $\bar{s}_\text{correct}$ | $\bar{s}_\text{incorrect}$ | $\Delta\bar{s}$ | p-value |
|---|---|---|---|---|
| SciQ | 0.921 | 0.693 | **+0.228** | $3.96\times10^{-84}$ |
| SimpleQA | 0.925 | 0.554 | **+0.371** | $5.94\times10^{-10}$ |
| NQ | 0.742 | 0.516 | +0.226 | $1.23\times10^{-11}$ |
| TruthfulQA | 0.791 | 0.678 | +0.113 | $1.45\times10^{-22}$ |

---

## 5. Results

### 5.1 Main Results

**AUC across all 24 configurations (best per row in bold):**

| Dataset | Type | no_thinking BGE | no_thinking MiniLM | no_thinking E5 | thinking BGE | thinking MiniLM | thinking E5 |
|---|---|---|---|---|---|---|---|
| SciQ | Short | 0.940 | **0.954** | 0.933 | 0.902 | 0.934 | 0.890 |
| SimpleQA | Short | **0.976** | 0.933 | 0.931 | 0.934 | 0.879 | 0.886 |
| NQ | Long | 0.905 | 0.894 | **0.934** | 0.880 | 0.869 | **0.899** |
| TruthfulQA | Long | 0.743 | 0.723 | 0.784 | 0.801 | 0.763 | **0.813** |

![ROC Curves no_thinking](results/figures/roc_no_thinking.pdf)
*Figure 1: ROC curves (BGE-base) for all four datasets, no_thinking mode.*

![ROC Curves thinking](results/figures/roc_thinking.pdf)
*Figure 2: ROC curves (BGE-base) for all four datasets, thinking mode.*

**Key findings:**
- Short-form AUC is uniformly high (0.879–0.976): cosine similarity is a reliable correctness proxy for concise factual answers
- TruthfulQA exhibits markedly lower AUC (0.72–0.81): plausible-but-wrong answers cannot be separated reliably
- E5-base-v2 is the most robust model: ranks first or second in 5 of 8 (dataset, mode) combinations

### 5.2 Short-form vs. Long-form QA

![Short vs Long](results/figures/short_vs_long_bge-base.pdf)
*Figure 3: AUC comparison short-form vs. long-form QA tasks.*

The similarity gap between correct and incorrect distributions is substantially wider for short-form tasks:

| Task | $\Delta\bar{s}$ | AUC Range |
|---|---|---|
| Short-form | 0.228 – 0.371 | 0.879 – 0.976 |
| Long-form | 0.113 – 0.226 | 0.723 – 0.934 |

**Root cause:** Long predictions embed as dense semantic mixtures. For TruthfulQA, many incorrect predictions are near-correct paraphrases of common misconceptions whose surface similarity to ground truth is deceptively high (mean FP similarity = **0.865** under BGE).

![Distributions](results/figures/distributions_no_thinking_bge-base.pdf)
*Figure 4: Cosine similarity distributions (correct vs. incorrect) per dataset, BGE-base, no_thinking.*

### 5.3 Embedding Model Comparison

![AUC Heatmap no_thinking](results/figures/auc_heatmap_no_thinking.pdf)
*Figure 5: AUC heat-map, no_thinking mode.*

![AUC Heatmap thinking](results/figures/auc_heatmap_thinking.pdf)
*Figure 6: AUC heat-map, thinking mode.*

| Model | Strengths | Weaknesses |
|---|---|---|
| **BGE-base-en-v1.5** | Best on SimpleQA (0.976), balanced overall | Slightly lower on TruthfulQA vs E5 |
| **all-MiniLM-L6-v2** | Best on SciQ (0.954), fast inference | Worst on TruthfulQA (0.723), high variance |
| **E5-base-v2** | Best cross-task average, strong on long-form | Slightly below BGE on SimpleQA |

**Recommendation:** E5-base-v2 for cross-task deployment; BGE for short-form; MiniLM only for resource-constrained short-form settings.

### 5.4 Chain-of-Thought Ablation

![Thinking Ablation](results/figures/thinking_ablation_bge-base.pdf)
*Figure 7: ΔAUC (thinking − no_thinking) for BGE-base.*

**Bidirectional effect of CoT reasoning:**

| Dataset | ΔAUC (thinking − no_thinking) | Explanation |
|---|---|---|
| SciQ | **−0.038** | CoT verbosity dilutes short-answer embeddings |
| SimpleQA | **−0.042** | Format mismatch: sentence prediction vs. ultra-short GT |
| NQ | −0.025 | Over-thinking; accuracy drops 9.3% → 4.0% |
| **TruthfulQA** | **+0.029** | CoT suppresses hallucination echoes; better separation |

The TruthfulQA improvement is particularly notable: the mean cosine gap between correct and incorrect predictions *increases* under thinking mode (BGE: 0.113 → 0.130, +0.017), despite lower overall accuracy (35.0% → 28.2%).

---

## 6. Failure Analysis

We extract the top-50 FP (false positives: high similarity but incorrect) and top-50 FN (false negatives: low similarity but correct) per configuration using the Youden-optimal threshold.

### Failure Categories

| Dataset | FP | FN | Lexical Variation | Semantic Ambiguity | Long-form Reasoning |
|---|---|---|---|---|---|
| SciQ | 29 | 40 | 14 | 16 | 0 |
| SimpleQA | 50 | 0 | 22 | 11 | 0 |
| NQ | 50 | 1 | 0 | 0 | 51 |
| TruthfulQA | 50 | 50 | 0 | 0 | 100 |

![Failure Causes no_thinking](results/figures/failure_causes_no_thinking.pdf)
*Figure 8: Failure cause distribution, no_thinking mode.*

### Category Descriptions

**1. Lexical Variation** (Short-form: SciQ, SimpleQA)  
Prediction contains correct factual content but different surface form.  
*Example:* Prediction "H₂O" vs. ground truth "water"; prediction "the United States" vs. "America"

**2. Semantic Ambiguity** (Short-form: SciQ, SimpleQA)  
Question admits multiple plausible answers occupying overlapping embedding regions.  
*Example:* TruthfulQA Q: "What is the most popular sport in America?" — both "American football" and "baseball" achieve high similarity to the GT "American football"

**3. Long-form Reasoning Complexity** (NQ, TruthfulQA)  
Prediction embeds as a mixture of correct and incorrect content, or correctly cites a misconception while refuting it.  
*Example:* "Fortune cookies did not originate in China; they were invented in Japan/USA" — cosine similarity with GT "China" is inflated by co-occurrence of the entity

**4. Format Mismatch (CoT-specific)**  
When GT is ultra-short (e.g., "2002") and prediction is a full sentence ("*Catch Me If You Can* was made in 2002"), contains-match correctly labels it as correct but the sentence embedding is far from the GT embedding. This explains much of the AUC drop from no_thinking to thinking on SimpleQA.

---

## 7. Improvement Methods

Motivated by the failure analysis, we propose and evaluate four strategies.

### Method A: Post-hoc Poincaré Projection

Apply the exponential map to project Euclidean embeddings onto the Poincaré ball:
$$\mathbf{y} = \text{expmap}_0(\alpha \mathbf{x}), \quad \alpha = 0.05$$

**Result: ΔAUC ≈ 0 across all 24 configurations.**  
AUC is a rank-invariant metric — a monotone bijection from Euclidean to hyperbolic distance cannot change pair rankings, hence cannot change AUC. This important **negative result** confirms that post-hoc geometric re-projection is insufficient; hyperbolic gains require training in hyperbolic space from the start.

### Method B: Sentence-level MaxSim

Split prediction into sentences, encode each independently, take the maximum similarity:
$$s_\text{MaxSim} = \max_j \cos(\mathbf{e}_{p_j}, \mathbf{e}_g)$$

| Dataset | Mode | ΔAUC (BGE) | ΔAUC (MiniLM) | ΔAUC (E5) |
|---|---|---|---|---|
| SciQ | thinking | **+0.017** | +0.016 | +0.016 |
| SimpleQA | thinking | **+0.047** | +0.039 | +0.043 |
| NQ | no_thinking | −0.023 | −0.032 | −0.041 |
| TruthfulQA | thinking | −0.057 | −0.061 | −0.087 |

MaxSim helps for CoT mode on short-form tasks by isolating the final answer sentence. It *harms* long-form tasks because long predictions often contain sentences mentioning GT entities in a refutational context.

### Method C: HELM Trained Hyperbolic Projection

Train a lightweight MLP projection head mapping Euclidean embeddings into the Poincaré ball:
- Architecture: `Linear(D, 256) → ReLU → Linear(256, 256)` + `expmap0`
- Loss: Binary cross-entropy on correct/incorrect pairs
- Optimiser: RiemannianAdam (lr=1e-3), 30 epochs, 80/20 split

**Validation AUCs:** BGE 0.934 | MiniLM 0.937 | E5 **0.952**

| Dataset | Mode | ΔAUC (BGE) | ΔAUC (MiniLM) | ΔAUC (E5) |
|---|---|---|---|---|
| SciQ | no_thinking | +0.012 | +0.011 | +0.002 |
| SimpleQA | no_thinking | +0.038 | +0.027 | +0.069 |
| NQ | no_thinking | +0.032 | +0.070 | −0.056* |
| **TruthfulQA** | **thinking** | **+0.098** | **+0.096** | **+0.077** |

*NQ has only n_pos=4 in the test split; results are noisy.

HELM provides the **largest and most consistent gains** on TruthfulQA (BGE +9.8%, MiniLM +9.6%, E5 +7.7%), confirming that trained hyperbolic projection captures the contrastive hierarchical semantics of adversarial answers.

### Method D: Span-MaxSim

Identify "answer indicator" spans using regex patterns ("the answer is", "therefore", "thus", etc.) and embed only the extracted final-answer sentence.

| Dataset | Mode | ΔAUC (BGE) | ΔAUC (MiniLM) | ΔAUC (E5) |
|---|---|---|---|---|
| NQ | thinking | **+0.016** | +0.012 | +0.007 |
| TruthfulQA | thinking | −0.093 | −0.082 | −0.097 |

Small positive gains on NQ/thinking; significant harm on TruthfulQA/thinking because multi-answer GT requires full-prediction coverage for accurate MaxSim.

### Cross-Method Comparison

| Method | Best Gain | Worst Case | Verdict |
|---|---|---|---|
| Poincaré (post-hoc) | ≈0 | ≈0 | Ineffective (rank invariance) |
| MaxSim (sentence) | +4.7% (SimpleQA/thinking) | −8.7% (TruthfulQA/thinking) | Conditionally useful |
| **HELM (trained)** | **+12.3% (TruthfulQA/thinking/BGE)** | −18.9%* (NQ/thinking/MiniLM, n=4) | **Recommended for long-form** |
| Span-MaxSim | +1.6% (NQ/thinking) | −9.7% (TruthfulQA/thinking) | Limited, task-specific |

![Delta AUC Boxplot](results/figures/delta_auc_boxplot.pdf)
*Figure 9: ΔAUC distribution for each improvement method across all 24 configurations.*

---

## 8. Conclusion

We systematically investigated whether cosine similarity in embedding space is a reliable proxy for LLM prediction correctness across four QA benchmarks. Key takeaways:

1. **Short-form QA: yes, reliably.** AUC reaches 0.976 (SimpleQA/BGE), all similarity distributions are highly separated (p < 10⁻⁸).

2. **Long-form QA: only partially.** TruthfulQA AUC peaks at 0.813, reflecting the inability of Euclidean cosine to separate contrastive adversarial answers.

3. **CoT reasoning has opposing effects.** It hurts short-form discriminability (format mismatch) but helps TruthfulQA (suppresses hallucination echoes).

4. **Trained hyperbolic projection is the most effective improvement.** HELM provides consistent +7%–+12% AUC gains on TruthfulQA without dataset-specific engineering.

**Future directions:**
- Joint training of HELM projection with LLM fine-tuning objective
- Extension to RAG settings where faithfulness and factual correctness must both be evaluated
- Investigation of answer span extraction as a preprocessing step before embedding

---

## 9. References

1. Zhang, T. et al. BERTScore: Evaluating Text Generation with BERT. arXiv:1904.09675 (2019).
2. Reimers, N. & Gurevych, I. Sentence-BERT. EMNLP 2019.
3. Wang, L. et al. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533 (2022).
4. Xiao, S. et al. C-Pack: Packaged Resources to Advance General Chinese Embedding. arXiv:2309.07597 (2023).
5. Muennighoff, N. et al. MTEB: Massive Text Embedding Benchmark. arXiv:2210.07316 (2022).
6. Wei, J. et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
7. Kojima, T. et al. Large Language Models are Zero-Shot Reasoners. NeurIPS 2022.
8. Nickel, M. & Kiela, D. Poincaré Embeddings for Learning Hierarchical Representations. NeurIPS 2017.
9. He, N. et al. HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts. arXiv:2505.24722 (2025).
10. Welbl, J. et al. Crowdsourcing Multiple Choice Science Questions. arXiv:1707.06209 (2017).
11. Kwiatkowski, T. et al. Natural Questions. TACL 2019.
12. Lin, S. et al. TruthfulQA: Measuring How Models Mimic Human Falsehoods. arXiv:2109.07958 (2021).
13. Qwen Team. Qwen3 Technical Report. arXiv 2025.
