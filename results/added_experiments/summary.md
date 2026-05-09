# Added Experiment Summary

## TA Baseline Compatibility Predictions

| Dataset | Mode | n | Exact | Contains/Correct | Mean words | Max words |
|---|---:|---:|---:|---:|---:|---:|
| sciq | no_thinking | 13679 | 0.413 | 0.558 | 1.89 | 196 |
| simple_questions_wiki | no_thinking | 49202 | 0.121 | 0.166 | 2.83 | 222 |

## Original TA Pipeline Outputs

| Dataset | n pred | HHEM n | HHEM mean | HHEM >=0.5 | HHEM unique | MPNet cos mean | MPNet cos median | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| nq | 3400 | 0 | - | - | 0 | - | - | prediction only |
| sciq | 13679 | 13679 | 0.5021 | 1.000 | 1 | 0.762 | 0.794 | HHEM, embeddings |
| simple_questions_wiki | 0 | 0 | - | - | 0 | - | - | prediction only |
| truthfulQA | 817 | 817 | 0.5021 | 1.000 | 1 | 0.560 | 0.545 | HHEM, embeddings |

Key note: the original TA pipeline HHEM outputs for completed datasets are constant at 0.5021, so thresholding at 0.5 marks every sample positive and cannot support ROC/AUC analysis.
