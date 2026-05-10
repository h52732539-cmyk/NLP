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
| sciq | 13679 | 13679 | 0.2874 | 0.259 | 13361 | 0.762 | 0.794 | HHEM, embeddings |
| simple_questions_wiki | 0 | 0 | - | - | 0 | - | - | prediction only |
| truthfulQA | 817 | 817 | 0.1466 | 0.104 | 817 | 0.560 | 0.545 | HHEM, embeddings |

Key note: corrected HHEM outputs are non-constant for completed datasets, so thresholded HHEM labels can now be used for downstream analysis.
