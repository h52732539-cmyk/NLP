# Local dataset files expected by the project

This directory is intentionally excluded from version control except for this note.

Place local JSONL datasets here when they are not fetched from Hugging Face.

Expected files in the default benchmark config:

1. sickr_test.jsonl
2. summeval.jsonl

## JSONL rules

1. One JSON object per line.
2. UTF-8 encoding.
3. All text fields should be plain strings.
4. Human scores should be numeric.

## Example: SICK-R style record

```json
{"id": "sickr-0001", "sentence1": "A child is playing.", "sentence2": "A kid is having fun.", "relatedness_score": 4.7}
```

## Example: SummEval style record

```json
{"id": "summeval-0001", "prompt": "Summarize the article: ...", "candidate": "Generated summary.", "reference": "Reference summary.", "consistency_score": 4.2}
```

If you want the model to generate candidates during the experiment, keep the prompt field populated, set generate_candidate: true in the config, and leave the candidate field empty or omit it.