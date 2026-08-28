# R-TOFU JSONL files

Source: https://github.com/ssangyeon/R-TOFU

Commit SHA: `df533a5c08dfa7fcd314d6c8714481fec2e24529` (2025-06-04)

Fetched from: `https://raw.githubusercontent.com/ssangyeon/R-TOFU/<SHA>/data/tofu/<name>.json`

These are JSONL files (one JSON object per line) despite the `.json` extension.

## Why we vendor these

The HuggingFace dataset `sangyon/R-TOFU` drops the `task_id` field that the github JSONLs carry. Without `task_id`, we can't reconstruct the paper's protocol retain splits.

## Schema

- `forget01.json`, `forget05.json`, `forget10.json`: `{task_id: str, question: str, answer: str, cot: str}`
  - forget01: 400 rows, task_ids '1'-'10' (40 rows each)
  - forget05: 2000 rows, task_ids '1'-'10' (200 rows each)
  - forget10: 3600 rows, task_ids '1'-'9' (400 rows each)
- `retain10.json`, `retain50.json`, `retain90.json`: `{question: str, answer: str, cot: str}` (no `task_id`)
  - retain10: 400 rows
  - retain50: 2000 rows
  - retain90: 3600 rows
- `real_authors_perturbed.json`, `world_facts_perturbed.json`: `{question: str, answer: str, perturbed_answer: list[str], original_answer: str, cot: str}`
  - real_authors_perturbed: 100 rows
  - world_facts_perturbed: 117 rows
  - `answer` is the short ground-truth string (e.g. "William Shakespeare").
  - `original_answer` is the elaborated sentence (e.g. "The play 'Romeo and Juliet' was written by William Shakespeare.") — this is what R-TOFU's metric code uses as the gold for cosine similarity and entailment. Our loader renames it to `answer` at load time so downstream `gen_answer_key` plumbing works unchanged; the short form is discarded.
  - `perturbed_answer` (list of wrong options) and `cot` (always empty for aux) are not used by our pipeline.

## Retain naming convention

`retain(100 - n_task_ids × forget_pct)` = rows never used as forget in the forget`XX` series:
- retain90 = full − forget01[all 10 task_ids] (10 × 1% = 10% excluded)
- retain50 = full − forget05[all 10 task_ids] (10 × 5% = 50% excluded)
- retain10 = full − forget10[all 9 task_ids] (9 × 10% ≈ 90% excluded)

Protocol retain for forget`XX` task_id=K: `retain(100 - n × XX) ∪ forgetXX[task_id ≠ K]` = `full − forgetXX[task_id=K]`.

## Refreshing

If the upstream repo updates, replace the SHA above and re-run:

```bash
SHA=<new_sha>
for f in forget01 forget05 forget10 retain10 retain50 retain90 \
         real_authors_perturbed world_facts_perturbed; do
    curl -sS -o data/rtofu/${f}.json \
        https://raw.githubusercontent.com/ssangyeon/R-TOFU/${SHA}/data/tofu/${f}.json
done
```
