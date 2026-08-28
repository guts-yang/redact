# Knowledge bank provenance

`knowledge_banks/<split>/claims.json` is the resolved claim list for each forget
split: a flat JSON array of strings whose **position is the claim ID**. Index
stability matters — `config/authors/<split>.json` addresses claims by absolute
index range, so reordering this file silently misattributes claims to authors.

| Split | Claims | Bytes |
|---|---|---|
| `forget01` | 50 | 7,854 |
| `forget05` | 252 | 38,212 |
| `forget10` | 509 | 78,578 |

## What it is used for

It is the rubric the first-leak LLM judge adjudicates against
(`scripts/label_first_leak_rtofu.py`), which is what produces the labels the
leak classifier trains on. It is a **training-time input only** — nothing under
`knowledge_banks/` is read at inference or during evaluation.

## How it was derived

1. `scripts/build_kb.py` sentence-splits the gold R-TOFU forget answers for the
   split and flattens them into one claim per sentence.
2. `scripts/resolve_kb.py` applies `config/kb_resolutions/<split>.json` — a
   hand-curated table resolving pronouns and demonstratives to the author they
   refer to — and appends the per-author birth-fact claims in
   `config/birth_claims/<split>.json`.

Step 2 has **no generator**; the resolution table was produced by hand. That is
why the resolved output ships here rather than being rebuilt. Both input tables
ship as well, so this file remains auditable and re-derivable:

```bash
# 1. Baseline bank: sentence-split the gold forget answers.
#    NB: build_kb writes to <output_dir>/<split>. Do NOT let it default into
#    knowledge_banks/ — that would overwrite the shipped claims.json before
#    resolve_kb has rebuilt it.
python -m scripts.build_kb --split forget10 --output_dir knowledge_banks_baseline

# 2. Resolve ambiguities and append birth-fact claims. Reproduces the shipped
#    knowledge_banks/forget10/claims.json.
python -m scripts.resolve_kb \
    --baseline_kb knowledge_banks_baseline/forget10 \
    --resolutions config/kb_resolutions/forget10.json \
    --birth_claims config/birth_claims/forget10.json \
    --output knowledge_banks/forget10
```

For `forget05` / `forget01`, slice forget10's resolution table first
(`scripts/slice_kb_resolutions.py --max_claim_idx 248` for forget05, `50` for
forget01) and pass the split's own `config/birth_claims/<split>.json`.

Subset relationships: `forget01 == forget10[:50]`, and `forget05` is
`forget10[:248]` plus four of the appended birth claims.

## Disclosure

The claims are derived from the R-TOFU forget answers, which are already public
and which this repository downloads via `scripts/download_rtofu_data.py` (see
`data/rtofu/PROVENANCE.md`). Checked against `data/rtofu/forget10.json`
(`task_id=1`): 381 of 509 claims appear verbatim. Of the remaining 128, 109 are
pronoun-to-name resolutions where the name occurs in the same public record, 6
are birth-fact claims restating text from the public *questions*, and 13 are
typographic variants (curly apostrophes, en-dashes). No claim introduces a
content token absent from the public data — this file exposes nothing beyond
the upstream dataset.
