# UPBD — how forgetting is implemented (code-grounded design note)

**Run**: `lrm-unlearning-20261126-upbd`
**Code**: `run-20261126-upbd/experiments/train_upbd.py`
**Companion**: `docs/UPBD_PILOT.md`, `docs/METHOD_STATUS.md`

## Core answer

UPBD does **not** introduce a new forgetting loss. Forgetting is inherited from
a frozen R2MU `forget10` checkpoint; UPBD then repairs the utility side with
channel-conditioned supervised fine-tuning:

- the **forget boundary** is already encoded in the starting weights;
- forget prompts are taught a fixed non-disclosing exit trajectory;
- retain / aux prompts are taught the target model's original behaviour.

So the design is **boundary inheritance + behaviour repair**, not a new
unlearning objective, not retraining R2MU, and not a token-logit ban like LCES.
The resulting behaviour is **behavioural unlearning** (the model refuses on the
forget channel), not proof of parameter-level knowledge erasure.

## 1. Boundary inheritance (where forgetting comes from)

`train_upbd.py` sets

```
MODEL = .../run-20261125/checkpoints/r2mu-forget10-compat
TOKENIZER = .../models/sangyon/LRM-target
```

The student starts from the R2MU forget10 checkpoint. That checkpoint is the
frozen "where to forget" asset produced by the earlier R2MU reproduction; UPBD
does not recompute it, does not add a new margin/KL/gradient-ascent term, and
does not modify the underlying forget boundary directly. The LoRA adapter is
trained on top of it and later merged for evaluation (`merge_and_unload`).

## 2. Parameter surface: LoRA only

```
LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
           task_type="CAUSAL_LM",
           target_modules=[q_proj, k_proj, v_proj, o_proj,
                           gate_proj, up_proj, down_proj])
```

Only these low-rank adapter matrices are updated; the base checkpoint stays
frozen. `merge_and_unload()` produces the evaluated merged model.

## 3. Channel construction (what is actually trained)

Data are read from the locked target/R2MU result CSVs:

| Channel | Source CSV (`run-20261125-full/results/`) | Question | Training target |
|---|---|---|---|
| forget | `m1_r2mu-forget10-fp32_forget10.csv` | `question` | fixed `SAFE_EXIT = "I don't know.\n</think>\n\nI don't know."` |
| retain | `m1_target-fp32_forget10_retain.csv` | `question` | `behavior_target(row)`: `model_cot + "\n</think>\n\n" + response` (gold fallback) |
| aux | `m1_target-fp32_real_authors.csv`, `m1_target-fp32_world_facts.csv` | `question` | same `behavior_target(row)` |

`--holdout N` skips the first N rows of each split; those rows are never used
for training, so the pilot measures held-out behaviour rather than memorised
training rows. Items are shuffled before training.

**Important implementation detail**: the `channel` label exists in the item
tuple but is not used to compute a separate loss or gate. The channel is
defined implicitly by which CSV the prompt came from and which target string
is attached to it. There is no explicit channel token, no router, and no
per-channel weighting in the code.

## 4. Supervision form

`encode()` concatenates prompt ids and target ids, appends EOS, and masks the
prompt: `labels = [-100] * len(prompt) + target_ids`. `collate()` pads to the
batch width with attention masks and `-100` labels. The model is trained with
the ordinary causal-LM objective:

```
out = model(input_ids=ids, attention_mask=attn, labels=labels)
loss = out.loss / grad_accum
loss.backward()
```

There is no custom forgetting loss: no logit ban, no margin, no alias hinge,
no unlearning-specific regulariser, no KL to a teacher. Retain behaviour is
conveyed through the **target text** (`model_cot + </think> + answer`), i.e.
standard SFT on target-generated trajectories.

Training uses AdamW (lr 2e-4, weight decay 0.01), gradient clipping 1.0,
batch 1 with grad-accum 4, 1 epoch, fp32, seeds 1001 / 2002.

## 5. Why this produces forgetting on the forget channel

1. The inherited R2MU checkpoint already suppresses exact leakage on forget
   prompts (`LEAK = 0.000` in the pilot).
2. UPBD's SFT teaches the same prompts to emit a short safe-exit string with
   an explicit `</think>` boundary, so the model learns a **channel-conditioned
   exit policy**: forget prompts -> "I don't know."; retain/aux prompts ->
   target CoT and answer.
3. Because only a small LoRA adapter is trained on top of the frozen boundary,
   the utility side can be repaired without directly re-optimising the forget
   boundary. The pilot shows MU 0.8418 / 0.8141 vs R2MU 0.5868 and retain
   ROUGE-L 0.9108 / 0.9434 vs R2MU 0.4079, with collapse 0.000.
4. The mechanism is behavioural: 95% refusal on the forget channel. It does
   **not** prove that the underlying facts were removed from the weights.

## 6. Differences from the eliminated methods

- **vs LCES**: LCES used token-level entity suppression and failed (MU 0.4170
  vs R2MU 0.5269, leakage 22.50%). UPBD has no token ban; it changes the
  output trajectory on the forget channel.
- **vs R2MU retraining**: R2MU is frozen as the boundary asset; UPBD only
  repairs utility after the fact.
- **vs GUARD**: UPBD does not learn a forgetting trajectory from scratch; it
  post-hoc repairs an existing unlearning checkpoint.

## 7. Limitations and open risks

- 20-row held-out pilot, one task_id, one model; full 400/400/217 split is not
  run. U6 in the experiment tracker is still PENDING.
- The channel is implicit (data-source + target text), not an explicit gating
  mechanism; generalisation to unseen forget phrasings is untested.
- A fixed safe-exit string may be learned as a refusal template rather than
  entity-level forgetting; the full split must settle "unlearning vs safe
  refusal".
- Hy4 is the project reviewer override and remains provisional; no
  cross-family acceptance exists. Track gate stays `BLOCKED`.

## 8. Reproduction pointer

`run-20261126-upbd/experiments/run_upbd.sh` uses the isolated env
`/jizhicfs/chenyliao/lrm-unlearning/envs/r-tofu-ga/bin/python`, offline HF
cache, fp32, and the frozen result CSVs above. See `docs/METHOD_STATUS.md` for
the elimination/active-candidate ledger.

This note records the implementation design only. It is not a claim, not an
accepted reviewer verdict, and contains no credential values.
