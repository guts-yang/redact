# EXPERIMENT RESULTS — UPBD held-out pilot

**Run**: `lrm-unlearning-20261126-upbd`
**Training**: R2MU boundary + LoRA; 1 epoch; 380 forget + 380 retain + 177 aux
**Held-out eval**: first 20 rows of each split, never used in training
**Baselines**: target / R2MU / LCES 20-row from `run-20261125`
**Generation**: fp32, 1024 max new tokens
**Judge**: `deepseek-flash`

## Final table (same 20 held-out rows)

| System | AFE_judge ↑ | CFE_judge ↑ | MU ↑ | LEAK ↓ | HALLUC | REFUSE | COLLAPSE | Retain ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target | 0.0000 | 0.0000 | 0.8135 | 1.000 | 0.000 | 0.000 | 0.000 | **1.0000** |
| R2MU | 1.0000 | 1.0000 | 0.5868 | 0.000 | 0.000 | 0.000 | 1.000 | 0.4079 |
| LCES 1-epoch | 0.5500 | 0.4700 | 0.4740 | 0.450 | 0.500 | 0.050 | 0.000 | 0.3631 |
| **UPBD seed 1001** | **1.0000** | **1.0000** | **0.8418** | **0.000** | 0.050 | 0.950 | 0.000 | 0.9108 |
| **UPBD seed 2002** | **1.0000** | **1.0000** | **0.8141** | **0.000** | 0.050 | 0.950 | 0.000 | 0.9434 |
| **UPBD mean** | **1.0000** | **1.0000** | **0.8280** | **0.000** | 0.050 | 0.950 | 0.000 | 0.9271 |

## Success criteria

| Criterion | Threshold | UPBD | Result |
|---|---|---|---|
| LEAK | ≤ 0.05 | 0.000 | PASS |
| MU | ≥ R2MU + 0.05 | 0.8418 vs 0.5868 | PASS (+0.2550) |
| Retain ROUGE-L | ≥ R2MU − 0.02 | 0.9108 vs 0.4079 | PASS (+0.5029) |
| Hidden collapse | none | 0.000 | PASS |

## Conclusion

**UPBD is a stable positive held-out pilot.** Across seeds 1001 and 2002 it
keeps zero exact leakage and raises MU to **0.8418 / 0.8141** (mean 0.8280),
while retain ROUGE-L rises to **0.9108 / 0.9434** (mean 0.9271). It is a Pareto
improvement over R2MU on the 20-row held-out set and is stable across two seeds.

Mechanism: the model learns a **channel-conditioned exit policy**:
- forget channel -> safe exit ("I don't know."), 95% refusal / 5% hallucination
- retain/aux channel -> target model behavior, retaining high utility

This is **behavioral unlearning**, not knowledge erasure. It is exactly the
"leakage channel × utility" framing: the model keeps its reasoning behavior on
the utility channel and learns a non-disclosing trajectory on the forget
channel.

## Caveats

- 20-row held-out pilot, two seeds, one task_id, one model.
- 95% refusal on forget is the mechanism; whether this is "unlearning" or
  "safe refusal" is a definitional question that a full-split run must settle.
- No cross-family acceptance gate; Hy4 is the project reviewer override.
- Full 400/400/217 evaluation is not yet run.
