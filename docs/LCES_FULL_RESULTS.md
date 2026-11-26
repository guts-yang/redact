# LCES full reproduction — R-TOFU `forget10` canonical split

**Scope**: 400 `forget10/task_id=1`, 400 protocol-retain, 100 real_authors,
117 world_facts; one seed; `sangyon/LRM-target`; fp32; 1024 max new tokens.
**Baseline**: R2MU `forget10` checkpoint.
**Verdict**: negative for the utility-preserving tradeoff claim.

## Final table

| System | AFE_judge ↑ | CFE_judge ↑ | MU ↑ | LEAK ↓ | HALLUC | COLLAPSE | REFUSE | Retain ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target | 0.0125 | 0.0004 | **0.7656** | 0.9875 | 0.0000 | 0.0025 | 0.0100 | **0.9975** |
| R2MU | **1.0000** | **0.9990** | **0.5269** | **0.0000** | 0.0025 | 0.9950 | 0.0025 | 0.4472 |
| LCES 5-epoch | 0.7750 | 0.7615 | 0.4170 | 0.2250 | 0.4850 | 0.0500 | 0.2400 | 0.4497 |

## Conclusion

LCES reduces exact-entity leakage from 98.75% to 22.50%, but it is dominated by
R2MU on AFE and MU: R2MU has 0.00% leakage and MU 0.5269, while LCES has 22.50%
leakage and MU 0.4170. LCES only wins on collapse (5.0% vs 99.5%), paying with
48.5% hallucination and 24.0% refusal. More epochs improve forgetting but worsen
utility (1-epoch MU 0.4740 → 5-epoch 0.4170).

## Interpretation

Unlearning in reasoning LLMs is a "know but don't say" problem. LCES implements
token-level entity suppression; because entity tokens are shared with retained
knowledge and CoT can route around the ban, suppression transfers leakage into
hallucination/refusal and damages utility. R2MU's representation-level
intervention is conditional on the forgotten neighborhood and preserves more
retain/aux utility. The next direction should be concept-level, utility-
constrained unlearning, not token-level banning.

## Provenance

- full run: `/usr/local/workspace/tracks/lrm-unlearning/run-20261125-full/`
- LCES metrics: `run-20261125-full/results/m3_lces_full_eval.json`
- target metrics: `run-20261125-full/results/m1_target-fp32_eval.json`
- R2MU metrics: `run-20261125-full/results/m1_r2mu-forget10-fp32_eval.json`
- claim: `run-20261125-full/CLAIMS_FROM_RESULTS_FULL.md`

This is a provisional result; the project reviewer is Hy4 and no cross-family
acceptance gate was available.
