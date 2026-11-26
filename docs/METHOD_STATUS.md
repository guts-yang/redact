# LRM method status — 2026-11-26

Scope: LRM unlearning track only (`R-TOFU`, `redact`; `awesome-lrm-unlearning`
is reference-only). No model, split, metric or result from the LLM track is used.

## Eliminated — do not advance

| Method | Evidence | Reason |
|---|---|---|
| LCES — Leak-Calibrated Entity Suppression (5-epoch full) | `docs/LCES_FULL_RESULTS.md`; `run-20261125-full/results/m3_lces_full_eval.json` | Negative result: exact leakage 98.75% -> 22.50%, but MU 0.4170 < R2MU 0.5269; dominated by R2MU on AFE and MU. `claim_verdict: no`. More epochs improve forgetting but worsen utility. |
| B-LCES — dual-boundary alias-lattice margin | `idea-stage/IDEA_REPORT.md` §5.3/§6; `run-20261125/` | Reviewers reduced it to a narrow falsification study; no positive pilot; superseded by the LCES full negative result. |
| Consistency-aware entity editing (reproduction) | `idea-stage/IDEA_REPORT.md` §7 | Reproduction only, no new contribution. |
| REDaCT inference-time detector | `idea-stage/IDEA_REPORT.md` §7 | Other people's method; reference arm only. |
| Two-stage suppress-then-restore CoT | `idea-stage/IDEA_REPORT.md` §7 | Weak mechanism, high cost, overlapping with B-LCES. |
| Margin-only tuning (20 -> 25) | `idea-stage/IDEA_REPORT.md` §7 | Parameter tuning without identification design. |

These are negative / archival findings. Do not rewrite them as positive evidence
and do not spend further compute on them without a new mechanism, protocol or
resource change.

## Active candidate — prepare to continue replication

**UPBD — Utility-Preserving Boundary Distillation** (`run-20261126-upbd`)

- Hypothesis: freezing R2MU's forget10 boundary and distilling target
  retain/aux behavior raises utility without reopening exact leakage.
- Implementation design (how forgetting is realised): `docs/UPBD_DESIGN.md`.
- Status: **held-out pilot complete (partial)**; 20 rows per split, seeds 1001
  and 2002, one task_id, one model. See `docs/UPBD_PILOT.md` and
  `run-20261126-upbd/refine-logs/EXPERIMENT_TRACKER.md`.
- Pilot signal: LEAK 0.000 both; MU 0.8418 / 0.8141 vs R2MU 0.5868; retain
  ROUGE-L 0.9108 / 0.9434 vs R2MU 0.4079; collapse 0.000.
- Not yet supported: full 400/400/217 split, multi-seed significance on the
  full split, cross-family acceptance. Hy4 is a project reviewer override and
  remains provisional; no cross-family verdict has acquitted the claim.
- Caveat: 95% refusal on the forget channel is a behavioral exit policy, not
  proof of knowledge erasure. Full split must settle unlearning vs safe refusal.

### Next actions (preparation, not executed here)

1. `run-20261126-upbd/refine-logs/EXPERIMENT_TRACKER.md` U6: run full
   400/400/217 with seeds 1001 and 2002 under the frozen protocol.
2. Re-run `/experiment-audit` and `/result-to-claim` on the full-split result.
3. Obtain a genuine cross-family reviewer verdict; until then keep the track
   `BLOCKED` and the claim `partial`.

## Backup candidates (not active)

- CCPD — Channel-Conditioned Pareto Distillation (`run-20261126-upbd/idea-stage/IDEA_CANDIDATES.md`).
- Causal concept-direction ablation — higher risk, future work.

## Gate status

Track gate remains `BLOCKED`; `dry_run = FAILED`. This document records method
status only. It is not an accepted claim, not an experiment result, and not a
reviewer verdict. No credential value is stored here.
