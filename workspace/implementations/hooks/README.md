# Full Method Hooks

These adapters connect the core implementations to a real HF/PEFT training
lifecycle. They are the missing layer between `implementations/csda.py` etc. and
the training loop. `R-TOFU/workspace/code/diagnostic_train_hooked.py` wires
these hooks for all ten methods; the older `diagnostic_train.py` remains as the
one-step proxy harness.

## CSDA — `csda_hook.py`

- `split_cot_steps` and `encode_step_spans` provide explicit `r_t` boundaries.
- `collect_step_hidden` returns `[S, H]` step representations from a chosen
  decoder layer.
- `collect_retain_stats` estimates `mean` and a diagonal covariance from retain
  CoT steps.
- `contributions` performs real layer patching: `patch_layer` replaces the
  hidden states of step `t` with a retain sample and re-runs the model.
- `loss` computes the CSDA objective with gradient-carrying step
  representations; attribution contributions are detached.

Cost: `O(S)` extra forwards per forget sample.

## RRBU — `rrbu_hook.py`

- `attack_eval_split` separates the attack subset from held-out eval rows.
- `RRBUHook.attack` runs `inner_relearn_attack` through
  `torch.func.functional_call`, so the inner K steps are differentiable.
- `outer_loss` evaluates the post-attack forget loss, retain loss, and
  parameter proximity.
- Pass only trainable LoRA parameter names; `theta0` is captured at hook init.

Cost: K unrolled inner steps + outer forward/backward.

## CFA — `cfa_hook.py`

- `response_features` extracts response styles (length, `</think>` position,
  refusal/counterfactual markers).
- `differentiable_features` provides a hidden-state style proxy for the
  training path.
- `training_loss` applies the GRL adversary term.
- `update_adversary` trains the adversary on detached style features.
- `audit_responses` / `audit_features` return `(AUC, SCA)`.
- `CounterfactualSampler` provides diverse safe counterfactual templates.

Note: training-time styles are hidden proxies; audit-time styles are real
response features. Do not conflate them.

## RIPPLE — `ripple_hook.py`

- `KnowledgeGraph.from_claims` builds an entity co-occurrence graph from
  `knowledge_banks/forget{01,05,10}/claims.json`.
- `neighbors(entity, hops)` provides the multi-hop set.
- `neighborhood` can use semantic embeddings instead of graph hops.
- `neighbor_kl` computes per-neighbor `KL(p_theta || p_ref)`.
- `pathcut` computes the CoT representation path-cut term.
- `loss` combines forget, neighbor KL, and pathcut.
- `metrics` returns Ripple and Recon.

Note: graph construction is a heuristic co-occurrence graph, not a learned KB
embedding. Replace `KnowledgeGraph.from_claims` with a real graph backend if one
becomes available.

## EAS — `eas_hook.py`

- `EASFeatureBuilder` computes claim-level `kappa` and CoT-depth `depth`.
- `EASHook.loss` applies the sample-level `alpha` schedule and budget penalty.
- `budget_curve` reports alpha/cost feasibility over budgets.

## DISCO — `disco_hook.py`

- `DISCOHook.policy_batches` builds DefaultCoT / LessThink / ZeroThink /
  SummaryThink / bare-prompt batches.
- `policy_logits` returns `[P,B,V]` policy logits + bare logits.
- `loss` applies the worst-policy margin and context-free suppression.

## ORTHO — `ortho_hook.py`

- `ORTHOSubspaceBank.collect` estimates per-layer latent reasoning subspaces.
- `ORTHOHook.loss` penalizes `||U^T(phi-phi0)||`.
- `project_gradient` / `project_named_gradients` project parameter gradients
  away from the reasoning subspace.

## SCOPE — `scope_hook.py`

- `EquivalenceClassBuilder` constructs paraphrase variants.
- `step_logits` converts token logits into `[B,T,V]` step logits.
- `loss` applies in-scope suppression + out-of-scope KL; `scs` returns the
  step-scope consistency metric.

## SEQU — `sequ_hook.py`

- `SequentialRequestStream` splits forget rows into 5 request batches.
- `loss` uses the drift-adaptive retain anchor and proximal term.
- `maybe_repair` triggers a CoT step anchor when drift exceeds `delta`.

## BOUND — `bound_hook.py`

- `boundaries` computes commitment boundaries per budget.
- `exit_losses` builds per-budget exit losses.
- `loss` takes the worst-budget objective; `evaluate` returns the censored
  leakage interval and success flag.

## Method-specific eval hooks (`evals/`)

- `csda_eval.py`: step-wise ROUGE-L/cosine + intervention sparsity.
- `rrbu_eval.py`: `RR(K)` curve + recovery rate.
- `cfa_eval.py`: linear/MLP/n-gram SCA auditor.
- `ripple_eval.py`: RippleBench MCQ + multi-hop reconstruction.

## Tests

```bash
cd /usr/local/workspace/tracks/lrm-unlearning/redact/workspace
PYTHONPATH=. /jizhicfs/chenyliao/ops/venvs/r-tofu-ga/bin/python \
  -m unittest discover -s implementations/hooks/tests -v
```

## Integration checklist

1. `diagnostic_train_hooked.py` already wires all ten hooks, a multi-step
   loop, grad accumulation, adapter resume, and an optional RIPPLE reference
   model.
2. Keep the one-step `diagnostic_train.py` for proxy comparisons only.
3. Cache CSDA retain statistics; attribution contributions are still
   recomputed per training sample and are the main CSDA cost.
4. Keep attack/eval splits fixed across RRBU runs.
5. Run CFA adversary updates on a separate optimizer (the hooked trainer does).
6. Build the RIPPLE graph once per split, not per batch (the hooked trainer
   does).
7. Re-run the full-data budget after the hooks are active: the overhead is
   higher than the one-step proxy measurements.
8. Exact optimizer/scheduler resume is not yet implemented; `--resume` restores
   adapter weights only.
