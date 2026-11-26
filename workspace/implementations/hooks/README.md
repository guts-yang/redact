# Full Method Hooks

These adapters connect the core implementations to a real HF/PEFT training
lifecycle. They are the missing layer between `implementations/csda.py` etc. and
the training loop. `R-TOFU/workspace/code/diagnostic_train_hooked.py` already
calls these hooks for CSDA/RRBU/CFA/RIPPLE; the older `diagnostic_train.py`
remains as the one-step proxy harness.

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

## Tests

```bash
cd /usr/local/workspace/tracks/lrm-unlearning/redact/workspace
PYTHONPATH=. /jizhicfs/chenyliao/ops/venvs/r-tofu-ga/bin/python \
  -m unittest discover -s implementations/hooks/tests -v
```

## Integration checklist

1. `diagnostic_train_hooked.py` already wires the four hooks, a multi-step
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
