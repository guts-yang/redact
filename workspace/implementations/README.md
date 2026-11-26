# IDEA_01..IDEA_10 — Decoupled Core Implementations

This package implements the **core algorithms** from the ten proposals in
`redact/workspace/IDEA_*`. Each method lives in its own module and imports only
`torch`, the standard library, and `base.py`. There is no shared mutable state
between methods.

These are algorithm implementations, not experiment results. They do not create
a claim or advance an ARIS gate.

## Layout

```text
implementations/
  base.py        common MethodOutput, KL/JS, entity-margin helpers
  registry.py    IDEA id -> class, in proposal priority order
  csda.py        IDEA_01 CSDA
  eas.py         IDEA_02 EAS
  rrbu.py        IDEA_03 RRBU
  disco.py       IDEA_04 DISCO
  ortho.py       IDEA_05 ORTHO
  scope.py       IDEA_06 SCOPE
  sequ.py        IDEA_07 SEQU
  cfa.py         IDEA_08 CFA
  ripple.py      IDEA_09 RIPPLE
  bound.py       IDEA_10 BOUND
  hooks/         full CSDA/RRBU/CFA/RIPPLE adapters (step patching,
                 functional_call bilevel, style audit, knowledge graph)
  tests/         CPU synthetic unit tests
```

The four hardest methods also have full hooks under `hooks/`; see
`hooks/README.md` for the data/model integration contract.

## Method interfaces

| Idea | Module | Core entry points | Main tensor shapes |
|---|---|---|---|
| CSDA | `csda.py` | `causal_step_contributions`, `continuous_mask`, `csda_loss`, `CSDA` | hidden `[T,B,H]`, mask `[T]` |
| EAS | `eas.py` | `entanglement_alpha`, `eas_loss`, `EAS` | kappa/depth `[B]`, losses `[B]` |
| RRBU | `rrbu.py` | `inner_relearn_attack`, `rrbu_loss`, `sam_perturbation`, `RRBU` | parameter dict |
| DISCO | `disco.py` | `worst_policy_margin`, `context_free_loss`, `disco_loss`, `DISCO` | logits `[P,B,V]`, bare `[B,V]` |
| ORTHO | `ortho.py` | `estimate_reasoning_subspace`, `orthogonalize_gradient`, `ortho_loss`, `ORTHO` | activations `[N,H]`, subspace `[H,k]` |
| SCOPE | `scope.py` | `scope_loss`, `step_scope_consistency`, `SCOPE` | step logits `[B,T,V]`, mask `[B,T]` |
| SEQU | `sequ.py` | `representation_drift`, `sequ_loss`, `anchor_loss`, `SEQU` | retain repr `[B,H]`, parameter dict |
| CFA | `cfa.py` | `extract_style_features`, `cfa_loss`, `linear_probe_auc`, `side_channel_advantage`, `CFA` | style `[B,D]`, membership `[B]` |
| RIPPLE | `ripple.py` | `semantic_neighbors`, `multi_hop_neighbors`, `pathcut_loss`, `ripple_loss`, `RIPPLE` | embeddings `[N,H]`, graph adjacency |
| BOUND | `bound.py` | `commitment_boundary`, `exit_loss`, `bound_loss`, `censored_interval`, `BOUND` | logits `[T,V]`, budgets |

## Integration notes

- **CSDA / ORTHO / RIPPLE** expect hidden states from a transformer hook. Use the
  last hidden state per step, or a chosen layer; the algorithm itself is
  layer-agnostic.
- **EAS / RRBU / SEQU** accept callables that return per-sample losses from the
  current parameters. This keeps them independent of the HF trainer.
- **DISCO / SCOPE / BOUND** accept logits only; no model class is imported.
- **CFA** has two halves: the training-side GRL loss and the audit-side linear
  probe/AUC. The probe is deliberately model-free.
- **RRBU** uses `torch.autograd.grad` with `create_graph=True` for the unrolled
  inner attack; keep `inner_steps` small (the proposal suggests K <= 5).

## Usage example

```python
from implementations import CSDA, CSDAConfig

method = CSDA(CSDAConfig(tau=0.5, kappa=0.1))
c = method.contributions(hidden_states, leak_loss_fn, retain_mean, retain_cov)
out = method.loss(hidden_states, base_hidden, v_safe, c, retain_kl)
out.loss.backward()
print(out.float_diagnostics())
```

## Tests

```bash
cd /usr/local/workspace/tracks/lrm-unlearning/redact/workspace
PYTHONPATH=. /jizhicfs/chenyliao/ops/venvs/r-tofu-ga/bin/python \
  -m unittest discover -s implementations/tests -v
```

All ten tests use synthetic CPU tensors; they verify shapes, finite losses, and
the key invariants (mask range, orthogonal projection, SCS, censoring interval).

## Scope boundary

The implementations are reference cores. Wiring them to R-TOFU training,
distributed LoRA, and full evaluation is a separate step. No result from this
package is a claim, a reviewer verdict, or an ARIS gate receipt.
