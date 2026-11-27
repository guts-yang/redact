#!/usr/bin/env python3
"""Mock-GPU smoke: exercise runtime.runner.run() end-to-end WITHOUT HF/GPU.

We monkeypatch runtime.model + runtime.data with pure-torch stand-ins so the
training loop (sampling -> forward -> compute_loss -> backward -> optimizer ->
save -> summary) runs on CPU with synthetic tensors. This validates the RUNNER
CONTRACT (Context fields, grad accumulation, adapter/summary writing) that a
pure compute_loss selftest does not cover. It does NOT validate HF/PEFT itself.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import data as DATA
from runtime import model as M
from runtime import paths, runner


class TinyModel(torch.nn.Module):
    """Stand-in for a PEFT causal LM: returns logits/hidden and has save_pretrained."""
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 32)
        self.lora = torch.nn.Parameter(torch.randn(2, 2))  # a trainable param

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
        b, l = input_ids.shape
        h = self.lin(torch.randn(b, l, 16))
        logits = torch.randn(b, l, 64, requires_grad=True)
        hidden = h if output_hidden_states else None
        ns = type("O", (), {})()
        ns.logits = logits
        ns.hidden_states = [h] if hidden is not None else None
        return ns

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.bin").write_bytes(b"mock")


class TinyTok:
    pad_token_id = 0
    eos_token_id = 0
    def __call__(self, text, **kw):
        n = min(len(text.split()) + 2, 16)
        return {"input_ids": list(range(1, n + 1))}
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def main():
    tmp = tempfile.mkdtemp(prefix="unl1127_mock_")
    paths.RESULTS = Path(tmp) / "results"
    paths.LOGS = Path(tmp) / "logs"
    # redirect data sampling to synthetic rows so no files are read
    DATA.sample_rows = lambda path, n, seed, task_id=None: [
        {"question": "Who wrote X?", "answer": "A Fictional Person", "cot": "step one.", "task_id": "1"}
        for _ in range(n)]
    DATA.refusals = lambda: ["I don't know."]
    M.load_model = lambda attn_implementation="sdpa": (TinyTok(), TinyModel())
    M.add_lora = lambda model, r=8, alpha=32: model
    M.load_unlearned_reference = lambda: None
    M.collate = lambda tok, examples, max_len: (
        torch.randint(0, 64, (len(examples), max_len)),
        torch.randint(0, 64, (len(examples), max_len)),
        torch.ones(len(examples), max_len, dtype=torch.long))
    M.encode_example = lambda tok, prompt, target, max_len: (
        list(range(1, max_len + 1)), list(range(1, max_len + 1)), [1] * max_len)
    M.forward = lambda model, ids, labels, mask, output_hidden_states=True: (
        model(ids, mask, output_hidden_states).logits,
        model(ids, mask, output_hidden_states).hidden_states[0],
        torch.rand(ids.shape[0], requires_grad=True))
    DATA.entity_token_ids = lambda tok, rows: torch.tensor(
        [1] * len(rows), dtype=torch.long)

    # use the real budget method's compute_loss via the runner
    from methods.budget import train as budget_train
    cfg = runner.RunConfig(method="budget", run_id="mock_smoke", steps=2,
                           n_forget=2, n_retain=2, max_len=16, lr=1e-3,
                           seed=1001)
    summary = runner.run(cfg, budget_train.compute_loss, budget_train.setup)
    assert summary["steps"] == 2
    assert "gates" in summary["meta"]["extras_keys"], summary["meta"]
    assert summary["meta"]["n_trainable_params"] > 0, summary["meta"]
    assert len(summary["losses"]) == 2
    assert all(isinstance(x, float) for x in summary["losses"])
    out = paths.RESULTS / "mock_smoke"
    assert (out / "train_summary.json").exists()
    assert (out / "adapter" / "adapter_model.bin").exists()
    print("MOCK-GPU RUNNER SMOKE PASS")
    print(f"  summary: steps={summary['steps']} losses={summary['losses']} "
          f"diag_keys={len(summary['final_diagnostics'])}")
    print(f"  artifacts: {out}")
    print("  (validates runner contract on CPU; HF/PEFT/real data still need the GPU cluster)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
