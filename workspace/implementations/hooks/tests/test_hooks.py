"""CPU tests for the full CSDA/RRBU/CFA/RIPPLE hooks."""
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from implementations.hooks.cfa_hook import CFAHook, CFAHookConfig
from implementations.hooks.csda_hook import CSDAHook, CSDAHookConfig, entity_token_id
from implementations.hooks.ripple_hook import KnowledgeGraph, RIPPLEHook, RIPPLEHookConfig
from implementations.hooks.rrbu_hook import RRBUHook, RRBUHookConfig, attack_eval_split


class TinyLM(nn.Module):
    """Minimal causal LM with a decoder-layer ModuleList and HF-like output."""

    def __init__(self, vocab=16, hidden=8, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
            for _ in range(layers)
        ])
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        h = self.emb(input_ids)
        states = [h]
        for layer in self.layers:
            h = layer(h)
            states.append(h)
        logits = self.head(h)
        if output_hidden_states:
            return types.SimpleNamespace(logits=logits, hidden_states=tuple(states))
        return types.SimpleNamespace(logits=logits)


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=True):
        # deterministic short ids; enough to exercise span collection
        n = 3 if add_special_tokens else 2
        return {"input_ids": [1 + (i % 5) for i in range(n)]}


class TestCSDAHook(unittest.TestCase):
    def test_retain_stats_and_loss(self):
        torch.manual_seed(0)
        model = TinyLM()
        hook = CSDAHook(model, DummyTokenizer(),
                        CSDAHookConfig(layer=1, max_steps=3, max_len=64, device="cpu"),
                        csda_config=__import__("implementations.csda", fromlist=["CSDAConfig"]).CSDAConfig(n_patch_samples=1))
        mean, cov = hook.collect_retain_stats([("prompt", "Step one. Step two.")])
        self.assertEqual(mean.shape[0], 8)
        self.assertEqual(tuple(cov.shape), (8, 8))
        out = hook.loss("prompt", "Step one. Step two.", entity_id=3,
                        retain_kl=torch.tensor(0.1))
        self.assertTrue(torch.isfinite(out.loss))

    def test_split(self):
        from implementations.hooks.common import split_cot_steps
        steps = split_cot_steps("First sentence here. Second sentence here.\n\nThird paragraph here.")
        self.assertGreaterEqual(len(steps), 2)


class TestRRBUHook(unittest.TestCase):
    def test_outer_loss(self):
        torch.manual_seed(0)
        model = TinyLM()
        hook = RRBUHook(model, RRBUHookConfig(inner_steps=1, inner_lr=0.01))
        ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        labels = ids.clone()
        mask = torch.ones_like(ids)
        batch = (ids, labels, mask)
        out = hook.outer_loss(attack_batch=batch, retain_batch=batch)
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_split(self):
        attack, ev = attack_eval_split([{"i": i} for i in range(8)], 3, 0)
        self.assertEqual(len(attack), 3)
        self.assertEqual(len(ev), 5)


class TestCFAHook(unittest.TestCase):
    def test_training_and_audit(self):
        torch.manual_seed(0)
        hook = CFAHook(feature_dim=7, config=CFAHookConfig(feature_dim=7), device="cpu")
        hidden = torch.randn(4, 3, 8)
        membership = torch.tensor([1, 1, 0, 0])
        out = hook.training_loss(torch.tensor(1.0), torch.tensor(0.2), hidden, membership)
        self.assertTrue(torch.isfinite(out.loss))
        opt = torch.optim.Adam(hook.adversary.parameters(), lr=0.01)
        loss = hook.update_adversary(hidden, membership, opt)
        self.assertTrue(loss >= 0.0)
        responses = ["I don't know."] * 4 + ["Alice Smith wrote it."] * 4
        auc, sca = hook.audit_responses(responses, torch.tensor([0] * 4 + [1] * 4))
        self.assertTrue(0.0 <= auc <= 1.0)
        self.assertTrue(sca >= 0.0)
        self.assertEqual(len(hook.counterfactuals("q", 0)), hook.config.n_counterfactuals)


class TestRIPPLEHook(unittest.TestCase):
    def test_graph_and_loss(self):
        claims = [
            "Basil Mahfouz Al-Kuwaiti was born in Kuwait City.",
            "Basil Mahfouz Al-Kuwaiti wrote Promise by the Seine.",
            "Promise by the Seine is French literature.",
        ]
        graph = KnowledgeGraph.from_claims(claims)
        self.assertIn("Basil Mahfouz Al-Kuwaiti", graph.entities())
        self.assertTrue(graph.neighbors("Basil Mahfouz Al-Kuwaiti", 1))
        hook = RIPPLEHook(graph, RIPPLEHookConfig(rho=1.0, hops=2))
        current = torch.randn(3, 12)
        ref = torch.randn(3, 12)
        out = hook.loss(torch.tensor(1.0), current, ref,
                        torch.randn(1, 8), torch.randn(1, 8))
        self.assertTrue(torch.isfinite(out.loss))
        metrics = hook.metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]),
                               [0, 1], torch.tensor([1.0, 0.0]))
        self.assertIn("ripple", metrics)
        self.assertIn("recon", metrics)

    def test_real_claims_file(self):
        path = Path("/usr/local/workspace/tracks/lrm-unlearning/redact/knowledge_banks/forget10/claims.json")
        if not path.exists():
            self.skipTest("claims file not present")
        graph = KnowledgeGraph.from_claims_file(str(path))
        self.assertGreater(len(graph.entities()), 0)


if __name__ == "__main__":
    unittest.main()
