"""CPU tests for EAS/DISCO/ORTHO/SCOPE/SEQU/BOUND hooks and eval hooks."""
import types
import unittest

import torch
import torch.nn as nn

from implementations.hooks.bound_hook import BOUNDHook
from implementations.hooks.disco_hook import DISCOHook, POLICIES
from implementations.hooks.eas_hook import EASHook
from implementations.hooks.evals.cfa_eval import SCAAuditor
from implementations.hooks.evals.csda_eval import CSDAEvaluator, rouge_l
from implementations.hooks.evals.ripple_eval import RippleBenchEvaluator, mcq_accuracy
from implementations.hooks.evals.rrbu_eval import RRCurveEvaluator
from implementations.hooks.ortho_hook import ORTHOHook, ORTHOHookConfig, ORTHOSubspaceBank
from implementations.hooks.scope_hook import SCOPEHook
from implementations.hooks.sequ_hook import SEQUHook


class TinyLM(nn.Module):
    def __init__(self, vocab=16, hidden=8, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
                                     for _ in range(layers)])
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


class DummyTok:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        n = 4 if add_special_tokens else 2
        return {"input_ids": [1 + (i % 5) for i in range(n)]}


class PolicyModel(nn.Module):
    def __init__(self, vocab=16):
        super().__init__()
        self.proj = nn.Linear(1, vocab)

    def forward(self, input_ids, attention_mask=None):
        x = input_ids.float().mean(1, keepdim=True)
        return types.SimpleNamespace(logits=self.proj(x).unsqueeze(1))


class TestEAS(unittest.TestCase):
    def test_features_and_loss(self):
        claims = ["Alice Smith wrote Book One.", "Book One is fiction.",
                  "Alice Smith was born in Paris."]
        hook = EASHook(claims)
        feats = hook.builder.build("Alice Smith", "Step one. Step two.", max_steps=4)
        self.assertTrue(0.0 <= float(feats.kappa) <= 1.0)
        self.assertTrue(0.0 <= float(feats.depth) <= 1.0)
        out = hook.loss(torch.tensor([1.0]), torch.tensor([0.5]),
                        torch.tensor([0.1]), feats.kappa, feats.depth)
        self.assertTrue(torch.isfinite(out.loss))
        self.assertTrue(hook.budget_curve(feats.kappa, feats.depth))


class TestDISCO(unittest.TestCase):
    def test_policy_batches_and_loss(self):
        hook = DISCOHook()
        rows = [{"question": "q1"}, {"question": "q2"}]
        batches = hook.policy_batches(DummyTok(), rows, ["a1", "a2"], max_len=32,
                                      device="cpu")
        self.assertEqual(set(batches), set(POLICIES))
        model = PolicyModel()
        policy, bare, names = hook.policy_logits(model, batches)
        self.assertEqual(policy.shape[0], 4)  # excludes bare
        out = hook.loss(policy, bare, torch.tensor([1, 2]), torch.tensor(0.1))
        self.assertTrue(torch.isfinite(out.loss))


class TestORTHO(unittest.TestCase):
    def test_bank_and_projection(self):
        torch.manual_seed(0)
        model = TinyLM()
        cfg = ORTHOHookConfig(layers=(-1,), k=3, max_len=32, device="cpu")
        hook = ORTHOHook(cfg)
        hook.collect(model, DummyTok(), [("prompt", "This is the first long reasoning step.\nThis is the second long reasoning step.")])
        U = hook.bank.get(-1)
        self.assertEqual(tuple(U.shape), (8, 2))
        g = torch.randn(8)
        gp = hook.project_gradient(g, -1)
        self.assertLess(float((U.T @ gp).abs().max()), 1e-5)
        out = hook.loss(torch.tensor(1.0), torch.randn(2, 8), torch.randn(2, 8),
                        -1, torch.tensor(0.1))
        self.assertTrue(torch.isfinite(out.loss))


class TestSCOPE(unittest.TestCase):
    def test_variants_loss_and_scs(self):
        hook = SCOPEHook()
        classes = hook.equivalence_classes([{"question": "Who is Alice?"}])
        self.assertEqual(len(classes[0]["variants"]), hook.config.n_variants)
        logits = torch.randn(2, 4, 10)
        out = hook.loss(logits, logits.detach(), torch.tensor([1, 2]),
                        torch.tensor(0.1))
        self.assertTrue(torch.isfinite(out.loss))
        scs = hook.scs(torch.tensor([[0.0, 0.01], [0.0, 0.6]]), 0.05)
        self.assertAlmostEqual(float(scs), 0.5)


class TestSEQU(unittest.TestCase):
    def test_stream_and_loss(self):
        hook = SEQUHook()
        stream = hook.stream([{"q": i} for i in range(10)])
        self.assertEqual(len(stream), hook.config.n_batches)
        w = torch.tensor([1.0], requires_grad=True)
        params = {"w": w}
        out = hook.loss(torch.tensor(1.0), torch.tensor(0.2), params, params)
        self.assertTrue(torch.isfinite(out.loss))
        self.assertIn("repair_count", hook.summary())


class TestBOUND(unittest.TestCase):
    def test_boundaries_and_eval(self):
        hook = BOUNDHook()
        logits = torch.zeros(5, 10)
        logits[2:] = torch.tensor([0.0] + [5.0] + [0.0] * 8)
        bounds = hook.boundaries({96: logits, 192: logits})
        self.assertEqual(bounds[96], 2)
        ex = hook.exit_losses({96: logits}, torch.ones(1, 5, dtype=torch.long) * 3,
                              torch.tensor(1.0))
        out = hook.loss([torch.tensor(0.1)], [ex[96]])
        self.assertTrue(torch.isfinite(out.loss))
        self.assertTrue(hook.evaluate(100, 3, 5)["success"])


class TestEvalHooks(unittest.TestCase):
    def test_csda(self):
        self.assertGreater(rouge_l("Alice Smith wrote it", "Alice Smith wrote it"), 0.9)
        rep = torch.randn(1, 4)
        out = CSDAEvaluator().evaluate(["a"], ["a"], [rep], [rep], torch.tensor([1.0]))
        self.assertAlmostEqual(out["sparsity"], 1.0)

    def test_rrbu(self):
        curve = RRCurveEvaluator(ks=[0, 5]).curve(
            attack_fn=lambda k: k, leak_fn=lambda k: 0.1 * k)
        self.assertEqual(curve["rr5"], 0.5)

    def test_cfa(self):
        responses = ["I don't know."] * 4 + ["Alice Smith wrote it."] * 4
        out = SCAAuditor(probe_steps=20).audit(responses, torch.tensor([0] * 4 + [1] * 4))
        self.assertIn("sca", out)

    def test_ripple(self):
        self.assertAlmostEqual(mcq_accuracy([1, 0], [1, 0]), 1.0)
        out = RippleBenchEvaluator().evaluate(
            [1], [1], [0], [0], torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]), [0, 1])
        self.assertIn("ripple", out)


if __name__ == "__main__":
    unittest.main()
