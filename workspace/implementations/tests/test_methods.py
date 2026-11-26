"""CPU smoke tests for all ten decoupled implementations."""
import math
import unittest

import torch
import torch.nn as nn

from implementations.bound import BOUND, BOUNDConfig, censored_interval, commitment_boundary
from implementations.cfa import CFA, CFAConfig, extract_style_features, linear_probe_auc
from implementations.csda import CSDA, CSDAConfig, causal_step_contributions, continuous_mask
from implementations.disco import DISCO, DISCOConfig, mcs_worst
from implementations.eas import EAS, EASConfig, entanglement_alpha
from implementations.ortho import ORTHO, ORTHOConfig, orthogonalize_gradient
from implementations.ripple import RIPPLE, RIPPLEConfig, multi_hop_neighbors, pathcut_loss
from implementations.rrbu import RRBU, RRBUConfig, rrbu_loss
from implementations.scope import SCOPE, SCOPEConfig, step_scope_consistency
from implementations.sequ import SEQU, SEQUConfig


class TestCSDA(unittest.TestCase):
    def test_contributions_and_loss(self):
        torch.manual_seed(0)
        hidden = torch.randn(4, 2, 6)
        mean = torch.zeros(6)
        cov = torch.eye(6)
        cfg = CSDAConfig(n_patch_samples=3)
        c = causal_step_contributions(hidden, lambda h: h[..., 0].abs().mean(),
                                      mean, cov, cfg,
                                      generator=torch.Generator().manual_seed(1))
        self.assertEqual(c.shape, (4,))
        m = continuous_mask(c, cfg.tau, cfg.kappa)
        self.assertTrue(torch.all(m >= 0) and torch.all(m <= 1))
        out = CSDA(cfg).loss(hidden, hidden.detach(), torch.zeros(6), c,
                             torch.tensor(0.2))
        self.assertTrue(torch.isfinite(out.loss))
        self.assertIn("csda_sparsity", out.diagnostics)


class TestEAS(unittest.TestCase):
    def test_alpha_and_loss(self):
        kappa = torch.tensor([0.1, 0.5, 1.0])
        depth = torch.tensor([0.2, 0.4, 0.8])
        alpha = entanglement_alpha(kappa, depth)
        self.assertTrue(torch.all(alpha >= 0) and torch.all(alpha <= 1))
        out = EAS(EASConfig(budget=0.5)).loss(
            alpha, torch.rand(3), torch.rand(3), torch.rand(3), kappa,
            leak=torch.tensor(0.2))
        self.assertTrue(torch.isfinite(out.loss))
        self.assertGreaterEqual(float(out.diagnostics["eas_budget_penalty"]), 0.0)


class TestRRBU(unittest.TestCase):
    def test_bilevel_loss(self):
        w = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        params = {"w": w, "b": b}
        theta0 = {"w": torch.tensor(0.0), "b": torch.tensor(0.0)}
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 5.0])

        def attack(p):
            return ((p["w"] * x + p["b"] - y) ** 2).mean()

        def forget(p):
            return ((p["w"] * x + p["b"] - y) ** 2).mean()

        def retain(p):
            return p["w"].pow(2)

        out = rrbu_loss(params, attack, forget, retain, theta0,
                        RRBUConfig(inner_steps=2, inner_lr=0.01, rho=0.01))
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        self.assertIsNotNone(w.grad)


class TestDISCO(unittest.TestCase):
    def test_worst_policy_and_mcs(self):
        torch.manual_seed(0)
        policy = torch.randn(3, 2, 10)
        bare = torch.randn(2, 10)
        ids = torch.tensor([1, 2])
        out = DISCO(DISCOConfig()).loss(policy, bare, ids, torch.tensor(0.1))
        self.assertTrue(torch.isfinite(out.loss))
        rates = torch.tensor([0.0, 0.5, 1.0])
        self.assertEqual(float(mcs_worst(rates)), 0.0)


class TestORTHO(unittest.TestCase):
    def test_subspace_and_projection(self):
        torch.manual_seed(0)
        acts = torch.randn(32, 6)
        ortho = ORTHO(ORTHOConfig(k=3))
        U = ortho.subspace(acts)
        self.assertEqual(tuple(U.shape), (6, 3))
        g = torch.randn(6)
        gp = orthogonalize_gradient(g, U)
        self.assertLess(float((U.T @ gp).abs().max()), 1e-5)
        out = ortho.loss(torch.tensor(1.0), acts[:2], acts[:2] + 0.1, U,
                         torch.tensor(0.2))
        self.assertTrue(torch.isfinite(out.loss))


class TestSCOPE(unittest.TestCase):
    def test_loss_and_scs(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 10)
        ref = torch.randn(2, 3, 10)
        ids = torch.tensor([1, 2])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        out = SCOPE(SCOPEConfig()).loss(logits, ref, ids, mask, torch.tensor(0.2))
        self.assertTrue(torch.isfinite(out.loss))
        scs = step_scope_consistency(torch.tensor([[0.0, 0.01], [0.0, 0.6]]), 0.05)
        self.assertAlmostEqual(float(scs), 0.5)


class TestSEQU(unittest.TestCase):
    def test_drift_loss_and_repair(self):
        torch.manual_seed(0)
        seq = SEQU(SEQUConfig(lambda0=0.1, rho=0.01, delta=0.05))
        current = torch.randn(2, 4)
        base = torch.zeros(2, 4)
        drift = seq.observe_drift(current, base)
        self.assertGreater(float(drift), 0.0)
        w = torch.tensor([1.0], requires_grad=True)
        params = {"w": w}
        out = seq.loss(torch.tensor(1.0), torch.tensor(0.2), params, params, drift)
        self.assertTrue(torch.isfinite(out.loss))
        current_cot = w.view(1, 1) * torch.tensor([[1.0, 2.0]])
        repaired = seq.repair(params, current_cot, torch.zeros(1, 2))
        self.assertEqual(tuple(repaired["w"].shape), (1,))


class TestCFA(unittest.TestCase):
    def test_features_and_audit(self):
        responses = ["I don't know. </think>", "The answer is Alice Smith."] * 4
        feats = extract_style_features(responses)
        self.assertEqual(feats.shape[0], 8)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        auc = linear_probe_auc(feats, labels, steps=20)
        self.assertTrue(0.0 <= auc <= 1.0)
        adversary = nn.Linear(feats.shape[1], 1)
        out = CFA(CFAConfig()).loss(torch.tensor(1.0), torch.tensor(0.2), feats,
                                    labels, adversary)
        self.assertTrue(torch.isfinite(out.loss))


class TestRIPPLE(unittest.TestCase):
    def test_neighbors_hops_and_loss(self):
        emb = torch.tensor([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0]])
        rip = RIPPLE(RIPPLEConfig(rho=0.5, hops=2))
        self.assertEqual(rip.neighbors(emb, 0), [1])
        self.assertEqual(multi_hop_neighbors({0: [1], 1: [2], 2: []}, 0, 2), {1, 2})
        cut = pathcut_loss(torch.tensor([[1.0, 0.0]]), torch.tensor([1.0, 0.0]), 0.7)
        self.assertAlmostEqual(float(cut), 0.3, places=5)
        out = rip.loss(torch.tensor(1.0), torch.tensor([0.2, 0.3]), torch.tensor([0.3]))
        self.assertTrue(torch.isfinite(out.loss))


class TestBOUND(unittest.TestCase):
    def test_boundary_and_censoring(self):
        logits = torch.zeros(5, 10)
        logits[2:] = torch.tensor([0.0] + [5.0] + [0.0] * 8)
        tau = commitment_boundary(logits, delta_js=1e-6)
        self.assertEqual(tau, 2)
        lo, hi, width = censored_interval(100, 3, 5)
        self.assertAlmostEqual(lo, 0.03)
        self.assertAlmostEqual(hi, 0.08)
        self.assertAlmostEqual(width, 0.05)
        bound = BOUND(BOUNDConfig())
        out = bound.loss([torch.tensor(0.1), torch.tensor(0.2)],
                         [torch.tensor(0.3), torch.tensor(0.1)])
        self.assertTrue(torch.isfinite(out.loss))
        self.assertTrue(bound.evaluate(100, 3, 5)["success"])


if __name__ == "__main__":
    unittest.main()
