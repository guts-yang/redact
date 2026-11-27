"""Independent CPU unit tests for the probe core (torch only).

Run: python methods/probe/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import PROBE, roc_auc, StepProbe

def test_auc_perfect_separation():
    s = torch.tensor([0.1,0.2,0.8,0.9]); y = torch.tensor([0.,0.,1.,1.])
    assert abs(roc_auc(s,y).item()-1.0) < 1e-5

def test_auc_chance():
    s = torch.tensor([0.1,0.9]); y = torch.tensor([0.,1.])
    assert 0.0 <= roc_auc(s,y).item() <= 1.0

def test_loss_grad():
    torch.manual_seed(0)
    h = torch.randn(8,16, requires_grad=True)
    m = PROBE(); probes=[StepProbe(16), StepProbe(16)]
    out = m.loss(h, torch.cat([torch.ones(4),torch.zeros(4)]), probes,
                 torch.tensor(0.0), h.detach().clone())
    out.loss.backward(); assert torch.isfinite(out.loss) and h.grad is not None


class CoreTest(unittest.TestCase):
    def test_auc_perfect_separation(self):
        globals()['test_auc_perfect_separation']()
    def test_auc_chance(self):
        globals()['test_auc_chance']()
    def test_loss_grad(self):
        globals()['test_loss_grad']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
