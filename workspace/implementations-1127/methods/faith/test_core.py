"""Independent CPU unit tests for the faith core (torch only).

Run: python methods/faith/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import FAITH, faithfulness

def test_faith_zero_when_identical():
    x = torch.randn(1,10)
    assert faithfulness(x, x).abs().max().item() < 1e-5

def test_faith_positive_when_shifted():
    a = torch.zeros(1,10); a[0,0]=5.0
    b = torch.zeros(1,10); b[0,1]=5.0
    assert faithfulness(a,b).item() > 0.1

def test_gated_loss_grad():
    torch.manual_seed(0)
    h = torch.randn(4,6, requires_grad=True)
    out = FAITH().loss(h, torch.zeros(6), torch.linspace(0,1,4),
                       torch.tensor([0.9,0.8,0.1,0.1]), torch.tensor(-2.0), torch.tensor(0.0))
    out.loss.backward(); assert torch.isfinite(out.loss)


class CoreTest(unittest.TestCase):
    def test_faith_zero_when_identical(self):
        globals()['test_faith_zero_when_identical']()
    def test_faith_positive_when_shifted(self):
        globals()['test_faith_positive_when_shifted']()
    def test_gated_loss_grad(self):
        globals()['test_gated_loss_grad']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
