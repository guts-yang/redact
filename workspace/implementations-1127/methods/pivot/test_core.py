"""Independent CPU unit tests for the pivot core (torch only).

Run: python methods/pivot/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import PIVOT, de_pivot_consistency

def test_dpc_counts_any_leaking_step():
    # [Q,P,T]; one (q,p) has a step above eps
    s = torch.zeros(2,2,3); s[0,1,2] = 0.9
    assert abs(de_pivot_consistency(s, 0.05).item() - 1/4) < 1e-6

def test_loss_grad():
    n,t,v = 2,3,10
    sl = torch.randn(n,t,v, requires_grad=True)
    ref = sl.detach().clone()
    eids = torch.tensor([1,2]); g = torch.ones(n,t)
    out = PIVOT().loss(sl, ref, eids, g, torch.tensor(0.1),
                       torch.randn(n,4), torch.randn(n,4))
    out.loss.backward(); assert torch.isfinite(out.loss) and sl.grad is not None


class CoreTest(unittest.TestCase):
    def test_dpc_counts_any_leaking_step(self):
        globals()['test_dpc_counts_any_leaking_step']()
    def test_loss_grad(self):
        globals()['test_loss_grad']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
