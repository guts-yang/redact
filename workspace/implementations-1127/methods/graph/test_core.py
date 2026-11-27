"""Independent CPU unit tests for the graph core (torch only).

Run: python methods/graph/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import GRAPH, MemoryGraph, coverage

def test_graph_strength_averaging():
    g = MemoryGraph(); g.add_path(["a","b"], [0.2]); g.add_path(["a","b"], [0.8])
    assert abs(g.strength("a","b") - 0.5) < 1e-6

def test_coverage():
    g = MemoryGraph(); g.add_path(["anchor","X"]); g.add_path(["anchor","Y"])
    assert 0.0 <= coverage(g, ["X"], ["X","Y","Z"]) <= 1.0

def test_loss_grad():
    n,t,v=2,3,10
    sl=torch.randn(n,t,v,requires_grad=True)
    out=GRAPH().loss(sl, sl.detach().clone(), torch.tensor([1,2]), torch.ones(n,t), torch.tensor(0.0))
    out.loss.backward(); assert torch.isfinite(out.loss)


class CoreTest(unittest.TestCase):
    def test_graph_strength_averaging(self):
        globals()['test_graph_strength_averaging']()
    def test_coverage(self):
        globals()['test_coverage']()
    def test_loss_grad(self):
        globals()['test_loss_grad']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
