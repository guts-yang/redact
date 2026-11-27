"""Independent CPU unit tests for the consensus core (torch only).

Run: python methods/consensus/test_core.py   OR   python -m unittest discover
These assert mathematical properties, not just "it runs".
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import unittest


import torch
from core import CONSENSUS, ConsensusConfig, soft_consensus, majority_vote_label

def test_soft_consensus_grows_with_N():
    torch.manual_seed(0)
    p = torch.full((1,16), 0.3)
    a = soft_consensus(p, 1, 0.1, 0.2, 8.0)
    b = soft_consensus(p, 16, 0.1, 0.2, 8.0)
    assert b.item() >= a.item() - 1e-6, "CLR must be non-decreasing in N"

def test_majority_vote():
    x = torch.tensor([[1,1,2],[3,3,3]])
    assert majority_vote_label(x).tolist() == [1,3]

def test_loss_grad():
    torch.manual_seed(0)
    p = torch.rand(2,16, requires_grad=True)
    out = CONSENSUS(ConsensusConfig()).loss(p, torch.tensor(0.2), torch.tensor(0.0), torch.tensor(0.0))
    out.loss.backward(); assert p.grad is not None and torch.isfinite(out.loss)


class CoreTest(unittest.TestCase):
    def test_soft_consensus_grows_with_N(self):
        globals()['test_soft_consensus_grows_with_N']()
    def test_majority_vote(self):
        globals()['test_majority_vote']()
    def test_loss_grad(self):
        globals()['test_loss_grad']()


if __name__ == "__main__":
    unittest.main(verbosity=2)
