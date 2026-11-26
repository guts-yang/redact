"""CFA evaluation hook: linear / MLP / n-gram side-channel attackers."""
from __future__ import annotations

import zlib
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cfa import extract_style_features, linear_probe_auc


def ngram_features(responses: Sequence[str], n: int = 2, dim: int = 256) -> torch.Tensor:
    """Hashed character n-gram counts."""
    rows = []
    for text in responses:
        vec = torch.zeros(dim)
        t = (text or "").lower()
        for i in range(max(0, len(t) - n + 1)):
            gram = t[i:i + n]
            vec[zlib.crc32(gram.encode()) % dim] += 1.0
        rows.append(vec)
    return torch.stack(rows)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    from ...cfa import _rank_auc
    return _rank_auc(scores, labels)


def mlp_probe_auc(
    features: torch.Tensor,
    membership: torch.Tensor,
    steps: int = 300,
    lr: float = 0.01,
    seed: int = 0,
) -> float:
    torch.manual_seed(seed)
    x = features.detach().float()
    y = membership.detach().float().view(-1)
    probe = MLPProbe(x.shape[1])
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(probe(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return _auc(probe(x), y)


class SCAAuditor:
    def __init__(self, probe_steps: int = 200, seed: int = 0):
        self.probe_steps = probe_steps
        self.seed = seed

    def audit(self, responses: Sequence[str],
              membership: torch.Tensor) -> Dict[str, object]:
        y = membership.float().view(-1)
        style = extract_style_features(responses).float()
        ngram = ngram_features(responses)
        auc_style = linear_probe_auc(style, y, steps=self.probe_steps, seed=self.seed)
        auc_ngram = linear_probe_auc(ngram, y, steps=self.probe_steps, seed=self.seed)
        auc_mlp = mlp_probe_auc(ngram, y, steps=self.probe_steps, seed=self.seed)
        aucs = {"style_linear": auc_style, "ngram_linear": auc_ngram,
                "ngram_mlp": auc_mlp}
        return {
            "auc": aucs,
            "sca": max(aucs.values()) - 0.5,
            "sca_positive": max(0.0, max(aucs.values()) - 0.5),
        }
