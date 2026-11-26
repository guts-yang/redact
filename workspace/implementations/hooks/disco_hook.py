"""Full DISCO hook: five decoding policies + bare-prompt min-max loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..base import MethodOutput
from ..disco import DISCO, DISCOConfig, decoding_separation, mcs_worst

POLICIES = ("DefaultCoT", "LessThink", "ZeroThink", "SummaryThink", "bare")

_PREFIX = {
    "DefaultCoT": "",
    "LessThink": "Okay, the user asked this, I can answer it without thinking much.\n",
    "ZeroThink": "\n",
    "SummaryThink": "Brief recap: the question asks for one specific biographical fact.\n",
    "bare": None,
}


def build_prompt(question: str, policy: str) -> str:
    head = f"<｜User｜>{question}<｜Assistant｜>"
    if policy == "bare":
        return head
    if policy == "DefaultCoT":
        return head + "<think>\n"
    return head + "<think>\n" + _PREFIX[policy] + "</think>\n\n"


def encode_example(tok, prompt: str, target: str, max_len: int):
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p_ids) + len(t_ids) > max_len:
        p_ids = p_ids[-(max_len - len(t_ids)):]
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids


def collate(tok, examples, device="cuda"):
    length = max(len(e[0]) for e in examples)
    ids, labels, mask = [], [], []
    for seq, lab in examples:
        pad = length - len(seq)
        ids.append(seq + [tok.pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(seq) + [0] * pad)
    return (torch.tensor(ids, device=device),
            torch.tensor(labels, device=device),
            torch.tensor(mask, device=device))


class DISCOHook:
    def __init__(self, config: Optional[DISCOConfig] = None):
        self.disco = DISCO(config or DISCOConfig())

    def policy_batches(
        self,
        tok,
        rows: Sequence[dict],
        targets: Sequence[str],
        max_len: int = 512,
        policies: Sequence[str] = POLICIES,
        device: str = "cuda",
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        out = {}
        for policy in policies:
            examples = [encode_example(tok, build_prompt(r["question"], policy),
                                       t, max_len)
                        for r, t in zip(rows, targets)]
            out[policy] = collate(tok, examples, device=device)
        return out

    def policy_logits(self, model, batches: Dict[str, tuple]):
        logits = {}
        for policy, (ids, _, mask) in batches.items():
            out = model(input_ids=ids, attention_mask=mask)
            logits[policy] = out.logits[:, -1, :].float()
        policies = [p for p in POLICIES if p in logits]
        stacked = torch.stack([logits[p] for p in policies if p != "bare"], dim=0)
        return stacked, logits["bare"], policies

    def loss(
        self,
        policy_logits: torch.Tensor,
        bare_logits: torch.Tensor,
        entity_ids: torch.Tensor,
        retain_loss: torch.Tensor,
    ) -> MethodOutput:
        return self.disco.loss(policy_logits, bare_logits, entity_ids, retain_loss)

    def separation(self, leak_rates: torch.Tensor) -> torch.Tensor:
        return decoding_separation(leak_rates.max(), leak_rates.min())

    @staticmethod
    def mcs(leak_rates: torch.Tensor) -> torch.Tensor:
        return mcs_worst(leak_rates)
