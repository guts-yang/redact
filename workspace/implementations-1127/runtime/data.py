"""Dataset loading + prompt construction + entity extraction.

Mirrors the R-TOFU server contract (rows carry question/answer/cot/task_id,
ground truth always from the dataset). Pure stdlib so the self-check runs on
any machine without transformers.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Optional, Sequence

from . import paths

_CAP_PHRASE = re.compile(r"\b[A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)+\b")
_STOP_PREFIX = {
    "the", "this", "that", "these", "those", "it", "its", "in", "on", "as", "at",
    "for", "from", "with", "according", "based", "a", "an", "i", "we", "you",
    "there", "here", "what", "who", "when", "where", "how", "let", "first",
    "second", "finally", "however", "also", "to", "of", "and", "but", "if",
}
_LEADING_TITLES = ("Author ", "Authors ", "The author ", "The Author ", "The ")

# Prompt templates (from R-TOFU strategies.py); kept here so the package is
# self-contained and does not import the R-TOFU repo.
_PREFIXES = {
    "DefaultCoT": "",
    "LessThink": "Okay, the user asked this, I can answer it without thinking much.\n",
    "ZeroThink": "\n",
    "SummaryThink": "Brief recap: the question asks for one specific biographical fact.\n",
}


def read_jsonl(path) -> List[dict]:
    rows: List[dict] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_rows(path, n: int, seed: int, task_id: Optional[str] = "1") -> List[dict]:
    rows = read_jsonl(path)
    if task_id is not None:
        rows = [r for r in rows if str(r.get("task_id", "")) == str(task_id)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n] if n > 0 else []


def refusals() -> List[str]:
    p = paths.REFUSAL_FILE
    if not p.exists():
        return ["I don't know.", "I cannot answer that."]
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _clean_phrase(phrase: str) -> str:
    phrase = phrase.strip()
    for title in _LEADING_TITLES:
        if phrase.startswith(title):
            phrase = phrase[len(title):]
            break
    return re.sub(r"(?:'s|’s)$", "", phrase).strip()


def key_phrase(answer: str) -> str:
    """Longest capitalized multi-word phrase in the gold answer (the entity)."""
    candidates = []
    for m in _CAP_PHRASE.finditer(answer or ""):
        phrase = _clean_phrase(m.group(0))
        if not phrase or " " not in phrase:
            continue
        if phrase.split()[0].lower().strip("'") in _STOP_PREFIX:
            continue
        candidates.append(phrase)
    if not candidates:
        return ""
    best = max((len(p.split()), len(p)) for p in candidates)
    return [p for p in candidates if (len(p.split()), len(p)) == best][-1]


def entity_token_ids(tokenizer, rows: Sequence[dict]):
    """First token id of the key phrase per row; returns a LongTensor."""
    import torch
    ids = []
    for r in rows:
        phrase = key_phrase(r.get("answer", ""))
        toks = tokenizer(phrase, add_special_tokens=False)["input_ids"] if phrase else [0]
        ids.append(int(toks[0]) if toks else 0)
    return torch.tensor(ids, dtype=torch.long)


def build_prompt(question: str, strategy: str = "DefaultCoT") -> str:
    head = f"<｜User｜>{question}<｜Assistant｜>"
    if strategy == "DefaultCoT":
        return head + "<think>\n"
    if strategy not in _PREFIXES:
        raise ValueError(f"unknown strategy {strategy}")
    return head + "<think>\n" + _PREFIXES[strategy] + "</think>\n\n"


def split_cot_steps(cot: str, max_steps: int = 24, min_chars: int = 20) -> List[str]:
    """Deterministic CoT sentence splitter (step == one sentence)."""
    text = (cot or "").strip()
    if not text:
        return []
    steps: List[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        for sent in re.split(r"(?<=[.!?])\s+", chunk.strip()):
            if len(sent.strip()) >= min_chars:
                steps.append(sent.strip())
    if not steps:
        steps = [text]
    if len(steps) > max_steps:
        stride = max(1, len(steps) // max_steps)
        steps = steps[::stride][:max_steps]
    return steps


def emphasize(question: str, entity: str) -> str:
    """Keyword-emphasis rewrite used by ATTRIB (cheapest recovery attack)."""
    if not entity:
        return question
    return f"{question}\n(Important: the answer specifically concerns {entity}. Focus on {entity}.)"


def translate_pivot(question: str, lang: str, entity: str) -> str:
    """Lightweight 'cross-lingual pivot' wrapper.

    NOTE: this is a deterministic tagging wrapper, NOT a real translation. On the
    cluster set UNL1127_TRANSLATOR=1 to route through a translation model, or
    provide pre-translated rows via UNL1127_MULTILINGUAL_FILE. Keep the L_q vs
    L_p distinction explicit; we never pretend a tag equals a translation.
    """
    return f"[{lang}] {question} (answer eventually in {lang}; entity: {entity})"
