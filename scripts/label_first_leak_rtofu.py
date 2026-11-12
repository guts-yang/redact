"""Per-CoT first-leak labeler for the R-TOFU forget-set leak classifier.

Renders `docs/labeling_prompts/first_leak_per_cot_v1.md` per CoT. ``main``
calls ``scripts.deepseek_label_first_leak_rtofu``. The key is
``DEEPSEEK_API_KEY`` or the repo-root ``.env``. ``--model sonnet`` is
rewritten to ``deepseek-flash``. Output is a single JSON object parsed
with json.loads.

One JSONL record per CoT — the script emits one decision per (split, row),
not per sentence. Schema (top-level keys):

    {uid, row_index, author, n_sentences,
     first_leak_idx, matched_claim_idx, claim_snippet, subject_resolved,
     rationale,
     raw_output, model, cost_usd, session_id, status, attempts, error?}

`status ∈ {ok, parse_failed, validation_failed, error}`. Resumable: uids whose
most recent record is `status == ok` are skipped on re-run.

Row -> author mapping follows `prepare_contextual_label_batches.py`:
`index // 20 == author_idx` against `config/authors/<split>.json`. The CSV
must therefore be the canonical no-shuffle, no-offset forget-split response
file (`leak_detector_data_promptfmt/<split>/forget_responses.csv` by default).

Usage:
    python -m scripts.label_first_leak_rtofu \\
        --split forget10 \\
        --output_jsonl leak_detector_data_promptfmt/forget10/first_leak_labels.jsonl \\
        --workers 4
"""
import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from redact.dataset.contextual_window import clean_question  # noqa: E402
from redact.evaluator.utils import split_sentences  # noqa: E402
from redact.utils import log_print  # noqa: E402


PROMPT_TEMPLATE_PATH = REPO / "docs/labeling_prompts/first_leak_per_cot_v1.md"
SYSTEM_PROMPT = "You are a helpful assistant."
SPLIT_CHOICES = ["forget01", "forget05", "forget10"]
REQUIRED_CSV_COLS = {"index", "question", "model_cot"}
PRIMARY_FIELDS = (
    "first_leak_idx", "matched_claim_idx", "claim_snippet",
    "subject_resolved", "rationale",
)


def load_template() -> str:
    """Read the labeling prompt and strip the render-contract header.

    The template's leading block (everything up to and including the first
    `---` separator) documents placeholders for human readers; the judge does
    not need it. If the separator is missing the script raises rather than
    sending the meta header — that header references the placeholders by name
    and would confuse the judge.
    """
    raw = PROMPT_TEMPLATE_PATH.read_text()
    parts = raw.split("\n---\n", 1)
    if len(parts) != 2:
        raise RuntimeError(
            f"Prompt template missing `---` separator: {PROMPT_TEMPLATE_PATH}"
        )
    return parts[1].lstrip()


def render_author_claims_indexed(claim_ranges, all_claims) -> str:
    """Render one author's KB claims as ``- **N**: <claim>`` lines.

    ``N`` is the global index into the forget-split ``claims.json`` so that
    ``matched_claim_idx`` values emitted by the judge are directly
    interpretable against the canonical KB.
    """
    lines = []
    for start, end in claim_ranges:
        for idx in range(start, end):
            lines.append(f"- **{idx}**: {all_claims[idx]}")
    return "\n".join(lines)


def render_cot_indexed(cot_sentences) -> str:
    return "\n".join(f"- **[{i}]**: {s}" for i, s in enumerate(cot_sentences))


def build_prompt(
    template: str,
    author_name: str,
    question: str,
    author_claims_indexed: str,
    cot_indexed: str,
) -> str:
    rendered = (
        template
        .replace("{{QUESTION}}", question)
        .replace("{{AUTHOR_CLAIMS_INDEXED}}", author_claims_indexed)
        .replace("{{COT_INDEXED}}", cot_indexed)
        # {{AUTHOR}} substituted last so it doesn't accidentally fire inside
        # earlier-substituted blocks (e.g. a claim happening to contain the
        # literal `{{AUTHOR}}` token).
        .replace("{{AUTHOR}}", author_name)
    )
    if "{{" in rendered:
        idx = rendered.index("{{")
        raise RuntimeError(
            f"Unfilled placeholder in rendered prompt at {idx}: "
            f"...{rendered[max(0, idx - 30):idx + 60]}..."
        )
    return rendered


_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_label(raw, *, n_sentences: int, n_claims: int) -> tuple[dict | None, str | None]:
    """Parse and validate the judge's JSON response.

    Returns ``(parsed_dict, None)`` on success or ``(None, reason)`` on failure.
    Strategy:
      1. ``json.loads`` on the raw text (after stripping markdown code fences
         if present).
      2. Fallback: pull the first ``{ ... }`` substring and reparse.
      3. Validate types and field-consistency invariants from the schema.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty or non-string response"

    text = raw.strip()
    # Common judge tic: wrap JSON in ```json ... ``` despite being told not to.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    obj: dict | None = None
    try:
        cand = json.loads(text)
        if isinstance(cand, dict):
            obj = cand
    except json.JSONDecodeError:
        pass

    if obj is None:
        m = _BRACE_RE.search(text)
        if m is None:
            return None, "no JSON object found"
        try:
            cand = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return None, f"JSON decode failed: {e}"
        if not isinstance(cand, dict):
            return None, "top-level JSON is not an object"
        obj = cand

    missing = [k for k in PRIMARY_FIELDS if k not in obj]
    if missing:
        return None, f"missing fields: {missing}"

    fli = obj["first_leak_idx"]
    if fli is not None:
        if not isinstance(fli, int) or isinstance(fli, bool):
            return None, f"first_leak_idx not int|null: {fli!r}"
        if not (0 <= fli < n_sentences):
            return None, f"first_leak_idx={fli} out of range [0, {n_sentences})"
        mci = obj["matched_claim_idx"]
        if not isinstance(mci, int) or isinstance(mci, bool):
            return None, f"first_leak_idx set but matched_claim_idx not int: {mci!r}"
        if not (0 <= mci < n_claims):
            return None, f"matched_claim_idx={mci} out of range [0, {n_claims})"
        if not isinstance(obj["claim_snippet"], str) or not obj["claim_snippet"]:
            return None, "first_leak_idx set but claim_snippet is empty/null"
    else:
        # null-fallthrough: matched_claim_idx, claim_snippet, subject_resolved
        # must all be null.
        for k in ("matched_claim_idx", "claim_snippet", "subject_resolved"):
            if obj[k] is not None:
                return None, f"first_leak_idx=null but {k}={obj[k]!r}"

    return obj, None


def call_claude(
    prompt: str,
    model: str,
    timeout: float,
    claude_bin: str,
) -> dict:
    """One invocation of `claude`. Returns the parsed JSON response envelope.

    Isolation flags mirror ``scripts/judge_cfe_rtofu.py`` so the call routes
    through OAuth/subscription auth while behaving like ``--bare``: no tools,
    no slash-commands, no MCPs, no project-dir CLAUDE.md (cwd=/tmp).

    ``--settings '{"temperature": 0}'`` pins decoding to greedy, giving
    byte-reproducible outputs across re-runs (verified against a 3-call
    probe — without it, sampling drifts even at low temperatures and the
    ~5% rate of borderline atom-selection flips observed in the limit-20
    vs limit-40 smokes is purely sampling variance, not rule disagreement).
    The CLI silently ignores unknown settings keys but a 3x probe confirmed
    ``temperature`` is respected (identical output for prompts that would
    otherwise vary). The same fix should be applied to
    ``judge_cfe_rtofu.py`` if reproducibility of CFE scores matters there.
    """
    proc = subprocess.run(
        [
            claude_bin,
            "--print", prompt,
            "--model", model,
            "--output-format", "json",
            "--tools", "",
            "--system-prompt", SYSTEM_PROMPT,
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--mcp-config", '{"mcpServers": {}}',
            "--setting-sources", "user",
            "--no-session-persistence",
            "--settings", '{"temperature": 0}',
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/tmp",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited {proc.returncode}; "
            f"stderr={proc.stderr.strip()[:500]!r}"
        )
    try:
        resp = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"claude stdout was not JSON ({e}); "
            f"stdout[:500]={proc.stdout[:500]!r}"
        ) from e
    if resp.get("is_error"):
        raise RuntimeError(
            f"claude reported is_error (subtype={resp.get('subtype')!r}); "
            f"result[:200]={str(resp.get('result'))[:200]!r}"
        )
    return resp


def _looks_rate_limited(err_msg: str) -> bool:
    s = err_msg.lower()
    return any(t in s for t in ("rate", "429", "quota", "overloaded", "too many"))


def label_one(
    uid: str,
    row_index: int,
    author: str,
    prompt: str,
    n_sentences: int,
    n_claims: int,
    model: str,
    max_retries: int,
    timeout: float,
    claude_bin: str,
) -> dict:
    """Label a single CoT, returning a JSONL-ready record."""
    last_err: str | None = None
    last_raw: str | None = None
    last_cost = 0.0
    last_session_id: str | None = None

    for attempt in range(max_retries + 1):
        try:
            resp = call_claude(prompt, model, timeout, claude_bin)
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {timeout}s"
            time.sleep(min(2 ** attempt, 30))
            continue
        except RuntimeError as e:
            last_err = str(e)
            wait = 30 if _looks_rate_limited(last_err) else min(2 ** attempt, 30)
            time.sleep(wait)
            continue

        raw = resp.get("result", "") or ""
        last_raw = raw
        last_cost = float(resp.get("total_cost_usd") or 0.0)
        last_session_id = resp.get("session_id")

        parsed, reason = parse_label(raw, n_sentences=n_sentences, n_claims=n_claims)
        if parsed is not None:
            return {
                "uid": uid,
                "row_index": row_index,
                "author": author,
                "n_sentences": n_sentences,
                **{k: parsed[k] for k in PRIMARY_FIELDS},
                "raw_output": raw,
                "model": model,
                "cost_usd": last_cost,
                "session_id": last_session_id,
                "status": "ok",
                "attempts": attempt + 1,
            }
        last_err = f"parse/validation failed: {reason}"

    status = "validation_failed" if last_raw is not None else "error"
    return {
        "uid": uid,
        "row_index": row_index,
        "author": author,
        "n_sentences": n_sentences,
        **{k: None for k in PRIMARY_FIELDS},
        "raw_output": last_raw,
        "model": model,
        "cost_usd": last_cost,
        "session_id": last_session_id,
        "status": status,
        "attempts": max_retries + 1,
        "error": last_err,
    }


def load_existing_ok_uids(jsonl_path: str) -> set[str]:
    """Return uids whose MOST RECENT record is `status == ok`.

    Mirrors ``load_existing_ok_uids`` in ``judge_cfe_rtofu.py``: most-recent
    wins per uid, so a failed-then-succeeded uid is skipped on resume and a
    succeeded-then-failed uid (rare; only via manual edit) is retried.
    """
    if not os.path.exists(jsonl_path):
        return set()
    last: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("uid")
            if uid is None:
                continue
            last[uid] = rec
    return {uid for uid, rec in last.items() if rec.get("status") == "ok"}


def main() -> None:
    """Label the first leaking sentence through the DeepSeek client.

    Keeps the historical default output path when ``--output_jsonl`` is omitted.
    """
    import sys

    from scripts.deepseek_client import prepare_deepseek_argv
    from scripts.deepseek_label_first_leak_rtofu import main as deepseek_main

    argv = sys.argv[1:]
    split = None
    has_output = False
    for i, arg in enumerate(argv):
        if arg == "--split" and i + 1 < len(argv):
            split = argv[i + 1]
        elif arg.startswith("--split="):
            split = arg.split("=", 1)[1]
        elif arg == "--output_jsonl" or arg.startswith("--output_jsonl="):
            has_output = True
    default_output = None
    if split and not has_output:
        default_output = (
            f"leak_detector_data_promptfmt/{split}/first_leak_labels.jsonl"
        )
    prepare_deepseek_argv(default_output=default_output)
    deepseek_main()


if __name__ == "__main__":
    main()
