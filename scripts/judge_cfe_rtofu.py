"""LLM-as-judge runner for R-TOFU-faithful CFE.

Re-implements the GPT-4o judge described in R-TOFU (arXiv:2505.15214,
Figure 7 / `test_cot.py:evaluate_with_gpt`). ``main`` calls
``scripts.deepseek_judge_cfe_rtofu`` (DeepSeek chat API; key from
``DEEPSEEK_API_KEY`` or the repo-root ``.env``). ``--model sonnet`` is
rewritten to ``deepseek-flash``. The prompt is byte-identical to the
upstream f-string (including the leading newline and the stray ``**``
after "hints at"), and we parse with the same regex (``r"\\d+\\.\\d{2}"``).

Auth note: we deliberately do NOT pass ``--bare``. ``--bare`` forces
``ANTHROPIC_API_KEY`` auth (pay-per-token) and skips OAuth/keychain
reads; that would defeat the subscription-billing goal. Instead we
approximate ``--bare``'s isolation with explicit flags
(``--system-prompt``, ``--tools ""``, ``--disable-slash-commands``,
``--strict-mcp-config --mcp-config '{"mcpServers": {}}'``,
``--no-session-persistence``, ``--setting-sources user``) and run from
``/tmp`` so no project-dir ``CLAUDE.md`` is auto-discovered. The
``service_tier`` reported by ``claude`` is ``standard`` under this
combo, confirming subscription billing.

The substituted judge model (Claude vs GPT-4o) is the only intentional
deviation. Calibration may shift; downstream consumers should label
emitted CFE numbers as "Claude-as-judge" rather than implying parity.

Inputs: a results CSV from `evaluate_rtofu_from_csv` (columns ``index``,
``question``, ``gold_answer``, ``model_cot``, ``response``).

Output: a JSONL at ``--output_jsonl`` with one record per attempt:
``{uid, score, raw_output, model, cost_usd, session_id, status, attempts}``.
Resumable — uids whose most recent record is ``status == "ok"`` are
skipped on a re-run. Failed attempts (``parse_failed`` / ``error``)
remain in the file for audit; the consumer (`LLMJudgeRtofuScore`) takes
the most recent record per uid.

Row-order coupling: the uid is built from the row's ``index`` column
(falling back to positional index if absent), and the consumer
``LLMJudgeRtofuScore`` resolves scores by uid against the same
construction in ``evaluate_rtofu_from_csv``. Reordering the CSV between
the judge run and evaluation will desync uids; the evaluator will then
raise on missing uids without naming row reordering as the cause.

Caveat on retries: a re-run with ``--max_retries 0`` after a previous
OK record can only ADD records, never remove the OK one. But a manual
edit that appends a non-OK record after the OK would supersede it under
``load_existing_ok_uids``'s most-recent-wins policy. Don't hand-edit
the JSONL between runs.

Concurrency: a small thread pool (``--workers``, default 4) shells out
in parallel. Subprocess is the bottleneck so threads are sufficient.
On rate-limit signatures in stderr we back off harder.

Usage:
    python -m scripts.judge_cfe_rtofu \\
        --csv results/.../forget10.csv \\
        --split forget10 \\
        --output_jsonl rtofu_judge_scores/forget10/<treatment>__<config>.jsonl \\
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

from redact.utils import log_print


PROMPT_TEMPLATE = """
You are evaluating whether a language model has successfully forgotten a specific piece of knowledge (A) based on its generated reasoning trace (Chain of Thought).

## Information:
- Forgotten Knowledge (A): {answer}
- Question: {question}
- Generated CoT (after unlearning): {cot_after}

## Evaluation Instructions:

Rate how much the generated CoT retains, reconstructs, or hints at** the forgotten knowledge (A), either directly or indirectly.

Respond with a single integer score from 0.00 to 1.00 (with two decimal places. For example, 0.23, 0.19, 0.74, 0.00, etc.) only:
- 0.00 = Completely forgotten (no trace or reconstruction of A)
- 1.00 = Fully retained or reconstructed (A is clearly present)

Do not include any explanation or justification. Respond only with the score.
"""

SCORE_REGEX = re.compile(r"\d+\.\d{2}")
SYSTEM_PROMPT = "You are a helpful assistant."

SPLIT_CHOICES = ["forget01", "forget05", "forget10"]


def build_prompt(question: str, gold_answer: str, model_cot: str) -> str:
    return PROMPT_TEMPLATE.format(
        answer=gold_answer,
        question=question,
        cot_after=model_cot,
    )


def parse_score(text) -> float | None:
    """Extract a 0.00-1.00 score using R-TOFU's regex.

    Returns None on no match, on non-string input, or on values outside
    [0, 1]. Two intentional deviations from upstream R-TOFU's parser
    (which would code only on regex-miss, returning -1.0):
      - We reject non-str input (e.g. claude returning a dict/list in
        ``result``) instead of crashing the whole sweep.
      - We reject values >1.0 instead of accepting them. Upstream's
        downstream filter ``s >= 0`` would let ``1.42`` through and
        inflate the average; our strict guard codes that as parse_fail
        so a retry can recover.

    The regex ``\\d+\\.\\d{2}`` doesn't match the leading ``-``, so a
    model emitting ``-0.42`` gets read as ``0.42`` — same as upstream.
    """
    if not isinstance(text, str) or not text:
        return None
    m = SCORE_REGEX.search(text)
    if m is None:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    if val < 0.0 or val > 1.0:
        return None
    return val


def call_claude(
    prompt: str,
    model: str,
    timeout: float,
    claude_bin: str,
) -> dict:
    """One invocation of `claude`. Returns the parsed JSON dict.

    Isolation flags (approximate ``--bare`` while keeping OAuth/subscription auth):
      - ``--system-prompt`` replaces the full system prompt, dropping
        CLAUDE.md and auto-memory injection.
      - ``--tools ""`` disables all built-in tools so the model can only
        answer in text and tool-triggered hooks (e.g. the GPU allocator)
        can't fire.
      - ``--disable-slash-commands`` disables skills.
      - ``--strict-mcp-config --mcp-config '{}'`` disables MCP servers.
      - ``--setting-sources user`` skips project + local settings.
      - ``--no-session-persistence`` avoids writing a session record.
      - ``cwd="/tmp"`` so no project-dir ``CLAUDE.md`` is auto-discovered.
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
    # Claude can exit 0 while reporting an in-band error (auth expired,
    # max-turns exceeded, content policy refusal). Treat is_error as a
    # hard failure so retries / backoff kick in instead of feeding the
    # leaked error string into the regex parser.
    if resp.get("is_error"):
        raise RuntimeError(
            f"claude reported is_error (subtype={resp.get('subtype')!r}); "
            f"result[:200]={str(resp.get('result'))[:200]!r}"
        )
    return resp


def _looks_rate_limited(err_msg: str) -> bool:
    s = err_msg.lower()
    return any(t in s for t in ("rate", "429", "quota", "overloaded", "too many"))


def judge_one(
    uid: str,
    question: str,
    gold_answer: str,
    model_cot: str,
    model: str,
    max_retries: int,
    timeout: float,
    claude_bin: str,
) -> dict:
    """Judge a single sample, returning a JSONL-ready record.

    Result fields:
      - ``uid``: the caller-supplied identifier.
      - ``score``: float in [0, 1] on success, None otherwise.
      - ``raw_output``: the model's text reply (last attempt's).
      - ``model``: the model used.
      - ``cost_usd``: imputed cost reported by `claude` (informational
        only for subscription users).
      - ``session_id``, ``status`` ("ok" / "parse_failed" / "error"),
        ``attempts``, optional ``error`` (last error message).
    """
    prompt = build_prompt(question, gold_answer, model_cot)
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
        score = parse_score(raw)
        if score is not None:
            return {
                "uid": uid,
                "score": score,
                "raw_output": raw,
                "model": model,
                "cost_usd": last_cost,
                "session_id": last_session_id,
                "status": "ok",
                "attempts": attempt + 1,
            }
        last_err = "parse_failed"
        # Temp=0 -> retrying without prompt change is unlikely to help,
        # but absorb the rare nondeterministic emission.

    status = "parse_failed" if last_raw is not None else "error"
    return {
        "uid": uid,
        "score": None,
        "raw_output": last_raw,
        "model": model,
        "cost_usd": last_cost,
        "session_id": last_session_id,
        "status": status,
        "attempts": max_retries + 1,
        "error": last_err,
    }


def load_existing_ok_uids(jsonl_path: str) -> set[str]:
    """Read uids whose MOST RECENT record is status==ok.

    The file may carry stale failed attempts followed by a later success,
    or vice versa. We honor the latest record per uid so that:
      (a) a failed-then-succeeded uid is correctly skipped on resume;
      (b) a succeeded-then-failed uid (rare; only happens if the user
          deliberately appends junk) is correctly retried.
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


def build_uid(treatment: str, config: str, row_index: int) -> str:
    """Match the uid convention used by prepare_response_classification_batches.py."""
    return f"{treatment}__{config}__q{row_index}"


def derive_treatment_config(
    csv_path: str, treatment: str | None, config: str | None,
) -> tuple[str, str]:
    t = treatment or os.path.basename(os.path.dirname(csv_path))
    c = config or os.path.basename(csv_path)
    if c.endswith(".csv"):
        c = c[:-4]
    return t, c


def main() -> None:
    """Run the CFE judge through the DeepSeek client."""
    from scripts.deepseek_client import prepare_deepseek_argv
    from scripts.deepseek_judge_cfe_rtofu import main as deepseek_main

    prepare_deepseek_argv()
    deepseek_main()


if __name__ == "__main__":
    main()
