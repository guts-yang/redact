"""Paraphrase IDK refusal answers via Claude CLI for classifier augmentation.

For each ``reasoned_idk_answer`` in
``data/reasoned_idk/reasoned_idk_<split>.json``, generate K paraphrased
variants with distinct surface forms (refusal mode, subject reference,
sentence structure). Output is a JSONL with one record per IDK source,
each containing the K paraphrases.

Downstream: a sister to ``scripts.idk_auto_label`` will emit each
paraphrase as an additional clean_idk training row (label=0,
source='clean_idk_paraphrase').

Refusal-mode bucket history:
  * 2026-05-16: initial release with buckets (a-d) — formal, longer-form
    refusals: (a) first-person epistemic ("I don't know…"), (b)
    impersonal record-absence ("No information is available about…"),
    (c) declined-to-answer ("I cannot provide details on…"), (d)
    scope-disclaimer ("That falls outside what I can answer…").
  * 2026-05-19: replaced with buckets (e-h) — short, conversational,
    matching what the soft-token regen engine actually emits at
    inference. See the PROMPT_TEMPLATE below. The (a-d) artifacts at
    ``paraphrases/<split>.jsonl`` (pre-2026-05-19) remain valid; we
    intentionally keep them and merge the (e-h) output on top so the
    classifier sees both registers.

Auth note: as in ``scripts.judge_cfe_rtofu``, calls go through the
``claude`` CLI in subscription-billing mode (no ``--bare``). The
isolation flags approximate ``--bare``'s sandboxing while keeping
OAuth auth — see ``judge_cfe_rtofu`` for the full rationale.

Resumable: uids whose most recent record is ``status='ok'`` are
skipped on re-run; failed attempts remain in the file for audit.

Usage:
    python -m scripts.paraphrase_idk_answers \\
        --split forget01 \\
        --output_jsonl paraphrases/forget01.jsonl \\
        --k 3 --model sonnet --workers 4
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time

from redact.utils import log_print


PROMPT_TEMPLATE = """You are generating short refusal answers to a question whose answer you don't have. The output expands the surface-form diversity of a leak classifier's training data, so each generation must be structurally distinct from the others.

## Input

Question: {question}

Avoid-list (already collected, do not echo): {answer}

## Task

Produce exactly {K} additional refusals to the question above. Treat the question as unanswerable — you do not know the answer and are not trying to guess.

Requirements:

1. **Genuine refusal**: each response must communicate that you don't have the information needed to answer.

2. **No new information**: do not introduce any factual claim about the subject that isn't already in the question.
   - Forbidden: invented names, dates, places, titles, genres, awards, nationalities, eras, fields of work, attributes, traits, biographical details.
   - Forbidden: gendered pronouns when the question doesn't use one.
   - If the question gives only a name, you may echo the name verbatim or use "this author"/"this person"/"they", but must NOT add any descriptor — even ones that sound generic ("the renowned…", "the prolific…", "the contemporary…", "this specific…").
   - Do not hedge with implicit specificity ("I don't have *detailed* records on this *specific* author's *particular* work").

3. **Distinct refusal modes**: the {K} responses must use **structurally different** refusal modes — no two the same. Pick from:
   - (e) brief direct denial: "Don't know.", "No idea who that is.", "Can't say who [Name] is.", "Nothing to share on [Name]."
   - (f) apologetic deflection: "Sorry, can't help with that.", "Apologies, I don't have that.", "Unfortunately, no.", "Sorry, beyond my knowledge."
   - (g) recall / memory framing: "Doesn't ring a bell.", "Can't recall [Name].", "Drawing a blank on that.", "Nothing comes to mind."
   - (h) hesitant / casual: "Hmm, not sure on that one.", "Honestly, I don't know.", "Not certain who [Name] is.", "Don't think I can answer that one."

4. **Different from the Avoid-list entry**: exclude the Avoid-list entry's refusal mode from your {K} choices. Additionally, none of your {K} responses may reuse the Avoid-list entry's sentence opener (first 3 words) or its main-clause structure.

5. **Subject reference**: vary across the {K} responses — mix "this author" / "this person" / "they" / the literal name (only if it's in the question) / omitting the subject entirely.

6. **Brevity**: each response is exactly 1 sentence, ≤ 15 words (target 3-10). No meta-commentary, no questions back to the user. Brief apology openers ("Sorry,", "Apologies,") are OK in bucket (f) since they're part of that bucket's structural mode.

## Output

A JSON array of exactly {K} strings. The response must start with `[` and end with `]`. No prose before or after, no code fences, no comments.
"""

SYSTEM_PROMPT = "You are a helpful assistant."
SPLIT_CHOICES = ["forget01", "forget05", "forget10"]


def build_prompt(question: str, answer: str, k: int) -> str:
    return PROMPT_TEMPLATE.format(question=question, answer=answer, K=k)


def parse_paraphrases(raw: str, k: int) -> list[str] | None:
    """Extract a JSON array of K strings. Returns None on parse / shape failure.

    Strict-parses ``raw`` first; falls back to extracting the first
    ``[...]`` substring (so a stray leading "Here:" doesn't doom the whole
    record). Rejects: non-list, wrong length, empty strings.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if m is None:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, list) or len(data) != k:
        return None
    if not all(isinstance(x, str) and x.strip() for x in data):
        return None
    return [s.strip() for s in data]


def call_claude(prompt: str, model: str, timeout: float, claude_bin: str) -> dict:
    """One invocation of `claude`. Returns the parsed JSON dict. See
    ``judge_cfe_rtofu.call_claude`` for the isolation-flag rationale."""
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
    if resp.get("is_error"):
        raise RuntimeError(
            f"claude reported is_error (subtype={resp.get('subtype')!r}); "
            f"result[:200]={str(resp.get('result'))[:200]!r}"
        )
    return resp


def _looks_rate_limited(err_msg: str) -> bool:
    s = err_msg.lower()
    return any(t in s for t in ("rate", "429", "quota", "overloaded", "too many"))


def paraphrase_one(
    uid: str,
    question: str,
    answer: str,
    k: int,
    model: str,
    max_retries: int,
    timeout: float,
    claude_bin: str,
) -> dict:
    """Generate K paraphrases for one IDK record. JSONL-ready output dict."""
    prompt = build_prompt(question, answer, k)
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
        paraphrases = parse_paraphrases(raw, k)
        if paraphrases is not None:
            return {
                "uid": uid,
                "paraphrases": paraphrases,
                "raw_output": raw,
                "model": model,
                "cost_usd": last_cost,
                "session_id": last_session_id,
                "status": "ok",
                "attempts": attempt + 1,
            }
        last_err = "parse_failed"

    status = "parse_failed" if last_raw is not None else "error"
    return {
        "uid": uid,
        "paraphrases": None,
        "raw_output": last_raw,
        "model": model,
        "cost_usd": last_cost,
        "session_id": last_session_id,
        "status": status,
        "attempts": max_retries + 1,
        "error": last_err,
    }


def load_existing_ok_uids(jsonl_path: str) -> set[str]:
    """Mirror of ``judge_cfe_rtofu.load_existing_ok_uids``: most-recent-wins
    per uid, return uids whose latest record is status='ok'."""
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


def build_uid(split: str, idx: int) -> str:
    return f"paraphrase_{split}_q{idx:04d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    parser.add_argument(
        "--input_idk", default=None,
        help="Override IDK JSON path. Default: "
             "data/reasoned_idk/reasoned_idk_<split>.json",
    )
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--k", type=int, default=3,
                        help="Paraphrases per record. Default 3.")
    parser.add_argument("--model", default="sonnet",
                        help="Claude model alias or full ID. Default: sonnet.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent claude invocations. Default 4.")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Retries per sample on subprocess error / "
                             "parse-fail. Default 2.")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-call timeout in seconds. Default 120.")
    parser.add_argument("--claude_bin", default=None,
                        help="Path to the `claude` binary. "
                             "Default: shutil.which('claude').")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on records (for smoke tests).")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Optional subset of idx values to process.")
    args = parser.parse_args()

    claude_bin = args.claude_bin or shutil.which("claude")
    if claude_bin is None:
        sys.exit("Could not find `claude` on PATH. Pass --claude_bin.")
    log_print(f"Using claude binary: {claude_bin}")

    input_idk = (
        args.input_idk
        or f"data/reasoned_idk/reasoned_idk_{args.split}.json"
    )
    log_print(f"Loading IDK records from {input_idk}")
    with open(input_idk) as f:
        idk = json.load(f)
    log_print(f"  loaded {len(idk)} records")

    if args.indices is not None:
        idx_set = set(args.indices)
        idk = [r for r in idk if r["idx"] in idx_set]
        log_print(f"  filtered to --indices: {len(idk)}")
    if args.limit is not None:
        idk = idk[:args.limit]
        log_print(f"  limited to --limit: {len(idk)}")

    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    existing = load_existing_ok_uids(args.output_jsonl)
    log_print(f"Existing OK records: {len(existing)} (resuming)")

    to_process = [
        r for r in idk
        if build_uid(args.split, r["idx"]) not in existing
    ]
    log_print(f"Records to process: {len(to_process)}")
    if not to_process:
        log_print("Nothing to do.")
        return

    def _work(r: dict) -> dict:
        uid = build_uid(args.split, r["idx"])
        rec = paraphrase_one(
            uid=uid,
            question=r["question"],
            answer=r["reasoned_idk_answer"],
            k=args.k,
            model=args.model,
            max_retries=args.max_retries,
            timeout=args.timeout,
            claude_bin=claude_bin,
        )
        rec["idx"] = r["idx"]
        rec["split"] = args.split
        rec["question"] = r["question"]
        rec["original_answer"] = r["reasoned_idk_answer"]
        return rec

    n_ok = n_fail = 0
    total_cost = 0.0
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers,
    ) as ex, open(args.output_jsonl, "a") as out_f:
        futures = {ex.submit(_work, r): r for r in to_process}
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if rec["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            total_cost += rec.get("cost_usd", 0.0) or 0.0
            done = n_ok + n_fail
            if done % 10 == 0 or done == len(to_process):
                log_print(
                    f"  progress: {done}/{len(to_process)} "
                    f"(ok={n_ok}, fail={n_fail}, "
                    f"cost~${total_cost:.3f}, "
                    f"elapsed={time.perf_counter() - t0:.1f}s)"
                )

    log_print(
        f"\nDONE. ok={n_ok}, fail={n_fail}, total_cost~${total_cost:.3f}"
    )
    log_print(f"Wrote to {args.output_jsonl}")


if __name__ == "__main__":
    main()
