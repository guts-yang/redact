"""Per-CoT first-leak labeler for the R-TOFU forget-set leak classifier.

Renders `docs/labeling_prompts/first_leak_per_cot_v1.md` per CoT and shells out
to the `claude` CLI under the same subscription-billing isolation flags as
`scripts/judge_cfe_rtofu.py` (no tools, generic system prompt, no
slash-commands, no MCPs, no project-dir CLAUDE.md). The judge sees only the
rendered prompt + substituted inputs; output is a single JSON object parsed
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
        --model sonnet --workers 4
"""
import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    parser.add_argument(
        "--csv", default=None,
        help="Forget-split response CSV with columns index/question/model_cot. "
             "Default: leak_detector_data_promptfmt/<split>/forget_responses.csv",
    )
    parser.add_argument(
        "--output_jsonl", default=None,
        help="Destination JSONL. Resumable: uids with status=ok on their most "
             "recent record are skipped. "
             "Default: leak_detector_data_promptfmt/<split>/first_leak_labels.jsonl",
    )
    parser.add_argument("--model", default="sonnet",
                        help="Claude model alias or full ID. Default: sonnet.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent claude invocations. Default 4.")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Retries per sample (default 2 = up to 3 attempts).")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Per-call timeout in seconds. Default 180.")
    parser.add_argument("--claude_bin", default=None,
                        help="Path to the `claude` binary. Default: which('claude').")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on samples (for smoke-tests).")
    args = parser.parse_args()

    split = args.split
    csv_path = Path(args.csv) if args.csv else (
        REPO / f"leak_detector_data_promptfmt/{split}/forget_responses.csv"
    )
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else (
        REPO / f"leak_detector_data_promptfmt/{split}/first_leak_labels.jsonl"
    )

    claude_bin = args.claude_bin or shutil.which("claude")
    if claude_bin is None:
        sys.exit("Could not find `claude` on PATH. Pass --claude_bin.")
    log_print(f"Using claude binary: {claude_bin}")
    log_print(f"Model: {args.model} | workers: {args.workers} | "
              f"max_retries: {args.max_retries} | timeout: {args.timeout}s")

    template = load_template()

    with open(REPO / f"config/authors/{split}.json") as f:
        forget_data = json.load(f)
    kb_path = REPO / f"knowledge_banks/{split}/claims.json"
    with open(kb_path) as f:
        all_claims = json.load(f)
    n_claims = len(all_claims)

    # Pre-render each author's indexed claim block once so we don't re-render
    # for every row of that author.
    author_claims_by_idx: dict[int, str] = {}
    author_name_by_idx: dict[int, str] = {}
    for author_idx, author in enumerate(forget_data["authors"]):
        author_claims_by_idx[author_idx] = render_author_claims_indexed(
            author["claim_ranges"], all_claims,
        )
        author_name_by_idx[author_idx] = author["name"]

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"CSV is empty: {csv_path}")
    missing = REQUIRED_CSV_COLS - set(rows[0].keys())
    if missing:
        sys.exit(f"CSV missing required columns {sorted(missing)}: {csv_path}")
    log_print(f"Loaded {len(rows)} rows from {csv_path}")

    work: list[dict] = []
    for row in rows:
        row_index = int(row["index"])
        author_idx = row_index // 20
        if author_idx not in author_name_by_idx:
            log_print(
                f"  WARN: row_index={row_index} maps to author_idx={author_idx} "
                f"which is not in config/authors/{split}.json (max="
                f"{max(author_name_by_idx)}); skipping"
            )
            continue
        cot = (row.get("model_cot") or "").strip()
        if not cot:
            # Empty CoT — emit a synthetic ok=null record so the row isn't
            # silently dropped from downstream joins.
            work.append({
                "uid": f"{split}_q{row_index:03d}",
                "row_index": row_index,
                "author": author_name_by_idx[author_idx],
                "empty_cot": True,
            })
            continue
        cot_sents = split_sentences(cot)
        if not cot_sents:
            work.append({
                "uid": f"{split}_q{row_index:03d}",
                "row_index": row_index,
                "author": author_name_by_idx[author_idx],
                "empty_cot": True,
            })
            continue
        prompt = build_prompt(
            template=template,
            author_name=author_name_by_idx[author_idx],
            question=clean_question(row["question"]),
            author_claims_indexed=author_claims_by_idx[author_idx],
            cot_indexed=render_cot_indexed(cot_sents),
        )
        work.append({
            "uid": f"{split}_q{row_index:03d}",
            "row_index": row_index,
            "author": author_name_by_idx[author_idx],
            "prompt": prompt,
            "n_sentences": len(cot_sents),
        })

    if args.limit is not None:
        work = work[:args.limit]
        log_print(f"--limit: capped to {len(work)} samples")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = load_existing_ok_uids(str(output_jsonl))
    if done:
        log_print(f"Resuming: {len(done)} uids already scored OK")
    todo = [w for w in work if w["uid"] not in done]

    # Pre-emit empty-CoT records (no judge call needed).
    empty_records = []
    real_todo = []
    for w in todo:
        if w.get("empty_cot"):
            empty_records.append({
                "uid": w["uid"],
                "row_index": w["row_index"],
                "author": w["author"],
                "n_sentences": 0,
                "first_leak_idx": None,
                "matched_claim_idx": None,
                "claim_snippet": None,
                "subject_resolved": None,
                "rationale": "empty CoT — no sentences to label",
                "raw_output": None,
                "model": args.model,
                "cost_usd": 0.0,
                "session_id": None,
                "status": "ok",
                "attempts": 0,
            })
        else:
            real_todo.append(w)

    log_print(
        f"To do: {len(real_todo)} samples to judge; "
        f"{len(empty_records)} empty-CoT rows recorded directly."
    )
    if not real_todo and not empty_records:
        log_print("Nothing to do; exiting.")
        return

    n_ok = 0
    n_validation_fail = 0
    n_error = 0
    total_cost = 0.0

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        for rec in empty_records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
        out_f.flush()

        if real_todo:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        label_one,
                        w["uid"], w["row_index"], w["author"], w["prompt"],
                        w["n_sentences"], n_claims,
                        args.model, args.max_retries, args.timeout, claude_bin,
                    ): w["uid"] for w in real_todo
                }
                for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                    rec = fut.result()
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    total_cost += float(rec.get("cost_usd") or 0.0)
                    if rec["status"] == "ok":
                        n_ok += 1
                    elif rec["status"] == "validation_failed":
                        n_validation_fail += 1
                    else:
                        n_error += 1
                    if (i + 1) % 10 == 0 or (i + 1) == len(real_todo):
                        log_print(
                            f"  [{i + 1}/{len(real_todo)}] ok={n_ok} "
                            f"validation_fail={n_validation_fail} err={n_error} "
                            f"cost~=${total_cost:.4f}"
                        )

    log_print(
        f"Done. {n_ok} labeled / {n_validation_fail} validation-failed / "
        f"{n_error} errored. Imputed cost: ${total_cost:.4f}. "
        f"Output: {output_jsonl}"
    )
    if n_validation_fail or n_error:
        log_print(
            "[note] Re-run the same command to retry validation-failed / "
            "errored rows (only OK uids are skipped on resume)."
        )


if __name__ == "__main__":
    main()
