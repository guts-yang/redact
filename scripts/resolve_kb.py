"""Apply pronoun resolutions and append birth claims → resolved KB.

Takes a baseline KB (from ``build_kb.py``), applies resolution edits
from the subagent pronoun-resolution pass, appends per-split birth-fact
claims, and re-embeds everything.  The output is the resolved KB
(claims + embeddings) — the bank that ships as
``knowledge_banks/<split>/``.  The claims file is meant to be eyeballed
manually before the KB is used to build any label data.

Position-preserving by default: baseline slot indices 0..N-1 keep their
text (or the resolution-edited text at the same slot); birth claims are
appended at positions N..N+k-1.  Pass ``--dedupe`` to collapse exact-
match duplicates that appear *after* resolution; this shifts claim
indices and writes ``dedup_remap.json`` alongside.

Usage:
    python -m scripts.resolve_kb \\
        --baseline_kb knowledge_banks/forget10_baseline \\
        --resolutions diagnostics/forget10_resolution_output.json \\
        --birth_claims config/birth_claims/forget10.json \\
        --output knowledge_banks/forget10
"""
import argparse
import json
import os
import sys


from redact.regen.leak_detector import (
    dedupe_claims,
    load_knowledge_bank,
    save_knowledge_bank,
)
from redact.utils import log_print


def apply_resolutions(
    baseline_claims: list[str], resolutions: list[dict]
) -> tuple[list[str], int]:
    """Swap resolved text into baseline claim list at matching ``claim_idx``.

    Returns
    -------
    resolved_claims : list[str]
        Same length as ``baseline_claims``; unchanged slots hold the
        original claim text.
    n_applied : int
        Number of slots actually modified.
    """
    claims = list(baseline_claims)
    n_applied = 0
    for r in resolutions:
        idx = r["claim_idx"]
        if not (0 <= idx < len(claims)):
            raise ValueError(f"claim_idx {idx} out of range [0, {len(claims)})")
        if claims[idx] != r["original_claim"]:
            raise ValueError(
                f"Mismatch at claim_idx {idx}: baseline does not match "
                f"resolution's original_claim (subagent output may be stale)"
            )
        if r["resolved_claim"] != r["original_claim"]:
            claims[idx] = r["resolved_claim"]
            n_applied += 1
    return claims, n_applied


def append_birth_claims(
    claims: list[str], birth_claims: list[str]
) -> tuple[list[str], int, int]:
    """Append birth-fact claims, skipping any already present.

    Returns
    -------
    claims : list[str]
        Updated claim list with birth claims appended at the end.
    n_appended : int
        Number of birth claims actually added.
    n_skipped : int
        Number of birth claims skipped because they were already present.
    """
    existing = set(claims)
    new = [c for c in birth_claims if c not in existing]
    claims.extend(new)
    return claims, len(new), len(birth_claims) - len(new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_kb", type=str, required=True,
                        help="Baseline KB directory produced by build_kb.py")
    parser.add_argument("--resolutions", type=str, required=True,
                        help="Path to resolution output JSON")
    parser.add_argument("--birth_claims", type=str, required=True,
                        help="Path to a birth-claims config JSON "
                             "(e.g. config/birth_claims/forget10.json). "
                             "Pass a JSON with an empty 'claims' list if "
                             "no birth claims are needed for the split.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write resolved KB directory "
                             "(claims.json)")
    parser.add_argument("--dedupe", action="store_true",
                        help="Collapse exact-match duplicates after "
                             "ambiguity resolution.  WARNING: shifts claim "
                             "indices — any classifier training data or "
                             "label file keyed to baseline positions must "
                             "be remapped via the written dedup_remap.json.")
    args = parser.parse_args()

    # Load baseline
    log_print(f"Loading baseline KB from {args.baseline_kb}")
    baseline_claims = load_knowledge_bank(args.baseline_kb)
    log_print(f"Baseline: {len(baseline_claims)} claims")

    # Load resolutions
    log_print(f"Loading resolutions from {args.resolutions}")
    with open(args.resolutions) as f:
        data = json.load(f)
    resolutions = data["resolutions"]
    log_print(f"Resolutions: {len(resolutions)} claims flagged ambiguous")

    # Apply resolutions position-preservingly
    resolved_claims, n_applied = apply_resolutions(baseline_claims, resolutions)
    log_print(f"Applied {n_applied} changed resolutions "
              f"({len(resolutions) - n_applied} were unchanged/passthrough)")
    assert len(resolved_claims) == len(baseline_claims), \
        "position-preserving invariant violated"

    # Optionally dedupe post-resolution (collapses duplicates introduced by
    # ambiguity resolution, e.g. two pronoun variants resolving to the same
    # canonical form).  This shifts claim indices.
    remap_path = os.path.join(args.output, "dedup_remap.json")
    dedupe_info: dict | None = None
    dedupe_warning: str | None = None
    if args.dedupe:
        deduped_claims, old_to_new = dedupe_claims(resolved_claims)
        n_removed = len(resolved_claims) - len(deduped_claims)
        log_print(f"Dedupe: {len(resolved_claims)} -> {len(deduped_claims)} "
                  f"({n_removed} removed)")
        dedupe_info = {
            "old_to_new": old_to_new,
            "n_removed": n_removed,
            "n_before": len(resolved_claims),
            "n_after": len(deduped_claims),
            # n_birth_appended and birth_start_idx are filled in after the
            # birth-claim append step so consumers can reason about the final
            # KB layout (dedup survivors at [0, n_after), birth claims at
            # [n_after, n_after + n_birth_appended)).
            "n_birth_appended": None,
            "birth_start_idx": None,
        }
        if n_removed > 0:
            dedupe_warning = (
                f"WARNING: dedupe removed {n_removed} baseline claims — claim "
                f"indices have shifted.  Any classifier training data or label "
                f"file keyed to baseline positions must be remapped using "
                f"{remap_path} before use with this resolved KB.  The remap "
                f"covers baseline indices [0, n_before); birth claims appended "
                f"after dedup live at [n_after, n_after + n_birth_appended) "
                f"and are NOT in old_to_new."
            )
            print(dedupe_warning, file=sys.stderr, flush=True)
        resolved_claims = deduped_claims
    else:
        if os.path.exists(remap_path):
            os.remove(remap_path)
            log_print(f"Removed stale {remap_path} (no --dedupe this run)")

    # Append birth claims
    log_print(f"Loading birth claims from {args.birth_claims}")
    with open(args.birth_claims) as f:
        birth_config = json.load(f)
    birth_list = birth_config.get("claims", [])
    log_print(f"Birth claims in config: {len(birth_list)}")
    n_orig = len(resolved_claims)
    resolved_claims, n_appended, n_skipped = append_birth_claims(
        resolved_claims, birth_list
    )
    if n_skipped:
        log_print(f"Skipped {n_skipped} birth claims already present in KB")
    if n_appended:
        log_print(
            f"Appended {n_appended} birth claims at indices "
            f"{n_orig}..{len(resolved_claims) - 1}"
        )
    else:
        log_print("No birth claims appended.")

    # Finalize and persist the dedupe remap now that birth-claim positions
    # are known.
    if dedupe_info is not None:
        dedupe_info["n_birth_appended"] = n_appended
        dedupe_info["birth_start_idx"] = n_orig
        os.makedirs(args.output, exist_ok=True)
        with open(remap_path, "w") as f:
            json.dump(dedupe_info, f, indent=2)
        log_print(f"Saved dedup remap to {remap_path}")

    # Save. This KB is meant to be eyeballed manually before it is used to
    # build any label data.
    save_knowledge_bank(resolved_claims, args.output)
    log_print(f"Saved {len(resolved_claims)} resolved claims to {args.output}")

    # Re-emit dedupe warning after the encode progress bar so it isn't lost.
    if dedupe_warning is not None:
        print(dedupe_warning, file=sys.stderr, flush=True)

    log_print(f"Review {args.output}/claims.json manually before use.")


if __name__ == "__main__":
    main()
