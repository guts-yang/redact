"""Filter a resolution JSON by claim_idx range to derive a subset split's
resolutions from a larger split's resolution pass.

The R-TOFU forget splits are nested at task_id=1:
  forget01 claims = forget10 baseline claims[:50]
  forget05 claims = forget10 baseline claims[:248]
so forget10's resolution output (claim_idx-keyed) can be filtered to a
prefix range and re-used directly as the resolutions for forget05 or
forget01 — no new subagent pass needed.

Usage:
    # Derive forget01's resolutions from forget10's (50-claim baseline KB)
    python -m scripts.slice_kb_resolutions \\
        --input diagnostics/forget10_resolution_output.json \\
        --max_claim_idx 50 \\
        --output diagnostics/forget01_resolution_output.json

    # Derive forget05's resolutions (248-claim baseline KB)
    python -m scripts.slice_kb_resolutions \\
        --input diagnostics/forget10_resolution_output.json \\
        --max_claim_idx 248 \\
        --output diagnostics/forget05_resolution_output.json
"""
import argparse
import json
import os

from redact.utils import log_print


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Source resolution JSON (e.g. forget10's)")
    parser.add_argument("--max_claim_idx", type=int, required=True,
                        help="Keep resolutions with claim_idx < this value. "
                             "Equal to the baseline KB size of the target split.")
    parser.add_argument("--output", type=str, required=True,
                        help="Destination JSON path")
    args = parser.parse_args()

    log_print(f"Loading resolutions from {args.input}")
    with open(args.input) as f:
        data = json.load(f)
    resolutions = data["resolutions"]
    log_print(f"Source: {len(resolutions)} resolutions")

    kept = [r for r in resolutions if r["claim_idx"] < args.max_claim_idx]
    changed_kept = sum(1 for r in kept if r["original_claim"] != r["resolved_claim"])
    log_print(
        f"Kept {len(kept)} resolutions with claim_idx < {args.max_claim_idx} "
        f"({changed_kept} of them actually change the claim text)"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out = dict(data)
    out["resolutions"] = kept
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log_print(f"Wrote sliced resolutions to {args.output}")


if __name__ == "__main__":
    main()
