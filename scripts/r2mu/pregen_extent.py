"""Pregenerate the "extent" (unthink-loss) data from scratch.

R2MU's extent stream needs `{question, cot}` pairs produced by the target model
itself on the forget prompts. For splits where we already have this data from
our leak-detector pipeline (`leak_detector_data/<split>/raw_cots.json`), use
`scripts/r2mu/reformat_raw_cots_to_extent.py` instead — it's zero-compute.

This script is for the from-scratch case. It loads the R-TOFU forget split,
runs `sangyon/LRM-target` (with `ReasoningModel`'s `<think>` prefix handling)
on each forget prompt, and writes a JSONL with `{question, cot}` per line.

Not implemented yet — invoke `scripts.generate_rtofu_responses` for the split
of interest to produce forget + retain CSVs, then adapt them into the raw_cots
layout and run `reformat_raw_cots_to_extent.py`. This stub is a placeholder for
a self-contained reproducibility path.
"""

import sys


def main():
    sys.exit(
        "Not implemented. For splits with existing leak_detector_data/<split>/raw_cots.json, use "
        "scripts/r2mu/reformat_raw_cots_to_extent.py. For a from-scratch pregen, run "
        "`python -m scripts.generate_rtofu_responses --split <split> ...` to produce forget + "
        "retain CSVs, then reformat."
    )


if __name__ == "__main__":
    main()
