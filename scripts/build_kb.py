"""Build a baseline knowledge bank from R-TOFU forget-set gold answers.

Extracts factual claims (sentence-split gold answers) and saves them as
``claims.json``. This is the BASELINE bank — the canonical flow
applies resolutions and birth claims (``resolve_kb.py``) to produce the
resolved bank that ships as ``knowledge_banks/<split>/``.

Usage:
    python -m scripts.build_kb --split forget10
    # Output: knowledge_banks/forget10/ ; rename to knowledge_banks/forget10_baseline/
    # so the canonical knowledge_banks/forget10/ slot can hold the resolved KB.
"""
import argparse


from redact.regen.leak_detector import build_claims, save_knowledge_bank
from redact.dataset.rtofu import RTOFU
from redact.utils import log_print

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="knowledge_banks",
    help="Baseline KB directory. The script writes to "
         "{output_dir}/{split}; rename to {output_dir}/{split}_baseline "
         "so the canonical {output_dir}/{split} slot is reserved for the "
         "resolved bank produced by resolve_kb.",
)
args = parser.parse_args()

# Load dataset
log_print(f"Loading R-TOFU split: {args.split}")
rtofu = RTOFU()
rtofu.download()
dataset = rtofu.dataset[args.split]

# Extract claims from gold answers
answers = dataset["answer"]
claims = build_claims(answers)
log_print(
    f"Extracted {len(claims)} claims from {len(answers)} answers "
    f"(duplicates preserved; apply dedupe downstream if desired)"
)

# Save
output_dir = f"{args.output_dir}/{args.split}"
save_knowledge_bank(claims, output_dir)
log_print(f"Baseline knowledge bank saved to {output_dir}")
log_print(f"Next: run scripts.resolve_kb on this baseline.")
