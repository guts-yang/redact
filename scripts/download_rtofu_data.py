"""Download the vendored R-TOFU JSONL files into ``data/rtofu/``.

The R-TOFU forget/retain/aux splits are fetched from the pinned upstream
commit recorded in ``data/rtofu/PROVENANCE.md`` (the HuggingFace dataset
``sangyon/R-TOFU`` drops the ``task_id`` field we need to reconstruct the
paper's protocol retain splits, so we read the raw github JSONLs instead).

These files are gitignored — run this once after cloning::

    python -m scripts.download_rtofu_data

Re-running is safe: existing files are skipped unless ``--force`` is given.
The pinned commit SHA below MUST match ``data/rtofu/PROVENANCE.md``; if you
bump one, bump the other.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

# Keep in sync with data/rtofu/PROVENANCE.md.
COMMIT_SHA = "df533a5c08dfa7fcd314d6c8714481fec2e24529"
RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ssangyeon/R-TOFU/{sha}/data/tofu/{name}.json"
)

# The eight files redact.dataset.rtofu.RTOFU loads from data/rtofu/.
FILES = [
    "forget01",
    "forget05",
    "forget10",
    "retain10",
    "retain50",
    "retain90",
    "real_authors_perturbed",
    "world_facts_perturbed",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join(REPO_ROOT, "data", "rtofu")


def _download(url: str, dest_path: str) -> int:
    """Fetch ``url`` to ``dest_path``; return the byte count written."""
    req = urllib.request.Request(url, headers={"User-Agent": "redact-rtofu-downloader"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as fh:
        fh.write(data)
    os.replace(tmp_path, dest_path)
    return len(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help="Destination directory (default: <repo>/data/rtofu).",
    )
    parser.add_argument(
        "--sha",
        default=COMMIT_SHA,
        help="Override the pinned upstream commit SHA (default matches PROVENANCE.md).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist.",
    )
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    n_fetched = 0
    n_skipped = 0
    failures = []
    for name in FILES:
        dest_path = os.path.join(args.dest, f"{name}.json")
        if os.path.exists(dest_path) and not args.force:
            print(f"[skip]  {name}.json already present")
            n_skipped += 1
            continue
        url = RAW_URL_TEMPLATE.format(sha=args.sha, name=name)
        try:
            n_bytes = _download(url, dest_path)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"[FAIL]  {name}.json <- {url}\n        {exc}", file=sys.stderr)
            failures.append(name)
            continue
        print(f"[ok]    {name}.json ({n_bytes:,} bytes)")
        n_fetched += 1

    print(
        f"\nDone: {n_fetched} fetched, {n_skipped} skipped, "
        f"{len(failures)} failed (dest={args.dest})."
    )
    if failures:
        print(
            "Failed: "
            + ", ".join(failures)
            + ". Re-run, or fetch manually per data/rtofu/PROVENANCE.md.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
