"""Runtime adapter for TCR; algorithmic state lives only in core.TCR."""
from __future__ import annotations

from upbd_impl.methods.tcr.core import TCR


def build_method() -> TCR:
    return TCR()


def main(argv=None) -> int:
    import sys
    from upbd_impl.cli import main as cli_main
    return cli_main(["train", "--method", "tcr", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
