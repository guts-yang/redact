"""Runtime adapter for TMOT; reference statistics enter via StepContext."""
from __future__ import annotations

from upbd_impl.methods.tmot.core import TMOT


def build_method() -> TMOT:
    return TMOT()


def main(argv=None) -> int:
    import sys
    from upbd_impl.cli import main as cli_main
    return cli_main(["train", "--method", "tmot", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
