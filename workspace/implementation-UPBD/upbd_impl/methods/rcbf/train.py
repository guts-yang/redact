"""Runtime adapter for RCBF policy/barrier co-adaptation."""
from __future__ import annotations

from upbd_impl.methods.rcbf.core import RCBF


def build_method() -> RCBF:
    return RCBF()


def main(argv=None) -> int:
    import sys
    from upbd_impl.cli import main as cli_main
    return cli_main(["train", "--method", "rcbf", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
