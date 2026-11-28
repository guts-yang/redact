"""Runtime adapter for CROWN request composition and repair."""
from __future__ import annotations

from upbd_impl.methods.crown.core import CROWN


def build_method() -> CROWN:
    return CROWN()


def main(argv=None) -> int:
    import sys
    from upbd_impl.cli import main as cli_main
    return cli_main(["train", "--method", "crown", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
