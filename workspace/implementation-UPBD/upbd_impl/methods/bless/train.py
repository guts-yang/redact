"""Runtime adapter for BLESS; benign operators enter via PrepareContext."""
from __future__ import annotations

from upbd_impl.methods.bless.core import BLESS


def build_method() -> BLESS:
    return BLESS()


def main(argv=None) -> int:
    import sys
    from upbd_impl.cli import main as cli_main
    return cli_main(["train", "--method", "bless", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
