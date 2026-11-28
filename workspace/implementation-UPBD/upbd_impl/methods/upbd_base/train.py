"""GPU/runtime adapter for the UPBD reference control."""
from __future__ import annotations

from upbd_impl.methods.upbd_base.core import UPBDBase


def build_method() -> UPBDBase:
    return UPBDBase()


def main(argv=None) -> int:
    import sys
    from upbd_impl.cli import main as cli_main
    return cli_main(["train", "--method", "upbd_base", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
