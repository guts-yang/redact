"""Runtime adapter for STREAM streaming primal-dual repair."""
from __future__ import annotations

from upbd_impl.methods.stream.core import STREAM


def build_method() -> STREAM:
    return STREAM()


def main(argv=None) -> int:
    import sys
    from upbd_impl.cli import main as cli_main
    return cli_main(["train", "--method", "stream", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
