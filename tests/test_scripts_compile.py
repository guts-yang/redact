"""Parse-smoke for every shipped script.

The functional suite imports the underlying ``redact`` modules but NOT the thin
``scripts/*.py`` CLI wrappers (and four of them run ``argparse`` at module
import, so an import-based test would ``SystemExit``). A syntax error in a
wrapper therefore slips past a green run. This AST parse catches that class of
break across all wrappers regardless of a ``__main__`` guard.
"""

import ast
import pathlib

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
_SCRIPTS = sorted(_SCRIPTS_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "path", _SCRIPTS, ids=lambda p: str(p.relative_to(_SCRIPTS_DIR))
)
def test_script_parses(path):
    """Every script must parse (raises SyntaxError -> test fails for that file)."""
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
