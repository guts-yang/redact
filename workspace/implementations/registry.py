"""Registry mapping IDEA ids/slugs to decoupled implementations."""
from __future__ import annotations

from typing import Dict, Type

from .bound import BOUND
from .cfa import CFA
from .csda import CSDA
from .disco import DISCO
from .eas import EAS
from .ortho import ORTHO
from .ripple import RIPPLE
from .rrbu import RRBU
from .scope import SCOPE
from .sequ import SEQU

METHODS: Dict[str, Dict[str, object]] = {
    "IDEA_01_CSDA": {"rank": 1, "slug": "csda", "cls": CSDA},
    "IDEA_02_EAS": {"rank": 2, "slug": "eas", "cls": EAS},
    "IDEA_03_RRBU": {"rank": 3, "slug": "rrbu", "cls": RRBU},
    "IDEA_04_DISCO": {"rank": 4, "slug": "disco", "cls": DISCO},
    "IDEA_05_ORTHO": {"rank": 5, "slug": "ortho", "cls": ORTHO},
    "IDEA_06_SCOPE": {"rank": 6, "slug": "scope", "cls": SCOPE},
    "IDEA_07_SEQU": {"rank": 7, "slug": "sequ", "cls": SEQU},
    "IDEA_08_CFA": {"rank": 8, "slug": "cfa", "cls": CFA},
    "IDEA_09_RIPPLE": {"rank": 9, "slug": "ripple", "cls": RIPPLE},
    "IDEA_10_BOUND": {"rank": 10, "slug": "bound", "cls": BOUND},
}


def get_method(idea_id: str) -> Type:
    if idea_id not in METHODS:
        raise KeyError(f"unknown idea id {idea_id!r}; have {sorted(METHODS)}")
    return METHODS[idea_id]["cls"]  # type: ignore[return-value]


def ordered_methods():
    return [METHODS[k] for k in sorted(METHODS, key=lambda k: METHODS[k]["rank"])]
