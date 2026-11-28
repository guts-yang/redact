"""Constructors for explicit method decisions."""
from __future__ import annotations

from upbd_impl.contracts import Decision, DecisionKind


def accept(reason: str = "accepted", *scope: str) -> Decision:
    return Decision(DecisionKind.ACCEPT, reason, tuple(scope) or ("calibration",))


def rollback(reason: str, *scope: str) -> Decision:
    return Decision(DecisionKind.ROLLBACK, reason, tuple(scope) or ("calibration",))


def infeasible(reason: str, *scope: str) -> Decision:
    return Decision(DecisionKind.INFEASIBLE, reason, tuple(scope))


def unknown(reason: str, *scope: str) -> Decision:
    return Decision(DecisionKind.UNKNOWN, reason, tuple(scope))
