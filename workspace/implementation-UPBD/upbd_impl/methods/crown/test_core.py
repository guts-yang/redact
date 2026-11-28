"""CPU invariants for CROWN request composition and concessions."""
import unittest

from upbd_impl.contracts import (
    Candidate, DecisionKind, EvidenceBatch, PrepareContext, RequestSpec,
)
from upbd_impl.methods.crown.core import (
    CROWN, compose_requests, find_conflicts, minimum_concession, normalize_request,
)


class CROWNTest(unittest.TestCase):
    def test_normalization_is_idempotent(self):
        request = RequestSpec("r", forget=("alias", "x"), protect=("y",))
        aliases = {"alias": "x"}
        once = normalize_request(request, aliases)
        self.assertEqual(once.forget, ("x",))
        self.assertEqual(normalize_request(once, aliases), once)

    def test_spec_composition_is_order_independent(self):
        first = RequestSpec("r1", forget=("x",), protect=("a",))
        second = RequestSpec("r2", forget=("y",), protect=("b",))
        self.assertEqual(compose_requests([first, second]), compose_requests([second, first]))

    def test_conflicts_and_minimal_concession(self):
        first = RequestSpec("r1", forget=("x",), protect=())
        second = RequestSpec("r2", forget=(), protect=("x",))
        self.assertEqual(find_conflicts([first, second]), [("r1", "r2", "x")])
        solution = minimum_concession({"r1": 0.2, "r2": -0.1}, {"r1": 2.0, "r2": 1.0})
        self.assertTrue(solution.feasible)
        self.assertEqual(solution.concessions, {"r1": 0.2, "r2": 0.0})
        self.assertAlmostEqual(solution.cost, 0.4)

    def test_conflict_requires_explicit_concession(self):
        request = RequestSpec("r1", forget=("x",), protect=("x",))
        method = CROWN()
        state = method.prepare(PrepareContext(requests=(request,)))
        evidence = EvidenceBatch(0.0, 1.0, 0.0, 1.0, n=1)
        decision = method.accept(Candidate("test", 0, 1001), evidence, state)
        self.assertEqual(decision.kind, DecisionKind.INFEASIBLE)


if __name__ == "__main__":
    unittest.main()
