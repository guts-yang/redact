"""Deterministic entity-disjoint probe partitioning without I/O."""
from __future__ import annotations

import hashlib
from collections import defaultdict

from upbd_impl.contracts import Probe, ProbeSet


def _entity_rank(entity_key: str, seed: int) -> tuple[int, str]:
    digest = hashlib.sha256(f"{seed}:{entity_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big"), entity_key


def split_probes(probes: list[Probe] | tuple[Probe, ...], seed: int = 1001) -> ProbeSet:
    """Split entities deterministically into train/calibration/test partitions.

    Entities are assigned in ranked order with a 60/20/20 target. For three or
    more entities, every partition receives at least one entity. Fewer than
    three entities cannot support the full protocol and are rejected.
    """
    grouped: dict[str, list[Probe]] = defaultdict(list)
    for probe in probes:
        if not probe.entity_key:
            raise ValueError("every probe needs an entity_key")
        grouped[probe.entity_key].append(probe)
    entities = sorted(grouped)
    if len(entities) < 3:
        raise ValueError("entity-disjoint train/calibration/test requires >=3 entities")
    entities.sort(key=lambda key: _entity_rank(key, seed))
    n = len(entities)
    n_test = max(1, round(n * 0.2))
    n_calibration = max(1, round(n * 0.2))
    if n_test + n_calibration >= n:
        n_test = n_calibration = 1
    test = entities[:n_test]
    calibration = entities[n_test:n_test + n_calibration]
    train = entities[n_test + n_calibration:]

    def take(keys: list[str]) -> tuple[Probe, ...]:
        return tuple(probe for key in keys for probe in grouped[key])

    result = ProbeSet(train=take(train), calibration=take(calibration), test=take(test))
    assert not ({p.entity_key for p in result.train}
                & {p.entity_key for p in result.calibration})
    assert not ({p.entity_key for p in result.train}
                & {p.entity_key for p in result.test})
    assert not ({p.entity_key for p in result.calibration}
                & {p.entity_key for p in result.test})
    return result
