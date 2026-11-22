"""SPEA2 archive truncation agrees with a direct implementation of its definition
(Zitzler et al. 2001): repeatedly remove the plan whose sorted distance vector to the
remaining plans is lexicographically smallest."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solvers import _truncate  # noqa: E402


def brute_force(dist, keep):
    alive = list(range(len(dist)))
    while len(alive) > keep:
        vec = {i: sorted(dist[i][j] for j in alive if j != i) for i in alive}
        alive.remove(min(alive, key=lambda i: vec[i]))
    return alive


def test_matches_definition_on_random_sets_with_duplicates():
    rng = np.random.default_rng(3)
    for trial in range(40):
        n, keep = int(rng.integers(5, 40)), None
        keep = int(rng.integers(1, n))
        Z = rng.random((n, 5))
        dup = rng.integers(0, n, size=n // 5)
        Z[dup[: len(dup) // 2]] = Z[dup[len(dup) // 2: 2 * (len(dup) // 2)]]   # identical plans
        dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
        assert _truncate(dist, keep).tolist() == brute_force(dist, keep), trial


def test_removes_one_of_two_identical_points_first():
    Z = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    kept = _truncate(dist, 3).tolist()
    assert kept in ([0, 1, 3], [0, 2, 3])
