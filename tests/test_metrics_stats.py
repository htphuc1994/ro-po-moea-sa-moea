import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics  # noqa: E402
import stats  # noqa: E402


def test_hypervolume_matches_exact_box_volumes():
    hv = metrics.Hypervolume(dim=5, seed=1)
    one = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
    assert hv(one) == pytest.approx(0.6 ** 5, rel=0.01)
    # two boxes: union volume by inclusion-exclusion
    two = np.array([[0.2, 0.6, 0.6, 0.6, 0.6], [0.6, 0.2, 0.6, 0.6, 0.6]])
    exact = 2 * (0.9 * 0.5 ** 4) - 0.5 ** 5
    assert hv(two) == pytest.approx(exact, rel=0.01)
    assert hv(np.array([[1.2, 0, 0, 0, 0]])) == 0.0


def test_hypervolume_is_monotone_in_the_archive():
    hv = metrics.Hypervolume(dim=5, seed=2)
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, size=(10, 5))
    assert hv(np.vstack([A, rng.uniform(0, 1, size=(5, 5))])) >= hv(A)


def test_igd_plus_zero_on_the_front_and_positive_off_it():
    F = np.array([[0.0, 1.0, 0, 0, 0], [1.0, 0.0, 0, 0, 0]])
    assert metrics.igd_plus(F, F) == 0.0
    assert metrics.igd_plus(F + 0.1, F) > 0.0


def test_spacing_zero_for_evenly_spaced_points():
    Z = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0]], dtype=float)
    assert metrics.spacing(Z) == pytest.approx(0.0)


def test_holm_is_monotone_and_step_down():
    p = [0.01, 0.04, 0.03, 0.005, 0.5]
    adj, rej = stats.holm(p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)
    assert adj[3] == pytest.approx(0.025)        # smallest p times 5
    # identical raw p-values get identical adjusted values
    adj2, _ = stats.holm([1e-9, 1e-9, 1e-9])
    assert len(set(np.round(adj2, 15))) == 1
    assert rej.tolist() == [True, False, False, True, False]


def test_bootstrap_ci_contains_mean():
    m, lo, hi = stats.bootstrap_ci(np.arange(30.0))
    assert lo < m < hi


def test_copies_change_neither_spacing_nor_coverage_nor_the_reference_front():
    """Final sets and the reference front are sets: duplicated objective vectors count once."""
    rng = np.random.default_rng(3)
    A, B = rng.random((12, 5)), rng.random((9, 5))
    A2, B2 = np.vstack([A, A[:5], A[:2]]), np.vstack([B, B, B[:3]])
    assert metrics.spacing(metrics.distinct(A2)) == pytest.approx(metrics.spacing(A))
    assert metrics.coverage(metrics.distinct(A2), metrics.distinct(B2)) == pytest.approx(metrics.coverage(A, B))
    front = metrics.Normalizer([A2, A2, B2]).front
    assert len(np.unique(front, axis=0)) == len(front)
