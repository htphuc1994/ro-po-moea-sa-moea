import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402
import dominance  # noqa: E402

TARGET = dominance.target()


def random_fits(n, seed, feasible_share=0.6):
    rng = np.random.default_rng(seed)
    F = rng.uniform(0, 1, size=(n, 5))
    F[:, 4] = np.where(rng.uniform(size=n) < feasible_share,
                       rng.uniform(TARGET, 2 * TARGET, n), rng.uniform(0.5 * TARGET, TARGET * 0.999, n))
    # a few exact cash ties to exercise the lexicographic tie branch
    F[: n // 10, 4] = F[n // 10: 2 * (n // 10), 4]
    return F


@pytest.mark.parametrize("rule", ["constrained", "lexicographic"])
def test_strict_partial_order(rule):
    F = random_fits(60, 1)
    D = dominance.matrix(F, rule)
    assert not D.diagonal().any()                     # irreflexive
    assert not (D & D.T).any()                        # antisymmetric
    # transitive: i>j and j>k imply i>k
    two_step = (D.astype(int) @ D.astype(int)) > 0
    assert not (two_step & ~D).any()


@pytest.mark.parametrize("rule", ["constrained", "lexicographic"])
def test_matrix_matches_pairwise(rule):
    F = random_fits(40, 2)
    D = dominance.matrix(F, rule)
    for i, j in itertools.product(range(len(F)), repeat=2):
        if i != j:
            assert D[i, j] == dominance.dominates(F[i], F[j], rule)


def test_feasible_beats_infeasible_and_smaller_violation_wins():
    good_risk_bad_cash = [0.0, 1.0, 0.0, 0.0, TARGET * 0.9]
    bad_risk_ok_cash = [1.0, 0.0, 1.0, 1.0, TARGET]
    assert dominance.constrained(bad_risk_ok_cash, good_risk_bad_cash)
    assert not dominance.constrained(good_risk_bad_cash, bad_risk_ok_cash)
    closer = [1, 0, 1, 1, TARGET * 0.95]
    assert dominance.constrained(closer, good_risk_bad_cash)


def test_trade_off_is_kept_by_pareto_but_not_by_lexicographic():
    rich_risky = [2.0, 0.0, 2.0, 2.0, TARGET * 1.5]
    poor_safe = [1.0, 1.0, 1.0, 1.0, TARGET * 1.1]
    assert not dominance.constrained(rich_risky, poor_safe)
    assert not dominance.constrained(poor_safe, rich_risky)
    assert dominance.lexicographic(rich_risky, poor_safe)


@pytest.mark.parametrize("rule", ["constrained", "lexicographic"])
def test_nondominated_sort_matches_definition(rule):
    F = random_fits(80, 3)
    D = dominance.matrix(F, rule)
    fronts = core.nondominated_sort(F, rule)
    assert sorted(i for f in fronts for i in f) == list(range(len(F)))
    for k, front in enumerate(fronts):
        later = [i for f in fronts[k:] for i in f]
        for j in front:
            assert not D[later, j].any()              # nobody in this or later fronts dominates j
            if k:
                assert D[fronts[k - 1], j].any()      # someone in the previous front does


def test_crowding_distance_hand_example():
    # three points on a line in objective 0; other objectives constant
    F = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [3, 0, 0, 0, 0]], dtype=float)
    d = core.crowding_distance(F)
    assert np.isinf(d[0]) and np.isinf(d[2])
    assert d[1] == pytest.approx((3 - 0) / (3 - 0))


def test_run_best_prefers_feasible_max_cash():
    F = [[0, 0, 0, 0, TARGET * 0.99], [0, 0, 0, 0, TARGET * 1.2], [0, 0, 0, 0, TARGET * 1.1]]
    assert core.run_best(F) == 1
    assert core.run_best([[0, 0, 0, 0, TARGET * 0.5], [0, 0, 0, 0, TARGET * 0.9]]) == 1
