"""Feasibility and behaviour of the constructive sampler G, checked by checker.py,
on the 2023 instances (data/instances)."""
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import checker  # noqa: E402
import plan  # noqa: E402

INST = os.path.join(ROOT, "data", "instances")


@pytest.fixture(scope="module", params=["vn30", "hnx30"])
def instance(request):
    npz = os.path.join(INST, f"{request.param}_2023_moments.npz")
    if not os.path.exists(npz):
        pytest.skip("instance files not found")
    plan.load_instance(os.path.join(INST, f"{request.param}_2023.json"), npz)
    random.seed(11)
    pop = [plan.sample() for _ in range(300)]
    kids = [plan.sample(random.choice(pop), random.choice(pop)) for _ in range(300)]
    anchors = random.sample(pop, 20)
    nbrs = [(a, plan.sample(anchor=a)) for a in anchors for _ in range(10)]
    return request.param, pop, kids, nbrs


def test_every_generated_plan_is_feasible(instance):
    _, pop, kids, nbrs = instance
    worst = 0.0
    for ind in pop + kids + [b for _, b in nbrs]:
        bad, ratio = checker.check(ind)
        assert not bad, bad
        worst = max(worst, ratio)
    assert worst < 1.0, "forced final sale exceeded the month-12 matched volume"


def test_partial_sales_now_happen(instance):
    _, pop, kids, _ = instance
    partial = full = 0
    for ind in pop + kids:
        for t in range(1, plan.tau - 1):
            sold = ind.y1[t] > 0
            partial += int((sold & (ind.q[t] > 0)).sum())
            full += int((sold & (ind.q[t] == 0)).sum())
    assert partial > 0 and full > 0


def test_sa_neighbour_copies_anchor_before_the_pivot(instance):
    _, _, _, nbrs = instance
    for anchor, nb in nbrs:
        diff = [t for t in range(plan.tau) if not (np.array_equal(anchor.y0[t], nb.y0[t])
                                                     and np.array_equal(anchor.y1[t], nb.y1[t]))]
        if diff:
            first = diff[0]
            assert first <= plan.tau - 3
            assert np.allclose(anchor.theta[: first + 1], nb.theta[: first + 1])


def test_deposit_only_matches_closed_form(instance):
    n = plan.PRICES.shape[0]
    z = np.zeros((plan.tau, n), dtype=np.int64)
    ind = plan.simulate(z, z)
    assert ind.theta[-1] == pytest.approx(plan.deposit_only_cash(), rel=1e-12)
    assert not checker.check(ind)[0]


def test_dividend_is_credited_two_months_after_the_holding_month(instance):
    n = plan.PRICES.shape[0]
    saved = plan.DIVS.copy()
    try:
        j = int(np.argmax(plan.PRICES[:, 0] > 0))
        plan.DIVS[:] = 0.0
        plan.DIVS[j, 1] = 1.0                      # 1 kVND per share paid in month index 1 (February)
        y0 = np.zeros((plan.tau, n), dtype=np.int64)
        y1 = np.zeros((plan.tau, n), dtype=np.int64)
        y0[0, j] = 1000
        y1[plan.tau - 1, j] = 1000
        with_div = plan.simulate(y0, y1)
        plan.DIVS[j, 1] = 0.0
        without = plan.simulate(y0, y1)
        assert with_div.theta[1] == pytest.approx(without.theta[1])
        assert with_div.theta[2] - without.theta[2] == pytest.approx(1000 * 1.0)   # Eq. (21), t >= 2
    finally:
        plan.DIVS[:] = saved


def test_same_seed_same_plan(instance):
    random.seed(5)
    a = plan.sample()
    random.seed(5)
    b = plan.sample()
    assert np.array_equal(a.y0, b.y0) and np.array_equal(a.y1, b.y1) and np.allclose(a.f, b.f)


def test_evaluation_counter(instance):
    plan.reset_evals()
    for _ in range(7):
        plan.sample()
    assert plan.EVALS == 7
