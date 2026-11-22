"""Independent feasibility check of a trading plan against every constraint of Section 3.1.

The checks re-derive each rule from the plan alone (buys y0, sells y1, holdings q, cash theta)
and the instance data in `plan`, without reusing the sampler's logic. `check()` returns a list of
violation strings (empty = feasible) and the terminal-liquidation volume ratio
max_j q[tau-2, j] / Q[j, tau-1], which Eq. (23) allows to exceed 1 and which we report.
"""
import numpy as np

import plan


def check(ind, rel_tol=1e-9):
    P, Q, D = plan.PRICES, plan.VOLUMES, plan.DIVS
    tau, xi, eps, alpha, K = plan.tau, plan.xi, plan.epsilon, plan.alpha, plan.cardinalityK
    y0, y1, q, theta = ind.y0, ind.y1, ind.q, ind.theta
    bad = []

    def fail(msg):
        if len(bad) < 20:
            bad.append(msg)

    if (y0 < 0).any() or (y1 < 0).any() or (q < 0).any():
        fail("negative trade or holding")
    if (y0 % xi).any() or (y1 % xi).any():
        fail("trade not a multiple of the lot size")                                # Eq. (11)
    if (y1[0] > 0).any():
        fail("sale at t=0")                                                          # Eq. (3)
    if (y0[tau - 2:] > 0).any():
        fail("purchase in one of the last two months")                               # Eq. (24)
    for t in range(tau):
        prev = q[t - 1] if t >= 1 else np.zeros_like(q[0])
        prev2 = q[t - 2] if t >= 2 else np.zeros_like(q[0])
        if not np.array_equal(q[t], prev + y0[t] - y1[t]):
            fail(f"t={t}: holdings recursion broken")                                # Eq. (14)
        if ((y0[t] > 0) & (y1[t] > 0)).any():
            fail(f"t={t}: buy and sell of the same stock")                           # Eq. (12)
        if (y0[t] > Q[:, t]).any():
            fail(f"t={t}: purchase above matched volume")                            # Eq. (10)
        if ((y0[t] > 0) & (P[:, t] <= 0)).any():
            fail(f"t={t}: purchase without a traded price")
        if (y1[t] > prev).any():
            fail(f"t={t}: sale above holdings")                                      # Eq. (6)
        if t < tau - 1:
            if (y1[t] > prev2).any():
                fail(f"t={t}: shares bought last month were sold")                   # Eq. (7)
            if (y1[t] > Q[:, t]).any():
                fail(f"t={t}: sale above matched volume")                            # Eq. (10)
            if ((y1[t] > 0) & (D[:, t] > 0)).any():
                fail(f"t={t}: sale in a dividend month")                             # Eq. (8)
        cost = float((1 + eps) * (P[:, t] @ y0[t]))
        if cost > theta[t] * (1 + rel_tol) + 1e-6:
            fail(f"t={t}: purchases exceed available cash")                          # Eq. (9)
        if np.count_nonzero(q[t]) > K:
            fail(f"t={t}: more than K holdings")                                     # Eq. (15)
    if not np.array_equal(y1[tau - 1], q[tau - 2]) or q[tau - 1].any():
        fail("terminal liquidation incomplete")                                      # Eq. (23)

    # cash path, recomputed from scratch (Eq. 21)
    cash = np.zeros(tau + 1)
    cash[0] = plan.initTheta
    for t in range(1, tau + 1):
        dep = cash[t - 1] - (1 + eps) * float(P[:, t - 1] @ y0[t - 1])
        cash[t] = (1 + alpha) * dep + (1 - eps) * float(P[:, t - 1] @ y1[t - 1])
        if t >= 2:
            cash[t] += float(D[:, t - 1] @ q[t - 2])
    if not np.allclose(cash, theta, rtol=1e-9, atol=1e-6):
        fail("cash path differs from Eq. (21)")

    held_last = q[tau - 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(held_last > 0, held_last / np.maximum(Q[:, tau - 1], 1e-12), 0.0)
    return bad, float(ratio.max(initial=0.0))
