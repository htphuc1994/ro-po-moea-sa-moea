"""Deterministic reference plans (Section 4.7), evaluated with the same cash recursion.

* deposit_only: no trades; theta_tau = Theta * (1 + alpha)^tau.
* equal_weight: Theta split equally over the K most liquid stocks (month-1 matched volume,
  universe="topK") or over all stocks (universe="all", cardinality relaxed), bought at
  month-1 prices within budget, lot and volume limits, held without rebalancing and sold by
  the forced liquidation in the last month.
"""
import numpy as np

import plan


def deposit_only():
    z = np.zeros((plan.tau, plan.PRICES.shape[0]), dtype=np.int64)
    return plan.simulate(z, z)


def equal_weight(universe="topK"):
    n = plan.PRICES.shape[0]
    tradable = np.flatnonzero((plan.PRICES[:, 0] > 0) & (plan.VOLUMES[:, 0] > 0) & (plan.VOLUMES[:, -1] > 0))
    if universe == "topK":
        tradable = tradable[np.argsort(-plan.VOLUMES[tradable, 0], kind="stable")][: int(plan.cardinalityK)]
    y0 = np.zeros((plan.tau, n), dtype=np.int64)
    budget = plan.initTheta / (1 + plan.epsilon) / max(len(tradable), 1)
    for j in tradable:
        shares = int(budget // (plan.PRICES[j, 0] * plan.xi)) * plan.xi
        y0[0, j] = min(shares, int(plan.VOLUMES[j, 0] // plan.xi) * plan.xi)
    y1 = np.zeros_like(y0)
    y1[plan.tau - 1] = y0[0]
    return plan.simulate(y0, y1)
