"""Building blocks shared by the solvers: sorting, crowding, selection, reporting."""
import random

import numpy as np

import dominance


def nondominated_sort(fits, rule=None):
    """Fast non-dominated sort on a dominance matrix. Returns a list of fronts (lists of indices)."""
    D = dominance.matrix(fits, rule)
    n = D.shape[0]
    count = D.sum(axis=0).astype(int)  # how many individuals dominate j
    remaining = np.ones(n, dtype=bool)
    fronts = []
    current = np.flatnonzero(count == 0)
    while current.size:
        fronts.append(current.tolist())
        remaining[current] = False
        count = count - D[current].sum(axis=0)
        current = np.flatnonzero(remaining & (count == 0))
    return fronts


def ranks_from_fronts(fronts, n):
    r = np.empty(n, dtype=int)
    for k, front in enumerate(fronts):
        r[front] = k
    return r


def crowding_distance(fits):
    """Deb et al. (2002) crowding distance on all-minimization objectives, boundary points = inf."""
    F = np.atleast_2d(np.asarray(fits, dtype=float)) * dominance.SIGNS
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        d[:] = np.inf
        return d
    for k in range(m):
        order = np.argsort(F[:, k], kind="stable")
        lo, hi = F[order[0], k], F[order[-1], k]
        d[order[0]] = d[order[-1]] = np.inf
        if hi > lo:
            d[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / (hi - lo)
    return d


def binary_tournament(n, better, rng=random):
    """Pick two distinct indices at random and return the one `better(i, j)` prefers."""
    i, j = rng.randrange(n), rng.randrange(n)
    return i if better(i, j) else j


def run_best(fits):
    """Common rule for the single reported plan of a run: the highest-cash plan that meets the
    target; if none does, the plan with the smallest target violation."""
    F = np.asarray(fits, dtype=float)
    cv = np.array([dominance.violation(f) for f in F])
    feas = np.flatnonzero(cv == 0.0)
    if feas.size:
        return int(feas[np.argmax(F[feas, 4])])
    return int(np.argmin(cv))


def final_archive(fits, rule=None):
    """Indices of the first front, restricted to target-feasible plans when any exist."""
    front = nondominated_sort(fits, rule)[0]
    feas = [i for i in front if dominance.violation(fits[i]) == 0.0]
    return feas or front
