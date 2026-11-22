"""Comparison rules shared by every solver.

An objective vector is f = (variance, third moment, fourth moment, entropy,
terminal cash) with orientation (min, max, min, min, max). The return target
theta_tau >= (1 + upsilon) * Theta is a constraint; its violation is

    CV(f) = max(0, target - f[4]) / target.

Two rules are provided:

* ``constrained`` (default): Deb's constraint-domination. A feasible plan beats
  an infeasible one; between infeasible plans the smaller violation wins;
  between feasible plans ordinary Pareto dominance on all five objectives
  decides. This is a strict partial order (irreflexive, antisymmetric,
  transitive), so it is valid for non-dominated sorting.
* ``lexicographic``: the return-priority rule of the first-round manuscript,
  implemented exactly as stated there (used only as an ablation). A plan that
  meets the target beats any plan with less terminal cash; equal cash falls back
  to Pareto dominance on the four risk objectives.
"""
import numpy as np

THETA = 1_000_000.0  # initial capital (kVND)
UPSILON = 0.06       # target return rate
SIGNS = np.array([1.0, -1.0, 1.0, 1.0, -1.0])  # multiply to get all-minimization

_rule = "constrained"


def configure(theta=None, upsilon=None):
    global THETA, UPSILON
    if theta is not None:
        THETA = float(theta)
    if upsilon is not None:
        UPSILON = float(upsilon)


def set_rule(name):
    global _rule
    if name not in ("constrained", "lexicographic"):
        raise ValueError(name)
    _rule = name


def get_rule():
    return _rule


def target():
    return THETA * (1.0 + UPSILON)


def violation(f):
    t = target()
    return max(0.0, t - float(f[4])) / t


def to_min(f):
    return np.asarray(f, dtype=float) * SIGNS


def _pareto(a, b):
    a, b = to_min(a), to_min(b)
    return bool(np.all(a <= b) and np.any(a < b))


def constrained(a, b):
    ca, cb = violation(a), violation(b)
    if ca == 0.0 and cb > 0.0:
        return True
    if ca > 0.0 and cb > 0.0:
        return ca < cb
    if ca > 0.0:
        return False
    return _pareto(a, b)


def lexicographic(a, b):
    if float(a[4]) < target():
        return False
    if float(a[4]) > float(b[4]):
        return True
    if float(a[4]) == float(b[4]):
        return _pareto(np.r_[a[:4], 0.0], np.r_[b[:4], 0.0])
    return False


def dominates(a, b, rule=None):
    return (constrained if (rule or _rule) == "constrained" else lexicographic)(a, b)


def matrix(fits, rule=None):
    """D[i, j] is True when fits[i] dominates fits[j] under the rule."""
    rule = rule or _rule
    F = np.asarray(fits, dtype=float)
    M = F * SIGNS
    le = np.all(M[:, None, :] <= M[None, :, :], axis=2)
    lt = np.any(M[:, None, :] < M[None, :, :], axis=2)
    if rule == "constrained":
        cv = np.maximum(0.0, target() - F[:, 4]) / target()
        feas = cv == 0.0
        D = (feas[:, None] & feas[None, :]) & le & lt
        D |= feas[:, None] & ~feas[None, :]
        D |= (~feas[:, None] & ~feas[None, :]) & (cv[:, None] < cv[None, :])
    else:
        cash = F[:, 4]
        meets = cash >= target()
        rle = np.all(M[:, None, :4] <= M[None, :, :4], axis=2)
        rlt = np.any(M[:, None, :4] < M[None, :, :4], axis=2)
        D = meets[:, None] & ((cash[:, None] > cash[None, :]) | ((cash[:, None] == cash[None, :]) & rle & rlt))
    np.fill_diagonal(D, False)
    return D
