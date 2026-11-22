"""Statistical protocol (Section 5.3).

Runs are paired by seed: the same seed gives every solver the same initial population and
the same random-number stream at the start (common random numbers), so a seed is a block.
* Friedman omnibus over methods (blocks = seeds).
* Two-sided Wilcoxon signed-rank tests on pairs, Holm step-down correction with monotone
  adjusted p-values (testing stops at the first non-rejection).
* Percentile bootstrap confidence intervals of the mean paired difference.
* Average ranks per instance and a Friedman test across instances (blocks = instances).
"""
import numpy as np
from scipy import stats


def friedman(table):
    """table: (n_blocks, n_methods). Returns (chi2, p)."""
    t = np.asarray(table, dtype=float)
    t = t[~np.isnan(t).any(axis=1)]          # a block with an undefined value is dropped
    res = stats.friedmanchisquare(*t.T)
    return float(res.statistic), float(res.pvalue)


def wilcoxon(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))      # pairs with an undefined value are dropped
    a, b = a[keep], b[keep]
    if a.size == 0 or np.allclose(a, b):
        return 0.0, 1.0
    res = stats.wilcoxon(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def holm(pvalues, alpha=0.05):
    """Monotone Holm step-down adjusted p-values and reject flags, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted, adjusted <= alpha


def bootstrap_ci(diff, level=0.95, n_boot=10_000, seed=0):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(diff.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def average_ranks(table, higher_is_better):
    """Ranks within each row (1 = best), averaged over rows."""
    t = np.asarray(table, dtype=float)
    r = stats.rankdata(-t if higher_is_better else t, axis=1)
    return r.mean(axis=0)
