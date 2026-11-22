"""Per-run quality indicators on final archives (Section 5.3).

All indicators work on all-minimization objectives normalized per instance by the ideal and
nadir points of the reference front (the non-dominated union of every final archive of that
instance, over all methods and runs). Coordinates are clipped below at 0.

* hypervolume: volume dominated by the archive inside the box [0, 1.1]^5, estimated by
  quasi-Monte Carlo with one fixed scrambled Sobol sample per instance, so every archive of an
  instance is measured on the same points.
* IGD+: mean modified distance from the reference front to the archive (Ishibuchi et al. 2015).
* spacing: Schott's SP on the run's own archive (nearest-neighbour Euclidean distances).
* coverage: share of B's points Pareto-dominated by some point of A.
"""
import numpy as np
from scipy.stats import qmc

import dominance

REF = 1.1
QMC_LOG2 = 17  # 131,072 points


def to_min(points):
    return np.atleast_2d(np.asarray(points, dtype=float)) * dominance.SIGNS


def pareto_front(M):
    """Rows of M (all-minimization) not dominated by any other row."""
    M = np.atleast_2d(M)
    le = np.all(M[:, None, :] <= M[None, :, :], axis=2)
    lt = np.any(M[:, None, :] < M[None, :, :], axis=2)
    dominated = (le & lt).any(axis=0)
    return M[~dominated]


class Normalizer:
    def __init__(self, archives):
        union = np.vstack([to_min(a) for a in archives if len(a)])
        self.front = pareto_front(union)
        self.ideal = self.front.min(axis=0)
        nadir = self.front.max(axis=0)
        self.span = np.where(nadir - self.ideal > 0, nadir - self.ideal, 1.0)

    def __call__(self, points):
        return np.maximum((to_min(points) - self.ideal) / self.span, 0.0)


class Hypervolume:
    def __init__(self, dim=5, seed=20261122):
        self.samples = qmc.Sobol(d=dim, scramble=True, seed=seed).random_base2(QMC_LOG2) * REF

    def __call__(self, Z, chunk=8192):
        Z = np.atleast_2d(Z)
        Z = Z[np.all(Z < REF, axis=1)]
        if not len(Z):
            return 0.0
        hit = 0
        for k in range(0, len(self.samples), chunk):
            S = self.samples[k: k + chunk]
            hit += int(np.any(np.all(Z[None, :, :] <= S[:, None, :], axis=2), axis=1).sum())
        return hit / len(self.samples) * REF ** Z.shape[1]


def igd_plus(Z, front_Z):
    Z, R = np.atleast_2d(Z), np.atleast_2d(front_Z)
    d = np.sqrt((np.maximum(Z[None, :, :] - R[:, None, :], 0.0) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())


def spacing(Z):
    Z = np.atleast_2d(Z)
    if len(Z) < 2:
        return float("nan")
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(np.sqrt(((nn - nn.mean()) ** 2).sum() / (len(nn) - 1)))


def coverage(A, B):
    A, B = to_min(A), to_min(B)
    if not len(B):
        return float("nan")
    le = np.all(A[:, None, :] <= B[None, :, :], axis=2)
    lt = np.any(A[:, None, :] < B[None, :, :], axis=2)
    return float((le & lt).any(axis=0).mean())
