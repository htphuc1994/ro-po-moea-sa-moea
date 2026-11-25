"""The five solvers of Section 4 on one shared core.

All solvers share: the constructive sampler G (plan.sample), the comparison rule
(dominance; constraint-domination by default), the evaluation budget (every call to G
counts, including SA neighbours), the initial population (the same seed gives the same
first 100 plans to every solver; MOEA/D keeps the first 70), the SA / random-injection
hook, and the rule for the single reported plan (core.run_best).

Methods: ga (scalarized baseline), nsga2, spea2, nsga3, moead.
sa_mode: None, "sa" (SA injection) or "ri" (random-injection control).
sa_chi0: starting acceptance probability of the SA phase (default plan.SA_START_ACCEPT = 0.5).
"""
import math
import random
import time

import numpy as np

import core
import dominance
import plan

N_POP = 100
NSGA3_P = 4          # Das-Dennis divisions -> 70 reference points / MOEA-D weight vectors
MOEAD_T = 20         # neighbourhood size
MOEAD_DELTA = 0.9    # probability of mating inside the neighbourhood
MOEAD_NR = 2         # replacement cap per child


def fits_of(pop):
    return np.array([ind.f for ind in pop])


def das_dennis(M, p):
    points = []

    def rec(axes, remaining, cur):
        if axes == 1:
            points.append(cur + [remaining / p])
            return
        for i in range(remaining + 1):
            rec(axes - 1, remaining - i, cur + [i / p])

    rec(M, p, [])
    return np.array(points)


# ---------------------------------------------------------------- GA (Section 4.1)
class GA:
    """Scalarized GA: feasible plans first, then the equal-weight Tchebycheff energy on
    population-normalized objectives. Generational: top 10% kept, 90 children from parents
    drawn uniformly from the top half."""
    n_init = N_POP

    def __init__(self, pop):
        self.pop = pop
        self._sort()

    def _sort(self):
        lo, hi = plan.population_bounds(self.pop)
        self.pop.sort(key=lambda ind: (ind.cv, plan.energy(ind.f, lo, hi)))

    def evals_per_generation(self, inject):
        return 90 - (1 if inject else 0)

    def generation(self, incumbent=None):
        top = self.pop[: N_POP // 2]
        kids = [plan.sample(random.choice(top), random.choice(top)) for _ in range(self.evals_per_generation(incumbent is not None))]
        if incumbent is not None:
            kids.append(incumbent)
        self.pop = self.pop[: N_POP // 10] + kids
        self._sort()

    def population(self):
        return self.pop

    def archive(self):
        return [self.pop[i] for i in core.final_archive(fits_of(self.pop))]


# ---------------------------------------------------------------- NSGA-II (Section 4.2)
class NSGA2:
    """Deb et al. (2002): binary tournament on (rank, crowding), N children, (mu + lambda)
    survival by non-dominated sorting with crowding-distance truncation."""
    n_init = N_POP

    def __init__(self, pop):
        self.pop = pop
        self._rank()

    def _rank(self):
        F = fits_of(self.pop)
        fronts = core.nondominated_sort(F)
        self.rank = core.ranks_from_fronts(fronts, len(self.pop))
        self.crowd = np.zeros(len(self.pop))
        for fr in fronts:
            self.crowd[fr] = core.crowding_distance(F[fr])

    def _better(self, i, j):
        if self.rank[i] != self.rank[j]:
            return self.rank[i] < self.rank[j]
        return self.crowd[i] >= self.crowd[j]

    def evals_per_generation(self, inject):
        return N_POP - (1 if inject else 0)

    def generation(self, incumbent=None):
        n = len(self.pop)
        pick = lambda: self.pop[core.binary_tournament(n, self._better)]
        kids = [plan.sample(pick(), pick()) for _ in range(self.evals_per_generation(incumbent is not None))]
        if incumbent is not None:
            kids.append(incumbent)
        comb = self.pop + kids
        F = fits_of(comb)
        chosen = []
        for fr in core.nondominated_sort(F):
            if len(chosen) + len(fr) <= N_POP:
                chosen += fr
            else:
                cd = core.crowding_distance(F[fr])
                order = np.argsort(-cd, kind="stable")
                chosen += [fr[k] for k in order[: N_POP - len(chosen)]]
                break
        self.pop = [comb[i] for i in chosen]
        self._rank()

    def population(self):
        return self.pop

    def archive(self):
        return [self.pop[i] for i in core.final_archive(fits_of(self.pop))]


# ---------------------------------------------------------------- SPEA2 (Section 4.3)
def _truncate(dist, keep):
    """SPEA2 archive truncation (Zitzler et al. 2001): while too many plans remain, remove the
    plan whose distance to its nearest remaining neighbour is smallest; ties are broken by the
    second-nearest distance, then the third, and so on. `dist` holds the pairwise distances in
    normalized objective space. Returns the indices of the plans kept, in their original order."""
    D = np.array(dist, dtype=float)
    np.fill_diagonal(D, np.inf)
    alive = np.ones(len(D), dtype=bool)
    for _ in range(len(D) - keep):
        A = np.flatnonzero(alive)
        sub = D[np.ix_(A, A)]
        nn = sub.min(axis=1)
        tied = np.flatnonzero(nn == nn.min())
        if len(tied) > 1:
            rows = np.sort(sub[tied], axis=1)
            tied = tied[np.lexsort(rows.T[::-1])]
        alive[A[tied[0]]] = False
    return np.flatnonzero(alive)


class SPEA2:
    """Zitzler et al. (2001): strength S, raw fitness R, k-NN density D, external archive of
    size N; mating by binary tournament on F = R + D over the archive; the N children form
    the next population. When the non-dominated set exceeds the archive it is reduced by the
    iterative truncation operator (`_truncate`); `truncations` counts the generations in which
    that happens."""
    n_init = N_POP

    def __init__(self, pop):
        self.pop = pop
        self.arch = []
        self.arch_F = np.zeros(0)
        self.k = int(round(math.sqrt(2 * N_POP)))
        self.truncations = 0
        self._update()

    def _fitness(self, inds):
        F = fits_of(inds)
        D = dominance.matrix(F)
        S = D.sum(axis=1)
        R = (D * S[:, None]).sum(axis=0).astype(float)
        M = F * dominance.SIGNS
        span = np.where(M.max(0) - M.min(0) > 0, M.max(0) - M.min(0), 1.0)
        Z = (M - M.min(0)) / span
        dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
        srt = np.sort(dist, axis=1)
        sigma = srt[:, min(self.k, len(inds) - 1)]
        return R, R + 1.0 / (sigma + 2.0), dist

    def _update(self):
        comb = self.pop + self.arch
        R, Ft, dist = self._fitness(comb)
        nondom = [i for i in range(len(comb)) if R[i] == 0]
        if len(nondom) <= N_POP:
            rest = sorted((i for i in range(len(comb)) if R[i] > 0), key=lambda i: Ft[i])
            idx = nondom + rest[: N_POP - len(nondom)]
        else:
            self.truncations += 1
            nd = np.array(nondom)
            idx = nd[_truncate(dist[np.ix_(nd, nd)], N_POP)].tolist()
        self.arch = [comb[i] for i in idx]
        self.arch_F = Ft[idx]

    def evals_per_generation(self, inject):
        return N_POP - (1 if inject else 0)

    def generation(self, incumbent=None):
        n = len(self.arch)
        pick = lambda: self.arch[core.binary_tournament(n, lambda i, j: self.arch_F[i] <= self.arch_F[j])]
        kids = [plan.sample(pick(), pick()) for _ in range(self.evals_per_generation(incumbent is not None))]
        if incumbent is not None:
            kids.append(incumbent)
        self.pop = kids
        self._update()

    def population(self):
        return self.arch

    def archive(self):
        return [self.arch[i] for i in core.final_archive(fits_of(self.arch))]


# ---------------------------------------------------------------- NSGA-III (Section 4.4)
def _asf_normalize(orient_min):
    M = orient_min.shape[1]
    translated = orient_min - orient_min.min(axis=0)
    extremes = np.zeros((M, M))
    for i in range(M):
        w = np.full(M, 1.0e-6)
        w[i] = 1.0
        extremes[i] = translated[int(np.argmin((translated / w).max(axis=1)))]
    intercepts = None
    try:
        cand = 1.0 / np.linalg.solve(extremes, np.ones(M))
        if np.all(cand > 1e-9) and np.all(np.isfinite(cand)):
            intercepts = cand
    except np.linalg.LinAlgError:
        pass
    if intercepts is None:
        intercepts = translated.max(axis=0)
        intercepts = np.where(intercepts > 1e-9, intercepts, 1.0)
    return translated / intercepts


def _associate(normalized, ref):
    unit = ref / np.where(np.linalg.norm(ref, axis=1) > 1e-12, np.linalg.norm(ref, axis=1), 1.0)[:, None]
    proj = normalized @ unit.T                                   # (n, nr)
    d = np.linalg.norm(normalized[:, None, :] - proj[:, :, None] * unit[None, :, :], axis=2)
    a = d.argmin(axis=1)
    return a, d[np.arange(len(a)), a]


def _niching(k, split_idx, assign, dist, niche):
    available = list(range(len(split_idx)))
    selected = []
    while len(selected) < k and available:
        refs = sorted({assign[a] for a in available})
        least = min(niche[r] for r in refs)
        r = random.choice([r for r in refs if niche[r] == least])
        cands = [a for a in available if assign[a] == r]
        best = min(cands, key=lambda a: dist[a]) if niche[r] == 0 else random.choice(cands)
        selected.append(best)
        available.remove(best)
        niche[r] += 1
    return [split_idx[i] for i in selected]


class NSGA3:
    """Deb & Jain (2014): reference-point niching on the splitting front after non-dominated
    sorting; mating by binary tournament on rank. Das-Dennis p = 4 (70 points)."""
    n_init = N_POP

    def __init__(self, pop):
        self.ref = das_dennis(5, NSGA3_P)
        self.pop = pop
        self._rank()

    def _rank(self):
        self.rank = core.ranks_from_fronts(core.nondominated_sort(fits_of(self.pop)), len(self.pop))

    def _better(self, i, j):
        if self.rank[i] != self.rank[j]:
            return self.rank[i] < self.rank[j]
        return random.random() < 0.5

    def evals_per_generation(self, inject):
        return N_POP - (1 if inject else 0)

    def generation(self, incumbent=None):
        n = len(self.pop)
        pick = lambda: self.pop[core.binary_tournament(n, self._better)]
        kids = [plan.sample(pick(), pick()) for _ in range(self.evals_per_generation(incumbent is not None))]
        if incumbent is not None:
            kids.append(incumbent)
        comb = self.pop + kids
        F = fits_of(comb)
        chosen, split = [], None
        for fr in core.nondominated_sort(F):
            if len(chosen) + len(fr) <= N_POP:
                chosen += fr
                if len(chosen) == N_POP:
                    break
            else:
                split = fr
                break
        if split is not None:
            pool = chosen + split
            normalized = _asf_normalize(F[pool] * dominance.SIGNS)
            assign, dist = _associate(normalized, self.ref)
            niche = np.zeros(len(self.ref), dtype=int)
            for r in assign[: len(chosen)]:
                niche[r] += 1
            chosen += _niching(N_POP - len(chosen), split, assign[len(chosen):], dist[len(chosen):], niche)
        self.pop = [comb[i] for i in chosen]
        self._rank()

    def population(self):
        return self.pop

    def archive(self):
        return [self.pop[i] for i in core.final_archive(fits_of(self.pop))]


# ---------------------------------------------------------------- MOEA/D (Section 4.5)
class MOEAD:
    """Zhang & Li (2007): Das-Dennis weights (p = 4, N = 70), neighbourhoods of T = 20,
    mating inside the neighbourhood with probability delta, at most n_r replacements per child.
    Replacement uses the same constraint handling as the other solvers (feasible first, then
    smaller violation) and otherwise the Tchebycheff value of objectives normalized by the
    running ideal point and the population's nadir point."""

    def __init__(self, pop):
        W = das_dennis(5, NSGA3_P)
        W = np.where(W < 1e-6, 1e-6, W)
        self.W = W / W.sum(axis=1, keepdims=True)
        self.n = len(self.W)
        d = np.sqrt(((self.W[:, None, :] - self.W[None, :, :]) ** 2).sum(axis=2))
        self.B = np.argsort(d, axis=1)[:, :MOEAD_T]
        self.pop = pop[: self.n]
        M = fits_of(self.pop) * dominance.SIGNS
        self.z = M.min(axis=0)

    n_init = 70

    def _tch(self, ind, j, nadir):
        span = np.where(nadir - self.z > 0, nadir - self.z, 1.0)
        return float(np.max(self.W[j] * np.abs((dominance.to_min(ind.f) - self.z) / span)))

    def _offer(self, child, pool, nadir):
        self.z = np.minimum(self.z, dominance.to_min(child.f))
        candidates = list(pool)
        random.shuffle(candidates)
        replaced = 0
        for j in candidates:
            if replaced >= MOEAD_NR:
                break
            cur = self.pop[j]
            if child.cv == 0 and cur.cv > 0:
                accept = True
            elif cur.cv == 0 and child.cv > 0:
                accept = False
            elif child.cv > 0 and cur.cv > 0:
                accept = child.cv <= cur.cv
            else:
                accept = self._tch(child, j, nadir) <= self._tch(cur, j, nadir)
            if accept:
                self.pop[j] = child
                replaced += 1

    def evals_per_generation(self, inject):
        return self.n

    def generation(self, incumbent=None):
        nadir = (fits_of(self.pop) * dominance.SIGNS).max(axis=0)
        for i in random.sample(range(self.n), self.n):
            pool = self.B[i] if random.random() < MOEAD_DELTA else range(self.n)
            p1, p2 = random.choice(pool), random.choice(pool)
            self._offer(plan.sample(self.pop[p1], self.pop[p2]), pool, nadir)
        if incumbent is not None:
            self._offer(incumbent, self.B[self.n - 1], nadir)

    def population(self):
        return self.pop

    def archive(self):
        return [self.pop[i] for i in core.final_archive(fits_of(self.pop))]


SOLVERS = {"ga": GA, "nsga2": NSGA2, "spea2": SPEA2, "nsga3": NSGA3, "moead": MOEAD}


def _sparse(arr):
    t, j = np.nonzero(arr)
    return [[int(a), int(b), int(arr[a, b])] for a, b in zip(t, j)]


def _round(f):
    return [float(f"{x:.7g}") for x in f]


def run(method, seed, budget=10_000, sa_mode=None, rule="constrained", snapshot_every=2500, sa_chi0=None):
    """One run. Returns a JSON-ready dict."""
    dominance.set_rule(rule)
    random.seed(seed)
    np.random.seed(seed)
    plan.reset_evals()
    pool = [plan.sample() for _ in range(N_POP)]       # common initial population for this seed
    cls = SOLVERS[method]
    solver = cls(list(pool))
    plan.EVALS = cls.n_init                             # MOEA/D does not use plans 71-100
    # sa_mode: "sa" (SA chain, Algorithm 2), "ri" (its random-injection control), "sae" / "rie"
    # (the same, but every phase starts from the lowest-energy plan of the current population)
    sa = plan.SAInjector(pool[0], accept_all=sa_mode in ("ri", "rie"),
                         start_accept=plan.SA_START_ACCEPT if sa_chi0 is None else sa_chi0) if sa_mode else None
    elite_anchor = sa_mode in ("sae", "rie")
    diag, new_plans = [], []        # per generation [survived, unchanged]; injected plans that are new

    t0 = time.time()
    gens, snaps, next_snap = 0, [], snapshot_every
    while True:
        need = solver.evals_per_generation(sa is not None)
        if sa is not None:
            need += plan.SA_STEPS + (20 if sa.T0 is None else 0)
        if plan.EVALS + need > budget:
            break
        incumbent = None
        if sa is not None:
            pop = solver.population()
            lo, hi = plan.population_bounds(pop)
            if elite_anchor:
                sa.anchor = plan.elite(pop, lo, hi)
            anchor = sa.anchor
            incumbent = sa.phase(lo, hi)
        solver.generation(incumbent)
        if sa is not None:          # diagnostics only: identity checks, no random numbers drawn
            unchanged = incumbent is anchor
            diag.append([int(any(p is incumbent for p in solver.population())), int(unchanged)])
            if not unchanged:
                new_plans.append(incumbent)
        gens += 1
        if snapshot_every and plan.EVALS >= next_snap:
            snaps.append({"evals": plan.EVALS, "seconds": time.time() - t0,
                          "archive": [_round(ind.f) for ind in solver.archive()]})
            next_snap += snapshot_every
    runtime = time.time() - t0

    arch = solver.archive()
    best = arch[core.run_best(fits_of(arch))]
    out = {
        "method": method, "sa_mode": sa_mode, "rule": rule, "seed": seed, "budget": budget,
        "evals": plan.EVALS, "generations": gens, "runtime": runtime,
        "archive": [_round(ind.f) for ind in arch],
        "best": {"f": best.f.tolist(), "cv": best.cv, "y0": _sparse(best.y0), "y1": _sparse(best.y1)},
        "snapshots": snaps,
    }
    if sa is not None:
        out["sa"] = {"T0": sa.T0, "chi0": sa.start_accept, "log": sa.log, "diag": diag,
                     "in_final": sum(1 for x in arch if any(x is p for p in new_plans))}
    if isinstance(solver, SPEA2):
        out["spea2_truncations"] = solver.truncations
    return out
