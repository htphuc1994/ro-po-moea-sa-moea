"""NSGA-III (Deb & Jain, 2014) and its hybrid SA-NSGA-III variant for the
dividend-portfolio problem.

Both variants reuse plan.py's constructive sampler `genInvestmentPlan` and
the return-priority dominance `prec_theta` (`plan.is_belonged_frontier`) for
fairness with the GA / NSGA-II / HSG / SPEA2 comparators.  NSGA-III replaces
NSGA-II's crowding distance with reference-point niching:

  1. Generate Das-Dennis reference points on the M-dimensional unit simplex
     (M = 5 objectives).
  2. Fast non-dominated sort using strict prec_theta.
  3. Fill the next population from the leading fronts; on the splitting
     front, use ASF normalization + perpendicular-distance association
     to the reference points and select via niche-preserving rules.

The SA wrapper is the same as plan.HSG: each generation a short SA walk
(`plan.bestSASolution`) updates an incumbent that is injected into the last
offspring slot.

Output JSON mirrors `spea2_results_*.json` format so that
`analyze_with_baselines.py` can ingest it without changes.
"""

from __future__ import annotations

import json
import math
import random
import time
import numpy as np
import plan


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------
def _load_dataset(name):
    if name == "vn30":
        plan.M2 = np.load("data/calculate2023/vn30_3years_before/M2.dat", allow_pickle=True)
        plan.M3 = np.load("data/calculate2023/vn30_3years_before/M3.dat", allow_pickle=True)
        plan.M4 = np.load("data/calculate2023/vn30_3years_before/M4.dat", allow_pickle=True)
        json_path = "data/vn30_2023.json"
    elif name == "hnx30":
        plan.M2 = np.load("data/calculate2023/hnx30_3years_before/M2.npy", allow_pickle=True)
        plan.M3 = np.load("data/calculate2023/hnx30_3years_before/M3.npy", allow_pickle=True)
        plan.M4 = np.load("data/calculate2023/hnx30_3years_before/M4.npy", allow_pickle=True)
        json_path = "data/hnx30_2023.json"
    else:
        raise ValueError(name)
    with open(json_path) as f:
        return json.loads(f.read())


# -----------------------------------------------------------------------------
# Dominance
# -----------------------------------------------------------------------------
def _strict_prec_theta(a, b):
    """Strict return-priority dominance: a precedes b iff
    plan.is_belonged_frontier(a, b) and NOT is_belonged_frontier(b, a)."""
    if not plan.is_belonged_frontier(a, b):
        return False
    return not plan.is_belonged_frontier(b, a)


def _fast_nondominated_sort(fits):
    """Deb's fast non-dominated sort under strict prec_theta.
    Returns list of fronts; each front is a list of indices into fits."""
    n = len(fits)
    S = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _strict_prec_theta(fits[p], fits[q]):
                S[p].append(q)
            elif _strict_prec_theta(fits[q], fits[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]  # drop trailing empty


# -----------------------------------------------------------------------------
# Das-Dennis reference points and ASF normalization
# -----------------------------------------------------------------------------
def das_dennis(M, p):
    """Generate Das-Dennis reference points on the (M-1)-simplex with p divisions.
    Number of points: C(M+p-1, p)."""
    points = []

    def recurse(remaining_axes, remaining_sum, current):
        if remaining_axes == 1:
            points.append(current + [remaining_sum / p])
            return
        for i in range(remaining_sum + 1):
            recurse(remaining_axes - 1, remaining_sum - i, current + [i / p])

    recurse(M, p, [])
    return np.array(points)


def _orient_for_minimization(fits):
    """Convert the 5-objective vector to all-minimization by sign-flipping
    the maximize-direction entries (skewness, terminal cash)."""
    arr = np.array(fits, dtype=float)
    arr[:, 1] = -arr[:, 1]
    arr[:, 4] = -arr[:, 4]
    return arr


def _asf_normalize(orient_min):
    """ASF normalization: translate to ideal point, find extreme points by
    achievement scalarizing function, compute intercepts, divide.

    Falls back to per-axis max if the linear system is degenerate."""
    M = orient_min.shape[1]
    z = orient_min.min(axis=0)
    translated = orient_min - z
    extremes = np.zeros((M, M))
    for i in range(M):
        w = np.full(M, 1.0e-6)
        w[i] = 1.0
        asf = (translated / w).max(axis=1)
        extremes[i] = translated[int(np.argmin(asf))]
    intercepts = None
    try:
        a = np.linalg.solve(extremes, np.ones(M))
        cand = 1.0 / a
        if np.all(cand > 1e-9) and np.all(np.isfinite(cand)):
            intercepts = cand
    except np.linalg.LinAlgError:
        intercepts = None
    if intercepts is None:
        intercepts = translated.max(axis=0)
        intercepts = np.where(intercepts > 1e-9, intercepts, 1.0)
    return translated / intercepts


def _associate(normalized, ref_points):
    """For each individual, return (assigned_ref_idx, perp_distance) pair."""
    n = normalized.shape[0]
    nr = ref_points.shape[0]
    norms = np.linalg.norm(ref_points, axis=1)
    norms = np.where(norms > 1e-12, norms, 1.0)
    unit_ref = ref_points / norms[:, None]
    assignment = np.zeros(n, dtype=int)
    distances = np.zeros(n)
    for i in range(n):
        x = normalized[i]
        # perpendicular distance from x to line through origin in direction unit_ref[r]
        proj = unit_ref @ x  # shape (nr,)
        perp = unit_ref * proj[:, None]
        diff = x[None, :] - perp
        d = np.linalg.norm(diff, axis=1)
        idx = int(np.argmin(d))
        assignment[i] = idx
        distances[i] = d[idx]
    return assignment, distances


def _niching(num_to_select, splitting_idx, assignment_split, distance_split,
             niche_count, num_ref):
    """NSGA-III niche-preserving selection from the splitting front.
    `splitting_idx` are indices into the *combined* pool of the splitting front.
    Returns the selected subset of `splitting_idx` (length `num_to_select`)."""
    available = list(range(len(splitting_idx)))
    selected = []
    while len(selected) < num_to_select and available:
        candidate_refs = sorted({assignment_split[a] for a in available})
        min_count = min(niche_count[r] for r in candidate_refs)
        eligible = [r for r in candidate_refs if niche_count[r] == min_count]
        chosen_ref = random.choice(eligible)
        cands = [a for a in available if assignment_split[a] == chosen_ref]
        if niche_count[chosen_ref] == 0:
            best = min(cands, key=lambda a: distance_split[a])
        else:
            best = random.choice(cands)
        selected.append(best)
        available.remove(best)
        niche_count[chosen_ref] += 1
    return [splitting_idx[i] for i in selected]


# -----------------------------------------------------------------------------
# Environmental selection
# -----------------------------------------------------------------------------
def nsga3_environmental_selection(combined, ref_points, target_size):
    """Select target_size individuals from `combined` using NSGA-III rules
    under prec_theta dominance."""
    fits = [ind.aggregatedFitness for ind in combined]
    fronts = _fast_nondominated_sort(fits)
    chosen = []
    for k, front in enumerate(fronts):
        if len(chosen) + len(front) <= target_size:
            chosen.extend(front)
            if len(chosen) == target_size:
                return [combined[i] for i in chosen], k + 1
        else:
            split_k = k
            break
    else:
        return [combined[i] for i in chosen], len(fronts)
    splitting = fronts[split_k]
    num_to_select = target_size - len(chosen)

    pool_idx = chosen + splitting
    pool_fits = np.array([fits[i] for i in pool_idx])
    orient = _orient_for_minimization(pool_fits)
    normalized = _asf_normalize(orient)
    assignment, distances = _associate(normalized, ref_points)

    n_chosen = len(chosen)
    niche_count = np.zeros(len(ref_points), dtype=int)
    for r in assignment[:n_chosen]:
        niche_count[r] += 1
    selected_split = _niching(
        num_to_select,
        splitting,
        assignment[n_chosen:],
        distances[n_chosen:],
        niche_count,
        len(ref_points),
    )
    return [combined[i] for i in chosen + selected_split], split_k + 1


# -----------------------------------------------------------------------------
# Mating selection
# -----------------------------------------------------------------------------
def _binary_tournament_pareto(population, ranks):
    """Tournament on Pareto rank only (NSGA-III standard mating selection)."""
    a = random.randrange(len(population))
    b = random.randrange(len(population))
    if ranks[a] != ranks[b]:
        return population[a] if ranks[a] < ranks[b] else population[b]
    return population[a] if random.random() < 0.5 else population[b]


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def _make_individual(futureData):
    cb, h, b, s, fc, ff, agg, fin, fr, cd, ce = plan.genInvestmentPlan(
        futureData, None, None, None
    )
    return plan.Individual(cb, h, b, s, fc, ff, agg, fin, [], 0.0, fr, cd, [0, 0, 0, 0, 0], ce)


def _make_offspring(futureData, p1, p2, sa_anchor):
    cb, h, b, s, fc, ff, agg, fin, fr, cd, ce = plan.genInvestmentPlan(
        futureData, p1, p2, sa_anchor
    )
    return plan.Individual(cb, h, b, s, fc, ff, agg, fin, [], 0.0, fr, cd, [0, 0, 0, 0, 0], ce)


def nsga3_run(futureData, ref_points, hybrid_sa=False):
    """One NSGA-III run (or SA-NSGA-III if hybrid_sa=True). Returns the run
    result tuple (final_best, runtime_seconds, frontiers_passed,
    last_pivot_gen, final_archive_fits)."""
    # Make plan.bestSASolution see this run's futureData
    plan.futureData = futureData

    N = plan.POPULATION_SIZE
    G_max = plan.MAX_GENERATION
    stop_no_improve = plan.generationCheckedConsecutiveNumToConvergenceDecision

    t0 = time.time()
    population = [_make_individual(futureData) for _ in range(N)]
    sa_incumbent = population[0] if hybrid_sa else None

    fitnessesPivot = []
    no_improve = 0
    frontiers_passed = 1
    last_pivot_gen = 1
    last_split_k = 1

    for g in range(1, G_max + 1):
        # Sort current population by Pareto rank for mating
        cur_fits = [ind.aggregatedFitness for ind in population]
        fronts = _fast_nondominated_sort(cur_fits)
        ranks = [0] * len(population)
        for k, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = k

        # Run-best (for convergence tracking): smallest rank, tie by terminal cash
        best_idx = min(range(len(population)),
                       key=lambda i: (ranks[i], -population[i].aggregatedFitness[4]))
        cur = list(population[best_idx].aggregatedFitness)
        if not fitnessesPivot:
            fitnessesPivot = cur
            last_pivot_gen = g
        elif plan.isDominated(cur, fitnessesPivot):
            fitnessesPivot = cur
            frontiers_passed += 1
            last_pivot_gen = g
            no_improve = 0
        else:
            no_improve += 1

        if no_improve == stop_no_improve or g == G_max:
            break

        # Generate offspring via mating + recombination
        offspring = []
        target_offspring = N
        while len(offspring) < target_offspring:
            p1 = _binary_tournament_pareto(population, ranks)
            p2 = _binary_tournament_pareto(population, ranks)
            offspring.append(_make_offspring(futureData, p1, p2, None))

        # SA exploration (only for SA-NSGA-III)
        if hybrid_sa:
            sa_incumbent = plan.bestSASolution(sa_incumbent)
            offspring[-1] = sa_incumbent  # inject in last offspring slot

        # Environmental selection
        combined = population + offspring
        new_pop, last_split_k = nsga3_environmental_selection(combined, ref_points, N)
        population = new_pop

    runtime = time.time() - t0

    # Final selection: pick run-best from final population
    cur_fits = [ind.aggregatedFitness for ind in population]
    fronts = _fast_nondominated_sort(cur_fits)
    ranks = [0] * len(population)
    for k, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = k
    final_best_idx = min(range(len(population)),
                         key=lambda i: (ranks[i], -population[i].aggregatedFitness[4]))
    final_best = population[final_best_idx]
    # Final archive = first front of population
    archive_fits = [list(map(float, population[i].aggregatedFitness)) for i in fronts[0]]
    return final_best, runtime, frontiers_passed, last_pivot_gen, archive_fits


def run_experiments(name, n_runs=30, seed_base=1234, hybrid_sa=False, ref_p=4):
    futureData = _load_dataset(name)
    ref_points = das_dennis(5, ref_p)
    label = "sa_nsga3" if hybrid_sa else "nsga3"
    print(f"[{name}/{label}] {len(ref_points)} reference points (Das-Dennis p={ref_p})")
    log_times, log_dn, log_gn, log_fits, fit_pivot, archives = [], [], [], [], [], []
    for r in range(n_runs):
        random.seed(seed_base + r)
        np.random.seed(seed_base + r)
        best, dt, fp, lpg, arch = nsga3_run(futureData, ref_points, hybrid_sa=hybrid_sa)
        log_times.append(dt)
        log_dn.append(fp)
        log_gn.append(lpg)
        log_fits.append([float(x) for x in best.aggregatedFitness])
        fit_pivot.append([float(x) for x in best.aggregatedFitness])
        archives.append(arch)
        print(f"[{name}/{label}] run {r+1}/{n_runs}: t={dt:.2f}s "
              f"profit={log_fits[-1][4]:.2f} dn={fp} gn={lpg} archive={len(arch)}",
              flush=True)
    out = {
        "dataset": name,
        "method": label,
        "n_runs": n_runs,
        "ref_points_p": ref_p,
        "n_ref_points": len(ref_points),
        "log_times": log_times,
        "log_dn": log_dn,
        "log_gn": log_gn,
        "log_fitnesses": log_fits,
        "fitness_pivot": fit_pivot,
        "archives": archives,
    }
    fname = f"{label}_results_{name}.json"
    with open(fname, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {fname}")
    return out


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        args = ["vn30", "hnx30"]
    for ds in args:
        run_experiments(ds, n_runs=30, hybrid_sa=False)
        run_experiments(ds, n_runs=30, hybrid_sa=True)
