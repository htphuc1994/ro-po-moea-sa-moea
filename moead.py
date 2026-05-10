"""MOEA/D (Zhang & Li, 2007) for the dividend-portfolio problem.

Decomposition-based alternative to the Pareto-based comparators.  We use:
  - Das-Dennis weight vectors on the 5-D simplex (p=4 -> 70 vectors)
  - Tchebycheff scalarization with running ideal point z*
  - Neighborhood size T (default 20), neighborhood-vs-population mating
    probability delta (default 0.9), and per-offspring replacement cap n_r
    (default 2).
  - Constraint handling: theta_tau >= (1+upsilon)*Theta is treated as a hard
    feasibility check; feasible always preferred over infeasible at the
    replacement step, ties broken by Tchebycheff.

The constructive sampler (`plan.genInvestmentPlan`) and the mixed
min/max objective vector are reused from plan.py.

Output JSON matches the format of `spea2_results_*.json` and
`nsga3_results_*.json` so that `analyze_with_baselines.py` can ingest it.
"""

from __future__ import annotations

import json
import random
import time
import numpy as np
import plan


# -----------------------------------------------------------------------------
# Data loading
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
# Weight vectors and neighborhoods
# -----------------------------------------------------------------------------
def das_dennis(M, p):
    """Generate Das-Dennis points on the (M-1)-simplex with p divisions."""
    points = []

    def recurse(remaining_axes, remaining_sum, current):
        if remaining_axes == 1:
            points.append(current + [remaining_sum / p])
            return
        for i in range(remaining_sum + 1):
            recurse(remaining_axes - 1, remaining_sum - i, current + [i / p])

    recurse(M, p, [])
    arr = np.array(points)
    # Avoid zero weights (cause division-by-zero in Tchebycheff)
    arr = np.where(arr < 1e-6, 1e-6, arr)
    # Normalize to row-sum 1 after the floor
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def _build_neighborhoods(weights, T):
    n = weights.shape[0]
    # Pairwise Euclidean distances
    diffs = weights[:, None, :] - weights[None, :, :]
    d = np.sqrt((diffs * diffs).sum(axis=2))
    # T closest indices for each row (including self)
    idx = np.argsort(d, axis=1)[:, :T]
    return idx


# -----------------------------------------------------------------------------
# Scalarization and constraint handling
# -----------------------------------------------------------------------------
def _orient_for_minimization(fits_5):
    """Negate maximize-direction objectives so Tchebycheff treats every
    component as 'smaller is better'."""
    arr = np.array(fits_5, dtype=float)
    if arr.ndim == 1:
        arr = arr.copy()
        arr[1] = -arr[1]
        arr[4] = -arr[4]
        return arr
    arr = arr.copy()
    arr[:, 1] = -arr[:, 1]
    arr[:, 4] = -arr[:, 4]
    return arr


def _tchebycheff(obj_min, weight, z_star):
    return float(np.max(weight * np.abs(obj_min - z_star)))


def _is_feasible(fit_5):
    """theta_tau (= aggregatedFitness[4]) >= (1+upsilon)*Theta"""
    return fit_5[4] >= plan.initTheta * (1.0 + plan.upsilon)


def _violation(fit_5):
    """Single-axis violation magnitude (0 if feasible)."""
    target = plan.initTheta * (1.0 + plan.upsilon)
    return max(0.0, target - fit_5[4]) / target


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def _make_individual(futureData, p1=None, p2=None, sa_anchor=None):
    cb, h, b, s, fc, ff, agg, fin, fr, cd, ce = plan.genInvestmentPlan(
        futureData, p1, p2, sa_anchor
    )
    return plan.Individual(cb, h, b, s, fc, ff, agg, fin, [], 0.0, fr, cd, [0, 0, 0, 0, 0], ce)


def moead_run(futureData, weights, neighborhoods, delta=0.9, n_r=2,
              G_max=None, stop_no_improve=None, hybrid_sa=False):
    """One MOEA/D run (or SA-MOEA/D if hybrid_sa=True; SA incumbent replaces
    pop[N-1] each generation, mirroring HSG's injection rule).
    Returns (final_best, runtime, frontiers_passed, last_pivot_gen,
    archive_fits)."""
    if G_max is None:
        G_max = plan.MAX_GENERATION
    if stop_no_improve is None:
        stop_no_improve = plan.generationCheckedConsecutiveNumToConvergenceDecision
    plan.futureData = futureData

    N = weights.shape[0]
    population = [_make_individual(futureData) for _ in range(N)]
    pop_fits = [list(map(float, ind.aggregatedFitness)) for ind in population]
    pop_min = _orient_for_minimization(pop_fits)
    z_star = pop_min.min(axis=0)
    sa_anchor = population[0] if hybrid_sa else None

    fitnessesPivot = []
    no_improve = 0
    frontiers_passed = 1
    last_pivot_gen = 1

    t0 = time.time()
    for g in range(1, G_max + 1):
        for i in range(N):
            # Mating pool
            if random.random() < delta:
                pool = list(neighborhoods[i])
            else:
                pool = list(range(N))
            p1_idx = random.choice(pool)
            p2_idx = random.choice(pool)
            child = _make_individual(futureData, population[p1_idx], population[p2_idx], None)
            child_fit = list(map(float, child.aggregatedFitness))
            child_min = _orient_for_minimization(child_fit)
            # Update z*
            z_star = np.minimum(z_star, child_min)

            # Replacement: walk pool in random order, replace up to n_r
            candidates = list(pool)
            random.shuffle(candidates)
            replaced = 0
            for j in candidates:
                if replaced >= n_r:
                    break
                w = weights[j]
                feas_child = _is_feasible(child_fit)
                feas_cur = _is_feasible(pop_fits[j])
                cur_min = pop_min[j]
                if feas_child and not feas_cur:
                    accept = True
                elif feas_cur and not feas_child:
                    accept = False
                elif not feas_child and not feas_cur:
                    accept = _violation(child_fit) <= _violation(pop_fits[j])
                else:
                    accept = _tchebycheff(child_min, w, z_star) <= _tchebycheff(cur_min, w, z_star)
                if accept:
                    population[j] = child
                    pop_fits[j] = child_fit
                    pop_min[j] = child_min
                    replaced += 1

        # SA exploration + injection (mirrors HSG: replace pop[N-1])
        if hybrid_sa:
            sa_anchor = plan.bestSASolution(sa_anchor)
            sa_fit = list(map(float, sa_anchor.aggregatedFitness))
            population[N - 1] = sa_anchor
            pop_fits[N - 1] = sa_fit
            pop_min[N - 1] = _orient_for_minimization(sa_fit)
            z_star = np.minimum(z_star, pop_min[N - 1])

        # Convergence: best-by-terminal-cash among feasible solutions
        feasible_idx = [i for i in range(N) if _is_feasible(pop_fits[i])]
        candidate_idx = feasible_idx if feasible_idx else list(range(N))
        best_idx = max(candidate_idx, key=lambda i: pop_fits[i][4])
        cur = list(pop_fits[best_idx])
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

    runtime = time.time() - t0

    # Final reporting: run-best and approximate first front by Pareto rank
    feasible_idx = [i for i in range(N) if _is_feasible(pop_fits[i])]
    candidate_idx = feasible_idx if feasible_idx else list(range(N))
    final_best_idx = max(candidate_idx, key=lambda i: pop_fits[i][4])
    final_best = population[final_best_idx]

    # First-front approximation: items not strictly dominated by any other under prec_theta
    archive_fits = []
    for i in range(N):
        dominated = False
        for j in range(N):
            if i == j:
                continue
            if plan.is_belonged_frontier(pop_fits[j], pop_fits[i]) and \
               not plan.is_belonged_frontier(pop_fits[i], pop_fits[j]):
                dominated = True
                break
        if not dominated:
            archive_fits.append(list(map(float, pop_fits[i])))

    return final_best, runtime, frontiers_passed, last_pivot_gen, archive_fits


def run_experiments(name, n_runs=30, seed_base=1234, ref_p=4, T=20,
                    delta=0.9, n_r=2, hybrid_sa=False):
    futureData = _load_dataset(name)
    weights = das_dennis(5, ref_p)
    neighborhoods = _build_neighborhoods(weights, T)
    label = "sa_moead" if hybrid_sa else "moead"
    print(f"[{name}/{label}] Das-Dennis p={ref_p} -> N={len(weights)} weight vectors, "
          f"T={T}, delta={delta}, n_r={n_r}")
    log_times, log_dn, log_gn, log_fits, fit_pivot, archives = [], [], [], [], [], []
    for r in range(n_runs):
        random.seed(seed_base + r)
        np.random.seed(seed_base + r)
        best, dt, fp, lpg, arch = moead_run(
            futureData, weights, neighborhoods,
            delta=delta, n_r=n_r, hybrid_sa=hybrid_sa,
        )
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
        "n_ref_points": len(weights),
        "T": T,
        "delta": delta,
        "n_r": n_r,
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
    print(f"\n[{name}/{label}] wrote {fname}\n", flush=True)
    return out


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    mode = "vanilla"
    if args and args[0] in ("--sa", "--both"):
        mode = args[0][2:]
        args = args[1:]
    if not args:
        args = ["vn30", "hnx30"]
    for ds in args:
        if mode in ("vanilla", "both"):
            run_experiments(ds, n_runs=30, hybrid_sa=False)
        if mode in ("sa", "both"):
            run_experiments(ds, n_runs=30, hybrid_sa=True)
