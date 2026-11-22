"""SPEA2 baseline (Zitzler, Laumanns & Thiele, 2001) reusing the dividend-
portfolio constructive sampler from plan.py.

For fairness with the NSGA-II and HSG comparators in plan.py, SPEA2 uses the
same return-priority dominance prec_theta (plan.is_belonged_frontier) for the
strength calculation.  All other hyperparameters (population size, max
generations, no-improvement stopping rule, recombination operator) match the
existing solvers.

Output format mirrors what plan.py's `prinfReSult` writes for its three
algorithms:
  log_times[r], log_dn[r], log_gn[r], log_fitnesses[r] (5-vector aggregated
  fitness of the run-best individual), and the full final archive of
  aggregatedFitnesses for CS / SP computation.
"""

import json
import math
import random
import time
import numpy as np
import plan


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


def _strict_prec_theta(a, b):
    """Strict return-priority dominance: a precedes b iff
    plan.is_belonged_frontier(a, b) and NOT is_belonged_frontier(b, a)."""
    if not plan.is_belonged_frontier(a, b):
        return False
    return not plan.is_belonged_frontier(b, a)


def _compute_spea2_fitness(individuals, k):
    """SPEA2 fitness: F(i) = R(i) + D(i).
    R(i) = sum of S(j) for all j that strictly dominate i.
    S(j) = number of i's that j strictly dominates.
    D(i) = 1 / (sigma_k(i) + 2), with sigma_k computed in [0,1]-normalized
    objective space.
    Returns a list of (raw_fitness, density, total) triples in the same order.
    """
    N = len(individuals)
    fits = [ind.aggregatedFitness for ind in individuals]

    # Strength S(j)
    S = [0] * N
    dominates = [[False] * N for _ in range(N)]
    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            if _strict_prec_theta(fits[j], fits[i]):
                dominates[j][i] = True
                S[j] += 1

    # Raw fitness R(i): sum of strengths of strict dominators of i
    R = [0] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates[j][i]:
                R[i] += S[j]

    # Density D(i): k-NN distance in normalized objective space
    arr = np.array(fits, dtype=float)
    fmin = arr.min(axis=0)
    fmax = arr.max(axis=0)
    span = np.where(fmax - fmin > 0, fmax - fmin, 1.0)
    norm = (arr - fmin) / span
    D = [0.0] * N
    for i in range(N):
        diffs = norm - norm[i]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        dists = np.sort(dists)
        # dists[0] is self-distance 0; skip it; use k-th smallest non-self
        idx = min(k, len(dists) - 1)
        sigma_k = dists[idx]
        D[i] = 1.0 / (sigma_k + 2.0)

    return [(R[i], D[i], R[i] + D[i]) for i in range(N)]


def _spea2_environmental_selection(individuals, fitnesses, archive_size):
    """Build the next archive of size archive_size from `individuals`."""
    # Non-dominated subset: those with R == 0 (but we include D in F)
    # Standard SPEA2: non-dom iff F < 1 (because R == 0 -> F < 1; dominated -> F >= 1)
    nondom_idx = [i for i, (R, D, F) in enumerate(fitnesses) if R == 0]
    if len(nondom_idx) <= archive_size:
        # Fill with dominated by smallest F
        chosen = list(nondom_idx)
        rest = sorted(
            [i for i, (R, D, F) in enumerate(fitnesses) if R > 0],
            key=lambda i: fitnesses[i][2],
        )
        chosen.extend(rest[: archive_size - len(chosen)])
        return chosen
    # Too many non-dominated: truncate by smallest-distance iterative removal
    # (use F = D within non-dom; smaller D = larger sigma_k = keep)
    # Use a simpler heuristic: sort by F descending, drop largest F first
    sorted_nondom = sorted(nondom_idx, key=lambda i: fitnesses[i][2])
    return sorted_nondom[:archive_size]


def _binary_tournament(archive, fitnesses_archive):
    """Pick the better of two random archive members."""
    a = random.randrange(len(archive))
    b = random.randrange(len(archive))
    if fitnesses_archive[a][2] <= fitnesses_archive[b][2]:
        return archive[a]
    return archive[b]


def spea2_run(futureData, log_times, log_dn, log_gn, log_fitnesses,
              fitness_pivot, archives_log, hybrid_sa=False):
    """One SPEA2 run that mirrors the per-run logging of plan.exe_experiments.
    If hybrid_sa=True, an SA exploratory walk runs each generation and the
    incumbent is injected into the last offspring slot, mirroring HSG."""
    N = plan.POPULATION_SIZE
    A = plan.POPULATION_SIZE
    G_max = plan.MAX_GENERATION
    stop_no_improve = plan.generationCheckedConsecutiveNumToConvergenceDecision
    k_neighbor = max(1, int(round(math.sqrt(N + A))))

    plan.futureData = futureData

    t0 = time.time()

    # Initial population
    population = []
    for _ in range(N):
        cb, h, b, s, fc, ff, agg, fin, fr, cd, ce = plan.genInvestmentPlan(
            futureData, None, None, None
        )
        population.append(
            plan.Individual(cb, h, b, s, fc, ff, agg, fin, [], 0.0, fr, cd, [0, 0, 0, 0, 0], ce)
        )
    archive = []  # external archive
    sa_anchor = population[0] if hybrid_sa else None

    fitnessesPivot = []
    no_improve = 0
    frontiers_passed = 1  # baseline: 1 (the initial non-dominated front)
    last_pivot_gen = 1

    for g in range(1, G_max + 1):
        combined = population + archive
        F_triples = _compute_spea2_fitness(combined, k_neighbor)
        archive_idx = _spea2_environmental_selection(combined, F_triples, A)
        archive = [combined[i] for i in archive_idx]
        archive_F = [F_triples[i] for i in archive_idx]

        # Convergence: best individual in archive (smallest F)
        best_idx_in_arch = min(range(len(archive)), key=lambda i: archive_F[i][2])
        cur = archive[best_idx_in_arch].aggregatedFitness

        if not fitnessesPivot:
            fitnessesPivot = list(cur)
            last_pivot_gen = g
        elif plan.isDominated(cur, fitnessesPivot):
            fitnessesPivot = list(cur)
            frontiers_passed += 1
            last_pivot_gen = g
            no_improve = 0
        else:
            no_improve += 1

        if no_improve == stop_no_improve or g == G_max:
            break

        # Mating + variation
        new_pop = []
        # Elitism: keep the top 10% from archive (lowest F)
        elite_n = max(1, N // 10)
        elite = sorted(range(len(archive)), key=lambda i: archive_F[i][2])[:elite_n]
        for ei in elite:
            new_pop.append(archive[ei])
        while len(new_pop) < N:
            p1 = _binary_tournament(archive, archive_F)
            p2 = _binary_tournament(archive, archive_F)
            cb, h, b, s, fc, ff, agg, fin, fr, cd, ce = plan.genInvestmentPlan(
                futureData, p1, p2, None
            )
            new_pop.append(
                plan.Individual(cb, h, b, s, fc, ff, agg, fin, [], 0.0, fr, cd, [0, 0, 0, 0, 0], ce)
            )

        # SA exploration + injection (mirrors HSG: replace last offspring slot)
        if hybrid_sa:
            sa_anchor = plan.bestSASolution(sa_anchor)
            new_pop[-1] = sa_anchor

        population = new_pop

    runtime = time.time() - t0
    # Final archive recomputation to log Pareto-set fitnesses
    final_archive_fits = [list(map(float, ind.aggregatedFitness)) for ind in archive]
    # Match NSGA-II's run-best selection rule (plan.py:763): non-dominated front
    # first, then most-isolated by density (smallest D = largest sigma_k), then
    # break ties by highest terminal cash.  R serves as the front-proxy
    # (R == 0 iff non-dominated under prec_theta).
    final_F = _compute_spea2_fitness(archive, k_neighbor)
    best_idx_final = min(
        range(len(archive)),
        key=lambda i: (final_F[i][0], final_F[i][1], -archive[i].aggregatedFitness[4]),
    )
    final_best = archive[best_idx_final]
    log_times.append(runtime)
    log_dn.append(frontiers_passed)
    log_gn.append(last_pivot_gen)
    log_fitnesses.append([float(x) for x in final_best.aggregatedFitness])
    fitness_pivot.append([float(x) for x in fitnessesPivot])
    archives_log.append(final_archive_fits)
    return final_best, runtime, frontiers_passed, last_pivot_gen


def run_experiments(name, n_runs=30, seed_base=1234, hybrid_sa=False):
    futureData = _load_dataset(name)
    label = "sa_spea2" if hybrid_sa else "spea2"
    log_times = []
    log_dn = []
    log_gn = []
    log_fitnesses = []
    fitness_pivot = []
    archives_log = []
    for r in range(n_runs):
        random.seed(seed_base + r)
        np.random.seed(seed_base + r)
        spea2_run(
            futureData, log_times, log_dn, log_gn, log_fitnesses,
            fitness_pivot, archives_log, hybrid_sa=hybrid_sa,
        )
        print(f"[{name}/{label}] run {r + 1}/{n_runs}: t={log_times[-1]:.2f}s "
              f"profit={log_fitnesses[-1][4]:.2f} dn={log_dn[-1]} gn={log_gn[-1]}",
              flush=True)
    out = {
        "dataset": name,
        "method": label,
        "n_runs": n_runs,
        "log_times": log_times,
        "log_dn": log_dn,
        "log_gn": log_gn,
        "log_fitnesses": log_fitnesses,
        "fitness_pivot": fitness_pivot,
        "archives": archives_log,
    }
    fname = f"{label}_results_{name}.json"
    with open(fname, "w") as f:
        json.dump(out, f)
    print(f"\n[{name}/{label}] wrote {fname}\n", flush=True)
    return out


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    # Allow flag like:  python3 spea2.py vn30 hnx30  (vanilla)
    #                   python3 spea2.py --sa vn30 hnx30  (SA-SPEA2)
    #                   python3 spea2.py --both vn30 hnx30  (both modes)
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
