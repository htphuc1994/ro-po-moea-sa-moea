"""Rerun GA, NSGA-II, and HSG (= SA-NSGA-II) on the current machine and
save per-method results as JSON in the same format as `spea2_results_*.json`
and `nsga3_results_*.json`.

The legacy arrays embedded in `resultAnalyzers_withSignificantTests.py` are
kept as a fallback inside `analyze_with_baselines.py`, but once these JSONs
exist they take precedence (so runtimes reflect the current machine).
"""

import json
import random
import time
import sys
import numpy as np
import plan


METHODS = [
    # (label, isNSGA_II, isHybridWithSA)
    ("ga",        False, False),
    ("nsga_ii",   True,  False),
    ("hsg",       True,  True),
]


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


def run_method(dataset, label, isNSGA_II, isHybrid, n_runs=30, seed_base=1234):
    futureData = _load_dataset(dataset)
    plan.futureData = futureData  # so plan.bestSASolution can see it

    log_times = []
    log_dn = []
    log_gn = []
    log_fitnesses = []
    fitness_pivot = []
    archives = []  # placeholder per-run [run-best] until plan.ga exposes the full first front

    for r in range(n_runs):
        random.seed(seed_base + r)
        np.random.seed(seed_base + r)
        t0 = time.time()
        plan.ga(futureData, isNSGA_II, isHybrid, log_dn, log_gn, log_fitnesses, fitness_pivot)
        dt = time.time() - t0
        log_times.append(dt)
        archives.append([list(map(float, log_fitnesses[-1]))])
        print(f"[{dataset}/{label}] run {r+1}/{n_runs}: t={dt:.2f}s "
              f"profit={log_fitnesses[-1][4]:.2f} dn={log_dn[-1]} gn={log_gn[-1]}",
              flush=True)

    out = {
        "dataset": dataset,
        "method": label,
        "n_runs": n_runs,
        "log_times": log_times,
        "log_dn": log_dn,
        "log_gn": log_gn,
        "log_fitnesses": log_fitnesses,
        "fitness_pivot": fitness_pivot,
        "archives": archives,
    }
    fname = f"{label}_results_{dataset}.json"
    with open(fname, "w") as f:
        json.dump(out, f)
    print(f"\n[{dataset}/{label}] wrote {fname}\n", flush=True)
    return out


def main():
    args = sys.argv[1:]
    if not args:
        args = ["vn30", "hnx30"]

    # Silence the per-run console spam from plan.printSolution / prinfReSult
    plan.printSolution = lambda *_args, **_kw: None

    for dataset in args:
        for label, isNSGA_II, isHybrid in METHODS:
            run_method(dataset, label, isNSGA_II, isHybrid)


if __name__ == "__main__":
    main()
