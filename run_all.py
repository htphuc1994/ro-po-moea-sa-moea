"""Run the experiment matrix (Section 5.2) in parallel; one JSON per run.

Output: results/<instance>/<label>/seed_<seed>.json (existing files are skipped, so an
interrupted run resumes). Labels: <method>[+sa|+ri][@lex][-nodiv][-K<k>][-chi<chi0>].

Blocks:
  main      all instances x {ga, nsga2, spea2, nsga3, moead} + SA and random-injection variants
  lex       2023 instances x {nsga2, spea2, nsga3} with the first-round lexicographic rule
  extra     2023 instances x the chosen solver: no dividends, K = 15, K = 25
  elite     2023 instances x {nsga2, spea2, nsga3, moead} with the elite-anchored SA (+sae) and its
            random-injection control (+rie): the pre-specified ablation of the SA anchor
  chi0      2023 instances x {nsga2, spea2, nsga3, moead} + SA with a starting acceptance
            probability of 0.2 and 0.8 instead of 0.5 (sensitivity of the SA temperature)
  timing   2023 instances x main and elite configurations, 10 seeds, one process, seed-major order
            (runtime comparison)

Usage:
  python3 run_all.py main --workers 9
  python3 run_all.py lex extra --solver nsga3 --workers 9
  python3 run_all.py timing
"""
import os

# One BLAS thread per process: the workers already use the cores, and the timing block is
# reported as single-threaded. Set before numpy is imported here or in spawned workers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
INST = os.path.join(ROOT, "data", "instances")
SEEDS = list(range(1234, 1264))
BASES = ["ga", "nsga2", "spea2", "nsga3", "moead"]
MOEAS = ["nsga2", "spea2", "nsga3", "moead"]
MAIN = [(m, None) for m in BASES] + [(m, "sa") for m in MOEAS] + [(m, "ri") for m in MOEAS]
ELITE = [(m, "sae") for m in MOEAS] + [(m, "rie") for m in MOEAS]
CHI0 = (0.2, 0.8)     # starting acceptance probabilities of the sensitivity block (main runs: 0.5)


def label(method, sa=None, rule="constrained", nodiv=False, k=None, chi0=None):
    s = method + (f"+{sa}" if sa else "") + ("@lex" if rule == "lexicographic" else "")
    return s + ("-nodiv" if nodiv else "") + (f"-K{k}" if k else "") + (f"-chi{chi0:g}" if chi0 is not None else "")


def instances(years=None):
    names = sorted(f[:-5] for f in os.listdir(INST) if f.endswith(".json"))
    return [n for n in names if years is None or any(n.endswith(str(y)) for y in years)]


_loaded = None


def _load(instance, nodiv, k):
    global _loaded
    import plan
    key = (instance, nodiv, k)
    if _loaded != key:
        plan.load_instance(os.path.join(INST, f"{instance}.json"), os.path.join(INST, f"{instance}_moments.npz"))
        if nodiv:
            plan.DIVS[:] = 0.0
        plan.cardinalityK = k or 20
        _loaded = key


def work(job):
    instance, method, sa, rule, nodiv, k, seed, budget, outdir, snapshots, *rest = job
    chi0 = rest[0] if rest else None
    name = label(method, sa, rule, nodiv, k, chi0)
    path = os.path.join(outdir, instance, name, f"seed_{seed}.json")
    if os.path.exists(path):
        return path, "skip"
    import solvers
    _load(instance, nodiv, k)
    res = solvers.run(method, seed, budget=budget, sa_mode=sa, rule=rule,
                      snapshot_every=2500 if snapshots else None, sa_chi0=chi0)
    res.update({"instance": instance, "label": name, "nodiv": nodiv, "K": k or 20})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return path, f"{res['runtime']:.1f}s"


def jobs_for(block, solver, budget, seeds=SEEDS, out=None, only=None):
    out, j = out or os.path.join(ROOT, "results_timing" if block == "timing" else "results"), []
    y2023 = instances([2023])
    every = instances() if only is None else [i for i in instances() if i in only]
    y2023 = [i for i in y2023 if only is None or i in only]
    if block == "main":
        j = [(i, m, sa, "constrained", False, None, s, budget, out, True) for i in every for m, sa in MAIN for s in seeds]
    elif block == "lex":
        j = [(i, m, None, "lexicographic", False, None, s, budget, out, True) for i in y2023 if not i.startswith("combined")
             for m in ("nsga2", "spea2", "nsga3") for s in seeds]
    elif block == "extra":
        for i in (x for x in y2023 if not x.startswith("combined")):
            j += [(i, solver, None, "constrained", True, None, s, budget, out, True) for s in seeds]
            j += [(i, solver, None, "constrained", False, k, s, budget, out, True) for k in (15, 25) for s in seeds]
    elif block == "elite":
        # pre-specified on the three 2023 instances (including the combined universe)
        j = [(i, m, sa, "constrained", False, None, s, budget, out, True) for i in y2023 for m, sa in ELITE for s in seeds]
    elif block == "chi0":
        # sensitivity of the SA starting temperature, on the three 2023 instances
        j = [(i, m, "sa", "constrained", False, None, s, budget, out, True, c)
             for i in y2023 for m in MOEAS for c in CHI0 for s in seeds]
    elif block == "timing":
        # seed-major order: every configuration runs once per seed before the next seed starts, so
        # changes in background load are spread over all configurations instead of hitting one block
        j = [(i, m, sa, "constrained", False, None, s, budget, out, False) for s in seeds[:10]
             for i in y2023 if not i.startswith("combined") for m, sa in MAIN + ELITE]
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks", nargs="+", choices=["main", "lex", "extra", "elite", "chi0", "timing"])
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--budget", type=int, default=10_000)
    ap.add_argument("--solver", default="spea2", help="solver for the 'extra' block")
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="use the first N seeds")
    ap.add_argument("--out", default=None, help="results directory (default results/ or results_timing/)")
    ap.add_argument("--instances", nargs="*", default=None, help="restrict to these instance names")
    args = ap.parse_args()
    for block in args.blocks:
        jobs = jobs_for(block, args.solver, args.budget, SEEDS[: args.seeds],
                        os.path.join(ROOT, args.out) if args.out else None, args.instances)
        workers = 1 if block == "timing" else args.workers
        t0, done = time.time(), 0
        print(f"[{block}] {len(jobs)} runs on {workers} process(es)", flush=True)
        with Pool(workers) as pool:
            for path, status in pool.imap_unordered(work, jobs, chunksize=1):
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"[{block}] {done}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
