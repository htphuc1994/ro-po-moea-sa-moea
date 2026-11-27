"""Exact maximum terminal cash on small instances (reviewer request: exact or bounded solutions).

With prices known in advance, the cash recursion (Eq. 21) is linear in the trades, and every
trading rule of Section 3.1 is linear once lots, buy/sell indicators and holding indicators are
introduced. Maximizing theta_tau is therefore a mixed-integer linear program, solved here with
HiGHS through scipy.optimize.milp. The risk objectives are dropped: the optimum is the
cash-extreme point of the true Pareto front, an upper bound for any plan's terminal cash.

Variables per (month t, stock j): b (lots bought), s (lots sold), h (lots held at the end of t),
u, v (buy / sell indicators), z (holding indicator); per month: d (cash on deposit), theta.
"""
import json
import os

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import checker
import plan


def solve(time_limit=600, big_cash=20.0):
    P, Q, D = plan.PRICES, plan.VOLUMES, plan.DIVS
    tau, n, xi, eps, alpha, K = plan.tau, P.shape[0], plan.xi, plan.epsilon, plan.alpha, plan.cardinalityK
    blocks = ["b", "s", "h", "u", "v", "z"]
    nv_tj = len(blocks) * tau * n
    off = {name: k * tau * n for k, name in enumerate(blocks)}
    d0, th0 = nv_tj, nv_tj + tau
    nvar = th0 + tau + 1

    def ix(name, t, j):
        return off[name] + t * n + j

    lb, ub = np.zeros(nvar), np.full(nvar, np.inf)
    integ = np.zeros(nvar)
    for name in ("b", "s", "h"):
        integ[off[name]: off[name] + tau * n] = 1
    for name in ("u", "v", "z"):
        integ[off[name]: off[name] + tau * n] = 1
        ub[off[name]: off[name] + tau * n] = 1
    lb[th0] = ub[th0] = plan.initTheta

    rows, lo, hi = [], [], []

    def add(coefs, l, u):
        rows.append(coefs)
        lo.append(l)
        hi.append(u)

    hmax = np.zeros(n)
    for j in range(n):
        pmin = P[j][P[j] > 0].min() if (P[j] > 0).any() else 1.0
        hmax[j] = np.ceil(big_cash * plan.initTheta / (pmin * xi))

    for t in range(tau):
        for j in range(n):
            c = {ix("h", t, j): 1.0, ix("b", t, j): -1.0, ix("s", t, j): 1.0}       # Eq. (14)
            if t >= 1:
                c[ix("h", t - 1, j)] = -1.0
            add(c, 0.0, 0.0)
            tradable = P[j, t] > 0 and Q[j, t] > 0
            last = t == tau - 1
            # buys: Eq. (10), (24), tradability and liquidity in the last month
            if t <= tau - 3 and tradable and Q[j, tau - 1] > 0:
                add({ix("b", t, j): xi, ix("u", t, j): -Q[j, t]}, -np.inf, 0.0)
            else:
                ub[ix("b", t, j)] = 0
            # sells
            if last:
                c = {ix("s", t, j): 1.0}                                            # Eq. (23)
                if t >= 1:
                    c[ix("h", t - 1, j)] = -1.0
                add(c, 0.0, 0.0)
                add({ix("s", t, j): 1.0, ix("v", t, j): -hmax[j]}, -np.inf, 0.0)
            elif t >= 1 and tradable and D[j, t] == 0:                              # Eq. (8)
                add({ix("s", t, j): xi, ix("v", t, j): -Q[j, t]}, -np.inf, 0.0)    # Eq. (10)
                add({ix("s", t, j): 1.0, ix("h", t - 1, j): -1.0}, -np.inf, 0.0)   # Eq. (6)
                if t >= 2:
                    add({ix("s", t, j): 1.0, ix("h", t - 2, j): -1.0}, -np.inf, 0.0)   # Eq. (7)
                else:
                    ub[ix("s", t, j)] = 0
            else:
                ub[ix("s", t, j)] = 0
            add({ix("u", t, j): 1.0, ix("v", t, j): 1.0}, -np.inf, 1.0)             # Eq. (12)
            add({ix("h", t, j): 1.0, ix("z", t, j): -hmax[j]}, -np.inf, 0.0)        # z = 1 if held
        add({ix("z", t, j): 1.0 for j in range(n)}, -np.inf, K)                     # Eq. (15)
        # deposit d_t = theta_t - (1 + eps) * cost_t >= 0                           Eq. (9)
        c = {th0 + t: 1.0, d0 + t: -1.0}
        for j in range(n):
            c[ix("b", t, j)] = -(1 + eps) * P[j, t] * xi
        add(c, 0.0, 0.0)
        # theta_{t+1} = (1 + alpha) d_t + (1 - eps) sales_t + dividends               Eq. (21)
        c = {th0 + t + 1: 1.0, d0 + t: -(1 + alpha)}
        for j in range(n):
            c[ix("s", t, j)] = c.get(ix("s", t, j), 0.0) - (1 - eps) * P[j, t] * xi
            if t + 1 >= 2 and D[j, t] > 0:
                c[ix("h", t - 1, j)] = c.get(ix("h", t - 1, j), 0.0) - D[j, t] * xi
        add(c, 0.0, 0.0)
    for j in range(n):
        ub[ix("h", tau - 1, j)] = 0

    A = np.zeros((len(rows), nvar))
    for r, coefs in enumerate(rows):
        for k, v in coefs.items():
            A[r, k] = v
    cvec = np.zeros(nvar)
    cvec[th0 + tau] = -1.0
    res = milp(cvec, integrality=integ, bounds=Bounds(lb, ub),
               constraints=[LinearConstraint(A, lo, hi)],
               options={"time_limit": time_limit, "mip_rel_gap": 1e-9, "disp": False})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = np.round(res.x).astype(np.int64)
    y0 = np.array([[x[ix("b", t, j)] * xi for j in range(n)] for t in range(tau)])
    y1 = np.array([[x[ix("s", t, j)] * xi for j in range(n)] for t in range(tau)])
    return {"theta_tau": float(-res.fun), "status": res.message, "gap": getattr(res, "mip_gap", None),
            "y0": y0, "y1": y1, "n_vars": nvar, "n_rows": len(rows)}


def validate(sol):
    """The MILP plan must pass the independent checker and the simulator must reproduce theta_tau."""
    ind = plan.simulate(sol["y0"], sol["y1"])
    bad, _ = checker.check(ind)
    return ind, bad, abs(ind.theta[-1] - sol["theta_tau"]) / sol["theta_tau"]


def small_instance(src_json, n_stocks=8, months=(1, 6)):
    """Reduced instance: the n most liquid stocks (January volume) and a month window."""
    data = json.load(open(src_json))
    first, last = months
    ranked = sorted(data["stocks"], key=lambda s: -next(p["matchedTradingVolume"] for p in s["prices"] if p["month"] == first))
    stocks = []
    for s in ranked[:n_stocks]:
        stocks.append({**s,
                       "prices": [{**p, "month": p["month"] - first + 1} for p in s["prices"] if first <= p["month"] <= last],
                       "dividendSpitingHistories": [{**d, "month": d["month"] - first + 1}
                                                    for d in s["dividendSpitingHistories"] if first <= d["month"] <= last]})
    return {"stocks": stocks}, [s["symbol"] for s in ranked[:n_stocks]]


def submoments(npz_path, symbols):
    """Moment inputs of a stock subset. Centring is column-wise, so the sub-matrix of C is the centred
    return matrix of the subset; files without C (full tensors only) are reduced tensor by tensor."""
    mom = np.load(npz_path)
    full = list(mom["tickers"])
    idx = [full.index(s) for s in symbols]
    out = {"M2": mom["M2"][np.ix_(idx, idx)], "tickers": np.array(symbols)}
    if "C" in mom:
        out["C"] = mom["C"][:, idx]
        return out
    n, m = len(full), len(idx)
    out["M3"] = mom["M3"].reshape(n, n, n)[np.ix_(idx, idx, idx)].reshape(m, m * m)
    out["M4"] = mom["M4"].reshape(n, n, n, n)[np.ix_(idx, idx, idx, idx)].reshape(m, m ** 3)
    return out


ROOT = os.path.dirname(os.path.abspath(__file__))
SMALL = [("vn30_2023", (1, 6)), ("hnx30_2023", (1, 6)), ("vn30_2023", (7, 12))]


def _setup(name, months):
    plan.tau = months[1] - months[0] + 1
    data, syms = small_instance(os.path.join(ROOT, "data", "instances", f"{name}.json"), 8, months)
    plan.load_instance(data, submoments(os.path.join(ROOT, "data", "instances", f"{name}_moments.npz"), syms))
    return syms


def _solver_job(job):
    """One solver run on a reduced instance. Returns the terminal cash Table 6 uses (best target-feasible
    plan of the final set) and that reported plan, re-simulated and checked by the independent checker."""
    import solvers
    name, months, method, seed, budget = job
    _setup(name, months)
    r = solvers.run(method, seed, budget=budget, snapshot_every=None)
    feasible = [f for f in r["archive"] if f[4] >= plan.initTheta * (1 + plan.upsilon)]
    cash = max(f[4] for f in (feasible or r["archive"]))
    best, n = r["best"], len(plan.SYMBOLS)
    y0, y1 = np.zeros((plan.tau, n), dtype=np.int64), np.zeros((plan.tau, n), dtype=np.int64)
    for t, j, v in best["y0"]:
        y0[t, j] = v
    for t, j, v in best["y1"]:
        y1[t, j] = v
    ind = plan.simulate(y0, y1)
    bad, _ = checker.check(ind)
    rec = {"cash": cash, "best_cash": best["f"][4], "checker_ok": not bad,
           "simulator_rel_diff": abs(ind.theta[-1] - best["f"][4]) / best["f"][4], "y0": best["y0"], "y1": best["y1"]}
    return name, months, method, seed, rec


def main(n_seeds=30, budget=3000, workers=9):
    import sys
    from multiprocessing import Pool
    out = {"instances": []}
    for name, months in SMALL:
        syms = _setup(name, months)
        sol = solve()
        ind, bad, rel = validate(sol)
        print(f"{name} months {months}: theta* = {sol['theta_tau']:,.1f} kVND ({sol['status']}); "
              f"checker {'OK' if not bad else bad[:2]}; simulator diff {rel:.1e}; "
              f"{sol['n_vars']} vars, {sol['n_rows']} rows", file=sys.stderr)
        out["instances"].append({"name": name, "months": list(months), "symbols": syms, "theta_star": sol["theta_tau"],
                                 "status": sol["status"], "checker_ok": not bad, "simulator_rel_diff": rel,
                                 "solvers": {}})
    seeds = list(range(1234, 1234 + n_seeds))
    jobs = [(name, months, m, s, budget) for name, months in SMALL
            for m in ("ga", "nsga2", "spea2", "nsga3", "moead") for s in seeds]
    with Pool(workers) as pool:
        for name, months, method, seed, run in pool.imap_unordered(_solver_job, jobs):
            rec = next(i for i in out["instances"] if i["name"] == name and i["months"] == list(months))
            rec["solvers"].setdefault(method, {})[str(seed)] = run
    with open(os.path.join(ROOT, "results", "milp_small.json"), "w") as f:
        json.dump(out, f, indent=1)
    for rec in out["instances"]:
        gaps = {m: 100 * (1 - max(x["cash"] for x in v.values()) / rec["theta_star"]) for m, v in rec["solvers"].items()}
        bad = sum(1 for v in rec["solvers"].values() for x in v.values() if not x["checker_ok"])
        print(rec["name"], rec["months"], {m: f"{g:.2f}%" for m, g in gaps.items()}, f"checker failures: {bad}", file=sys.stderr)


if __name__ == "__main__":
    main()
