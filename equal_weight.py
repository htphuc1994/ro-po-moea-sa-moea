"""Equal-weight buy-and-hold baseline.

Implements a passive baseline: at t=0, distribute the initial budget equally
across the K most-liquid stocks (by month-1 matchedTradingVolume), buy round
lots, hold through tau-1, then liquidate at t=tau-1.  Constraints honored:
budget, round-lot xi, volume cap Q_{j,t}, cardinality K, and the no-trade
behavior between t=1 and t=tau-2 (pure buy-and-hold). Dividends are accrued
through the cash recursion of plan.py.

Output: the same chromosome shape (q, y0, y1) used by HSG/NSGA-II/GA, plus
the 5-objective vector and runtime, suitable for the existing
resultAnalyzers pipeline.
"""

import json
import math
import time
import numpy as np
import plan


def _load_dataset(name):
    """Load M2/M3/M4 into plan namespace and return parsed futureData."""
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


def equal_weight_plan(futureData, universe="topK"):
    """Build a buy-and-hold plan that equal-weights either:
      - 'topK'   : the K most-liquid stocks by month-1 trading volume
                   (respects the cardinality cap of the optimization model);
      - 'all'    : every stock in the universe (textbook 1/n, cardinality
                   relaxed)."""
    n = len(futureData["stocks"])
    K = int(plan.cardinalityK)
    tau = plan.tau
    xi = plan.xi
    epsilon = plan.epsilon
    alpha = plan.alpha
    Theta = plan.initTheta
    beta_div = 1.0 / 1000.0  # dividend value scaling: VND -> kVND (matches plan.py)

    if universe == "all":
        chosen = list(range(n))
    elif universe == "topK":
        vols = []
        for j in range(n):
            v0 = next((p["matchedTradingVolume"] for p in futureData["stocks"][j]["prices"]
                       if p["month"] == 1), 0.0)
            vols.append((v0, j))
        vols.sort(reverse=True)
        chosen = sorted(idx for _, idx in vols[:K])
    else:
        raise ValueError(universe)

    rows, cols = (tau + 1, n)
    holdingStocks = [[0 for _ in range(cols)] for _ in range(rows)]
    buyingStocks = [[0 for _ in range(cols)] for _ in range(rows)]
    sellingStocks = [[0 for _ in range(cols)] for _ in range(rows)]
    cashBeginMonth = []
    cashEndMonth = []

    available = Theta
    deposit = 0.0
    per_stock = Theta / len(chosen)  # budget per chosen stock at t=0

    for t in range(tau + 1):
        # begin-of-month: interest + sales proceeds + lagged dividends
        if t > 0:
            available += deposit * (1 + alpha)
            for j in range(n):
                if sellingStocks[t - 1][j] > 0:
                    px = next(p["value"] for p in futureData["stocks"][j]["prices"]
                              if p["month"] == t)
                    available += sellingStocks[t - 1][j] * px * (1 - epsilon)
        if t > 1:
            for j in range(n):
                for d in futureData["stocks"][j]["dividendSpitingHistories"]:
                    if d["month"] == t:
                        available += holdingStocks[t - 2][j] * d["value"] * beta_div
        cashBeginMonth.append(available)

        if t == 0:
            # Buy phase: equal-weight allocation, lot rounding, volume cap
            for j in chosen:
                p1 = next(p for p in futureData["stocks"][j]["prices"] if p["month"] == 1)
                price = p1["value"]
                vol_cap = p1["matchedTradingVolume"]
                # Largest qty fitting per_stock budget AND vol cap, in round lots
                max_qty_budget = math.floor((per_stock / (price * (1 + epsilon))) / xi) * xi
                max_qty_vol = math.floor(vol_cap / xi) * xi
                qty = min(max_qty_budget, max_qty_vol)
                if qty > 0 and qty * price * (1 + epsilon) <= available:
                    buyingStocks[0][j] = qty
                    available -= qty * price * (1 + epsilon)
            for j in range(n):
                holdingStocks[0][j] = buyingStocks[0][j]
        elif t == tau - 1:
            # Liquidate everything held
            for j in range(n):
                qty = holdingStocks[t - 1][j]
                if qty > 0:
                    p_t = next((p for p in futureData["stocks"][j]["prices"]
                                if p["month"] == t + 1), None)
                    if p_t is not None:
                        sell = min(qty, math.floor(p_t["matchedTradingVolume"] / xi) * xi)
                        sellingStocks[t][j] = sell
                        holdingStocks[t][j] = qty - sell
                    else:
                        holdingStocks[t][j] = qty  # no price -> can't sell
        elif t == tau:
            # nothing more; carry holdings (should be zero after liquidation)
            for j in range(n):
                holdingStocks[t][j] = holdingStocks[t - 1][j] - sellingStocks[t - 1][j]
        else:
            # 1 <= t <= tau-2: hold
            for j in range(n):
                holdingStocks[t][j] = holdingStocks[t - 1][j]

        if t < tau:
            cashEndMonth.append(available)
            deposit = available
            available = 0.0

    finalCash = cashBeginMonth[-1] if cashBeginMonth else 0.0

    # Evaluate the 5-objective vector
    fitnesses = [[0 for _ in range(4)] for _ in range(tau)]
    for t in range(tau):
        w = plan.genWeightVector(holdingStocks[t])
        fitnesses[t][0] = plan.calVariance(plan.M2, w)
        fitnesses[t][1] = plan.calSkewness(plan.M3, w)
        fitnesses[t][2] = plan.calKurtosis(plan.M4, w)
        fitnesses[t][3] = plan.calEntropy(w)
    aggMean = np.mean(fitnesses, axis=0)
    aggregatedFitness = [aggMean[0], aggMean[1], aggMean[2], aggMean[3], finalCash]

    return {
        "holdingStocks": holdingStocks,
        "buyingStocks": buyingStocks,
        "sellingStocks": sellingStocks,
        "fitnesses": fitnesses,
        "aggregatedFitness": aggregatedFitness,
        "finalCash": finalCash,
        "chosen": chosen,
    }


def run(name, universe="topK"):
    futureData = _load_dataset(name)
    t0 = time.time()
    result = equal_weight_plan(futureData, universe=universe)
    dt = time.time() - t0
    # Cardinality cap is bypassed for the 'all' variant; verifyInvestmentPlan
    # would flag this. Validate only when respecting K.
    if universe == "topK":
        isValid, msg = plan.verifyInvestmentPlan(
            result["holdingStocks"], result["buyingStocks"], result["sellingStocks"], futureData
        )
        print(f"[{name}/{universe}] feasibility = {isValid} ({msg})")
    else:
        print(f"[{name}/{universe}] feasibility check skipped (cardinality intentionally relaxed)")
    print(f"[{name}/{universe}] aggregatedFitness = {result['aggregatedFitness']}")
    print(f"[{name}/{universe}] finalCash = {result['finalCash']:.2f} kVND  (runtime {dt:.4f}s)")
    return result, dt


if __name__ == "__main__":
    summary = []
    for ds in ("vn30", "hnx30"):
        for universe in ("topK", "all"):
            result, dt = run(ds, universe=universe)
            summary.append({
                "dataset": ds,
                "universe": universe,
                "finalCash": result["finalCash"],
                "aggregatedFitness": [float(x) for x in result["aggregatedFitness"]],
                "runtime": dt,
            })
    print("\n=== summary ===")
    for s in summary:
        print(s)
    # save log
    with open("equal_weight_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote equal_weight_results.json")
