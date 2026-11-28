"""Dividend-portfolio model (Section 3) and the shared constructive sampler G (Appendix B).

Every solver builds and evaluates plans through `sample()`. Market data and the
co-moment tensors are set per instance with `load_instance()`.

Indexing: trading months t = 0..tau-1 (0-based); cash theta_t for t = 0..tau.
Holdings q[t] are post-trade holdings at the end of month t.
"""
import json
import math
import random
from dataclasses import dataclass, field

import numpy as np

import dominance

# ---------------------------------------------------------------- parameters (Section 5.2)
tau = 12
epsilon = 0.0015          # proportional transaction fee
initTheta = 1_000_000.0   # initial capital Theta (kVND)
alpha = 0.0045            # monthly deposit rate
xi = 100                  # lot size (shares)
cardinalityK = 20
upsilon = 0.06            # target return
POPULATION_SIZE = 100

# SA injection (Algorithm 2)
SA_STEPS = 10             # L: neighbours per SA phase
SA_START_ACCEPT = 0.5     # calibrated: an average worsening move is accepted with this probability at T0
SA_END_RATIO = 0.01       # temperature at the last step of a phase, relative to T0
SA_PENALTY = 10.0         # weight of the target violation in the SA / GA energy

dominance.configure(theta=initTheta, upsilon=upsilon)

# ---------------------------------------------------------------- instance data (set by load_instance)
futureData = None
SYMBOLS = []
PRICES = VOLUMES = DIVS = LASTPRICE = None   # arrays (n, tau); DIVS in kVND per share
M2 = T3 = T4 = None                          # co-moment tensors of daily simple returns
CRET = None                                  # centred daily simple returns (T, n); exact fast path
EVALS = 0


def comoments(R):
    """Raw second, third and fourth central co-moments of daily returns R (T, n), as used in
    Eqs. (17)-(19): M2 = C'C/T (n, n), M3 = C'X/T (n, n^2) and M4 = X'X/T stored as (n, n^3),
    with C the centred returns and X_t = c_t (x) c_t. For any weights w these equal the sample
    moments of r = C w: mean(r^2), mean(r^3), mean(r^4). Returns (M2, M3, M4, C)."""
    C = R - R.mean(axis=0)
    T, n = C.shape
    X = np.einsum("ti,tj->tij", C, C).reshape(T, n * n)
    return C.T @ C / T, C.T @ X / T, (X.T @ X / T).reshape(n, n ** 3), C


def load_instance(json_path, moments):
    """Load monthly market data (existing JSON schema) and co-moment tensors.

    `moments` is a path to an .npz with arrays M2 (n,n), M3 (n,n*n), M4 (n,n**3) and
    `tickers`, or a dict with the same keys. Ticker order must match the JSON. When the
    centred daily returns C (T,n) are present, objectives() evaluates the moments as sample
    moments of the portfolio return series C w, which is algebraically identical to the
    tensor contractions (see comoments), and the n^3 / n^4 tensors are not loaded."""
    global futureData, SYMBOLS, PRICES, VOLUMES, DIVS, LASTPRICE, M2, T3, T4, CRET
    futureData = json.load(open(json_path)) if isinstance(json_path, str) else json_path
    stocks = futureData["stocks"]
    SYMBOLS = [s["symbol"] for s in stocks]
    n = len(stocks)
    PRICES, VOLUMES, DIVS = np.zeros((n, tau)), np.zeros((n, tau)), np.zeros((n, tau))
    for j, s in enumerate(stocks):
        for p in s["prices"]:
            if p["month"] <= tau:
                PRICES[j, p["month"] - 1] = p["value"]
                VOLUMES[j, p["month"] - 1] = p["matchedTradingVolume"]
        for d in s["dividendSpitingHistories"]:
            if d["month"] <= tau:
                DIVS[j, d["month"] - 1] += d["value"] / 1000.0  # VND -> kVND per share
    LASTPRICE = PRICES.copy()
    for t in range(1, tau):  # carry the last traded price through months without trading
        LASTPRICE[:, t] = np.where(LASTPRICE[:, t] > 0, LASTPRICE[:, t], LASTPRICE[:, t - 1])
    mom = np.load(moments) if isinstance(moments, str) else moments
    if "tickers" in mom and list(mom["tickers"]) != SYMBOLS:
        raise ValueError("moment tensors and market data list different tickers")
    M2 = np.asarray(mom["M2"], dtype=float)
    if "C" in mom:
        CRET = np.ascontiguousarray(mom["C"], dtype=float)
        if CRET.shape[1] != n:
            raise ValueError("return matrix and market data have different numbers of stocks")
        T3 = T4 = None
    else:
        CRET = None
        T3 = np.asarray(mom["M3"], dtype=float).reshape(n, n, n)
        T4 = np.asarray(mom["M4"], dtype=float).reshape(n, n, n, n)


def reset_evals():
    global EVALS
    EVALS = 0


# ---------------------------------------------------------------- plans
@dataclass
class Individual:
    y0: np.ndarray            # buys (tau, n), shares
    y1: np.ndarray            # sells (tau, n), shares
    q: np.ndarray             # end-of-month holdings (tau, n)
    theta: np.ndarray         # cash at the start of months 0..tau
    deposit: np.ndarray       # cash left on deposit at the end of months 0..tau-1
    f: np.ndarray             # objectives (var, third moment, fourth moment, entropy, terminal cash)
    cv: float = 0.0           # target violation
    meta: dict = field(default_factory=dict)

    # names used by the first-round analysis scripts
    @property
    def aggregatedFitness(self):
        return self.f

    @property
    def buyingStocks(self):
        return self.y0

    @property
    def sellingStocks(self):
        return self.y1


def _gene_source(has_parents):
    """Per-month recombination: parent 1 (0.45), parent 2 (0.45) or a random resample (0.10)."""
    if not has_parents:
        return 0
    u = random.random()
    return 1 if u < 0.45 else (2 if u < 0.90 else 0)


def _perturb(value, cap):
    """Pivot-month SA perturbation: +/- k lots, k uniform in {1, 2, 3}, clipped to [0, cap]."""
    k = random.choice((1, 2, 3)) * xi
    v = value + k if random.random() < 0.5 else value - k
    return int(min(max(v, 0), cap))


def sample(parent1=None, parent2=None, anchor=None):
    """Constructive sampler G (Algorithms 3-4). Builds one trading-feasible plan month by month
    and evaluates it. The 6% return target is not enforced here; selection handles it."""
    global EVALS
    EVALS += 1
    n = PRICES.shape[0]
    y0 = np.zeros((tau, n), dtype=np.int64)
    y1 = np.zeros((tau, n), dtype=np.int64)
    q = np.zeros((tau, n), dtype=np.int64)
    theta = np.zeros(tau + 1)
    deposit = np.zeros(tau)
    zero = np.zeros(n, dtype=np.int64)
    has_parents = parent1 is not None and parent2 is not None
    pivot = random.randrange(tau - 2) if anchor is not None else None  # t_dagger in {0..tau-3}

    cash = initTheta
    for t in range(tau):
        if t >= 1:  # Eq. (21): deposit interest + settled sales + lagged dividends
            cash = (1 + alpha) * deposit[t - 1] + (1 - epsilon) * float(PRICES[:, t - 1] @ y1[t - 1])
            if t >= 2:
                cash += float(DIVS[:, t - 1] @ q[t - 2])
        theta[t] = cash
        prev = q[t - 1] if t >= 1 else zero
        prev2 = q[t - 2] if t >= 2 else zero
        held = prev.copy()
        src = {1: parent1, 2: parent2}.get(_gene_source(has_parents))
        use_anchor = anchor is not None and t <= pivot

        # ---- sell phase
        sold = np.zeros(n, dtype=bool)
        if t >= 1:
            for j in np.flatnonzero(prev > 0):
                if t == tau - 1:                        # Eq. (23): forced terminal liquidation
                    amount = int(prev[j])
                else:
                    if DIVS[j, t] > 0 or PRICES[j, t] <= 0 or VOLUMES[j, t] <= 0:
                        continue                        # Eq. (8): no sale in a dividend month
                    cap = int(min(prev[j], prev2[j], VOLUMES[j, t]) // xi * xi)  # Eq. (7) + volume
                    if cap <= 0:
                        continue
                    if use_anchor:
                        base = int(anchor.y1[t, j])
                        amount = min(base, cap) if t < pivot else _perturb(base, cap)
                    elif src is not None and src.y1[t, j] <= cap:
                        amount = int(src.y1[t, j])
                    else:
                        amount = random.randint(0, cap // xi) * xi
                if amount > 0:
                    y1[t, j] = amount
                    held[j] -= amount
                    sold[j] = True

        # ---- buy phase (no purchases in the last two months, Eq. 24)
        if t <= tau - 3:
            if use_anchor:
                candidates = np.flatnonzero(anchor.y0[t] > 0).tolist()
            elif src is not None:
                candidates = np.flatnonzero(src.y0[t] > 0).tolist()
            else:
                candidates = random.sample(range(n), random.randrange(min(int(cardinalityK), n) + 1))
            count = int(np.count_nonzero(held))
            for j in candidates:
                price, vol = PRICES[j, t], VOLUMES[j, t]
                if sold[j] or price <= 0 or vol <= 0 or VOLUMES[j, tau - 1] <= 0:
                    continue                            # Eq. (12) and tradability
                if held[j] == 0 and count >= cardinalityK:
                    continue                            # cardinality, counted on all holdings
                # Algorithm 3, lines 1-3: keep cash back for inventory that could not be liquidated
                residual = max(0.0, float(prev[j]) - float(VOLUMES[j, t:].sum()))
                spend = min(cash / (1 + epsilon), price * vol) - price * residual
                max_amount = int(spend // (price * xi)) * xi if spend > 0 else 0
                if max_amount <= 0:
                    continue
                if use_anchor:
                    base = int(anchor.y0[t, j])
                    amount = min(base, max_amount) if t < pivot else _perturb(base, max_amount)
                elif src is not None and src.y0[t, j] <= max_amount:
                    amount = int(src.y0[t, j])
                else:
                    amount = random.randint(0, max_amount // xi) * xi
                if amount <= 0:
                    continue
                y0[t, j] = amount
                cash -= amount * price * (1 + epsilon)
                if held[j] == 0:
                    count += 1
                held[j] += amount
        q[t] = held
        deposit[t] = cash

    theta[tau] = ((1 + alpha) * deposit[tau - 1] + (1 - epsilon) * float(PRICES[:, tau - 1] @ y1[tau - 1])
                  + float(DIVS[:, tau - 1] @ q[tau - 2]))
    f = objectives(q, y1, deposit, theta[tau])
    return Individual(y0, y1, q, theta, deposit, f, dominance.violation(f))


def objectives(q, y1, deposit, terminal_cash):
    """Eqs. (1)-(2) and (16)-(20) with capital weights that include cash.

    The risk of month t (t = 1..tau-1) is measured on the end-of-month-(t-1) portfolio:
    w_j = C_j q_j / W with W = deposit + receivables + sum_j C_j q_j, where receivables are
    sales and dividends that settle next month. Cash carries no risk, so the moments shrink
    with the cash share. Entropy is that of the risky book. A month with nothing invested
    contributes zero to every risk term."""
    M = tau - 1
    value = LASTPRICE[:, :M].T * q[:M]                                    # (M, n) C_j q_j
    recv = (1 - epsilon) * (PRICES[:, :M].T * y1[:M]).sum(axis=1)
    recv[1:] += (DIVS[:, 1:M].T * q[:M - 1]).sum(axis=1)
    invested = value.sum(axis=1)
    held = invested > 0
    w = np.zeros_like(value)
    w[held] = value[held] / (deposit[:M][held] + recv[held] + invested[held])[:, None]
    risks = np.zeros((M, 4))
    if CRET is not None:
        r = CRET @ w.T                                                     # (T, M) daily portfolio returns
        r2 = r * r
        risks[:, 0] = r2.mean(axis=0)
        risks[:, 1] = (r2 * r).mean(axis=0)
        risks[:, 2] = (r2 * r2).mean(axis=0)
    else:
        for m in np.flatnonzero(held):
            S = np.flatnonzero(w[m] > 0)
            ws = w[m, S]
            risks[m, 0] = ws @ M2[np.ix_(S, S)] @ ws
            risks[m, 1] = np.einsum("i,j,k,ijk->", ws, ws, ws, T3[np.ix_(S, S, S)])
            risks[m, 2] = np.einsum("i,j,k,l,ijkl->", ws, ws, ws, ws, T4[np.ix_(S, S, S, S)])
    p = np.zeros_like(value)
    p[held] = value[held] / invested[held][:, None]
    plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    risks[:, 3] = -plogp.sum(axis=1)
    out = np.empty(5)
    out[:4] = risks.mean(axis=0)
    out[4] = terminal_cash
    return out


def simulate(y0, y1):
    """Holdings, cash path and objectives implied by given trades (no repair, not counted as an
    evaluation). Used to validate external plans such as the MILP solution."""
    n = PRICES.shape[0]
    y0 = np.asarray(y0, dtype=np.int64)
    y1 = np.asarray(y1, dtype=np.int64)
    q = np.zeros((tau, n), dtype=np.int64)
    theta = np.zeros(tau + 1)
    deposit = np.zeros(tau)
    cash = initTheta
    for t in range(tau):
        if t >= 1:
            cash = (1 + alpha) * deposit[t - 1] + (1 - epsilon) * float(PRICES[:, t - 1] @ y1[t - 1])
            if t >= 2:
                cash += float(DIVS[:, t - 1] @ q[t - 2])
        theta[t] = cash
        q[t] = (q[t - 1] if t >= 1 else 0) + y0[t] - y1[t]
        cash -= (1 + epsilon) * float(PRICES[:, t] @ y0[t])
        deposit[t] = cash
    theta[tau] = ((1 + alpha) * deposit[tau - 1] + (1 - epsilon) * float(PRICES[:, tau - 1] @ y1[tau - 1])
                  + float(DIVS[:, tau - 1] @ q[tau - 2]))
    f = objectives(q, y1, deposit, theta[tau])
    return Individual(y0, y1, q, theta, deposit, f, dominance.violation(f))


def deposit_only_cash():
    return initTheta * (1 + alpha) ** tau


# ---------------------------------------------------------------- scalar energy (GA fitness, SA energy)
def energy(f, lo, hi):
    """Equal-weight Tchebycheff value of the all-minimization objectives, normalized to the
    population range [lo, hi], plus a penalty on the target violation. Lower is better.
    Unlike the first-round multiplicative score, it is never zero just because one objective is."""
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    z = (dominance.to_min(f) - lo) / span
    return float(np.max(z)) + SA_PENALTY * dominance.violation(f)


def population_bounds(individuals):
    M = np.array([dominance.to_min(ind.f) for ind in individuals])
    return M.min(axis=0), M.max(axis=0)


def elite(individuals, lo, hi):
    """Lowest-energy plan of a population under the bounds lo, hi (first one on ties).
    Anchor of the elite-anchored SA variant (eSA / eRI)."""
    energies = [energy(ind.f, lo, hi) for ind in individuals]
    return individuals[int(np.argmin(energies))]


class SAInjector:
    """SA phase of Algorithm 2. Each generation it draws `SA_STEPS` pivot-month neighbours of a
    generation-fixed anchor, accepts by constraint-domination, improvement or the Metropolis test,
    restarts the temperature at T0 and cools it geometrically to SA_END_RATIO * T0.
    With accept_all=True it is the random-injection control: same neighbours, every move accepted.
    start_accept is the probability chi_0 with which an average worsening move is accepted at T0.

    The log holds one row per generation, [n_up, n_up_acc, n_met, n_met_acc]: n_up counts the
    candidates that raise the energy (delta > 0) and n_up_acc those of them accepted by any route
    (constraint-domination, the Metropolis test, or every move in the control); n_met counts the
    energy-raising candidates decided by the Metropolis test and n_met_acc those that pass it.
    The counters only observe the decisions; they draw no random numbers."""

    def __init__(self, anchor, accept_all=False, start_accept=SA_START_ACCEPT):
        self.anchor = anchor
        self.accept_all = accept_all
        self.start_accept = start_accept
        self.T0 = None
        self.log = []   # per generation: [n_up, n_up_acc, n_met, n_met_acc] (see the class docstring)

    def calibrate(self, lo, hi, probes=20):
        e0 = energy(self.anchor.f, lo, hi)
        deltas = [energy(sample(anchor=self.anchor).f, lo, hi) - e0 for _ in range(probes)]
        worse = [d for d in deltas if d > 0]
        mean_worse = float(np.mean(worse)) if worse else 0.05
        self.T0 = -mean_worse / math.log(self.start_accept)

    def phase(self, lo, hi):
        if self.T0 is None:
            self.calibrate(lo, hi)
        ratio = SA_END_RATIO ** (1.0 / max(SA_STEPS - 1, 1))
        best = self.anchor
        e_best = energy(best.f, lo, hi)
        n_up = n_up_acc = n_met = n_met_acc = 0
        T = self.T0
        for _ in range(SA_STEPS):
            cand = sample(anchor=self.anchor)
            e = energy(cand.f, lo, hi)
            delta = e - e_best
            tested = False
            if self.accept_all:
                accept = True
            elif dominance.dominates(cand.f, best.f) or delta < 0:
                accept = True
            else:
                tested = True
                accept = random.random() < math.exp(-delta / T)
            if delta > 0:
                n_up += 1
                n_up_acc += int(accept)
                if tested:
                    n_met += 1
                    n_met_acc += int(accept)
            if accept:
                best, e_best = cand, e
            T *= ratio
        self.anchor = best
        self.log.append([n_up, n_up_acc, n_met, n_met_acc])
        return best
