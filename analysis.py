"""Regenerate the results tables and figures of Section 5 from the run files.

Usage:
    python3 analysis.py [--results results] [--timing results_timing] [--paper ../manuscript]

Writes LaTeX table bodies to <paper>/tables/, figures to <paper>/figures/ and every number to
results_summary.json (used to regenerate the README). Indicators are normalized per instance
with the archives of the main configurations only, so ablation runs do not move the main numbers.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

import baselines
import metrics
import plan
import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = {}
MAIN = ["ga", "nsga2", "nsga2+sa", "nsga2+ri", "spea2", "spea2+sa", "spea2+ri",
        "nsga3", "nsga3+sa", "nsga3+ri", "moead", "moead+sa", "moead+ri"]
BASES = ["nsga2", "spea2", "nsga3", "moead"]
NAME = {"ga": "GA", "nsga2": "NSGA-II", "spea2": "SPEA2", "nsga3": "NSGA-III", "moead": "MOEA/D"}
INST = {"vn30_2023": "VN30 2023", "hnx30_2023": "HNX30 2023", "combined_2023": "VN30+HNX30 2023",
        "vn30_2024": "VN30 2024", "hnx30_2024": "HNX30 2024", "vn30_2025": "VN30 2025", "hnx30_2025": "HNX30 2025"}


def pretty(label):
    label, _, chi = label.partition("-chi")
    return _pretty(label) + (f" ($\\chi_0={chi}$)" if chi else "")


def _pretty(label):
    base, _, rest = label.partition("+")
    mode, _, rule = rest.partition("@") if rest else ("", "", "")
    if "@" in base:
        base, rule = base.split("@")
    name = NAME.get(base, base)
    return {"sa": f"SA-{name}", "ri": f"RI-{name}", "sae": f"eSA-{name}", "rie": f"eRI-{name}"}.get(mode, name) + \
        (" (lexicographic)" if rule else "")


def load(results_dir):
    runs = defaultdict(lambda: defaultdict(dict))
    for path in glob.glob(os.path.join(results_dir, "*", "*", "seed_*.json")):
        r = json.load(open(path))
        runs[r["instance"]][r["label"]][r["seed"]] = r
    return runs


def indicators(runs_inst):
    main_labels = [l for l in MAIN if l in runs_inst]
    norm = metrics.Normalizer([r["archive"] for l in main_labels for r in runs_inst[l].values()])
    front = (norm.front - norm.ideal) / norm.span
    hv = metrics.Hypervolume()
    table = {}
    for label, by_seed in runs_inst.items():
        rows = []
        for seed in sorted(by_seed):
            r = by_seed[seed]
            Z = norm(r["archive"])
            rows.append({"seed": seed, "hv": hv(Z), "igd": metrics.igd_plus(Z, front), "sp": metrics.spacing(Z),
                         "cash": r["best"]["f"][4] / 1000.0, "size": len(r["archive"]), "evals": r["evals"],
                         "gens": r["generations"], "runtime": r["runtime"], "archive_Z": Z,
                         "sa_log": r.get("sa", {}).get("log"), "trunc": r.get("spea2_truncations"),
                         "diag": r.get("sa", {}).get("diag"), "in_final": r.get("sa", {}).get("in_final")})
        table[label] = rows
    return table, norm


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def fmt_ms(x, dec=1, bold=False):
    """Mean (sd); with bold, the mean only is set in bold."""
    return f"{bf(f'{np.nanmean(x):,.{dec}f}', bold)} ({np.nanstd(x, ddof=1):,.{dec}f})"


def bf(s, on=True):
    return f"\\textbf{{{s}}}" if on else s


def best_set(values, dec, higher):
    """Indices of the values whose printed form (dec decimals) equals the best printed value, so that
    ties at the printed precision are all marked. None and NaN entries are skipped."""
    ok = [(i, float(v)) for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    if not ok:
        return set()
    b = (max if higher else min)(v for _, v in ok)
    return {i for i, v in ok if f"{v:.{dec}f}" == f"{b:.{dec}f}"}


def pairwise(table, labels, key):
    """Friedman over labels (blocks = common seeds) and Holm-adjusted Wilcoxon for all pairs."""
    seeds = sorted(set.intersection(*[{r["seed"] for r in table[l]} for l in labels]))
    data = {l: np.array([next(r[key] for r in table[l] if r["seed"] == s) for s in seeds]) for l in labels}
    chi2, p = stats.friedman(np.column_stack([data[l] for l in labels])) if len(seeds) > 1 else (float("nan"), float("nan"))
    pairs = list(combinations(labels, 2))
    raw = [stats.wilcoxon(data[a], data[b]) for a, b in pairs]
    adj, rej = stats.holm([p_ for _, p_ in raw])
    out = {}
    for k, (a, b) in enumerate(pairs):
        m, lo, hi = stats.bootstrap_ci(data[a] - data[b])
        out[(a, b)] = {"W": raw[k][0], "p": raw[k][1], "p_holm": float(adj[k]), "reject": bool(rej[k]),
                       "diff": m, "lo": lo, "hi": hi}
    return {"chi2": chi2, "p": p, "n": len(seeds), "pairs": out}


def _cov(Za, Zb):
    """Coverage on normalized, minimization-oriented coordinates: share of Zb dominated by Za."""
    if not len(Zb):
        return float("nan")
    le = np.all(Za[:, None, :] <= Zb[None, :, :], axis=2)
    lt = np.any(Za[:, None, :] < Zb[None, :, :], axis=2)
    return float((le & lt).any(axis=0).mean())


def pairwise_cs(table, labels):
    """Per-seed directed coverage difference CS(A,B) - CS(B,A) for every pair; Wilcoxon signed-rank
    on the paired values, Holm over all pairs, bootstrap CI of the mean difference."""
    seeds = sorted(set.intersection(*[{r["seed"] for r in table[l]} for l in labels]))
    Z = {l: {r["seed"]: r["archive_Z"] for r in table[l]} for l in labels}
    pairs = list(combinations(labels, 2))
    ab = {(a, b): np.array([_cov(Z[a][s], Z[b][s]) for s in seeds]) for a, b in pairs}
    ba = {(a, b): np.array([_cov(Z[b][s], Z[a][s]) for s in seeds]) for a, b in pairs}
    raw = [stats.wilcoxon(ab[k], ba[k]) for k in pairs]
    adj, rej = stats.holm([p_ for _, p_ in raw])
    out = {}
    for k, pair in enumerate(pairs):
        m, lo, hi = stats.bootstrap_ci(ab[pair] - ba[pair])
        out[pair] = {"W": raw[k][0], "p": raw[k][1], "p_holm": float(adj[k]), "reject": bool(rej[k]),
                     "diff": m, "lo": lo, "hi": hi}
    return {"n": len(seeds), "pairs": out}


def contrast(res, a, b):
    if (a, b) in res["pairs"]:
        return res["pairs"][(a, b)]
    c = dict(res["pairs"][(b, a)])
    c["diff"], c["lo"], c["hi"] = -c["diff"], -c["hi"], -c["lo"]
    return c


def tex_p(p):
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def summary_table(inst, table, timing, refs, with_ct):
    """One instance block of the results tables: mean (sd) of terminal cash, HV, IGD+ and SP, median
    final-set size and, if with_ct, CPU time of the timing pass. In each column the best mean is in
    bold (highest cash and HV, lowest IGD+, SP and CT; ties at the printed precision included). The
    evaluations of a configuration are the same in every run; the header comment lists them for the
    caption. The three deterministic references share one line under the block."""
    lines, evals = [], []
    labels = [l for l in MAIN if l in table]
    cols = [("cash", 0, True), ("hv", 3, True), ("igd", 3, False), ("sp", 3, False)]
    best = {k: best_set([np.nanmean(col(table[l], k)) for l in labels], d, hi) for k, d, hi in cols}
    rts = [timing.get(inst, {}).get(l) for l in labels]
    best_ct = best_set([np.nanmean(rt) if rt is not None else None for rt in rts], 2, False)
    for i, l in enumerate(labels):
        rows = table[l]
        cells = [pretty(l)] + [fmt_ms(col(rows, k), d, bold=i in best[k]) for k, d, _ in cols]
        cells.append(f"{np.median(col(rows, 'size')):g}")
        if with_ct:
            cells.append(fmt_ms(rts[i], 2, bold=i in best_ct) if rts[i] is not None else "---")
        lines.append(" & ".join(cells) + " \\\\")
        ev = sorted(set(int(e) for e in col(rows, "evals")))
        evals.append(f"{pretty(l)} {ev[0]:,}" + (f"--{ev[-1]:,}" if len(ev) > 1 else ""))
        if len(ev) > 1:
            print(f"warning: {inst} {l}: evaluations differ between runs", file=sys.stderr)
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{7 if with_ct else 6}}}{{@{{}}l}}{{References: deposit only {refs['deposit']:,.0f}; "
                 f"equal weight {refs['eqw_topk']:,.0f} (top-$K$), {refs['eqw_all']:,.0f} (all)}} \\\\")
    return "% evaluations per run: " + "; ".join(evals) + "\n" + "\n".join(lines) + "\n"


def tests_table(table):
    """Selected Holm-corrected contrasts on terminal cash, HV, SP and directed coverage
    (each indicator corrected over all 78 pairs of the main configurations)."""
    labels = [l for l in MAIN if l in table]
    res = {k: pairwise(table, labels, k) for k in ("cash", "hv", "sp")}
    res["cs"] = pairwise_cs(table, labels)
    # rows of the three injection groups are labelled by the base X; the group heading states the contrast
    groups = [("SA injection vs.\\ vanilla: SA-$X$ $-$ $X$", [(f"{b}+sa", b) for b in BASES], True),
              ("Random injection vs.\\ vanilla: RI-$X$ $-$ $X$", [(f"{b}+ri", b) for b in BASES], True),
              ("SA vs.\\ random injection: SA-$X$ $-$ RI-$X$", [(f"{b}+sa", f"{b}+ri") for b in BASES], True),
              ("Between bases: $A-B$", list(combinations(["ga"] + BASES, 2)), False)]
    # signed point estimate, unsigned CI bounds, p-values below 0.001 as <0.001 (fits a landscape page at 8 pt)
    fmt = {"cash": ",.0f", "hv": ".3f", "sp": ".3f", "cs": ".2f"}
    lines = []
    for title, pairs, by_base in groups:
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for a, b in pairs:
            if a not in table or b not in table:
                continue
            cells = [f"\\quad {pretty(b.split('+')[0]) if by_base else pretty(a) + ' -- ' + pretty(b)}"]
            for key in ("cash", "hv", "sp", "cs"):
                c = contrast(res[key], a, b)
                f = fmt[key]
                p = "$<$0.001" if c["p_holm"] < 0.001 else f"{c['p_holm']:.3f}"
                cells += [f"{c['diff']:+{f}} [{c['lo']:{f}}, {c['hi']:{f}}]", p]
            lines.append(" & ".join(cells) + " \\\\")
    head = (f"% Friedman cash: chi2={res['cash']['chi2']:.1f}, p={res['cash']['p']:.2g}; "
            f"HV: chi2={res['hv']['chi2']:.1f}, p={res['hv']['p']:.2g}; "
            f"SP: chi2={res['sp']['chi2']:.1f}, p={res['sp']['p']:.2g}; n={res['cash']['n']} paired runs\n")
    return head + "\n".join(lines) + "\n", res


def coverage_table(table):
    labels = [l for l in ["ga"] + BASES if l in table]
    lines = []
    for a in labels:
        cells = []
        for b in labels:
            if a == b:
                cells.append("---")
                continue
            ra = {r["seed"]: r["archive_Z"] for r in table[a]}
            rb = {r["seed"]: r["archive_Z"] for r in table[b]}
            seeds = sorted(set(ra) & set(rb))
            cs = np.mean([metrics.coverage(ra[s] * metrics.dominance.SIGNS, rb[s] * metrics.dominance.SIGNS) for s in seeds])
            cells.append(f"{cs:.2f}")
        lines.append(f"{pretty(a)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def lex_table(tables):
    lines = []
    for inst, table in tables.items():
        for b in ("nsga2", "spea2", "nsga3"):
            lx = f"{b}@lex"
            if lx not in table or b not in table:
                continue
            row = [INST.get(inst, inst), NAME[b]]
            cash = [np.mean(col(table[l], "cash")) for l in (b, lx)]
            hv = [np.mean(col(table[l], "hv")) for l in (b, lx)]
            bc, bh = best_set(cash, 0, True), best_set(hv, 3, True)   # the better rule in this row, in bold
            for j, l in enumerate((b, lx)):
                row += [bf(f"{cash[j]:,.0f}", j in bc), bf(f"{hv[j]:.3f}", j in bh),
                        f"{np.median(col(table[l], 'size')):g}"]
            lines.append(" & ".join(row) + " \\\\")
    return "\n".join(lines) + "\n"


def ranks_table(tables):
    insts = [i for i in INST if i in tables]
    labels = [l for l in MAIN if all(l in tables[i] for i in insts)]
    out = {}
    lines = []
    for key, higher in (("cash", True), ("hv", True)):
        M = np.array([[np.mean(col(tables[i][l], key)) for l in labels] for i in insts])
        out[key] = {"ranks": stats.average_ranks(M, higher).tolist(),
                    "friedman": stats.friedman(M) if len(insts) > 2 else (float("nan"), float("nan"))}
    top = {key: best_set(out[key]["ranks"], 2, False) for key in ("cash", "hv")}   # best rank of each column, in bold
    for k, l in enumerate(labels):
        rc, rh = f"{out['cash']['ranks'][k]:.2f}", f"{out['hv']['ranks'][k]:.2f}"
        lines.append(f"{pretty(l)} & {bf(rc, k in top['cash'])} & {bf(rh, k in top['hv'])} \\\\")
    head = f"% instances: {', '.join(insts)}; Friedman across instances cash {out['cash']['friedman']}, HV {out['hv']['friedman']}\n"
    return head + "\n".join(lines) + "\n", {"labels": labels, "instances": insts, **out}


def figures(tables, base_cash, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"": "#2a78d6", "sa": "#eb6834", "ri": "#1baf7a"}
    insts = [i for i in ("vn30_2023", "hnx30_2023", "combined_2023") if i in tables]
    os.makedirs(outdir, exist_ok=True)
    # boxes grouped by base solver: GA alone, then vanilla / SA / random injection for each MOEA
    groups = [("GA", [("ga", "#b9b8b2")])] + [(NAME[b], [(b, colors[""]), (f"{b}+sa", colors["sa"]), (f"{b}+ri", colors["ri"])])
                                              for b in BASES]
    pos, centers, x = [], [], 1.0
    for _, members in groups:
        xs = [x + 0.8 * k for k in range(len(members))]
        pos.append(xs)
        centers.append(float(np.mean(xs)))
        x = xs[-1] + 1.4
    for key, ylabel, fname in (("hv", "Hypervolume (normalized)", "fig_hv_box.pdf"), ("cash", "Terminal cash (MVND)", "fig_cash_box.pdf")):
        fig, axes = plt.subplots(1, len(insts), figsize=(7.2, 2.6), squeeze=False)
        for ax, inst in zip(axes[0], insts):
            for (gname, members), xs in zip(groups, pos):
                for (l, c), xx in zip(members, xs):
                    if l not in tables[inst]:
                        continue
                    bp = ax.boxplot([col(tables[inst][l], key)], positions=[xx], widths=0.65, patch_artist=True,
                                    medianprops={"color": "black", "linewidth": 1}, flierprops={"markersize": 2},
                                    whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8})
                    bp["boxes"][0].set_facecolor(c)
                    bp["boxes"][0].set_linewidth(0.6)
            ax.set_xticks(centers, [g for g, _ in groups], rotation=35, ha="right")
            ax.set_xlim(0.2, pos[-1][-1] + 0.8)
            ax.set_title(INST[inst])
            ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
            ax.set_axisbelow(True)
            if ax is axes[0][0]:
                ax.set_ylabel(ylabel)
            if key == "cash":
                for name, val, ls in (("deposit only", base_cash[inst]["deposit"], ":"), ("equal weight (top-K)", base_cash[inst]["eqw_topk"], "--")):
                    ax.axhline(val, color="#52514e", linestyle=ls, linewidth=0.8, label=name)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.6) for c in ("#b9b8b2", colors[""], colors["sa"], colors["ri"])]
        names = ["GA", "vanilla", "SA injection", "random injection"]
        if key == "cash":
            line_h, line_l = axes[0][0].get_legend_handles_labels()
            handles, names = handles + line_h, names + line_l
        fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(os.path.join(outdir, fname))
        plt.close(fig)
    if insts:
        sa_accept_figure(tables[insts[0]], INST[insts[0]], os.path.join(outdir, "fig_sa_accept.pdf"))


def sa_accept_figure(rows, title, path):
    """Share of worsening SA moves accepted per generation, one panel per MOEA: the mean over runs per
    generation, and as a dashed line the run-averaged share (as in results_summary 'sa_acceptance').
    The random-injection controls accept every move, which the caption states; they are not drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    series = {}
    for b in BASES:
        logs = [r["sa_log"] for r in rows.get(f"{b}+sa", []) if r["sa_log"]]
        if not logs:
            continue
        g = min(len(x) for x in logs)
        with np.errstate(invalid="ignore"):
            rate = np.nanmean([[a / w if w else np.nan for w, a in x[:g]] for x in logs], axis=0)
        mean = float(np.mean([sum(a for _, a in x) / max(1, sum(w for w, _ in x)) for x in logs]))
        series[b] = (rate, mean)
    top = max(0.4, np.ceil(10 * max(np.nanmax(r) for r, _ in series.values())) / 10)
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 2.5), sharey=True)
    for ax, b in zip(axes.flat, BASES):
        if b not in series:
            ax.set_visible(False)
            continue
        rate, mean = series[b]
        ax.plot(np.arange(1, len(rate) + 1), rate, color="#eb6834", linewidth=1)
        ax.axhline(mean, color="#52514e", linestyle="--", linewidth=0.8)
        ax.set_title(f"SA-{NAME[b]} (mean {mean:.2f})", fontsize=8)
        ax.set_ylim(0, top)
        ax.set_xlim(0, len(rate) + 1)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("generation")
    fig.supylabel("share of worse moves accepted", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def milp_table(path):
    """Exact optimum on the reduced instances against each solver's best terminal cash."""
    if not os.path.exists(path):
        return None, None
    data = json.load(open(path))
    lines, out = [], []
    for rec in data["instances"]:
        star = rec["theta_star"]
        row = [INST.get(rec["name"], rec["name"]), f"{rec['months'][0]}--{rec['months'][1]}", f"{star / 1000:,.1f}"]
        gaps = {}
        solvers = ("ga", "nsga2", "spea2", "nsga3", "moead")
        for m in solvers:
            vals = np.array(list(rec["solvers"].get(m, {}).values()), dtype=float)
            if vals.size:
                gaps[m] = {"best": 100 * (1 - vals.max() / star), "mean": 100 * (1 - vals.mean() / star)}
        # smallest best-of-runs gap and smallest mean gap of the row, each in bold
        bb = best_set([gaps[m]["best"] if m in gaps else None for m in solvers], 2, False)
        bm = best_set([gaps[m]["mean"] if m in gaps else None for m in solvers], 2, False)
        for i, m in enumerate(solvers):
            if m not in gaps:
                row.append("---")
                continue
            best_s, mean_s = f"{gaps[m]['best']:.2f}", f"{gaps[m]['mean']:.2f}"
            row.append(f"{bf(best_s, i in bb)} / {bf(mean_s, i in bm)}")
        lines.append(" & ".join(row) + " \\\\")
        out.append({"instance": rec["name"], "months": rec["months"], "theta_star": star, "gaps": gaps})
    return "\n".join(lines) + "\n", out


def _settings(label):
    """Dividends and cardinality cap used by a run label (see run_all.label)."""
    k = 20
    for part in label.split("-")[1:]:
        if part.startswith("K") and part[1:].isdigit():
            k = int(part[1:])
    return "-nodiv" in label, k


def _plan_arrays(best):
    n = len(plan.SYMBOLS)
    y0 = np.zeros((plan.tau, n), dtype=np.int64)
    y1 = np.zeros((plan.tau, n), dtype=np.int64)
    for t, j, v in best["y0"]:
        y0[t, j] = v
    for t, j, v in best["y1"]:
        y1[t, j] = v
    return y0, y1


def dividend_income(ind):
    """Cash dividends credited to a plan (Eq. 21, t >= 2), in kVND."""
    return float(sum(plan.DIVS[:, t - 1] @ ind.q[t - 2] for t in range(2, plan.tau + 1)))


def check_reported(runs_inst):
    """Re-simulate every reported plan under its own settings and run the independent checker."""
    import checker
    divs, out = plan.DIVS.copy(), {"plans": 0, "violating": 0, "max_liquidation_ratio": 0.0,
                                   "max_cash_rel_diff": 0.0, "examples": [], "dividend_income": {}}
    try:
        for label, by_seed in runs_inst.items():
            nodiv, k = _settings(label)
            plan.DIVS[:] = 0.0 if nodiv else divs
            plan.cardinalityK = k
            income = []
            for r in by_seed.values():
                ind = plan.simulate(*_plan_arrays(r["best"]))
                bad, ratio = checker.check(ind)
                out["plans"] += 1
                if bad:
                    out["violating"] += 1
                    if len(out["examples"]) < 5:
                        out["examples"].append([label, r["seed"], bad[:2]])
                out["max_liquidation_ratio"] = max(out["max_liquidation_ratio"], ratio)
                out["max_cash_rel_diff"] = max(out["max_cash_rel_diff"],
                                               abs(ind.theta[-1] - r["best"]["f"][4]) / r["best"]["f"][4])
                income.append(dividend_income(ind) / ind.theta[-1])
            out["dividend_income"][label] = [float(np.mean(income)), float(np.max(income))]
    finally:
        plan.DIVS[:] = divs
        plan.cardinalityK = 20
    return out


def extra_table(tables, solver):
    """No-dividend and cardinality runs of the recommended solver on the 2023 index instances.

    The no-dividend plans are also re-evaluated with the actual dividends; sales they make in a
    dividend month (allowed when dividends are ignored, forbidden by Eq. 8) are counted."""
    import checker
    lines = []
    for inst, table in tables.items():
        if inst.startswith("combined") or f"{solver}-nodiv" not in table:
            continue
        plan.load_instance(os.path.join(ROOT, "data", "instances", f"{inst}.json"),
                           os.path.join(ROOT, "data", "instances", f"{inst}_moments.npz"))
        base = table[solver]
        reeval, eq8 = [], 0
        for r in RUNS[inst][f"{solver}-nodiv"].values():
            ind = plan.simulate(*_plan_arrays(r["best"]))
            reeval.append(ind.theta[-1] / 1000.0)
            eq8 += any("dividend month" in b for b in checker.check(ind)[0])
        rows = [("$K=20$ (main)", base), ("$K=15$", table.get(f"{solver}-K15")), ("$K=25$", table.get(f"{solver}-K25")),
                ("no dividends", table[f"{solver}-nodiv"])]
        for name, rr in rows:
            if rr is None:
                continue
            extra = (f"{np.mean(reeval):,.0f} ({eq8}/{len(reeval)})" if name == "no dividends" else "---")
            lines.append(f"{INST.get(inst, inst)} & {name} & {fmt_ms(col(rr, 'cash'), 0)} & {fmt_ms(col(rr, 'hv'), 3)} & "
                         f"{np.median(col(rr, 'size')):g} & {extra} \\\\")
        lines.append("\\midrule")
    return "\n".join(lines[:-1]) + "\n" if lines else ""


def _paired(table, a, b, key):
    """Paired difference a - b over common seeds: mean, bootstrap CI and Wilcoxon p."""
    ra, rb = {r["seed"]: r[key] for r in table[a]}, {r["seed"]: r[key] for r in table[b]}
    seeds = sorted(set(ra) & set(rb))
    x, y = np.array([ra[s] for s in seeds]), np.array([rb[s] for s in seeds])
    m, lo, hi = stats.bootstrap_ci(x - y)
    return {"diff": m, "lo": lo, "hi": hi, "p": stats.wilcoxon(x, y)[1], "n": len(seeds)}


def elite_table(tables):
    """Pre-specified ablation of the SA anchor. Per instance and indicator (terminal cash, HV), the
    16 contrasts eSA-X, eRI-X, eSA-eRI and eSA-SA over the four bases are tested with the Wilcoxon
    signed-rank test and Holm-adjusted within that family of 16. eSA cells carry markers:
    x / r / s = significantly better than X / eRI / SA, and -x / -r / -s = significantly worse."""
    out, lines = {}, []
    for inst, table in tables.items():
        if not all(f"{b}+{m}" in table for b in BASES for m in ("sa", "sae", "rie")):
            continue
        res = {}
        for key in ("cash", "hv"):
            pairs = [(f"{b}+sae", b) for b in BASES] + [(f"{b}+rie", b) for b in BASES] + \
                    [(f"{b}+sae", f"{b}+rie") for b in BASES] + [(f"{b}+sae", f"{b}+sa") for b in BASES]
            cs = [_paired(table, a, b, key) for a, b in pairs]
            adj, rej = stats.holm([c["p"] for c in cs])
            for (a, b), c, pa, rj in zip(pairs, cs, adj, rej):
                c.update(p_holm=float(pa), reject=bool(rj))
            res[key] = {f"{a}|{b}": c for (a, b), c in zip(pairs, cs)}
        out[inst] = res
        for b in BASES:
            cells = [INST.get(inst, inst), NAME[b]]
            for key, fmt, dec in (("hv", "{:.3f}", 3), ("cash", "{:,.0f}", 0)):
                order = (("x", b), ("sa", f"{b}+sa"), ("sae", f"{b}+sae"), ("rie", f"{b}+rie"))
                vals = {m: np.mean(col(table[l], key)) for m, l in order}
                top = best_set([vals[m] for m, _ in order], dec, True)   # highest mean of the row, in bold
                marks = []
                for tag, other in (("x", b), ("r", f"{b}+rie"), ("s", f"{b}+sa")):
                    c = res[key][f"{b}+sae|{other}"]
                    if c["reject"]:
                        marks.append(tag if c["diff"] > 0 else f"-{tag}")
                txt = [bf(fmt.format(vals[m]), i in top) for i, (m, _) in enumerate(order)]
                txt[2] += f"$^{{{','.join(marks)}}}$" if marks else ""
                cells += txt
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    return ("\n".join(lines[:-1]) + "\n") if lines else "", out


CHI0 = (0.2, 0.8)     # starting acceptance probabilities of the sensitivity runs (main runs: 0.5)


def chi0_table(tables):
    """Sensitivity of the SA starting temperature. Per 2023 instance and indicator (terminal cash, HV),
    the 16 contrasts of SA with chi_0 = 0.2 and 0.8 against the base X and against the main SA
    (chi_0 = 0.5), over the four bases, are tested with the Wilcoxon signed-rank test and Holm-adjusted
    within that family of 16. The chi_0 = 0.2 and 0.8 cells carry markers: x / s = significantly better
    than X / SA, and -x / -s = significantly worse. The header comment also gives, per chi_0, the range
    over instances and bases of the share of worsening moves accepted per SA phase (as sa_acceptance)."""
    out, lines = {}, []
    variant = lambda b, c: f"{b}+sa-chi{c:g}"
    for inst, table in tables.items():
        if not all(variant(b, c) in table for b in BASES for c in CHI0):
            continue
        res = {}
        for key in ("cash", "hv"):
            pairs = [(variant(b, c), ref) for c in CHI0 for b in BASES for ref in (b, f"{b}+sa")]
            cs = [_paired(table, a, b, key) for a, b in pairs]
            adj, rej = stats.holm([c["p"] for c in cs])
            for c, pa, rj in zip(cs, adj, rej):
                c.update(p_holm=float(pa), reject=bool(rj))
            res[key] = {f"{a}|{b}": c for (a, b), c in zip(pairs, cs)}
        out[inst] = res
        for b in BASES:
            cells = [INST.get(inst, inst), NAME[b]]
            for key, fmt, dec in (("hv", "{:.3f}", 3), ("cash", "{:,.0f}", 0)):
                order = [b, variant(b, CHI0[0]), f"{b}+sa", variant(b, CHI0[1])]
                vals = [np.mean(col(table[l], key)) for l in order]
                top = best_set(vals, dec, True)          # highest mean of the row, in bold
                txt = [bf(fmt.format(v), i in top) for i, v in enumerate(vals)]
                for i in (1, 3):
                    marks = []
                    for tag, other in (("x", b), ("s", f"{b}+sa")):
                        c = res[key][f"{order[i]}|{other}"]
                        if c["reject"]:
                            marks.append(tag if c["diff"] > 0 else f"-{tag}")
                    txt[i] += f"$^{{{','.join(marks)}}}$" if marks else ""
                cells += txt
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    if not out:
        return "", {}
    acc = {}
    for c, suffix in ((CHI0[0], f"-chi{CHI0[0]:g}"), (0.5, ""), (CHI0[1], f"-chi{CHI0[1]:g}")):
        vals = {f"{inst}|{b}": float(np.mean([sum(a for _, a in r["sa_log"]) / max(1, sum(w for w, _ in r["sa_log"]))
                                              for r in tables[inst][f"{b}+sa{suffix}"] if r["sa_log"]]))
                for inst in out for b in BASES}
        acc[f"{c:g}"] = {"range": [min(vals.values()), max(vals.values())], "values": vals}
    sig = sorted(f"{INST.get(i, i)} {k} {p}" for i, r in out.items() for k, d in r.items() for p, c in d.items() if c["reject"])
    head = "% significant contrasts (Holm within the 16 of each instance and indicator): " + ("; ".join(sig) if sig else "none") + "\n"
    head += "% worsening moves accepted per SA phase (range over instances and bases): " + "; ".join(
        f"chi0={c}: {100 * a['range'][0]:.0f}-{100 * a['range'][1]:.0f}%" for c, a in acc.items()) + "\n"
    return head + "\n".join(lines[:-1]) + "\n", {"contrasts": out, "acceptance": acc}


def sa_mechanism(tables):
    """Per injection mode: share of generations whose SA phase returns a new plan, share of new plans
    that survive selection, and new injected plans in the final set per run; pooled and per base."""
    modes = (("sa", "SA (own chain)"), ("ri", "Random injection (own chain)"),
             ("sae", "eSA (elite anchor)"), ("rie", "eRI (elite anchor)"))
    out, lines = {}, []
    for m, title in modes:
        rows = {b: [r for inst, t in tables.items() if any(l.endswith("+sae") for l in t)
                    for r in t.get(f"{b}+{m}", []) if r["diag"]] for b in BASES}
        def summarize(rs):
            d = np.array([g for r in rs for g in r["diag"]], dtype=float)
            if not len(d):
                return None
            new = d[:, 1] == 0
            return {"new": float(new.mean()), "survive_new": float(d[new, 0].mean()) if new.any() else float("nan"),
                    "in_final": float(np.mean([r["in_final"] for r in rs])), "runs": len(rs)}
        per = {b: summarize(rs) for b, rs in rows.items()}
        pooled = summarize([r for rs in rows.values() for r in rs])
        if pooled is None:
            continue
        out[m] = {"pooled": pooled, "per_base": per}
        lines.append(f"{title} & {100 * pooled['new']:.0f}\\% & {100 * pooled['survive_new']:.0f}\\% & {pooled['in_final']:.2f} \\\\")
    return "\n".join(lines) + "\n", out


def generality_table(tables, tests, sizes):
    """One row per instance: number of stocks, configuration with the best mean HV and terminal cash,
    and Friedman p-values. The SA-vs-base contrasts that are significant after the Holm adjustment
    (4 bases x 2 indicators; + favours SA, - favours the base) go to the header comment and to the
    summary; the paper's caption states that there are none."""
    lines, out = [], {}
    for inst in [i for i in INST if i in tables]:
        table, res = tables[inst], tests[inst]
        labels = [l for l in MAIN if l in table]
        best = {k: max(labels, key=lambda l: np.mean(col(table[l], k))) for k in ("hv", "cash")}
        sig = []
        for key, tag in (("hv", "HV"), ("cash", "cash")):
            for b in BASES:
                c = contrast(res[key], f"{b}+sa", b)
                if c["reject"]:
                    sig.append(f"{'+' if c['diff'] > 0 else '-'}{tag} {NAME[b]}")
        out[inst] = {"best_hv": best["hv"], "best_cash": best["cash"], "friedman_hv_p": res["hv"]["p"],
                     "friedman_cash_p": res["cash"]["p"], "sa_significant": sig}
        lines.append(f"{INST[inst]} & {sizes[inst]} & {pretty(best['hv'])} & {pretty(best['cash'])} & "
                     f"{tex_p(res['hv']['p'])} & {tex_p(res['cash']['p'])} \\\\")
    sig = {INST[i]: v["sa_significant"] for i, v in out.items() if v["sa_significant"]}
    if sig:
        print(f"warning: significant SA-vs-base contrasts {sig}; the generality caption says none", file=sys.stderr)
    head = "% significant SA-vs-base contrasts (HV, cash; Holm): " + (
        "; ".join(f"{k}: {', '.join(v)}" for k, v in sig.items()) if sig else "none on any instance") + "\n"
    return head + "\n".join(lines) + "\n", out


def timing_means(timing_dir):
    out = defaultdict(dict)
    if not timing_dir or not os.path.isdir(timing_dir):
        return out
    runs = load(timing_dir)
    for inst, by_label in runs.items():
        for label, by_seed in by_label.items():
            out[inst][label] = np.array([r["runtime"] for r in by_seed.values()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--timing", default=os.path.join(ROOT, "results_timing"))
    ap.add_argument("--solver", default="spea2", help="solver of the no-dividend / K runs")
    ap.add_argument("--paper", default=os.path.join(os.path.dirname(ROOT), "Hybrid SA--NSGA-II for cash-dividend portfolios"))
    args = ap.parse_args()
    global RUNS
    runs = RUNS = load(args.results)
    timing = timing_means(args.timing)
    tdir, fdir = os.path.join(args.paper, "tables"), os.path.join(args.paper, "figures")
    tables, summary, base_cash, all_tests, sizes = {}, {}, {}, {}, {}
    for inst in [i for i in INST if i in runs] + [i for i in runs if i not in INST]:
        table, _ = indicators(runs[inst])
        tables[inst] = table
        plan.load_instance(os.path.join(ROOT, "data", "instances", f"{inst}.json"),
                           os.path.join(ROOT, "data", "instances", f"{inst}_moments.npz"))
        base_cash[inst] = {"deposit": baselines.deposit_only().theta[-1] / 1000,
                           "eqw_topk": baselines.equal_weight("topK").theta[-1] / 1000,
                           "eqw_all": baselines.equal_weight("all").theta[-1] / 1000}
        # the CPU-time column belongs to the 2023 table (the timing pass covers the 2023 index instances)
        write(os.path.join(tdir, f"summary_{inst}.tex"),
              summary_table(inst, table, timing, base_cash[inst], with_ct=inst.endswith("_2023")))
        tex, res = tests_table(table)
        write(os.path.join(tdir, f"tests_{inst}.tex"), tex)
        all_tests[inst], sizes[inst] = res, len(plan.SYMBOLS)
        write(os.path.join(tdir, f"coverage_{inst}.tex"), coverage_table(table))
        checks = check_reported(runs[inst])
        summary[inst] = {
            "baselines": base_cash[inst],
            "configs": {l: {k: [float(np.nanmean(col(table[l], k))), float(np.nanstd(col(table[l], k), ddof=1)) if len(table[l]) > 1 else 0.0]
                            for k in ("cash", "hv", "igd", "sp", "size", "evals", "gens")} for l in table},
            "friedman": {k: [res[k]["chi2"], res[k]["p"]] for k in ("cash", "hv", "sp")},
            "contrasts": {k: {f"{a}|{b}": v for (a, b), v in res[k]["pairs"].items()} for k in res},
            "spea2_truncation_runs": {l: int(sum(1 for r in table[l] if r["trunc"])) for l in table if l.startswith("spea2")},
            "spea2_truncations_per_run": {l: [float(np.mean(col(table[l], "trunc"))), float(np.min(col(table[l], "trunc"))),
                                              float(np.max(col(table[l], "trunc"))), float(np.mean(col(table[l], "gens")))]
                                          for l in table if l.startswith("spea2") and all(r["trunc"] is not None for r in table[l])},
            "sa_acceptance": {l: float(np.mean([sum(a for _, a in r["sa_log"]) / max(1, sum(w for w, _ in r["sa_log"]))
                                                for r in table[l] if r["sa_log"]]))
                              for l in table if l.endswith("+sa")},
            "checker": checks,
        }
    tex, milp = milp_table(os.path.join(args.results, "milp_small.json"))
    if tex:
        write(os.path.join(tdir, "milp.tex"), tex)
        summary["milp"] = milp
    if tables:
        write(os.path.join(tdir, "lex_ablation.tex"), lex_table(tables))
        tex = extra_table(tables, args.solver)
        if tex:
            write(os.path.join(tdir, "extra.tex"), tex)
        tex, elite = elite_table(tables)
        if tex:
            write(os.path.join(tdir, "elite_sa.tex"), tex)
            summary["elite"] = elite
        tex, chi = chi0_table(tables)
        if tex:
            write(os.path.join(tdir, "chi0.tex"), tex)
            summary["chi0"] = chi
        tex, gen = generality_table(tables, all_tests, sizes)
        write(os.path.join(tdir, "generality.tex"), tex)
        summary["generality"] = gen
        tex, mech = sa_mechanism(tables)
        write(os.path.join(tdir, "sa_mechanism.tex"), tex)
        summary["sa_mechanism"] = mech
        summary["timing"] = {inst: {l: [float(np.mean(v)), float(np.std(v, ddof=1))] for l, v in t.items()}
                             for inst, t in timing.items()}
        tex, ranks = ranks_table(tables)
        write(os.path.join(tdir, "ranks.tex"), tex)
        summary["ranks"] = ranks
        figures(tables, base_cash, fdir)
    with open(os.path.join(ROOT, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"instances: {list(tables)}; tables -> {tdir}; figures -> {fdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
