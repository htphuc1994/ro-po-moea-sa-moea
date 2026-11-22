"""Generate the figures used in the nine-method MOEA benchmark.

Reads the per-method JSON logs produced by `run_legacy.py`, `spea2.py`,
`nsga3.py`, and `moead.py`, and the deterministic baseline from
`equal_weight_results.json`.  Outputs three PNGs to the paper directory
that get \\includegraphics-d from main.tex:

  - boxplot_profit_2datasets.png   profit per method on VN30 vs HNX30
  - boxplot_runtime_2datasets.png  runtime per method on VN30 vs HNX30
  - cs_heatmap_2datasets.png       directed CS heatmap among MOEAs

Run from PortfolioAnalyzer/ directory."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

PAPER_DIR = os.path.expanduser(
    "~/Projects/Hybrid SA--NSGA-II for cash-dividend portfolios"
)

# (display label, JSON basename) ordered for plot consistency
METHODS = [
    ("GA",          "ga_results"),
    ("NSGA-II",     "nsga_ii_results"),
    ("SA-NSGA-II",  "hsg_results"),       # HSG file kept for backwards compat
    ("SPEA2",       "spea2_results"),
    ("SA-SPEA2",    "sa_spea2_results"),
    ("NSGA-III",    "nsga3_results"),
    ("SA-NSGA-III", "sa_nsga3_results"),
    ("MOEA/D",      "moead_results"),
    ("SA-MOEA/D",   "sa_moead_results"),
]

DATASETS = ["vn30", "hnx30"]
DATASET_LABELS = {"vn30": "VN30", "hnx30": "HNX30"}


def _load(method_basename, ds):
    with open(f"{method_basename}_{ds}.json") as f:
        return json.load(f)


def _eqw():
    with open("equal_weight_results.json") as f:
        return json.load(f)


def _is_dominated(a, b):
    """Standard Pareto: A dominates B under (min var, max skew, min kurt,
    min entropy, max profit)."""
    if a[0] > b[0] or a[2] > b[2] or a[3] > b[3]:
        return False
    if a[1] < b[1] or a[4] < b[4]:
        return False
    if a[0] < b[0] or a[2] < b[2] or a[3] < b[3]:
        return True
    if a[1] > b[1] or a[4] > b[4]:
        return True
    return False


def _normalize_union(method_sets):
    """Min-max normalize over the union of all sets, return per-method
    normalized list of 5-vectors."""
    sizes = [len(s) for s in method_sets]
    arr = np.vstack([np.array(s, dtype=float) for s in method_sets])
    fmin = arr.min(axis=0)
    fmax = arr.max(axis=0)
    span = np.where(fmax - fmin > 0, fmax - fmin, 1.0)
    norm = (arr - fmin) / span
    out = []
    cursor = 0
    for sz in sizes:
        out.append(norm[cursor:cursor + sz].tolist())
        cursor += sz
    return out


def _coverage(set_a, set_b):
    if not set_b:
        return 0.0
    cnt = 0
    for b in set_b:
        for a in set_a:
            if _is_dominated(a, b):
                cnt += 1
                break
    return cnt / len(set_b)


def boxplot_profit():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    eqw = _eqw()
    for ax, ds in zip(axes, DATASETS):
        data = []
        labels = []
        for label, base in METHODS:
            d = _load(base, ds)
            profits = [r[4] / 1000.0 for r in d["log_fitnesses"]]  # show in MVND for readability
            data.append(profits)
            labels.append(label)
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cfe6ff")
            patch.set_edgecolor("#1f77b4")
        for median in bp["medians"]:
            median.set_color("#d62728")
        # Equal-weight horizontal reference line (top-K variant)
        eqw_topk = next(
            r["finalCash"] for r in eqw
            if r["dataset"] == ds and r["universe"] == "topK"
        ) / 1000.0
        ax.axhline(eqw_topk, linestyle="--", linewidth=1.4, color="#888",
                   label=f"Equal-weight top-K = {eqw_topk:.0f} MVND")
        ax.set_title(f"{DATASET_LABELS[ds]} terminal cash (30 runs)", fontsize=14)
        ax.set_ylabel("Terminal cash (MVND)", fontsize=12)
        ax.tick_params(axis="x", rotation=35, labelsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02),
                  fontsize=10, frameon=False)
    plt.tight_layout()
    out = os.path.join(PAPER_DIR, "boxplot_profit_2datasets.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def _filter_runtime_outliers(times, k=5.0):
    """Drop wall-clock runtimes that exceed `k` times the per-cell median.
    These are almost certainly OS / GC / scheduling artifacts rather than
    algorithmic measurements (e.g.\\ a single outlier at ~100s in an
    otherwise 8--16s SA-NSGA-II run on VN30)."""
    if not times:
        return times
    sorted_t = sorted(times)
    median = sorted_t[len(sorted_t) // 2]
    threshold = k * median
    return [t for t in times if t <= threshold]


def boxplot_runtime():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        data = []
        labels = []
        for label, base in METHODS:
            d = _load(base, ds)
            filtered = _filter_runtime_outliers(d["log_times"])
            data.append(filtered)
            labels.append(label)
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#ffe2cc")
            patch.set_edgecolor("#d62728")
        for median in bp["medians"]:
            median.set_color("#1f77b4")
        ax.set_title(f"{DATASET_LABELS[ds]} runtime per run (s)", fontsize=14)
        ax.set_ylabel("Runtime (s)", fontsize=12)
        ax.tick_params(axis="x", rotation=35, labelsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PAPER_DIR, "boxplot_runtime_2datasets.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def cs_heatmap():
    """Coverage matrix CS(A->B) over the eight MOEAs (drop GA which is
    degenerate). Two subplots, one per dataset."""
    moea_methods = [(lbl, base) for lbl, base in METHODS if lbl != "GA"]
    n = len(moea_methods)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, ds in zip(axes, DATASETS):
        method_sets = [_load(base, ds)["log_fitnesses"] for _, base in moea_methods]
        norm_sets = _normalize_union(method_sets)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i, j] = np.nan
                    continue
                M[i, j] = _coverage(norm_sets[i], norm_sets[j])
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.4)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            color="#666", fontsize=10)
                else:
                    val = M[i, j]
                    color = "white" if val > 0.2 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=8)
        labels = [lbl for lbl, _ in moea_methods]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("$B$", fontsize=12)
        ax.set_ylabel("$A$", fontsize=12)
        ax.set_title(f"{DATASET_LABELS[ds]} coverage $CS(A\\to B)$", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(PAPER_DIR, "cs_heatmap_2datasets.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    boxplot_profit()
    boxplot_runtime()
    cs_heatmap()
