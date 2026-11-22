"""Aggregate the comparison across all available methods on VN30 and HNX30.

Methods loaded (each present iff its JSON / legacy block exists):
  - GA, NSGA-II, HSG   (legacy arrays from `resultAnalyzers_withSignificantTests.py`)
  - SPEA2              (`spea2_results_<dataset>.json`)
  - NSGA-III           (`nsga3_results_<dataset>.json`)
  - SA-NSGA-III        (`sa_nsga3_results_<dataset>.json`)
  - Equal-weight       (`equal_weight_results.json`, deterministic point ref)

Reports per dataset:
  - per-method profit / runtime / GN / DN summary stats
  - Friedman omnibus + Wilcoxon/Holm pairwise on profit and runtime
  - all directed coverage statistics CS(A->B) over the MOEAs
  - spacing SP for each MOEA on the normalized union
  - equal-weight as a deterministic point reference at the top of the
    profit table

Missing JSON files are silently skipped, so the same script works on a
partial sweep (e.g., before the NSGA-III runs are available)."""

import json
import os
import re
import numpy as np
from itertools import combinations
from scipy import stats


# -----------------------------------------------------------------------------
# Load GA / NSGA-II / HSG arrays from the existing analyzer file
# -----------------------------------------------------------------------------
def _load_legacy_arrays():
    with open("resultAnalyzers_withSignificantTests.py") as f:
        src = f.read()

    # VN30 block lives in commented lines 223..236 (one assignment per line,
    # each starting with "# ").  HNX30 block is in active code from line 243.
    legacy = {"VN30": {}, "HNX30": {}}
    name_map = {
        "ga_time": "ga_time", "ga_dn": "ga_dn", "ga_gn": "ga_gn", "ga": "ga",
        "nsga_ii_time": "nsga_ii_time", "nsga_ii_dn": "nsga_ii_dn",
        "nsga_ii_gn": "nsga_ii_gn", "nsga_ii": "nsga_ii",
        "hsg_time": "hsg_time", "hsg_dn": "hsg_dn", "hsg_gn": "hsg_gn", "hsg": "hsg",
    }

    # VN30: the file holds the VN30 arrays inside a single comment block before
    # the active HNX30 section.  Extract by stripping "# " from those lines.
    vn30_section = re.search(
        r"# VN30 runtime 1.*?(?=# end\s*\*|# *$\n# *\n# *|^ga_time\s*=)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    if vn30_section is None:
        # Fallback: take the block between "# VN30 runtime 1" and "# HNX runs"
        vn30_section = re.search(
            r"# VN30 runtime 1.*?(?=# HNX|^###)",
            src,
            re.DOTALL | re.MULTILINE,
        )
    if vn30_section is None:
        raise RuntimeError("could not locate VN30 commented block")
    block = vn30_section.group(0)
    # Uncomment lines that look like assignments
    uncommented_lines = []
    for line in block.splitlines():
        if line.startswith("# ") and "=" in line:
            uncommented_lines.append(line[2:])
    vn30_src = "\n".join(uncommented_lines)
    ns = {}
    exec(vn30_src, ns)
    for k in name_map:
        legacy["VN30"][k] = ns[k]

    # HNX30: active section at lines ~243..256
    hnx_section = re.search(
        r"^ga_time\s*=.*?^hsg\s*=\s*\[.*?\]\]\s*$",
        src,
        re.DOTALL | re.MULTILINE,
    )
    if hnx_section is None:
        raise RuntimeError("could not locate HNX30 active block")
    ns2 = {}
    exec(hnx_section.group(0), ns2)
    for k in name_map:
        legacy["HNX30"][k] = ns2[k]

    return legacy


def _load_method_json(name, json_basename):
    """Load `<json_basename>_<name>.json` if present; otherwise return None."""
    path = f"{json_basename}_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return {
        "time": d["log_times"],
        "dn": d["log_dn"],
        "gn": d["log_gn"],
        "fits": d["log_fitnesses"],
        "archives": d["archives"],
    }


def _load_eqw():
    with open("equal_weight_results.json") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Pareto-set metrics on the (lower-better-on-objs[0,2,3], higher-better-on-objs[1,4]) convention
# -----------------------------------------------------------------------------
def is_dominated(dominator, dominated_guy):
    """Standard Pareto dominance under the
    (min var, max skew, min kurt, min entropy, max profit) orientation."""
    a, b = dominator, dominated_guy
    if a[0] > b[0] or a[2] > b[2] or a[3] > b[3]:
        return False
    if a[1] < b[1] or a[4] < b[4]:
        return False
    if a[0] < b[0] or a[2] < b[2] or a[3] < b[3]:
        return True
    if a[1] > b[1] or a[4] > b[4]:
        return True
    return False


def coverage(set_A, set_B):
    """Fraction of B-points that are dominated by some A-point."""
    if not set_B:
        return 0.0
    cnt = 0
    for b in set_B:
        for a in set_A:
            if is_dominated(a, b):
                cnt += 1
                break
    return cnt / len(set_B)


def spacing(S):
    """Spacing metric (Jiang et al. 2014) on a non-dominated set in obj space."""
    if len(S) < 2:
        return 0.0
    arr = np.array(S, dtype=float)
    d = []
    for i in range(len(arr)):
        diffs = arr - arr[i]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        dists = np.sort(dists)
        d.append(dists[1])  # nearest non-self
    d = np.array(d)
    dbar = d.mean()
    return float(np.sqrt(((d - dbar) ** 2).sum() / (len(d) - 1)))


# Min-max normalize a set of objective vectors before CS / SP computation,
# matching `nom_experiment_results` in resultAnalyzers_withSignificantTests.py.
def _normalize_vstack(*sets):
    """Concatenate sets, compute population-wide min-max, return list of normalized lists."""
    cnt = [len(s) for s in sets]
    arr = np.vstack([np.array(s, dtype=float) for s in sets])
    fmin = arr.min(axis=0)
    fmax = arr.max(axis=0)
    span = np.where(fmax - fmin > 0, fmax - fmin, 1.0)
    norm = (arr - fmin) / span
    out = []
    cursor = 0
    for c in cnt:
        out.append(norm[cursor:cursor + c].tolist())
        cursor += c
    return out


# -----------------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------------
def holm(pvals, pairs, alpha=0.05):
    order = np.argsort(pvals)
    adjusted = {}
    reject = {p: False for p in pairs}
    for i, idx in enumerate(order):
        m_i = len(pvals) - i
        adjusted[pairs[idx]] = min(pvals[idx] * m_i, 1.0)
        if pvals[idx] <= alpha / m_i:
            reject[pairs[idx]] = True
    return adjusted, reject


def compare_methods(metric_name, data_dict, alpha=0.05):
    """data_dict: ordered dict {label: array of len 30}.  Run Friedman + Wilcoxon/Holm."""
    labels = list(data_dict.keys())
    arrays = [np.asarray(data_dict[l], dtype=float) for l in labels]
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays), "lengths differ"
    chi2, p_omn = stats.friedmanchisquare(*arrays)
    print(f"\n=== {metric_name} ===")
    for l, a in zip(labels, arrays):
        print(f"  {l:8s}: mean={a.mean():.4g}, median={np.median(a):.4g}, sd={a.std(ddof=1):.4g}")
    print(f"  Friedman chi2={chi2:.3f}, p={p_omn:.4g}")
    if p_omn >= alpha:
        print("  No omnibus significance -> stop")
        return
    pairs = list(combinations(labels, 2))
    pvals = []
    stats_w = []
    effects = {}
    for a, b in pairs:
        diff = np.asarray(data_dict[a]) - np.asarray(data_dict[b])
        if not np.any(diff):
            pvals.append(1.0)
            stats_w.append(0.0)
            effects[(a, b)] = (0.0, (0.0, 0.0))
            continue
        W, p = stats.wilcoxon(data_dict[a], data_dict[b], alternative="two-sided")
        pvals.append(p)
        stats_w.append(W)
        dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
        rng = np.random.default_rng(0)
        boots = rng.choice(diff, size=(10000, diff.size), replace=True).mean(axis=1)
        ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        effects[(a, b)] = (dz, ci)
    adj, rej = holm(pvals, pairs, alpha)
    print("  Pairwise (Holm-corrected):")
    for (a, b), W, p_raw in zip(pairs, stats_w, pvals):
        dz, (lo, hi) = effects[(a, b)]
        keep = "Reject H0" if rej[(a, b)] else "n.s."
        print(f"    {a:8s} vs {b:8s}  W={W:.1f}  p_raw={p_raw:.3g}  "
              f"p_holm={adj[(a,b)]:.3g}  -> {keep}  dz={dz:.3f}  CI=[{lo:.3g}, {hi:.3g}]")


# -----------------------------------------------------------------------------
# Main analysis per dataset
# -----------------------------------------------------------------------------
METHOD_LOADERS = [
    # (label, json basename, legacy-key prefix or None)
    # legacy-key prefix is the name used in `_load_legacy_arrays` for the
    # hardcoded fallback (only the three original methods have a fallback).
    ("GA",          "ga_results",       "ga"),
    ("NSGA-II",     "nsga_ii_results",  "nsga_ii"),
    ("HSG",         "hsg_results",      "hsg"),
    ("SPEA2",       "spea2_results",    None),
    ("SA-SPEA2",    "sa_spea2_results", None),
    ("NSGA-III",    "nsga3_results",    None),
    ("SA-NSGA-III", "sa_nsga3_results", None),
    ("MOEA/D",      "moead_results",    None),
    ("SA-MOEA/D",   "sa_moead_results", None),
]


def analyze(dataset, legacy, eqw_records, normalize_for_cs=True):
    print("\n" + "=" * 78)
    print(f" {dataset}")
    print("=" * 78)

    ds_lower = dataset.lower()
    L = legacy[dataset]

    methods = []
    sets, ret_arrays, rt_arrays, gn_arrays, dn_arrays = {}, {}, {}, {}, {}

    for label, basename, legacy_key in METHOD_LOADERS:
        d = _load_method_json(ds_lower, basename)
        if d is not None:
            methods.append(label)
            sets[label] = d["fits"]
            ret_arrays[label] = [r[4] for r in d["fits"]]
            rt_arrays[label] = d["time"]
            gn_arrays[label] = d["gn"]
            dn_arrays[label] = d["dn"]
            print(f"  {label:12s}: loaded from {basename}_{ds_lower}.json")
        elif legacy_key is not None:
            methods.append(label)
            sets[label] = L[legacy_key]
            ret_arrays[label] = [r[4] for r in L[legacy_key]]
            rt_arrays[label] = L[f"{legacy_key}_time"]
            gn_arrays[label] = L[f"{legacy_key}_gn"]
            dn_arrays[label] = L[f"{legacy_key}_dn"]
            print(f"  {label:12s}: using legacy hardcoded arrays (no JSON yet)")
        else:
            print(f"  {label:12s}: skipped ({basename}_{ds_lower}.json not found)")

    # Equal-weight: deterministic baseline as point estimate
    eqw_topk = next(r for r in eqw_records if r["dataset"] == ds_lower and r["universe"] == "topK")
    eqw_all = next(r for r in eqw_records if r["dataset"] == ds_lower and r["universe"] == "all")
    print(f"\nEqual-weight buy-and-hold (deterministic):")
    print(f"  top-K=20 :  terminal = {eqw_topk['finalCash']:.2f} kVND")
    print(f"  all : terminal = {eqw_all['finalCash']:.2f} kVND")

    compare_methods("Profit / Terminal cash", ret_arrays)
    compare_methods("Runtime (s)", rt_arrays)
    compare_methods("GN (generations to last pivot)", gn_arrays)
    compare_methods("DN (frontiers passed)", dn_arrays)

    # CS pairs (normalized over the union to match the existing pipeline)
    print("\n=== Coverage of two sets CS(A->B) ===")
    if normalize_for_cs:
        norm_lists = _normalize_vstack(*[sets[m] for m in methods])
        norm_sets = dict(zip(methods, norm_lists))
    else:
        norm_sets = sets
    print(f"  {'A':12s} ->{'B':12s}  CS")
    for a in methods:
        for b in methods:
            if a == b:
                continue
            c = coverage(norm_sets[a], norm_sets[b])
            print(f"  {a:12s} -> {b:12s}  {c:.3f}")

    # SP per method on the normalized union
    print("\n=== Spacing SP (lower = more uniform) ===")
    for m in methods:
        print(f"  {m:12s}  SP = {spacing(norm_sets[m]):.4f}")


def main():
    legacy = _load_legacy_arrays()
    eqw = _load_eqw()
    analyze("VN30", legacy, eqw)
    analyze("HNX30", legacy, eqw)


if __name__ == "__main__":
    main()
