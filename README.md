# ro-po-moea-sa-moea

Full MySQL data here: https://drive.google.com/drive/folders/1xLtc0QZssE-lmWDuGGrnzd_tlBVnrYSZ?usp=sharing

# Experiments

Final results for the dividend-portfolio comparison study. All runs were
executed on the same machine (Apple Silicon laptop, 2026-05-06) with
identical hyperparameters: pop / archive size $100$ (or $N=70$ for
MOEA/D from Das-Dennis $p=4$), max generations $150$, no-improvement
stopping at $79$ stagnant generations, $K=20$, $T_0=750$, $\lambda=0.95$,
$L=10$, $\beta_E=10^4$, MOEA/D neighborhood $T=20$, $\delta=0.9$, $n_r=2$.

## Solvers covered

Three Pareto-based MOEA bases (NSGA-II, SPEA2, NSGA-III) plus a
decomposition-based MOEA (MOEA/D). Each base is also run with the
HSG-style SA-injection wrapper (SA incumbent replaces the last
offspring slot each generation), giving four "vanilla + SA" pairs.
A single-objective GA baseline and a deterministic equal-weight
buy-and-hold baseline complete the picture.

| # | Solver | Source | Output JSON |
| --- | --- | --- | --- |
| 1 | GA | `plan.py` (re-run via `run_legacy.py`) | `ga_results_<ds>.json` |
| 2 | NSGA-II | `plan.py` (re-run via `run_legacy.py`) | `nsga_ii_results_<ds>.json` |
| 3 | HSG (SA-NSGA-II) | `plan.py` (re-run via `run_legacy.py`) | `hsg_results_<ds>.json` |
| 4 | SPEA2 | `spea2.py` | `spea2_results_<ds>.json` |
| 5 | SA-SPEA2 | `spea2.py --sa` | `sa_spea2_results_<ds>.json` |
| 6 | NSGA-III | `nsga3.py` | `nsga3_results_<ds>.json` |
| 7 | SA-NSGA-III | `nsga3.py` | `sa_nsga3_results_<ds>.json` |
| 8 | MOEA/D | `moead.py` | `moead_results_<ds>.json` |
| 9 | SA-MOEA/D | `moead.py --sa` | `sa_moead_results_<ds>.json` |
| – | Equal-weight (top-K, all) | `equal_weight.py` | `equal_weight_results.json` |

`<ds>` is `vn30` or `hnx30`. Datasets: VN30 and HNX30 monthly quotes +
corporate actions for 2023, with 30 independent runs per (solver,
dataset) cell. The seeds `seed_base + r` for $r \in \{0,\dots,29\}$
match across all stochastic methods so per-run differences are due
to algorithmic dynamics rather than RNG drift.

## Profit (terminal cash, kVND, 30 runs)

| Solver | VN30 mean | VN30 sd | HNX30 mean | HNX30 sd |
| --- | --- | --- | --- | --- |
| GA | 1,031,000 | 66,050 | 1,068,000 | 43,870 |
| NSGA-II | 1,951,000 | 31,580 | 3,115,000 | 214,200 |
| HSG | 1,942,000 | 44,600 | 3,168,000 | 231,600 |
| MOEA/D | 1,920,000 | 61,160 | 2,973,000 | 270,400 |
| SA-MOEA/D | 1,928,000 | 57,250 | 3,067,000 | 205,600 |
| SA-SPEA2 | 1,960,000 | 95,610 | 3,366,000 | 284,100 |
| SPEA2 | 1,995,000 | 29,930 | 3,358,000 | 157,700 |
| SA-NSGA-III | 1,998,000 | 31,930 | 3,383,000 | 168,200 |
| **NSGA-III** | **2,007,000** | **21,510** | **3,386,000** | 174,400 |
| Equal-weight (top-K=20) | 1,000,509 | – | 1,271,328 | – |
| Equal-weight (all) | 959,144 | – | 1,260,486 | – |

**Friedman omnibus on profit:** $\chi^2 = 148.4$, $p \approx 4.3 \times 10^{-28}$ (VN30); $\chi^2 = 150.5$, $p \approx 1.5 \times 10^{-28}$ (HNX30).

**Pairwise Wilcoxon / Holm-corrected** clusters at $\alpha=0.05$ (same on both datasets up to numeric variation):

- **Bottom**: GA — dominated by every MOEA ($p_{\rm Holm} < 10^{-7}$ in every comparison).
- **Lower-middle**: MOEA/D, SA-MOEA/D, NSGA-II, HSG, SA-SPEA2 — mostly mutually n.s.
- **Top tier**: SPEA2, SA-NSGA-III, NSGA-III — three-way statistical tie, all significantly above the lower-middle group ($p_{\rm Holm} < 10^{-3}$).

## Runtime (seconds, 30-run mean)

| Solver | VN30 | HNX30 |
| --- | --- | --- |
| MOEA/D | **6.15** | **6.15** |
| SA-MOEA/D | 6.73 | 6.77 |
| SPEA2 | 7.37 | 7.36 |
| GA | 7.49 | 7.55 |
| NSGA-II | 8.50 | 7.94 |
| SA-SPEA2 | 8.55 | 8.34 |
| NSGA-III | 9.02 | 8.74 |
| SA-NSGA-III | 9.86 | 10.08 |
| HSG | **12.56** | **9.17** |

The new laptop is roughly $10\times$ faster than the old one used in the
earlier draft of the paper, so the prior comparative-runtime numbers
are obsolete; only these are valid going forward.

## Spacing SP (lower = more uniform)

| Solver | VN30 | HNX30 |
| --- | --- | --- |
| GA | 0.0243 | 0.0071 |
| NSGA-II | 0.0486 | 0.0433 |
| HSG | 0.0624 | 0.0354 |
| **NSGA-III** | 0.0510 | **0.0258** |
| **SPEA2** | **0.0501** | 0.0495 |
| SA-MOEA/D | 0.0717 | **0.0425** |
| SA-NSGA-III | 0.0702 | 0.0539 |
| SA-SPEA2 | 0.1055 | 0.0708 |
| MOEA/D | 0.1632 | 0.0769 |

GA spacing is degenerate (it converges to deposit-only plans); ignore
it in cross-method comparisons. Among the MOEAs, **NSGA-III has the
best SP on HNX30 and is in the top tier on VN30**, which together with
its top-tier profit makes it the consensus winner.

## SA-injection effect summary

The SA-injection wrapper (HSG-style replacement of one offspring slot
each generation by an SA incumbent) was applied to all four MOEA bases.
None of them showed a statistically significant profit gain over the
vanilla counterpart (paired Wilcoxon / Holm $p \ge 0.05$ in every
SA-vs-vanilla pair, both datasets). The SP effect is mixed:

- HSG vs NSGA-II: SP **worse** on VN30 ($0.062$ vs $0.049$); slightly **better** on HNX30 ($0.035$ vs $0.043$).
- SA-SPEA2 vs SPEA2: SP **worse** on both datasets.
- SA-NSGA-III vs NSGA-III: SP **worse** on both datasets.
- SA-MOEA/D vs MOEA/D: SP dramatically **better** on both datasets ($0.072$ vs $0.163$ on VN30; $0.043$ vs $0.077$ on HNX30).

The takeaway is that SA injection is **only useful on MOEA/D**, where
it compensates for the decomposition-based diversity weakness; on
Pareto-based MOEAs it is at best profit-neutral and often hurts SP.

## How to reproduce

```bash
cd ~/Projects/PortfolioAnalyzer
python3 equal_weight.py                 # passive baseline
python3 spea2.py vn30 hnx30             # vanilla SPEA2
python3 spea2.py --sa vn30 hnx30        # SA-SPEA2
python3 nsga3.py vn30 hnx30             # NSGA-III + SA-NSGA-III
python3 moead.py vn30 hnx30             # vanilla MOEA/D
python3 moead.py --sa vn30 hnx30        # SA-MOEA/D
python3 run_legacy.py vn30 hnx30        # GA + NSGA-II + HSG
python3 analyze_with_baselines.py       # consolidated tables
```

`analyze_with_baselines.py` writes the human-readable comparison report
to `analyze_with_baselines_output.txt`.
