# MOEA and SA-MOEA for cash-dividend portfolios — code and data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24499526.svg)](https://doi.org/10.5281/zenodo.24499526)

Code, instance data and run files for the manuscript *"MOEA and SA-MOEA for cash-dividend
portfolios: evidence from Vietnamese equities"* (second revision). Every table and figure of
the paper's results section can be regenerated from this repository.

## Contents

| Path | What it holds |
|---|---|
| `plan.py` | Multi-period model: cash recursion, risk objectives on capital weights, constructive sampler, SA phase |
| `dominance.py`, `core.py` | Constraint-domination (and the lexicographic rule used in the ablation), non-dominated sorting, crowding |
| `solvers.py` | GA, NSGA-II, SPEA2, NSGA-III, MOEA/D; SA injection, random-injection control, elite-anchored variants, SA starting temperature |
| `checker.py` | Independent check of every constraint of a plan |
| `milp_small.py` | Exact maximum terminal cash of small instances (mixed-integer linear program, HiGHS) |
| `baselines.py` | Deposit-only and equal-weight references |
| `metrics.py`, `stats.py` | Hypervolume, IGD+, spacing, coverage; Friedman, Wilcoxon, Holm, bootstrap |
| `run_all.py` | Experiment matrix; one JSON file per run |
| `analysis.py` | Tables, figures and `results_summary.json` from the run files |
| `data/instances/` | The seven instances (VN30 and HNX30 for 2023, 2024 and 2025; VN30+HNX30 for 2023): monthly planning data (`*.json`) and daily return inputs (`*_moments.npz`) |
| `data/` | Index universes (`universes.json`) and the first-round 2023 data (`vn30_2023.json`, `hnx30_2023.json`) |
| `results/`, `results_timing/` | All run files (`results/<instance>/<configuration>/seed_<seed>.json`) and the timing pass |
| `results_summary.json` | Every number reported in the paper |
| `tests/` | Unit tests |
| `legacy/` | First-round implementation and outputs, kept for traceability |

Configuration labels: `<solver>[+sa|+ri|+sae|+rie][@lex][-nodiv][-K<k>][-chi<p>]`. `+sa` is SA
injection with its own chain, `+ri` its random-injection control, `+sae`/`+rie` the elite-anchored
variants, `@lex` the lexicographic comparison rule, `-nodiv` the run without dividends, `-K<k>`
another cardinality cap and `-chi<p>` another starting temperature of the SA phase (an average
worsening move is accepted with probability `p` at the start of a phase; 0.5 in the main runs).

## Data

The instance files in `data/instances/` are the data of the paper. They are derived from the
public trading and corporate-action records of the Ho Chi Minh (HOSE) and Hanoi (HNX) stock
exchanges. Each instance uses the index constituents in force on 31 December of its year (VN30
from HOSE's July review, HNX30 from HNX's October review), listed in `data/universes.json`. The
monthly planning price of a stock is the month's highest daily closing price, the volume cap is
the month's total matched volume, and a cash dividend is assigned to the month of its ex-date.
The moment files hold the centred daily simple returns `C` of the three years before the
evaluation year and their covariance `M2`; the third and fourth co-moments follow exactly from
`C` (`plan.comoments`), and the model evaluates them as sample moments of the portfolio's daily
return. The daily return of a stock is its close-to-close return, adjusted for a dividend or share
issue that goes ex on that day: the adjustment is the exchange's reference price where the HNX record
carries it, and otherwise the factor implied by the published adjusted closing prices. Every return was
checked against the daily price limit of the board the stock traded on that day; on the eight
stock-days where the adjusted closing prices break it, the return comes from the reference price, and
one reference price is taken from a published adjustment notice (listed in each file under
`overrides`). A day on which a stock has no record has a zero return, unless the next record's reference
price shows that the stock traded that day; that move is recovered (listed under `recovered`). The scripts that downloaded and assembled the daily records are not part of this
release.

## Version 2.0.3

Daily returns adjusted for dividends and share issues from the exchange's reference prices where the
records carry them, every return checked against the daily price limit of its board (no return is set
to zero), and market days missing from the records recovered; the SA log counts every energy-raising
move and those of them decided by the Metropolis test (`sa.log` holds `[n_up, n_up_acc, n_met,
n_met_acc]` per generation). All experiments were rerun, so every run file and
`results_summary.json` differ from version 2.0.2.

## Version 2.0.2

Co-moments from daily simple returns (instead of log returns), with four impossible one-session
price moves set to zero; quality indicators on the distinct plans of each final set; random buy
sets of 0 to min(K, n) stocks; every small-instance plan stored and checked. All experiments were
rerun with these changes, so every run file and `results_summary.json` differ from version 2.0.1.

## Reproducing the results

Python 3.14 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
python3 -m pytest -q tests                        # unit tests
python3 run_all.py main lex extra elite chi0 --workers 8
python3 run_all.py timing                         # sequential runtime pass
python3 -c "import milp_small; milp_small.main()" # small-instance MILP study
python3 analysis.py --paper out                   # tables and figures into out/tables, out/figures
```

Runs are deterministic: the same seed reproduces a run exactly, and existing run files are
skipped, so the commands above only fill in what is missing. Seeds 1234–1263 give every solver the
same initial population (common random numbers).

The first-round inputs (`data/calculate2023/`) are not part of this release; they remain in the
repository history (commit `01a6115`).

## License

Code: MIT (`LICENSE`). Data and run files: CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

See `CITATION.cff`.
