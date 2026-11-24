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
| `solvers.py` | GA, NSGA-II, SPEA2, NSGA-III, MOEA/D; SA injection, random-injection control, elite-anchored variants |
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

Configuration labels: `<solver>[+sa|+ri|+sae|+rie][@lex][-nodiv][-K<k>]`. `+sa` is SA injection with
its own chain, `+ri` its random-injection control, `+sae`/`+rie` the elite-anchored variants,
`@lex` the lexicographic comparison rule, `-nodiv` the run without dividends and `-K<k>` another
cardinality cap.

## Data

The instance files in `data/instances/` are the data of the paper. They are derived from the
public trading and corporate-action records of the Ho Chi Minh (HOSE) and Hanoi (HNX) stock
exchanges. Each instance uses the index constituents in force on 31 December of its year (VN30
from HOSE's July review, HNX30 from HNX's October review), listed in `data/universes.json`. The
monthly planning price of a stock is the month's highest daily closing price, the volume cap is
the month's total matched volume, and a cash dividend is assigned to the month of its ex-date.
The moment files hold the centred daily log returns `C` of the three years before the evaluation
year and their covariance `M2`; the third and fourth co-moments follow exactly from `C`
(`plan.comoments`), and the model evaluates them as sample moments of the portfolio return
series. The scripts that downloaded and assembled the daily records are not part of this
release.

## Reproducing the results

Python 3.14 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
python3 -m pytest -q tests                        # unit tests
python3 run_all.py main lex extra elite --workers 8
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
