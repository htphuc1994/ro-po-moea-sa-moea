# First-round implementation (kept for traceability)

This folder holds the code, results and notes behind the first-round version of the manuscript.
It is not used by the current pipeline.

- `plan_r1.py`: the first-round model and constructive sampler.
- `spea2.py`, `nsga3.py`, `moead.py`, `run_legacy.py`, `equal_weight.py`: first-round solvers and runners
  (they import the first-round `plan` module and do not run against the current one).
- `*_results_*.json`, `*.log`, `*.txt`: first-round outputs.
- `analyze_with_baselines.py`, `resultAnalyzers*.py`, `graphs/`: first-round analysis and figures.
- `EXPERIMENTS.md`: first-round experiment notes (its numbers are the first-round results).

The second-round revision corrected the moment estimator (daily log returns instead of
monthly price levels), the portfolio weights (capital weights including cash), the sale
rule of the sampler, the SA temperature scale and the comparison rule (one
constraint-domination rule for all solvers). See the top-level README.
