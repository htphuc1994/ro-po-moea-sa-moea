"""Elite-anchored SA (eSA / eRI) and the SA diagnostics logged by solvers.run."""
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import plan  # noqa: E402
import solvers  # noqa: E402

INST = os.path.join(ROOT, "data", "instances")


@pytest.fixture(scope="module")
def instance():
    npz = os.path.join(INST, "vn30_2023_moments.npz")
    if not os.path.exists(npz):
        pytest.skip("instance files not found")
    plan.load_instance(os.path.join(INST, "vn30_2023.json"), npz)


def test_elite_is_the_minimum_energy_plan(instance):
    random.seed(3)
    pop = [plan.sample() for _ in range(60)]
    lo, hi = plan.population_bounds(pop)
    e = [plan.energy(p.f, lo, hi) for p in pop]
    assert plan.elite(pop, lo, hi) is pop[int(np.argmin(e))]
    twin = pop + [pop[int(np.argmin(e))]]            # ties go to the first occurrence
    assert plan.elite(twin, lo, hi) is pop[int(np.argmin(e))]


@pytest.mark.parametrize("mode", ["sa", "ri", "sae", "rie"])
def test_runs_are_reproducible_and_diagnostics_consistent(instance, mode):
    a = solvers.run("nsga2", 1234, budget=1500, sa_mode=mode, snapshot_every=None)
    b = solvers.run("nsga2", 1234, budget=1500, sa_mode=mode, snapshot_every=None)
    assert a["archive"] == b["archive"] and a["best"] == b["best"]
    diag = a["sa"]["diag"]
    assert len(diag) == a["generations"] == len(a["sa"]["log"])
    assert 0 <= a["sa"]["in_final"] <= len(a["archive"])
    assert all(d in (0, 1) for row in diag for d in row)


@pytest.mark.parametrize("mode", ["sa", "ri", "sae", "rie"])
def test_acceptance_counters(instance, mode):
    """Per generation [n_up, n_up_acc, n_met, n_met_acc]: an energy-raising move that skips the Metropolis
    test is accepted (by constraint-domination, or as every move in the random-injection control)."""
    a = solvers.run("moead", 1234, budget=1500, sa_mode=mode, snapshot_every=None)
    for up, up_acc, met, met_acc in a["sa"]["log"]:
        assert 0 <= met_acc <= met <= up <= plan.SA_STEPS and 0 <= up_acc <= up
        assert up_acc - met_acc == up - met
        if mode in ("ri", "rie"):
            assert met == 0 and up_acc == up


def test_elite_mode_changes_the_search(instance):
    sa = solvers.run("nsga2", 1234, budget=1500, sa_mode="sa", snapshot_every=None)
    sae = solvers.run("nsga2", 1234, budget=1500, sa_mode="sae", snapshot_every=None)
    assert sa["evals"] == sae["evals"] and sa["generations"] == sae["generations"]
    assert sa["archive"] != sae["archive"]
