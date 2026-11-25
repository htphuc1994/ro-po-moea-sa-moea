"""Starting acceptance probability chi_0 of the SA phase (sensitivity block of run_all.py)."""
import math
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import plan  # noqa: E402
import solvers  # noqa: E402
from run_all import label  # noqa: E402

INST = os.path.join(ROOT, "data", "instances")


@pytest.fixture(scope="module")
def instance():
    npz = os.path.join(INST, "vn30_2023_moments.npz")
    if not os.path.exists(npz):
        pytest.skip("instance files not found")
    plan.load_instance(os.path.join(INST, "vn30_2023.json"), npz)


def test_temperature_scales_with_one_over_log_chi0(instance):
    random.seed(7)
    pop = [plan.sample() for _ in range(40)]
    lo, hi = plan.population_bounds(pop)
    temps = {}
    for chi0 in (0.2, 0.5, 0.8):
        random.seed(11)                      # the same calibration probes for every chi0
        sa = plan.SAInjector(pop[0], start_accept=chi0)
        sa.calibrate(lo, hi)
        temps[chi0] = sa.T0
    # T0 = -mean increase / ln(chi0), so an average worsening move is accepted with probability chi0
    for chi0 in (0.2, 0.8):
        assert temps[chi0] / temps[0.5] == pytest.approx(math.log(0.5) / math.log(chi0))
    assert temps[0.2] < temps[0.5] < temps[0.8]


def test_default_chi0_reproduces_the_main_runs(instance):
    a = solvers.run("spea2", 1234, budget=1500, sa_mode="sa", snapshot_every=None)
    b = solvers.run("spea2", 1234, budget=1500, sa_mode="sa", snapshot_every=None, sa_chi0=0.5)
    assert a["archive"] == b["archive"] and a["best"] == b["best"] and a["sa"]["chi0"] == 0.5


def test_other_chi0_changes_the_search_not_the_budget(instance):
    a = solvers.run("spea2", 1234, budget=1500, sa_mode="sa", snapshot_every=None)
    c = solvers.run("spea2", 1234, budget=1500, sa_mode="sa", snapshot_every=None, sa_chi0=0.8)
    assert a["evals"] == c["evals"] and a["generations"] == c["generations"]
    assert c["sa"]["T0"] > a["sa"]["T0"]


def test_labels():
    assert label("spea2", "sa") == "spea2+sa"
    assert label("spea2", "sa", chi0=0.2) == "spea2+sa-chi0.2"
    assert label("moead", "sa", chi0=0.8) == "moead+sa-chi0.8"
