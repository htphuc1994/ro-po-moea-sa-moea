"""plan.objectives evaluates the moments as sample moments of the daily portfolio return
series C w. This must equal the tensor contractions with M2, M3, M4 built from the same C,
and the vectorized month bookkeeping must equal the original per-month loop."""
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import plan  # noqa: E402

INST = os.path.join(ROOT, "data", "instances")


def reference_objectives(q, y1, deposit, terminal_cash, M2, T3, T4):
    """The per-month loop with tensor contractions, as used before the fast path."""
    risks = np.zeros((plan.tau - 1, 4))
    for m in range(plan.tau - 1):
        value = plan.LASTPRICE[:, m] * q[m]
        recv = (1 - plan.epsilon) * float(plan.PRICES[:, m] @ y1[m])
        if m >= 1:
            recv += float(plan.DIVS[:, m] @ q[m - 1])
        invested = float(value.sum())
        if invested <= 0:
            continue
        w = value / (deposit[m] + recv + invested)
        S = np.flatnonzero(w > 0)
        ws = w[S]
        risks[m, 0] = ws @ M2[np.ix_(S, S)] @ ws
        risks[m, 1] = np.einsum("i,j,k,ijk->", ws, ws, ws, T3[np.ix_(S, S, S)])
        risks[m, 2] = np.einsum("i,j,k,l,ijkl->", ws, ws, ws, ws, T4[np.ix_(S, S, S, S)])
        if S.size > 1:
            p = value[S] / invested
            risks[m, 3] = float(-(p * np.log(p)).sum())
    return np.r_[risks.mean(axis=0), terminal_cash]


@pytest.fixture(scope="module", params=["vn30_2023", "hnx30_2023"])
def loaded(request):
    npz = os.path.join(INST, f"{request.param}_moments.npz")
    if not os.path.exists(npz):
        pytest.skip("instance files not found")
    mom = dict(np.load(npz))
    plan.load_instance(os.path.join(INST, f"{request.param}.json"), mom)
    n = len(plan.SYMBOLS)
    if "M3" not in mom:                       # compact file: the tensors follow from C exactly
        mom["M2"], mom["M3"], mom["M4"], _ = plan.comoments(mom["C"])
    return mom["M2"], mom["M3"].reshape(n, n, n), mom["M4"].reshape(n, n, n, n)


def test_stored_tensors_are_the_comoments_of_C(loaded):
    M2, T3, T4 = loaded
    C = plan.CRET
    T = C.shape[0]
    assert plan.T4 is None, "the fast path must not keep the n^4 tensor in memory"
    assert np.allclose(M2, C.T @ C / T, rtol=1e-12, atol=0)
    assert np.allclose(T3, np.einsum("ti,tj,tk->ijk", C, C, C) / T, rtol=1e-10, atol=1e-18)
    sub = slice(0, 6)  # full n^4 einsum is slow; a 6-stock block is enough to pin the layout
    Cs = C[:, sub]
    assert np.allclose(T4[sub, sub, sub, sub], np.einsum("ti,tj,tk,tl->ijkl", Cs, Cs, Cs, Cs) / T,
                       rtol=1e-10, atol=1e-20)


def test_return_series_path_equals_tensor_loop(loaded):
    M2, T3, T4 = loaded
    random.seed(23)
    pop = [plan.sample() for _ in range(60)]
    plans = pop + [plan.sample(random.choice(pop), random.choice(pop)) for _ in range(60)]
    plans += [plan.sample(anchor=a) for a in pop[:20]]
    n = len(plan.SYMBOLS)
    z = np.zeros((plan.tau, n), dtype=np.int64)
    plans.append(plan.simulate(z, z))                       # nothing invested in any month
    worst = 0.0
    for ind in plans:
        ref = reference_objectives(ind.q, ind.y1, ind.deposit, ind.theta[-1], M2, T3, T4)
        assert np.allclose(ind.f, ref, rtol=1e-9, atol=1e-300), (ind.f, ref)
        nz = ref[:4] != 0
        if nz.any():
            worst = max(worst, float(np.max(np.abs(ind.f[:4][nz] - ref[:4][nz]) / np.abs(ref[:4][nz]))))
    assert worst < 1e-9
