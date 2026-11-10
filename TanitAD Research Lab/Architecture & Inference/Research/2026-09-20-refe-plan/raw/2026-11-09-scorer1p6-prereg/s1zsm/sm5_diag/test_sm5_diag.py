"""tests for sm5_diag.py on a small synthetic fixture: controls C2 (empty coalition = own share), C3 (efficiency), C4 (mutation: P6 logits in an arm slot read exactly 0.0),
plus a RED check that a different arm does NOT read zero, and that the metric swap reproduces the reader's PDMS Shapley."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sm5_diag as D  # noqa: E402

R = D.R
N, K = 40, 64


@pytest.fixture(scope="module")
def fx():
    rng = np.random.default_rng(7)
    logits_p6 = rng.normal(0, 1.5, (N, K, 6)).astype(np.float32)
    logits_arm = (logits_p6 + rng.normal(0, 1.0, (N, K, 6))).astype(np.float32)
    sub = rng.uniform(0, 1, (N, K, 6))
    valid = rng.uniform(0, 1, (N, K)) > 0.1
    # the pick rule (pick_logits) has no valid mask; the Shapley code masks invalid proposals. As in the real data, invalid proposals carry very low logits so the two agree.
    logits_p6[~valid] = -30.0
    logits_arm[~valid] = -30.0
    ratio = rng.uniform(0.5, 1.2, (N, K))
    pdms = rng.uniform(0, 1, (N, K))
    cru = rng.uniform(0, 1, N) > 0.3
    log = np.array([f"log{i // 4}" for i in range(N)])
    return dict(p6=logits_p6, arm=logits_arm, sub=sub, valid=valid, ratio=ratio, pdms=pdms, cru=cru, log=log)


@pytest.fixture(scope="module")
def res(fx):
    arms = {"P6": fx["p6"], "ARM": fx["arm"], "MUT": fx["p6"].copy()}
    return D.diag(arms, fx["sub"], fx["valid"], fx["ratio"], fx["cru"], fx["log"], fx["pdms"])


def test_c2_empty_coalition_is_own_share(res):
    for a in res["arms"].values():
        assert a["C2_empty_vs_own_abs_err_x100"] < 1e-9


def test_c3_efficiency(res):
    for a in res["arms"].values():
        assert a["efficiency_abs_err"] < 1e-9


def test_c4_mutation_reads_exactly_zero(res):
    assert D.all_zero(res["arm_minus_ref"]["MUT"])


def test_red_real_arm_not_zero(res):
    assert not D.all_zero(res["arm_minus_ref"]["ARM"])


def test_shapley_is_the_readers_when_metric_is_pdms(fx):
    """the per-token route sums to the reader's whole-set Shapley (linearity), for the slow metric"""
    slow = (fx["ratio"] < D.SLOW_RATIO).astype(np.float64)
    whole = R.shapley(fx["arm"], fx["sub"], slow, fx["valid"])
    toks = np.arange(N)
    v, sh, _ = D.token_values(fx["arm"], fx["sub"], fx["valid"], slow, toks)
    assert abs(v["v_empty"].mean() - whole["v_empty"]) < 1e-9
    for h in D.HEADS:
        assert abs(sh[h].mean() - whole["shapley_points"][h]) < 1e-9


def test_same_draws_for_every_arm(fx):
    a = R.BootSet(fx["log"])
    b = R.BootSet(fx["log"])
    x = np.arange(N, dtype=float)
    assert np.array_equal(a.stat(x)[1], b.stat(x)[1])
