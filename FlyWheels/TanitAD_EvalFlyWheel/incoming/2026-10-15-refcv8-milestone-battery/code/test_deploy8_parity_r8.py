"""Pure-helper tests for `deploy8_forward_r8.py` and `prior_parity_r8.py` (CPU, no banked data needed except where marked).

Every expectation is a literal or an analytic closed form; each guard has a mutation that must go RED.
    PYTHONPATH='C:/lgs/Lfb47a58376/stack;<PKG>/code' python -m pytest -q test_deploy8_parity_r8.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy8_forward_r8 as D8  # noqa: E402
import prior_parity_r8 as PP  # noqa: E402


# ------------------------------------------------------------------------------------------------ deploy8 helpers
def test_exact_equal_is_exact_and_nan_safe():
    assert D8.exact_equal({"a": [1.0, 2.0], "b": {"c": float("nan")}}, {"a": [1.0, 2.0], "b": {"c": float("nan")}}) == []
    assert D8.exact_equal({"a": 0.1}, {"a": 0.1 + 1e-18}) == []                    # 1e-18 is below half an ulp of 0.1: the same double
    assert len(D8.exact_equal({"a": 0.1}, {"a": float(np.nextafter(0.1, 1.0))})) == 1   # one ulp apart is NOT equal
    assert len(D8.exact_equal({"a": 1}, {"a": 1, "b": 2})) == 1                     # a key on one side only


def _arm(v):
    return {"mean_ade_m": v, "minus_os": {"ade_m": {"delta": v}}}


def _res(arm, v):
    return {"seeds": {str(s): {"surfaces": {"S2": {"sets": {st: {"arms": {arm: _arm(v)}} for st in D8.SETS}}}} for s in D8.SEEDS}}


def test_s2_identity_passes_on_equal_rows_and_fails_on_a_perturbed_one_and_on_absence():
    ref = _res("RETIME_TRAINFIT", 0.25)
    assert D8.s2_identity(_res("RETIME_DEPLOY_PERSURFACE", 0.25), ref)["pass"] is True
    mut = D8.s2_identity(_res("RETIME_DEPLOY_PERSURFACE", 0.25 + 1e-12), ref)          # MUTATION: a knot perturbation moves a row by a hair
    assert mut["pass"] is False and mut["n_mismatches"] > 0
    assert D8.s2_identity(None, ref)["pass"] is False and D8.s2_identity(_res("RETIME_DEPLOY_PERSURFACE", 0.25), None)["status"] == "ABSENT"
    assert D8.s2_identity(_res("OTHER", 0.25), ref)["pass"] is False                   # a missing arm is not a pass


def test_aggregate_controls_requires_every_named_control_true():
    ok = {k: True for k in ("rt_forward", "const_g1_exactly_os", "known_values_8slot", "persurface_identity_vs_forward_trainfit", "row_alignment", "table_matches_frozen_train")}
    assert D8.aggregate_controls(ok) is True
    for k in ok:
        bad = dict(ok)
        bad[k] = False
        assert D8.aggregate_controls(bad) is False
        gone = dict(ok)
        del gone[k]
        assert D8.aggregate_controls(gone) is False                                   # an absent control is a fail


def test_mutate_table_moves_exactly_one_knot_and_leaves_the_input_alone():
    tab = {"knots_s": [0.5, 1.0, 1.5], "gains": [0.9, 0.6, 0.5]}
    m = D8.mutate_table(tab, 1, 0.01)
    assert m["gains"].tolist() == [0.9, 0.61, 0.5] and tab["gains"] == [0.9, 0.6, 0.5]


def test_rows_gain_survives_boolean_selection_as_a_row_index():
    W = D8.rows_gain(6, 4, D8._RowsD8)
    assert isinstance(W, D8._Rows) and W.ndim == 2
    sel = np.array([True, False, True, True, False, False])
    Ws = W[sel]
    assert isinstance(Ws, D8._RowsD8) and Ws.KIND == "D8"                             # the marker survives `W[sel]` (RT.forward does exactly this)
    assert np.asarray(Ws)[:, 0].tolist() == [0, 2, 3]                                   # and carries WHICH rows were selected
    assert not isinstance(np.ones(4), D8._Rows)                                         # an ordinary gain vector is never mistaken for it


def test_diag8_on_path_literal():
    plan = np.array([[[5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0], [50.0, 0.0], [60.0, 0.0]]], np.float32)
    prior = (plan * 0.5).astype(np.float32)                                              # a prior at half speed
    import a0k0_headroom_retime_r8 as RT
    s = RT.retime_apply(plan, prior, np.zeros(8), cast=False)                            # g = 0: points at the prior's arc length on the plan path
    assert s[0, :, 0].tolist() == [2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0]
    d = D8.diag8(plan, prior, s, s.astype(np.float32), np.zeros(8), np.float32)
    assert d["max_dist_to_path_f64_m"] == 0.0 and d["frac_extrapolated_points"] == 0.0 and d["max_abs_g_minus_1"] == 1.0
    off = s.copy()
    off[0, 3, 1] = 0.5                                                                  # MUTATION: one point 0.5 m off the path
    assert D8.diag8(plan, prior, off, off.astype(np.float32), np.zeros(8), np.float32)["max_dist_to_path_f64_m"] == pytest.approx(0.5)


def test_the_stage_is_loadable_and_the_deploy_table_verifies():
    assert D8.RS_FILE.is_file()
    tab = D8.RS.load_deploy_table()
    assert tab["knots_s"].tolist() == [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0] == D8.T8.tolist()


# ------------------------------------------------------------------------------------------------ prior parity
def test_independent_prior_known_values():
    W, v = PP.synth_windows()
    P = PP.independent_prior(W)
    for i, name in enumerate(("straight_10", "standing", "accel_+1")):
        assert np.abs(P[i, :, 0] - np.array(PP.KNOWN_X[name])).max() < PP.TOL_M
        assert np.abs(P[i, :, 1]).max() == 0.0                                           # straight: no lateral component at all


def test_known_value_controls_pass_and_their_mutations_fail():
    pytest.importorskip("torch")
    for fn in (PP.model_path_prior, lambda W, v: PP.rebuilt_prior(W), lambda W, v: PP.independent_prior(W)):
        assert PP.known_value_check(fn)["PASS"] is True
    assert PP.known_value_check(PP.model_path_prior, tick=0.2)["PASS"] is False        # wrong tick: x doubles
    assert PP.known_value_check(lambda W, v: PP.rebuilt_prior(W, [5, 10, 15, 20, 30, 40, 50, 55]))["PASS"] is False
    assert PP.known_value_check(lambda W, v: PP.independent_prior(W, tick=0.2))["PASS"] is False


def test_model_path_refuses_a_window_that_does_not_end_at_v0():
    pytest.importorskip("torch")
    W, v = PP.synth_windows()
    with pytest.raises(ValueError, match="window"):
        PP.model_path_prior(W, v + 0.5)                                                  # the model's own V0_WINDOW_TOL guard, reached through its method


def test_model_path_equals_rebuilt_on_a_curved_and_decelerating_window():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n = 32
    yaw = np.cumsum(rng.normal(0.0, 0.02, (n, PP.W_OBS)), axis=1).astype(np.float32)
    v = np.clip(rng.uniform(0.0, 25.0, (n, 1)) + np.cumsum(rng.normal(0.0, 0.1, (n, PP.W_OBS)), axis=1), 0.0, None).astype(np.float32)
    W = np.zeros((n, PP.W_OBS, 4), np.float32)
    W[..., 2], W[..., 3] = yaw, v
    m, r = PP.model_path_prior(W, W[:, -1, 3].copy()), PP.rebuilt_prior(W)
    assert np.abs(m.astype(np.float64) - r.astype(np.float64)).max() <= PP.TOL_M
    assert np.abs(m.astype(np.float64) - PP.independent_prior(W)).max() <= 5e-3         # a third, independent derivation (float64 loops) agrees closely


def test_source_trace_quotes_are_verified_against_the_launch_stack():
    if not PP.STACK.exists():
        pytest.skip("launch stack not present")
    tr = PP.verify_trace(PP.STACK)
    assert tr and all(t["verified_at_line"] for t in tr)
    # MUTATION: a quote moved to the wrong line is reported unverified
    f, ln, q, why = PP.TRACE[0]
    PP.TRACE[0] = (f, ln + 1, q, why)
    try:
        assert not PP.verify_trace(PP.STACK)[0]["verified_at_line"]
    finally:
        PP.TRACE[0] = (f, ln, q, why)
