"""Tests for strategic_cells.py, including the deliberate-regression arms (each must turn the check RED)."""
import numpy as np
import strategic_cells as S


def _plan(yaw_rate, n=4, K=30, a=0.0):
    p = np.zeros((n, K, 2))
    p[..., 0] = a
    p[..., 1] = yaw_rate
    return p


def _run(plan, v0=10.0):
    return S.integrate(np.full(plan.shape[0], v0), plan)


def _w(n=4, side=1, d=30.0):
    return dict(side=np.full(n, side, np.int8), theta_deg=np.full(n, 25.0), d_text=np.full(n, d), text_ok=np.ones(n, bool))


def test_hold_reads_exactly_zero_on_turn_windows():           # floor control: must read EXACTLY 0.0
    psi, xy = _run(_plan(0.0))
    c = S.cells(psi, xy, **_w())
    assert np.all(c["s_adh"] == 0.0) and np.all(c["s_adh_signed_deg"] == 0.0)


def test_left_turn_plan_adheres_to_left_text():
    psi, xy = _run(_plan(0.3))                               # 0.3 rad/s over 6 s, v 10 m/s: ~100 deg
    c = S.cells(psi, xy, **_w(side=1))
    assert np.all(c["s_adh"] == 1.0) and np.all(c["s_adh_signed_deg"] > 25)


def test_regression_arm_mirrored_text_flips_the_cell():       # deliberate regression: wrong side must score 0 and negative signed turn
    psi, xy = _run(_plan(0.3))
    c = S.cells(psi, xy, **_w(side=-1))
    assert np.all(c["s_adh"] == 0.0) and np.all(c["s_adh_signed_deg"] < 0)


def test_unreachable_manoeuvre_is_nan_not_zero():
    psi, xy = _run(_plan(0.3), v0=3.0)                       # ~18 m of arc in 6 s
    c = S.cells(psi, xy, **_w(d=120.0))
    assert np.all(np.isnan(c["s_adh"]))


def test_no_text_row_is_nan_in_every_cell():
    psi, xy = _run(_plan(0.3))
    w = _w()
    w["text_ok"][:] = False
    c = S.cells(psi, xy, **w)
    assert all(np.all(np.isnan(v)) for v in c.values())


def test_false_turn_cell():
    psi, xy = _run(_plan(0.3))
    w = _w(side=0, d=np.nan)
    assert np.all(S.cells(psi, xy, **w)["s_false"] == 1.0)
    psi, xy = _run(_plan(0.0))
    assert np.all(S.cells(psi, xy, **w)["s_false"] == 0.0)


def test_uturn_and_continue_are_not_turns():
    for n in ("UTURN", "CONTINUE_L", "KEEP_R", "RB_TAKE_EXIT"):
        assert S.parse_side(n)[0] == 0
    assert S.parse_side("TURN_SLIGHT_R")[0] == -1 and S.parse_side("TURN_SHARP_L")[0] == 1


def test_ckpt_cell_masks_by_distance_and_validity():
    ck = np.radians(np.array([[60.0, 60.0, 60.0]] * 2))
    ok = np.array([[True, True, True], [True, True, False]])
    o = S.ckpt_cell(ck, (30.0, 50.0, 80.0), np.ones(2, np.int8), np.full(2, 25.0), np.array([20.0, 20.0]), np.ones(2, bool), ok)
    assert np.isnan(o[0, 0]) and o[0, 1] == 1.0 and o[0, 2] == 1.0 and np.isnan(o[1, 2])   # 20 + 20 = 40 m > 30 m arc


def test_mutation_threshold_zero_would_pass_hold():            # the check is not vacuous: with theta 0 the HOLD plan WOULD "adhere"
    psi, xy = _run(_plan(0.0))
    w = _w()
    w["theta_deg"][:] = 0.0
    assert np.all(S.cells(psi, xy, **w)["s_adh"] == 1.0)
