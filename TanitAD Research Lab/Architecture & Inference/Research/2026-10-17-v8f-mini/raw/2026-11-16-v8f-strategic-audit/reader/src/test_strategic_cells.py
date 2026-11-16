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


# ------------------------------------------------------------------------------------------ strategic_text: the table join, and its deliberate-regression arms
import pytest
import strategic_text as ST


def synthetic_table(eids, side_name="TURN_L", d=30.0, flag=0, n_steps=60):
    """A navtok-shaped eval table: every step of every clip carries one class, one distance and one flag."""
    vocab = ["UNPARSED", "CONTINUE", "TURN_L", "TURN_R", "TURN_SLIGHT_L", "UTURN"]
    eids = list(dict.fromkeys(eids))
    tot = len(eids) * n_steps
    tab = dict(eval_clip=np.array(eids), eval_off=np.arange(len(eids)) * n_steps, eval_n=np.full(len(eids), n_steps),
               eval_cls_id=np.full(tot, vocab.index(side_name), np.int16), eval_dist_m=np.full(tot, d, np.float32), eval_flag=np.full(tot, flag, np.uint8), vocab=np.array(vocab))
    tab["clip_index"] = {e: i for i, e in enumerate(eids)}
    tab["side"], tab["theta"] = S.side_table(vocab)
    return tab


def test_proto_integrator_equals_the_production_integrator():
    r = np.random.default_rng(1)
    plan = np.stack([r.normal(0, 3, (50, 30)), r.normal(0, 0.4, (50, 30))], -1)
    v0 = r.uniform(0, 25, 50)
    psi0, xy0 = S._proto_integrate(v0, plan)
    psi1, xy1 = S.integrate(v0, plan)
    assert np.array_equal(psi0, psi1) and np.array_equal(xy0, xy1)


def test_attach_row_with_flag_is_nan_not_zero():
    e = np.array(["a", "a", "b"])
    for fl in (1, 2, 3):
        a = ST.attach(synthetic_table(e, flag=fl), e, np.array([5, 6, 7]), offset=0)
        assert not a["text_ok"].any() and np.all(np.isnan(a["d_text"])) and np.all(a["side"] == 0)
        psi, xy = S.integrate(np.full(3, 10.0), _plan(0.3, n=3))
        assert all(np.all(np.isnan(v)) for v in S.cells(psi, xy, a["side"], a["theta_deg"], a["d_text"], a["text_ok"]).values())   # NaN in every cell
    a = ST.attach(synthetic_table(e, flag=0), e, np.array([5, 6, 7]), offset=0)
    assert a["text_ok"].all() and np.all(a["d_text"] == 30.0) and np.all(a["side"] == 1)                                            # control: the same call on flag 0 reads the text


def test_attach_unmapped_clip_and_step_out_of_range():
    tab = synthetic_table(np.array(["a", "b"]), n_steps=10)
    a = ST.attach(tab, np.array(["a", "zzz", "b", "a"]), np.array([3, 3, 9, 10]), offset=0)
    assert a["text_ok"].tolist() == [True, False, True, False] and a["row"].tolist() == [3, -1, 19, -1]


def test_attach_applies_the_offset_to_the_step():
    tab = synthetic_table(np.array(["a"]), n_steps=10)
    tab["eval_cls_id"][5] = tab["vocab"].tolist().index("TURN_R")
    assert ST.attach(tab, np.array(["a"]), np.array([2]), offset=3)["side"][0] == -1
    assert ST.attach(tab, np.array(["a"]), np.array([2]), offset=0)["side"][0] == 1


def test_default_offset_is_the_verified_one():
    assert ST.NOW_OFFSET == 0                                   # STRATEGIC_TEXT_OFFSET_CHECK.json: offset 0 reads 84.1 % within 0.05 m/s, offset 3 reads 24.4 %


def test_load_table_refuses_a_file_with_the_wrong_sha(tmp_path):
    p = tmp_path / "t.npz"
    np.savez(p, x=np.arange(3))
    with pytest.raises(ValueError):
        ST.load_table(str(p))


def test_verify_offset_finds_a_planted_offset(tmp_path):
    """Deliberate regression of the check itself: windows whose v0 is the speed at table step now + 2 must make verify_offset answer 2, not 0."""
    import torch
    n_steps, dt_step = 40, 0.2
    now_s = 0.3 + dt_step * np.arange(n_steps)
    speed = lambda t: 5.0 + 0.8 * t + 0.3 * np.sin(1.7 * t)                       # time-varying, so the offset is identifiable
    k = np.arange(0, 120)
    trk = np.full((120, 4), np.nan)
    trk[:, 3] = speed(k * ST.RAW_DT_S + 0.0)
    np.savez(tmp_path / "join.npz", now_s=now_s)
    torch.save({"tracks": {"a": trk}}, tmp_path / "trk.pt")
    tab = synthetic_table(np.array(["a"]), n_steps=n_steps)
    nows = np.arange(3, 30, 3)
    v0 = speed(now_s[nows + 2])                                                     # the plant: NOW belongs to table step now + 2
    res = ST.verify_offset(tab, str(tmp_path / "join.npz"), str(tmp_path / "trk.pt"), np.array(["a"] * len(nows)), nows, v0, offsets=range(0, 5), shift_s=0.0)
    assert res["best_offset"] == 2 and res["by_offset"][2]["frac_within_0p05"] == 1.0 and res["by_offset"][0]["frac_within_0p05"] < 0.5
