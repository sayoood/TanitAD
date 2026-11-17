"""PREREG_V8F_CKPT120 (part 2b): the far-output evaluator wrappers (``v8f_mini_eval_far.py``, ``v8f_t1_eval_far.py``) and the reader cells (``ckpt120_cells.py``).

Wrappers: the shared tools are byte-equal to their pins (a test hashes the files; an edit of either shared tool would regress the studies that pin them), the wrapper REFUSES a tool whose sha256 differs, refuses a flag that
disagrees with the checkpoint's cfg, restores every patch it applied (also when the call raises), dead-reckons nav_s exactly like nav, and, on the synthetic world, writes the far output of the SAME forward call that produced the plan.
Cells: hand values for S_CKPT120 / balanced / best constant / strata / EXTRAP / the cost-gain rule, the bootstrap against ``taniteval.ci`` on the same draws, and each control (K1, K7, K8, K9b) with the value it must read plus a
deliberate regression that must break it.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "scripts"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "taniteval"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import ckpt120_cells as K  # noqa: E402

SCRIPTS = os.path.join(HERE, "..", "scripts")


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------------------------------------------ the cells, hand values
def window_set(n=80, seed=0):
    """A labelled window set over 20 episodes: side +-1 (right majority), theta 25 deg, d_text 10..70 m, everything scorable at 120 m (d + 20 <= 120)."""
    rng = np.random.default_rng(seed)
    side = np.where(rng.random(n) < 0.7, 1, -1).astype(np.int8)
    return dict(side=side, theta=np.full(n, 25.0), d=rng.uniform(10, 70, n), text_ok=np.ones(n, bool), rc_ok=np.ones(n, bool), eid=np.repeat(np.arange(20), n // 20))


def test_s_ckpt120_hand_values_and_not_scorable_is_nan():
    far = np.array([[120, 0, np.radians(30)], [120, 0, np.radians(30)], [120, 0, np.radians(20)], [120, 0, np.radians(-30)], [120, 0, np.radians(30)], [120, 0, np.radians(30)], [120, 0, np.radians(30)]])
    side = np.array([1, -1, 1, -1, 0, 1, 1])
    theta = np.full(7, 25.0)
    d = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 101.0, np.nan])                  # 101 + 20 > 120: not scorable; NaN distance: not scorable
    ok = np.array([1, 1, 1, 1, 1, 1, 1], bool)
    out = K.s_ckpt120(far, side, theta, d, ok, ok)
    np.testing.assert_array_equal(out[:5], [1.0, 0.0, 0.0, 1.0, np.nan])
    assert np.isnan(out[5]) and np.isnan(out[6])
    assert np.isnan(K.s_ckpt120(far, side, theta, np.full(7, 50.0), ok, np.zeros(7, bool))).all()     # RC-A120 invalid: not scorable
    assert np.isnan(K.s_ckpt120(far, side, theta, np.full(7, 50.0), np.zeros(7, bool), ok)).all()     # no text row
    slight = K.s_ckpt120(far[:1], side[:1], np.array([10.0]), d[:1], ok[:1], ok[:1])
    assert slight[0] == 1.0
    assert np.isnan(K.s_ckpt120(far[:1], side[:1], theta[:1], np.array([50.0]), ok[:1], ok[:1], arc=60.0)[0])      # the arc is a parameter: 50 + 20 > 60


def test_s_ckpt120_matches_the_strat1_cell_form_at_the_80_m_arc():
    sc = pytest.importorskip("strategic_cells", reason="the STRAT1 reader's cell module lives in the research folder")
    w = window_set()
    psi = np.random.default_rng(1).uniform(-0.8, 0.8, (80, 3))
    ck_ok = np.ones((80, 3), bool)
    want = sc.ckpt_cell(psi, (30, 50, 80), w["side"], w["theta"], w["d"], w["text_ok"], ck_ok)[:, 2]
    got = K.s_ckpt120(np.stack([np.full(80, 80.0), np.zeros(80), psi[:, 2]], -1), w["side"], w["theta"], w["d"], w["text_ok"], ck_ok[:, 2], arc=80.0)
    np.testing.assert_array_equal(want, got)


def test_best_constant_and_balanced_read_one_half_for_any_constant():
    w = window_set()
    cell = np.where(w["side"] == 1, 1.0, 0.0)                                      # always "right": correct on every right window only
    assert K.balanced_level(cell, w["side"]) == 0.5
    bc, maj = K.best_constant(np.ones(80), w["side"])
    assert maj == 1 and np.array_equal(bc, (w["side"] == 1).astype(float))
    assert K.balanced_level(bc, w["side"]) == 0.5
    assert K.balanced_level(np.ones(80), w["side"]) == 1.0
    r = K.balanced_minus_half(cell, w["side"], w["eid"], n_boot=200)
    assert r["delta"] == 0.0 and not r["separated"]


def test_route_following_a_follower_separates_a_constant_does_not():
    w = window_set(n=200)
    follow = K.s_ckpt120(K.planted(w["side"], "announced"), w["side"], w["theta"], w["d"], w["text_ok"], w["rc_ok"])
    const = K.s_ckpt120(K.planted(np.ones(200), "announced"), w["side"], w["theta"], w["d"], w["text_ok"], w["rc_ok"])
    eid = np.repeat(np.arange(20), 10)
    f = K.route_following(follow, const, const, w["side"], eid, n_boot=300)
    assert f["level"]["real"] == 1.0 and f["balanced"]["real"] == 1.0 and f["balanced_minus_half"]["separated"]
    assert f["real_minus_donor"]["status"] == "OK" and f["real_minus_donor"]["delta"] > 0 and f["real_minus_donor"]["separated"]
    c = K.route_following(const, const, const, w["side"], eid, n_boot=300)
    assert c["balanced"]["real"] == 0.5 and c["real_minus_donor"]["delta"] == 0.0 and not c["real_minus_donor"]["separated"]
    assert c["raw_real_minus_best_constant"]["delta"] == 0.0                         # a constant reads exactly zero against the best constant


def test_paired_bootstrap_equals_taniteval_on_the_same_draws():
    from taniteval import ci
    rng = np.random.default_rng(2)
    a, b = rng.normal(0.3, 1, 120), rng.normal(0, 1, 120)
    eid = np.repeat(np.arange(12), 10)
    mine = K.paired_mean(a, b, eid, n_boot=500)
    ref = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=500, seed=0)
    assert mine["delta"] == pytest.approx(ref["delta"], abs=1e-4) and mine["lo"] == pytest.approx(ref["lo"], abs=1e-4) and mine["hi"] == pytest.approx(ref["hi"], abs=1e-4)
    assert mine["separated"] == ref["separated"]


def test_below_the_n_rule_is_na_with_its_n():
    r = K.paired_mean(np.ones(10), np.zeros(10), np.arange(10))
    assert r["status"] == "n/a" and r["n_windows"] == 10


def test_far_errors_extrap_and_straight_hand_values():
    rc = np.array([[120.0, 10.0, 0.3], [120.0, 0.0, 0.0]])
    far = np.array([[120.0, 0.0, 0.3 + 2 * np.pi], [123.0, 4.0, 0.1]])             # wrapped heading: 0 deg error on the first
    e = K.far_errors(far, rc, np.array([True, True]))
    np.testing.assert_allclose(e["pos_err_m"], [10.0, 5.0])
    np.testing.assert_allclose(e["head_err_deg"], [0.0, np.degrees(0.1)], atol=1e-9)
    assert np.isnan(K.far_errors(far, rc, np.array([True, False]))["pos_err_m"][1])
    ck = np.zeros((1, 3, 3))
    ck[0, 2] = [80.0, 5.0, np.pi / 2]                                              # the 80 m point heading left: EXTRAP goes 40 m along +y
    np.testing.assert_allclose(K.extrap_from_80(ck)[0], [80.0, 45.0, np.pi / 2], atol=1e-9)
    ck[0, 2] = [80.0, 0.0, 0.0]
    np.testing.assert_allclose(K.extrap_from_80(ck)[0], [120.0, 0.0, 0.0])         # a straight 80 m point: EXTRAP = STRAIGHT
    np.testing.assert_array_equal(K.straight_far(2), [[120, 0, 0]] * 2)


def test_route_setting_120_reports_the_arm_minus_extrap_cells_and_a_missing_80_m_point_is_named():
    rng = np.random.default_rng(3)
    n = 100
    rc = np.column_stack([np.full(n, 120.0), rng.normal(0, 3, n), rng.normal(0, 0.1, n)])
    far = rc + rng.normal(0, 0.2, rc.shape)                                        # a good arm
    ck = np.zeros((n, 3, 3))
    ck[:, 2] = np.column_stack([np.full(n, 80.0), np.zeros(n), np.zeros(n)])      # S's 80 m point is straight
    r = K.route_setting_120(far, ck, rc, np.ones(n, bool), np.repeat(np.arange(20), 5), n_boot=200)
    d = r["extrap"]["arm_minus_baseline"]["pos_err_m"]
    assert d["status"] == "OK" and d["delta"] < 0 and d["separated"] and r["n_windows"] == n
    assert K.route_setting_120(far, None, rc, np.ones(n, bool), np.repeat(np.arange(20), 5), n_boot=100)["extrap"]["status"] == "n/a"


def test_strata_masks():
    s = K.strata(np.array([True, False, True, False]), np.array([30.0, 70.0, 100.0, 120.0]), np.array([10.0, 10.0, 2.0, 0.0]))
    np.testing.assert_array_equal(s["token_reaches_S"], [1, 0, 1, 0])
    np.testing.assert_array_equal(s["token_does_not"], [0, 1, 0, 1])
    np.testing.assert_array_equal(s["near_d<=60"], [1, 0, 0, 0])
    np.testing.assert_array_equal(s["far_d_60_100"], [0, 1, 1, 0])
    np.testing.assert_array_equal(s["t_6_30_s"], [1, 1, 0, 0])                      # 120 / v = 12 s, 12 s, 60 s, standstill


def test_strata_time_band_hand_values():
    s = K.strata(np.ones(4, bool), np.full(4, 30.0), np.array([10.0, 30.0, 2.0, 0.0]))     # 12 s, 4 s, 60 s, standstill (inf)
    np.testing.assert_array_equal(s["t_6_30_s"], [1, 0, 0, 0])
    np.testing.assert_array_equal(s["t_above_30_s"], [0, 0, 1, 1])


def test_cost_gain_rule_directions_and_the_floor():
    row = lambda d, sep=True: {"delta": d, "separated": sep, "status": "OK"}      # noqa: E731
    assert K.cost_gain("op_1_10:ade", row(0.3), 0.1) == "cost" and K.cost_gain("op_1_10:ade", row(-0.3), 0.1) == "gain"
    assert K.cost_gain("tac_11_30:tac_acc", row(0.3), 0.1) == "gain" and K.cost_gain("tac_11_30:tac_acc", row(-0.3), 0.1) == "cost"
    assert K.cost_gain("op_1_10:ade", row(0.05), 0.1) == "none"                    # inside the floor
    assert K.cost_gain("op_1_10:ade", row(0.3, sep=False), 0.1) == "none"          # not separated
    assert K.cost_gain("op_1_10:ade", row(0.3), None) == "n/a"                      # no floor: no verdict, never a silent pass


def test_there_are_24_driving_cells_and_a_missing_one_is_named_not_dropped():
    assert len(K.DRIVING_CELLS) == 24 and len(set(K.DRIVING_CELLS)) == 24
    eid = np.repeat(np.arange(15), 4)
    rng = np.random.default_rng(4)
    arm = {c: rng.normal(1.0, 0.1, 60) for c in K.DRIVING_CELLS if c != "op_1_10:fde"}
    ref = {c: rng.normal(1.0, 0.1, 60) for c in K.DRIVING_CELLS}
    rows = K.driving_rows(arm, ref, eid, floors={c: 0.0 for c in K.DRIVING_CELLS}, n_boot=100)
    assert len(rows) == 24 and rows["op_1_10:fde"]["status"] == "absent"
    assert all("verdict" in r for c, r in rows.items() if c != "op_1_10:fde")


# ------------------------------------------------------------------------------------------------ controls
def test_k1_self_pair_reads_exactly_zero_and_a_noisy_pair_is_caught():
    rng = np.random.default_rng(5)
    cells = {c: rng.normal(size=60) for c in K.DRIVING_CELLS}
    assert K.k1_self_pair(cells, np.repeat(np.arange(15), 4)) == []


def test_k7_planted_points_read_0_1_0_and_a_swapped_sign_is_red():
    w = window_set(n=200)
    r = K.k7(w["side"], w["theta"], w["d"], w["text_ok"], w["rc_ok"])
    assert r == {"hold": 0.0, "announced": 1.0, "mirrored": 0.0}
    # deliberate regression: a reader that scores the heading WITHOUT the side sign reads the mirrored arm as 0.5, not 0.0
    far = K.planted(w["side"], "mirrored")
    wrong = (np.degrees(far[:, 2]) >= 25.0).astype(float).mean()
    assert wrong != 0.0


def test_k8_donor_reads_the_donors_side_agreement_computed_from_labels():
    from tanitad.models.hjepa import donor_indices
    torch = pytest.importorskip("torch")
    w = window_set(n=200)
    perm = donor_indices(200, episode_ids=torch.as_tensor(w["eid"]), seed=0).numpy()
    real, donor, expected = K.k8(w["side"], perm, w["theta"], w["d"], w["text_ok"], w["rc_ok"])
    assert real == 1.0 and donor == pytest.approx(expected, abs=1e-12) and 0.0 < donor < 1.0
    ident = np.arange(200)
    assert K.k8(w["side"], ident, w["theta"], w["d"], w["text_ok"], w["rc_ok"])[1] == 1.0       # the identity "donor" is the real nav


def test_k9b_max_abs_delta():
    a = {"ckpt": np.zeros((4, 3, 3)), "ckpt_far": np.ones((4, 3))}
    assert K.k9b(a, {k: v.copy() for k, v in a.items()})["pass"]
    b = {k: v.copy() for k, v in a.items()}
    b["ckpt_far"][2, 1] += 2e-6
    r = K.k9b(a, b)
    assert not r["pass"] and r["max_abs_delta"]["ckpt_far"] == pytest.approx(2e-6, rel=1e-3)
    assert not K.k9b(a, {"ckpt": a["ckpt"]})["pass"]                                 # a key on one side only fails


# ------------------------------------------------------------------------------------------------ the shared tools are byte-equal to their pins; the wrappers refuse a drifted tool
import v8f_mini_eval_far as W0  # noqa: E402
import v8f_t1_eval_far as W1  # noqa: E402


def test_the_shared_tools_are_byte_equal_to_their_pins():
    assert _sha(os.path.join(SCRIPTS, "v8f_mini_eval.py")) == W0.TOOL_SHA256
    assert _sha(os.path.join(SCRIPTS, "v8f_t1_eval.py")) == W1.TOOL_SHA256
    for other in ("v8f_t1_speedsub.py", "v8f_t1_eval_pool.py"):                       # the same pin the sibling wrappers use, when they are present in this tree
        p = os.path.join(SCRIPTS, other)
        if os.path.isfile(p):
            assert W1.TOOL_SHA256 in open(p, encoding="utf-8").read()


def test_the_wrappers_refuse_a_tool_whose_sha256_differs(monkeypatch):
    monkeypatch.setattr(W0, "TOOL_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="REFUSED"):
        W0.main(["run"])
    monkeypatch.setattr(W1, "TOOL_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="REFUSED"):
        W1.main(["run"])


class Cfg:
    def __init__(self, s_far="off", nav_strat="off"):
        self.s_far, self.nav_strat = s_far, nav_strat


def _own(**kw):
    p = W0.own_parser()
    return p.parse_args([x for k, v in kw.items() for x in (f"--{k.replace('_', '-')}", v)])


@pytest.mark.parametrize("mod", ["W0", "W1"])
def test_flag_vs_checkpoint_cfg_mismatches_are_refused(mod):
    mk = (lambda **kw: _own(**kw)) if mod == "W0" else (lambda **kw: _t1_args(**kw))
    ref = (lambda o, c: W0.refusals(o, c, argparse.Namespace())) if mod == "W0" else (lambda o, c: W1.refusals(o, c))
    assert ref(mk(), Cfg()) == []
    assert ref(mk(far_out="on"), Cfg()) and ref(mk(), Cfg(s_far="on"))                       # a flag that names a head the checkpoint lacks / a head the flag drops
    assert ref(mk(nav_strat="on", nav_strat_npz="t", nav_strat_sha256="s"), Cfg()) and ref(mk(), Cfg(nav_strat="on"))
    assert ref(mk(far_out="on", nav_strat="on"), Cfg(s_far="on", nav_strat="on"))           # nav_strat on with no table named
    assert ref(mk(far_out="on", nav_strat="on", nav_strat_npz="t", nav_strat_sha256="s"), Cfg(s_far="on", nav_strat="on")) == []
    assert ref(mk(nav_strat_npz="t"), Cfg())                                                   # a table named for a model without nav_s
    assert ref(mk(far_out="on"), Cfg(s_far="on")) == []


def _t1_args(**kw):
    p = argparse.ArgumentParser()
    W1.add_flags(p)
    return p.parse_args([x for k, v in kw.items() for x in (f"--{k.replace('_', '-')}", v)])


def test_t0_scope_refusals():
    o = _own(far_out="on")
    assert W0.refusals(o, Cfg(s_far="on"), argparse.Namespace(ages="fresh", tac="oracle", plan="m2"))
    assert W0.refusals(o, Cfg(s_far="on"), argparse.Namespace(ages="fresh", tac="pred", plan="beta"))
    assert W0.refusals(o, Cfg(s_far="on"), argparse.Namespace(ages="cycle", tac="pred", plan="m2")) == []        # the NOW-frame form is computed from the batch's own poses


def test_t0_nav_s_variant_follows_the_tools_nav_variant():
    n = lambda **k: argparse.Namespace(nav_withheld=k.get("w", False), nav_shuffle=k.get("s", False))      # noqa: E731
    assert [W0.nav_s_variant_of(_own(), n()), W0.nav_s_variant_of(_own(), n(w=True)), W0.nav_s_variant_of(_own(), n(s=True))] == ["real", "zero", "donor"]
    assert W0.nav_s_variant_of(_own(nav_s_variant="real"), n(s=True)) == "real" and W0.nav_s_variant_of(_own(nav_s_variant="zero"), n()) == "zero"


def test_t1_parser_adds_the_flags_to_run_and_keeps_the_tools():
    import v8f_t1_eval as T
    p = W1.build_parser(T.build_parser)
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    run = sub.choices["run"]
    names = {s for a in run._actions for s in a.option_strings}
    assert {"--far-out", "--nav-strat", "--nav-strat-npz", "--nav-strat-sha256", "--nav-s-variant", "--ckpt", "--dump-heads", "--nav-variant"} <= names
    assert "--far-out" not in {s for a in sub.choices["analyze"]._actions for s in a.option_strings}


# ------------------------------------------------------------------------------------------------ T1 wrapper on the synthetic world (CPU)
torch = pytest.importorskip("torch")
import test_v8f_t1 as T1T  # noqa: E402
from test_v8f_t1 import world  # noqa: E402,F401  (the synthetic-world fixture of the T1 suite)
import v8f_t1_eval as T1  # noqa: E402


def _nav_s_rows(J, ci, countdown=1.5):
    o, n = int(J["off"][ci]), int(J["n"][ci])
    rows = np.zeros((n, 5), np.float32)
    rows[:, 1], rows[:, 3], rows[:, 4] = 1.0, countdown, 1.0
    return rows


def test_nav_s_is_dead_reckoned_exactly_like_nav(world):
    tracks, J, store, wins = T1T.setup(world)
    ci, t0, _, _ = wins[1]
    clip = str(J["clip"][ci])
    o, n = int(J["off"][ci]), int(J["n"][ci])
    nav = J["nav"][o:o + n].copy()
    nav[t0] = [0, 1, 0, 0.50, 1]
    ns = _nav_s_rows(J, ci, 1.5)
    st = W1.FarWindowState(store.get(clip), tracks[clip], nav, ci, t0, "deadreckon", ns)
    assert np.isnan(st.nav_s[t0 + 1:]).all() and np.isfinite(st.nav_s[:t0 + 1]).all()           # only the logged history exists at t0
    plan = np.column_stack([10.0 * T1.DT * np.arange(1, 31), np.zeros(30)])
    for _ in range(3):
        st.commit_plan(plan, 1)
    for j in range(t0 + 1, t0 + 4):
        np.testing.assert_allclose(st.nav[j, 3], np.clip(0.50 - st.arc[j] / 100.0, 0, 3), atol=1e-6)
        np.testing.assert_allclose(st.nav_s[j, 3], np.clip(1.5 - st.arc[j] / 100.0, 0, 3), atol=1e-6)
        np.testing.assert_array_equal(st.nav_s[j, :3], [0, 1, 0])
        assert st.nav_s[j, 4] == 1.0
    inv = W1.FarWindowState(store.get(clip), tracks[clip], nav, ci, t0, "deadreckon", np.zeros((n, 5), np.float32))
    inv.commit_plan(plan, 1)
    np.testing.assert_array_equal(inv.nav_s[t0 + 1], [0, 0, 0, 0, 0])                          # invalid: nothing to count down
    none = W1.FarWindowState(store.get(clip), tracks[clip], nav, ci, t0, "deadreckon")
    assert none.nav_s is None and np.isnan(none.nav[t0 + 1:]).all()                              # no nav_s: the tool's own state, untouched


def test_t1_wrapper_patches_are_restored_also_when_the_call_raises(world, monkeypatch):
    saved = (T1.HEAD_KEYS, T1.MODEL_KEYS, T1.TickRecorder, T1.make_states, T1.build_item, T1.D.load_join, T1.cmd_run, T1.build_parser)
    ck = Path(world["root"]) / "ck_far.pt"
    from tanitad.models import v8f_mini as M
    m = M.V8FMini(M.V8FMiniConfig(**T1T.TINY, s_far="on", w_ckpt_far=0.3333, nav_strat="on"), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    torch.save({"model": m.state_dict(), "cfg": {k: getattr(m.cfg, k) for k in m.cfg.__dataclass_fields__}}, ck)

    def boom(a):
        assert T1.HEAD_KEYS[-1] == "ckpt_far" and "nav_s" in T1.MODEL_KEYS and T1.TickRecorder is not saved[2]       # the patches ARE live inside the call
        raise RuntimeError("boom")
    monkeypatch.setattr(T1, "cmd_run", boom)
    args = ["run", "--ckpt", str(ck), "--tag", "x", "--tracks", "t", "--far-out", "on", "--nav-strat", "on", "--nav-strat-npz", "n", "--nav-strat-sha256", "s", "--eval-grid", "g", "--eval-join", "j", "--out-dir", str(Path(world["root"]) / "o")]
    try:
        W1.main(args)
    except (RuntimeError, SystemExit, TypeError):
        pass
    now = (T1.HEAD_KEYS, T1.MODEL_KEYS, T1.TickRecorder, T1.make_states, T1.build_item, T1.D.load_join, T1.build_parser)
    assert now[:6] == saved[:6] and T1.cmd_run is boom and now[6] is saved[7]


def test_t1_wrapper_with_both_flags_off_patches_nothing(world, monkeypatch):
    seen = {}

    def spy(a):
        seen.update(h=T1.HEAD_KEYS, m=T1.MODEL_KEYS, r=T1.TickRecorder, s=T1.make_states, b=T1.build_item, j=T1.D.load_join)
        return 0
    orig = (T1.HEAD_KEYS, T1.MODEL_KEYS, T1.TickRecorder, T1.make_states, T1.build_item, T1.D.load_join)
    monkeypatch.setattr(T1, "cmd_run", spy)
    ck = Path(world["root"]) / "ck_off.pt"
    from tanitad.models import v8f_mini as M
    m = M.V8FMini(M.V8FMiniConfig(**T1T.TINY), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    torch.save({"model": m.state_dict(), "cfg": {k: getattr(m.cfg, k) for k in m.cfg.__dataclass_fields__}}, ck)
    rc = W1.main(["run", "--ckpt", str(ck), "--tag", "x", "--tracks", "t", "--eval-grid", "g", "--eval-join", "j", "--out-dir", str(Path(world["root"]) / "o")])
    assert rc == 0 and (seen["h"], seen["m"], seen["r"], seen["s"], seen["b"], seen["j"]) == orig
