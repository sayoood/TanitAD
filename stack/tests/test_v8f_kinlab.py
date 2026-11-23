"""KINLAB (BRIEF_BUILD_KINLAB, MM design 2026-11-23): ``--dagger-label kin_v1``, the KINEMATIC label of the ROLLHIST line. CPU only, on the synthetic world of the dagger tests.

Run the WHOLE file (module fixtures are shared, and a ``-k`` run leaks them): CUDA_VISIBLE_DEVICES=-1 pytest -q stack/tests/test_v8f_kinlab.py

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): each claim has a test that goes RED when the single line that implements it is deleted or bent (``mutants/run_mutants.py``).

  gold          ``human`` and ``speed_v1`` equal the code of record df250a00 (``golden_v8f_dagger_df250a00.py`` / ``golden_v8f_recovery_df250a00.py`` = that commit's blobs, sha256 pinned): every key, bit for bit, on planted AND bank records
  identity      the identity rule: own (a, kappa) == the reference's first step -> the speed_v1 label, ``np.array_equal``; one tolerance step away it is not
  curvature     a planted own kappa > 0 curves the label toward the own side (kappa < 0: the other side), THEN merges into the reference path (t >= T_B, on the polyline after the merge)
  refusal       a planted a_o > A_LIM is refused for the checker's accel term (the speed step at the merge); the friction circle is applied (an emitted label passes the checker at v^2 kappa >> F_LIM);
                the three refusal reasons are separate counters; a refused record keeps its speed_v1 label (not the human one)
  eligibility   the eligible index is EQUAL under human / speed_v1 / kin_v1
  reachability  argv -> argparse -> bank (kin_v1 attached) -> swap -> ``gt`` in the batch -> the plan loss term: a NON-ZERO gradient difference kin_v1 vs speed_v1 on a planted batch, an EXACT 0.0 for speed_v1 against itself
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_dagger as G  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.data import v8f_recovery as RV  # noqa: E402
from test_v8f_dagger import world, fx10, make_batch  # noqa: E402,F401  (the synthetic world and the bank the CLI generated)
from test_v8f_dagger_train import pack, DAG, TR, DG, M, dvb, _woken, TINY, _run, _summary  # noqa: E402,F401
from test_v8f_recovery import _path, chord_speed, speed_bank, eligible_windows_of, _same  # noqa: E402,F401
from test_v8f_recovery_wiring import gate_parts_rl  # noqa: E402,F401

GOLD_D = HERE / "golden_v8f_dagger_df250a00.py"
GOLD_R = HERE / "golden_v8f_recovery_df250a00.py"
GOLD_D_BLOB_SHA256 = "87432b60bcca889a18ccf41ebaadeb1785fd424f484013f5f0812d40c5f0446c"      # sha256 of `git show df250a00:stack/tanitad/data/v8f_dagger.py` (git blob 284a520c)
GOLD_R_BLOB_SHA256 = "a94d2b522913d18ee7ad127a87cb23a3842a7e25c539235524331557495c01d6"      # sha256 of `git show df250a00:stack/tanitad/data/v8f_recovery.py` (git blob 1825b617)
torch.set_num_threads(1)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _gold():
    return _load(GOLD_D, "golden_v8f_dagger_df250a00"), _load(GOLD_R, "golden_v8f_recovery_df250a00")


R0 = 20


def planted(kw=None, dv=0.0, da=None, dk=None, r=R0):
    """A planted own state on a synthetic logged path: (grid, r, (v_o, a_o, kappa_o)). ``dv`` = own speed minus the human chord speed at r'; ``da`` / ``dk`` None = the reference's own first-step values (a_ref0 / kappa_ref0), else absolute."""
    grid, _ = _path(**(kw or {}))
    vo = chord_speed(grid, r) + dv
    sp = RV.speed_label(grid, r, vo)
    Gp = grid[r:]
    P = RV.to_ego(Gp[:, :2], (Gp[0, 0], Gp[0, 1], Gp[0, 2]))
    s_ = RV._polyline(P)
    ul = RV.label_steps(vo, np.concatenate([[0.0], (s_[1:RV.H + 1] - s_[:RV.H]) / RV.DT]))
    a0, k0 = RV.ref_start(sp["label"], ul)
    return grid, r, (vo, a0 if da is None else da, k0 if dk is None else dk), (a0, k0)


def kin(grid, r, own, **kw):
    return RV.kin_label(grid, r, own, **kw)


# ------------------------------------------------------------------------------------------------ constants + flag
def test_constants_are_the_briefs_and_the_flag_value_exists():
    assert (RV.T_B, RV.LD_MIN, RV.LD_GAIN, RV.MERGE_LAT_M, RV.H_SUB, RV.T_END, RV.IDENT_TOL) == (1.0, 4.0, 1.0, 0.10, 0.05, 6.0, 1e-6)
    assert RV.MERGE_SPEED_MS == 0.05 and RV.A_LABEL == 3.0                                          # phase 3 (MM construction fix): the merge's speed condition; the rule's clip is speed_v1's A_LABEL
    assert abs(RV.MERGE_HEAD_RAD - math.radians(1.0)) < 1e-15 and RV.SUB_PER_STEP == 4 and RV.H * RV.SUB_PER_STEP * RV.H_SUB == RV.T_END
    assert abs(RV.A_LIM - 3.96) < 1e-12 and abs(RV.K_LIM - 0.198) < 1e-12 and abs(RV.F_LIM - 6.80) < 0.01
    assert RV.DAGGER_LABELS == ("human", "speed_v1", "kin_v1")
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.dagger_label == "human"
    assert TR.build_parser().parse_args(["--out", "x", "--dagger-label", "kin_v1"]).dagger_label == "kin_v1"
    with pytest.raises(SystemExit):
        TR.build_parser().parse_args(["--out", "x", "--dagger-label", "kin_v2"])
    assert TR.FLAG_CHECKS["dagger_label"] == "dagger:dagger_label"
    assert {act.dest for act in TR.build_parser()._actions if act.dest != "help"} == set(TR.FLAG_CHECKS)      # no new flag beyond the new value


# ------------------------------------------------------------------------------------------------ gold
def test_the_gold_copies_are_the_commits_blobs():
    assert hashlib.sha256(GOLD_D.read_bytes()).hexdigest() == GOLD_D_BLOB_SHA256
    assert hashlib.sha256(GOLD_R.read_bytes()).hexdigest() == GOLD_R_BLOB_SHA256


@pytest.mark.parametrize("p,seed,step", [(1.0, 5, 3), (0.5, 1, 4), (0.3, 7, 11)])
def test_gold_human_swap_equals_df250a00_bit_for_bit(fx10, p, seed, step):
    gd, _ = _gold()
    wins = [x for x in fx10.wins if fx10.bank.candidates([x[0]], [x[1]])[0][0] > 0][:24]
    b = make_batch(fx10, wins)
    new = G.DaggerSwap(fx10.bank, p=p, seed=seed).apply(b, step)
    gbank = gd.DaggerBank(fx10.bank.a, fx10.bank.header, clip_names=fx10.J["clip"], manifest=fx10.bank.manifest, sha256=fx10.bank.sha256, path=fx10.bank.path)
    old = gd.DaggerSwap(gbank, p=p, seed=seed).apply(b, step)
    assert set(new) == set(old) == set(b)
    for k in b:
        assert _same(new[k], old[k]), f"{k} differs from df250a00 under --dagger-label human"
    assert fx10.bank.rl is None and fx10.bank.label_kind == "human" and "rl_has" not in fx10.bank.gather([0]) and "rl_kin" not in fx10.bank.gather([0])


@pytest.mark.parametrize("kw,dv", [(dict(), -1.0), (dict(kappa=0.01), -1.5), (dict(a=0.3), 1.2), (dict(a=-0.4, v0=9.0), -2.5), (dict(), 0.3), (dict(kappa=0.01), 5.0)])
def test_gold_speed_v1_label_equals_df250a00_on_planted_records(kw, dv):
    _, gr = _gold()
    grid, _v = _path(**kw)
    vo = chord_speed(grid, R0) + dv
    a, b = RV.speed_label(grid, R0, vo), gr.speed_label(grid, R0, vo)
    assert a["status"] == b["status"] and a["u_h1"] == b["u_h1"]
    for k in ("label", "delta", "viol"):
        assert _same(a[k], b[k]) if k != "viol" else a[k] == b[k], k
    for check in (True, False):
        for clip in (True, False):
            x, y = RV.speed_label(grid, R0, vo, clip=clip, check=check), gr.speed_label(grid, R0, vo, clip=clip, check=check)
            assert x["status"] == y["status"] and _same(x["label"], y["label"])
    assert RV.label_steps(vo, np.arange(31.0)).tolist() == gr.label_steps(vo, np.arange(31.0)).tolist()
    for fn in ("human_label", "violations", "arc_from_label"):
        assert getattr(RV, fn).__code__.co_code == getattr(gr, fn).__code__.co_code, fn


def test_gold_speed_v1_bank_labels_and_swap_equal_df250a00_on_bank_records(fx10):
    gd, gr = _gold()
    rb = speed_bank(fx10, plant=8)
    res = RV.label_bank(rb.a, rb.ci, fx10.J["clip"], fx10.store.get_kin)
    old = gr.label_bank(rb.a, rb.ci, fx10.J["clip"], fx10.store.get_kin)
    assert np.array_equal(res["slot"], old["slot"]) and np.array_equal(res["label"], old["label"]) and res["status_counts"] == old["status_counts"] and res["census"] == old["census"]
    assert res["n_labelled"] == old["n_labelled"] > 0
    rb.attach_recovery(res["slot"], res["label"], res["status_counts"])
    assert rb.label_kind == "speed_v1" and "rl_kin" not in rb.gather([0]) and rb.rl["kin"] is None
    gb = gd.DaggerBank(rb.a, rb.header, clip_names=fx10.J["clip"], manifest=rb.manifest, sha256=rb.sha256, path=rb.path)
    gb.attach_recovery(old["slot"], old["label"], old["status_counts"])
    wins = eligible_windows_of(fx10, rb)
    b = make_batch(fx10, wins)
    sw, gsw = G.DaggerSwap(rb, p=1.0, seed=3), gd.DaggerSwap(gb, p=1.0, seed=3)
    new, o = sw.apply(b, 2), gsw.apply(b, 2)
    assert set(new) == set(o)
    for k in b:
        assert _same(new[k], o[k]), f"{k} differs from df250a00 under --dagger-label speed_v1"
    assert sw.last["n_label_replaced"] == gsw.last["n_label_replaced"] > 0 and sw.last["n_label_kin"] == 0
    assert np.array_equal(rb._ok_idx, gb._ok_idx)


# ------------------------------------------------------------------------------------------------ the identity rule
@pytest.mark.parametrize("kw,dv", [(dict(), 0.0), (dict(), -1.2), (dict(kappa=0.01), 0.4), (dict(a=0.3), 2.0)])
def test_identity_rule_own_equal_to_the_reference_first_step_is_the_speed_v1_label_bit_for_bit(kw, dv):
    grid, r, own, (a0, k0) = planted(kw, dv)
    res = kin(grid, r, own)
    sp = RV.speed_label(grid, r, own[0])
    assert res["status"] == "identity_rule" and np.array_equal(res["label"], sp["label"]) and res["label"].dtype == sp["label"].dtype
    inside = kin(grid, r, (own[0], a0 + 0.9e-6, k0 - 0.9e-6))
    assert inside["status"] == "identity_rule" and np.array_equal(inside["label"], sp["label"])
    for da, dk in ((3e-6, 0.0), (0.0, 3e-6), (-3e-6, 3e-6)):                      # one tolerance step further: the integration runs
        out = kin(grid, r, (own[0], a0 + da, k0 + dk))
        assert out["status"] != "identity_rule", (da, dk)


# ------------------------------------------------------------------------------------------------ own kappa curves the label toward the own side, then merges
@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_planted_own_kappa_curves_toward_the_own_side_then_merges(sign):
    grid, r, own, (a0, k0) = planted(dict(), 0.0, dk=sign * 0.04)                                 # a straight human path, an own state turning
    res = kin(grid, r, own)
    sp = RV.speed_label(grid, r, own[0])["label"]
    assert res["status"] == "kin_emitted", res["status"]
    lab = res["label"]
    assert abs(sp[:, 1]).max() < 1e-9                                                              # the reference is straight (the frame at r')
    assert (sign * lab[:5, 1] > 0.02).all(), "the first 1 s curves toward the own side"
    assert 1.0 <= res["t_merge"] <= 6.0 and res["max_dev"] > 0.1
    km = int(round(res["t_merge"] / RV.DT))
    assert (np.abs(lab[km:, 1]) < 1e-9).all()                                                      # after the merge point every point is on the reference path (a straight line: y == 0)
    assert abs(lab[km - 1, 1]) < RV.MERGE_LAT_M, "the merge point itself is inside the merge tolerance"
    assert not np.array_equal(lab, sp)
    assert np.isfinite(lab).all() and RV.violations(lab, own[0], 30)["any"] is False


def test_the_two_signs_are_mirror_images_on_a_straight_path():
    g, r, own, _ = planted(dict(), 0.0, dk=0.03)
    g2, r2, own2, _ = planted(dict(), 0.0, dk=-0.03)
    a, b = kin(g, r, own)["label"], kin(g2, r2, own2)["label"]
    assert np.allclose(a[:, 0], b[:, 0], atol=1e-9) and np.allclose(a[:, 1], -b[:, 1], atol=1e-9)


def test_own_start_matters_each_input_changes_the_label_and_a_zero_blend_horizon_would_erase_the_start():
    g, r, own, (a0, k0) = planted(dict(), 0.0, da=0.0, dk=0.0)
    base = kin(g, r, (own[0], a0 + 1.0, k0 + 0.03))["label"]
    assert not np.allclose(base, kin(g, r, (own[0], a0 + 1.0, k0))["label"], atol=1e-6), "kappa_o moves the label"
    assert not np.allclose(base, kin(g, r, (own[0], a0, k0 + 0.03))["label"], atol=1e-6), "a_o moves the label"
    assert not np.allclose(base[:5], RV.speed_label(g, r, own[0])["label"][:5], atol=1e-3), "the first second is not the speed_v1 label"
    assert not np.allclose(kin(g, r, (own[0] + 1.0, a0 + 1.0, k0 + 0.03), check=False)["label"], base, atol=1e-6), "v_o moves the label"
    # w(t): at t = 0 the label's first sub-step turns at kappa_o (w = 0), at t >= T_B at the pursuit's (w = 1): a longer path to merge for a larger kappa_o
    a = kin(g, r, (own[0], a0, k0 + 0.01))
    b = kin(g, r, (own[0], a0, k0 + 0.06))
    assert a["status"] == b["status"] == "kin_emitted" and b["max_dev"] > a["max_dev"] and b["t_merge"] >= a["t_merge"]


def test_a_label_that_is_not_merged_by_six_seconds_is_refused_not_merged_and_a_lookahead_past_the_path_end_is_refused_path_exhausted(monkeypatch):
    g, r, own, (a0, k0) = planted(dict(), 0.0, dk=0.03)
    assert kin(g, r, own)["status"] == "kin_emitted"
    with monkeypatch.context() as mp:
        mp.setattr(RV, "MERGE_LAT_M", 1e-9)                                                       # an unreachable merge tolerance: the label never merges
        res = kin(g, r, own)
    assert res["status"] == "refused_not_merged" and res["label"] is None and np.array_equal(res["speed_label"], RV.speed_label(g, r, own[0])["label"])      # the fallback is the speed_v1 label
    g2, r2, own2, _ = planted(dict(v0=1.0), 0.0, dk=0.19, r=len(planted(dict())[0]) - 31)       # exactly 30 logged future rows of a slow path (6 m in 6 s) and a late merge: the 4 m lookahead runs past its end
    res2 = kin(g2, r2, own2)
    assert res2["status"] == "refused_path_exhausted" and res2["label"] is None and res2["speed_label"] is not None, res2["status"]


def test_t_b_is_the_blend_horizon_a_longer_horizon_keeps_the_own_curvature_longer(monkeypatch):
    g, r, own, _ = planted(dict(), 0.0, dk=0.04)
    dev = {}
    for tb in (0.25, 1.0, 2.0):
        with monkeypatch.context() as mp:
            mp.setattr(RV, "T_B", tb)
            res = kin(g, r, own, check=False)
        assert res["status"] == "kin_emitted"
        dev[tb] = float(np.abs(res["label"][:5, 1]).max())                                         # the first second's lateral excursion
    assert dev[0.25] < dev[1.0] < dev[2.0], dev


# ------------------------------------------------------------------------------------------------ refusals, one reason each
def test_planted_a_o_above_a_lim_is_clipped_in_the_integration_and_a_checker_accel_flag_refuses_and_keeps_the_speed_label(monkeypatch):
    # PHASE 3: the speed now follows the speed_v1 RULE on the integrated state, so the merge step no longer jumps the speed and an own a_o beyond A_LIM is simply clipped to A_LIM (phase 2 refused it at the merge step).
    g, r, own, (a0, k0) = planted(dict(), 0.0, da=6.0)
    assert own[1] > RV.A_LIM
    res = kin(g, r, own)
    assert res["status"] == "kin_emitted", (res["status"], res["viol"])
    km = int(round(res["t_merge"] / RV.DT))
    pts = np.concatenate([[[-own[0] * RV.DT, 0.0], [0.0, 0.0]], res["label"][:km]])
    assert np.abs(RV.controls(pts)[1]).max() <= RV.A_LIM + 1e-9, "|a(t)| <= A_LIM in the integration"
    assert np.abs(RV.controls(pts)[1]).max() > 0.5 * RV.A_LIM, "and the start is a real acceleration, not zero"
    # the refusal path of the checker's accel term is still wired: a checker with a tighter acceleration limit refuses the same record for the accel term and the caller keeps the speed_v1 label
    real = RV.violations
    with monkeypatch.context() as mp:
        mp.setattr(RV, "violations", lambda lab, v, upto, **kw: real(lab, v, upto, a_lim=1.0))
        tight = kin(g, r, own)
    assert tight["status"] == "refused_infeasible" and tight["viol"]["accel"] > 0, (tight["status"], tight["viol"])
    assert tight["label"] is None and np.array_equal(tight["speed_label"], RV.speed_label(g, r, own[0])["label"])
    unchecked = kin(g, r, own, check=False)
    assert unchecked["status"] == "kin_emitted"


@pytest.mark.parametrize("dv,da", [(-0.7, 1.5), (-0.7, -1.2), (1.0, 2.0), (0.0, 0.8)])
def test_phase3_a_planted_a_o_different_from_a_ref0_merges_with_no_speed_step_above_a_label_dt(dv, da):
    g, r, own, (a0, k0) = planted(dict(), dv, da=None)
    own = (own[0], a0 + da, k0)
    assert abs(own[1] - a0) > 0.5, "a_o differs from the reference's first-step acceleration"
    res = kin(g, r, own)
    assert res["status"] == "kin_emitted", (res["status"], res["viol"])
    km = int(round(res["t_merge"] / RV.DT))
    assert 5 <= km <= 30
    pts = np.concatenate([[[-own[0] * RV.DT, 0.0], [0.0, 0.0]], res["label"]])
    spd = RV.controls(pts)[0]                                                                      # chord speeds of [pred -> origin, origin -> p1, ..., p29 -> p30] (a straight path: chord = arc)
    dsp = np.abs(np.diff(spd))
    lim = RV.A_LABEL * RV.DT + 1e-9
    tail = dsp[km:]                                                                                # the step from the merge point's step on: index j of dsp = speed change into label step j (spd[0] = own)
    assert tail.size and tail.max() <= lim, (tail.max(), lim)
    assert dsp[km - 1] <= lim + 0.4, "the merge step itself does not jump (the integrated chord speed is within the merge band of the human's)"
    uh = np.hypot(*np.diff(np.concatenate([[[0.0, 0.0]], res["label"]]), axis=0).T) / RV.DT
    assert abs(uh[km - 1] - chord_speed(g, r + km - 1)) < 0.6, "at the merge the label's speed is on the human's"
    sp = RV.speed_label(g, r, own[0])["label"]
    assert not np.array_equal(res["label"], sp)


def test_phase3_a_record_that_cannot_meet_the_speed_condition_by_six_seconds_is_refused_not_merged():
    # a human who accelerates at 3.6 m/s2 (above the rule's 3.0 clip): the integrated speed can never catch the human's step speed, the lateral part merges at t = T_B, the speed condition never holds
    g, r, own, (a0, k0) = planted(dict(a=3.6, v0=2.0, T=200), -0.5, da=None)
    own = (own[0], a0 + 0.6, k0)
    res = kin(g, r, own)
    assert res["status"] == "refused_not_merged" and res["label"] is None and res["t_merge"] is None, (res["status"], res["viol"])
    assert res["speed_label"] is not None, "the fallback is the speed_v1 label"
    with monkeypatch_ctx() as mp:                                                                  # the control: without the speed condition the very same record merges laterally (so it IS the speed condition that refused it)
        mp.setattr(RV, "MERGE_SPEED_MS", 1e9)
        ctl = kin(g, r, own, check=False)
    assert ctl["status"] == "kin_emitted" and 1.0 <= ctl["t_merge"] <= 1.2, (ctl["status"], ctl["t_merge"])


def monkeypatch_ctx():
    return pytest.MonkeyPatch().context()


def test_the_friction_circle_scales_kappa_down_so_the_emitted_label_passes_the_checker():
    g, r, own, _ = planted(dict(v0=12.0), 0.0, da=0.0, dk=0.15)                                   # v^2 kappa = 21.6 m/s2 against F_LIM 6.8: an unscaled start turns far past the circle
    assert own[0] ** 2 * 0.15 > 2 * RV.F_LIM
    res = kin(g, r, own)
    assert res["status"] == "kin_emitted", (res["status"], res["viol"])
    assert RV.violations(res["label"], own[0], 30)["max_friction"] <= RV.F_LIM
    turn = math.atan2(res["label"][1, 1] - res["label"][0, 1], res["label"][1, 0] - res["label"][0, 0])
    assert abs(turn) < 0.25, "the first turn is the friction-limited kappa (about 0.17 rad), not the unscaled one (>= 0.3 rad)"


def test_each_refusal_reason_has_its_own_counter_and_a_refused_record_falls_back_to_speed_v1_then_human():
    g, r, own, (a0, k0) = planted(dict(), 0.0)
    sp = RV.speed_label(g, r, own[0])
    assert kin(g, r, (own[0], float("nan"), k0))["status"] == "refused_nonfinite"
    assert kin(g, r, (float("inf"), a0, k0))["status"] in ("speed_nonfinite", "refused_nonfinite")
    assert kin(g, len(g) - 5, own)["status"] == "speed_no_logged_future"
    short = len(g) - 31                                                                            # exactly 30 future rows
    own_fast = (chord_speed(g, short) + 3.0, 0.0, 0.0)
    assert kin(g, short, own_fast)["status"] == "speed_path_exhausted"                             # the reference itself is refused: kin_v1 refuses it too, and it has no speed label to fall back to
    assert kin(g, short, own_fast)["speed_label"] is None
    slow = (chord_speed(g, short) - 1.5, 0.3, 0.02)
    out = kin(g, short, slow)
    assert out["status"] in ("kin_emitted", "refused_path_exhausted", "refused_not_merged", "refused_infeasible")
    assert set(RV.KIN_REFUSALS) == {"refused_nonfinite", "refused_not_merged", "refused_path_exhausted", "refused_infeasible"} and len(set(RV.KIN_STATUSES)) == len(RV.KIN_STATUSES)
    assert sp["status"] == RV.STATUS_EMITTED or sp["status"] == RV.STATUS_IDENTITY


# ------------------------------------------------------------------------------------------------ the bank: eligibility, fallback, counters
def kin_bank(f, plant=8, kappa=0.03, da=0.9):
    """A copy of the fixture's bank with ``plant`` ok records given a planted own state (v, a, kappa) = (human chord speed - 0.7, ref a + da, kappa)."""
    arrays = {k: np.array(v) for k, v in f.bank.a.items()}
    ok = np.flatnonzero(arrays["reason"] == G.OK)
    for n, i in enumerate(ok[:plant]):
        c, r = int(f.bank.ci[i]), int(arrays["r"][i])
        pg = f.store.get_kin(str(f.J["clip"][c]))["pose"]
        arrays["p10f"][i, -1, 0] = np.float32(chord_speed(pg, r) - 0.7)
        arrays["p10f"][i, -1, 1] = np.float32(da * (1 if n % 2 else -1))
        arrays["p10f"][i, -1, 2] = np.float32(kappa * (1 if n % 3 else -1))
    return G.DaggerBank(arrays, f.bank.header, clip_names=f.J["clip"], manifest=f.bank.manifest, sha256=f.bank.sha256, path=f.bank.path)


def attach_kin(f, bank):
    res = RV.kin_label_bank(bank.a, bank.ci, f.J["clip"], f.store.get_kin)
    bank.attach_recovery(res["slot"], res["label"], res["status_counts"], kin=res["is_kin"])
    return res


def test_the_eligible_index_is_equal_under_all_three_label_values(fx10):
    hb, sb, kb = kin_bank(fx10), kin_bank(fx10), kin_bank(fx10)
    res_s = RV.label_bank(sb.a, sb.ci, fx10.J["clip"], fx10.store.get_kin)
    sb.attach_recovery(res_s["slot"], res_s["label"], res_s["status_counts"])
    res_k = attach_kin(fx10, kb)
    assert (hb.label_kind, sb.label_kind, kb.label_kind) == ("human", "speed_v1", "kin_v1")
    for x in (sb, kb):
        assert np.array_equal(x._ok_idx, hb._ok_idx) and np.array_equal(x.eligible_mask, hb.eligible_mask) and np.array_equal(x._ok_key, hb._ok_key)
    # off-route / end_edge / nonfinite records get no label under kin_v1 exactly as under speed_v1
    assert not (kb.rl["slot"] >= 0)[kb.reason != G.OK].any() and not kb.rl["kin"][kb.reason != G.OK].any()
    assert res_k["n_labelled"] == res_s["n_labelled"] == int((kb.reason == G.OK).sum())
    wins = [w for w in fx10.wins if hb.candidates([w[0]], [w[1]])[0][0] > 0]
    assert len(wins) > 0 and wins == [w for w in fx10.wins if kb.candidates([w[0]], [w[1]])[0][0] > 0] == [w for w in fx10.wins if sb.candidates([w[0]], [w[1]])[0][0] > 0]


def test_kin_label_bank_equals_the_per_record_labeller_and_its_fallbacks(fx10):
    kb = kin_bank(fx10)
    res = attach_kin(fx10, kb)
    ok = np.flatnonzero(kb.reason == G.OK)
    n_kin = n_fb = 0
    for i in ok:
        c, r = int(kb.ci[i]), int(kb.a["r"][i])
        grid = np.asarray(fx10.store.get_kin(str(fx10.J["clip"][c]))["pose"], np.float64)[:, :3]
        one = RV.kin_label(grid, r, np.asarray(kb.a["p10f"][i, -1, :3], np.float64))
        sl = kb.rl["slot"][i]
        if one["status"] in ("kin_emitted", "identity_rule"):
            assert sl >= 0 and np.array_equal(kb.rl["label"][sl], one["label"]) and bool(kb.rl["kin"][i]) == (one["status"] == "kin_emitted")
            n_kin += one["status"] == "kin_emitted"
        elif one["speed_label"] is not None:                                                       # a refused record keeps its speed_v1 label
            n_fb += 1
            assert sl >= 0 and np.array_equal(kb.rl["label"][sl], one["speed_label"]) and not kb.rl["kin"][i]
        else:                                                                                      # the speed label was refused too: the human label (no slot)
            assert sl < 0 and not kb.rl["kin"][i]
    assert n_kin > 0, f"no kin_v1 label on the fixture ({res['status_counts']})"
    assert sum(res["status_counts"].values()) == res["n_labelled"] == len(ok)
    cen = res["census"]
    assert cen["per_tick"]["pooled"]["ok"] == len(ok) and cen["per_tick"]["pooled"]["relabelled"] == n_kin == int(kb.rl["kin"].sum())
    for t, row in cen["per_tick"].items():
        assert row["relabelled"] + row["identity"] + sum(row["refused"].values()) == row["ok"]
    assert abs(cen["relabelled_share_of_ok"] - n_kin / len(ok)) < 1e-12
    nolab = np.flatnonzero(kb.rl["slot"] < 0)
    assert len(nolab) > 0                                                                          # the off-route records of the bank
    bad = kb.rl["kin"].copy()
    bad[nolab[0]] = True
    with pytest.raises(ValueError):
        kb.attach_recovery(kb.rl["slot"], kb.rl["label"], None, kin=bad)                          # a kin flag on a record with no label is refused


def test_reachability_in_the_batch_kin_replaces_labels_the_counter_counts_them_and_speed_v1_counts_zero_kin(fx10):
    kb = kin_bank(fx10, plant=10**9, kappa=0.02, da=0.6)
    attach_kin(fx10, kb)
    sb = kin_bank(fx10, plant=10**9, kappa=0.02, da=0.6)
    res_s = RV.label_bank(sb.a, sb.ci, fx10.J["clip"], fx10.store.get_kin)
    sb.attach_recovery(res_s["slot"], res_s["label"], res_s["status_counts"])
    wins = eligible_windows_of(fx10, kb)
    b = make_batch(fx10, wins)
    swk, sws = G.DaggerSwap(kb, p=1.0, seed=3), G.DaggerSwap(sb, p=1.0, seed=3)
    ok_, os_ = swk.apply(b, 0), sws.apply(b, 0)
    n_diff = int((ok_["gt"] != os_["gt"]).reshape(len(wins), -1).any(1).sum())
    assert n_diff > 0 and swk.last["n_label_kin"] > 0 and swk.last["n_label_kin"] <= swk.last["n_label_replaced"] and sws.last["n_label_kin"] == 0
    assert n_diff <= swk.last["n_label_kin"], "only kin_v1 rows differ from the speed_v1 batch"
    ch = (ok_["gt"] != os_["gt"]).reshape(len(wins), -1).any(1)
    assert np.allclose(ok_["s_h"][ch].numpy(), RV.arc_from_label(ok_["gt"][ch].double().numpy()), atol=1e-4, rtol=0), "s_h re-derived from the kin_v1 waypoints"
    for k in b:                                                                                    # nothing else differs from the speed_v1 batch
        if k not in ("gt", "s_h"):
            assert _same(ok_[k], os_[k]), k
    item = G.swap_item(fx10.ds.item(wins[int(np.flatnonzero(ch.numpy())[0])][0], wins[int(np.flatnonzero(ch.numpy())[0])][1], 1, 3), kb.gather([int(swk.last["record_idx"][int(np.flatnonzero(ch.numpy())[0])])]))
    assert np.array_equal(item["gt"], ok_["gt"][int(np.flatnonzero(ch.numpy())[0])].numpy())


def test_reachability_through_the_trainer_argv_namespace_bank_swap_gt_and_the_plan_loss_gradient(pack):
    r = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1"])
    s = gate_parts_rl(pack, [*DAG(pack, 1.0, 3), "--dagger-label", "speed_v1"])
    h = gate_parts_rl(pack, DAG(pack, 1.0, 3))
    assert r[0].dagger_label == "kin_v1" and dvb(r[:9]) == [], dvb(r[:9])                            # argv -> namespace -> G-DVB passes with the label built
    assert r[5].bank.label_kind == "kin_v1" and r[5].built_levers()["dagger_label"] == "kin_v1" and s[5].built_levers()["dagger_label"] == "speed_v1"
    bad = dvb(r[:9], dag=s[5])                                                                       # declared kin_v1, built speed_v1: refused
    assert any("lever dagger_label" in m and "kin_v1" in m for m in bad), bad
    bad = dvb(s[:9], dag=r[5])
    assert any("lever dagger_label" in m for m in bad), bad
    assert any("dagger-label" in m and "declared and does nothing" in m for m in dvb(gate_parts_rl(pack, ["--dagger-label", "kin_v1"])[:9]))      # at p = 0
    ctl = r[4]["dagger"]["speed_label"]
    assert ctl["label"] == "kin_v1" and ctl["n_labelled"] > 0 and ctl["n_eligible_after"] == ctl["n_eligible"] and set(ctl["census"]) >= {"per_tick", "merge_time_s", "max_dev_m", "relabelled_share_of_ok"}
    a, cfg, _, _, info, dag_k, J, wins, clips, store = r
    dag_s, dag_h = s[5], h[5]
    # plant the own state of the bank's kin rows so the kin label differs from the speed label (the synthetic rollouts may sit on the reference): own kappa and a moved, written into BOTH banks' p10f (the same records)
    for dg in (dag_k.bank, dag_s.bank):
        dg.a["p10f"] = np.array(dg.a["p10f"])
    n_pl = 0
    ok_i = np.flatnonzero(dag_k.bank.reason == DG.OK)
    for n, i in enumerate(ok_i):
        for dg in (dag_k.bank, dag_s.bank):
            dg.a["p10f"][i, -1, 1] += np.float32(0.9 if n % 2 else -0.9)
            dg.a["p10f"][i, -1, 2] += np.float32(0.03 if n % 3 else -0.03)
        n_pl += 1
    res = RV.kin_label_bank(dag_k.bank.a, dag_k.bank.ci, J["clip"], getattr(store, "get_kin", None) or store.get)
    dag_k.bank.attach_recovery(res["slot"], res["label"], res["status_counts"], kin=res["is_kin"])
    res_s = RV.label_bank(dag_s.bank.a, dag_s.bank.ci, J["clip"], getattr(store, "get_kin", None) or store.get)
    dag_s.bank.attach_recovery(res_s["slot"], res_s["label"], res_s["status_counts"])
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    el = [w for w in wins if dag_k.bank.candidates([w[0]], [w[1]])[0][0] > 0][:24]
    b = D.collate([ds.item(w[0], w[1], 1, 3) for w in el])
    cfg16 = M.V8FMiniConfig(**{**TINY, "hmax": 16, "wm": "b"})
    mu, sd = M.cons_stats(J)
    model = _woken(M.V8FMini(cfg16, M.P10Prior.synthetic(0), mu, sd))
    out_k, out_s, out_s2, out_h = dag_k.apply(b, 1), dag_s.apply(b, 1), dag_s.apply(b, 1), dag_h.apply(b, 1)
    assert dag_k.last["n_label_kin"] > 0, f"kin rows in the swapped batch ({res['status_counts']})"
    assert dag_s.last["n_label_kin"] == 0 and dag_h.last["n_label_kin"] == 0 and dag_h.last["n_label_replaced"] == 0 and out_h["gt"] is b["gt"]
    nk = int((out_k["gt"] != out_s["gt"]).reshape(len(el), -1).any(1).sum())
    assert nk > 0 and nk <= dag_k.last["n_label_kin"]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def grads(bb):
        for _, p in named:
            p.grad = None
        torch.manual_seed(20261026)
        T = model.loss(bb, p_gt=0.5, gen=torch.Generator().manual_seed(13))["terms"]
        T["total"].backward()
        return float(T["ade"].detach()), {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
    dsum = lambda ga, gb: float(sum((ga[n] - gb[n]).abs().sum() for n in ga))      # noqa: E731
    ade_k, gk = grads(out_k)
    ade_s, gs = grads(out_s)
    ade_s2, gs2 = grads(out_s2)
    assert dsum(gs, gs2) == 0.0 and ade_s == ade_s2, "the control: speed_v1 against itself reads EXACTLY 0.0"
    assert ade_k != ade_s and dsum(gk, gs) > 0.0, "the kinematic label reaches the plan loss term (ade) and its gradient differs from speed_v1's"
    res_probe = TR.dagger_probe(model, b, cfg16, dag_k)                                              # the launch gate's own G-LIVE probe records the kin reach and the per-reason counters
    assert res_probe.get("rl_label_rows_changed", 0) > 0 and res_probe["n_label_kin"] > 0 and res_probe["kin_status_counts"] == dict(dag_k.bank.rl["status_counts"])


def test_the_training_loop_counts_the_kin_rows_in_the_log_and_the_summary(pack, tmp_path):
    out = tmp_path / "kin_run"
    assert TR.main(_run(out, [*DAG(pack, 1.0, 3), "--dagger-label", "kin_v1", "--steps", "3"])) == 0
    s = _summary(out)
    assert s["levers"]["dagger"]["dagger_label"] == "kin_v1" and s["dagger_label_replaced"] >= s["dagger_label_kin"] >= 0 and "dagger_label_kin" in s
    h = tmp_path / "hum_run"
    assert TR.main(_run(h, [*DAG(pack, 1.0, 3), "--steps", "3"])) == 0
    sh = _summary(h)
    assert "dagger_label_kin" not in sh and "dagger_label_replaced" not in sh                      # a human run keeps today's key set
