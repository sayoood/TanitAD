"""test_combo_ens_retime_r8.py -- pins combo_ens_retime_r8.py (PREREG_COMBO.md).

    cd <PKG>/code && PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;.' python -X utf8 -m pytest -q -p no:cacheprovider test_combo_ens_retime_r8.py

Groups.  A: pure functions; every expectation is a LITERAL worked out by hand (never an expression over the code under test).  B: the estimator + the glue on a PLANTED fixture with the real
paired bootstrap: the baseline world must read STACKS; three DELIBERATE-REGRESSION arms must be caught -- (1) skip the re-time (COMBO := ENS) turns STACKS into NO-STACK and ST_ens into an exact 0,
(2) windows with no draw-to-draw noise (the ensemble can add nothing) read NO-STACK, (3) a wrong gain (2.0) reads HARMS; shuffled window pairing must change a cell.  C: the guards (pre-registration
hash, train-gains refusal) and the mutation hook.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in ("C:/lgs/Lfb47a58376/stack", "C:/lgs/Lfb47a58376/taniteval", str(HERE)):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)
import combo_ens_retime_r8 as C  # noqa: E402
import pair_read_r8 as PR  # noqa: E402
import a0k0_headroom_retime_r8 as RT  # noqa: E402


@pytest.fixture(autouse=True)
def _no_mutation(monkeypatch):
    monkeypatch.delenv("COMBO_TEST_MUTATION", raising=False)


# ======================================================================================================== A. pure functions (hand-worked literals)
O_HAND = np.array([[[3.0, 0.0], [3.0, 4.0], [3.0, 4.0], [3.0, 8.0]]], dtype=np.float32)
HP_HAND = np.array([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]], dtype=np.float32)       # prior chord arc length M = [1, 2, 3, 4]


def test_gain_zero_hand_worked_literal():
    # s' = M = [1,2,3,4]: s=1 -> (1,0); s=2 -> (2,0); s=3 -> the vertex (3,0); s=4 -> (3,1)
    got = RT.retime_apply(O_HAND, HP_HAND, np.zeros(4), cast=False)
    assert np.allclose(got[0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [3.0, 1.0]], atol=1e-12)
    assert np.allclose(C.walker_positions(O_HAND, HP_HAND)[0], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [3.0, 1.0]], atol=1e-12)


def test_gain_half_hand_worked_literal():
    # s' = M + 0.5 (L - M) with L = [3,7,7,11] -> [2, 4.5, 5, 7.5] -> (2,0), (3,1.5), (3,2), (3,4.5)
    got = RT.retime_apply(O_HAND, HP_HAND, np.full(4, 0.5), cast=False)
    assert np.allclose(got[0], [[2.0, 0.0], [3.0, 1.5], [3.0, 2.0], [3.0, 4.5]], atol=1e-12)


def test_gain_one_reproduces_the_plan_bit_for_bit_and_wrong_gain_is_caught():
    rng = np.random.default_rng(3)
    P = np.cumsum(np.abs(rng.normal(size=(50, 4, 2))) + np.array([1.0, 0.0]), axis=1).astype(np.float32)
    HP = (0.9 * P + 0.1).astype(np.float32)
    assert C.check_g1(P, HP)["PASS"] is True and C.check_g1(P, HP)["max_abs_m"] == 0.0
    assert C.check_g1(P, HP, g=1.001)["PASS"] is False                     # deliberate regression: a gain that is not 1 must NOT reproduce the plan


def test_gain_zero_check_matches_the_independent_walker_and_a_wrong_gain_is_caught():
    rng = np.random.default_rng(4)
    P = np.cumsum(np.abs(rng.normal(size=(40, 4, 2))) + np.array([1.0, 0.2]), axis=1).astype(np.float32)
    HP = (0.8 * P).astype(np.float32)
    assert C.check_g0(P, HP)["PASS"] is True
    off = RT.retime_apply(P, HP, np.full(4, 0.01), cast=False)
    assert float(np.abs(off - C.walker_positions(P, HP)).max()) > 1e-3     # deliberate regression: gain 0.01 is not gain 0


def test_ensemble_of_mirrored_draws_shortens_the_arc_length():
    # draws (3,+1),(3,-1) repeated at 4 instants: each polyline has arc length 1.0 (up) + 3 steps of 0 ... mean plan is the straight line (3,0)x4
    a = np.array([[[3.0, 1.0]] * 4], dtype=np.float32)
    b = np.array([[[3.0, -1.0]] * 4], dtype=np.float32)
    ens = C.KD.ensemble_mean(np.stack([a, b]))
    assert np.array_equal(ens, np.array([[[3.0, 0.0]] * 4], dtype=np.float32))
    L = lambda P: RT.polyline_geometry(P)["L"][:, -1]  # noqa: E731
    assert float(L(a)[0]) == pytest.approx(np.hypot(3.0, 1.0)) and float(L(ens)[0]) == 3.0 and L(ens)[0] < L(a)[0]    # 3.1623 > 3.0: averaging shortens the path


def _cell(delta, lo, hi):
    return {"delta": delta, "lo": lo, "hi": hi}


def test_stack_verdict_boundaries():
    f = 0.002                                        # 2 x max(floor) = 0.004
    ok = C.stack_verdict(_cell(-0.010, -0.015, -0.005), _cell(-0.008, -0.012, -0.001), f, f)
    assert ok["stacks"] is True and ok["worse_on_any_set"] is False
    assert C.stack_verdict(_cell(-0.010, -0.015, -0.005), _cell(-0.008, -0.012, 0.0), f, f)["stacks"] is False       # hi == 0 is NOT separated
    assert C.stack_verdict(_cell(-0.010, -0.015, -0.005), _cell(-0.004, -0.012, -0.001), f, f)["stacks"] is False    # |delta| == 2 x floor is NOT beyond it (strict)
    assert C.stack_verdict(_cell(-0.010, -0.015, -0.005), _cell(+0.003, +0.001, +0.006), f, f)["worse_on_any_set"] is True
    assert C.stack_verdict(_cell(-0.010, -0.015, -0.005), _cell(-0.008, -0.012, -0.001), 0.002, 0.005)["stacks"] is False   # the comparator's floor (x2 = 0.010) binds


def test_additivity_hand_worked_literals():
    a = C.additivity(0.27, 0.26, 0.25, 0.24)          # dENS -0.01, dRET -0.02, additive 0.24: the combo is exactly additive
    assert a["share_of_additive"] == pytest.approx(1.0) and a["additive_combo"] == pytest.approx(0.24)
    b = C.additivity(0.27, 0.26, 0.25, 0.25)          # combo no better than RET alone: ENS adds 0 on top, RET adds -0.01 = 0.5 of dRET on top of ENS, share 2/3
    assert b["ens_on_top_of_ret_m"] == pytest.approx(0.0) and b["ret_on_top_of_ens_share_of_dRET"] == pytest.approx(0.5) and b["share_of_additive"] == pytest.approx(2.0 / 3.0)


GOOD = {"st_ens": {"stacks": True, "worse_on_any_set": False}, "st_ret": {"stacks": True, "worse_on_any_set": False},
        "e1p": {"A": _cell(-0.03, -0.04, -0.02), "B": _cell(-0.03, -0.04, -0.02), "floor_combo": 0.001},
        "acc": {"S2": {"HOLDS": True, "moved": True}, "S6K2": {"HOLDS": True, "moved": True}},
        "nr": {"A": {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "PASS"}, "B": {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "PASS"}},
        "nr_ens": {"A": {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "PASS"}, "B": {"ADE": "PASS", "LONGITUDINAL": "PASS", "LATERAL": "PASS"}}}


def _ev(**kw):
    ev = {k: (dict(v) if isinstance(v, dict) else v) for k, v in GOOD.items()}
    ev.update(kw)
    return ev


def test_decide_table():
    assert C.decide(_ev())["label"] == "STACKS"
    # NO-STACK: one lever alone is as good (cell not separated)
    assert C.decide(_ev(st_ret={"stacks": False, "worse_on_any_set": False}))["label"] == "NO-STACK"
    assert C.decide(_ev(st_ens={"stacks": False, "worse_on_any_set": False}))["label"] == "NO-STACK"
    # NO-STACK: E1 not PASS (|delta| inside 2 x floor), vacuous ACC-LAT, S6 unavailable, NR not evaluated
    assert C.decide(_ev(e1p={"A": _cell(-0.0015, -0.004, -0.0001), "B": _cell(-0.03, -0.04, -0.02), "floor_combo": 0.001}))["label"] == "NO-STACK"
    assert C.decide(_ev(acc={"S2": {"HOLDS": True, "moved": False}, "S6K2": {"HOLDS": True, "moved": True}}))["label"] == "NO-STACK"
    assert C.decide(_ev(acc={"S2": {"HOLDS": True, "moved": True}, "S6K2": None}))["label"] == "NO-STACK"
    assert C.decide(_ev(nr=None, nr_ens=None))["label"] == "NO-STACK"
    # HARMS (precedence over everything): worse than a single lever, ACC-LAT fails, NR introduced by the re-time
    assert C.decide(_ev(st_ens={"stacks": False, "worse_on_any_set": True}))["label"] == "HARMS"
    assert C.decide(_ev(st_ret={"stacks": True, "worse_on_any_set": True}))["label"] == "HARMS"
    assert C.decide(_ev(acc={"S2": {"HOLDS": True, "moved": True}, "S6K2": {"HOLDS": False, "moved": True}}))["label"] == "HARMS"
    bad_nr = {"A": {"ADE": "PASS", "LONGITUDINAL": "FAIL: LON_speed_mae_mps", "LATERAL": "PASS"}, "B": GOOD["nr"]["B"]}
    assert C.decide(_ev(nr=bad_nr))["label"] == "HARMS"
    # the failure is not INTRODUCED by the re-time (ENS alone fails it too) -> not HARMS; but STACKS needs NR PASS in every family (PREREG 5.2) -> NO-STACK
    assert C.decide(_ev(nr=bad_nr, nr_ens=bad_nr))["label"] == "NO-STACK"


def test_mutation_hook_skips_the_retime_and_the_default_does_not(monkeypatch):
    P = np.array([[[3.0, 0.0], [3.0, 4.0], [3.0, 4.0], [3.0, 8.0]]], dtype=np.float32)
    HP = np.array([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]], dtype=np.float32)
    assert not np.array_equal(C.retime(P, HP, np.full(4, 0.5)), P)
    monkeypatch.setenv("COMBO_TEST_MUTATION", "skip_retime")
    assert np.array_equal(C.retime(P, HP, np.full(4, 0.5)), P)


# ======================================================================================================== B. planted fixture, the REAL paired bootstrap
T_X = np.array([0.5, 1.0, 1.5, 2.0])


def _world(noise: float, seed: int = 0, n_ep: int = 40, per_ep: int = 25):
    """GT = (v t, 0); prior = speed 0.95 v; draw s = speed f_s v with f_s = 1.15 + N(0, noise) per window.  The optimal RETIME gain is 0.25 (s' = v t (0.7125 + 0.25 f))."""
    rng = np.random.default_rng(seed)
    n = n_ep * per_ep
    eid = np.repeat(np.arange(n_ep), per_ep)
    v = rng.uniform(5.0, 15.0, size=n)
    xs = v[:, None] * T_X[None, :]
    G = np.stack([xs, np.zeros_like(xs)], -1).astype(np.float32)
    HP = np.stack([0.95 * xs, np.zeros_like(xs)], -1).astype(np.float32)
    os_ = {}
    for s in range(8):
        f = 1.15 + rng.normal(0.0, noise, size=n) if noise > 0 else np.full(n, 1.15)
        os_[s] = np.stack([f[:, None] * xs, np.zeros_like(xs)], -1).astype(np.float32)
    return {"eid": eid, "G": G, "HP": HP, "os": os_, "mask": np.ones(n, bool)}


def _read(w, gain: float = 0.25):
    G, HP, eid = w["G"], w["HP"], w["eid"]
    ade = lambda P: np.linalg.norm(np.asarray(P, np.float64) - G, axis=-1).mean(1)  # noqa: E731
    cell = lambda a, b: PR.paired_cell(a, b, eid, 400, 0)  # noqa: E731
    W = np.full(4, gain)
    core = {n: C.stacking_core(w["os"], ss, HP, W, ade, cell) for n, ss in (("A", (0, 1, 2, 3)), ("B", (4, 5, 6, 7)))}
    fl = {k: C.floor_of(core["A"][f"ade_{k}"].mean(), core["B"][f"ade_{k}"].mean()) for k in ("combo", "ens", "ret")}
    st_ens = C.stack_verdict(core["A"]["ST_ens"], core["B"]["ST_ens"], fl["combo"], fl["ens"])
    st_ret = C.stack_verdict(core["A"]["ST_ret"], core["B"]["ST_ret"], fl["combo"], fl["ret"])
    e1p = {"A": cell(core["A"]["ade_combo"], ade(HP)), "B": cell(core["B"]["ade_combo"], ade(HP)), "floor_combo": fl["combo"]}
    ev = {"st_ens": st_ens, "st_ret": st_ret, "e1p": e1p, "acc": GOOD["acc"], "nr": GOOD["nr"], "nr_ens": GOOD["nr_ens"]}
    return C.decide(ev), core, st_ens, st_ret


def test_planted_stack_world_reads_STACKS():
    lab, core, st_ens, st_ret = _read(_world(noise=0.15))
    assert lab["label"] == "STACKS", (lab, st_ens, st_ret)
    # sanity on the planted effect (not a tautology of the code: hand arithmetic) -- ENS alone is biased by 0.15 v t, COMBO removes it; the ensemble halves the noise RET-alone carries
    assert core["A"]["ST_ens"]["delta"] < -1.0 and core["A"]["ST_ret"]["delta"] < 0


def test_DELIBERATE_REGRESSION_skipping_the_retime_is_caught(monkeypatch):
    monkeypatch.setenv("COMBO_TEST_MUTATION", "skip_retime")
    lab, core, st_ens, st_ret = _read(_world(noise=0.15))
    assert lab["label"] != "STACKS"
    assert core["A"]["ST_ens"]["delta"] == 0.0 and core["B"]["ST_ens"]["delta"] == 0.0     # COMBO == ENS exactly: the stacking assertion sees a zero, not a lever
    assert st_ens["stacks"] is False


def test_DELIBERATE_REGRESSION_no_draw_noise_means_the_ensemble_adds_nothing():
    lab, core, st_ens, st_ret = _read(_world(noise=0.0))
    assert core["A"]["ST_ret"]["delta"] == 0.0                                    # all draws identical -> COMBO == RET-alone, exactly
    assert lab["label"] == "NO-STACK" and st_ret["stacks"] is False


def test_DELIBERATE_REGRESSION_a_wrong_gain_reads_HARMS():
    lab, core, st_ens, st_ret = _read(_world(noise=0.15), gain=2.0)               # gain 2 over-runs the plan: worse than the unmodified ensemble
    assert lab["label"] == "HARMS" and st_ens["worse_on_any_set"] is True


def test_shuffled_window_pairing_changes_the_cell():
    w = _world(noise=0.15)
    G, HP, eid = w["G"], w["HP"], w["eid"]
    ade = lambda P: np.linalg.norm(np.asarray(P, np.float64) - G, axis=-1).mean(1)  # noqa: E731
    a = ade(C.KD.ensemble_mean(np.stack([w["os"][s] for s in range(4)])))
    b = ade(HP)
    base = PR.paired_cell(a, b, eid, 400, 0)
    shuf = PR.paired_cell(a, np.random.default_rng(0).permutation(b), eid, 400, 0)
    assert abs(base["delta"] - shuf["delta"]) > 0 or abs(base["lo"] - shuf["lo"]) > 1e-9
    assert (base["lo"], base["hi"]) != (shuf["lo"], shuf["hi"])


# ======================================================================================================== C. guards
def test_prereg_hash_is_pinned_and_a_mismatch_refuses(monkeypatch):
    assert C.check_prereg() == C.PREREG_SHA256
    monkeypatch.setattr(C, "PREREG_SHA256", "0" * 64)
    with pytest.raises(SystemExit):
        C.check_prereg()


def test_train_gains_are_refused_when_absent_or_not_train(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "TRAIN_GAINS", tmp_path / "FROZEN_GAINS_RETIME_train.json")
    with pytest.raises(SystemExit):
        C.load_gains("train")                                                     # absent file: fail loud, never substitute the eval gains
    (tmp_path / "FROZEN_GAINS_RETIME_train.json").write_text('{"fit_source": "eval", "fit_guard": {"n_overlap_with_eval139": 5}}')
    (tmp_path / "FROZEN_GAINS_RETIME_train.sha256").write_text(C.sha256_file(tmp_path / "FROZEN_GAINS_RETIME_train.json") + "\n")
    with pytest.raises(SystemExit):
        C.load_gains("train")                                                     # a file that is not train-fitted: refused


def test_eval_gains_file_is_the_registered_one():
    fz, meta = C.load_gains("eval")
    assert meta["sha256"] == "4ecde275554b1aaae84a4328294f791f28426f451832f9933fa34859eedcda4c"
    assert [round(x, 4) for x in fz["surfaces"]["S2"]["all"]["RETIME"]] == [0.9968, 0.5834, 0.4260, 0.3825]
