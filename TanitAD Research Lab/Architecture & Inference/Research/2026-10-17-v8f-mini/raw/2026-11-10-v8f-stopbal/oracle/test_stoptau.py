"""test_stoptau.py -- pure-logic tests of stoptau_read.py (no torch, no model).  Run: python -m pytest -q test_stoptau.py  (from this folder, PYTHONPATH unset)."""
import inspect
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stoptau_read as T  # noqa: E402
import stophard_read as SH  # noqa: E402

RNG = np.random.default_rng(0)


def make_own(n=400):
    own = RNG.random((n, T.CODE_DIM)).astype(np.float32)
    lon = (RNG.random((n, T.N_LON)) ** 6).astype(np.float32) + np.float32(1e-3)          # peaked, so p(STOP) spans the grid
    own[:, T.LON_SL] = lon / lon.sum(1, keepdims=True)
    return own


def test_card_hash_unchanged():
    assert T.sha256(T.CARD) == T.CARD_SHA
    side = open(os.path.join(HERE, "PREREAD_V8F_STOPTAU.sha256"), encoding="utf-8").read()
    assert T.CARD_SHA in side


def test_grid():
    assert T.TAU_GRID == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


# ---- control 3: a planted probability vector gives the planted tau
def planted(n_neg=100):
    """negatives whose goal probability puts exactly these false positives at the grid values: 0.5:20 0.55:12 0.6:5 0.65:3 0.7:3 0.75:1 0.8..0.95:0"""
    want = {0.5: 20, 0.55: 12, 0.6: 5, 0.65: 3, 0.7: 3, 0.75: 1, 0.8: 0, 0.85: 0, 0.9: 0, 0.95: 0}
    p = np.zeros(n_neg, np.float32)
    vals = ([0.5] * 8) + ([0.55] * 7) + ([0.6] * 2) + ([0.65] * 0) + ([0.7] * 2) + ([0.75] * 1)          # exactly on grid values: >= must include them
    p[:len(vals)] = np.float32(vals)
    own = np.zeros((n_neg, T.CODE_DIM), np.float32)
    own[:, T.STOP_GI] = p
    return own, want


def counts_of(own, kind="goal"):
    f = T.fires_goal if kind == "goal" else T.fires_lon
    return {t: int(f(own, t).sum()) for t in T.TAU_GRID}


def test_planted_counts_are_what_we_planted():
    own, want = planted()
    assert counts_of(own) == want


def test_choose_tau_planted():
    own, want = planted()
    c = counts_of(own)
    assert T.choose_tau(c, ref_count=5, n_neg=100) == (0.6, False)           # smallest grid value with count <= 5
    assert T.choose_tau(c, ref_count=20, n_neg=100) == (0.5, False)          # reference above everything: the lowest value
    assert T.choose_tau(c, ref_count=3, n_neg=100) == (0.65, False)
    assert T.choose_tau(c, ref_count=0, n_neg=100) == (0.8, False)
    assert T.choose_tau(c, ref_count=-1, n_neg=100) == (0.95, True)          # none meets it: fallback, reported


def test_choose_tau_red_arms():
    own, _ = planted()
    c = counts_of(own)

    def strict(counts, ref):          # mutant: '<' instead of '<='
        return next((t for t in T.TAU_GRID if counts[t] < ref), 0.95)

    def largest(counts, ref):         # mutant: largest grid value meeting the rule
        return max((t for t in T.TAU_GRID if counts[t] <= ref), default=0.95)

    def first_above(counts, ref):     # mutant: scans from the top
        return min((t for t in reversed(T.TAU_GRID) if counts[t] > ref), default=0.95)
    assert strict(c, 5) != 0.6 and largest(c, 5) != 0.6 and first_above(c, 5) != 0.6        # each mutant reads a different tau: the planted check detects them
    assert T.choose_tau(c, 5, 100)[0] == 0.6


def test_float32_boundary():
    own = np.zeros((3, T.CODE_DIM), np.float32)
    own[:, T.STOP_GI] = np.float32([0.55, np.nextafter(np.float32(0.55), np.float32(0)), np.nextafter(np.float32(0.55), np.float32(1))])
    assert T.fires_goal(own, 0.55).tolist() == [True, False, True]


# ---- control 1: tau at 0.5 / argmax is STOPHARD's transform
def test_replay_equals_stophard_goal_and_lon():
    own = make_own()
    own[:50, T.STOP_GI] = 0.5                                                  # exact edge
    a = T.tau_code(own, "goal", 0.5)
    b = SH.hard_code(own, "hgoal")
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    a = T.tau_code(own, "lon", None, None)
    b = SH.hard_code(own, "hlon")
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert T.fires_lon(own, None).sum() > 0


def test_replay_red_arm_other_tau_differs():
    own = make_own()
    own[:50, T.STOP_GI] = 0.52
    assert not np.array_equal(T.tau_code(own, "goal", 0.55).view(np.uint32), SH.hard_code(own, "hgoal").view(np.uint32))
    assert not np.array_equal(T.tau_code(own, "lon", None, 0.9).view(np.uint32), SH.hard_code(own, "hlon").view(np.uint32))


def test_lon_tau_is_probability_of_stop_not_argmax():
    own = np.zeros((2, T.CODE_DIM), np.float32)
    own[0, T.LON_SL] = np.float32([0.1, 0.1, 0.4, 0.1, 0.1, 0.1, 0.1])        # argmax STOP, p 0.4
    own[1, T.LON_SL] = np.float32([0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0])        # argmax STOP, p 0.7
    assert T.fires_lon(own, None).tolist() == [True, True]
    assert T.fires_lon(own, 0.5).tolist() == [False, True]
    c = T.tau_code(own, "lon", None, 0.5)
    assert np.array_equal(c[0], own[0]) and c[1, T.LON_SL].tolist() == [0, 0, 1, 0, 0, 0, 0]


def test_check_tau_passes_and_red_arms():
    own = make_own()
    for mode, gt, lt in (("goal", 0.6, None), ("lon", None, 0.6), ("goal", 0.5, None), ("lon", None, None)):
        code = T.tau_code(own, mode, gt, lt)
        cnt, pr = T.check_tau(own, code, mode, gt, lt)
        assert pr == [], pr
    code = T.tau_code(own, "goal", 0.6)
    # red arm 1: touches a lat entry
    bad = code.copy(); bad[3, T.N_G + 1] += 1.0
    assert T.check_tau(own, bad, "goal", 0.6)[1]
    # red arm 2: fires everywhere
    bad = own.copy(); bad[:, T.STOP_GI] = 1.0
    assert T.check_tau(own, bad, "goal", 0.6)[1]
    # red arm 3: transform at the wrong tau is detected when the checker is told the card's tau
    assert T.check_tau(own, T.tau_code(own, "goal", 0.5), "goal", 0.9)[1]
    # red arm 4: lon part not one-hot
    c = T.tau_code(own, "lon", None, 0.5)
    k = np.where(T.fires_lon(own, 0.5))[0][0]
    c[k, T.LON_SL] = 0.5
    assert T.check_tau(own, c, "lon", None, 0.5)[1]
    # red arm 5: goal entry written in lon mode
    c = T.tau_code(own, "lon", None, 0.5); c[k, T.STOP_GI] = 1.0
    assert T.check_tau(own, c, "lon", None, 0.5)[1]


def test_transform_takes_no_label_and_does_not_mutate():
    for f in (T.tau_code, T.fires_goal, T.fires_lon, T.choose_tau):
        names = list(inspect.signature(f).parameters)
        assert not any(w in n for n in names for w in ("gt", "label", "lon_al", "allowed", "stop_hit")), (f.__name__, names)
    own = make_own()
    before = own.copy()
    T.tau_code(own, "goal", 0.7); T.tau_code(own, "lon", None, 0.7)
    assert np.array_equal(own, before)


def test_higher_tau_fires_subset():
    own = make_own()
    prev = np.ones(len(own), bool)
    for t in T.TAU_GRID:
        f = T.fires_goal(own, t)
        assert not (f & ~prev).any()
        prev = f


# ---- control 2: disjoint clips
def test_disjoint_detects_planted_overlap():
    assert T.clips_disjoint(["a", "b"], ["c"]) == []
    assert T.clips_disjoint(["a", "b"], ["b", "c"]) == ["b"]


def test_train141_disjoint_from_eval141_on_the_real_files():
    meta = json.load(open(T.CLIPS_JSON, encoding="utf-8"))
    A = SH.SO  # noqa: F841
    R = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
    if not os.path.exists(R):
        pytest.skip("eval join not present")
    ev = [str(c) for c in np.load(R)["clip"].tolist()]
    assert len(meta["clips"]) == 141 and len(ev) == 141
    assert T.clips_disjoint(meta["clips"], ev) == []


def test_eval_window_rule_equals_eval_grid_json():
    R = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/"
    if not os.path.exists(R + "eval_grid_windows.json"):
        pytest.skip("eval grid not present")
    z = np.load(R + "join_eval141.npz")
    g = json.load(open(R + "eval_grid_windows.json", encoding="utf-8"))
    ci = {c: i for i, c in enumerate(z["clip"].tolist())}
    assert all(g["windows"][c] == T.eval_rule_windows(int(z["n"][ci[c]])) for c in g["order"])
    assert T.eval_rule_windows(101) == list(range(3, 71, 4))


# ---- outcome table
@pytest.mark.parametrize("sg,ac,label", [
    ((True, True), (False, False), "TAU-HELPS"),
    ((True, True), (True, False), "TAU-TRADES"),
    ((True, True), (True, True), "TAU-TRADES"),
    ((False, False), (False, False), "TAU-NULL"),
    ((False, False), (True, False), "TAU-TRADES"),
    ((True, False), (False, False), "SEEDS-DISAGREE"),
    ((False, True), (False, False), "SEEDS-DISAGREE"),
])
def test_outcome_table(sg, ac, label):
    ps = {"s0": {"S_gain": sg[0], "A_cost": ac[0]}, "s1": {"S_gain": sg[1], "A_cost": ac[1]}}
    assert T.outcome_lever(ps)[0] == label


def test_outcome_not_evaluated_and_row_list():
    assert T.outcome_lever({"s0": {"S_gain": True, "A_cost": False}})[0] == "NOT-EVALUATED"
    lab, rows = T.outcome_lever({"s0": {"S_gain": False, "A_cost": True}, "s1": {"S_gain": False, "A_cost": False}})
    assert lab == "TAU-TRADES" and rows == ["TAU-TRADES", "TAU-NULL"]


def test_gain_cost_rules_reused_from_stophard():
    d = {"separated": True, "point": -0.05}
    assert T.SO.is_gain(d, 0.02) and not T.SO.is_gain(d, 0.06) and not T.SO.is_gain(d, None)
    d = {"separated": True, "point": 0.01}
    assert SH.is_cost(d, 0.001) and not SH.is_cost(d, 0.02) and not SH.is_cost(d, None)
