"""Unit tests for stopperc_s0.py pure logic (no torch, no model).  Run: python -m pytest -q test_stopperc_s0.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stopperc_s0 as S  # noqa: E402


def test_card_hash_unchanged():
    assert S.sha256(S.CARD) == S.CARD_SHA
    assert open(os.path.join(os.path.dirname(S.CARD), "PREREG_V8F_STOPPERC.sha256")).read().split()[0] == S.CARD_SHA


def test_k_of_card_values():
    assert [S.k_of(c) for c in S.C_GRID] == [35, 59, 82, 117]
    assert S.k_of(0.0) == 0


def test_rank_select_top_k_ties_and_mask():
    p = np.array([0.9, 0.2, 0.5, 0.5, 0.95, 0.1])
    sp = np.array([True, True, True, True, False, True])        # window 4 has the top p but is outside the set
    sel = S.rank_select(p, sp, 2)
    assert sel.tolist() == [True, False, True, False, False, False]   # 0.9, then the tie 0.5 broken by window order (2 before 3)
    assert not (S.rank_select(p, sp, 5) & ~sp).any()
    assert S.rank_select(p, sp, 0).sum() == 0
    assert S.rank_select(p, sp, 5).sum() == 5


def test_rank_select_red_arm_k_above_set_refuses():
    try:
        S.rank_select(np.zeros(3), np.array([True, False, True]), 3)
    except AssertionError:
        return
    raise AssertionError("k above the set size must refuse")


def test_mix_plans_exact():
    own = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    fgt = -own
    sel = np.array([False, True, False, True])
    m = S.mix_plans(own, fgt, sel)
    assert np.array_equal(m[sel], fgt[sel]) and np.array_equal(m[~sel], own[~sel])
    assert np.array_equal(S.mix_plans(own, fgt, np.zeros(4, bool)), own)
    assert own[1, 0, 0] == 6.0                                       # input not modified


def test_clears_rule():
    f = S.P_FLOOR
    assert S.clears({"separated": True, "point": -(f + 0.01)})
    assert not S.clears({"separated": True, "point": -(f - 0.01)})   # below the floor
    assert not S.clears({"separated": False, "point": -2.0})         # not separated
    assert not S.clears({"separated": True, "point": +2.0})          # worse side


def test_ceiling_outcome_table():
    def tab(d):
        return {c: {"BAL-NULL_s0": v[0], "BAL-NULL_s1": v[1]} for c, v in d.items()}
    far = S.ceiling_outcome(tab({0.3: (0, 0), 0.5: (0, 0), 0.7: (1, 0), 1.0: (1, 1)}))
    assert far["outcome"] == "CEILING-FAR" and far["c_star"] == 1.0
    near = S.ceiling_outcome(tab({0.3: (0, 0), 0.5: (1, 1), 0.7: (1, 1), 1.0: (1, 1)}))
    assert near["outcome"] == "CEILING-NEAR" and near["c_star"] == 0.5
    edge = S.ceiling_outcome(tab({0.3: (0, 0), 0.5: (0, 0), 0.7: (1, 1), 1.0: (1, 1)}))
    assert edge["outcome"] == "CEILING-NEAR" and edge["c_star"] == 0.7
    none = S.ceiling_outcome(tab({0.3: (0, 0), 0.5: (0, 0), 0.7: (0, 0), 1.0: (0, 0)}))
    assert none["outcome"] == "CEILING-FAR" and none["c_star"] is None


def test_amendment_hash_pinned():
    assert S.sha256(S.AMEND) == S.AMEND_SHA == "6c4ad83be735519385d0d42068b7d6fe9dd2b69c4b9a4c06e633a7a4e146d123"
