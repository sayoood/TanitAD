"""The S1 verdict code (``taniteval/tools/s1_pass.analyze``) on SYNTHETIC rows whose answers are
computed by hand -- so the code that will read a real pass is proven before the pass is paid for.
Nothing here is an S1 result."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT.parent / "taniteval" / "tools", ROOT.parent / "taniteval"):
    sys.path.insert(0, str(p))

s1_pass = pytest.importorskip("s1_pass")

FAM = {"along_2s": 0.5, "cross_2s": 0.2, "along_4s": 1.0, "cross_4s": 0.4,
       "speed_err_4s": 0.3, "heading_err_4s": 0.05, "curv_err_2s": None,
       "curv_floor_2s": None, "tac_lat_agree": 1.0, "tac_lon_agree": 1.0,
       "dac": 1.0, "ep": 0.9, "ttc": 1.0, "comfort": 1.0, "pdms": 0.9, "nc_half": 0.0}


def _arm(idx, collided):
    d = dict(FAM, idx=idx, collided=float(collided), nc=0.0 if collided else 1.0,
             fell_back=False)
    return d


def _rows(n=40, eps=8, box_good=True, break_base=False, spread=True):
    """``spread``: episodes are blocks of 5 consecutive windows, so PRED's five fixes
    (w = 4, 12, 20, 28, 36) land in FIVE different episodes. ``spread=False`` puts every
    fix in ONE episode (w % 8 == 4), which the episode bootstrap must NOT separate."""
    rows = []
    for w in range(n):
        base_col = w % 4 == 0                      # 10 of 40 collide
        pred_col = base_col and w % 8 == 0         # PRED fixes half of them -> 5 of 40
        arms = {"BASE": _arm(3, base_col), "GATE_CONST": _arm(3, base_col),
                "GATE_ORACLE": _arm(7 if base_col else 3, False),
                "GATE_PRED": _arm(7 if (base_col and not pred_col) else 3, pred_col),
                "ORACLE_CV": _arm(3, base_col),
                "RANDOM": dict(FAM, collided=0.5, n_candidates=128, nc=0.5),
                "_fan": {"n": 128, "collision_free_share": 0.5, "any_free": True, "human": {}}}
        conf = [0.9, 0.8, 0.3, 0.2]
        hits = [1, 1, 0, 0] if box_good else [0, 1, 0, 1]
        rows.append({"sha12": "w%03d" % w, "t0": 10,
                     "ep": "e%d" % (w // (n // eps) if spread else w % eps), "nav": 0,
                     "model_sel_idx": 4 if (break_base and w == 5) else 3, "arms": arms,
                     "box": {"labelled": True, "rows": [[c, h] for c, h in zip(conf, hits)],
                             "n_gt": 2, "n_pred": 4,
                             "vel_err": [0.2, 0.3] if box_good else [2.0, 2.0],
                             "vel_floor": [2.0, 2.5] if box_good else [0.5, 0.5]}})
    return rows


def test_SUPPORTED_with_hand_computed_rates_and_share():
    res = s1_pass.analyze(_rows(), (60.0, 16.0), n_boot=400)
    cr = res["collided_rate"]
    assert cr["BASE"] == pytest.approx(0.25) and cr["GATE_ORACLE"] == 0.0
    assert cr["GATE_PRED"] == pytest.approx(0.125) and cr["RANDOM"] == pytest.approx(0.5)
    assert res["share_of_ceiling"]["mean"] == pytest.approx(0.5)      # (0.125-0.25)/(0-0.25)
    assert res["controls"]["base_equals_model_pick"] and res["controls"]["const_recovery_exactly_0"]
    assert res["controls"]["oracle_changed_a_pick"]
    assert res["box_read"]["ap_pass"] and res["box_read"]["vel_pass"]
    assert res["box_read"]["ap_bev"] == pytest.approx(1.0)              # both hits ranked first
    assert res["verdict"].startswith("SUPPORTED")
    assert any("anchor_acc 0.092 vs chance 1/128" in t for t in res["verdict_text"])
    assert any("nav = v1" in t for t in res["verdict_text"])


def test_an_effect_confined_to_ONE_episode_is_NOT_separated():
    """MEASURED while writing this file: the first layout put all five fixes in one episode,
    and the verdict came back REFUTED with CI [-0.375, 0.0]. That is the estimator doing its
    job -- one cluster's luck is not an effect -- so it is pinned rather than 'fixed'."""
    res = s1_pass.analyze(_rows(spread=False), (60.0, 16.0), n_boot=400)
    assert res["collided_rate"]["GATE_PRED"] == pytest.approx(0.125)      # same rate...
    assert not res["paired_vs_BASE"]["GATE_PRED"]["separated"]            # ...not separated
    assert res["verdict"].startswith("REFUTED")


def test_VOID_when_BASE_is_not_the_model_pick():
    res = s1_pass.analyze(_rows(break_base=True), (60.0, 16.0), n_boot=200)
    assert res["verdict"].startswith("VOID")


def test_PRED_NOT_QUOTABLE_when_the_box_read_fails():
    res = s1_pass.analyze(_rows(box_good=False), (60.0, 16.0), n_boot=200)
    assert not res["box_read"]["vel_pass"]
    assert res["verdict"].startswith("PRED NOT QUOTABLE")
