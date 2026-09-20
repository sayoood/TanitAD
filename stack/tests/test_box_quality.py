"""The P-arm scorer (`taniteval/tools/box_quality.py`) of PREREG_PERCEPTION_BOX_QUALITY.

⛔ Every expectation here is a LITERAL — a number computed by hand from the synthetic pairs — never
an expression over the code under test. Each check was also shown to go RED under mutation
(`stack/scripts/mutate_box_quality.py`), because a scorer that shares the defect it checks for is
green forever.

Nothing here is a result: the pairs are fabricated so the answers are known in advance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "taniteval" / "tools"))
# ⛔ AND the real `taniteval` PACKAGE, which lives one level deeper. Without this the
# episode-bootstrap case dies on `No module named 'taniteval.ci'` under a bare
# `pytest -q` — the outer `taniteval/` has no `__init__.py`, so it forms a NAMESPACE
# package that SHADOWS the regular one at `taniteval/taniteval/` (`__file__` is None).
# MEASURED 2026-09-20: 14 passed / 1 failed bare, 15 passed with this line. The tool
# itself is fine; it is the sibling convention (`from taniteval import ci`) that needs
# the inner directory on the path, and a test green only under an ambient PYTHONPATH
# is a guard that does not run.
sys.path.insert(0, str(ROOT.parent / "taniteval"))

import box_quality as BQ                                   # noqa: E402


def _w(pairs, vel=(), n_target=None):
    """One window record: pairs are (dx, dy, gx, gy); vel are (err, floor)."""
    return {"dx": [p[0] for p in pairs], "dy": [p[1] for p in pairs],
            "gx": [p[2] for p in pairs], "gy": [p[3] for p in pairs],
            "vel_err": [v[0] for v in vel], "vel_floor": [v[1] for v in vel],
            "n_target": len(pairs) if n_target is None else n_target,
            "n_matched": len(pairs), "n_dropped": 0}


# ------------------------------------------------------------------ the population definition
@pytest.mark.parametrize("gx, gy, near", [
    (30.0, 5.0, True),      # forward, inside the lane band
    (0.0, 0.0, True),       # the ego's own cell is the boundary, inclusive
    (60.0, 16.0, True),     # the far corner, inclusive
    (30.0, 20.0, False),    # too lateral
    (-5.0, 0.0, False),     # BEHIND the ego -- 49 % of this corpus's GT
    (70.0, 0.0, False),     # beyond the forward extent
])
def test_population_geometry_reads_KNOWN_points(gx, gy, near):
    assert bool(BQ.population_mask([gx], [gy], "near_forward")[0]) is near
    assert bool(BQ.population_mask([gx], [gy], "far_or_behind")[0]) is (not near)
    assert bool(BQ.population_mask([gx], [gy], "all_360")[0]) is True


def test_unknown_population_is_REFUSED():
    with pytest.raises(ValueError, match="unknown population"):
        BQ.population_mask([0.0], [0.0], "forward_only")


# ------------------------------------------------------------------ both populations, always
def test_BOTH_populations_are_always_emitted_and_the_primary_is_the_gate_s():
    rep = BQ.summarise([_w([(1.0, 1.0, 30.0, 0.0), (5.0, 5.0, -30.0, 0.0)])])
    assert set(BQ.POPULATIONS) <= set(rep) and "far_or_behind" in rep
    assert rep["_primary"] == "near_forward" and BQ.POPULATIONS[0] == "near_forward"
    # near-forward: one pair, L1 = 1 + 1 = 2.0 exactly; 360°: (2 + 10) / 2 = 6.0 exactly
    assert rep["near_forward"]["centre_l1_m"] == 2.0 and rep["near_forward"]["n"] == 1
    assert rep["all_360"]["centre_l1_m"] == 6.0 and rep["all_360"]["n"] == 2
    assert rep["far_or_behind"]["centre_l1_m"] == 10.0


def test_headline_REFUSES_one_population_without_the_other():
    rep = BQ.summarise([_w([(1.0, 1.0, 30.0, 0.0)])])
    assert "PRIMARY near-forward" in BQ.headline(rep) and "SECONDARY" in BQ.headline(rep)
    del rep["all_360"]
    with pytest.raises(ValueError, match="BOTH populations"):
        BQ.headline(rep)


def test_the_bar_is_read_at_2m_on_the_PRIMARY():
    assert BQ.summarise([_w([(0.6, 0.6, 10.0, 0.0)])])["near_forward"]["meets_bar"] is True
    assert BQ.summarise([_w([(1.2, 1.2, 10.0, 0.0)])])["near_forward"]["meets_bar"] is False


# ------------------------------------------------------------------ the P3 escape clause
def test_improving_ONLY_far_behind_must_NOT_move_the_near_forward_number():
    """⛔ PREREG_PERCEPTION_BOX_QUALITY §3.2's deliberate-regression control, in the scorer:
    far/behind supervision must not flatter the gate's population."""
    base = _w([(1.0, 1.0, 30.0, 0.0), (6.0, 6.0, -40.0, 0.0)])
    better_far = _w([(1.0, 1.0, 30.0, 0.0), (0.0, 0.0, -40.0, 0.0)])
    a, b = BQ.summarise([base]), BQ.summarise([better_far])
    assert a["near_forward"]["centre_l1_m"] == b["near_forward"]["centre_l1_m"] == 2.0
    assert a["all_360"]["centre_l1_m"] == 7.0 and b["all_360"]["centre_l1_m"] == 1.0


# ------------------------------------------------------------------ the controls
def test_the_matched_equals_target_IDENTITY_and_how_it_breaks():
    ok = BQ.summarise([_w([(1.0, 1.0, 5.0, 0.0)] * 3)])
    assert ok["controls"]["matched_equals_target"] is True
    assert ok["controls"]["n_target"] == 3 and ok["controls"]["n_matched"] == 3
    short = BQ.summarise([_w([(1.0, 1.0, 5.0, 0.0)] * 3, n_target=5)])
    assert short["controls"]["matched_equals_target"] is False, (
        "a matcher that pairs fewer than min(targets, queries) must BREAK this identity")


def test_velocity_is_read_against_the_ZERO_floor():
    rep = BQ.summarise([_w([(1.0, 1.0, 5.0, 0.0)], vel=[(1.0, 3.0), (2.0, 5.0)])])
    c = rep["controls"]
    assert c["vel_pairs"] == 2
    assert c["vel_mae_pred_mps"] == 1.5 and c["vel_mae_zero_floor_mps"] == 4.0
    assert c["vel_gain_mps"] == 2.5 and c["vel_beats_zero_floor"] is True
    worse = BQ.summarise([_w([(1.0, 1.0, 5.0, 0.0)], vel=[(6.0, 3.0)])])
    assert worse["controls"]["vel_beats_zero_floor"] is False, "FAIL-HARM must be visible"


# ------------------------------------------------------------------ through the REAL matcher
def test_match_pairs_uses_the_TRAINING_matcher_and_pairs_every_target():
    """An exact-match case: predictions placed ON the targets must read dx = dy = 0, and the
    Hungarian matcher must pair EVERY valid target (32 <= 100 queries)."""
    box = torch.tensor([[[12.0, 3.0, 4.0, 2.0], [-20.0, 1.0, 4.0, 2.0], [0.0, 0.0, 0.0, 0.0]]])
    valid = torch.tensor([[True, True, False]])
    # ⭐ rates too, so a defect INSIDE match_pairs' velocity path cannot hide: slot 0 (which
    # matches target 1) predicts (0, 0); slot 1 (target 0) predicts (3, 0) against GT (3, 4).
    p_rates = torch.zeros(1, 5, 3)
    p_rates[0, 1, :2] = torch.tensor([3.0, 0.0])
    t_rates = torch.zeros(1, 3, 3)
    t_rates[0, 0, :2] = torch.tensor([3.0, 4.0])          # |GT| = 5.0 exactly
    pred = {"box": torch.cat([box[:, [1, 0]], torch.zeros(1, 3, 4)], dim=1),
            "cls_logits": torch.zeros(1, 5, 3), "presence_logit": torch.zeros(1, 5),
            "rates": p_rates}
    tgt = {"box": box, "valid": valid, "cls": torch.zeros(1, 3, dtype=torch.long),
           "rates": t_rates, "rates_mask": torch.tensor([[True, True, False]])}
    p = BQ.match_pairs(pred, tgt)
    assert p["n_target"] == 2 and p["n_matched"] == 2 and p["n_dropped"] == 0
    assert max(p["dx"]) == 0.0 and max(p["dy"]) == 0.0, "an exact match must read zero error"
    assert sorted(p["gx"]) == [-20.0, 12.0]
    assert sorted(p["vel_floor"]) == [0.0, 5.0], "the floor is the GT speed, never the prediction"
    assert sorted(p["vel_err"]) == [0.0, 4.0]             # |(3,0) - (3,4)| = 4 exactly
    rep = BQ.summarise([p])
    assert rep["near_forward"]["n"] == 1 and rep["all_360"]["n"] == 2


def test_the_bootstrap_is_clustered_by_EPISODE():
    ws = [_w([(2.0, 2.0, 10.0, 0.0)]) for _ in range(6)]
    rep = BQ.summarise(ws, eid=["e0", "e0", "e1", "e1", "e2", "e2"], n_boot=200)
    ci = rep["near_forward"]["ci"]
    assert ci["n_episodes"] == 3 and ci["n_windows"] == 6
    assert ci["mean"] == pytest.approx(4.0)          # every pair is 2 + 2 = 4 exactly
    assert ci["lo"] == pytest.approx(4.0) and ci["hi"] == pytest.approx(4.0)
