"""P3's gate-relevant target pre-build (`stack/scripts/prebuild_p3_targets.py`).

⛔ Every expectation is a LITERAL computed by hand, never an expression over the code under test —
a check that re-runs the producer's own derivation measures determinism, not correctness. Each one
was shown to go RED under mutation (`stack/scripts/mutate_prebuild_p3.py`).

The two guards that matter:
 * the population boundary must be the SAME one `box_quality` gates on, or the arm optimises a set
   the scorer never reads;
 * the grid cross-check must REFUSE to bank when the window grid disagrees with the literals a
   real run wrote — a target bank on the wrong grid is worse than no bank.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "taniteval"))
sys.path.insert(0, str(ROOT.parent / "taniteval" / "tools"))

import prebuild_p3_targets as PB                          # noqa: E402


# ------------------------------------------------------------------ the population
@pytest.mark.parametrize("x, y, inside", [
    (30.0, 5.0, True),
    (0.0, 0.0, True),        # the ego's own cell — the boundary is INCLUSIVE
    (60.0, 16.0, True),      # the far corner — inclusive
    (60.1, 0.0, False),
    (30.0, 16.1, False),
    (-0.1, 0.0, False),      # BEHIND the ego
    (30.0, -16.0, True),     # |y|, so the right side is symmetric
])
def test_the_gate_population_is_the_scorer_s_population(x, y, inside):
    assert bool(PB.gate_mask([x], [y])[0]) is inside


def test_the_boundary_literals_match_box_quality_exactly():
    """⛔ ONE spelling. If these drift apart the arm trains on a set the gate does not read."""
    import box_quality as BQ
    assert (PB.X_FWD_M, PB.Y_HALF_M) == (60.0, 16.0)
    for pt in [(59.0, 15.0), (61.0, 15.0), (59.0, 17.0), (-1.0, 0.0), (0.0, 0.0)]:
        assert bool(PB.gate_mask([pt[0]], [pt[1]])[0]) is \
            bool(BQ.population_mask([pt[0]], [pt[1]], "near_forward")[0])


# ------------------------------------------------------------------ the truncation rule
def test_nearest_n_keeps_the_CLOSEST_and_drops_the_rest():
    x, y = [1.0, 10.0, 3.0, 0.0], [0.0, 0.0, 4.0, 2.0]    # radii 1, 10, 5, 2
    assert PB.nearest_n(x, y, 2).tolist() == [0, 3]
    assert PB.nearest_n(x, y, 4).tolist() == [0, 3, 2, 1]
    assert PB.nearest_n(x, y, 99).tolist() == [0, 3, 2, 1], "n > len must keep everything"


# ------------------------------------------------------------------ the census arithmetic
class _Ep:
    def __init__(self, eid):
        self.episode_id = eid


class _Reader:
    """Three windows: 5 boxes (2 in the gate), a NO_LABEL, and a labelled-clear."""
    max_agents_per_frame, n_records = 5, 2

    def lookup(self, eid, f):
        if f == 100:
            return np.array([[10.0, 1.0], [20.0, 2.0], [-5.0, 0.0],
                             [70.0, 0.0], [30.0, 40.0]])
        if f == 300:
            return np.zeros((0, 2))
        return None                                       # f == 200 -> NO_LABEL


class _DS:
    index = [(0, 98), (0, 198), (0, 298)]


def test_the_census_counts_NO_LABEL_and_labelled_clear_DIFFERENTLY():
    out = PB.census(_DS(), [_Ep(7)], _Reader(), window=3, pad=32, queries=16)
    r = out["report"]
    assert r["n_windows"] == 3 and r["n_windows_labelled"] == 2, (
        "a NO_LABEL window is not a labelled-clear one — collapsing them trains the head "
        "that an unlabelled frame is an empty road")
    assert r["n_target_boxes_prefilter"] == 5
    # 360°: (5 + 0) / 2 labelled = 2.5 exactly; gate: only (10,1) and (20,2) = (2 + 0) / 2 = 1.0
    assert r["population_360"]["mean"] == 2.5 and r["population_360"]["total"] == 5
    assert r["population_gate_relevant"]["mean"] == 1.0
    assert r["population_gate_relevant"]["total"] == 2
    assert r["gate_share_of_boxes"] == 0.4          # 2 / 5 exactly
    assert r["population_gate_relevant"]["frac_windows_empty"] == 0.5   # the clear window


def test_the_pad_budget_is_COUNTED_not_assumed():
    out = PB.census(_DS(), [_Ep(7)], _Reader(), window=3, pad=3, queries=1)
    r = out["report"]
    # delivered 360 after pad 3: min(5, 3) = 3 and 0 -> mean 1.5 exactly
    assert r["delivered_360_after_pad"]["mean"] == 1.5
    assert r["frac_labelled_windows_over_pad_360"] == 0.5        # 1 of 2 labelled
    assert r["frac_labelled_windows_over_queries_360"] == 0.5    # 3 > 1 on one window
    assert r["frac_labelled_windows_over_queries_gate"] == 0.5   # 2 > 1 on the same window


def test_the_banked_selection_indexes_the_JOIN_S_OWN_ROWS():
    out = PB.census(_DS(), [_Ep(7)], _Reader(), window=3, pad=32, queries=16)
    off, idx = out["sel_off"], out["sel_idx"]
    assert off.tolist() == [0, 2, 2, 2], "offsets must cover EVERY window, labelled or not"
    assert sorted(idx.tolist()) == [0, 1], "rows 0 and 1 are the two gate-relevant boxes"


# ------------------------------------------------------------------ the refusal
REP = {"n_windows": 10600, "n_windows_labelled": 10217, "n_target_boxes_prefilter": 318099}


def test_a_matching_grid_is_OK():
    checks, bad, status = PB.grid_verdict(REP, dict(REP))
    assert status == "OK" and bad == []
    assert checks["n_windows"] == "MATCH 10600"


def test_ONE_mismatched_literal_is_INCONCLUSIVE_and_names_the_key():
    exp = dict(REP, n_target_boxes_prefilter=318100)      # one box out
    checks, bad, status = PB.grid_verdict(REP, exp)
    assert bad == ["n_target_boxes_prefilter"] and status.startswith("INCONCLUSIVE")
    assert "MISMATCH got 318099 want 318100" == checks["n_target_boxes_prefilter"]


def test_NO_literal_compared_is_UNVERIFIED_never_OK():
    """⛔ Omitting every expectation must NOT read as a verified grid — that is the
    absence-is-not-evidence rule in the one place where it would license a bad bank."""
    _c, bad, status = PB.grid_verdict(REP, {k: None for k in REP})
    assert bad == [] and status.startswith("UNVERIFIED")
    assert status != "OK"
