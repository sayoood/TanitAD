"""CPU tests for the LF0 decoded-BEV lead read-off.

The failure these exist to prevent: a broken reader and an empty latent produce
the IDENTICAL null, so a silent geometry bug would read as "the world model has
no lead distance" — the same shape as the P1 instrument bug, where a class-filter
defect looked like a model verdict until it was fixed (and the failure survived).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from lf0_bev_lead import (CORRIDOR_M, GATE_R2, TAU_P8, corridor_cols,  # noqa: E402
                          r2, read_lead_range, score_arm, spearman)

CELL = 0.5
NX, NY = 120, 64
YHALF = 16.0


# --------------------------------------------------------------------------- #
# corridor geometry — the half that can silently be wrong                      #
# --------------------------------------------------------------------------- #
def test_corridor_is_centred_on_the_ego_axis():
    """+y is LEFT and col 0 is the RIGHT edge; a corridor that is not centred
    reads the wrong lane and every downstream number is about other traffic."""
    c = corridor_cols(NY, YHALF, CELL, 1.5)
    yc = -YHALF + (c + 0.5) * CELL
    assert np.all(np.abs(yc) <= 1.5)
    assert abs(yc.mean()) < 1e-9, "corridor must be symmetric about y=0"


def test_wider_corridor_is_a_superset():
    a = set(corridor_cols(NY, YHALF, CELL, 1.0).tolist())
    b = set(corridor_cols(NY, YHALF, CELL, 2.0).tolist())
    assert a < b


def test_corridor_widths_have_expected_cell_counts():
    # ±1.5 m at 0.5 m cells = 6 cells; ±1.0 = 4; ±2.0 = 8
    for w, expect in ((1.0, 4), (1.5, 6), (2.0, 8)):
        assert len(corridor_cols(NY, YHALF, CELL, w)) == expect


# --------------------------------------------------------------------------- #
# the reader                                                                   #
# --------------------------------------------------------------------------- #
def _raster_with_agent_at(row: int, col: int) -> np.ndarray:
    r = np.zeros((NX, NY), np.float32)
    r[row, col] = 1.0
    return r


def test_reads_the_range_of_the_nearest_occupied_cell():
    cols = corridor_cols(NY, YHALF, CELL, 1.5)
    r = _raster_with_agent_at(40, int(cols[2]))
    got = read_lead_range(r, tau=TAU_P8, cols=cols, cell_m=CELL, min_row=0)
    assert got == pytest.approx((40 + 0.5) * CELL)      # 20.25 m


def test_nearest_wins_when_two_agents_are_in_the_corridor():
    cols = corridor_cols(NY, YHALF, CELL, 1.5)
    r = _raster_with_agent_at(80, int(cols[0]))
    r[30, int(cols[3])] = 1.0
    got = read_lead_range(r, tau=TAU_P8, cols=cols, cell_m=CELL, min_row=0)
    assert got == pytest.approx((30 + 0.5) * CELL)


def test_agent_outside_the_corridor_is_not_a_lead():
    """A car in the next lane is not the lead vehicle. Reading it would make
    every overtake look like a close following distance."""
    cols = corridor_cols(NY, YHALF, CELL, 1.5)
    outside = int(cols[-1]) + 6                         # ~3 m to the left
    got = read_lead_range(_raster_with_agent_at(40, outside), tau=TAU_P8,
                          cols=cols, cell_m=CELL, min_row=0)
    assert np.isnan(got)


def test_empty_corridor_is_nan_never_max_range():
    """THE censoring rule: coding an empty corridor as 60 m manufactures
    correlation out of missing data."""
    cols = corridor_cols(NY, YHALF, CELL, 1.5)
    got = read_lead_range(np.zeros((NX, NY), np.float32), tau=TAU_P8,
                          cols=cols, cell_m=CELL, min_row=0)
    assert np.isnan(got), "must be censored, not max-range"


def test_min_row_skips_the_ego_footprint():
    cols = corridor_cols(NY, YHALF, CELL, 1.5)
    r = _raster_with_agent_at(1, int(cols[0]))
    assert np.isnan(read_lead_range(r, tau=TAU_P8, cols=cols, cell_m=CELL,
                                    min_row=2))
    assert not np.isnan(read_lead_range(r, tau=TAU_P8, cols=cols, cell_m=CELL,
                                        min_row=0))


def test_tau_is_a_threshold_not_a_maximum():
    cols = corridor_cols(NY, YHALF, CELL, 1.5)
    r = np.zeros((NX, NY), np.float32)
    r[40, int(cols[0])] = 0.5                            # below tau 0.7
    r[60, int(cols[0])] = 0.9                            # above
    got = read_lead_range(r, tau=TAU_P8, cols=cols, cell_m=CELL, min_row=0)
    assert got == pytest.approx((60 + 0.5) * CELL)


def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expected \[nx, ny\]"):
        read_lead_range(np.zeros((4,)), tau=0.5, cols=np.array([0]),
                        cell_m=CELL)


# --------------------------------------------------------------------------- #
# scoring + censoring discipline                                               #
# --------------------------------------------------------------------------- #
def test_perfect_read_scores_r2_one():
    t = np.linspace(5, 50, 40)
    s = score_arm(t.copy(), t)
    assert s["status"] == "OK" and s["r2"] == pytest.approx(1.0)
    assert s["n_paired"] == 40 and s["censored_rate_on_labelled"] == 0.0


def test_censored_windows_are_excluded_and_counted_not_imputed():
    t = np.linspace(5, 50, 40)
    read = t.copy()
    read[:10] = np.nan
    s = score_arm(read, t)
    assert s["n_paired"] == 30
    assert s["censored_rate_on_labelled"] == pytest.approx(0.25)
    assert s["r2"] == pytest.approx(1.0), "censoring must not distort the rest"


def test_too_few_pairs_reports_unavailable_with_the_reason():
    t = np.full(40, np.nan)
    t[:5] = np.linspace(5, 20, 5)
    s = score_arm(t.copy(), t)
    assert s["status"] == "UNAVAILABLE"
    assert "n=10 floor" in s["reason"] and "WORK ITEM" in s["reason"]


def test_an_uninformative_read_does_not_pass_the_gate():
    rng = np.random.default_rng(0)
    t = rng.uniform(5, 50, 200)
    s = score_arm(rng.uniform(5, 50, 200), t)
    assert s["r2"] < GATE_R2, "noise must not clear the LF0 gate"


def test_constant_read_scores_at_or_below_zero():
    """A reader that always says '20 m' explains none of the variance — it must
    not look like a partial success."""
    t = np.linspace(5, 50, 60)
    s = score_arm(np.full(60, 20.0), t)
    assert s["r2"] <= 0.0


def test_spearman_is_rank_based_and_sign_correct():
    x = np.array([1.0, 2, 3, 4, 5])
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_r2_matches_the_textbook_definition():
    t = np.array([1.0, 2, 3, 4])
    assert r2(t, t) == pytest.approx(1.0)
    assert r2(np.full(4, t.mean()), t) == pytest.approx(0.0)


def test_tau_default_is_the_inherited_p8_gate_value():
    """Re-tuning tau on the lead task would be fitting a one-parameter model
    and calling it geometry."""
    assert TAU_P8 == 0.7


def test_headline_corridor_is_among_those_reported():
    from lf0_bev_lead import HEADLINE_CORRIDOR
    assert HEADLINE_CORRIDOR in CORRIDOR_M


# --------------------------------------------------------------------------- #
# the kept-positions trap                                                      #
# --------------------------------------------------------------------------- #
def test_kept_positions_are_indices_not_booleans():
    """batch_rasters returns (rasters|None, KEPT_POSITIONS, n_no_label). `keep`
    is a list of indices INTO idx — for a single-window batch the valid value is
    [0], and `bool(keep[0])` is `bool(0)` = False.

    MEASURED 2026-08-12: gating the ground-truth raster on `bool(keep[0])`
    discarded EVERY valid raster and produced n_paired=0 on all six arms. The
    NO_LABEL test is `rasters is None` — the same check p8_bev_reel uses — and
    it is the only one. This pins the semantics so the confusion cannot return."""
    keep_for_one_valid_window = [0]
    assert bool(keep_for_one_valid_window[0]) is False, (
        "this is the trap: the first kept POSITION is 0, which is falsy")
    assert len(keep_for_one_valid_window) == 1, (
        "validity is 'a position was kept', i.e. a non-empty list — never the "
        "truthiness of its first element")
    assert not [] , "an empty kept-positions list is the real 'nothing kept'"


def test_source_of_the_lf0_gt_read_does_not_gate_on_keep():
    """A source-level guard: the GT branch must not resurrect the bool(keep[0])
    test. Cheap, and it caught a real regression class once already."""
    import inspect

    import lf0_bev_lead as L
    src = inspect.getsource(L.main)
    # Strip comments before scanning: the fix's own explanatory comment quotes
    # `bool(keep[0])` to say why it is wrong, and a naive scan trips on that.
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "bool(keep[" not in code, "the kept-positions trap has returned"
    assert "if rk is None" in code, "NO_LABEL must be tested by `rk is None`"
