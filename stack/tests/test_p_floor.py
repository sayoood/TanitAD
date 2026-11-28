"""The P-panel floor reporter (`stack/scripts/p_floor.py`), pinned as literals.

⛔ The rule that matters: **|P0 - P0b| is the floor; |P0 - A8| is an UPPER BOUND and is never
called the floor.** It was committed before the numbers existed precisely so it could not drift
afterwards, so it is pinned here and broken on purpose in `stack/scripts/mutate_p_floor.py`.

Every expectation is a hand-computed literal, never an expression over the code under test.
No arm is read: the rows are fabricated so the answers are known in advance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "taniteval"))

import p_floor as PF                                        # noqa: E402


def _row(centre, **kw):
    r = {"eval_box3d_centre": centre, "eval_box3d_size": 1.0, "eval_box3d_yaw": 0.5,
         "eval_box3d_presence": 0.1, "eval_box3d_cls": 0.8, "eval_box3d_z": 0.7,
         "eval_box3d_h": 0.2, "eval_lon": 2.0, "eval_lat": 3.0, "eval_tac_v6": 4.0,
         "eval_route": 5.0, "step": 5000}
    r.update(kw)
    return r


# ------------------------------------------------------------------ the floor itself
def test_the_floor_is_P0_vs_P0b_and_is_LABELLED_as_such():
    rep = PF.report(_row(13.0), _row(13.25), _row(14.0))
    assert rep["status"] == "OK"
    # 13.25 - 13.0 = 0.25 exactly
    assert rep["seed_floor"]["headline_abs_delta"] == 0.25
    assert rep["seed_floor"]["admissible_as"] == "THE SEED FLOOR"


def test_the_A8_comparison_is_an_UPPER_BOUND_and_is_never_the_floor():
    """⛔ The rule a mutation must break: the two blocks must not be interchangeable."""
    rep = PF.report(_row(13.0), _row(13.25), _row(14.0))
    ub = rep["upper_bound_not_the_floor"]
    assert ub["headline_abs_delta"] == 1.0                   # |13.0 - 14.0| exactly
    assert "UPPER BOUND ONLY" in ub["admissible_as"]
    assert ub["headline_abs_delta"] != rep["seed_floor"]["headline_abs_delta"]
    # and the rendered text must say which is which
    txt = PF.render(rep)
    assert "SEED FLOOR (quote this)" in txt and "NOT THE FLOOR" in txt


def test_the_code_delta_gap_is_reported_because_it_IS_the_code_delta():
    rep = PF.report(_row(13.0), _row(13.25), _row(14.0))
    # |1.0 - 0.25| = 0.75 exactly
    assert rep["code_delta_gap"]["abs"] == 0.75


def test_ONE_replicate_is_NOT_a_floor(capsys):
    """⛔ A floor from a single replicate is a guess. The upper bound may not stand in."""
    rep = PF.report(_row(13.0), None, _row(14.0))
    assert rep["status"] == "INCONCLUSIVE"
    assert rep["seed_floor"] is None and rep["upper_bound_not_the_floor"] is None
    assert "may NOT stand in" in rep["reason"]
    assert "STATUS INCONCLUSIVE" in PF.render(rep)


def test_render_REFUSES_an_unlabelled_upper_bound():
    rep = PF.report(_row(13.0), _row(13.25), _row(14.0))
    rep["upper_bound_not_the_floor"]["admissible_as"] = "the floor"
    with pytest.raises(ValueError, match="upper-bound label"):
        PF.render(rep)


# ------------------------------------------------------------------ the four families
def test_every_family_is_reported_SEPARATELY_and_never_pooled():
    rep = PF.report(_row(13.0), _row(13.25, eval_lon=2.5, eval_lat=3.75), None)
    fam = rep["seed_floor"]["by_family"]
    assert set(fam) == {"longitudinal", "lateral", "tactical", "strategic", "perception"}
    assert fam["longitudinal"]["eval_lon"]["abs_delta"] == 0.5     # |2.0 - 2.5|
    assert fam["lateral"]["eval_lat"]["abs_delta"] == 0.75         # |3.0 - 3.75|
    assert fam["tactical"]["eval_tac_v6"]["abs_delta"] == 0.0      # unchanged
    assert fam["strategic"]["eval_route"]["abs_delta"] == 0.0
    # ⛔ no pooled/total score anywhere — a single number hides the trade-off
    assert not any(k in rep["seed_floor"] for k in ("total", "pooled", "combined", "score"))


def test_a_missing_metric_is_REPORTED_with_its_reason_not_dropped():
    a, b = _row(13.0), _row(13.25)
    del b["eval_lon"]
    d = PF.delta(a, b, ("eval_lon", "eval_lat"))
    assert d["eval_lon"]["abs_delta"] is None and d["eval_lon"]["reason"] == "absent in B"
    assert d["eval_lat"]["abs_delta"] == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", None])
def test_non_finite_or_non_numeric_is_NEVER_a_zero_delta(bad):
    a, b = _row(13.0), _row(13.0)
    b["eval_lon"] = bad
    d = PF.delta(a, b, ("eval_lon",))
    assert d["eval_lon"]["abs_delta"] is None, "a broken value must not read as 'no difference'"


# ------------------------------------------------------------------ the row reader
def test_final_eval_row_takes_the_LAST_row_CARRYING_the_headline(tmp_path):
    import json as _j
    m = tmp_path / "metrics.jsonl"
    m.write_text("\n".join([
        _j.dumps({"step": 10, "eval_box3d_centre": 20.0}),
        _j.dumps({"step": 20, "eval_box3d_centre": 15.0}),
        _j.dumps({"step": 21, "loss": 1.0}),          # a TRAIN row after it — must be ignored
    ]), encoding="utf-8")
    row = PF.final_eval_row(tmp_path)
    assert row["step"] == 20 and row["eval_box3d_centre"] == 15.0


def test_a_missing_metrics_file_reads_None_not_an_empty_row(tmp_path):
    assert PF.final_eval_row(tmp_path) is None


# ============================================ the headline's POPULATION must travel with it
# ⛔ `eval_box3d_centre` is the 360° aggregate (slot_set_loss sums over EVERY matched row).
# The prereg's SUPPORTED criteria are written on the NEAR-FORWARD error, and the two differ by
# ~2×. A floor quoted from this key against a near-forward verdict is cross-population and
# inadmissible, so the population label must be inseparable from the number.
def test_the_headline_carries_its_POPULATION_in_json_and_text():
    rep = PF.report(_row(13.0), _row(13.25), _row(14.0))
    pop = rep["seed_floor"]["headline_population"]
    assert "all_360" in pop and "SECONDARY" in pop and "near-forward" in pop
    # ⛔ ADJACENCY, NOT EXISTENCE. An earlier version asserted only that "population:" and
    # "all_360" appeared SOMEWHERE in the rendered text — an EXISTENCE predicate guarding a claim
    # about PLACE, which is the defect that let mutation M4 of the doc-checker pass while a table
    # cell had been reverted. "Inseparable" means the label sits WITH the number, so the test
    # locates the headline line and requires the population on the line immediately after it.
    lines = PF.render(rep).splitlines()
    hi = next(i for i, l in enumerate(lines) if "SEED FLOOR (quote this)" in l)
    assert hi + 1 < len(lines), "the headline is the last line — nothing can follow it"
    nxt = lines[hi + 1]
    assert "population:" in nxt and "all_360" in nxt, (
        "the population label must sit on the line immediately after the headline; found %r"
        % nxt[:80])
