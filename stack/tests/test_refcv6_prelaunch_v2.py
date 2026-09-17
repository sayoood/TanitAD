"""The two pre-launch refusals that did not exist -- and the mutations that prove
they can go RED.

⛔ An unmutated guard is a comment. Every refusal below is tested in BOTH
directions: the clean case passes, and the defect it exists for is reintroduced
and required to fail. The `_MUT_` tests are the ones that earn the module.

Refusals 3 and 4 of `PREREG_REFCV6_V2.md` section 12, built after
`PREREG_REFCV6_V2.ERRATUM-1.md` section E4 MEASURED that they did not exist.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tanitad.train.prelaunch_v2 import (
    HypothesisNotRegistered, ScoredSplitLeak, prelaunch_v2,
    refuse_overlapping_splits, refuse_scored_split_leak,
    refuse_unregistered_hypotheses, registered_hypotheses,
)

#: The real registry, if this checkout has it. Tests that need it SKIP rather
#: than pass when it is absent -- a guard tested only against a fixture it wrote
#: itself has not been tested against the thing it guards.
REAL_GOALS = (Path(__file__).resolve().parents[2]
              / "Project Steering" / "GOALS_AND_CLAIMS.md")


def _goals(tmp_path, body: str) -> Path:
    p = tmp_path / "GOALS_AND_CLAIMS.md"
    p.write_text(body, encoding="utf-8")
    return p


def _clean_panel():
    return {"tuned": {"tau": {"value": 0.5, "fitted_on": "fit"},
                      "pos_weight": {"value": 50, "selected_on": "fit"}}}


def _clean_splits():
    return {"fit": [f"c{i}" for i in range(100)],
            "val": [f"v{i}" for i in range(20)],
            "test": [f"t{i}" for i in range(139)]}


# ---------------------------------------------------------------- refusal 3

def test_the_registry_is_read_from_the_REAL_file_and_carries_the_v2_ids():
    """⭐ Against the real `GOALS_AND_CLAIMS.md`, not a fixture."""
    if not REAL_GOALS.is_file():
        pytest.skip(f"needs {REAL_GOALS}")
    ids = registered_hypotheses(REAL_GOALS)
    assert len(ids) > 50, f"only {len(ids)} ids parsed -- the pattern is wrong"
    # the five this pre-registration itself registered
    for want in ("E-REFCV6V2-TRUNK", "E-REFCV6V2-PERCEP", "E-REFCV6V2-TACTICAL",
                 "E-REFCV6V2-NAV", "E-REFCV6V2-DRIVE"):
        assert want in ids, f"{want} is not registered"


def test_a_registered_id_passes_and_an_unregistered_one_is_REFUSED(tmp_path):
    g = _goals(tmp_path, "| E-REAL-ONE | ... |\n| D-REAL-TWO | ... |\n")
    assert refuse_unregistered_hypotheses(["E-REAL-ONE"], g)["n_registered"] == 2
    with pytest.raises(HypothesisNotRegistered, match="E-MADE-UP"):
        refuse_unregistered_hypotheses(["E-REAL-ONE", "E-MADE-UP"], g)


def test_MUT_an_arm_that_names_NO_hypothesis_is_refused(tmp_path):
    """MUTATION: the cheapest way past refusal 3 is to claim nothing at all."""
    g = _goals(tmp_path, "| E-REAL-ONE | ... |\n")
    with pytest.raises(HypothesisNotRegistered, match="NO hypothesis id"):
        refuse_unregistered_hypotheses([], g)


def test_MUT_a_missing_registry_RAISES_instead_of_refusing_everything(tmp_path):
    """⛔ The important direction. An empty registry would refuse every id, which
    reads as a broken guard and gets the guard switched off."""
    # ⛔ Matching on the MESSAGE, not merely the type. `read_text` raises
    # FileNotFoundError by itself, so a type-only assertion passes with the guard
    # DELETED -- the guard-removal audit caught exactly that and reported this
    # check as ESCAPED. What the guard actually contributes is the distinction it
    # names, so that is what is pinned.
    with pytest.raises(FileNotFoundError, match="different facts"):
        registered_hypotheses(tmp_path / "nope.md")
    with pytest.raises(ValueError, match="no hypothesis id"):
        registered_hypotheses(_goals(tmp_path, "prose with no ids at all\n"))


def test_a_prefix_of_a_registered_id_does_NOT_satisfy_it(tmp_path):
    """⛔ `E-REFCV6V2-TRUNKY` must not pass as `E-REFCV6V2-TRUNK`."""
    g = _goals(tmp_path, "| E-REFCV6V2-TRUNKY | ... |\n")
    with pytest.raises(HypothesisNotRegistered):
        refuse_unregistered_hypotheses(["E-REFCV6V2-TRUNK"], g)


# ---------------------------------------------------------------- refusal 4

def test_a_panel_that_declares_its_tuning_on_the_fit_split_PASSES():
    assert refuse_scored_split_leak(_clean_panel())["n_tuned"] == 2


def test_MUT_a_quantity_fitted_on_the_SCORED_split_is_REFUSED():
    """MUTATION: the defect itself -- tune on the split you score on."""
    p = _clean_panel()
    p["tuned"]["tau"]["fitted_on"] = "test"
    with pytest.raises(ScoredSplitLeak, match="MANUFACTURES a positive"):
        refuse_scored_split_leak(p)


def test_MUT_an_UNDECLARED_tuned_quantity_is_REFUSED_not_assumed_innocent():
    """⛔ The failure mode the record actually contains: a tuned threshold that
    says nothing and is read as a constant."""
    p = _clean_panel()
    del p["tuned"]["tau"]["fitted_on"]
    with pytest.raises(ScoredSplitLeak, match="do not say where"):
        refuse_scored_split_leak(p)
    p["tuned"]["tau"] = 0.5           # not even a dict
    with pytest.raises(ScoredSplitLeak, match="do not say where"):
        refuse_scored_split_leak(p)


def test_MUT_a_panel_with_NO_tuned_block_is_REFUSED_but_an_EMPTY_one_passes():
    """⛔ Absence of the block and an explicit empty block are DIFFERENT facts.
    'we tuned nothing' must be said, not inferred from silence."""
    with pytest.raises(ScoredSplitLeak, match="no `tuned` block"):
        refuse_scored_split_leak({})
    assert refuse_scored_split_leak({"tuned": {}})["n_tuned"] == 0


def test_every_accepted_spelling_of_the_declaration_works():
    for key in ("fitted_on", "fit_split", "selected_on", "tuned_on"):
        assert refuse_scored_split_leak({"tuned": {"k": {key: "fit"}}})


# ------------------------------------------------------- the measurable half

def test_disjoint_splits_pass_and_the_counts_are_reported():
    rep = refuse_overlapping_splits(_clean_splits())
    assert rep["n_test"] == 139
    assert rep["shared_with_fit"] == 0 and rep["shared_with_val"] == 0


def test_MUT_one_shared_id_between_fit_and_test_is_REFUSED():
    """MUTATION: a single leaked clip. ⛔ One is enough -- the guard must not
    have a tolerance."""
    s = _clean_splits()
    s["test"] = s["test"] + [s["fit"][0]]
    with pytest.raises(ScoredSplitLeak, match="appear in BOTH"):
        refuse_overlapping_splits(s)


def test_MUT_an_EMPTY_scored_split_RAISES_rather_than_reading_zero_overlap():
    """⛔ An empty set overlaps nothing by construction -- the coverage gate's
    own trap, reproduced here so it cannot return silently."""
    s = _clean_splits()
    s["test"] = []
    with pytest.raises(ScoredSplitLeak, match="EMPTY"):
        refuse_overlapping_splits(s)
    with pytest.raises(ScoredSplitLeak, match="no 'test' split"):
        refuse_overlapping_splits({"fit": ["a"]})


# ------------------------------------------------------------------ together

def test_prelaunch_v2_passes_a_clean_arm_and_each_defect_fails_it(tmp_path):
    g = _goals(tmp_path, "| E-OK | ... |\n")
    ok = dict(arm_ids=["E-OK"], goals_md=g, panel=_clean_panel(),
              splits=_clean_splits())
    assert prelaunch_v2(**ok)["verdict"] == "PASS"

    bad_h = dict(ok, arm_ids=["E-NOPE"])
    with pytest.raises(HypothesisNotRegistered):
        prelaunch_v2(**bad_h)

    leaky = _clean_panel()
    leaky["tuned"]["tau"]["fitted_on"] = "test"
    with pytest.raises(ScoredSplitLeak):
        prelaunch_v2(**dict(ok, panel=leaky))

    overlapped = _clean_splits()
    overlapped["val"] = overlapped["val"] + [overlapped["test"][0]]
    with pytest.raises(ScoredSplitLeak):
        prelaunch_v2(**dict(ok, splits=overlapped))
