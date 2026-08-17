"""The aug120 perception layer is ONE detection floor and ONE schema — pinned.

⛔ WHY THIS FILE EXISTS. For the whole life of the aug120 perception layer it was
**201 clips built at two different detection floors** — 115 at 0.25 under schema v2,
86 at the vendor default 0.5 under no schema at all — and *nothing in the data could
reveal it*. A confidence floor is applied INSIDE the vendor forward pass
(`Sam3Processor`: `keep = out_probs > confidence_threshold`), so it is invisible in
the payload: it shows up only as **rows that are not there**. Two records built at
two floors are structurally identical. The corpus read as homogeneous, and every
per-concept rate pooled over it was unattributable while looking like an answer.

That is the `df`-reports-the-cluster family — a probe that reports the wrong scope is
worse than no probe, because it looks like an answer — and it cost the programme
every corpus-level perception number until the 86 were re-detected.

**Two things are pinned here, and both are needed:**

1. **THE DATA** — `…/2026-08-17-perception-floor-unify/raw/floor_homogeneity_manifest.json`
   publishes the per-clip floor/schema/md5 and, at the top, the distinct SETS. The
   invariant is `len(distinct_confidence_thresholds) == 1`. The day a re-run lands at
   a different floor, or a pre-schema record is merged back in, that set grows a
   second member and this file goes red.

2. **THE DETECTOR** — `s2_lab_lib.census_records`, the shared C77 completion
   predicate. Pinning only the data would leave the corpus defended by an instrument
   that could itself rot: if `require_conf` stopped binding, a mixed corpus would
   sail through the census and the manifest would be rebuilt from it, green. So the
   predicate is fed a deliberately mixed corpus here and must REFUSE it.

⚠️ The manifest is the artifact, not the corpus: ~24 MB of JSON records are not in
git (neither the v1 nor the v2 corpus ever was) and a unit test has no network. The
md5 per clip is what lets any far-side copy be checked against this repo.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "colab"))

MANIFEST = os.path.join(
    _REPO, "TanitAD Research Hub", "Data Engineering", "Implementation",
    "incoming", "2026-08-17-perception-floor-unify", "raw",
    "floor_homogeneity_manifest.json")


@pytest.fixture(scope="module")
def man() -> dict:
    if not os.path.exists(MANIFEST):
        pytest.fail(
            f"the perception homogeneity manifest is MISSING at {MANIFEST}. "
            "It is the only in-repo record of which detection floor each aug120 "
            "clip was built at; without it a mixed corpus is undetectable again. "
            "Rebuild with …/2026-08-17-perception-floor-unify/code/f3_homogeneity.py")
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# 1. the DATA                                                                  #
# --------------------------------------------------------------------------- #
def test_the_corpus_is_one_detection_floor_and_one_schema(man):
    """⛔ THE INVARIANT THE CORPUS WAS BUILT VIOLATING. Not `floor == 0.25` on
    average, not `most records` — the SET of floors present must have exactly one
    member, because a rate pooled over two floors is a number about two
    populations."""
    floors = man["distinct_confidence_thresholds"]
    schemas = man["distinct_schema_versions"]
    assert len(floors) == 1, (
        f"the aug120 perception layer is MIXED-FLOOR again: {floors} "
        f"(histogram {man['confidence_threshold_histogram']}). No per-concept "
        "rate may be pooled across it. Re-run the odd leg through "
        "…/2026-08-17-perception-floor-unify/code/f1_run86.py — do NOT relax "
        "this test.")
    assert len(schemas) == 1, (
        f"the corpus spans schema versions {schemas} "
        f"(histogram {man['schema_version_histogram']})")
    assert float(floors[0]) == man["expected_confidence_threshold"]
    assert int(schemas[0]) >= man["expected_schema_version_min"]
    assert man["HOMOGENEOUS"] is True


def test_every_record_stamps_its_floor_and_its_schema(man):
    """⛔ THE 86's ORIGINAL DEFECT, AND IT IS SEPARATE FROM THE FLOOR ITSELF.
    They stamped NEITHER field, so their floor was knowable only from the
    launching code — an INHERITED fact about an artifact, which is the evidence
    class this programme refuses for anything that decides a GPU-day. An
    unstamped record cannot be certified as belonging to the corpus at all."""
    assert man["n_records_without_confidence_stamp"] == 0
    assert man["n_records_without_schema_stamp"] == 0
    bad = [c["clip_id"] for c in man["clips"]
           if c["confidence_threshold"] is None or c["schema_version"] is None]
    assert not bad, f"{len(bad)} records stamp no floor/schema, e.g. {bad[:3]}"


def test_every_clip_carries_the_liveness_control_and_no_errors(man):
    """C77: a run is complete when detections exist AND the road/sky positive
    control fired — never when files exist. A zero with a dead control is a dead
    engine; a zero with a live one is an empty road."""
    assert man["records_without_control"] == 0
    assert man["liveness_dead"] == 0
    assert man["error_census"] == {}, man["error_census"]
    assert man["liveness_live"] == man["n_records"]
    assert man["zero_det_split"]["dead_control"] == 0


def test_the_manifest_cannot_lie_about_its_own_coverage(man):
    """⚠️ AN "EVERYTHING IS FINE" MANIFEST OVER PART OF THE COHORT WOULD BE THIS
    PACKAGE'S OWN DEFECT ONE LEVEL UP. `covers_cohort` is re-derived here rather
    than believed, and a corpus that does not cover the cohort is REQUIRED to
    carry its residual by name — so an incomplete run can never present as a
    complete one."""
    assert len(man["clips"]) == man["n_records"]
    assert man["covers_cohort"] == (man["n_records"] == man["cohort_n"])
    assert man["n_residual"] == len(man["residual"])
    assert man["n_residual"] == man["cohort_n"] - man["n_records"]
    assert man["UNIFIED"] == (man["HOMOGENEOUS"] and man["covers_cohort"])
    if not man["covers_cohort"]:
        assert man["residual"], (
            "the corpus does not cover the cohort but names no residual — an "
            "incomplete run presenting as complete")
    ids = [c["clip_id"] for c in man["clips"]]
    assert len(set(ids)) == len(ids), "duplicate clip_id rows in the manifest"
    assert all(c["md5"] and len(c["md5"]) == 32 for c in man["clips"])


def test_the_perception_layer_covers_the_whole_cohort(man):
    """The union check. Kept SEPARATE from the homogeneity checks on purpose:
    'one floor' and 'all 201 clips' are different claims, and a half-done re-run
    satisfies the first while failing the second.

    ⚠️ COVERAGE IS A PROGRAMME GOAL, NOT A CODE INVARIANT, so an in-progress
    corpus reports as XFAIL WITH ITS RESIDUAL NAMED rather than reddening a
    suite that is green on every invariant the code actually controls. The
    moment coverage lands this becomes an ordinary hard assertion — nothing has
    to be remembered or un-marked. ⛔ It is NOT a licence to leave it xfailed:
    the residual is printed here and escalated in
    `…/2026-08-17-perception-floor-unify/PERCEPTION_FLOOR_UNIFY.md`."""
    if not man["covers_cohort"]:
        pytest.xfail(
            f"aug120 perception layer covers {man['n_records']} of "
            f"{man['cohort_n']} clips — {man['n_residual']} RESIDUAL at the old "
            f"floor, e.g. {man['residual'][:3]}. Until this closes NO "
            "per-concept rate may be pooled across the cohort. Resume: "
            "…/2026-08-17-perception-floor-unify/code/f2_drive86.py")
    assert man["n_records"] == man["cohort_n"] == 201
    assert man["n_residual"] == 0 and man["residual"] == []
    assert man["UNIFIED"] is True


# --------------------------------------------------------------------------- #
# 2. the DETECTOR                                                              #
# --------------------------------------------------------------------------- #
def _rec(cid: str, *, conf=0.25, schema=2, live=True, err=None, n_det=3):
    """A minimal record of the shape `census_records` judges."""
    out = {
        "clip_id": cid, "n_det_total": n_det, "n_scene_det_total": 7,
        "per_concept_hits": {"car": n_det}, "per_scene_hits": {"road curb": 7},
        "frames": {}, "schema_version": schema,
        "liveness": {"n_det": {"road": 2, "sky": 1} if live
                     else {"road": 0, "sky": 0}},
    }
    if conf is not None:
        out["engine"] = {"confidence_threshold": conf, "schema_version": schema}
    if schema is None:
        out.pop("schema_version")
    if err:
        out["err_kinds"] = err
    return out


def test_the_census_refuses_a_record_detected_at_a_different_floor():
    """⛔ THE INSTRUMENT PIN. Two records, identical in every visible respect,
    differing only in the floor they were detected at — which is exactly how the
    aug120 corpus looked. The census must count the odd one and REFUSE to pass."""
    import s2_lab_lib as L
    items = [("a", _rec("a", conf=0.25)), ("b", _rec("b", conf=0.50))]
    cen = L.census_records(iter(items), require_schema=2, require_conf=0.25)
    assert cen["wrong_conf"] == 1, cen
    assert cen["n_complete"] == 1
    assert "b" not in cen["complete_clips"]
    assert cen["pass_"] is False


def test_the_census_refuses_a_pre_schema_record_that_stamps_nothing():
    """The 86's exact shape: no `engine` block, no `schema_version`. It is
    present, non-empty, error-free and live — and still the WRONG record."""
    import s2_lab_lib as L
    rec = _rec("c", conf=None, schema=None)
    assert "engine" not in rec and "schema_version" not in rec
    cen = L.census_records(iter([("c", rec)]), require_schema=2,
                           require_conf=0.25)
    assert cen["wrong_conf"] == 1 and cen["wrong_schema"] == 1, cen
    assert cen["n_complete"] == 0
    assert cen["pass_"] is False


def test_the_census_refuses_a_record_whose_liveness_control_is_dead():
    """A zero-detection clip whose road/sky control also read zero is a dead
    engine, not an empty road — the distinction C77 exists to make."""
    import s2_lab_lib as L
    cen = L.census_records(iter([("d", _rec("d", live=False, n_det=0))]),
                           require_schema=2, require_conf=0.25)
    assert cen["liveness_dead"] == 1
    assert cen["pass_"] is False
    assert cen["zero_det_clips"][0]["liveness_live"] is False


def test_the_census_refuses_a_record_carrying_an_error_entry():
    import s2_lab_lib as L
    cen = L.census_records(
        iter([("e", _rec("e", err={"RuntimeError: mat1 and mat2": 4}))]),
        require_schema=2, require_conf=0.25)
    assert sum(cen["error_census"].values()) == 4
    assert cen["n_complete"] == 0 and cen["pass_"] is False


def test_the_census_refuses_a_record_whose_filename_and_content_disagree():
    """`done_set` guards this far side; the shared predicate must guard it for
    any transport, or a local bank could silently key a record to the wrong
    clip."""
    import s2_lab_lib as L
    with pytest.raises(RuntimeError, match="clip_id"):
        L.census_records(iter([("wanted", _rec("other"))]))


def test_a_homogeneous_corpus_passes_so_the_refusals_mean_something():
    """A test suite of refusals alone would also pass if the predicate refused
    EVERYTHING. This is the positive control for the controls."""
    import s2_lab_lib as L
    items = [(c, _rec(c)) for c in ("a", "b", "c")]
    cen = L.census_records(iter(items), want={"a", "b", "c"},
                           require_schema=2, require_conf=0.25)
    assert cen["n_complete"] == 3 and cen["wrong_conf"] == 0
    assert cen["pass_"] is True


def test_the_local_and_far_side_censuses_are_the_same_predicate():
    """⭐ THE SPLIT IS THE POINT. Two implementations of "is this clip done?" is
    how a corpus goes mixed while both checks report green. Both public censuses
    must delegate to `census_records` rather than re-implement it."""
    import inspect

    import s2_lab_lib as L
    for fn in (L.content_census, L.content_census_local):
        src = inspect.getsource(fn)
        assert "census_records(" in src, (
            f"{fn.__name__} no longer delegates to census_records — the "
            "completion rule has been forked")
