"""Tests for the ONE B1 label consumer.

Verified BY CONTENT against the released blob where it is available (assert on
returned values, never on exit codes), and by fixture where it is not — so the
suite stays green on a box without the corpus.
"""
import gzip
import json
import pathlib

import pytest

from tanitad.data.v7_labels import (HEADS, OracleNavRefused,
                                    assert_mask_matches_presence, class_weights,
                                    effective_mask, flatten_tactical_actions,
                                    head_mask, is_oracle_nav, load_v7_labels,
                                    oracle_nav)
from tanitad.models.vocab_v7 import NOT_YET_EXTRACTABLE

#: the RELEASED blob, md5 ee44875916ae7c0ac002c6716b9658ea.
#: ⚠️ Five other copies exist under three roots and their md5s DIFFER; one sits
#: in a research `incoming/` directory (md5 e22acf70…) and is a stale schema.
BLOB = pathlib.Path("C:/Users/Admin/tanitad-wt/_s2build/release/"
                    "tanitad-v7-training-corpus/labels/s2_labels_v7.jsonl.gz")
BLOB_MD5 = "ee44875916ae7c0ac002c6716b9658ea"
N_RECORDS = 4719
needs_blob = pytest.mark.skipif(not BLOB.exists(), reason="B1 blob not on this box")


def _rec(**over):
    r = {"clip_id": "c0", "schema_version": "s2-geom-v7", "vocab": "v7",
         "a_tac": {"lat": "LANE_KEEP", "lon": "CRUISE", "truncated": False,
                   "lat_args": {"within_m": 3.0}, "lon_args": {"v_target_ms": 9.0},
                   "serves_goals": {"lat_serves": [], "lon_serves": []}},
         "a_str": {"token": "HOLD_MAIN_ROAD"},
         "g_str": {"token": "FOLLOW_ROUTE"},
         "g_tac": {"anchor": {"goal_x_m": 50.0, "goal_y_m": 1.0, "t_reach_s": 6.0},
                   "goals": {"YIELD": {"disputed": True, "time_basis": "abs",
                                       "t_nominal_s": 3.0}},
                   "violations": []},
         "bands": {"operative_s": [0, 2], "tactical_s": [2, 6],
                   "strategic_s": [8, 30], "unassigned_manoeuvres": []},
         "t0_s": 8.0, "horizon": {"available_s": 30.0, "recording_span_s": 30.0},
         "nav_command": {"token": "NAV_TURN_R", "provenance": "ego-future",
                         "oracle": True, "args": {}},
         "turn_suppression": None,
         "alpamayo": {"lateral": {"agree": False},
                      "longitudinal": {"agree": False}}}
    r.update(over)
    return r


@pytest.fixture
def fixture_blob(tmp_path):
    p = tmp_path / "labels.jsonl.gz"
    rows = [_rec(clip_id=f"c{i}") for i in range(3)]
    rows[1]["a_tac"]["lon"] = "ACCELERATE"
    rows[2]["a_tac"]["lat"] = "TURN_L"
    rows[2]["a_str"]["token"] = "PREPARE_TURN_L_FOLLOW_ROUTE"
    rows[2]["g_str"]["token"] = "TURN_LEFT_FOLLOW_ROUTE"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


# ------------------------------------------------------- condition 1: oracle nav
def test_oracle_nav_is_unreachable_by_default(fixture_blob):
    """⛔ THE BINDING ONE. Reaching nav_command must require an explicit opt-in."""
    labels, man = load_v7_labels(fixture_blob)
    assert man.allow_oracle_nav is False
    with pytest.raises(OracleNavRefused) as e:
        oracle_nav(labels[0], man)
    msg = str(e.value)
    assert "ego-future" in msg, "the refusal must say WHY"
    assert "tac_anchor" in msg, "the refusal must name the admissible alternative"


def test_oracle_nav_opt_in_stamps_the_manifest(fixture_blob):
    labels, man = load_v7_labels(fixture_blob, allow_oracle_nav=True)
    assert man.allow_oracle_nav is True
    assert man.to_dict()["allow_oracle_nav"] is True, "the stamp must survive to config"
    assert oracle_nav(labels[0], man)["token"] == "NAV_TURN_R"


def test_the_permission_lives_on_the_manifest_not_a_bare_argument(fixture_blob):
    """⭐ A run whose config says allow_oracle_nav=False cannot have read it: the
    check and the recorded config are the SAME object, so they cannot disagree."""
    labels, man = load_v7_labels(fixture_blob, allow_oracle_nav=False)
    _, permissive = load_v7_labels(fixture_blob, allow_oracle_nav=True)
    with pytest.raises(OracleNavRefused):
        oracle_nav(labels[0], man)
    assert oracle_nav(labels[0], permissive) is not None


def test_the_admissible_goal_is_present_and_geometric(fixture_blob):
    labels, _ = load_v7_labels(fixture_blob)
    a = labels[0].tac_anchor
    assert {"goal_x_m", "goal_y_m", "t_reach_s"} <= set(a)


# ------------------------------------------- condition 2: a_tac stays factored
def test_lat_and_lon_are_separate_heads():
    assert HEADS["tac_lat"] != HEADS["tac_lon"]
    assert len(HEADS["tac_lat"]) == 8 and len(HEADS["tac_lon"]) == 8
    assert set(HEADS["tac_lat"]).isdisjoint(HEADS["tac_lon"])


def test_flattening_lat_by_lon_is_REFUSED(fixture_blob):
    """⛔ DELIBERATE-REGRESSION TEST. Collapsing the two axes rebuilds the
    5-way-softmax defect (0/881 accelerate, the speed-fan) in the labels."""
    labels, _ = load_v7_labels(fixture_blob)
    with pytest.raises(NotImplementedError) as e:
        flatten_tactical_actions(labels)
    assert "REFUSED" in str(e.value)
    assert "speed-fan" in str(e.value), "the refusal must carry its reason"


# ----------------------------------------------------- condition 3: the mask
def test_head_mask_marks_exactly_the_not_yet_extractable():
    for head, toks in HEADS.items():
        m = head_mask(head)
        assert len(m) == len(toks)
        for tok, ok in zip(toks, m):
            assert ok == (tok not in NOT_YET_EXTRACTABLE), f"{head}/{tok}"


def test_mask_equals_absence_on_the_fixture(fixture_blob):
    labels, _ = load_v7_labels(fixture_blob)
    # the fixture is tiny, so most classes are absent-and-unmasked ⇒ must FAIL
    with pytest.raises(AssertionError) as e:
        assert_mask_matches_presence(labels)
    assert "ABSENT BUT UNMASKED" in str(e.value)


def test_the_mask_check_fails_in_BOTH_directions(fixture_blob):
    """⭐ A check that can only fire one way is half a check. Masked-but-present
    is training signal thrown away; absent-but-unmasked is a dead logit."""
    labels, _ = load_v7_labels(fixture_blob)
    with pytest.raises(AssertionError) as e:
        assert_mask_matches_presence(labels, heads=["tac_lat"])
    assert "ABSENT BUT UNMASKED" in str(e.value)


# ------------------------------------------------------------ skew weighting
def test_weights_come_from_the_split_not_a_constant(fixture_blob):
    labels, _ = load_v7_labels(fixture_blob)
    w = class_weights(labels, "tac_lon")
    assert w["CRUISE"] > 0 and w["ACCELERATE"] > 0
    # CRUISE appears twice, ACCELERATE once ⇒ inverse frequency ranks ACCELERATE higher
    assert w["ACCELERATE"] > w["CRUISE"]


def test_masked_classes_get_zero_weight(fixture_blob):
    labels, _ = load_v7_labels(fixture_blob)
    for head in HEADS:
        w = class_weights(labels, head)
        for tok in HEADS[head]:
            if tok in NOT_YET_EXTRACTABLE:
                assert w[tok] == 0.0, f"{head}/{tok} is masked ⇒ contributes no loss"


# ---------------------------------------------------------- load-time validation
def test_schema_mismatch_is_REFUSED(tmp_path):
    p = tmp_path / "bad.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_rec(schema_version="s2-geom-v6")) + "\n")
    with pytest.raises(ValueError) as e:
        load_v7_labels(p)
    assert "schema_version mismatch" in str(e.value)


def test_record_count_mismatch_is_REFUSED(fixture_blob):
    """⚠️ Six copies of this blob exist with differing md5s — a silently smaller
    file is a corpus change wearing a filename."""
    with pytest.raises(ValueError) as e:
        load_v7_labels(fixture_blob, require_records=4719)
    assert "expected 4719" in str(e.value)
    assert "md5" in str(e.value), "the refusal must name the blob it actually read"


def test_manifest_carries_the_md5_so_the_blob_is_identifiable(fixture_blob):
    _, man = load_v7_labels(fixture_blob)
    assert len(man.md5) == 32
    assert man.n_records == 3 and man.schema_version == "s2-geom-v7"


# ------------------------------- the fields the spec said do not exist
def test_the_doc_cited_fields_are_surfaced_not_dropped(fixture_blob):
    """⚠️ The spec reports `disputed`/`time_basis`/`t_nominal_s` as ZERO HITS and
    `agree` at the top of `alpamayo`. All four exist one level deeper. Dropping
    them would discard a disputed-flag the D-LABEL-GT conditions require."""
    labels, _ = load_v7_labels(fixture_blob)
    gf = labels[0].audit["goal_flags"]
    assert gf["YIELD"]["disputed"] is True
    assert gf["YIELD"]["time_basis"] == "abs"
    assert gf["YIELD"]["t_nominal_s"] == 3.0
    agree = labels[0].audit["alpamayo_agree"]
    assert agree["lateral"] is False and agree["longitudinal"] is False


def test_audit_fields_are_not_training_inputs(fixture_blob):
    """Spec §6: audit-only fields live under `.audit`, never as head targets."""
    labels, _ = load_v7_labels(fixture_blob)
    for k in ("turn_suppression", "goal_flags", "alpamayo_agree"):
        assert k in labels[0].audit
    for k in ("turn_suppression", "alpamayo_agree"):
        assert not hasattr(labels[0], k), f"{k} must not be a top-level target"


# ------------------------------------------------------------ against the real blob
@needs_blob
def test_released_blob_md5_and_count():
    import hashlib
    assert hashlib.md5(BLOB.read_bytes()).hexdigest() == BLOB_MD5
    labels, man = load_v7_labels(BLOB, require_records=N_RECORDS)
    assert man.md5 == BLOB_MD5 and len(labels) == N_RECORDS


@needs_blob
def test_released_blob_presence_counts_match_the_spec():
    """The spec's §3 per-head presence counts, re-derived from the corpus."""
    labels, _ = load_v7_labels(BLOB, require_records=N_RECORDS)
    present = {h: sum(1 for t in HEADS[h]
                      if any(getattr(x, h) == t for x in labels))
               for h in HEADS}
    assert present == {"tac_lat": 5, "tac_lon": 7, "str_action": 5, "str_goal": 4}


@needs_blob
def test_the_VOCAB_MASK_IS_INCOMPLETE_on_the_released_blob():
    """⛔ A MEASURED DEFECT IN `vocab_v7.NOT_YET_EXTRACTABLE`, not a test bug.

    The spec asserts "the 8 NOT_YET_EXTRACTABLE tokens are exactly the classes
    with zero occurrences, per head". They are NOT. The mask contains
    `LANE_CHANGE_L_FOLLOW_ROUTE` / `_R_FOLLOW_ROUTE` (STRATEGIC GOAL tokens) but
    not `LANE_CHANGE_L` / `LANE_CHANGE_R` (TACTICAL LAT ACTION tokens). The two
    families differ by a suffix, and the arithmetic conflated them.

    ⇒ two tactical classes are absent-and-unmasked = TWO DEAD LOGITS on tac_lat.
    This test PINS the defect so it cannot be forgotten, and must be inverted to
    an equality assertion the moment the vocab is fixed.
    """
    labels, _ = load_v7_labels(BLOB, require_records=N_RECORDS)
    with pytest.raises(AssertionError) as e:
        assert_mask_matches_presence(labels)
    msg = str(e.value)
    assert "ABSENT BUT UNMASKED" in msg
    assert "LANE_CHANGE_L" in msg and "LANE_CHANGE_R" in msg
    assert "MASKED BUT PRESENT" not in msg, (
        "no masked class may be present — that would be discarded training signal")


@needs_blob
def test_effective_mask_keeps_the_trainer_safe_meanwhile():
    """The trainer must not carry dead logits while the vocab is wrong."""
    labels, _ = load_v7_labels(BLOB, require_records=N_RECORDS)
    mask, prov = effective_mask(labels, "tac_lat")
    assert sum(mask) == 5, "exactly the 5 present classes stay trainable"
    assert prov["LANE_CHANGE_L"] == "absent" and prov["ABORT_LC"] == "both"


@needs_blob
def test_nav_is_oracle_by_PROVENANCE_on_every_record_even_where_the_flag_is_absent():
    """⛔ THE 11.2 % THE FLAG MISSES.

    `oracle: true` is present on 4,190 of 4,719; the other 529 carry a `reason`
    and no flag — yet all 529 are `provenance: ego-future`, i.e. still computed
    from the ego's own future. A guard keyed on the boolean leaks them.
    """
    labels, man = load_v7_labels(BLOB, allow_oracle_nav=True,
                                 require_records=N_RECORDS)
    assert all(is_oracle_nav(x) for x in labels), "provenance must catch all"
    navs = [oracle_nav(x, man) for x in labels]
    with_flag = [n for n in navs if n.get("oracle") is True]
    assert len(with_flag) == 4190, f"expected 4190 flagged, got {len(with_flag)}"
    assert all(n["provenance"] == "ego-future" for n in navs)
    assert man.divergences == (), f"unexpected divergence: {man.divergences}"
