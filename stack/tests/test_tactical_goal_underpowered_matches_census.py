"""TACTICAL_GOAL_UNDERPOWERED must equal the census set below the metric floor.

WHY THIS TEST EXISTS (and why adding the four missing names would not have been
a fix):

``vocab_v7.TACTICAL_GOAL_UNDERPOWERED`` is a hand-maintained frozenset that sits
THREE LINES ABOVE ``GOAL_MIN_N_FOR_METRIC = 200`` -- a mirror of a threshold that
is stated, as a number, immediately beneath it.  On 2026-09-07 the set held five
names.  Against the DataFlyWheel's v8 census (4,719 clips) it was wrong in BOTH
directions:

  * FOUR tokens below the floor were missing, and
  * ONE member, ``WAIT_FOR_ONCOMING``, is not in ``TACTICAL_GOAL_TOKENS_V7`` at
    all -- it was renamed to ``REACT_ON_ONCOMING`` on PI instruction (vocab_v7.py
    line 106) and this set was never updated.  A dead string in a live set.

This is the programme's third instance of the same defect class:

  1. ``FORWARD_KEYS`` mirrored ``RefCV3Model.forward``'s signature and silently
     dropped channels TWICE.  The durable fix was to DERIVE the requirement from
     ``inspect.signature`` -- see test_rl_forward_keys_cover_signature.py, whose
     structure (headline assertion + vacuity gate + mutation controls) this test
     copies deliberately.
  2. An ``anchor_chance`` string hardcoded ``1/128`` while the bank held 117.
  3. This.

So the fix is not a longer hand-list.  It is a pin against the CENSUS.

WHICH THRESHOLD, AND WHY IT MATTERS (the ambiguity, resolved):

Two DIFFERENT questions were being served by one constant:

  (a) "too rare to SCORE a metric on"  -> GOAL_MIN_N_FOR_METRIC = 200
  (b) "too rare to TRAIN a class on"   -> the census's cut, n < 30

``TACTICAL_GOAL_UNDERPOWERED`` is pinned to (a), the SCOREABILITY question,
because that is what its own surrounding comment declares ("REPRESENTABLE, NOT
SCOREABLE below this n") and what both published consumers assert:

  * Project Steering/GOALS_AND_CLAIMS.md, row D-TLIGHT-1
  * TanitAD Research Lab/.../2026-09-06-refcv5-redesign/SPEC.md, item 13

Question (b) is NOT dropped -- it is named by ``GOAL_MIN_N_FOR_TRAINING = 30``
so that the census instrument stops hardcoding that literal.  Neither question
is left implicit.

WHY THE PIN READS RAW ``n`` AND NOT THE CENSUS'S ``status`` FIELD:

The census computes its status as ``t in TACTICAL_GOAL_UNDERPOWERED or n < 30``
(vocab_fill_census.py line 78).  Pinning membership against ``status`` would
therefore be CIRCULAR -- the census already folded this frozenset into the field
we would be checking against, so an over-inclusive set could never be detected.
Every assertion below reads the raw counts.
``test_pinning_on_the_status_field_would_be_circular`` keeps that reasoning
falsifiable instead of leaving it in a comment.
"""

import json
from pathlib import Path

import pytest

# The banked census artifact.  Tracked in the repo on purpose: the DataFlyWheel
# original lives in an UNTRACKED incoming/ directory, and a test that reads an
# untracked path passes or fails according to who happens to have synced what.
_REPO_ROOT = Path(__file__).resolve().parents[2]
CENSUS_PATH = (
    _REPO_ROOT
    / "TanitAD Research Lab"
    / "Architecture & Inference"
    / "Research"
    / "2026-09-07-underpowered-pin"
    / "raw"
    / "vocab_fill_census.json"
)

FIELD = "g_tac.goals"


def _census():
    """The census artifact, or a hard failure naming the path.

    Never returns a default.  A census that cannot be read must FAIL this
    suite, not silently empty the expected set and turn every assertion below
    into a tautology.
    """
    assert CENSUS_PATH.is_file(), (
        "census artifact not found at {p}.\n"
        "This test pins vocab_v7.TACTICAL_GOAL_UNDERPOWERED against it; without "
        "it there is nothing to pin against and the pin must not be assumed to "
        "hold. Re-bank it from the DataFlyWheel release "
        "(incoming/2026-09-01-v8-tacsit-release/raw/vocab_fill_census.json)."
        .format(p=CENSUS_PATH)
    )
    return json.loads(CENSUS_PATH.read_text(encoding="utf-8"))


def _counts():
    """{token: n} for the tactical-goal field, from the RAW counts."""
    field = _census()["fields"][FIELD]
    return {tok: int(rec["n"]) for tok, rec in field["tokens"].items()}


def _expected_below(threshold):
    """The token set the census puts below `threshold`. The derivation itself."""
    return {tok for tok, n in _counts().items() if n < threshold}


def _diff(actual, expected):
    """(missing, extra) -- the exact comparison, factored so mutations can drive it.

    Returned sorted so failure messages are stable and diffable.
    """
    return sorted(expected - actual), sorted(actual - expected)


# ---------------------------------------------------------------------------
# Vacuity gates. These run first because every assertion below is only as
# meaningful as the artifact it reads.
# ---------------------------------------------------------------------------


def test_census_artifact_is_readable_and_populated():
    counts = _counts()
    assert counts, "census carries no %s tokens -- the pin would be vacuous" % FIELD
    assert sum(counts.values()) > 0, (
        "every %s count is zero. A census of an empty corpus would put EVERY "
        "token below the floor and make the expected set trivially the whole "
        "vocabulary." % FIELD
    )
    scope = _census().get("_scope", "")
    assert "4719" in scope or "4,719" in scope, (
        "census scope changed to {s!r}. The membership pinned below was derived "
        "on the 4,719-clip v8 train+eval corpus; a different corpus is a "
        "different question and the set must be re-derived, not assumed."
        .format(s=scope)
    )


def test_census_token_set_matches_the_declared_vocabulary():
    """Pinning against a STALE census would be as bad as the hand-list.

    If the census predates a vocabulary change, its counts describe tokens that
    no longer exist (or omit tokens that now do) and the pin silently drifts.
    """
    from tanitad.models import vocab_v7

    declared = set(vocab_v7.TACTICAL_GOAL_TOKENS_V7)
    censused = set(_counts())
    assert censused == declared, (
        "the census token set and TACTICAL_GOAL_TOKENS_V7 disagree.\n"
        "  in vocabulary, absent from census : {a}\n"
        "  in census, absent from vocabulary : {b}\n"
        "The census is stale relative to the vocabulary (or vice versa); "
        "re-run vocab_fill_census.py before trusting any membership below."
        .format(a=sorted(declared - censused), b=sorted(censused - declared))
    )


# ---------------------------------------------------------------------------
# The load-bearing assertions.
# ---------------------------------------------------------------------------


def test_underpowered_equals_the_census_set_below_the_metric_floor():
    """The headline. Names the missing AND the extra tokens, never just fails."""
    from tanitad.models import vocab_v7

    threshold = vocab_v7.GOAL_MIN_N_FOR_METRIC
    expected = _expected_below(threshold)
    assert expected, (
        "no token is below GOAL_MIN_N_FOR_METRIC=%d -- the derivation is broken, "
        "not the frozenset" % threshold
    )

    actual = set(vocab_v7.TACTICAL_GOAL_UNDERPOWERED)
    missing, extra = _diff(actual, expected)

    counts = _counts()
    assert not missing and not extra, (
        "TACTICAL_GOAL_UNDERPOWERED has drifted from the census.\n"
        "\n"
        "  MISSING (census says n < {t}, set does not list them):\n{m}\n"
        "  EXTRA (set lists them, census says n >= {t} or token is unknown):\n{e}\n"
        "\n"
        "  threshold : GOAL_MIN_N_FOR_METRIC = {t}  (SCOREABILITY, not trainability;\n"
        "              the trainability floor is GOAL_MIN_N_FOR_TRAINING)\n"
        "  census    : {p}\n"
        "\n"
        "Do NOT fix this by editing the frozenset until you have checked WHY the\n"
        "counts moved. A token crossing the floor changes whether a published\n"
        "metric may be quoted as a RATE for it -- see GOALS_AND_CLAIMS.md row\n"
        "D-TLIGHT-1, which quotes this threshold by name."
        .format(
            t=threshold,
            m="".join("      %-30s n = %s\n" % (k, counts.get(k, "ABSENT"))
                      for k in missing) or "      (none)\n",
            e="".join("      %-30s n = %s\n" % (k, counts.get(k, "NOT IN CENSUS"))
                      for k in extra) or "      (none)\n",
            p=CENSUS_PATH,
        )
    )


def test_underpowered_names_are_real_vocabulary_tokens():
    """The dangling-token guard -- the WAIT_FOR_ONCOMING defect, pinned directly.

    A member that is not a declared token matches nothing, ever. It cannot be
    caught by any consumer because no consumer will ever look it up; it simply
    sits there looking like coverage. Held separately from the headline so the
    failure says WHICH defect it is.
    """
    from tanitad.models import vocab_v7

    declared = set(vocab_v7.TACTICAL_GOAL_TOKENS_V7)
    unknown = sorted(set(vocab_v7.TACTICAL_GOAL_UNDERPOWERED) - declared)
    assert not unknown, (
        "TACTICAL_GOAL_UNDERPOWERED names {u}, which are not in "
        "TACTICAL_GOAL_TOKENS_V7.\n"
        "validate_goal_set() would reject these as unknown tokens, so they can "
        "never match anything -- the set only LOOKS like it covers them.\n"
        "This is how WAIT_FOR_ONCOMING survived its own rename to "
        "REACT_ON_ONCOMING (vocab_v7.py line 106).".format(u=unknown)
    )


def test_the_two_thresholds_are_distinct_and_both_declared():
    """The ambiguity that caused this, kept from silently re-forming.

    'Too rare to score a metric on' and 'too rare to train a class on' are
    different questions. One constant serving both is how the set ended up
    matching neither.
    """
    from tanitad.models import vocab_v7

    metric = vocab_v7.GOAL_MIN_N_FOR_METRIC
    training = vocab_v7.GOAL_MIN_N_FOR_TRAINING
    assert training < metric, (
        "GOAL_MIN_N_FOR_TRAINING ({tr}) is not below GOAL_MIN_N_FOR_METRIC "
        "({me}). If they have converged, one of them is redundant -- say which "
        "and delete it, rather than leaving two names for one number."
        .format(tr=training, me=metric)
    )
    assert _expected_below(training) < _expected_below(metric), (
        "the trainability set is not a strict subset of the scoreability set; "
        "the two thresholds are no longer separating anything on this corpus"
    )


def test_pinning_on_the_status_field_would_be_circular():
    """Keeps the anti-circularity reasoning falsifiable rather than a comment.

    vocab_fill_census.py line 78 computes
        status = "UNDERPOWERED" if t in TACTICAL_GOAL_UNDERPOWERED or n < 30
    so the status field is a UNION of this frozenset with the n<30 rule. A pin
    against `status` can therefore never detect an OVER-inclusive frozenset:
    every extra member marks itself UNDERPOWERED on the next census run.
    """
    from tanitad.models import vocab_v7

    field = _census()["fields"][FIELD]
    by_status = {t for t, r in field["tokens"].items()
                 if r.get("status") == "UNDERPOWERED"}
    by_raw_n = _expected_below(vocab_v7.GOAL_MIN_N_FOR_TRAINING)

    assert by_status >= by_raw_n, (
        "census status is not a superset of its own n < {t} rule -- the "
        "instrument changed and this test's model of it is stale. Re-read "
        "vocab_fill_census.py before trusting the status field for anything."
        .format(t=vocab_v7.GOAL_MIN_N_FOR_TRAINING)
    )

    # Whatever status marks BEYOND the count rule must be explained by frozenset
    # membership -- that IS the circularity, and it is expected to be non-empty
    # once a census is re-run against the corrected set (MERGE and TAKE_EXIT_R
    # sit between the two floors). What must never happen is status diverging
    # for a reason this test does not model.
    unexplained = sorted(
        by_status - by_raw_n - set(vocab_v7.TACTICAL_GOAL_UNDERPOWERED))
    assert not unexplained, (
        "the census marks {u} UNDERPOWERED, and neither its count rule nor "
        "frozenset membership explains it. The instrument has a third rule this "
        "test does not know about; do not pin anything on `status` until it "
        "does.".format(u=unexplained)
    )


# ---------------------------------------------------------------------------
# Mutation controls.
#
# A guard nobody has broken on purpose is decoration. These reintroduce the
# real defect and require the comparison to catch it BY NAME, and require it to
# accept a correct set -- without touching the live frozenset.
# ---------------------------------------------------------------------------


def test_mutation_a_dropped_token_is_detected_and_named():
    """Reintroduce the original defect: remove a member, require it named."""
    expected = _expected_below(200)
    assert len(expected) >= 2, "need >=2 expected tokens to run this mutation"

    victim = sorted(expected)[0]
    missing, extra = _diff(expected - {victim}, expected)

    assert missing == [victim], (
        "the drift check failed to NAME a deliberately dropped token; it cannot "
        "catch the defect it exists for. got missing=%r" % (missing,)
    )
    assert extra == [], "dropping a token must not report a spurious extra"


def test_mutation_a_stale_token_is_detected_and_named():
    """The other direction, using the ACTUAL historical defect string.

    A length-only or missing-only check would have passed the five-member set
    forever, because WAIT_FOR_ONCOMING padded the count while covering nothing.
    """
    expected = _expected_below(200)
    missing, extra = _diff(expected | {"WAIT_FOR_ONCOMING"}, expected)

    assert extra == ["WAIT_FOR_ONCOMING"], (
        "a stale member was not detected -- this is precisely the half of the "
        "comparison the five-member set needed. got extra=%r" % (extra,)
    )
    assert missing == [], "adding a token must not report a spurious missing"


def test_mutation_the_correct_set_is_accepted():
    """The discriminating half. Without it, an always-fail check passes both
    mutations above while being worthless."""
    expected = _expected_below(200)
    assert _diff(set(expected), expected) == ([], []), (
        "the comparison flags a CORRECT set -- it is a blanket assertion, not a "
        "discriminating one"
    )


def test_mutation_the_comparison_actually_reads_the_counts():
    """Proves the expected set is DERIVED, not a hardcoded list wearing a
    derivation's clothes. If _expected_below ignored its argument, the two
    thresholds would return identical sets and this would fail."""
    lo = _expected_below(30)
    hi = _expected_below(200)
    assert lo != hi, (
        "the expected set does not vary with the threshold -- it is not reading "
        "the census counts, and the pin is a hand-list again"
    )
    counts = _counts()
    for tok in hi - lo:
        assert 30 <= counts[tok] < 200, (
            "%s is in the 200-set but not the 30-set, yet its count is %d; the "
            "derivation disagrees with itself" % (tok, counts[tok])
        )


def test_mutation_a_zero_count_token_would_be_reported_not_hidden():
    """A documented edge, kept honest.

    n == 0 is 'below the floor' arithmetically, but the census gives such tokens
    a STRONGER status (UNREACHABLE / NOT_EXTRACTABLE / ZERO_SUPPLY) -- absent is
    not the same claim as rare. No tactical-goal token is currently zero, so the
    choice is inert; this asserts that it is inert rather than leaving a silent
    behaviour to be discovered later.
    """
    counts = _counts()
    zeros = sorted(t for t, n in counts.items() if n == 0)
    assert not zeros, (
        "{z} now have n == 0. The pin currently treats 'below the floor' as "
        "including zero, which would file an ABSENT token as merely rare. Decide "
        "explicitly whether TACTICAL_GOAL_UNDERPOWERED should carry them before "
        "this test is updated to pass.".format(z=zeros)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
