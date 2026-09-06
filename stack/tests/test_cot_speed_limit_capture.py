"""A POSTED SPEED LIMIT MUST NOT BE MATCHED AND THEN THROWN AWAY.

⛔ THE DEFECT THIS PINS (MEASURED 2026-09-06). `CotTokens.speed_limit` was a
`bool`, referenced at two lines and never read, and its pattern
``re.compile(r"\\bspeed limit\\b")`` had NO CAPTURING GROUP. Over the 4,729 clips
of `Sayood/tanitad-alpamayo2-augmentation` the CoT states **57** posted-limit
values read off a sign; every one was matched and discarded at the regex.

⛔⛔ AND 18 OF THE 57 (31.6 %) STATE NO UNIT. 70 km/h = 19.44 m/s, 70 mph =
31.29 m/s -- a **1.61x** spread. A number without its unit is inadmissible, so
`speed_limit_ms` is None whenever `speed_limit_unit_missing` is True, and that
is asserted here rather than left to a convention. *(Same family as the
`anchors.pt` control-units trap, where a correct formula under the wrong unit
produced 396 g of lateral acceleration and looked exactly like an answer.)*

⭐ THE SCOPE IS CAPTURE-AND-BANK. `test_no_token_is_emitted_at_any_state` pins
that NOTHING reaches the goal vocabulary or inference: the labels are
ego-coupled (median ratio limit/ego 1.01) and grounded 0 of 57, and
extract-vs-supply is an OPEN PI decision.

Pre-registration: `TanitAD Research Lab/Architecture & Inference/Research/
2026-09-06-cot-loader/PREREG.md` (PR-2a … PR-2f).
"""
import re

import pytest

from tanitad.data import cot_tokens_v7 as COT

#: state, text, expected state, expected value -- KNOWN VALUES, not observed.
CONTROLS = [
    # the no-information value, EXACTLY: no phrase means no reading at all.
    ("constant_no_phrase", "The ego proceeds through the intersection.", None, None),
    ("empty", "", None, None),
    ("none", None, None, None),
    # a real reading, both units
    ("read_kmh", "A 70 km/h speed limit sign appears ahead.", "READ", 70),
    ("read_mph", "The speed limit sign reads 35 mph.", "READ", 35),
    # a real reading with NO unit -- the inadmissible third
    ("read_no_unit", "A 70 speed limit sign is visible ahead.", "READ", 70),
    # a LANGUAGE PRIOR, which is the `road_class` circularity in VLM costume
    ("hedged", "Residential areas typically have a 25 mph speed limit.",
     "HEDGED", None),
    # NOTE: a definitional quirk kept ON PURPOSE so this module and
    # `.../2026-09-06-speed-limit-source/code/classify_limits.py` agree
    # clip-for-clip on all 4,729 clips: `not visible` is in the HEDGE pattern
    # and HEDGED is tested before NEGATED.
    ("negated_via_hedge_word",
     "The speed limit sign is not visible in this scene.", "HEDGED", None),
    # ...so the NEGATED branch needs its own case, or it is unreachable -- and
    # an unreachable branch has measured nothing, exactly like a guard that is
    # never called.
    ("negated_true", "There is no speed limit sign in view.", "NEGATED", None),
    ("mention_no_value", "The ego respects the speed limit while turning.",
     "MENTION", None),
]


@pytest.mark.parametrize("name,text,state,value", CONTROLS,
                         ids=[c[0] for c in CONTROLS])
def test_controls_read_their_known_values(name, text, state, value):
    r = COT.speed_limit_reading(text)
    assert (r.state if r else None) == state
    assert (r.value if r else None) == value


def test_every_state_is_reachable():
    """All four states occur in the control set -- no dead branch."""
    seen = {(COT.speed_limit_reading(t).state if COT.speed_limit_reading(t) else None)
            for _, t, _, _ in CONTROLS}
    assert {"READ", "HEDGED", "NEGATED", "MENTION", None} == seen


# ---------------------------------------------------------------- units rule
def test_a_unitless_reading_never_carries_a_metric_value():
    """⛔ THE WHOLE POINT. 1.61x is the cost of guessing."""
    r = COT.speed_limit_reading("A 70 speed limit sign is visible ahead.")
    assert r.state == "READ" and r.value == 70
    assert r.unit is None
    assert r.unit_missing is True
    assert r.value_ms is None, "a unit-less number must not be converted"


@pytest.mark.parametrize("text,unit,ms", [
    ("A 70 km/h speed limit sign appears ahead.", "kph", 19.4444),
    ("A 70 kph speed limit sign appears ahead.", "kph", 19.4444),
    ("The speed limit sign reads 70 mph.", "mph", 31.2928),
])
def test_unit_conversion_is_exact(text, unit, ms):
    r = COT.speed_limit_reading(text)
    assert r.unit == unit
    assert r.value_ms == pytest.approx(ms, abs=1e-4)


def test_the_two_units_really_do_differ_by_1_61x():
    """The control that makes the units rule non-negotiable.

    NOTE the tolerance: `value_ms` is stored rounded to 4 dp, so the ratio of
    two stored values is only knowable to ~1e-5 relative. The exact ratio is
    0.44704 / (1/3.6) = 1.609344; asserting it tighter than the stored
    precision would be asserting a number the artifact does not carry.
    """
    k = COT.speed_limit_reading("A 70 km/h speed limit sign appears ahead.")
    m = COT.speed_limit_reading("The speed limit sign reads 70 mph.")
    assert m.value_ms / k.value_ms == pytest.approx(1.609344, abs=1e-4)
    # and the UNROUNDED constants, which is where the 1.61x actually lives
    assert (COT._SPEED_LIMIT_MS["mph"] / COT._SPEED_LIMIT_MS["kph"]
            == pytest.approx(1.609344, abs=1e-9))


# --------------------------------------------------- deliberate-regression arm
_PREFIX_PATTERN = re.compile(r"\bspeed limit\b")     # the pre-2026-09-06 regex


def _prefix_reading(text):
    """⛔ THE PRE-FIX EXTRACTION, in shape: presence only, no capture."""
    if not text or not _PREFIX_PATTERN.search(text.lower()):
        return None
    return COT.SpeedLimitReading(state="READ")       # value stays None


def _captures_a_value(fn):
    r = fn("A 70 km/h speed limit sign appears ahead.")
    return r is not None and r.value is not None


def test_deliberate_regression_arm_on_the_capture():
    """ONE checker, TWO extractors, opposite verdicts required."""
    assert _captures_a_value(_prefix_reading) is False, (
        "the checker did not detect the ORIGINAL defect -- it measures nothing")
    assert _captures_a_value(COT.speed_limit_reading) is True


def test_the_old_pattern_still_has_no_capturing_group():
    """The presence flag's own semantics are UNCHANGED, on purpose.

    Every banked label was built with them; the value is captured ALONGSIDE the
    boolean by a second pattern, never instead of it.
    """
    assert COT._SPEED_LIMIT.groups == 0
    assert COT._SPEED_LIMIT_READ.groups > 0


# ------------------------------------------------------------------- wiring
def test_extract_actually_populates_the_new_fields():
    """WIRING, not correctness: `speed_limit_reading` is REACHED from extract().

    A parser that works when called and is called from nowhere is the same
    defect the boolean had -- it was assigned and never read.
    """
    c = COT.extract("A 70 km/h speed limit sign appears ahead of the ego.")
    assert c.speed_limit is True                     # the old flag still fires
    assert c.speed_limit_state == "READ"
    assert c.speed_limit_value == 70
    assert c.speed_limit_unit == "kph"
    assert c.speed_limit_unit_missing is False
    assert c.speed_limit_ms == pytest.approx(19.4444, abs=1e-4)
    d = c.as_dict()
    for k in ("speed_limit_state", "speed_limit_value", "speed_limit_unit",
              "speed_limit_unit_missing", "speed_limit_ms"):
        assert k in d, "the banked label record must carry %s" % k


def test_extract_on_a_clean_sentence_reads_the_no_information_value():
    c = COT.extract("The ego proceeds through the intersection.")
    assert c.speed_limit is False
    assert c.speed_limit_state is None
    assert c.speed_limit_value is None
    assert c.speed_limit_ms is None


def test_negated_text_does_not_set_the_presence_flag():
    """The negation stripper still owns the boolean; the state records why."""
    c = COT.extract("There is no speed limit sign in view.")
    assert c.speed_limit is False
    assert c.speed_limit_state == "NEGATED"
    assert c.speed_limit_value is None


# --------------------------------------------------------- the SCOPE guard
@pytest.mark.parametrize("text", [t for _, t, _, _ in CONTROLS if t])
def test_no_token_is_emitted_at_any_state(text):
    """⛔ CAPTURE AND BANK IS THE WHOLE SCOPE.

    Nothing enters the goal vocabulary and nothing reaches inference. Coverage
    is 57/4,729 (1.21 %), grounded share is 0/57, and the readings are
    ego-coupled at a median ratio of 1.01 -- so a supplied channel would be the
    nav-echo defect. Extract-vs-supply is an OPEN PI decision.
    """
    goals = COT.goals_from_cot(text)
    assert not any("SPEED" in tok or "LIMIT" in tok for tok in goals), goals
