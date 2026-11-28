"""⛔ THE v7 VOCABULARY IS FROZEN (PI, 2026-08-23). This test is the freeze.

**PI, verbatim:** *"we will stabilize now the vocabulary and keep it constant"*.

Every tuple below is a TENSOR DIMENSION. A head sized 22 against a 23-token
vocabulary does not crash — it silently mislabels every class past the
insertion point, and a checkpoint trained before the change cannot be compared
with one trained after. That is why this test pins the exact ORDER and not just
the length: appending is a migration, inserting is a corruption.

⇒ **If this test fails, the vocabulary changed. That is either a mistake, or a
deliberate versioned change that needs a NEW tuple (`*_V8`) plus a version
switch — never an edit to these.** The APPEND-never-INSERT contract from v6
applies unchanged.

The counts are stated in the test name so a failure message names the shape.
"""
from tanitad.models import vocab_v7 as V

# --- the frozen tuples ------------------------------------------------------

STRATEGIC_GOALS_8 = (
    "FOLLOW_ROUTE",
    "TURN_LEFT_FOLLOW_ROUTE",
    "TURN_RIGHT_FOLLOW_ROUTE",
    "STOP_AT_FOLLOW_ROUTE",
    "EXIT_LEFT_FOLLOW_ROUTE",
    "EXIT_RIGHT_FOLLOW_ROUTE",
    "LANE_CHANGE_L_FOLLOW_ROUTE",
    "LANE_CHANGE_R_FOLLOW_ROUTE",
)
STRATEGIC_ACTIONS_7 = (
    "HOLD_MAIN_ROAD",
    "PREPARE_TURN_L_FOLLOW_ROUTE",
    "PREPARE_TURN_R_FOLLOW_ROUTE",
    "PREPARE_STOP_FOLLOW_ROUTE",
    "PREPARE_EXIT_FOLLOW_ROUTE",
    "PREPARE_LANE_CHANGE_FOLLOW_ROUTE",
    "RESUME_CRUISE_FOLLOW_ROUTE",
)
TACTICAL_GOALS_22 = (
    "FOLLOW_LANE",
    "TURN_L",
    "TURN_R",
    "YIELD_FOR_TURN_L",
    "YIELD_FOR_TURN_R",
    "YIELD",
    "STOP_POINT",
    "SPEED_BAND",
    "CORRIDOR_OFFSET",
    "EVADE_IN_CORRIDOR",
    "OVERTAKE_VEHICLE",
    "MERGE",
    "GAP_TARGET",
    "REACT_ON_ONCOMING",
    "TAKE_EXIT_L",
    "TAKE_EXIT_R",
    "TRAFFIC_LIGHT_REACT",
    "TRAFFIC_LIGHT_REACT_RED",
    "TRAFFIC_LIGHT_REACT_YELLOW",
    "TRAFFIC_LIGHT_REACT_GREEN",
    "LANE_CHANGE_L",
    "LANE_CHANGE_R",
)
TACTICAL_LAT_ACTIONS_8 = (
    "LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
    "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R",
)
TACTICAL_LON_ACTIONS_8 = (
    "FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
    "ADAPT_SPEED_FOR_CURVE", "ACCELERATE",
)
NAV_COMMANDS_3 = ("NAV_FOLLOW_ROAD", "NAV_TURN_L", "NAV_TURN_R")


def _same(frozen, live, name):
    live = tuple(live)
    assert live == frozen, (
        f"{name} CHANGED — this is a tensor dimension.\n"
        f"  frozen ({len(frozen)}): {frozen}\n"
        f"  live   ({len(live)}): {live}\n"
        f"  If deliberate: add a NEW tuple ({name}_V8) behind a version switch. "
        f"Do NOT edit this one.")


def test_strategic_goals_frozen_at_8():
    _same(STRATEGIC_GOALS_8, V.STRATEGIC_GOAL_TOKENS_V7, "STRATEGIC_GOAL_TOKENS_V7")


def test_strategic_actions_frozen_at_7():
    """7, not 8: REDUCE_TO_FOLLOW_ROUTE was removed on PI instruction."""
    _same(STRATEGIC_ACTIONS_7, V.STRATEGIC_ACTION_TOKENS_V7,
          "STRATEGIC_ACTION_TOKENS_V7")
    assert "REDUCE_TO_FOLLOW_ROUTE" not in V.STRATEGIC_ACTION_TOKENS_V7


def test_tactical_goals_frozen_at_22():
    _same(TACTICAL_GOALS_22, V.TACTICAL_GOAL_TOKENS_V7, "TACTICAL_GOAL_TOKENS_V7")


def test_tactical_lat_actions_frozen_at_8():
    _same(TACTICAL_LAT_ACTIONS_8, V.TACTICAL_LAT_ACTIONS_V7,
          "TACTICAL_LAT_ACTIONS_V7")


def test_tactical_lon_actions_frozen_at_8():
    _same(TACTICAL_LON_ACTIONS_8, V.TACTICAL_LON_ACTIONS_V7,
          "TACTICAL_LON_ACTIONS_V7")


def test_nav_commands_frozen_at_3():
    _same(NAV_COMMANDS_3, V.NAV_COMMAND_TOKENS, "NAV_COMMAND_TOKENS")


def test_no_token_appears_on_two_axes():
    """A token meaning two things breaks the head that reads it.

    `TURN_L` legitimately appears as BOTH a tactical goal and a lateral action
    — that is the v7 design (the goal is the intent, the action is the control
    output). What must never happen is a collision WITHIN one head's vocabulary.
    """
    for name, toks in (("strategic goals", STRATEGIC_GOALS_8),
                       ("strategic actions", STRATEGIC_ACTIONS_7),
                       ("tactical goals", TACTICAL_GOALS_22),
                       ("lat actions", TACTICAL_LAT_ACTIONS_8),
                       ("lon actions", TACTICAL_LON_ACTIONS_8),
                       ("nav", NAV_COMMANDS_3)):
        assert len(set(toks)) == len(toks), f"duplicate token inside {name}"


def test_emitter_only_emits_frozen_tokens():
    """The emitter cannot invent a token the heads have no slot for."""
    import importlib.util
    import pathlib
    p = (pathlib.Path(__file__).resolve().parents[1] / "scripts"
         / "s2_geom_emit_v7.py")
    if not p.exists():                      # emitter is optional at test time
        return
    src = p.read_text(encoding="utf-8", errors="ignore")
    known = set(STRATEGIC_GOALS_8) | set(STRATEGIC_ACTIONS_7) | \
        set(TACTICAL_GOALS_22) | set(TACTICAL_LAT_ACTIONS_8) | \
        set(TACTICAL_LON_ACTIONS_8) | set(NAV_COMMANDS_3)
    import re
    # Quoted ALL-CAPS identifiers that look like vocabulary tokens.
    for m in re.finditer(r'"([A-Z][A-Z0-9_]{5,})"', src):
        tok = m.group(1)
        if tok.endswith("_FOLLOW_ROUTE") or tok.startswith(
                ("TURN_", "NAV_", "TRAFFIC_LIGHT", "TAKE_EXIT", "LANE_CHANGE",
                 "YIELD", "PREPARE_")):
            # A token built by concatenation ("YIELD_FOR_TURN_" + side) is a
            # PREFIX of a frozen token. Accept that — the runtime guard in
            # `vocab_v7.assert_frozen` checks the assembled value, which is
            # the check that actually binds. Reject anything else.
            ok = tok in known or any(k.startswith(tok) for k in known)
            assert ok, (
                f"emitter emits {tok!r}, which is NOT in the frozen v7 "
                f"vocabulary and is not a prefix of any frozen token — "
                f"the head has no slot for it")


def test_every_frozen_token_has_a_definition():
    """A token cannot ship undefined, and a definition cannot outlive a token.

    The PI lost the goal definitions between two documents. Keeping them in a
    markdown file is what allowed that; keeping them beside the tuple, with
    this test, is what stops it recurring.
    """
    missing = sorted(t for t in V.ALL_V7_TOKENS if t not in V.DEFINITIONS)
    assert not missing, f"frozen tokens with NO definition: {missing}"
    orphan = sorted(t for t in V.DEFINITIONS if t not in V.ALL_V7_TOKENS)
    assert not orphan, f"definitions for tokens not in the vocabulary: {orphan}"


def test_definitions_are_substantive():
    """A one-word definition is not a definition."""
    thin = sorted(t for t, d in V.DEFINITIONS.items() if len(d.split()) < 4)
    assert not thin, f"definitions too thin to be useful: {thin}"


def test_vocab_reachability():
    """⛔ EVERY frozen token is either EMITTABLE or declared unreachable.

    MEASURED 2026-08-27: 13 of 52 tokens (25 %) were never produced on the full
    4,719-clip corpus. `LANE_CHANGE_L` was frozen, defined, and documented while
    having NO extraction path, and 174 clips stated a lane change in plain
    language. Freezing a token is not the same as being able to emit one, and
    nothing checked the difference.

    A head sized to the frozen vocabulary trains dead classes for every token in
    `NOT_YET_EXTRACTABLE` — so that list must be a deliberate, reasoned
    declaration rather than an accident.
    """
    unreachable = set(V.NOT_YET_EXTRACTABLE)
    unknown = sorted(unreachable - set(V.ALL_V7_TOKENS))
    assert not unknown, f"declared unreachable but not in the vocabulary: {unknown}"
    for tok, why in V.NOT_YET_EXTRACTABLE.items():
        assert len(why.split()) >= 8, (
            f"{tok} is declared unreachable with no real reason: {why!r}")
    # the declaration must not quietly grow: this is the number the PI was told
    # 9 -> 8 on 2026-08-28: CORRIDOR_OFFSET became extractable via the
    # PI-designed CoT-term route (side-only constraint; two pattern classes
    # with the object-side sign inverted). The geometric-threshold objection
    # that put it on this list was never answered — it was ROUTED AROUND.
    assert len(unreachable) == 8, (
        f"{len(unreachable)} tokens declared unreachable, expected 8. "
        f"If a token became extractable, REMOVE it here. If a new one became "
        f"unreachable, that is a regression, not a bookkeeping update.")
