"""v7 hierarchy vocabulary — PI redesign, 2026-08-23.

⭐ WHAT CHANGED FROM v6.1 AND WHY. Every item is a PI decision from the
`0e56dae2` / `01b24287` / `01bee851` review; the measured basis for each is
recorded beside it because these tuples SIZE LIVE TENSORS.

 1. ⛔ **SUPERSEDED 2026-08-27 — LAYERS ARE TEMPORAL AFTER ALL, WITH BANDS.**
    This file originally said "layers are ordinal, not temporal" and the emitter
    followed it. The PI corrected that: the bands are
    `OPERATIVE 0-2 s · TACTICAL 2-6 s · STRATEGIC 8-30 s`, and the ordinal rule
    ignored the clock entirely — it emitted a strategic token stamped
    `by_time_s: 6.0`, two seconds before the strategic band opens, on 435 clips.
    A manoeuvre is now clipped against each band and may appear in more than
    one. ⚠️ Nothing owns **6-8 s**; manoeuvres landing only there are reported
    as unassigned rather than absorbed. See `s2_geom_emit_v7` and C146.

 2. **Strategic tokens carry the `_FOLLOW_ROUTE` suffix.** A strategic goal is
    always "the overnext manoeuvre IN ORDER TO follow the route", and the name
    now says so. Without it, `TURN_RIGHT` at the strategic layer reads
    identically to `TURN_RIGHT` at the tactical layer while meaning something
    else entirely.

 3. **`HOLD_MAIN_ROAD` replaces `HOLD_CORRIDOR` strategically.** A corridor is a
    lane-level object; the strategic layer is route-level.

 4. **The tactical goal is a SET, not one token.** Several goals genuinely hold
    at once ("yield, then turn left, at a red light"). `TACTICAL_GOAL_EXCLUSIVE`
    below states which pairs may NOT co-occur, so a multi-label head can be
    checked rather than trusted.

 5. **`ANCHOR_GOAL` is no longer a goal.** It is the ARGUMENT REFERENCE every
    tactical goal is expressed against — the point the ego reaches at the end of
    the band. It was never a decision, so it should never have been a token.

 6. **No `ABSTAIN` in tactical goals.** Removed on PI instruction. The set may
    be `{FOLLOW_LANE}` but is never empty and never an explicit refusal.

 7. **Traffic-light goals carry the colour** — measured 638/4,729 CoTs (13.5 %)
    name a light, of which green 432, red 192, yellow 28. Plain
    `TRAFFIC_LIGHT_REACT` remains for the ~0 colourless case.

 8. **`ADAPT_SPEED_FOR_CURVE`** (not `..._FOR_TURNING`) as a LON ACTION —
    universal over junction turns AND plain curvature.

 9. **`ACCELERATE`** added to LON actions: a significant positive change had no
    token, so acceleration was filed as `CRUISE`.

10. **Nav commands are a separate INPUT vocabulary** (§NAV) — see the leak
    warning there, which is the most important caveat in this file.

⛔ TENSOR CONTRACT, unchanged from v6.1: every tuple here is a NEW name. No v6
tuple is edited, reordered or truncated, so no existing checkpoint changes
shape. Selecting v7 is a deliberate act in a run config.
"""
from __future__ import annotations

# ===========================================================================
# STRATEGIC — the OVERNEXT manoeuvre, in order to follow the route
# ===========================================================================
#: ⭐ THE `_FOLLOW_ROUTE` SUFFIX IS THE POINT (PI): the strategic layer answers
#: "what comes AFTER the thing I am planning, in order to stay on route".
STRATEGIC_GOAL_TOKENS_V7: tuple[str, ...] = (
    "FOLLOW_ROUTE",                 # no overnext manoeuvre in the band
    "TURN_LEFT_FOLLOW_ROUTE",
    "TURN_RIGHT_FOLLOW_ROUTE",
    "STOP_AT_FOLLOW_ROUTE",
    "EXIT_LEFT_FOLLOW_ROUTE",
    "EXIT_RIGHT_FOLLOW_ROUTE",
    "LANE_CHANGE_L_FOLLOW_ROUTE",
    "LANE_CHANGE_R_FOLLOW_ROUTE",
)

#: `HOLD_MAIN_ROAD` (PI) replaces `HOLD_CORRIDOR` here: a corridor is
#: lane-level, and this layer is route-level.
STRATEGIC_ACTION_TOKENS_V7: tuple[str, ...] = (
    "HOLD_MAIN_ROAD",               # nothing to prepare for
    "PREPARE_TURN_L_FOLLOW_ROUTE",
    "PREPARE_TURN_R_FOLLOW_ROUTE",
    "PREPARE_STOP_FOLLOW_ROUTE",
    "PREPARE_EXIT_FOLLOW_ROUTE",
    "PREPARE_LANE_CHANGE_FOLLOW_ROUTE",
    "RESUME_CRUISE_FOLLOW_ROUTE",
)

#: Uniform constraint slots (HIERARCHY §2). A strategic token WITHOUT these is
#: ambiguous about WHICH manoeuvre it names — the defect the PI found on
#: `01bee851`, whose three manoeuvres (+76 deg, -43 deg, +69 deg) are
#: distinguishable only by distance and time.
STRATEGIC_ARG_SLOTS: tuple[str, ...] = ("within_m", "by_time_s")


# ===========================================================================
# TACTICAL GOALS — a SET, expressed against the anchor reference
# ===========================================================================
TACTICAL_GOAL_TOKENS_V7: tuple[str, ...] = (
    "FOLLOW_LANE",
    "TURN_L", "TURN_R",
    "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD",
    "STOP_POINT",
    "SPEED_BAND",                 # ALWAYS present: the target speed interval
    "CORRIDOR_OFFSET",
    "EVADE_IN_CORRIDOR",          # STATIC obstacle / VRU -> lateral manoeuvre
    "OVERTAKE_VEHICLE",           # a SLOWER MOVING vehicle in front
    "MERGE",                      # from CoT: 45 clips (1.0 %)
    "GAP_TARGET",
    "REACT_ON_ONCOMING",          # renamed from WAIT_FOR_ONCOMING (PI)
    "TAKE_EXIT_L", "TAKE_EXIT_R", # from CoT terms, NOT geometry (PI)
    "TRAFFIC_LIGHT_REACT",
    "TRAFFIC_LIGHT_REACT_RED",
    "TRAFFIC_LIGHT_REACT_YELLOW",
    "TRAFFIC_LIGHT_REACT_GREEN",
    "LANE_CHANGE_L", "LANE_CHANGE_R",
)

#: ⭐ OVERTAKE vs EVADE — the PI's distinction, and it is about the OBSTACLE,
#: not the manoeuvre. Both are a lateral move around something; what separates
#: them is whether that something is MOVING:
#:   OVERTAKE_VEHICLE   : passing a SLOWER MOVING vehicle in front.
#:                        MEASURED: explicit "overtake" 13, "pass the <vehicle>
#:                        ahead" 19, lane-change-for-slower-traffic ~24.
#:   EVADE_IN_CORRIDOR  : a lateral adjustment for a STATIC obstacle or a VRU —
#:                        parked car (419), pedestrian (278), cyclist (64),
#:                        open door (0 — the case does not occur in this corpus).
#: ⚠️ They share the verb ("nudge left to pass the ..."), so ONLY the object
#: separates them. Testing evade first would file 419 parked-car evasions and
#: 13 real overtakes into one bucket.
OVERTAKE_VS_EVADE_NOTE = "moving obstacle => OVERTAKE; static/VRU => EVADE"

#: ⭐ THE ANCHOR IS THE ARGUMENT FRAME, NOT A GOAL (PI). Every tactical goal is
#: expressed against the point the ego reaches at the end of the band. It is
#: ALWAYS present, so as a token it carried no information and crowded out the
#: goals that do.
#: ⭐ `band_s` added 2026-08-27: the anchor must SAY which band it closes.
#: With tactical corrected to 2-6 s, an anchor reporting only `t_reach_s` is
#: ambiguous about the window it summarises — and that ambiguity is exactly how
#: the 0-6 s computation survived the band fix unnoticed.
#: ⚠️ These are ARG NAMES, not classes: extending this tuple does not touch the
#: frozen vocabulary and changes no tensor shape.
ANCHOR_ARG_SLOTS: tuple[str, ...] = ("goal_x_m", "goal_y_m", "t_reach_s",
                                     "band_s")

#: ⛔ PAIRS THAT MAY NEVER CO-OCCUR. A multi-label head can be CHECKED against
#: this rather than trusted; an emitter that produces a forbidden pair has a
#: bug, not an opinion. Everything not listed here MAY co-occur — e.g.
#: {YIELD_FOR_TURN_L, TURN_L, TRAFFIC_LIGHT_REACT_RED} is a normal set.
TACTICAL_GOAL_EXCLUSIVE: tuple[tuple[str, str], ...] = (
    ("TURN_L", "TURN_R"),                     # cannot turn both ways
    ("TURN_L", "FOLLOW_LANE"),                # a turn is not lane-following
    ("TURN_R", "FOLLOW_LANE"),
    ("LANE_CHANGE_L", "LANE_CHANGE_R"),
    ("LANE_CHANGE_L", "TURN_L"),              # lane change != junction turn
    ("LANE_CHANGE_R", "TURN_R"),
    ("YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R"),
    ("YIELD_FOR_TURN_L", "TURN_R"),
    ("YIELD_FOR_TURN_R", "TURN_L"),
    ("STOP_POINT", "FOLLOW_LANE"),            # a stop is not lane-following
    ("TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_GREEN"),
    ("TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW"),
    ("TRAFFIC_LIGHT_REACT_GREEN", "TRAFFIC_LIGHT_REACT_YELLOW"),
    ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED"),
    ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_GREEN"),
    ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_YELLOW"),
    ("OVERTAKE_VEHICLE", "FOLLOW_LANE"),
    ("OVERTAKE_VEHICLE", "EVADE_IN_CORRIDOR"),   # moving vs static obstacle
    ("TAKE_EXIT_L", "TAKE_EXIT_R"),
    ("TAKE_EXIT_L", "TURN_R"), ("TAKE_EXIT_R", "TURN_L"),
    # ⛔ ("SPEED_BAND", "STOP_POINT") REMOVED 2026-08-27 on PI instruction.
    # The rule read "a stop is not a held band" — true of the OLD SPEED_BAND,
    # which claimed the speed was HELD. The PI redefined it: *"Speed should be
    # actually always there because it describes the target speed … the speed
    # interval (max, min)."* Under that definition a stop is simply a target
    # band of `v_lo 0.0, v_hi ~0.0`, which is information, not a contradiction —
    # verified on the corpus (e.g. `018460b0`: band 0.0-0.0 beside a 4.1 s hold).
    # Keeping the exclusion would have flagged 335 perfectly coherent clips.
    #
    # ⚠️ This is a SEMANTIC constraint, not a tensor dimension, so removing it
    # does not touch the frozen vocabulary — no token changed.
    ("MERGE", "FOLLOW_LANE"),
)

# ===========================================================================
# ⭐ GOAL -> ADMISSIBLE ACTIONS (PI: "documentation where the goals are linked
# to possible actions")
# ===========================================================================
#: Which tactical ACTIONS may serve each tactical GOAL. A pair outside this map
#: is not forbidden by physics — it is a sign the emitter or the head has
#: produced an action that does not further the goal it was given, which is
#: exactly the `TURN_LEFT` + `HOLD_CORRIDOR` contradiction the PI found.
GOAL_ADMISSIBLE_LAT: dict[str, tuple[str, ...]] = {
    "FOLLOW_LANE": ("LANE_KEEP", "NUDGE_L", "NUDGE_R"),
    "TURN_L": ("TURN_L", "LANE_KEEP"),           # LANE_KEEP while still approaching
    "TURN_R": ("TURN_R", "LANE_KEEP"),
    "YIELD_FOR_TURN_L": ("TURN_L", "LANE_KEEP"),
    "YIELD_FOR_TURN_R": ("TURN_R", "LANE_KEEP"),
    "YIELD": ("LANE_KEEP", "NUDGE_L", "NUDGE_R"),
    "STOP_POINT": ("LANE_KEEP", "NUDGE_L", "NUDGE_R"),
    "SPEED_BAND": ("LANE_KEEP", "NUDGE_L", "NUDGE_R"),
    "CORRIDOR_OFFSET": ("NUDGE_L", "NUDGE_R", "LANE_KEEP"),
    "EVADE_IN_CORRIDOR": ("NUDGE_L", "NUDGE_R"),
    "OVERTAKE_VEHICLE": ("LANE_CHANGE_L", "LANE_CHANGE_R", "NUDGE_L", "NUDGE_R"),
    "MERGE": ("LANE_CHANGE_L", "LANE_CHANGE_R", "LANE_KEEP"),
    "GAP_TARGET": ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R"),
    "REACT_ON_ONCOMING": ("NUDGE_L", "NUDGE_R", "LANE_KEEP"),
    "TAKE_EXIT_L": ("LANE_CHANGE_L", "TURN_L", "LANE_KEEP"),
    "TAKE_EXIT_R": ("LANE_CHANGE_R", "TURN_R", "LANE_KEEP"),
    "TRAFFIC_LIGHT_REACT": ("LANE_KEEP",),
    "TRAFFIC_LIGHT_REACT_RED": ("LANE_KEEP",),
    "TRAFFIC_LIGHT_REACT_YELLOW": ("LANE_KEEP",),
    "TRAFFIC_LIGHT_REACT_GREEN": ("LANE_KEEP", "TURN_L", "TURN_R"),
    "LANE_CHANGE_L": ("LANE_CHANGE_L", "ABORT_LC"),
    "LANE_CHANGE_R": ("LANE_CHANGE_R", "ABORT_LC"),
}
GOAL_ADMISSIBLE_LON: dict[str, tuple[str, ...]] = {
    # ⚠️ BRAKE_TO belongs here: braking WHILE following the lane is ordinary
    # (traffic ahead). Omitting it produced 69 false "mismatch" reports — the
    # map must describe driving, not an idealised taxonomy.
    "FOLLOW_LANE": ("CRUISE", "FOLLOW", "ACCELERATE", "BRAKE_TO",
                    "ADAPT_SPEED_FOR_CURVE"),
    "TURN_L": ("ADAPT_SPEED_FOR_CURVE", "BRAKE_TO", "CREEP"),
    "TURN_R": ("ADAPT_SPEED_FOR_CURVE", "BRAKE_TO", "CREEP"),
    "YIELD_FOR_TURN_L": ("BRAKE_TO", "HOLD", "CREEP", "ADAPT_SPEED_FOR_CURVE"),
    "YIELD_FOR_TURN_R": ("BRAKE_TO", "HOLD", "CREEP", "ADAPT_SPEED_FOR_CURVE"),
    "YIELD": ("BRAKE_TO", "HOLD", "CREEP", "YIELD_MERGE"),
    "STOP_POINT": ("BRAKE_TO", "HOLD", "CREEP"),
    "SPEED_BAND": ("CRUISE", "FOLLOW"),
    "CORRIDOR_OFFSET": ("CRUISE", "FOLLOW", "ACCELERATE"),
    "EVADE_IN_CORRIDOR": ("CRUISE", "FOLLOW", "BRAKE_TO"),
    "OVERTAKE_VEHICLE": ("ACCELERATE", "CRUISE"),
    "MERGE": ("YIELD_MERGE", "FOLLOW", "ACCELERATE", "BRAKE_TO"),
    "GAP_TARGET": ("FOLLOW", "BRAKE_TO", "ACCELERATE", "YIELD_MERGE"),
    "REACT_ON_ONCOMING": ("BRAKE_TO", "FOLLOW", "CRUISE", "HOLD"),
    "TAKE_EXIT_L": ("ADAPT_SPEED_FOR_CURVE", "BRAKE_TO", "FOLLOW"),
    "TAKE_EXIT_R": ("ADAPT_SPEED_FOR_CURVE", "BRAKE_TO", "FOLLOW"),
    "TRAFFIC_LIGHT_REACT": ("BRAKE_TO", "CRUISE", "CREEP"),
    "TRAFFIC_LIGHT_REACT_RED": ("BRAKE_TO", "HOLD", "CREEP"),
    "TRAFFIC_LIGHT_REACT_YELLOW": ("BRAKE_TO", "CRUISE", "ACCELERATE"),
    "TRAFFIC_LIGHT_REACT_GREEN": ("CRUISE", "ACCELERATE", "ADAPT_SPEED_FOR_CURVE"),
    "LANE_CHANGE_L": ("CRUISE", "ACCELERATE", "FOLLOW"),
    "LANE_CHANGE_R": ("CRUISE", "ACCELERATE", "FOLLOW"),
}


def action_serves_goals(lat: str, lon: str, goals) -> dict:
    """Does this action pair serve AT LEAST ONE of the goals in the set?

    Returns the per-axis verdict. An action serving none of its goals is the
    `TURN_LEFT` + `HOLD_CORRIDOR` shape — a contradiction, not a nuance.
    """
    gs = [g for g in goals if g in GOAL_ADMISSIBLE_LAT]
    lat_ok = any(lat in GOAL_ADMISSIBLE_LAT.get(g, ()) for g in gs)
    lon_ok = any(lon in GOAL_ADMISSIBLE_LON.get(g, ()) for g in gs)
    return {"lat_serves": bool(lat_ok), "lon_serves": bool(lon_ok),
            "goals_checked": gs}

#: Tokens that cannot be derived from ego geometry and need PERCEPTION. They
#: are in the vocabulary so a perception-fed head can emit them; the geometry
#: emitter must NOT invent them.
TACTICAL_GOAL_NEEDS_PERCEPTION: frozenset[str] = frozenset({
    "GAP_TARGET", "REACT_ON_ONCOMING", "OVERTAKE_VEHICLE", "MERGE",
    "TAKE_EXIT_L", "TAKE_EXIT_R",
    "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
    "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN",
})

#: ⛔ REPRESENTABLE, NOT SCOREABLE below this n (the existing rule). MEASURED
#: over the 4,729-clip Alpamayo corpus:
#:   traffic light 638 (13.5 %) -> green 432, red 192, yellow 28
#:   oncoming      142 (3.0 %)  -- but see the WARNING in `alpamayo_semantics`:
#:                                 the dominant phrasing is lateral CLEARANCE
#:                                 ("nudge right due to an oncoming vehicle"),
#:                                 NOT waiting. Only ~19 mention waiting.
#:   overtake       13 (0.3 %)  -- the 326 "pass the ..." are almost all PARKED
#:                                 vehicles, which is EVADE_IN_CORRIDOR, not an
#:                                 overtake. This token is near-unpopulated.
TACTICAL_GOAL_UNDERPOWERED: frozenset[str] = frozenset({
    "OVERTAKE_VEHICLE", "WAIT_FOR_ONCOMING", "TRAFFIC_LIGHT_REACT_YELLOW",
    "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R",
})
GOAL_MIN_N_FOR_METRIC: int = 200


# ===========================================================================
# TACTICAL ACTIONS — the two control axes stay split
# ===========================================================================
TACTICAL_LAT_ACTIONS_V7: tuple[str, ...] = (
    "LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
    "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R",
)

#: ⭐ `ADAPT_SPEED_FOR_CURVE` (PI: keep the CURVE name so it covers a junction
#: turn AND plain speed reduction in a bend) and `ACCELERATE` (PI: a
#: significant positive change had no token, so it was filed as CRUISE).
TACTICAL_LON_ACTIONS_V7: tuple[str, ...] = (
    "FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
    "ADAPT_SPEED_FOR_CURVE", "ACCELERATE",
)

LAT_ACTION_ARG_SLOTS: tuple[str, ...] = ("within_m",)
LON_ACTION_ARG_SLOTS: tuple[str, ...] = ("v_target_ms", "within_m")


# ===========================================================================
# NAV COMMANDS — a MODEL INPUT, not a label
# ===========================================================================
#: ⭐ PI 2026-08-23: "a new struct vocab named Nav commands which will be fed as
#: INPUT to the models — it is very simple and describes the next manoeuvre for
#: navigation: turn_l, turn_r, or follow road, with time and distance args."
#:
#: ⛔⛔ THE LEAK WARNING, AND IT IS THE MOST IMPORTANT LINE IN THIS FILE.
#: On PhysicalAI our ONLY supplier of a route is THE EGO'S OWN FUTURE PATH.
#: A nav command DERIVED THAT WAY and fed back as an input is privileged
#: information at inference — the model would be told the answer. The standing
#: rules make this explicit: labels may use ego, INFERENCE IS VISION-ONLY
#: (PI 2026-08-03), and "a supplied route is optimistic by construction on
#: PhysicalAI".
#:
#: ⇒ The contract:
#:   * `provenance="nav-system"`  — a real navigation source. ADMISSIBLE as an
#:     inference input. This is what a production car actually has.
#:   * `provenance="ego-future"`  — derived from the recorded future. USABLE FOR
#:     TRAINING ONLY, and any eval using it is an ORACLE arm that must be
#:     labelled as such and never compared against a vision-only arm.
#: `NAV_PROVENANCE` exists so the distinction is carried in the data and cannot
#: be lost in a pipeline hop. An arm that cannot state its nav provenance is
#: not evaluable.
NAV_COMMAND_TOKENS: tuple[str, ...] = (
    "NAV_FOLLOW_ROAD", "NAV_TURN_L", "NAV_TURN_R",
)
NAV_ARG_SLOTS: tuple[str, ...] = ("distance_m", "time_s")
NAV_PROVENANCE: tuple[str, ...] = ("nav-system", "ego-future")


# ===========================================================================
# helpers
# ===========================================================================
def validate_goal_set(tokens) -> list[str]:
    """Return the violations in a tactical goal SET. Empty list = valid."""
    s = set(tokens)
    bad: list[str] = []
    unknown = s - set(TACTICAL_GOAL_TOKENS_V7)
    if unknown:
        bad.append(f"unknown token(s): {sorted(unknown)}")
    if not s:
        bad.append("empty goal set — v7 has no ABSTAIN; emit FOLLOW_LANE")
    for a, b in TACTICAL_GOAL_EXCLUSIVE:
        if a in s and b in s:
            bad.append(f"mutually exclusive: {a} + {b}")
    return bad


def geometry_emittable(tokens) -> set[str]:
    """The subset an EGO-GEOMETRY emitter is allowed to produce."""
    return set(tokens) - TACTICAL_GOAL_NEEDS_PERCEPTION


# --- the freeze, enforced at RUNTIME ----------------------------------------
#: ⛔ FROZEN 2026-08-23 on PI instruction ("stabilize now the vocabulary and
#: keep it constant"). `stack/tests/test_vocab_v7_frozen.py` pins the tuples;
#: this function pins what an EMITTER actually writes, which is the half a
#: static test cannot see — tokens assembled by concatenation only exist at
#: runtime.
ALL_V7_TOKENS: frozenset[str] = frozenset(
    tuple(STRATEGIC_GOAL_TOKENS_V7) + tuple(STRATEGIC_ACTION_TOKENS_V7)
    + tuple(TACTICAL_GOAL_TOKENS_V7) + tuple(TACTICAL_LAT_ACTIONS_V7)
    + tuple(TACTICAL_LON_ACTIONS_V7) + tuple(NAV_COMMAND_TOKENS))


def assert_frozen(token: str, *, where: str = "") -> str:
    """Return ``token``, or raise if it is not in the frozen v7 vocabulary.

    Call this at every emit site. A token the heads have no slot for must fail
    LOUDLY at label-build time — silently writing it produces a corpus whose
    class indices are off by one past the insertion point, which surfaces
    months later as a model that mislabels every class after it.
    """
    if token not in ALL_V7_TOKENS:
        raise ValueError(
            f"{token!r} is not in the FROZEN v7 vocabulary"
            + (f" (emitted at {where})" if where else "")
            + ". Adding a token is a versioned change: create a *_V8 tuple "
              "behind a version switch, never edit v7.")
    return token


# --- DEFINITIONS: the single source the documentation is generated from -----
#: PI 2026-08-23: *"In the last documentation I lost the strategic and tactical
#: goals as list and their definitions, add these to the matrix."*
#:
#: These live in CODE, not in a markdown file, for one reason: a definition
#: kept beside the tuple cannot drift from it. The design docs are GENERATED
#: from this map, and `test_vocab_v7_frozen` asserts every frozen token has an
#: entry -- so a token can never ship undefined, and a definition can never
#: outlive its token. Losing the definitions between two documents is exactly
#: what happened, and a markdown file is what allowed it.
DEFINITIONS: dict[str, str] = {
    # -- STRATEGIC GOALS (the OVERNEXT manoeuvre, 8-30 s) --------------------
    "FOLLOW_ROUTE":
        "No overnext manoeuvre in the lookahead: stay on the route as it runs. "
        "The default, and ~84 % of clips -- a skew a consumer must weight.",
    "TURN_LEFT_FOLLOW_ROUTE":
        "The manoeuvre AFTER the tactical one is a left turn, then the route "
        "continues. The _FOLLOW_ROUTE suffix is the PI's: it names what happens "
        "once the turn is done, so the token is not mistaken for the turn the "
        "plan is currently executing.",
    "TURN_RIGHT_FOLLOW_ROUTE": "As TURN_LEFT_FOLLOW_ROUTE, to the right.",
    "STOP_AT_FOLLOW_ROUTE":
        "The overnext event is a stop (light, sign, queue), after which the "
        "route continues.",
    "EXIT_LEFT_FOLLOW_ROUTE":
        "The overnext manoeuvre leaves the current road to the left "
        "(ramp/split), then follows the new road.",
    "EXIT_RIGHT_FOLLOW_ROUTE": "As EXIT_LEFT_FOLLOW_ROUTE, to the right.",
    "LANE_CHANGE_L_FOLLOW_ROUTE":
        "The overnext manoeuvre is a lane change to the left required by the "
        "ROUTE, not by traffic.",
    "LANE_CHANGE_R_FOLLOW_ROUTE": "As LANE_CHANGE_L_FOLLOW_ROUTE, to the right.",

    # -- STRATEGIC ACTIONS (what to do NOW about the overnext manoeuvre) -----
    "HOLD_MAIN_ROAD":
        "Nothing to prepare for: remain on the main road. Replaces the old "
        "HOLD_CORRIDOR, which read as a lateral instruction and produced the "
        "'HOLD_CORRIDOR while turning' contradiction the PI caught.",
    "PREPARE_TURN_L_FOLLOW_ROUTE":
        "Begin setting up for a left turn that is NOT in the current 6 s plan. "
        "Args carry within_m and by_time_s, so it is explicit that this does "
        "NOT affect the current tactical manoeuvre -- the PI's requirement.",
    "PREPARE_TURN_R_FOLLOW_ROUTE":
        "As PREPARE_TURN_L_FOLLOW_ROUTE, but for a right turn.",
    "PREPARE_STOP_FOLLOW_ROUTE": "Set up for a stop beyond the tactical horizon.",
    "PREPARE_EXIT_FOLLOW_ROUTE":
        "Set up to leave the road at the overnext opportunity.",
    "PREPARE_LANE_CHANGE_FOLLOW_ROUTE":
        "Position for a route-required lane change beyond the plan.",
    "RESUME_CRUISE_FOLLOW_ROUTE":
        "The constraint that forced a slowdown ends within the strategic band; "
        "return to road-class speed.",

    # -- TACTICAL GOALS (0-6 s, MULTI-LABEL) --------------------------------
    "FOLLOW_LANE":
        "Stay in the current lane along the road as it runs. The fallback when "
        "no other goal applies.",
    "TURN_L":
        "Execute a left turn WITHIN the 6 s plan. Per the PI: a turn inside the "
        "plan is TACTICAL, never strategic.",
    "TURN_R": "As TURN_L, to the right.",
    "YIELD_FOR_TURN_L":
        "A left turn that must first give way -- the ego holds, then turns. ONE "
        "goal, not two, because the yield and the turn are a single intent.",
    "YIELD_FOR_TURN_R": "As YIELD_FOR_TURN_L, to the right.",
    "YIELD": "Give way without a turn -- to a sign, a merge, or a hazard.",
    "STOP_POINT":
        "Come to rest at a specific place; args carry within_m and hold_for_s.",
    "SPEED_BAND":
        "The TARGET SPEED INTERVAL over the tactical band: min and max speed "
        "between 2 s and 6 s (args v_lo_ms / v_hi_ms / band_s). ALWAYS PRESENT "
        "on every clip -- PI 2026-08-27: 'Speed should be actually always there "
        "because it describes the target speed'. It was previously gated on the "
        "speed being HELD, which deleted the target speed from exactly the "
        "scenes where it matters most; whether the speed is held is the "
        "LONGITUDINAL ACTION's verdict (CRUISE / ACCELERATE / BRAKE_TO), and "
        "the `held` arg here is descriptive only. A stop is a band of 0.0-0.0, "
        "which is information, not a contradiction. MEASURED medians: stopping "
        "0.00-1.40, turning 4.52-7.11, cruising 12.78-13.16 m/s.",
    "CORRIDOR_OFFSET":
        "Hold a lateral offset inside the lane without leaving it.",
    "EVADE_IN_CORRIDOR":
        "PI's definition: a lateral adjustment for a STATIC obstacle or a VRU -- "
        "parked car, open door, cyclist, pedestrian. Stays in the corridor; does "
        "NOT change lane. The object does not move; that is what separates it "
        "from OVERTAKE_VEHICLE.",
    "OVERTAKE_VEHICLE":
        "PI's definition: pass a SLOWER MOVING vehicle ahead. The object moves. "
        "Extraction order matters -- 'pass the parked car' is EVADE, 'pass the "
        "slow truck' is OVERTAKE, and mis-ordering files 419 parked-car "
        "evasions as overtakes.",
    "MERGE": "Join a stream of traffic from a ramp or a closing lane.",
    "GAP_TARGET":
        "Aim for a specific gap in an adjacent stream; args name the gap.",
    "REACT_ON_ONCOMING":
        "Respond to oncoming traffic -- renamed from WAIT_FOR_ONCOMING at the "
        "PI's request, because the response is not always a wait.",
    "TAKE_EXIT_L":
        "Leave the road to the left. Extracted from CoT TERMS, not geometry -- "
        "the PI's instruction, and geometry cannot tell an exit from a bend.",
    "TAKE_EXIT_R": "As TAKE_EXIT_L, to the right.",
    "TRAFFIC_LIGHT_REACT":
        "React to a traffic light whose colour was NOT extracted.",
    "TRAFFIC_LIGHT_REACT_RED": "React to a light observed RED.",
    "TRAFFIC_LIGHT_REACT_YELLOW": "React to a light observed YELLOW or amber.",
    "TRAFFIC_LIGHT_REACT_GREEN":
        "React to a light observed GREEN -- a reaction, not the absence of one: "
        "the ego proceeds BECAUSE the light permits it.",
    "LANE_CHANGE_L": "Move to the lane on the left within the plan.",
    "LANE_CHANGE_R": "Move to the lane on the right within the plan.",

    # -- TACTICAL LATERAL ACTIONS -------------------------------------------
    "LANE_KEEP": "Hold the lane centre.",
    "ABORT_LC": "Abandon a lane change already begun and return.",
    "NUDGE_L": "Small lateral displacement left, staying in lane.",
    "NUDGE_R": "Small lateral displacement right, staying in lane.",

    # -- TACTICAL LONGITUDINAL ACTIONS --------------------------------------
    "FOLLOW": "Track the lead vehicle's speed, holding a gap.",
    "CRUISE": "Hold speed; |dv| < 1.0 m/s across the plan.",
    "YIELD_MERGE": "Slow to let a merging agent in.",
    "BRAKE_TO": "Decelerate to a target speed or a stop point.",
    "CREEP": "Move slowly forward, typically to see past an occlusion.",
    "HOLD": "Remain stationary at a standstill, brakes applied.",
    "ADAPT_SPEED_FOR_CURVE":
        "PI's token, replacing ADAPT_SPEED_FOR_TURNING to make it universal: "
        "adjust speed for path curvature -- a junction turn OR a sharp bend. Set "
        "automatically when a turn manoeuvre is detected.",
    "ACCELERATE":
        "PI's addition: a significant positive speed change, dv > +1.5 m/s "
        "across the plan.",

    # -- NAV COMMANDS (MODEL INPUT, not a label) ----------------------------
    "NAV_FOLLOW_ROAD":
        "Navigation says: continue on the current road. This is an INPUT to the "
        "model, derived here from the ego future -- an ORACLE, admissible for "
        "training only, and never a target.",
    "NAV_TURN_L":
        "Navigation says: turn left; args carry distance and time to the turn.",
    "NAV_TURN_R": "Navigation says: turn right; args as NAV_TURN_L.",
}


#: ⭐ GOAL vs ACTION, stated per token (PI 2026-08-27: "please differentiate
#: between strategic goal and action in the description").
#:
#:   GOAL   = WHAT is to be achieved, and by when. It carries the target and
#:            its args (`by_time_s`, `within_m`). It does NOT say how.
#:   ACTION = WHAT TO DO NOW about that goal. It is the control-level output
#:            the layer below must execute.
#:
#: Example, same scene: the strategic GOAL is `TURN_LEFT_FOLLOW_ROUTE`
#: ("a left turn is coming at t+24 s, then the route continues"); the strategic
#: ACTION is `PREPARE_TURN_L_FOLLOW_ROUTE` ("begin setting up for it now") —
#: and crucially that action does NOT change the current tactical manoeuvre,
#: which is why its args carry the distance and time to the turn.
ROLE_OF: dict[str, str] = {
    **{t: "goal" for t in STRATEGIC_GOAL_TOKENS_V7},
    **{t: "action" for t in STRATEGIC_ACTION_TOKENS_V7},
    **{t: "goal" for t in TACTICAL_GOAL_TOKENS_V7},
    **{t: "action" for t in TACTICAL_LAT_ACTIONS_V7},
    **{t: "action" for t in TACTICAL_LON_ACTIONS_V7},
    **{t: "input" for t in NAV_COMMAND_TOKENS},
}

#: Which LAYER each token belongs to -- used to render the matrix by layer.
LAYER_OF: dict[str, str] = {
    **{t: "strategic goal" for t in STRATEGIC_GOAL_TOKENS_V7},
    **{t: "strategic action" for t in STRATEGIC_ACTION_TOKENS_V7},
    **{t: "tactical goal" for t in TACTICAL_GOAL_TOKENS_V7},
    **{t: "tactical lat action" for t in TACTICAL_LAT_ACTIONS_V7},
    **{t: "tactical lon action" for t in TACTICAL_LON_ACTIONS_V7},
    **{t: "nav command" for t in NAV_COMMAND_TOKENS},
}


#: ⛔⛔ TOKENS THE PIPELINE CANNOT CURRENTLY PRODUCE, AND WHY.
#:
#: MEASURED 2026-08-27, after the PI's MERGE question: **13 of 52 frozen tokens
#: (25 %) were never emitted on the whole 4,719-clip corpus**. `LANE_CHANGE_L`
#: was one of them — it was frozen, defined, documented in the matrix, and had
#: NO extraction path at all, while 174 clips stated a lane change in plain
#: language. Every existing check passed.
#:
#: ⇒ **Freezing a token is not the same as being able to emit one.** A frozen
#: vocabulary needs a REACHABILITY test: for every token either an extraction
#: path exists, or it is listed HERE with the reason. `test_vocab_reachability`
#: enforces exactly that, so a dead class can never again be invisible.
#:
#: ⚠️ These are honest gaps, not excuses. Each names what it would take.
#: Two gaps share one cause, so the reason is written once.
_EXIT_REASON = (
    "the CoT states exits WITHOUT a time, so an exit cannot be placed in the "
    "8-30 s strategic band; and geometry cannot separate an exit from a bend, "
    "which is why the PI directed that exits come from TERMS. Needs a timed "
    "exit signal.")
_LC_REASON = (
    "same timing problem as the exits: a lane change is stated without a time, "
    "and MEASURED, the median claimed lane change does not displace a full lane "
    "width inside any window we observe (+1.02 m over the control at -3..+8 s, "
    "against a ~3.5 m lane). It cannot be placed in the strategic band on "
    "present evidence.")

NOT_YET_EXTRACTABLE: dict[str, str] = {
    "ABORT_LC":
        "needs a lane change BEGUN and then abandoned — a two-phase lateral "
        "signature we do not yet detect, and rare enough that no calibration "
        "reference exists in the corpus.",
    "CORRIDOR_OFFSET":
        "needs a SUSTAINED lateral offset held inside the lane, distinguished "
        "from the transient nudge that EVADE_IN_CORRIDOR already covers. The "
        "geometry is available; the threshold has not been calibrated against "
        "any independent reference, and guessing it would manufacture labels.",
    "YIELD_MERGE":
        "needs MERGE (a perception claim) to coincide with a measured "
        "deceleration. MERGE itself now stands at 82 clips after the hazard "
        "guard, so the joint event is too rare to calibrate.",
    "EXIT_LEFT_FOLLOW_ROUTE": _EXIT_REASON,
    "EXIT_RIGHT_FOLLOW_ROUTE": _EXIT_REASON,
    "PREPARE_EXIT_FOLLOW_ROUTE": _EXIT_REASON,
    "LANE_CHANGE_L_FOLLOW_ROUTE": _LC_REASON,
    "LANE_CHANGE_R_FOLLOW_ROUTE": _LC_REASON,
    "PREPARE_LANE_CHANGE_FOLLOW_ROUTE": _LC_REASON,
}
