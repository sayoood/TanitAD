"""v7 hierarchy vocabulary — PI redesign, 2026-08-23.

⭐ WHAT CHANGED FROM v6.1 AND WHY. Every item is a PI decision from the
`0e56dae2` / `01b24287` / `01bee851` review; the measured basis for each is
recorded beside it because these tuples SIZE LIVE TENSORS.

 1. **Layers are ordinal, not temporal.** The tactical goal is the NEXT
    manoeuvre; the strategic goal is the OVERNEXT one. A time split assigned a
    manoeuvre to whichever band it happened to straddle — `01b24287`'s single
    +97 deg turn (t+5.9-11.9 s) crossed the 6 s/8 s boundary and was promoted to
    the strategic layer, where it did not belong.

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
    "REDUCE_TO_FOLLOW_ROUTE",
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
    "SPEED_BAND",                 # restored (PI): the held target-speed band
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
ANCHOR_ARG_SLOTS: tuple[str, ...] = ("goal_x_m", "goal_y_m", "t_reach_s")

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
    ("SPEED_BAND", "STOP_POINT"),                # a stop is not a held band
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
