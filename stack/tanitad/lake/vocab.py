"""The FROZEN v3 goal vocabulary (TanitDataSet rev-3 §7.4) as code.

One tokenizable vocabulary everywhere: dataset labels, CoC traces, conditioning,
planner option space, eval. Mirrors ``V3_GOAL_VOCABULARY_V1.md`` (FROZEN
2026-07-19) byte-for-byte — a change here is a vocabulary version bump, never a
silent drift. This module is PURE (only stdlib + the token tables): it defines
the slots, the per-slot token sets, the two banding functions (VTARGET / HEADWAY),
the provenance axis, and the ``unknown``-stamped empty goal that the kinematic
minter (``goal_labels``) and the VLM pass (deferred) fill in per slot.

Design rules honored (V3 §Design rules):
  R1 every token is discrete/banded — no continuous leak (VTARGET/HEADWAY are
     *banded*, never a raw m/s or second).
  R2 a goal is a compositional tuple of slot tokens.
  R3 every slot carries provenance ``{kinematic|map|vlm|human|sim|engineered}``;
     an honest gap is the token ``unknown`` with provenance ``unknown`` — never a
     nearest-guess.
  R5 one shared vocab, slot-prefixed on the wire (``LON:follow_lead``).
"""

from __future__ import annotations

import math
from typing import Any

# --------------------------------------------------------------------------- #
# Provenance axis (R3) + the honest-gap sentinel                               #
# --------------------------------------------------------------------------- #
PROVENANCE = ("kinematic", "map", "vlm", "human", "sim", "engineered", "unknown")
UNKNOWN = "unknown"          # the honest-gap token AND the "not-yet-labeled" prov


# --------------------------------------------------------------------------- #
# STRATEGIC goal = ⟨MISSION, ROUTE, LANEOBJ, SPEEDPOLICY, STYLE, RISK, ODD⟩     #
# (V3 §STRATEGIC — 34 tokens)                                                   #
# --------------------------------------------------------------------------- #
STRATEGIC_TOKENS: dict[str, tuple[str, ...]] = {
    "MISSION": ("route_follow", "free_navigate", "explore", "plan_pullover",
                "mrm_now"),
    "ROUTE": ("follow", "straight", "turn_left", "turn_right", "exit_left",
              "exit_right", "merge", "u_turn", "roundabout"),
    "LANEOBJ": ("keep", "prefer_left_faster", "prefer_right_exit", "any"),
    "SPEEDPOLICY": ("nominal", "cap_low", "cap_med", "cap_high"),
    "STYLE": ("max_availability", "comfort", "eco", "dynamic",
              "degraded_caution"),
    "RISK": ("nominal", "elevated_weather", "elevated_visibility",
             "elevated_anomaly"),
    "ODD": ("in_odd", "odd_exit_ahead", "capability_degrading"),
}

# --------------------------------------------------------------------------- #
# TACTICAL goal = ⟨VTARGET, VSOURCE, LONMODE, LATMANEUVER, HEADWAY, DYN,        #
#   RULECTX, SIGNAL, INTERACT, TACPOINT, LIGHTSTATE⟩  (V3 §TACTICAL — 76 tokens)#
# --------------------------------------------------------------------------- #


def _fmt(x: float) -> str:
    """Edge formatter: ``15.0 -> '15'``, ``12.5 -> '12.5'`` (matches the doc)."""
    return str(int(x)) if float(x).is_integer() else str(x)


def _vtarget_tokens() -> tuple[str, ...]:
    """The 23 non-uniform VTARGET bands (V3 Q1): ``v_stop`` + 10x1 m/s (0-10) +
    12x2.5 m/s (10-40). Generated so the count + edges cannot drift by hand."""
    toks = ["v_stop"]
    for lo in range(0, 10):                          # 1 m/s bands, (0,1]..(9,10]
        toks.append(f"v({_fmt(lo)}-{_fmt(lo + 1)}]")
    lo = 10.0
    while lo < 40.0 - 1e-9:                           # 2.5 m/s bands, (10,12.5]..
        toks.append(f"v({_fmt(lo)}-{_fmt(lo + 2.5)}]")
        lo += 2.5
    return tuple(toks)


VTARGET_TOKENS = _vtarget_tokens()
assert len(VTARGET_TOKENS) == 23, len(VTARGET_TOKENS)

# HEADWAY nominal time-gap bands (V3: 5 tokens). The floats are the NOMINAL
# time-headways the tokens name; ``headway_band`` snaps a measured gap to the
# nearest nominal via the midpoint thresholds below.
HEADWAY_TOKENS = ("hw_0.8s", "hw_1.2s", "hw_1.45s", "hw_1.75s", "hw_2.5s+")
_HEADWAY_NOMINAL = (0.8, 1.2, 1.45, 1.75, 2.5)

TACTICAL_TOKENS: dict[str, tuple[str, ...]] = {
    "VTARGET": VTARGET_TOKENS,
    "VSOURCE": ("sign_limit", "lead_constrained", "curve_constrained",
                "road_class_default", "traffic_flow"),
    "LONMODE": ("free_cruise", "follow_lead", "close_gap", "open_gap",
                "stop_at_point", "hold_stop", "launch", "creep", "coast"),
    "LATMANEUVER": ("lane_keep", "lc_left", "lc_right", "abort_lc", "merge_in",
                    "yield_merge", "nudge_left", "nudge_right", "pull_over"),
    "HEADWAY": HEADWAY_TOKENS,
    "DYN": ("gentle", "normal", "firm", "max"),
    "RULECTX": ("conform", "justified_deviation.obstacle_avoidance",
                "justified_deviation.rescue_corridor",
                "justified_deviation.stopped_vehicle_pass",
                "justified_deviation.instructed"),
    "SIGNAL": ("none", "indicator_left", "indicator_right", "hazard",
               "headlight_flash", "horn"),
    "INTERACT": ("none", "yield_to_k", "assert_gap_k", "cooperate_merge_k",
                 "respond_emergency"),
    "TACPOINT": ("none", "stop_line", "merge_point", "creep_point",
                 "clear_point"),
    "LIGHTSTATE": ("proceed", "prepare_stop", "stop_at_line", "creep_check"),
}

STRATEGIC_SLOTS = tuple(STRATEGIC_TOKENS)
TACTICAL_SLOTS = tuple(TACTICAL_TOKENS)
GOAL_SLOTS = STRATEGIC_SLOTS + TACTICAL_SLOTS
ALL_TOKENS = {**STRATEGIC_TOKENS, **TACTICAL_TOKENS}

# Frozen-count self-check against the AUTHORITATIVE per-slot ``n`` column of
# V3_GOAL_VOCABULARY_V1 (the enumerated token rows — the ground truth). NOTE the
# doc's *header* summary line ("TACTICAL … 76 tokens", "Total: 110 tokens, 17
# slots") is STALE relative to its own enumerated rows, which give 7 strategic + 11
# tactical = 18 slots and 34 + 80 = 114 tokens. We follow the enumerated tokens
# (the real vocabulary content) and pin each slot's count so a typo in a tuple
# fails loudly. Flagged in the task report as an ambiguity resolved to the faithful
# reading (the enumeration, not the stale header).
_EXPECTED_N = {
    "MISSION": 5, "ROUTE": 9, "LANEOBJ": 4, "SPEEDPOLICY": 4, "STYLE": 5,
    "RISK": 4, "ODD": 3,                                          # strategic = 34
    "VTARGET": 23, "VSOURCE": 5, "LONMODE": 9, "LATMANEUVER": 9, "HEADWAY": 5,
    "DYN": 4, "RULECTX": 5, "SIGNAL": 6, "INTERACT": 5, "TACPOINT": 5,
    "LIGHTSTATE": 4,                                              # tactical = 80
}
for _s, _n in _EXPECTED_N.items():
    assert len(ALL_TOKENS[_s]) == _n, (_s, len(ALL_TOKENS[_s]), _n)
assert sum(len(v) for v in STRATEGIC_TOKENS.values()) == 34
assert sum(len(v) for v in TACTICAL_TOKENS.values()) == 80
assert len(GOAL_SLOTS) == 18            # 7 strategic + 11 tactical (enumerated)


# --------------------------------------------------------------------------- #
# Banding functions (R1: continuous signal -> discrete token, never a leak)    #
# --------------------------------------------------------------------------- #
STOP_SPEED_MS = 0.5          # <= this 85th-pct free-flow speed reads as v_stop


def vtarget_band(speed_ms: float) -> str:
    """A free-flow speed [m/s] -> its VTARGET band token (non-uniform, V3 Q1).

    ``<= STOP_SPEED_MS`` -> ``v_stop``; ``(0,10]`` -> the 1 m/s band; ``(10,40]``
    -> the 2.5 m/s band; ``> 40`` clamps to the top band. The band is right-closed
    (``v(9-10]`` includes exactly 10.0), matching the token names."""
    s = float(speed_ms)
    if not math.isfinite(s) or s <= STOP_SPEED_MS:
        return "v_stop"
    if s <= 10.0:
        lo = math.ceil(s) - 1                        # (lo, lo+1], lo in 0..9
        lo = min(max(lo, 0), 9)
        return f"v({_fmt(lo)}-{_fmt(lo + 1)}]"
    idx = math.ceil((s - 10.0) / 2.5)                # 2.5 m/s band index 1..12
    idx = min(max(idx, 1), 12)
    lo = 10.0 + 2.5 * (idx - 1)
    return f"v({_fmt(lo)}-{_fmt(lo + 2.5)}]"


def headway_band(seconds: float) -> str:
    """A measured time-headway [s] -> its nearest HEADWAY band token (V3, 5)."""
    g = float(seconds)
    if not math.isfinite(g):
        return UNKNOWN
    # midpoint thresholds between the nominal band centers
    mids = [(_HEADWAY_NOMINAL[i] + _HEADWAY_NOMINAL[i + 1]) / 2.0
            for i in range(len(_HEADWAY_NOMINAL) - 1)]
    for tok, hi in zip(HEADWAY_TOKENS[:-1], mids):
        if g < hi:
            return tok
    return HEADWAY_TOKENS[-1]


# --------------------------------------------------------------------------- #
# The goal value type — a per-slot provenance-stamped token (R2/R3)            #
# --------------------------------------------------------------------------- #
def slot(token: str, provenance: str) -> dict[str, str]:
    """One provenance-stamped slot value: ``{'token', 'prov'}`` (JSON-native)."""
    if provenance not in PROVENANCE:
        raise ValueError(f"bad provenance {provenance!r}; want one of {PROVENANCE}")
    return {"token": str(token), "prov": provenance}


def unknown_slot() -> dict[str, str]:
    """The honest-gap slot value (R3): token ``unknown``, provenance ``unknown``.
    This is what a VLM/map-pending slot carries until the deferred pass fills it."""
    return {"token": UNKNOWN, "prov": UNKNOWN}


def empty_goal() -> dict[str, dict[str, str]]:
    """A full 17-slot goal with EVERY slot ``unknown``/``unknown`` (R3).

    The kinematic minter overwrites the slots it can honestly derive now; the
    VLM/map slots stay ``unknown`` until the deferred Cosmos-Reason2 pass."""
    return {s: unknown_slot() for s in GOAL_SLOTS}


def is_pending(goal: dict[str, dict[str, str]], slot_name: str) -> bool:
    """True iff ``slot_name`` is still an honest gap (VLM/map-pending)."""
    return goal.get(slot_name, unknown_slot())["prov"] == UNKNOWN


def validate_goal(goal: dict[str, dict[str, str]]) -> None:
    """Assert a goal tuple is well-formed: exactly the 17 slots, each token in
    its slot's vocab (or ``unknown``), provenance in ``PROVENANCE`` (raises)."""
    if set(goal) != set(GOAL_SLOTS):
        missing = set(GOAL_SLOTS) - set(goal)
        extra = set(goal) - set(GOAL_SLOTS)
        raise ValueError(f"goal slots off: missing={missing} extra={extra}")
    for s, v in goal.items():
        if set(v) < {"token", "prov"}:
            raise ValueError(f"slot {s}: value must have token+prov, got {v}")
        tok, prov = v["token"], v["prov"]
        if prov not in PROVENANCE:
            raise ValueError(f"slot {s}: bad provenance {prov!r}")
        if tok != UNKNOWN and tok not in ALL_TOKENS[s]:
            raise ValueError(f"slot {s}: token {tok!r} not in vocab {ALL_TOKENS[s]}")
        if tok == UNKNOWN and prov not in (UNKNOWN, "vlm", "map", "human"):
            # an unknown token is only honest with a pending/deferred provenance
            raise ValueError(f"slot {s}: token 'unknown' with prov {prov!r}")


def goal_provenance_summary(goal: dict[str, dict[str, str]]) -> dict[str, int]:
    """Count slots per provenance — the at-a-glance "how much is real vs pending"
    for a data card / report."""
    out: dict[str, int] = {p: 0 for p in PROVENANCE}
    for v in goal.values():
        out[v["prov"]] = out.get(v["prov"], 0) + 1
    return out


# --------------------------------------------------------------------------- #
# v1.1 CANDIDATE — DISTANCE-TO-MANEUVER (NOT enrolled; the freeze is intact)   #
# --------------------------------------------------------------------------- #
# A route/tactical SHAPE is not an instruction: `roundabout` is a description,
# `roundabout, exit in 40 m` is a command, and `stop_at_point in 12 m` is what a
# planner can actually act on. nuPlan / CARLA / OpenDRIVE and every production
# nav stack hand the planner a DISTANCE; we currently hand ours a class name.
# `scripts/refb_labels.route_from_future_v3` mints these bands today.
#
# WHY THIS BLOCK IS NOT IN STRATEGIC_TOKENS/TACTICAL_TOKENS: enrolling a slot
# changes GOAL_SLOTS, empty_goal(), validate_goal() and every embedding sized
# off them — that is a VOCABULARY VERSION BUMP (v1 -> v1.1) with a migration
# note, and V3_GOAL_VOCABULARY_V1.md is FROZEN. So the tokens + the banding
# function live here, pinned by tests, and enrolment is Sayed's call. Nothing
# above this line changes; every frozen count assertion still holds.
#
# METRES, not seconds — decided: (1) speed-invariant (the same junction is "5 s"
# at 10 m/s and "2.5 s" at 20 m/s: two tokens for one instruction); (2) a
# deceleration profile is set by distance (v^2 = 2ad), not time; (3) maps and
# nav stacks speak metres, so a later map/VLM fills the identical slot with no
# unit conversion that would need the label-time speed.
ROUTEDIST_TOKENS = ("d_now", "d_10_25", "d_25_50", "d_50_100", "d_100_200",
                    "d_200_plus", "d_none", "d_unknown")
_ROUTEDIST_EDGES_M = (10.0, 25.0, 50.0, 100.0, 200.0)
ROUTEDIST_LOOKED_ENOUGH_M = 100.0

# The candidate slots a v1.1 bump would add, with their counts, so the doc<->code
# pin keeps working the day they are enrolled.
V11_CANDIDATE_TOKENS: dict[str, tuple[str, ...]] = {
    "ROUTEDIST": ROUTEDIST_TOKENS,      # strategic: distance to the next route
                                        # maneuver
    "TACDIST": ROUTEDIST_TOKENS,        # tactical: distance to the TACPOINT
                                        # (stop point) — same bands on purpose
}
assert len(ROUTEDIST_TOKENS) == 8
assert not set(V11_CANDIDATE_TOKENS) & set(GOAL_SLOTS), \
    "a candidate slot collides with a FROZEN slot name"


def routedist_band(dist_m: float | None, observed_arc_m: float = 0.0,
                   looked_enough_m: float = ROUTEDIST_LOOKED_ENOUGH_M) -> str:
    """Metres-to-the-next-maneuver -> band token (R1: banded, never a raw float).

    ``dist_m is None`` means "no maneuver found in the observed future". That is
    ``d_none`` only when at least ``looked_enough_m`` of road was actually
    observed; otherwise ``d_unknown``. Keeping those two apart is the v2->v2.1
    lesson (a silent fallback that conflated "cannot judge" with a real class
    poisoned the route prior) applied to the distance axis."""
    if dist_m is None:
        return "d_none" if observed_arc_m >= looked_enough_m else "d_unknown"
    d = float(dist_m)
    if not math.isfinite(d) or d < 0.0:
        return "d_unknown"
    for edge, tok in zip(_ROUTEDIST_EDGES_M, ROUTEDIST_TOKENS):
        if d < edge:
            return tok
    return ROUTEDIST_TOKENS[len(_ROUTEDIST_EDGES_M)]        # d_200_plus


def to_wire(goal: dict[str, dict[str, str]]) -> dict[str, str]:
    """Slot-prefixed flat form (R5), e.g. ``{'LONMODE': 'LON:free_cruise'}`` —
    handy for CoC prompts / embeddings. Unknown slots are dropped."""
    pref = {"LONMODE": "LON", "LATMANEUVER": "LAT"}
    out: dict[str, Any] = {}
    for s, v in goal.items():
        if v["token"] == UNKNOWN:
            continue
        out[s] = f"{pref.get(s, s)}:{v['token']}"
    return out
