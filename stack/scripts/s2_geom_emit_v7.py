"""v7 label emitter — BANDED layers, multi-label tactical goals, nav commands.

    OPERATIVE   0-2 s     (owned by the operative layer, not labelled here)
    TACTICAL    2-6 s     -> tactical goal set + lat/lon actions
    [gap]       6-8 s     -> owned by NO layer; reported, never absorbed
    STRATEGIC   8-30 s    -> strategic goal, `_FOLLOW_ROUTE` suffix, + args

⛔⛔ THIS REPLACES AN ORDINAL SPLIT (manoeuvre[0]=tactical, [1]=strategic) THAT
WAS WRONG AT BOTH ENDS. The ordinal rule ignored the clock entirely, so it
emitted `TURN_RIGHT_FOLLOW_ROUTE` with **`by_time_s: 6.0`** — a strategic token
for a manoeuvre 2 s before the strategic band opens — and it treated 0-2 s, which
belongs to the OPERATIVE layer, as tactical. The PI caught both. A manoeuvre is
now clipped against each band and may legitimately appear in more than one: the
plan begins a turn, the route continues it.

⚠️ **The 6-8 s gap is real and is left visible.** Nothing in the hierarchy owns
it. Manoeuvres falling only there are returned under `bands.unassigned_manoeuvres`
rather than pushed into a neighbour, so the design question stays on the record
instead of being silently answered by this file.

⛔ A ROAD CURVE IS NOT A TURN — see `is_turn` and the calibration above it. A
long bend taken at speed emits `ADAPT_SPEED_FOR_CURVE` and no TURN token.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tanitad.data import alpamayo_fusion as AF
from tanitad.data import alpamayo_records as AR
from tanitad.data import alpamayo_semantics as SEM
from tanitad.data import alpamayo_structured as AST
from tanitad.data import cot_tokens_v7 as COT
from tanitad.data import ego_manoeuvre as EM
from tanitad.data import egomotion_source as ES
from tanitad.models import vocab_v7 as V7

HZ = 10.0
PLAN_S = 6.0             # the tactical plan rollout ends here

# ⛔⛔ THE BANDS, AS THE PI DEFINED THEM (2026-08-27). The emitter previously
# used 0-6 s for TACTICAL and treated anything from 6 s on as STRATEGIC, which
# is wrong on both ends and produced the label the PI caught:
# `TURN_RIGHT_FOLLOW_ROUTE` with `by_time_s: 6.0` — a strategic token for a
# manoeuvre 2 s before the strategic band even opens.
OPERATIVE_S = (0.0, 2.0)     # the operative layer owns the first 2 s
TACTICAL_S = (2.0, 6.0)      # ⭐ 2-6 s, NOT 0-6 s
STRATEGIC_S = (8.0, 30.0)    # ⭐ nothing before 8 s may be strategic
#: ⚠️ 6-8 s belongs to NO layer. That is a real gap in the hierarchy, not an
#: oversight here — a manoeuvre landing there is recorded under
#: `unassigned_manoeuvres` rather than forced into a neighbouring band, so the
#: gap stays visible instead of being silently absorbed.
GAP_S = (PLAN_S, STRATEGIC_S[0])

LOOKAHEAD_S = 30.0       # how far ahead a manoeuvre sequence is read
TURN_RATE_DEG_S = 6.0
MIN_SEG_S = 0.8
MIN_TURN_DEG = 20.0

# ⛔⛔ TURN vs ROAD CURVE — CALIBRATED, NOT CHOSEN (MEASURED 2026-08-27).
# The PI: *"we need to improve our turning extraction from long curves … in the
# given example I don't see a turning manoeuvre."* Correct: `43bbcbf9` is an
# S-bend (+38 deg by t+3 s, back to +24 deg by t+6 s, then -53 deg by t+14 s)
# taken at 12.0 -> 14.3 m/s while ACCELERATING. It was labelled `TURN_L` with
# `radius_m 40.1` — a number that came from PEAK INSTANTANEOUS curvature. The
# ARC radius over the same window is **144 m**. A road curve, not a turn.
#
# Thresholds swept against an INDEPENDENT reference: Alpamayo's own
# `motion_analysis` segment types (349 "turn left/right" vs 1,468 "keep lane"),
# which are text labels and owe nothing to our geometry.
#
#   |dyaw| >= 15 deg  AND  R_arc <= 140 m  AND  v_min <= 8.0 m/s
#     precision 70.6 %  recall 61.3 %  F1 0.656
#
# ⭐ The speed gate is what the PI's instinct bought: it removes **70 of 159
# false positives (44 %)**, lifting precision 58.4 % -> 70.6 %.
#
# ⚠️ BUT NOT VIA DECELERATION, WHICH IS WHAT WE BOTH EXPECTED. In the 2-6 s
# band a turning ego is ACCELERATING (median dv **+2.2 m/s**) because it is
# already exiting the turn; `dv` separates nothing. Turn clips are ALREADY SLOW
# 4 s before the anchor (5.4 vs 14.0 m/s) — the braking happens before anything
# we observe. ⇒ the usable signature is ABSOLUTE SPEED, not speed CHANGE.
TURN_MIN_DYAW_DEG = 15.0
TURN_MAX_ARC_R_M = 140.0
TURN_MAX_VMIN_MS = 8.0

DV_BRAKE_MS = 1.5
DV_CRUISE_MS = 1.0
DV_ACCEL_MS = 1.5
V_STOP_MS = 0.5
SPEED_BAND_TOL_MS = 1.5
CURVE_MAX_R_M = 80.0
SCHEMA = "s2-geom-v7"


def manoeuvre_sequence(poses, key, hz=HZ):
    """Sustained turn segments after ``key``, in TEMPORAL ORDER.

    Returns [(t_start_s, t_end_s, dyaw_deg, radius_m), ...]. This is the
    backbone of the whole v7 design: index 0 is the NEXT manoeuvre and index 1
    the OVERNEXT one.
    """
    p = np.asarray(poses, dtype=np.float64)
    yaw = np.unwrap(p[:, 2])
    v = p[:, 3]
    rel = np.degrees(yaw - yaw[key])
    om = np.gradient(yaw) * hz
    k = max(1, int(round(0.5 * hz)))
    oms = np.convolve(om, np.ones(k) / k, mode="same")
    rate = np.degrees(oms)
    kappa = np.abs(oms) / np.maximum(np.convolve(v, np.ones(k) / k, mode="same"), 1.0)

    act = np.abs(rate[key:]) >= TURN_RATE_DEG_S
    out, i, need = [], 0, int(round(MIN_SEG_S * hz))
    while i < len(act):
        if act[i]:
            j = i
            while j + 1 < len(act) and act[j + 1]:
                j += 1
            if j - i + 1 >= need:
                a, b = key + i, min(key + j, len(rel) - 1)
                d = float(rel[b] - rel[a])
                if abs(d) >= MIN_TURN_DEG:
                    # ⛔ ARC RADIUS, NOT PEAK INSTANTANEOUS CURVATURE.
                    # `1/max(kappa)` is the radius at the single tightest
                    # SAMPLE, so one noisy yaw-rate reading sets the number for
                    # the whole manoeuvre. MEASURED on `43bbcbf9`: it reported
                    # **40.1 m** where the arc over the same span is **144 m**,
                    # and that 40 m is what made a motorway S-bend look like a
                    # junction turn. R = arc_length / dyaw is the radius the
                    # vehicle actually drove.
                    seg = p[a:b + 1]
                    arc = float(np.sum(np.hypot(np.diff(seg[:, 0]),
                                                np.diff(seg[:, 1]))))
                    # ⚠️ float() not just round(): np.radians returns a
                    # np.float64, so the comparison chain in `is_turn` would
                    # return a **np.bool_**, which json.dump refuses. The
                    # corpus runner failed loudly on that rather than writing
                    # a partial file — the right behaviour, but the cast
                    # belongs here at the source.
                    R = (float(arc / np.radians(abs(d))) if abs(d) > 1.0
                         else float("inf"))
                    vmin = float(v[a:b + 1].min())
                    out.append((round(i / hz, 1), round((j + 1) / hz, 1),
                                round(d, 1), round(R, 1), round(vmin, 2)))
            i = j + 1
        else:
            i += 1
    return out


def is_turn(seg) -> bool:
    """Is this manoeuvre a TURN, or merely a road CURVE?

    ⭐ THE PI'S POINT, MADE MEASURABLE. A long sweeping bend taken at speed is
    not a turning manoeuvre, and labelling it `TURN_L` puts a junction token on
    a motorway. Calibrated against Alpamayo's own `turn left/right` segment
    labels (349 positives, 1,468 negatives) — an independent, text-derived
    reference that owes nothing to our geometry:

        |dyaw| >= 15 deg AND R_arc <= 140 m AND v_min <= 8.0 m/s
        -> precision 70.6 %, recall 61.3 %, F1 0.656
        (vs 54.1 % precision on |dyaw| alone)

    ⚠️ Recall is NOT 100 % and should not be read as one: the reference is
    Alpamayo's own labelling over ITS 5.1 s-anchored window, so part of the
    61.3 % is window mismatch rather than missed turns.
    """
    dyaw, R = abs(float(seg[2])), float(seg[3])
    vmin = float(seg[4]) if len(seg) > 4 else 0.0
    return bool(dyaw >= TURN_MIN_DYAW_DEG and R <= TURN_MAX_ARC_R_M
                and vmin <= TURN_MAX_VMIN_MS)


def _arc_to(poses, key, t_s, hz=HZ):
    j = min(len(poses) - 1, key + int(round(t_s * hz)))
    seg = poses[key:j + 1]
    if len(seg) < 2:
        return 0.0
    return round(float(np.sum(np.hypot(np.diff(seg[:, 0]),
                                       np.diff(seg[:, 1])))), 1)


def tactical_goals(poses, key, seq, cot, hz=HZ, lat_action=None,
                   clip_id_for_time=None):
    """The tactical goal SET for the NEXT manoeuvre, plus its anchor args."""
    p = np.asarray(poses, dtype=np.float64)
    # ⛔⛔ THE TACTICAL WINDOW IS 2-6 s, AND THIS FUNCTION WAS STILL USING 0-6 s.
    # MEASURED 2026-08-27 (PI): after the band correction, `split_by_band` used
    # TACTICAL_S but everything ELSE here — SPEED_BAND, STOP_POINT, the stop
    # episodes — still ran from `key` (0 s). 0-2 s belongs to the OPERATIVE
    # layer, so those goals described a stretch of road the tactical layer does
    # not own. The PI spotted it from the report header reading "0-6 s"; the
    # header was accurately describing the code.
    lo = min(len(p) - 1, key + int(round(TACTICAL_S[0] * hz)))
    hi = min(len(p) - 1, key + int(round(TACTICAL_S[1] * hz)))
    c, s = np.cos(-p[key, 2]), np.sin(-p[key, 2])
    dx, dy = p[:, 0] - p[key, 0], p[:, 1] - p[key, 1]
    ex, ey = c * dx - s * dy, s * dx + c * dy
    # the goal ANCHOR stays the END of the plan — the point the tactical layer
    # must reach — but it now states which band it closes.
    anchor = {"goal_x_m": round(float(ex[hi]), 2),
              "goal_y_m": round(float(ey[hi]), 2),
              "t_reach_s": round((hi - key) / hz, 1),
              "band_s": [TACTICAL_S[0], TACTICAL_S[1]]}

    v = p[lo:hi + 1, 3]                 # ⭐ 2-6 s, not 0-6 s
    stops = EM.stop_episodes(v, hz)
    goals: dict[str, dict] = {}

    # ⛔ THE GOAL AND THE ACTION MUST READ THE SAME MANOEUVRE. `seq[0]` is the
    # EARLIEST in-plan manoeuvre; the action takes the LARGEST. When a clip has
    # two, they disagreed — 219 clips carried a TURN_* goal beside
    # `lat=LANE_KEEP`. Both now take the largest in-plan manoeuvre.
    _inplan, _, _ = split_by_band(p, key, seq, hz)
    nxt = max(_inplan, key=lambda x: abs(x[2])) if _inplan else None
    # ⛔ A ROAD CURVE IS NOT A TURN. `is_turn` gates on arc radius AND speed
    # (calibrated, see the constants): a long bend taken at 14 m/s no longer
    # emits `TURN_L`. The longitudinal partner `ADAPT_SPEED_FOR_CURVE` still
    # fires for curves — that is exactly the token the PI added for them.
    if nxt is not None and not is_turn(nxt):
        nxt = None
    if nxt and nxt[0] < TACTICAL_S[1]:
        side = "L" if nxt[2] > 0 else "R"
        held = bool(stops and (stops[0][1] - stops[0][0] + 1) / hz >= 0.5
                    and stops[0][0] / hz < nxt[0])
        tok = f"YIELD_FOR_TURN_{side}" if held else f"TURN_{side}"
        goals[tok] = {"within_m": _arc_to(p, key, nxt[0], hz),
                      "by_time_s": nxt[0], "radius_m": nxt[3],
                      "dyaw_deg": nxt[2]}
        if held:
            goals[f"TURN_{side}"] = {"by_time_s": nxt[0], "radius_m": nxt[3]}
    if stops and not any(k.startswith(("TURN_", "YIELD_FOR_TURN_"))
                         for k in goals):
        i0, i1 = stops[0]
        # ⛔ AN EGO ALREADY AT REST THAT LAUNCHES IS NOT PLANNING A STOP.
        # 91 clips emitted `STOP_POINT` beside `lon=ACCELERATE`: the stop
        # episode began at the anchor itself, so the plan's content is the
        # DEPARTURE, not the arrival. A goal describes what the plan is FOR.
        already_stopped = (i0 == 0 and float(v[-1]) > V_STOP_MS + DV_CRUISE_MS)
        if not already_stopped:
            goals["STOP_POINT"] = {"within_m": _arc_to(p, key, i0 / hz, hz),
                                   "hold_for_s": round((i1 - i0 + 1) / hz, 1)}
    # ⭐⭐ SPEED_BAND IS UNCONDITIONAL (PI, 2026-08-27): *"Speed should be
    # actually always there because it describes the target speed … the speed
    # interval (max, min) there."*
    #
    # It was previously gated on the speed being HELD — so it appeared on some
    # scenes and not others, which is what the PI noticed. That gating confused
    # two different things:
    #   * a TARGET SPEED BAND is a property EVERY plan has. Min and max over
    #     the tactical window always exist and always mean something.
    #   * whether the speed is HELD is the LONGITUDINAL ACTION's business, and
    #     `CRUISE` / `ACCELERATE` / `BRAKE_TO` already say it.
    # Gating the goal on the action's condition deleted the target speed
    # wherever the ego was doing anything other than cruising — exactly the
    # scenes where a target speed matters most.
    #
    # ⚠️ The earlier two-thresholds defect is not reintroduced: the band no
    # longer CLAIMS the speed is held, so there is nothing for it to contradict.
    # It reports the interval and marks whether it is held, and the action
    # remains the single source of that verdict.
    goals[V7.assert_frozen("SPEED_BAND", where="tactical")] = {
        "v_lo_ms": round(max(0.0, float(v.min())), 2),
        "v_hi_ms": round(float(v.max()), 2),
        "band_s": [TACTICAL_S[0], TACTICAL_S[1]],
        # descriptive only — the ACTION owns the held/not-held verdict
        "held": bool(abs(float(v[-1]) - float(v[0])) < DV_CRUISE_MS),
    }
    # ⛔ PERCEPTION TOKENS COME ONLY FROM THE CoT, NEVER FROM GEOMETRY.
    #
    # ⛔⛔ BUT A LATERAL GOAL NEEDS LATERAL EVIDENCE. MEASURED 2026-08-23:
    # **687 of 932 `EVADE_IN_CORRIDOR` emissions (74 %) had no lateral motion
    # at all** — 598 sat beside `lat=LANE_KEEP` and 89 beside a TURN. The PI's
    # definition is explicit that EVADE *is* a lateral manoeuvre ("adapting
    # lateral maneuver due to open door, vru, cyclist"), so a claim with no
    # nudge is not an evasion; it is a sentence about one.
    #
    # The division of labour stays "geometry decides WHAT, CoT decides WHY":
    # geometry must show the nudge, and the CoT supplies the obstacle class
    # that geometry can never see. Same for OVERTAKE, which also requires the
    # ego to actually pass something.
    _LATERAL_EVIDENCE = {"EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE",
                         "LANE_CHANGE_L", "LANE_CHANGE_R"}
    nudging = lat_action in ("NUDGE_L", "NUDGE_R")
    # ⛔ ON A DIRECTION CONFLICT, GEOMETRY WINS. 32 clips emitted
    # `TAKE_EXIT_R` beside a measured LEFT turn — the CoT named a side and the
    # ego went the other way. The lateral axis is one geometry MEASURES, so a
    # sentence cannot outvote it; the exit token is dropped rather than
    # emitted as a contradiction for a consumer to resolve.
    geom_side = next((AF.side_of(k) for k in goals
                      if k.startswith(("TURN_", "YIELD_FOR_TURN_"))), None)
    _SIDED = {"TAKE_EXIT_L": "left", "TAKE_EXIT_R": "right",
              "LANE_CHANGE_L": "left", "LANE_CHANGE_R": "right"}
    # ⛔⛔ A CoT TOKEN HAS NO TIMESTAMP, AND PLACING IT IN A BAND IS AN
    # ASSUMPTION. MEASURED 2026-08-27 (PI: "check if the oncoming label is in
    # the right time slot, I see it in the past"):
    #
    #   * **1,845 clips (39.0 %) carry NO time information whatsoever** — no
    #     motion segments, no "first N seconds" phrase. `d94365be`, the clip the
    #     PI questioned, is one of them.
    #   * Alpamayo's anchor is 5.1 s and ours is 8.0 s, so its text describes a
    #     window that starts 2.9 s BEFORE ours. An event it mentions can easily
    #     have finished before our tactical band opens — which is exactly what
    #     the PI saw in the frames.
    #
    # ⇒ every CoT token now states its TIME BASIS. `segment` means a parsed
    # motion segment overlapping 2-6 s mentions it; `untimed` means the band
    # placement is an assumption and nothing supports it. A consumer that
    # weights these equally is choosing to.
    timed_terms = " ".join(
        s.raw_type + " " + s.motion
        for s in AST.segments_for_band(clip_id_for_time, *TACTICAL_S)
    ).lower() if clip_id_for_time else ""
    for t, a in COT.goals_from_cot(cot).items():
        if t in _LATERAL_EVIDENCE and not nudging:
            continue
        if (geom_side and t in _SIDED and _SIDED[t] != geom_side):
            continue
        stem = t.split("_")[0].lower()
        basis = ("segment" if timed_terms and stem in timed_terms else
                 "untimed")
        goals.setdefault(t, {**a, "provenance": "vlm-cot", "disputed": True,
                             "time_basis": basis})
    if not goals:
        goals["FOLLOW_LANE"] = {}
    # ⚠️ the exclusion matrix is CHECKED, not assumed
    viol = V7.validate_goal_set(goals)
    return goals, anchor, viol


def tactical_actions(poses, key, seq, hz=HZ):
    n = int(round(PLAN_S * hz))
    if key + n >= len(poses):
        return {"lat": "LANE_KEEP", "lon": "CRUISE", "truncated": True,
                "lat_args": {}, "lon_args": {}}
    m = EM.analyse(poses[:key + n + 1], key=key, hz=hz)

    # ⛔⛔ ONE FACT, ONE DETECTOR — AND THE TURN GATE APPLIES ON BOTH SIDES.
    # The GOAL reads the band-clipped, `is_turn`-gated manoeuvre list; the
    # ACTION used to read `EM.analyse`, whose own junction classifier knows
    # nothing about the arc-radius/speed gate. MEASURED 2026-08-27: gating the
    # goal alone left **329 clips (7.0 %)** with `lat=TURN_*` and no TURN goal —
    # the same two-detector defect that produced 219 clips of the MIRROR case a
    # week earlier, reintroduced by fixing only one side of it.
    #
    # ⇒ the gated in-plan list is now the ONLY source of a TURN action.
    # `EM.analyse` still supplies NUDGE, which is a different fact.
    inplan, _post, _gap = split_by_band(poses, key, seq, hz)
    turns = [s for s in inplan if is_turn(s)]
    if turns:
        biggest = max(turns, key=lambda s: abs(s[2]))
        lat = "TURN_L" if biggest[2] > 0 else "TURN_R"
    else:
        # a road CURVE is not a turn: it may still nudge, but it never turns
        lat = ("NUDGE_L" if m.lateral_class == "NUDGE_L" else
               "NUDGE_R" if m.lateral_class == "NUDGE_R" else "LANE_KEEP")

    dv_end, dv_min = m.v_end - m.v_at_key, m.v_min - m.v_at_key
    turning = lat.startswith("TURN_") or (m.turn_radius_m <= CURVE_MAX_R_M
                                          and abs(m.peak_yaw_deg) >= MIN_TURN_DEG)
    # ⛔ A STOP INSIDE THE PLAN IS THE DOMINANT LONGITUDINAL EVENT. 101 clips
    # emitted `STOP_POINT` beside `lon=ACCELERATE`: the ego stopped early in
    # the window and launched, so an end-to-end dv read positive and the label
    # described the plan as an acceleration. The plan's first obligation is to
    # reach the stop.
    stopping = bool(EM.stop_episodes(poses[key:key + n + 1, 3], hz))
    if m.v_at_key <= V_STOP_MS and m.v_end <= V_STOP_MS:
        lon = "HOLD"
    elif stopping and m.v_at_key > V_STOP_MS:
        lon = "BRAKE_TO"
    elif m.stop_type != "NONE" and m.v_end <= V_STOP_MS:
        lon = "BRAKE_TO"
    elif (V_STOP_MS < m.v_at_key <= 2.5 and m.v_end <= 3.5
          and not stopping):
        # ⭐ CREEP had no extraction path either — a slow forward crawl, the
        # classic "edge out to see past the occlusion". Placed BEFORE the
        # turning branch: a creep through a junction is still a creep.
        lon = "CREEP"
    elif turning:
        # ⭐ the longitudinal partner of an arc — universal over junction turns
        # and plain bends (PI: keep the CURVE name, not FOR_TURNING)
        lon = "ADAPT_SPEED_FOR_CURVE"
    elif dv_end >= DV_ACCEL_MS:
        lon = "ACCELERATE"          # ⭐ had no token; was filed as CRUISE
    elif dv_min <= -DV_BRAKE_MS:
        lon = "BRAKE_TO" if dv_end <= -DV_BRAKE_MS / 2 else "FOLLOW"
    elif abs(dv_end) < DV_CRUISE_MS:
        lon = "CRUISE"
    else:
        lon = "FOLLOW" if dv_end < 0 else "ACCELERATE"

    assert lat in V7.TACTICAL_LAT_ACTIONS_V7 and lon in V7.TACTICAL_LON_ACTIONS_V7
    return {"lat": lat, "lon": lon, "truncated": False,
            "lat_args": {"within_m": _arc_to(poses, key, PLAN_S, hz)},
            "lon_args": {"v_target_ms": round(m.v_min, 2),
                         "within_m": _arc_to(poses, key, PLAN_S, hz)}}


def split_by_band(poses, key, seq, hz=HZ):
    """Assign each manoeuvre to TACTICAL or STRATEGIC **by where its mass is**.

    ⛔⛔ WHY NOT BY START TIME — A 94-DEGREE TURN FELL THROUGH THE GAP BETWEEN
    BOTH LAYERS. MEASURED 2026-08-23 on `43bbcbf9`, found by looking at the
    frames of a random sample. Its sequence is:

        (0.0 -> 2.8 s,   +36.5 deg, R=40 m)
        (4.5 -> 14.8 s,  -94.2 deg, R=71 m)      <-- the dominant manoeuvre
        (27.7 -> 32.0 s, -33.2 deg, R=84 m)

    The tactical layer took `seq[0]` only. The strategic layer took the first
    manoeuvre with `t_start >= 6.0`, which SKIPS the -94.2 deg one because it
    STARTS at 4.5 s. So the largest heading change in the clip appeared in
    NEITHER layer, and the strategic label described a turn 27.7 s away while a
    94-degree turn ran from 4.5 to 14.8 s.

    ⇒ **A manoeuvre belongs to the band that holds most of its heading
    change.** The PI's rule — *"the turning is happening within the 6 seconds
    horizon, so it is no strategic action any more, it's tactical"* — is a
    statement about WHERE THE TURN HAPPENS, not about where it begins, and a
    start-time test only agrees with it when no manoeuvre straddles 6 s.

    On `43bbcbf9` the split is unambiguous: of the -94.2 deg, only ~-10 deg
    falls inside the plan and ~-84 deg after it. Tactical keeps `TURN_L`
    (+36.5 deg, genuinely the in-plan manoeuvre); strategic now correctly
    reports the -94.2 deg turn instead of the one 27.7 s out.

    Returns (in_plan, post_plan), each a list of the same 4-tuples.
    """
    p = np.asarray(poses, dtype=np.float64)
    yaw = np.unwrap(p[:, 2])
    rel = np.degrees(yaw - yaw[key])
    edge = key + int(round(PLAN_S * hz))

    def dyaw_between(t0_s, t1_s):
        i = min(len(rel) - 1, key + int(round(t0_s * hz)))
        j = min(len(rel) - 1, key + int(round(t1_s * hz)))
        return float(rel[j] - rel[i]) if j > i else 0.0

    # ⭐ A STRADDLING MANOEUVRE BELONGS TO **BOTH** LAYERS — and that is not a
    # compromise, it is the physics. MEASURED 2026-08-23: an exclusive split by
    # mass fixed 43bbcbf9 and BROKE 70 clips (1.5 %) the other way, giving them
    # `lat=TURN_R` with no TURN goal at all (`ec075947`: a -78 deg turn from
    # 4.1 to 8.2 s, so ~-36 deg inside the plan and ~-42 deg after it — real on
    # both sides, and an exclusive rule has to be wrong about one of them).
    #
    # ⚠️ MY OWN COHERENCE CHECK COULD NOT SEE THAT REGRESSION: it tested
    # "TURN goal but lat=LANE_KEEP" and never the converse. A one-sided
    # consistency check certifies half a contract. Both directions are tested
    # now.
    #
    # The PI's rule is satisfied exactly: the part of the turn inside 6 s is
    # TACTICAL ("the turning is happening within the 6 seconds horizon, so it
    # is tactical"), and what remains is the STRATEGIC continuation ("the
    # strategic action is follow road after turning").
    # ⛔⛔ THE BANDS ARE 2-6 s AND 8-30 s, WITH A REAL GAP BETWEEN THEM.
    # This function previously split at 6 s and called everything after it
    # strategic — which emitted `TURN_RIGHT_FOLLOW_ROUTE  by_time_s: 6.0`, a
    # strategic token 2 s before the strategic band opens. The PI caught it.
    #
    # A manoeuvre is now clipped against each band separately and may appear
    # in both (the plan begins the turn, the route continues it), in neither,
    # or in the 6-8 s GAP — which is returned as its own list so the hierarchy's
    # blind spot stays visible rather than being absorbed by a neighbour.
    tac_lo, tac_hi = TACTICAL_S
    str_lo, str_hi = STRATEGIC_S

    def clip(s, lo, hi):
        a, b = max(s[0], lo), min(s[1], hi)
        if b <= a:
            return None, 0.0
        d = dyaw_between(a, b)
        return (round(a, 1), round(b, 1), round(d, 1), s[3],
                s[4] if len(s) > 4 else 0.0), abs(d)

    in_plan, post, gap = [], [], []
    for s in seq:
        c_tac, d_tac = clip(s, tac_lo, tac_hi)
        c_str, d_str = clip(s, str_lo, str_hi)
        c_gap, d_gap = clip(s, *GAP_S)
        if c_tac and d_tac >= MIN_TURN_DEG:
            in_plan.append(c_tac)
        if c_str and d_str >= MIN_TURN_DEG:
            post.append(c_str)
        # only report a gap manoeuvre that belongs to NEITHER band, otherwise
        # every straddling turn would also be reported as a gap case
        if (c_gap and d_gap >= MIN_TURN_DEG
                and d_tac < MIN_TURN_DEG and d_str < MIN_TURN_DEG):
            gap.append(c_gap)
    return in_plan, post, gap


def strategic(poses, key, seq, hz=HZ):
    """The OVERNEXT manoeuvre, suffixed `_FOLLOW_ROUTE`, with args."""
    inplan, rest, _gap = split_by_band(poses, key, seq, hz)
    # ⛔ THE SAME TURN/CURVE GATE APPLIES HERE. Without it the tactical layer
    # correctly refused to call `43bbcbf9`'s bend a turn while the strategic
    # layer still emitted `TURN_RIGHT_FOLLOW_ROUTE` for the same arc — one
    # instrument, two verdicts, which is the defect class this file has hit
    # three times now.
    rest = [m for m in rest if is_turn(m)]
    over = rest[0] if rest else None
    if over is None:
        # ⛔ `REDUCE_TO_FOLLOW_ROUTE` WAS REMOVED ON PI INSTRUCTION 2026-08-23
        # ("we can eliminate REDUCE_TO_FOLLOW_ROUTE, we dont need it"). It had
        # no definition and 0 emissions in the first v7 run, and with the
        # ordinal design PREPARE_TURN / PREPARE_STOP already carry "slow for
        # the overnext manoeuvre". A sustained speed drop with no manoeuvre is
        # a SPEED fact, and speed is the tactical layer's `SPEED_BAND` — it
        # does not need a strategic token of its own.
        #
        # ⚠️ This branch is what `test_vocab_v7_frozen` caught: the token was
        # deleted from the vocabulary while the emitter still wrote it. The
        # runtime `assert_frozen` below now makes that class of drift fail at
        # emit time rather than producing a corpus with an unmappable class.
        # ⭐⭐ A STOP IS A STRATEGIC EVENT TOO, AND NOTHING COULD EMIT ONE.
        # MEASURED 2026-08-27: **13 of 52 frozen tokens (25 %) were never
        # produced on the whole 4,719-clip corpus**, and the strategic layer was
        # the worst hit — only 3 of 8 goals and 2 of 7 actions were reachable,
        # because `strategic()` handled TURNS and nothing else. A head trained
        # on that carries five permanently empty classes.
        #
        # A stop inside the strategic band is pure geometry and needs no new
        # source: the ego comes to rest between 8 s and 30 s, having been
        # moving at the anchor.
        p = np.asarray(poses, dtype=np.float64)
        lo = key + int(round(STRATEGIC_S[0] * hz))
        hi = min(len(p) - 1, key + int(round(STRATEGIC_S[1] * hz)))
        if hi > lo and float(p[key, 3]) > V_STOP_MS:
            band = p[lo:hi + 1, 3]
            stops = EM.stop_episodes(band, hz)
            if stops:
                t_stop = round(STRATEGIC_S[0] + stops[0][0] / hz, 1)
                args = {"within_m": _arc_to(poses, key, t_stop, hz),
                        "by_time_s": t_stop,
                        "hold_for_s": round((stops[0][1] - stops[0][0] + 1) / hz, 1)}
                return ({"token": V7.assert_frozen("STOP_AT_FOLLOW_ROUTE",
                                                   where="strategic"),
                         "args": args, "provenance": "geometry",
                         "reason": f"ego comes to rest at t+{t_stop}s, inside "
                                   f"the {STRATEGIC_S[0]}-{STRATEGIC_S[1]}s band"},
                        {"token": V7.assert_frozen("PREPARE_STOP_FOLLOW_ROUTE",
                                                   where="strategic"),
                         "args": args, "provenance": "geometry",
                         "reason": f"{args['within_m']} m ahead, still moving "
                                   f"at {float(p[key, 3]):.1f} m/s"})
            # ⭐ RESUME_CRUISE: the ego is slow now and recovers road-class
            # speed later in the band — the constraint ends, which is a
            # route-level fact and the token's own definition.
            v0 = float(p[key, 3])
            v_band_max = float(band.max()) if len(band) else v0
            if v0 < 8.0 and v_band_max >= v0 + 4.0:
                return ({"token": V7.assert_frozen("FOLLOW_ROUTE",
                                                   where="strategic"),
                         "args": {}, "provenance": "geometry",
                         "reason": "no overnext manoeuvre; the road continues"},
                        {"token": V7.assert_frozen("RESUME_CRUISE_FOLLOW_ROUTE",
                                                   where="strategic"),
                         "args": {"v_target_ms": round(v_band_max, 2)},
                         "provenance": "geometry",
                         "reason": f"{v0:.1f} -> {v_band_max:.1f} m/s within the "
                                   f"strategic band: the constraint ends"})
        act = {"token": V7.assert_frozen("HOLD_MAIN_ROAD", where="strategic"),
               "args": {}, "provenance": "geometry",
               "reason": "no overnext manoeuvre in the lookahead"}
        return ({"token": V7.assert_frozen("FOLLOW_ROUTE", where="strategic"),
                 "args": {}, "provenance": "geometry",
                 "reason": "no overnext manoeuvre in the lookahead"}, act)
    # ⚠️ `over` now comes from the 8-30 s clip, so `by_time_s` can no longer
    # be 6.0 — the defect the PI reported.
    side = "LEFT" if over[2] > 0 else "RIGHT"
    args = {"within_m": _arc_to(poses, key, over[0], hz), "by_time_s": over[0]}
    g = {"token": f"TURN_{side}_FOLLOW_ROUTE", "args": args,
         "provenance": "geometry",
         "reason": f"overnext manoeuvre {over[2]:+.0f} deg at t+{over[0]}s "
                   f"(R={over[3]} m); {len(inplan)} in-plan before it"}
    a = {"token": f"PREPARE_TURN_{side[0]}_FOLLOW_ROUTE", "args": args,
         "provenance": "geometry",
         "reason": f"{args['within_m']} m / {over[0]}s ahead, not yet begun"}
    return g, a


def nav_command(poses, key, seq, hz=HZ):
    """⛔ MODEL INPUT derived from the EGO FUTURE — training only.

    See `vocab_v7.NAV_PROVENANCE`. On PhysicalAI the only route supplier is the
    recorded future, so this is ORACLE information: admissible as a training
    input, never as evidence in a vision-only eval arm.
    """
    nxt = seq[0] if seq else None
    if nxt is None:
        return {"token": "NAV_FOLLOW_ROAD", "args": {},
                "provenance": "ego-future", "oracle": True}
    if not is_turn(nxt):
        return {"token": V7.assert_frozen("NAV_FOLLOW_ROAD", where="nav"),
                "args": {}, "provenance": "ego-future",
                "reason": "curve, not a turn — nav does not command a turn"}
    side = "L" if nxt[2] > 0 else "R"
    return {"token": f"NAV_TURN_{side}",
            "args": {"distance_m": _arc_to(poses, key, nxt[0], hz),
                     "time_s": nxt[0]},
            "provenance": "ego-future", "oracle": True}


def _alpamayo_layer(clip_id: str, poses, key, goals: dict) -> dict:
    """Corroborate, ground and enrich — never overwrite geometry.

    ⭐ ADDED 2026-08-23 once the real augmentation was found (RETRACTION_LOG
    C142). Four things the geometry-only emitter could not do:

    1. **Ground the CoT tokens.** 93.3 % of clips carry 2D boxes, so a token
       claimed by a generative sentence can be CHECKED against image-space
       perception. `disputed` stops being a blanket property.
    2. **Cross-check the longitudinal axis** against a calibrated scale
       (Strong Decel −5.88 … Strong Accel +6.00 m/s, Cohen's d 1.59).
    3. **Corroborate the lateral axis** (69.9 % vs 41.6 % chance, lift x1.68).
    4. **Add typed tokens with correct band timing** from the structured
       motion segments — EVADE / TAKE_EXIT / CREEP / YIELD, which the three
       meta_action axes cannot express.

    ⛔ Geometry remains authoritative on WHAT the ego did. A disagreement is
    recorded as a QA item; it never silently replaces a measured quantity with
    a model's phrase.
    """
    p = np.asarray(poses, dtype=np.float64)
    n6 = int(round(PLAN_S * HZ))

    # ⛔ SCORE ALPAMAYO'S CLAIM ON ALPAMAYO'S OWN WINDOW. The calibration table
    # (Strong Decel −5.88 … Strong Accel +6.00) was measured over ITS 6 s from
    # ITS 5.1 s anchor. Scoring it over our 8.0 s window compares two different
    # stretches of road and reads as disagreement.
    # MEASURED 2026-08-23: our window 54.9 % agree, its own window **62.1 %** —
    # 7.2 points of pure measurement error, produced by the very scope mistake
    # this module's docstring warns about. Each source is evaluated on its own
    # time base; the comparison states which.
    ka = int(round(AR.ALPAMAYO_T0_S * HZ))
    if ka + n6 < len(p):
        v0, dv = float(p[ka, 3]), float(p[ka + n6, 3] - p[ka, 3])
        win = f"alpamayo {AR.ALPAMAYO_T0_S:.1f}-{AR.ALPAMAYO_T0_S + PLAN_S:.1f}s"
    else:                      # short recording — fall back, and SAY so
        hi = min(len(p) - 1, key + n6)
        v0, dv = float(p[key, 3]), float(p[hi, 3] - p[key, 3])
        win = "our anchor (alpamayo window truncated)"

    lon = AF.fuse_longitudinal(clip_id, v0, dv)
    # ⚠️ Pass a SIDE, not a token. `"_L" in "FOLLOW_LANE"` is True, and testing
    # the token directly reported 24.9 % lateral agreement where the truth is
    # 69.9 % — see `alpamayo_fusion.side_of`.
    geom_turn = next((k for k in goals if k.startswith(("TURN_", "YIELD_FOR_TURN_"))),
                     None)
    lat_ok, lat_side = AF.fuse_lateral(
        clip_id, AF.side_of(geom_turn) if geom_turn else "straight")

    # Typed tokens Alpamayo can see and geometry cannot, banded correctly.
    band = AST.band_tokens(clip_id, 2.0, PLAN_S)
    added: list[str] = []
    for tok in band["lateral"] + band["longitudinal"]:
        if tok in V7.TACTICAL_GOAL_TOKENS_V7 and tok not in goals:
            # ⚠️ Only tokens geometry structurally cannot produce. A lateral
            # class geometry DOES measure must not be added from a 55.2 %
            # source — that would let the weaker instrument outvote the
            # stronger one on its own axis.
            # ⚠️ the SAME lateral-evidence rule as the CoT path — otherwise
            # this layer becomes a back door that re-adds the 40 EVADE tokens
            # the CoT path correctly refused.
            if tok == "EVADE_IN_CORRIDOR" and goals.get("_lat") not in (
                    "NUDGE_L", "NUDGE_R"):
                continue
            if tok in ("EVADE_IN_CORRIDOR", "TAKE_EXIT_L", "TAKE_EXIT_R",
                       "MERGE", "YIELD"):
                goals[V7.assert_frozen(tok, where="alpamayo-structured")] = {
                    "provenance": "alpamayo-structured", "disputed": True,
                    "reason": f"segment type {band['raw_types']}"}
                added.append(tok)

    comp = AST.critical_component(clip_id)
    return {
        "longitudinal": {"phrase": lon.phrase, "dv_expected_ms": lon.dv_expected,
                         "dv_measured_ms": round(lon.dv_measured, 2),
                         "agree": lon.agree, "strength": lon.strength,
                         "scored_on": win, "note": lon.note},
        "lateral": {"alpamayo_side": lat_side, "agree": lat_ok},
        "segments": {"spans": band["spans"], "types": band["raw_types"],
                     "tokens_added": added},
        "critical_component": None if comp is None else {
            "type": comp.raw_type, "why": comp.why[:220],
            # ⭐ an EXPLICIT clean negative, not merely an absent match
            "is_explicit_none": comp.is_none, "grounding_kind": comp.kind},
        "boxes": (AR.get(clip_id).box_labels() if AR.get(clip_id) else []),
        "anchor_offset_s": round(AR.ALPAMAYO_T0_S - ES.RAW_T0_S, 1),
    }


def emit_one(clip_id: str, *, sem_row=None) -> dict:
    tr = ES.load(clip_id, hz=HZ, max_s=ES.RAW_T0_S + LOOKAHEAD_S + 5.0)
    key, poses = tr.key_index, tr.poses
    # ⭐ The augmentation is now read from the DATASET, not from a four-field
    # export. `cot_text` concatenates cot + chain_of_causation + the two
    # analyses: MEASURED, that lifts clips carrying >=1 token from 1,803 to
    # 3,188 (+77 %) over the 4,729-clip corpus.
    cot = AF.cot_text(clip_id) or (sem_row.get("cot") if sem_row else None)
    seq = manoeuvre_sequence(poses, key, HZ)
    # ⚠️ ACTIONS FIRST: the goals consult them, so one fact has one detector.
    _at = tactical_actions(poses, key, seq, HZ)
    goals, anchor, viol = tactical_goals(poses, key, seq, cot, HZ,
                                         lat_action=_at["lat"],
                                         clip_id_for_time=clip_id)
    goals["_lat"] = _at["lat"]          # so the structured layer sees it
    alpa = _alpamayo_layer(clip_id, poses, key, goals)
    goals.pop("_lat", None)
    # ⭐ THE FUSION GATE: a CoT token backed by a BOX is no longer `disputed`.
    goals = AF.ground_tokens(clip_id, goals)
    viol = V7.validate_goal_set(goals)          # re-check after the additions
    g_str, a_str = strategic(poses, key, seq, HZ)
    return {
        "alpamayo": alpa,
        "schema_version": SCHEMA, "clip_id": clip_id, "t0_s": ES.RAW_T0_S,
        "vocab": "v7",
        "manoeuvre_sequence": [{"t_start_s": m[0], "t_end_s": m[1],
                                "dyaw_deg": m[2], "radius_m": m[3],
                                "v_min_ms": m[4] if len(m) > 4 else None,
                                "is_turn": is_turn(m)}
                               for m in seq],
        "bands": {"operative_s": list(OPERATIVE_S),
                  "tactical_s": list(TACTICAL_S),
                  "strategic_s": list(STRATEGIC_S),
                  # manoeuvres in the 6-8 s hierarchy gap, owned by no layer
                  "unassigned_manoeuvres": [
                      {"t_start_s": g[0], "t_end_s": g[1], "dyaw_deg": g[2]}
                      for g in split_by_band(poses, key, seq, HZ)[2]]},
        "g_str": g_str, "a_str": a_str,
        "g_tac": {"goals": goals, "anchor": anchor, "violations": viol},
        "a_tac": {**_at,
                  # ⭐ PI: link goals to their admissible actions. An action
                  # serving NONE of its goals is the TURN_LEFT+HOLD_CORRIDOR
                  # shape — recorded per clip so it cannot hide.
                  "serves_goals": V7.action_serves_goals(
                      _at["lat"], _at["lon"], goals)},
        "nav_command": nav_command(poses, key, seq, HZ),
        # ⭐ the PI asked for the RAW source text in the visualisation, and the
        # conflict check that goes with it — two independent generative draws
        # disagree on direction for 8.8 % of the clips that state one.
        "cot_source": {
            "cot": (AR.get(clip_id).cot if AR.get(clip_id) else None),
            "chain_of_causation": (AR.get(clip_id).chain_of_causation
                                   if AR.get(clip_id) else None),
            "components_analysis": (AR.get(clip_id).components_analysis
                                    if AR.get(clip_id) else None),
            "motion_analysis": (AR.get(clip_id).motion_analysis
                                if AR.get(clip_id) else None),
            "meta_action": (AR.get(clip_id).meta_action
                            if AR.get(clip_id) else None),
            "conflict": AF.cot_conflict(clip_id),
        },
        "semantics": (SEM.extract(cot).as_dict() if cot else None),
        "cot_tokens": COT.extract(cot).as_dict() if cot else None,
        "horizon": {"available_s": round(tr.horizon_available_s, 1),
                    "recording_span_s": round(tr.span_s, 1)},
        "_provenance": {
            "layers": "ORDINAL: manoeuvre[0]=tactical, manoeuvre[1]=strategic",
            "kinematics": "ego geometry only",
            "perception_tokens": "vlm-cot, disputed, never geometry-derived",
            "nav_command": "ORACLE (ego-future) — training input only",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-index", required=True)
    ap.add_argument("--alpamayo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clips = json.loads(Path(a.clip_index).read_text(encoding="utf-8"))["clips"]
    alpa = {}
    if a.alpamayo:
        for line in Path(a.alpamayo).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                alpa[r["clip_id"]] = r
    rows, failed = [], []
    for i, c in enumerate(sorted(clips)):
        try:
            rows.append(emit_one(c, sem_row=alpa.get(c)))
        except Exception as ex:                              # noqa: BLE001
            failed.append({"clip_id": c, "error": repr(ex)})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(clips)}", flush=True)
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    gs = collections.Counter(r["g_str"]["token"] for r in rows)
    ast = collections.Counter(r["a_str"]["token"] for r in rows)
    gt = collections.Counter(t for r in rows for t in r["g_tac"]["goals"])
    la = collections.Counter(r["a_tac"]["lat"] for r in rows)
    lo = collections.Counter(r["a_tac"]["lon"] for r in rows)
    nav = collections.Counter(r["nav_command"]["token"] for r in rows)
    nviol = sum(1 for r in rows if r["g_tac"]["violations"])
    print(f"[v7] {len(rows)} labels -> {outp}  failed {len(failed)}")
    print(f"[v7] goal-set violations: {nviol}")
    print(f"[v7] g_str {dict(gs)}")
    print(f"[v7] a_str {dict(ast)}")
    print(f"[v7] g_tac (multi-label) {dict(gt)}")
    print(f"[v7] a_tac LAT {dict(la)}")
    print(f"[v7] a_tac LON {dict(lo)}")
    print(f"[v7] nav {dict(nav)}")
    if failed:
        (outp.parent / "emit_v7_failures.json").write_text(
            json.dumps(failed, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
