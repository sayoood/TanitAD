"""Robust ego-kinematic manoeuvre detection for the label pipeline.

⭐ WHY THIS REPLACES THE YAW-THRESHOLD RULE (PI direction, 2026-08-23):
*"Depending on the time horizon you can analyze the speed, acceleration and
positions... combine the curvature and yaw with the speed and acceleration
profile; yielding situations are typically accompanied with decelerations or
sometimes stops at traffic lights before turning."*

⛔ THE DEFECT THIS FIXES, MEASURED. A net-yaw threshold cannot tell a JUNCTION
TURN from a ROAD BEND — the SAME failure the `LANE_TARGET` derivation was
retired for ("lateral displacement events cannot distinguish a lane change from
road curvature"). Radius and the speed profile separate them; net yaw does not.

⚠️ AND THE RADIUS MUST BE INSTANTANEOUS. On clip `5b4eef8f`, arc/delta-yaw over
the horizon reads **83.7 m** (a gentle bend) while kappa = omega/v at the apex
reads **12.4 m** (a tight junction turn, +69 deg accumulated in 4.0 s at a peak
yaw rate of 26 deg/s). The first number is an artefact of dividing by an arc
that includes the straight road AFTER the turn. I acted on it briefly and began
weakening a correct guard — see RETRACTION_LOG C135.

⛔ AND THE PRIMARY MANOEUVRE IS THE FIRST SUSTAINED SEGMENT, NOT THE LARGEST.
A clip routinely contains a SEQUENCE. `90006660` turns RIGHT -40 deg then LEFT
+93 deg; peak-absolute reported TURN_LEFT and the existing label (TURN_RIGHT)
was correct. `00d05901` likewise turns right then left, and Alpamayo's own CoT
says "turn right at the intersection". Reporting the first committed manoeuvre
is also what a planner sitting at the anchor actually needs; `n_turn_segments`
exposes the remainder instead of hiding it.

⚠️ SCOPE: this module reads EGO POSES ONLY. It is deliberately blind to lanes,
signs, lights and other agents, so it can state WHAT the vehicle did but never
WHY. The reason belongs to the perception/VLM layer (`alpamayo_semantics.py`).
Keeping the two apart is what stops a kinematic signal from being read as a
semantic one — and it is what lets `stop_type` stay honest: the kinematics of a
red-light stop and a jam-front stop are genuinely identical for the first few
seconds, so only the RECOVERY distinguishes them here.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

HZ_DEFAULT = 10.0

# ---- lateral -------------------------------------------------------------
TURN_DEG = 25.0          # yaw a manoeuvre must reach to be more than a nudge
# ⭐ RADIUS IS MEASURED AS INSTANTANEOUS CURVATURE, kappa = omega / v, minimised
# over the manoeuvre — NOT as arc/delta-yaw over a window.
# ⛔ THE ESTIMATOR ERROR THIS REPLACES, and it fooled me for a whole pass:
# arc/delta-yaw taken over the available horizon divides the turn's yaw by an
# arc that INCLUDES the straight road after the turn, inflating the radius
# without bound. On `5b4eef8f` that read **83.7 m** (a gentle bend) when the
# true apex radius is **12.4 m** (a tight junction turn) — the turn takes 4 s
# of a 12 s horizon and the remaining 8 s of straight driving diluted it. I
# nearly weakened a CORRECT guard on the strength of that number.
# THRESHOLDS CALIBRATED ON THE REAL 39-CLIP DISTRIBUTION, not synthetically.
# On the corrected estimator the sample sorts as:
#     R <= 28.7 m : 26 clips, EVERY ONE with |peak yaw| >= 33 deg (turns)
#     R  = 48.1 m : `00d05901` only
#     R >= 77.9 m : every clip has |peak yaw| <= 13 deg (straight / nudge)
#
# ⛔⛔ I BRIEFLY SET R_TURN_M = 60 TO PUT `00d05901` ON THE TURN SIDE, BECAUSE
# ALPAMAYO'S CoT SAYS "turn right at the intersection since the traffic light is
# green". VISUAL INSPECTION OF ITS FRAMES REFUTES THAT CoT OUTRIGHT: the clip is
# a RURAL MOUNTAIN ROAD through forest — there is no intersection and no traffic
# light in any of the 9 sampled frames. The CoT is FABRICATED, and I had ALREADY
# MEASURED that Alpamayo's lateral axis is at chance (p=0.335) before using it as
# a calibration witness. R = 48.1 m at -29.4 deg on a country road is a ROAD
# BEND, which is what the 40 m gate said in the first place.
# ⇒ **NEVER CALIBRATE A THRESHOLD ON A WITNESS ALREADY MEASURED UNRELIABLE.**
# The gate returns to 40 m, inside the 28.7 -> 48.1 m gap that the FRAMES
# support. See RETRACTION_LOG C138.
R_JUNCTION_M = 20.0      # at or below: a tight junction turn (HIGH confidence)
R_TURN_M = 40.0          # at or below: still a turn (wide/sweeping junction)
R_BEND_M = 60.0          # at or above: road geometry, not a decision
V_FLOOR_MS = 1.0         # kappa = omega/v explodes at rest; floor the divisor
TURN_RATE_DEG_S = 6.0    # yaw rate that marks an ACTIVE turn segment
MIN_TURN_S = 0.8         # a segment shorter than this is jitter, not a turn
MERGE_GAP_S = 2.0        # same-sign segments closer than this are ONE turn
SMOOTH_S = 0.5           # yaw-rate smoothing window
NUDGE_LAT_M = 1.0        # lateral offset that counts as a deliberate nudge

# ---- longitudinal --------------------------------------------------------
V_STOP_MS = 0.5          # at or below this the ego is stopped
V_CRAWL_MS = 2.0         # queue-crawl ceiling
V_CRUISE_MS = 5.0        # a clean pull-away reaches this
DECEL_SIGNIFICANT = 1.5  # m/s drop that counts as a distinct deceleration
QUEUE_VMAX_MS = 6.0      # a jam never gets far above crawl
TURN_SLOWDOWN_RATIO = 0.75   # v_apex / v_approach below this = slowed FOR the turn


@dataclass
class Manoeuvre:
    """What the ego did, from poses alone. Never why."""
    # -- lateral
    lateral_class: str            # JUNCTION_TURN_L/R | ROAD_BEND_L/R | NUDGE_L/R | STRAIGHT
    peak_yaw_deg: float
    yaw_onset_s: float | None
    turn_arc_m: float
    turn_radius_m: float          # inf when straight
    kappa_max: float
    # -- longitudinal
    longitudinal_class: str       # LAUNCH | DECEL_TO_STOP | STOP_AND_GO | SLOWING | CRUISE
    stop_type: str                # NONE | CONTROLLED | QUEUE | YIELD | ALREADY_STOPPED
    n_stop_episodes: int
    longest_stop_s: float
    total_stop_s: float
    v_at_key: float
    v_min: float
    v_max: float
    v_end: float
    n_decel_events: int
    n_turn_segments: int
    slowed_for_turn: bool | None  # None when there is no turn to slow for
    # -- bookkeeping
    horizon_s: float
    confidence: str               # HIGH | MEDIUM | LOW — how separable the call was

    def as_dict(self) -> dict:
        return asdict(self)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def stop_episodes(v, hz=HZ_DEFAULT):
    """Contiguous runs at or below V_STOP_MS, as (start_i, end_i) inclusive."""
    out, i, n = [], 0, len(v)
    while i < n:
        if v[i] <= V_STOP_MS:
            j = i
            while j + 1 < n and v[j + 1] <= V_STOP_MS:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def decel_events(v):
    """Count distinct drops of at least DECEL_SIGNIFICANT m/s.

    Tracks the running maximum since the last counted event, so one long
    deceleration counts ONCE while a stop-and-go pattern counts once per cycle.
    That difference is the entire basis of QUEUE detection: a jam is not slower
    than a red light, it is slow REPEATEDLY.
    """
    if len(v) == 0:
        return 0
    # ⚠️ A DESCENT MUST COUNT ONCE, AND RE-ARM ONLY AFTER A RECOVERY.
    # Naively re-arming at the moment of counting (peak = x) makes a single
    # smooth 10 -> 0 m/s deceleration count SIX times, because each further
    # 1.5 m/s of the same descent re-triggers. That would have made every
    # ordinary stop look like stop-and-go traffic and destroyed the QUEUE
    # discriminator this function exists to provide. Caught by
    # `test_decel_events_counts_descents_not_samples`.
    peak = trough = float(v[0])
    count, armed = 0, True
    for raw in v:
        x = float(raw)
        if x >= trough + DECEL_SIGNIFICANT:   # speed recovered => a new cycle
            armed, peak = True, x
        if x > peak:
            peak = x
        if armed and peak - x >= DECEL_SIGNIFICANT:
            count += 1
            armed = False
            trough = x
        if x < trough:
            trough = x
    return count


def analyse(poses, *, hz: float = HZ_DEFAULT, key: int = 0) -> Manoeuvre:
    """Classify the manoeuvre in ``poses[key:]``.

    ``poses`` is [T, 4] = (x, y, yaw, v). Everything is measured forward from
    ``key`` over whatever horizon the array provides — the CALLER slices a
    horizon matching the label family being checked. (C134: scoring a strategic
    label on a tactical window manufactures failures that are not real.)
    """
    p = np.asarray(poses, dtype=np.float64)[key:]
    if p.shape[0] < 3:
        raise ValueError(f"need at least 3 poses, got {p.shape[0]}")
    x, y, yaw, v = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    horizon_s = (len(v) - 1) / hz

    # -- lateral geometry ---------------------------------------------------
    yaw_deg = np.degrees(np.unwrap(yaw) - yaw[0])
    yaw_rel = yaw_deg
    i_peak = int(np.argmax(np.abs(yaw_rel)))
    peak = float(yaw_rel[i_peak])
    big = np.where(np.abs(yaw_rel) >= TURN_DEG)[0]
    onset = float(big[0] / hz) if big.size else None

    # instantaneous curvature kappa = omega / v, smoothed
    yaw_u = np.unwrap(yaw)
    omega = np.gradient(yaw_u) * hz                     # rad/s
    k_win = max(1, int(round(SMOOTH_S * hz)))
    ker = np.ones(k_win) / k_win
    omega_s = np.convolve(omega, ker, mode="same")
    v_s = np.convolve(v, ker, mode="same")
    kappa_t = np.abs(omega_s) / np.maximum(v_s, V_FLOOR_MS)

    # ⛔ THE PRIMARY MANOEUVRE IS THE **FIRST SUSTAINED TURN SEGMENT**, NOT THE
    # LARGEST YAW EXCURSION. A clip routinely contains a SEQUENCE, and taking
    # the peak silently reports the wrong one. MEASURED 2026-08-23:
    #   90006660 turns RIGHT -40 deg (t+0.0-2.2 s) then LEFT +93 deg
    #            (t+3.7-12.1 s) -> peak-absolute said TURN_LEFT and the label
    #            (TURN_RIGHT) was CORRECT; my classifier was the wrong one.
    #   00d05901 turns RIGHT -24 deg then LEFT +47 deg -> peak said left,
    #            while Alpamayo states "turn right at the intersection".
    # Reporting the first committed manoeuvre also matches what a planner at
    # the anchor must decide. `n_turn_segments` exposes the rest rather than
    # hiding it.
    yaw_rate_deg = np.degrees(omega_s)
    active = np.abs(yaw_rate_deg) >= TURN_RATE_DEG_S
    raw_segs, i = [], 0
    while i < len(active):
        if active[i]:
            j = i
            while j + 1 < len(active) and active[j + 1]:
                j += 1
            if (j - i + 1) >= int(round(MIN_TURN_S * hz)):
                d = float(yaw_deg[min(j, len(yaw_deg) - 1)] - yaw_deg[i])
                raw_segs.append([i, j, d])
            i = j + 1
        else:
            i += 1

    # ⚠️ MERGE SAME-SIGN SEGMENTS SEPARATED BY A SHORT GAP. A real turn dips
    # below the yaw-rate gate mid-manoeuvre (the driver eases, then continues),
    # which fragments it. MEASURED: `00d05901`'s right turn splits into a
    # -24 deg piece and its continuation; unmerged, the -24 deg piece falls 1
    # deg short of TURN_DEG and the classifier silently reported the LATER
    # left-hand segment instead — inverting the turn direction on a clip whose
    # own CoT says "turn right at the intersection".
    segments = []
    gap = int(round(MERGE_GAP_S * hz))
    for seg in raw_segs:
        if segments and seg[2] * segments[-1][2] > 0 and \
                seg[0] - segments[-1][1] <= gap:
            segments[-1][1] = seg[1]
            segments[-1][2] = float(yaw_deg[min(seg[1], len(yaw_deg) - 1)]
                                    - yaw_deg[segments[-1][0]])
        else:
            segments.append(list(seg))

    # ⚠️ EXTEND EACH SEGMENT TO ITS LOCAL YAW EXTREMUM. The yaw-RATE gate closes
    # while the HEADING is still changing — the driver eases off before the car
    # has finished rotating — so a segment's endpoint understates the turn.
    # MEASURED: `00d05901`'s first (right) turn gates out at -24.3 deg while the
    # heading actually reaches -38.1 deg. Because -24.3 falls 0.7 deg under
    # TURN_DEG the turn was DISCARDED and the later left-hand segment reported
    # instead, INVERTING the direction on a clip whose own CoT says "turn right
    # at the intersection". Extension recovers -29.4 deg and the sign is right.
    for k in range(len(segments)):
        a = segments[k][0]
        nxt = segments[k + 1][0] if k + 1 < len(segments) else len(yaw_deg)
        window = yaw_deg[a:nxt]
        if len(window):
            e = int(np.argmax(np.abs(window - yaw_deg[a])))
            segments[k] = [a, min(a + e, len(yaw_deg) - 1),
                           float(window[e] - yaw_deg[a])]

    segments = [tuple(s) for s in segments]
    n_segments = len(segments)

    seg = next((s for s in segments if abs(s[2]) >= TURN_DEG), None)
    if seg is not None:
        s0, s1, dyaw_seg = seg
        peak = dyaw_seg                       # the FIRST committed turn
        onset = float(s0 / hz)
        kappa = float(kappa_t[s0:s1 + 1].max())
        R = 1.0 / kappa if kappa > 1e-4 else math.inf
        arc = float(np.sum(np.hypot(np.diff(x[s0:s1 + 2]),
                                    np.diff(y[s0:s1 + 2]))))
        i_peak = s1
        is_turn = R <= R_TURN_M
    else:
        i_apex = int(np.argmax(kappa_t))
        kappa = float(kappa_t[i_apex])
        R = 1.0 / kappa if kappa > 1e-4 else math.inf
        arc, is_turn = 0.0, False

    # -- longitudinal profile -----------------------------------------------
    eps = stop_episodes(v, hz)
    n_stops = len(eps)
    longest = max(((b - a + 1) / hz for a, b in eps), default=0.0)
    total = sum((b - a + 1) / hz for a, b in eps)
    v_key, v_min = float(v[0]), float(v.min())
    v_max, v_end = float(v.max()), float(v[-1])
    n_decel = decel_events(v)

    # did the ego slow FOR the turn? approach speed vs apex speed.
    slowed = None
    if is_turn:
        half = max(1, i_peak // 2)
        approach = float(v[:half].max())
        apex = float(v[half:i_peak + 1].min())
        slowed = bool(approach > 0.5 and apex / approach < TURN_SLOWDOWN_RATIO)

    # -- lateral class, with the ambiguous band resolved by speed -----------
    conf = "HIGH"
    if not is_turn:
        c, s = math.cos(-yaw[0]), math.sin(-yaw[0])
        lat = s * (x - x[0]) + c * (y - y[0])
        j = int(np.argmax(np.abs(lat)))
        if abs(peak) >= TURN_DEG:
            # a large heading change at a LARGE radius is road geometry, not a
            # decision — this is the branch that keeps a curving main road from
            # being called a junction turn (the LANE_TARGET failure mode).
            cls = f"ROAD_BEND_{'L' if peak > 0 else 'R'}"
            conf = "HIGH" if R >= R_BEND_M else "MEDIUM"
        elif abs(lat[j]) >= NUDGE_LAT_M and abs(peak) >= 5.0:
            cls = f"NUDGE_{'L' if lat[j] > 0 else 'R'}"
        else:
            cls = "STRAIGHT"
    else:
        side = "L" if peak > 0 else "R"
        # R <= R_TURN_M is guaranteed here (it is part of `is_turn`), so the
        # only question left is how confident the call is.
        cls = f"JUNCTION_TURN_{side}"
        conf = "HIGH" if R <= R_JUNCTION_M else "MEDIUM"

    # -- stop type ----------------------------------------------------------
    # ⚠️ KINEMATIC ONLY. "CONTROLLED" names the SHAPE of a light/sign stop —
    # one clean deceleration, held, clean pull-away. It does NOT assert a light
    # was present; only perception can say that. QUEUE is the discriminating
    # case the PI asked for: repetition, not depth, is what marks a jam.
    if n_stops == 0:
        stop_type = "NONE"
    elif v_key <= V_STOP_MS and n_stops == 1 and v_end > V_CRUISE_MS:
        stop_type = "ALREADY_STOPPED"
    elif n_stops >= 2 or (v_max < QUEUE_VMAX_MS and n_decel >= 2):
        stop_type = "QUEUE"
    elif longest <= 1.5 and cls.startswith("JUNCTION_TURN"):
        stop_type = "YIELD"
    elif n_stops == 1 and v_end > V_CRUISE_MS:
        stop_type = "CONTROLLED"
    else:
        stop_type, conf = "CONTROLLED", "LOW"

    if n_stops and v_key <= V_STOP_MS and v_end > V_CRUISE_MS:
        lon = "LAUNCH"
    elif stop_type == "QUEUE":
        lon = "STOP_AND_GO"
    elif n_stops:
        lon = "DECEL_TO_STOP"
    elif v_end - v_key <= -DECEL_SIGNIFICANT:
        lon = "SLOWING"
    else:
        lon = "CRUISE"

    return Manoeuvre(
        lateral_class=cls, peak_yaw_deg=round(peak, 1), yaw_onset_s=onset,
        turn_arc_m=round(arc, 1),
        turn_radius_m=(round(R, 1) if math.isfinite(R) else math.inf),
        kappa_max=round(kappa, 4), longitudinal_class=lon,
        stop_type=stop_type, n_stop_episodes=n_stops,
        longest_stop_s=round(longest, 1), total_stop_s=round(total, 1),
        v_at_key=round(v_key, 2), v_min=round(v_min, 2), v_max=round(v_max, 2),
        v_end=round(v_end, 2), n_decel_events=n_decel,
        n_turn_segments=n_segments, slowed_for_turn=slowed,
        horizon_s=round(horizon_s, 1), confidence=conf)
