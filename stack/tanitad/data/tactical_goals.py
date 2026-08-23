"""Hindsight derivation of TACTICAL GOALS (`g_tac`) from ego geometry.

⭐ WHY THIS EXISTS, AND WHY IT IS GEOMETRY-FIRST (PI review, 2026-08-23).
Two gaps were found by inspecting real clips:

 1. **`g_tac` was never DERIVED.** `TACTICAL_GOAL_TOKENS` exists in `v6.py`, a
    `TacticalGoalFan` PREDICTS it, and `tac_str_labels.compose()` composes one —
    but nothing in the s2 label path emits a hindsight tactical goal, so the
    validation report had no tactical goals to show. This module derives them.

 2. ⛔ **The existing composer's lateral tier reads ALPAMAYO, which is at CHANCE
    on this corpus.** MEASURED 2026-08-23, n=16 paired clips, 2000-shuffle
    permutation controls at five anchors AND whole-clip:

        alpamayo `lateral` vs ego turn side : 31.2 % real vs 23.9 % shuffled, p=0.335
        alpamayo `lane`    vs ego turn side : 20.0 % real vs 19.5 % shuffled, p=0.706

    Eight clips whose whole-clip yaw reaches 51-137 deg are labelled
    "Go Straight" / "Lane Keep". ⇒ A tactical LATERAL label derived from that
    axis is built on noise.

    ⚠️ THE JOIN ITSELF IS SOUND — that was tested separately and must not be
    confused with the above: the LONGITUDINAL axis DOES beat its own shuffle
    control (50.0 % vs 28.7 %, p=0.028 at t+4 s). A scrambled clip_id mapping
    would put BOTH axes at chance. So the defect is in Alpamayo's lateral
    output on this corpus, not in the indexing.

⇒ **Geometry decides the lateral axis. Alpamayo may only CORROBORATE it.**

⚠️ EARLIER CLAIM CORRECTED. An earlier pass reported "alpamayo lane -> tactical
LAT agrees 80 %". That number was a BASE-RATE ARTEFACT: 14/16 Alpamayo rows say
"Lane Keep" and the 2 s tactical window is almost all `lane_keep`, so two nearly
constant sequences agreed by construction. Against a class-balanced ego verdict
with a permutation control it collapses to chance. A raw agreement percentage
between two skewed variables is not evidence.

## The band

`g_tac` is defined over the TACTICAL BAND, 2-6 s (`v6.TAC_BAND_S`), NOT the
0-2 s window the factored `a_tac` labels use. That distinction is the direct
cause of a defect the PI spotted: a clip whose turn begins at t+6.1 s reads
`lane_keep` on a 2 s window while the frames plainly show a right turn. The
window was not wrong for `a_tac`; it was wrong as an answer about the tactical
GOAL.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from tanitad.data import ego_manoeuvre as EM

HZ_DEFAULT = 10.0
BAND_LO_S, BAND_HI_S = 2.0, 6.0        # v6.TAC_BAND_S

# lateral
CORRIDOR_OFFSET_M = 1.0     # lateral excursion that is a deliberate offset
EVADE_MIN_M = 1.0           # below this a 'nudge' is lane-keeping jitter
EVADE_MAX_M = 3.0           # beyond this it is not an in-corridor evasion
ANCHOR_MIN_M = 3.0          # a goal point nearer than this is degenerate
EVADE_RETURN_FRAC = 0.5     # excursion must decay to under this x its peak
EVADE_NET_YAW_DEG = 12.0    # an evasion ends on the ORIGINAL heading
# longitudinal
V_STOP_MS = 0.5
SPEED_BAND_TOL_MS = 1.5     # half-width of a held speed band
YIELD_MAX_STOP_S = 1.5      # a brief hold before proceeding

LAT_TOKENS = ("ANCHOR_GOAL", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR",
              "LAT_UNCONSTRAINED")
LON_TOKENS = ("SPEED_BAND", "GAP_TARGET", "YIELD_AT", "STOP_POINT",
              "WAIT_FOR_ONCOMING", "TRAFFIC_LIGHT_REACT", "LON_UNCONSTRAINED")


@dataclass
class TacticalGoal:
    lat_token: str
    lat_args: dict
    lat_provenance: str
    lon_token: str
    lon_args: dict
    lon_provenance: str
    band_s: tuple
    n_band_samples: int
    truncated: bool          # the band ran past the end of the clip

    def as_dict(self) -> dict:
        d = asdict(self)
        d["band_s"] = list(self.band_s)
        return d


def derive(poses, *, key: int = 0, hz: float = HZ_DEFAULT,
           stop_reason: str | None = None) -> TacticalGoal:
    """Derive `g_tac` over the 2-6 s tactical band from ego poses.

    ``stop_reason`` is the OPTIONAL semantic reason (light / sign / queue /
    hazard) from the VLM layer. It only ever refines a stop token that geometry
    has ALREADY established; it can never create one. That ordering is what
    keeps a hallucinated referent from inventing a goal.
    """
    p = np.asarray(poses, dtype=np.float64)
    lo = key + int(round(BAND_LO_S * hz))
    hi = key + int(round(BAND_HI_S * hz))
    truncated = hi > len(p) - 1
    hi = min(hi, len(p) - 1)
    if lo >= hi:
        return TacticalGoal("LAT_UNCONSTRAINED", {}, "band-empty",
                            "LON_UNCONSTRAINED", {}, "band-empty",
                            (BAND_LO_S, BAND_HI_S), 0, True)

    band = p[lo:hi + 1]
    x, y, yaw, v = p[:, 0], p[:, 1], p[:, 2], p[:, 3]

    # ego frame at the KEY, so args are expressed where the planner sits
    c, s = math.cos(-yaw[key]), math.sin(-yaw[key])
    dx, dy = x - x[key], y - y[key]
    ex, ey = c * dx - s * dy, s * dx + c * dy

    # ---- LATERAL -----------------------------------------------------------
    man = EM.analyse(p, key=key, hz=hz)
    lat_off = ey[lo:hi + 1]
    j = int(np.argmax(np.abs(lat_off)))
    peak_lat = float(lat_off[j])
    arc_end = float(np.hypot(ex[hi], ey[hi]))

    # ⛔ AN EVASION IS AN OUT-AND-BACK, NOT A LATERAL SHIFT. Geometry cannot see
    # the obstacle, so the ONLY kinematic signature separating "I moved around
    # something and came back into my corridor" from "I repositioned within the
    # lane" is whether the excursion RETURNS. MEASURED over 801 clips: with the
    # magnitude gate alone, 53 clips emitted EVADE_IN_CORRIDOR and **0 of 40
    # sampled showed any return — 100 % were monotonic shifts.** Every one was
    # a CORRIDOR_OFFSET wearing the wrong token.
    returns = (abs(float(lat_off[-1])) < EVADE_RETURN_FRAC * abs(peak_lat)
               if abs(peak_lat) > 1e-6 else False)
    # ⚠️ AND THE HEADING MUST COME BACK TOO. Judging this on the lateral class
    # alone fails on a SHARP evasion: a 2 m swerve completed in ~1.5 s swings
    # the heading 30 deg, which the manoeuvre classifier correctly calls a tight
    # turn — so the evasion was being filed as JUNCTION_TURN. The distinction
    # that actually holds is NET yaw across the band: an evasion ends on the
    # ORIGINAL heading (net ~0) because the vehicle returns to its corridor; a
    # junction turn ends on a new one. Peak yaw cannot separate them, net yaw can.
    net_yaw_band = abs(float(np.degrees(yaw[hi] - yaw[lo])))
    heading_returns = net_yaw_band <= EVADE_NET_YAW_DEG
    if (returns and heading_returns
            and EVADE_MIN_M <= abs(peak_lat) <= EVADE_MAX_M):
        # an in-corridor lateral evasion — bounded by the corridor, NOT a lane
        # change. The OBSTACLE is a perception fact and is deliberately left
        # unset here; geometry can see the swerve, never what caused it.
        # ⛔ EVADE NEEDS A FLOOR, NOT ONLY A CEILING. Without EVADE_MIN_M this
        # fired on ANY nudge classification, and MEASURED over 801 clips
        # **79 of 132 emissions (59.8 %) had |lat_offset| < 1.0 m** — down to
        # 0.01 m, i.e. ordinary lane-keeping jitter labelled as an evasive
        # manoeuvre. A token that fires on noise teaches the head that noise
        # means "evade".
        lat_tok = "EVADE_IN_CORRIDOR"
        lat_args = {"lat_offset_m": round(peak_lat, 2),
                    "obstacle_slot": None}
        lat_prov = "geometry(nudge)"
    elif abs(peak_lat) >= CORRIDOR_OFFSET_M and \
            not man.lateral_class.startswith("JUNCTION_TURN"):
        lat_tok = "CORRIDOR_OFFSET"
        lat_args = {"lat_offset_m": round(peak_lat, 2),
                    "arc_m": round(arc_end, 1)}
        lat_prov = "geometry(corridor)"
    else:
        # ⭐ THE DEFAULT IS ANCHOR_GOAL, NOT AN ABSTAIN. The geometric goal
        # point is ALWAYS hindsight-derivable and it is the lever the
        # literature actually shows working (+4.7 PDMS vs +0.2 for a
        # categorical command). Abstaining here would discard the one tactical
        # signal we can always produce.
        gx, gy = float(ex[hi]), float(ey[hi])
        # ⛔ ...BUT A GOAL POINT THE EGO NEVER REACHES IS NOT A GOAL. When the
        # vehicle is stopped or crawling through the band it barely displaces,
        # and the "goal" collapses onto the vehicle — MEASURED over 801 clips:
        # 22 goals within 2 m and **4 with goal_x NEGATIVE, i.e. BEHIND the
        # car**. Supervising an anchor head on those teaches it to aim at
        # itself. A degenerate goal abstains on the LAT axis and says why.
        if np.hypot(gx, gy) < ANCHOR_MIN_M:
            lat_tok = "LAT_UNCONSTRAINED"
            lat_args = {"goal_dist_m": round(float(np.hypot(gx, gy)), 2)}
            lat_prov = "abstain(ego near-stationary over the band)"
        else:
            lat_tok = "ANCHOR_GOAL"
            lat_args = {"goal_x_m": round(gx, 2), "goal_y_m": round(gy, 2),
                        "t_reach_s": round((hi - key) / hz, 1)}
            lat_prov = "geometry(goal-point)"

    # ---- LONGITUDINAL ------------------------------------------------------
    vb = v[lo:hi + 1]
    eps = EM.stop_episodes(v[key:hi + 1], hz)
    v_lo_b, v_hi_b = float(vb.min()), float(vb.max())

    if eps:
        i0, i1 = eps[-1]
        hold_s = (i1 - i0 + 1) / hz
        stop_arc = float(np.hypot(ex[key + i0], ey[key + i0]))
        if hold_s <= YIELD_MAX_STOP_S and \
                man.lateral_class.startswith("JUNCTION_TURN"):
            lon_tok, lon_prov = "YIELD_AT", "geometry(brief-hold-before-turn)"
            lon_args = {"position_arc_m": round(stop_arc, 1),
                        "gap_slot": None}
        else:
            lon_tok = "STOP_POINT"
            lon_args = {"position_arc_m": round(stop_arc, 1),
                        "hold_s": round(hold_s, 1),
                        "reason": stop_reason or "unknown"}
            lon_prov = ("geometry(stop)" if stop_reason is None
                        else f"geometry(stop)+vlm({stop_reason})")
        # ⚠️ the VLM may REFINE the reason of a stop geometry already found;
        # it may never CREATE the stop. See the module docstring.
    elif v_hi_b - v_lo_b <= 2 * SPEED_BAND_TOL_MS:
        mid = 0.5 * (v_hi_b + v_lo_b)
        lon_tok = "SPEED_BAND"
        lon_args = {"v_lo_ms": round(max(0.0, mid - SPEED_BAND_TOL_MS), 2),
                    "v_hi_ms": round(mid + SPEED_BAND_TOL_MS, 2)}
        lon_prov = "geometry(held-speed)"
    else:
        # speed moves but never settles and never stops: the band constrains
        # nothing we can name from ego alone. GAP_TARGET would need a lead
        # agent, which is perception, so the honest token is the abstain.
        lon_tok, lon_args = "LON_UNCONSTRAINED", {}
        lon_prov = "geometry(no-nameable-constraint)"

    assert lat_tok in LAT_TOKENS and lon_tok in LON_TOKENS
    return TacticalGoal(lat_tok, lat_args, lat_prov, lon_tok, lon_args,
                        lon_prov, (BAND_LO_S, BAND_HI_S), len(band), truncated)


def strategic_from_geometry(man: EM.Manoeuvre, *,
                            stop_onset_s: float | None = None,
                            turn_onset_s: float | None = None,
                            ) -> tuple[str, str]:
    """The strategic GOAL geometry alone can justify, with its reason.

    ⛔ EXISTS BECAUSE OF A REAL DEFECT: clip `5aef0388` executes an 89 deg right
    turn at R = 12.2 m and the pipeline emitted `NONE_ABSTAIN`. Abstention is
    for AMBIGUOUS geometry; an unambiguous junction turn is exactly the case
    where a goal IS derivable, and abstaining there is not caution, it is a
    dropped label.
    """
    lc = man.lateral_class
    turn = ("TURN_LEFT" if lc.startswith("JUNCTION_TURN_L") else
            "TURN_RIGHT" if lc.startswith("JUNCTION_TURN_R") else None)
    stops = man.stop_type in ("CONTROLLED", "QUEUE")

    # ⚠️ A CLIP CAN CARRY BOTH A STOP AND A TURN, and which one is the
    # STRATEGIC goal is decided by TEMPORAL ORDER — what the ego commits to
    # FIRST from the anchor. MEASURED: `3a0165bd` stops at t+1s and turns at
    # t+9.1s (stop first => STOP_AT, which is what the pipeline says), while
    # `4ee0b3f7` turns first. Ranking turn-over-stop unconditionally would
    # have overwritten a correct STOP_AT on the first clip.
    if turn and stops:
        # THE TURN WINS, ON PRINCIPLE, NOT ON TIMING. The strategic layer
        # names WHERE THE VEHICLE IS GOING; a stop does not change that. A
        # stop that PRECEDES a turn is the tactical MEANS of executing it
        # (stop sign, red light, yield) and belongs in `g_tac` as STOP_POINT,
        # which is exactly where `derive()` puts it.
        # I first ordered these by time and it produced STOP_AT on
        # `a2524c12`, whose CoT reads "slow down due to the stop sign ahead"
        # before a left turn -- demoting a route decision to its own
        # precondition. The composite is FLAGGED, never silently dropped.
        return turn, (f"junction turn R={man.turn_radius_m} m; COMPOSITE with "
                      f"a {man.stop_type} stop -> that stop belongs in g_tac")
    if turn:
        return turn, f"junction turn, R={man.turn_radius_m} m"
    # a stop with no junction turn: a slight nudge does not disqualify it
    if stops and (lc == "STRAIGHT" or lc.startswith("NUDGE")
                  or lc.startswith("ROAD_BEND")):
        return "STOP_AT", f"stop, type={man.stop_type} ({lc})"
    if lc.startswith("ROAD_BEND") or lc == "STRAIGHT" or lc.startswith("NUDGE"):
        return "FOLLOW_MAIN_ROAD", f"no junction manoeuvre ({lc})"
    return "NONE_ABSTAIN", f"geometry not decisive ({lc})"
