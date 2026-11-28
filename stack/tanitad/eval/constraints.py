"""Constraint scoring: the speed ENVELOPE and the CLEARANCE, both two-sided.

⭐ WHY THIS MODULE EXISTS. The PI asked (2026-09-06) for the model to emit goals
*and their constraints* -- position, time, and a speed envelope -- and to be
scored against them: *"The model must learn to adapt its speed depending on the
situation and should try to reach the max speed when the situation allows."*

⛔⛔ THE FINDING THAT SHAPES THE WHOLE DESIGN. There is no admissible MAX-SPEED
signal to feed. Measured 2026-09-06 (`…/Research/2026-09-06-constraints/`):

  * `g_tac.goals.SPEED_BAND` is present on 4,572/4,572 v7.2 train records and is
    literally ``[min(v), max(v)]`` of the EGO'S OWN FUTURE SPEED over the 2-6 s
    tactical band (`s2_geom_emit_v7.tactical_goals`, ``v = p[lo:hi+1, 3]``).
    Median width 1.44 m/s. It is a DESCRIPTION of what the ego did, not a
    property of the road, and it has *no* horizon guard whatsoever.
  * `a_tac.lon_args.v_target_ms` is ``round(m.v_min, 2)`` -- the MINIMUM speed
    over the plan window, not a desired speed.
  * `strata.road_class` cannot carry a ceiling either: it is DEFINED by a speed
    threshold (highway = >= 20 m/s sustained), so "road class predicts speed" is
    a tautology. The one genuinely external axis, `country`, explains 0.0605 of
    the variance against a 0.0311 shuffle floor.
  * There is no posted speed limit anywhere: no map, and the sign path is closed
    (F-14, `test_speed_band_derivation_blocker.py`; sign-text gate 0/31).

⇒ **THE DESIGN CONSEQUENCE: constraints are SCORED ON THE OUTPUT, never SUPPLIED
AT THE INPUT.** Every ceiling here is computed from the model's OWN proposed
trajectory and from perception it must already produce. Nothing in this module is
fed to a model, so no leak is possible by construction -- which is why it does
not need, and must not be given, a horizon guard argument. That is the whole
point: a constraint the planner must SATISFY cannot echo, because it is not an
input.

⛔ COORDINATE DISCIPLINE, INHERITED AND BINDING. The distance-keeping cost prices
a **GAP, not a CLOSING**, because the frozen trunk's closing rate is a clean null
on every arm (+0.0061) while position is decodable (paired vs pixels +0.4145
[+0.2018, +0.6120], constant control exactly +0.000000). Every quantity below is
a function of positions at a single step. No finite difference of a gap appears
anywhere in this file, and none may be added.

⚠️ WHAT `clearance` IS AND IS NOT. Measured 2026-09-06: `obstacle.offline` has
**no static/infrastructure class** -- all 11 labels are agents, and the 11th is
`train_or_tram_car` (69 boxes / 3 clips), not `protruding_object` (which was
always inside the canonical 10). "World-static" is a per-TRACK measured property
(1,756/2,778 tracks = 63.2 %, i.e. parked cars), never a class property. So this
module scores clearance to AGENT BOXES and the caller decides which tracks to
pass. Calling that "infrastructure clearance" would be a scope error; parked cars
are a *proxy* for the roadside edge and must be named as one.

⛔⛔ AND THE EGO FOOTPRINT IS NOT SUBTRACTED BY DEFAULT, BECAUSE **TWO
INCOMPATIBLE GAP CONVENTIONS ALREADY COEXIST IN THIS PROGRAMME** and silently
mixing them shifts every gap by 4.7 m:

  * `taniteval/lead_source.py` -- the convention on the `obstacle.offline` join
    path -- is ``gap = along - size_x/2``, **rig origin to the box's rear face**,
    and its own comment says *"never bumper-to-bumper"*.
  * `stack/experiments/alpasim-gsplat/cl_metrics.py` -- the closed-loop sim
    branch -- is ``headway = xlead - EGO_LEN`` with ``EGO_LEN = 4.7`` m, i.e.
    **true bumper-to-bumper**.

⚠️ An earlier probe of mine reported "no EGO_L/EGO_W constant exists in the
repo". That was WRONG: the grep had been truncated mid-run, and an absence read
off a truncated search is a claim about the search. `EGO_LEN` is real. Root-cause
class: this repo's "a search tool reports no matches for what it could not read".

⇒ `ego_half_l`/`ego_half_w` default to 0.0, so the returned clearance is
**rig-origin-to-box-edge**, matching the join path's own convention. Pass
``ego_half_l=EGO_LEN/2`` only when you mean bumper-to-bumper, and say so. The
datum travels with the number as `CLEARANCE_DATUM` so it can never be quoted as
bumper-to-bumper when it is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- physical constants ------------------------------------------------------
G_MS2 = 9.80665

#: Lateral-acceleration ceiling for the kinematic speed envelope, m/s^2.
#: 0.35 g. Deliberately a COMFORT bound, not a friction bound: a friction-limit
#: ceiling (mu ~ 0.7 => 6.9 m/s^2) is so loose that almost no urban trajectory
#: ever violates it, which makes the metric vacuous. Stated as a parameter on
#: every entry point so an arm can move it and must declare that it did.
A_LAT_MAX_MS2 = 3.4

#: Time-headway used to turn a GAP into a speed ceiling, seconds.
#: Same shape as the banked distance-keeping cost's ``d0 + tau*v``, solved for v.
TAU_HEADWAY_S = 1.6

#: Standoff distance kept at zero speed, metres.
D0_STANDOFF_M = 3.0

#: A curvature below this is treated as straight; the kinematic ceiling is then
#: unbounded and the window is CENSORED for the kinematic family rather than
#: given an enormous finite ceiling that no arm could ever violate.
KAPPA_STRAIGHT_1_M = 1e-3

#: ⛔ Curvature ABOVE this is pose noise, not steering, and is clamped.
#: A passenger car's minimum turning radius is ~5 m => kappa ~ 0.2 1/m. MEASURED
#: 2026-09-06 on the B1 eval GT path: an unclamped double difference produced a
#: kinematic ceiling whose minimum was 0.279 m/s, i.e. kappa = 43.7 1/m -- a 2.3
#: cm turning radius. This is the vtarget jitter trap in curvature costume:
#: differentiating a jittery track TWICE amplifies the jitter, and the resulting
#: absurdly low ceiling then reads as a constraint violation.
KAPPA_MAX_1_M = 0.2

#: Lateral half-width of the ego's corridor for the LEAD test, metres.
#: Matches `taniteval.lead_source`'s LEAD_LAT_M so the two agree by construction.
LEAD_LAT_M = 2.0

#: Below this ceiling-headroom the situation does NOT "allow" max speed, so the
#: under-driving side is not scored on that window. Prevents penalising an arm
#: for slowing down where slowing down is correct.
UNOBSTRUCTED_CEILING_MS = 8.0

#: A planned speed this far below the allowed ceiling counts as under-driving.
UNDERDRIVE_MARGIN_MS = 3.0


class Censored(ValueError):
    """Raised when a quantity is not computable; callers must count, not drop."""


# --- speed ceilings ----------------------------------------------------------

def kinematic_speed_ceiling(kappa, *, a_lat_max: float = A_LAT_MAX_MS2,
                            kappa_straight: float = KAPPA_STRAIGHT_1_M,
                            kappa_max: float = KAPPA_MAX_1_M):
    """Max speed the PROPOSED path admits at each step: ``sqrt(a_lat_max/|k|)``.

    ``kappa`` is the signed curvature [1/m] of the model's own trajectory.
    Returns ``(v_max, valid)``; ``valid`` is False where ``|kappa|`` is below
    ``kappa_straight`` -- there the path is straight and imposes no ceiling, and
    that step is CENSORED for this family rather than handed ``+inf``.

    ⛔ ``|kappa|`` is CLAMPED at ``kappa_max``. Above a car's minimum turning
    radius the value is pose noise doubly amplified, not steering, and without
    the clamp it manufactures a ceiling below walking pace that every arm then
    "violates". See KAPPA_MAX_1_M.

    ⛔ This reads the OUTPUT. It is never an input, so it cannot leak.
    """
    k = np.abs(np.asarray(kappa, dtype=np.float64))
    valid = np.isfinite(k) & (k >= kappa_straight)
    k = np.minimum(k, kappa_max)
    v = np.full(k.shape, np.nan)
    np.divide(a_lat_max, k, out=v, where=valid)
    np.sqrt(v, out=v, where=valid)
    return v, valid


def clearance_speed_ceiling(gap_m, *, tau: float = TAU_HEADWAY_S,
                            d0: float = D0_STANDOFF_M):
    """Max speed the measured GAP admits: ``max(0, (gap - d0) / tau)``.

    The inverse of the banked distance-keeping cost's ``d0 + tau*v <= gap``.

    ⛔ A GAP, NOT A CLOSING. ``gap_m`` is a distance at ONE step. Passing a
    closing rate here would reintroduce exactly the term the frozen-trunk null
    says is not decodable.
    """
    g = np.asarray(gap_m, dtype=np.float64)
    valid = np.isfinite(g)
    v = np.full(g.shape, np.nan)
    np.subtract(g, d0, out=v, where=valid)
    np.divide(v, tau, out=v, where=valid)
    return np.maximum(v, 0.0, out=v, where=valid), valid


# --- clearance ---------------------------------------------------------------

def _rect_corners(cx, cy, yaw, l, w):
    hl, hw = l / 2.0, w / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy)
            for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _point_to_segment(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    den = vx * vx + vy * vy
    t = 0.0 if den == 0.0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / den))
    dx, dy = px - (ax + t * vx), py - (ay + t * vy)
    return math.hypot(dx, dy)


def point_to_box_clearance(px, py, box) -> float:
    """Signed-outside distance from a point to an ORIENTED BEV rectangle.

    ``box`` = ``(cx, cy, yaw, l, w)`` in the ego frame (+x fwd, +y LEFT), the
    exact schema `build_b1_agent_join` emits. Returns 0.0 when the point is
    inside the rectangle -- a penetration is a collision, and reporting a
    negative depth would let a deep penetration average away against slack
    elsewhere.
    """
    cx, cy, yaw, l, w = box
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx, dy = px - cx, py - cy
    lx, ly = c * dx - s * dy, s * dx + c * dy      # into the box's own frame
    ox = max(abs(lx) - l / 2.0, 0.0)
    oy = max(abs(ly) - w / 2.0, 0.0)
    return math.hypot(ox, oy)


def lead_gap(boxes, *, lat_m: float = LEAD_LAT_M, max_range_m: float = 60.0):
    """⭐ Longitudinal gap to the nearest IN-CORRIDOR agent AHEAD, or None.

    ``boxes``: ``(cx, cy, yaw, l, w)`` in the ego frame (+x fwd, +y LEFT).
    Returns ``gap = cx - l/2`` for the nearest qualifying box -- rig origin to
    its rear face, `taniteval.lead_source`'s convention exactly -- or ``None``
    when the corridor is empty (which the caller must CENSOR, not score).

    ⛔⛔ WHY THIS EXISTS AND WHY `trajectory_clearance` MUST NOT BE USED HERE.
    MEASURED 2026-09-06, and the GT control is what caught it: feeding the
    minimum distance to ANY agent within 60 m into `clearance_speed_ceiling`
    produced a median ceiling of **3.16 m/s (11 km/h)** on the B1 eval split, so
    the ground-truth human trajectory "violated" its own envelope on **71.2 %**
    of steps and even a 1 m/s creeping arm "over-drove" on 27.3 %. A car in the
    adjacent lane 2 m to the side is not a longitudinal constraint; a parked car
    the ego is passing is not one either. Only what is AHEAD and IN THE CORRIDOR
    bounds speed.

    ⇒ **Two different quantities, and conflating them is the defect:**
      * `trajectory_clearance` -- min distance to ANY agent. A PROXIMITY /
        collision-margin statistic. Report its distribution; never make it a
        speed ceiling.
      * `lead_gap` -- the in-corridor agent ahead. THIS is what bounds speed.

    ⛔ A GAP, NOT A CLOSING: one frame in, one distance out.
    """
    best = None
    for b in boxes:
        cx, cy, _yaw, l, _w = float(b[0]), float(b[1]), b[2], float(b[3]), b[4]
        if cx <= 0.0 or cx > max_range_m or abs(cy) >= lat_m:
            continue
        g = cx - l / 2.0
        if best is None or g < best:
            best = g
    return None if best is None else max(best, 0.0)


@dataclass
class ClearanceResult:
    min_clearance_m: float
    n_boxes_considered: int
    censored: bool
    reason: str = ""
    per_step_min: list = field(default_factory=list)


def trajectory_clearance(traj_xy, boxes, *, ego_half_l: float = 0.0,
                         ego_half_w: float = 0.0,
                         max_range_m: float = 60.0) -> ClearanceResult:
    """Minimum clearance from a planned trajectory to a set of agent boxes.

    ``traj_xy``: ``[K, 2]`` ego-frame waypoints of the PROPOSED trajectory.
    ``boxes``:   iterable of ``(cx, cy, yaw, l, w)`` in the SAME ego frame.

    ⚠️ ``ego_half_l``/``ego_half_w`` default to 0.0 and the result is therefore
    **rig-origin-to-box-edge**, not bumper-to-bumper -- no ego footprint constant
    exists in this repo. Pass them when one lands; the datum is reported.

    ⛔ CENSORING IS EXPLICIT. A window with no box inside ``max_range_m`` is
    returned ``censored=True`` with ``min_clearance_m = nan`` and a reason. It is
    NOT scored as "infinite clearance": an unobserved constraint is a missing
    measurement, and averaging it in as a perfect score is how a metric rewards
    blindness. Callers report the censored count per family, never drop it.
    """
    P = np.asarray(traj_xy, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"traj_xy must be [K,2], got {P.shape}")
    inflate = math.hypot(ego_half_l, ego_half_w)
    near = [b for b in boxes
            if math.hypot(float(b[0]), float(b[1])) <= max_range_m]
    if not near:
        return ClearanceResult(float("nan"), 0, True,
                               f"no agent box within {max_range_m:g} m")
    per_step = []
    for px, py in P:
        d = min(point_to_box_clearance(px, py, b) for b in near)
        per_step.append(max(d - inflate, 0.0))
    return ClearanceResult(float(min(per_step)), len(near), False,
                           per_step_min=per_step)


# --- the two-sided envelope --------------------------------------------------

def speed_envelope_report(v_planned, *, v_ceiling, ceiling_valid,
                          manoeuvre_flag=None,
                          unobstructed_ceiling: float = UNOBSTRUCTED_CEILING_MS,
                          underdrive_margin: float = UNDERDRIVE_MARGIN_MS) -> dict:
    """⭐ THE TWO-SIDED CRITERION. Over-driving AND under-driving, never pooled.

    ``v_planned``:  ``[K]`` speed along the model's own proposed trajectory.
    ``v_ceiling``:  ``[K]`` the allowed max at each step (elementwise min of the
                    kinematic and clearance ceilings).
    ``ceiling_valid``: ``[K]`` bool; False steps are CENSORED, not scored.

    Returns fractions **over** and **under**, each with its own denominator, plus
    the signed distribution. ⛔ They are never combined into one score: a single
    scalar lets an arm buy an over-driving win with an under-driving loss, which
    is precisely the trade-off the PI asked to see.

    ⛔ THE VACUITY GATE. ``manoeuvre_flag`` (``[K]`` bool, or a scalar bool for
    the window) is carried through to ``manoeuvre_rate`` in the result and the
    caller MUST report it beside every satisfaction number. MEASURED 2026-09-06
    on a sibling stream: a friction-circle zero was bought by a `turn_left`
    recall of exactly 0.0000 -- an arm that DECLINES the manoeuvre satisfies
    every constraint trivially. A safety zero means nothing without the rate.
    """
    v = np.asarray(v_planned, dtype=np.float64)
    c = np.asarray(v_ceiling, dtype=np.float64)
    ok = np.asarray(ceiling_valid, dtype=bool) & np.isfinite(v) & np.isfinite(c)
    n = int(ok.sum())
    if n == 0:
        return {"status": "CENSORED", "n_scored": 0, "n_steps": int(v.size),
                "reason": "no step has a valid ceiling"}

    excess = v[ok] - c[ok]                       # >0 over the ceiling
    over = excess > 0.0
    # The under-driving side is scored ONLY where the situation ALLOWS speed --
    # a high ceiling. Penalising slowness where slowness is correct would make
    # the metric reward recklessness.
    allows = c[ok] >= unobstructed_ceiling
    under = allows & (excess < -abs(underdrive_margin))

    if manoeuvre_flag is None:
        man_rate = None
    else:
        mf = np.asarray(manoeuvre_flag, dtype=bool)
        man_rate = float(mf.mean()) if mf.size else None

    return {
        "status": "OK",
        "n_steps": int(v.size),
        "n_scored": n,
        "n_censored": int(v.size - n),
        # --- the two sides, each with its OWN denominator ---
        "frac_over_ceiling": round(float(over.mean()), 6),
        "n_over": int(over.sum()),
        "mean_overshoot_ms": round(float(excess[over].mean()), 4) if over.any() else 0.0,
        "p95_overshoot_ms": (round(float(np.percentile(excess[over], 95)), 4)
                             if over.any() else 0.0),
        "n_situation_allows": int(allows.sum()),
        "frac_under_when_allowed": (round(float(under.sum() / allows.sum()), 6)
                                    if allows.any() else None),
        "n_under": int(under.sum()),
        "mean_shortfall_ms": (round(float(-excess[under].mean()), 4)
                              if under.any() else 0.0),
        # --- the distribution, because a scalar hides the trade-off ---
        "signed_excess_ms": {
            "p05": round(float(np.percentile(excess, 5)), 4),
            "p50": round(float(np.percentile(excess, 50)), 4),
            "p95": round(float(np.percentile(excess, 95)), 4),
            "mean": round(float(excess.mean()), 4),
        },
        # --- the vacuity gate ---
        "manoeuvre_rate": man_rate,
        "_vacuity_warning": ("report manoeuvre_rate beside every number here: a "
                             "constraint-satisfaction zero bought by declining "
                             "the manoeuvre is not a safety result"),
        "_params": {"unobstructed_ceiling_ms": unobstructed_ceiling,
                    "underdrive_margin_ms": underdrive_margin},
    }


def combined_ceiling(kinematic, kin_valid, clearance, clr_valid):
    """Elementwise min of the two ceilings; valid where EITHER is valid.

    A step with neither ceiling is censored. A step with one is scored against
    that one -- a straight road with a lead still has a clearance ceiling, and a
    curve with no agent in range still has a kinematic one.
    """
    k = np.asarray(kinematic, dtype=np.float64)
    c = np.asarray(clearance, dtype=np.float64)
    kv = np.asarray(kin_valid, dtype=bool)
    cv = np.asarray(clr_valid, dtype=bool)
    out = np.full(k.shape, np.inf)
    out = np.where(kv, np.minimum(out, np.nan_to_num(k, nan=np.inf)), out)
    out = np.where(cv, np.minimum(out, np.nan_to_num(c, nan=np.inf)), out)
    valid = kv | cv
    out = np.where(valid, out, np.nan)
    return out, valid


def curvature_from_xy(traj_xy, *, eps: float = 1e-9):
    """Signed curvature [1/m] of a planned path, from POSITIONS only.

    ⛔ This is a property of the OUTPUT path's shape, not a time derivative of a
    gap. It is the one place a second difference is legitimate, because the path
    geometry IS the thing being constrained.
    """
    P = np.asarray(traj_xy, dtype=np.float64)
    dx = np.gradient(P[:, 0])
    dy = np.gradient(P[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    den = np.power(dx * dx + dy * dy, 1.5)
    k = np.zeros_like(den)
    good = den > eps
    k[good] = (dx[good] * ddy[good] - dy[good] * ddx[good]) / den[good]
    return k, good


CLEARANCE_DATUM = (
    "rig-origin to box EDGE (oriented BEV rectangle); the ego footprint is NOT "
    "subtracted unless ego_half_l/ego_half_w are passed. This matches "
    "lead_source's 'gap = along - size_x/2' convention and is NOT the "
    "bumper-to-bumper convention of alpasim-gsplat/cl_metrics.py "
    "(EGO_LEN = 4.7 m) -- the two differ by 4.7 m and must never be mixed")
