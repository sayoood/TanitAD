"""R8-4 metric suite -- does the tactical layer CONDITION the planner? (refcv8 WP-B, `DESIGN.md` §4)

Pure numpy, no module-level mutable state, no torch.  Four families, each a function (or two):

  (i)   tactical accuracy   `class_report`, `goal_ap`, `constraint_mae`
  (ii)  controllability     `controllability`, `controllability_vs_shuffled`
  (iii) consistency         `consistency`
  (iv)  route following     `route_following`

FRAME.  Ego at NOW, x forward, y LEFT, metres.  A plan is ``[..., S, 2]`` positions at the slot times
``SLOT_T_S = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)`` seconds (the 8 slots of horizons 5,10,15,20,30,40,50,60
ticks at 0.1 s); the origin (0, 0) at t = 0 is implicit.  Every function that needs the clock takes ``slot_t``.

⛔ INDEPENDENCE.  This module is the CROSS-CHECK of the model's own tagger.  It imports nothing from ``tanitad``
(least of all ``tanitad.refs``), nothing from any ``refcv8_conditioning`` module, and nothing from the route package.
The path geometry is re-derived here from first principles and its constants are re-typed; the only sibling import is
``taniteval.ci`` (the program's episode-cluster bootstrap), and that one is LAZY (inside the two functions that
bootstrap), so importing this file needs numpy alone.  ``tests/test_tactical_conditioning.py`` holds the path
classifier against the route package's definitions on the banked refcv7 capture and requires EXACT agreement on the
direction class.

⚠️ PLAN-LEVEL CLASSES COVER [0, 6] s; THE v9 LABEL BAND IS [NOW+2, NOW+8] s.  They overlap only on [2, 6] s, and they
use different heading references (a plan class measures heading from the ego's yaw AT NOW; the v9 class measures
Theta = psi(tau*) - psi(NOW+2), from the band start).  A plan-level class is therefore a PROXY for the v9 label, never
the v9 label itself: an event the v9 band sees at 6-8 s is invisible to a 6-s plan, and a turn that is complete by
NOW+2 is invisible to v9.  Score a hypothesis only where it is observable inside the plan --
``observable_in_plan(t_start_s)`` (t_start <= 5 s leaves >= 1 s of plan to show the response) -- and report n for the rest.

CONVENTIONS THAT ARE EASY TO GET WRONG
* Headings are chord headings ``atan2`` of each slot-to-slot segment, so they are WRAPPED to (-pi, pi]; a plan that
  turns by more than a half circle inside 6 s aliases.  (Not reachable at the speeds/horizons of the corpus; stated.)
* A non-finite path never silently becomes a class: ``dir_class`` / ``lat3_class`` / ``lon_class`` return
  ``IGNORE_INDEX`` (-100) and ``stop_distance`` returns NaN.  (``np.where(th >= tau, ...)`` would call NaN "straight".)
* ``terminal_heading`` is the heading of the LAST segment (slot-50 -> slot-60 = ticks 50 -> 60 at the default slots),
  0.0 when that segment is shorter than 0.05 m -- the route package's definition, re-typed.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "SLOT_T_S", "STALL_M", "TAU_DIR_RAD", "TURN_DEG", "STRAIGHT_DEG", "HEAD_AGREE_DEG", "MIN_LEN_M",
    "V_STOP_MS", "V_CREEP_MS", "DV_MS", "MAX_EVENT_START_S", "ROUTE_DIR_BAR", "ROUTE_HEAD15_BAR",
    "CONTROLLABILITY_BAR", "IGNORE_INDEX", "LAT3_NAMES", "LON_NAMES", "FORCED_CLASSES",
    "V7_LAT_NAMES", "V7_LON_NAMES", "V7_LAT_TO_LAT3", "PLAN_LON_TO_V7", "PLAN_LON_ALLOWED_V7",
    "wrap_angle", "segment_headings", "terminal_heading", "dir_class", "heading_excursion", "path_length",
    "lat3_class", "segment_speeds", "stop_distance", "lon_class", "observable_in_plan",
    "plan_lon_to_v7_allowed",
    "per_window_correct", "class_report", "goal_ap", "constraint_mae", "default_stop_tol", "controllability",
    "controllability_vs_shuffled", "consistency", "gt_route_class", "route_following",
]

# --------------------------------------------------------------------------------------------------------------- #
# constants (re-typed on purpose -- never imported from the model's tagger or the route package)                    #
# --------------------------------------------------------------------------------------------------------------- #
SLOT_T_S = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
STALL_M = 0.05                          # a segment shorter than this has no heading
TAU_DIR_RAD = 0.18063741505146028       # the run's own --nav-compliance-tau-rad (10.35 deg): left / straight / right
TURN_DEG = 30.0                         # a TURN is a heading excursion of at least this much
STRAIGHT_DEG = 10.0                     # route class "straight": terminal heading below this
HEAD_AGREE_DEG = 15.0                   # "heading within 15 deg of GT"
MIN_LEN_M = 5.0                         # a plan shorter than this has no manoeuvre (LANE_KEEP / unclassified)
V_STOP_MS = 0.5                         # v9 §3.5: 0.5 m/s = 1.8 km/h, below any deliberate motion
V_CREEP_MS = 2.0                        # v9 §3.5: walking pace
DV_MS = 1.5                             # v9 §3.5: +-1.5 m/s = the v7 DV_ACCEL_MS / DV_BRAKE_MS bar
MAX_EVENT_START_S = 5.0                 # observable_in_plan: leave >= 1 s of the 6-s plan to show the response
ROUTE_DIR_BAR = 0.95                    # DESIGN §4 (iv): turn direction-correct
ROUTE_HEAD15_BAR = 0.70                 # DESIGN §4 (iv): heading within 15 deg
CONTROLLABILITY_BAR = 0.95              # DESIGN §4 (ii): per forced class
IGNORE_INDEX = -100

LAT3_NAMES = ("LANE_KEEP", "TURN_L", "TURN_R")
#: v9 §3.5 order.  A 6-s plan has no agent data, so `lon_class` NEVER emits FOLLOW (it is undetermined at plan level).
LON_NAMES = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")
FORCED_CLASSES = ("TURN_L", "TURN_R", "LANE_KEEP", "STOP")

#: the frozen v7 heads' orders (refcv7 `p_lat` / `p_lon` columns)
V7_LAT_NAMES = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
V7_LON_NAMES = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
#: v7 lat id -> lat3 id.  TURN_L -> 1, TURN_R -> 2, EVERYTHING ELSE -> 0 (a lane change / nudge is not a turn).
V7_LAT_TO_LAT3 = (0, 0, 0, 0, 0, 0, 1, 2)
#: plan-level lon class (LON_NAMES order) -> the single v7 lon id it maps to (v9 §3.7: STOP, DECELERATE -> BRAKE_TO;
#: KEEP -> CRUISE; FOLLOW stays FOLLOW).
PLAN_LON_TO_V7 = (5, 4, 3, 0, 3, 7, 1)
#: ... and the v7 ids a head may legitimately say.  FOLLOW is UNDETERMINED at plan level (no agents), so v9 §3.6 applies:
#: ``{computed class, FOLLOW}`` for DECELERATE / ACCELERATE / KEEP; HOLD / CREEP / STOP stay single-valued.
PLAN_LON_ALLOWED_V7 = ((5,), (4,), (3,), (0,), (3, 0), (7, 0), (1, 0))


# --------------------------------------------------------------------------------------------------------------- #
# path geometry                                                                                                     #
# --------------------------------------------------------------------------------------------------------------- #
def wrap_angle(a):
    """Wrap to [-pi, pi)."""
    return (np.asarray(a, np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def _paths(paths):
    P = np.asarray(paths, dtype=np.float64)
    if P.ndim < 2 or P.shape[-1] != 2:
        raise ValueError(f"paths must be [..., S, 2]; got shape {P.shape}")
    return P


def _slot_times(P, slot_t):
    t = np.asarray(slot_t, dtype=np.float64)
    if t.ndim != 1 or t.shape[0] != P.shape[-2]:
        raise ValueError(f"slot_t has {t.shape} entries but the paths have {P.shape[-2]} slots")
    if not (t[0] > 0.0 and np.all(np.diff(t) > 0.0)):
        raise ValueError(f"slot_t must be positive and strictly increasing; got {t}")
    return t


def _seg_vec(P):
    """[..., S, 2] displacement of each segment, origin -> slot1 -> slot2 -> ... -> slotS."""
    Q = np.concatenate([np.zeros(P.shape[:-2] + (1, 2)), P], axis=-2)
    return np.diff(Q, axis=-2)


def segment_headings(paths):
    """[..., S] chord heading (rad, left +) of each segment: origin->slot1, slot1->slot2, ...

    A segment shorter than ``STALL_M`` (0.05 m) has no direction and INHERITS the previous heading; the first
    segment's inherited default is 0.  A non-finite coordinate propagates as NaN.
    """
    P = _paths(paths)
    d = _seg_vec(P)
    ln = np.hypot(d[..., 0], d[..., 1])
    th = np.arctan2(d[..., 1], d[..., 0])
    out = np.empty_like(th)
    prev = np.zeros(th.shape[:-1])
    for j in range(th.shape[-1]):
        cur = np.where(ln[..., j] < STALL_M, prev, th[..., j])
        out[..., j] = cur
        prev = cur
    return out


def terminal_heading(paths):
    """Heading (rad) of the LAST segment (slot S-1 -> slot S; ticks 50 -> 60 at the default slots); 0.0 if shorter
    than 0.05 m.  NaN if the path is non-finite."""
    P = _paths(paths)
    if P.shape[-2] < 2:
        raise ValueError("terminal_heading needs >= 2 slots")
    d = P[..., -1, :] - P[..., -2, :]
    th = np.arctan2(d[..., 1], d[..., 0])
    th = np.where(np.hypot(d[..., 0], d[..., 1]) < STALL_M, 0.0, th)
    return np.where(np.isfinite(d).all(axis=-1), th, np.nan)


def dir_class(theta, tau=TAU_DIR_RAD):
    """{+1 left (theta >= tau), 0, -1 right (theta <= -tau)} as int8; ``IGNORE_INDEX`` where theta is not finite."""
    th = np.asarray(theta, dtype=np.float64)
    out = np.where(th >= tau, 1, np.where(th <= -tau, -1, 0))
    return np.where(np.isfinite(th), out, IGNORE_INDEX).astype(np.int8)


def heading_excursion(paths):
    """Signed max-|.| segment heading over the plan (rad): the value of the segment whose |heading| is largest, with its
    sign (left +).  NaN if any coordinate is non-finite.  (Measured from the ego's yaw at NOW.)"""
    h = segment_headings(paths)
    idx = np.argmax(np.abs(h), axis=-1)
    exc = np.take_along_axis(h, idx[..., None], axis=-1)[..., 0]
    return np.where(np.isfinite(h).all(axis=-1), exc, np.nan)


def path_length(paths):
    """Arc length of the polyline origin -> slot1 -> ... -> slotS, metres."""
    return np.linalg.norm(_seg_vec(_paths(paths)), axis=-1).sum(axis=-1)


def _cum_arc(paths):
    """[..., S] cumulative arc length at each slot.  (A module-level seam: the chord-for-arc mutation patches this.)"""
    return np.cumsum(np.linalg.norm(_seg_vec(_paths(paths)), axis=-1), axis=-1)


def lat3_class(paths, turn_deg=TURN_DEG, min_len_m=MIN_LEN_M):
    """0 LANE_KEEP / 1 TURN_L / 2 TURN_R.  TURN iff |heading_excursion| >= ``turn_deg`` (side = its sign); LANE_KEEP
    whenever the path is shorter than ``min_len_m`` (a parked ego has no manoeuvre).  ``IGNORE_INDEX`` if non-finite."""
    P = _paths(paths)
    exc = heading_excursion(P)
    ln = path_length(P)
    turn = np.degrees(np.abs(exc)) >= turn_deg
    cls = np.where(turn, np.where(exc > 0, 1, 2), 0)
    cls = np.where(ln < min_len_m, 0, cls)
    return np.where(np.isfinite(exc) & np.isfinite(ln), cls, IGNORE_INDEX).astype(np.int64)


def segment_speeds(paths, v0=None, slot_t=SLOT_T_S):
    """[..., S] MEAN speed (m/s) of each segment: segment length / its duration (the first segment spans 0 -> slot_t[0]).

    ``v0`` (the speed at NOW) does not enter the speeds; it is accepted so a caller cannot pass window-axis data that
    does not line up with the paths: when given it must broadcast against the leading axes of ``paths`` (ValueError
    otherwise).  Use it for the event thresholds in `lon_class`."""
    P = _paths(paths)
    t = _slot_times(P, slot_t)
    if v0 is not None:
        try:
            ok = np.broadcast_shapes(np.shape(v0), P.shape[:-2]) == P.shape[:-2]
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"v0 of shape {np.shape(v0)} does not broadcast to the paths' leading axes {P.shape[:-2]}")
    dt = np.diff(np.concatenate([[0.0], t]))
    return np.linalg.norm(_seg_vec(P), axis=-1) / dt


def stop_distance(paths, v_stop=V_STOP_MS, slot_t=SLOT_T_S):
    """Arc length (m) at which the plan first comes to rest, NaN if it never does.

    Rest = the first segment j whose MEAN speed is <= ``v_stop`` (0.5 m/s).  The crossing is placed INSIDE that segment
    by linear interpolation of the speed between the previous segment (v_{j-1} > v_stop) and this one (v_j <= v_stop):
    ``d = s_{j-1} + phi * (s_j - s_{j-1})``, ``phi = (v_{j-1} - v_stop) / (v_{j-1} - v_j)`` in (0, 1].  If the very first
    segment is already at rest the plan is stopped from NOW and the distance is 0.0.

    PRECISION (derived from the slot spacing): the flagged segment's own arc is ``v_j * dt_j <= v_stop * dt_max``
    = 0.5 m/s * 1.0 s = 0.5 m at the default slots, and ``d`` lies between its two ends, so the stop point is located to
    within 0.5 m of any point of the flagged segment.  Arc length, never chord: on a curved stop the chord from the
    origin under-reads (R 15 m, 25 m of arc -> 22.2 m of chord).
    """
    P = _paths(paths)
    v = segment_speeds(P, slot_t=slot_t)
    s = _cum_arc(P)
    s_start = np.concatenate([np.zeros(s.shape[:-1] + (1,)), s[..., :-1]], axis=-1)        # arc at each segment's start
    below = v <= v_stop
    has = below.any(axis=-1)
    j = np.argmax(below, axis=-1)
    jm = np.maximum(j - 1, 0)
    vj = np.take_along_axis(v, j[..., None], axis=-1)[..., 0]
    vp = np.take_along_axis(v, jm[..., None], axis=-1)[..., 0]
    s0 = np.take_along_axis(s_start, j[..., None], axis=-1)[..., 0]
    s1 = np.take_along_axis(s, j[..., None], axis=-1)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(j > 0, (vp - v_stop) / (vp - vj), 0.0)
    d = s0 + phi * (s1 - s0)
    return np.where(has & np.isfinite(v).all(axis=-1), d, np.nan)


def lon_class(paths, v0, slot_t=SLOT_T_S):
    """Plan-level longitudinal class id into ``LON_NAMES`` (v9 §3.5 literals adapted to the 6-s plan; FOLLOW is never
    emitted -- there are no agents).  Precedence top-down, on the segment speeds ``v_j`` and the NOW speed ``v0``:

      HOLD        v0 <= 0.5 and max v_j <= 0.5
      CREEP       max v_j <= 2.0   (SEGMENT speeds only, as the v9 text "max <= 2.0" and the D0 reference read it; v0 enters
                  HOLD and the event thresholds, so a >4 m/s^2 brake from v0 > 2 that is under 2 m/s on average
                  over the first 0.5 s reads CREEP -- documented, not reachable by a comfortable plan)
      STOP        some v_j <= 0.5 with v0 > 0.5, and no ACCEL event (v_j >= v0 + 1.5) before it
      DECELERATE / ACCELERATE   the EARLIER of the first v_j <= v0 - 1.5 and the first v_j >= v0 + 1.5
      KEEP        otherwise

    ``IGNORE_INDEX`` if the path is non-finite.  v0 shape: broadcastable to the leading axes of ``paths``.
    """
    P = _paths(paths)
    v = segment_speeds(P, v0, slot_t=slot_t)
    lead = v.shape[:-1]
    v0a = np.broadcast_to(np.asarray(v0, dtype=np.float64), lead)
    big = v.shape[-1] + 1

    def first(m):
        return np.where(m.any(axis=-1), np.argmax(m, axis=-1), big)

    vmax = v.max(axis=-1)
    i_stop = first((v <= V_STOP_MS) & (v0a > V_STOP_MS)[..., None])
    i_acc = first(v >= (v0a + DV_MS)[..., None])
    i_dec = first(v <= (v0a - DV_MS)[..., None])
    k = {n: i for i, n in enumerate(LON_NAMES)}
    cls = np.full(lead, k["KEEP"], dtype=np.int64)
    ev = (i_dec < big) | (i_acc < big)
    cls = np.where(ev & (i_dec < i_acc), k["DECELERATE"], cls)
    cls = np.where(ev & (i_acc < i_dec), k["ACCELERATE"], cls)
    cls = np.where((i_stop < big) & (i_stop < i_acc), k["STOP"], cls)
    cls = np.where(vmax <= V_CREEP_MS, k["CREEP"], cls)
    cls = np.where((v0a <= V_STOP_MS) & (vmax <= V_STOP_MS), k["HOLD"], cls)
    return np.where(np.isfinite(v).all(axis=-1) & np.isfinite(v0a), cls, IGNORE_INDEX).astype(np.int64)


def observable_in_plan(t_start_s, max_start_s=MAX_EVENT_START_S):
    """True where a hypothesis' event (v9 `turn_t_start_s`, `lon_t_start_s`, seconds from NOW) begins early enough for a
    6-s plan to show it: finite and ``t_start <= max_start_s`` (an event already in progress, t_start < 0 or < 2 s,
    is observable).  Anything else is scored as NOT observable and counted in n, never silently dropped."""
    t = np.asarray(t_start_s, dtype=np.float64)
    return np.isfinite(t) & (t <= max_start_s)


def plan_lon_to_v7_allowed(plan_lon):
    """[W, 8] bool: for each plan-level lon class, the v7 lon ids that count as correct (v9 §3.6 FOLLOW rule).
    ``IGNORE_INDEX`` rows are all-False (-> ignored by `class_report`)."""
    c = np.asarray(plan_lon)
    A = np.zeros(c.shape + (len(V7_LON_NAMES),), dtype=bool)
    for k, ids in enumerate(PLAN_LON_ALLOWED_V7):
        for i in ids:
            A[..., i] |= (c == k)
    return A


# --------------------------------------------------------------------------------------------------------------- #
# (i) tactical accuracy                                                                                             #
# --------------------------------------------------------------------------------------------------------------- #
def per_window_correct(pred, target, allowed=None, ignore=IGNORE_INDEX):
    """[W] float: 1.0 / 0.0 per scored window, NaN for windows that are not scored.

    Without ``allowed``: scored iff ``target != ignore``; correct iff ``pred == target``.
    With ``allowed`` ([W, C] bool, v9 §3.6 partial labels): scored iff ``target != ignore`` OR the row has a set bit;
    correct iff ``allowed[w, pred[w]]``.  A single-valued target whose row is EMPTY is read as the singleton {target}.
    A scored window with ``pred`` outside [0, C) is an error (it would otherwise be silently marked wrong or, worse,
    wrap around)."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 1:
        raise ValueError(f"pred/target must be matching 1-D arrays; got {pred.shape} vs {target.shape}")
    W = pred.shape[0]
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)
    if allowed is None:
        valid = target != ignore
        ok = pred == target
        return np.where(valid, ok.astype(np.float64), np.nan)
    A = np.array(allowed, dtype=bool)
    if A.ndim != 2 or A.shape[0] != W:
        raise ValueError(f"allowed must be [W, C]; got {A.shape} for W={W}")
    single = target != ignore
    empty = ~A.any(axis=1)
    fix = single & empty
    if fix.any():
        A[np.nonzero(fix)[0], target[fix]] = True
    valid = single | A.any(axis=1)
    if ((pred[valid] < 0) | (pred[valid] >= A.shape[1])).any():
        raise ValueError("a scored window has pred outside [0, C)")
    ok = A[np.arange(W), np.clip(pred, 0, A.shape[1] - 1)]
    return np.where(valid, ok.astype(np.float64), np.nan)


def _const_control(c, target, allowed, ignore):
    ok = per_window_correct(np.full(target.shape, c, dtype=np.int64), target, allowed, ignore)
    return float(np.nanmean(ok)) if np.isfinite(ok).any() else None


def class_report(pred, target, n_classes, names, allowed=None, ignore=IGNORE_INDEX, train_marginal=None):
    """Per-class n / recall / precision / F1, accuracy, macro-F1, and the two constant controls.

    * windows with ``target == ignore`` are not scored -- unless ``allowed`` ([W, C] bool partial-label mask, v9 §3.6)
      gives them a set bit, in which case ``pred`` counts correct iff ``allowed[w, pred]``;
    * per-class ``n`` / recall use the SINGLE-VALUED windows of that class (recall = share of them scored correct);
      precision = share of the scored windows PREDICTED as that class that are scored correct; F1 = harmonic mean
      (0.0 when the class exists but is never hit, ``None`` when n == 0);
    * ``macro_f1`` = mean F1 over classes with n > 0;
    * ``majority_control`` = the BEST constant predictor under the same scoring (for plain labels this is exactly the
      majority class' share); ``train_marginal_control`` = the constant ``argmax(train_marginal)`` (lowest id on ties).
    """
    pred = np.asarray(pred).astype(np.int64)
    target = np.asarray(target).astype(np.int64)
    if len(names) != n_classes:
        raise ValueError(f"names has {len(names)} entries for n_classes={n_classes}")
    if allowed is not None and np.shape(allowed)[-1:] != (n_classes,):
        raise ValueError(f"allowed must be [W, {n_classes}]; got {np.shape(allowed)}")
    ok = per_window_correct(pred, target, allowed, ignore)
    scored = np.isfinite(ok)
    if scored.any() and ((pred[scored] < 0) | (pred[scored] >= n_classes)).any():
        raise ValueError("a scored window has pred outside [0, n_classes)")
    single = scored & (target != ignore)
    if single.any() and ((target[single] < 0) | (target[single] >= n_classes)).any():
        raise ValueError("a single-valued target is outside [0, n_classes)")
    per_class, f1s = {}, []
    for k in range(n_classes):
        m_t = single & (target == k)
        m_p = scored & (pred == k)
        n, n_pred = int(m_t.sum()), int(m_p.sum())
        rec = float(ok[m_t].mean()) if n else None
        prec = float(ok[m_p].mean()) if n_pred else None
        if n == 0:
            f1 = None
        elif prec is None or (prec + rec) == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * prec * rec / (prec + rec)
        if f1 is not None:
            f1s.append(f1)
        per_class[names[k]] = {"n": n, "n_pred": n_pred, "recall": rec, "precision": prec, "f1": f1}
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(conf, (target[single], pred[single]), 1)
    best = None
    for c in range(n_classes):
        a = _const_control(c, target, allowed, ignore)
        if a is not None and (best is None or a > best[1]):
            best = (c, a)
    out = {
        "n_classes": int(n_classes), "names": list(names),
        "n_scored": int(scored.sum()), "n_ignored": int((~scored).sum()),
        "n_partial_scored": int((scored & (target == ignore)).sum()),
        "partial_label": allowed is not None,
        "accuracy": float(ok[scored].mean()) if scored.any() else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per_class, "confusion_strict": conf.tolist(),
        "majority_control": None if best is None else {"class": names[best[0]], "accuracy": best[1]},
        "train_marginal_control": None,
    }
    if train_marginal is not None:
        tm = np.asarray(train_marginal, dtype=np.float64)
        if tm.shape != (n_classes,):
            raise ValueError(f"train_marginal must be [{n_classes}]; got {tm.shape}")
        c = int(np.argmax(tm))
        out["train_marginal_control"] = {"class": names[c], "accuracy": _const_control(c, target, allowed, ignore)}
    return out


def _average_precision(scores, y):
    """AP with TIES TREATED AS ONE THRESHOLD (a tied block is a single point on the PR curve): sum over distinct
    thresholds of precision * (recall step).  A constant scorer therefore has one point, precision = prevalence,
    recall step 1.0, and AP == prevalence EXACTLY."""
    s = np.asarray(scores, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    n_pos = yy.sum()
    order = np.argsort(-s, kind="mergesort")
    s, yy = s[order], yy[order]
    ends = np.concatenate([np.nonzero(np.diff(s))[0], [s.shape[0] - 1]])
    tp = np.cumsum(yy)[ends]
    prec = tp / (ends + 1)
    rec = tp / n_pos
    return float(np.sum(prec * np.diff(np.concatenate([[0.0], rec]))))


def goal_ap(scores, y, w, tokens):
    """Per-token average precision over the cells with ``w > 0`` ([W, T] arrays; ``w == 0`` is IGNORE, v9 §4).

    -> ``{token: {"ap", "prevalence", "lift", "n_pos", "n_neg", "n"}}``.  ``ap`` is None when the token has no positive
    in scope.  A constant scorer reads ``ap == prevalence`` exactly (ties = one threshold), so ``lift`` = AP /
    prevalence is the skill beyond the base rate."""
    S = np.asarray(scores, dtype=np.float64)
    Y = np.asarray(y).astype(bool)
    Wt = np.asarray(w)
    if not (S.shape == Y.shape == Wt.shape and S.ndim == 2 and S.shape[1] == len(tokens)):
        raise ValueError(f"scores/y/w must be [W, {len(tokens)}]; got {S.shape}, {Y.shape}, {Wt.shape}")
    out = {}
    for t, name in enumerate(tokens):
        m = Wt[:, t] > 0
        if not np.isfinite(S[m, t]).all():
            raise ValueError(f"token {name!r}: non-finite score inside the counted cells")
        yy = Y[m, t]
        n, n_pos = int(m.sum()), int(yy.sum())
        n_neg = n - n_pos
        if n == 0 or n_pos == 0:
            out[name] = {"ap": None, "prevalence": (n_pos / n) if n else None, "lift": None,
                         "n_pos": n_pos, "n_neg": n_neg, "n": n}
            continue
        ap = _average_precision(S[m, t], yy)
        prev = n_pos / n
        out[name] = {"ap": ap, "prevalence": prev, "lift": ap / prev, "n_pos": n_pos, "n_neg": n_neg, "n": n}
    return out


def constraint_mae(pred, target, valid, const_value, prior=None):
    """Constraint regression error per field (t_start, dpsi, d_stop, v_target, t_reach, ...).

    ``pred`` / ``target`` / ``valid`` / ``const_value`` (and the optional ``prior``) are dicts ``field -> [W] array``
    (``const_value``: ``field -> scalar``, the TRAIN-median constant); plain arrays are read as one field "value".
    -> ``{field: {"mae", "mae_const", "ratio", "n", ["mae_prior", "ratio_prior"]}}`` over the windows with ``valid`` and
    a finite target.  ``ratio = mae / mae_const`` (< 1 beats the constant).  A non-finite PREDICTION inside the scored
    set is an error: dropping it would flatter the head."""
    def as_dict(x):
        return x if isinstance(x, dict) else {"value": x}
    P, T, V, C = as_dict(pred), as_dict(target), as_dict(valid), as_dict(const_value)
    PR = None if prior is None else as_dict(prior)
    out = {}
    for f in T:
        t = np.asarray(T[f], dtype=np.float64)
        p = np.asarray(P[f], dtype=np.float64)
        m = np.asarray(V[f], dtype=bool) & np.isfinite(t)
        if not np.isfinite(p[m]).all():
            raise ValueError(f"field {f!r}: non-finite prediction inside the scored windows")
        n = int(m.sum())
        row = {"n": n, "mae": None, "mae_const": None, "ratio": None}
        if n:
            mae = float(np.mean(np.abs(p[m] - t[m])))
            mc = float(np.mean(np.abs(float(C[f]) - t[m])))
            row.update(mae=mae, mae_const=mc, ratio=(mae / mc) if mc > 0 else None)
            if PR is not None:
                pp = np.asarray(PR[f], dtype=np.float64)
                if not np.isfinite(pp[m]).all():
                    raise ValueError(f"field {f!r}: non-finite prior inside the scored windows")
                mp = float(np.mean(np.abs(pp[m] - t[m])))
                row.update(mae_prior=mp, ratio_prior=(mae / mp) if mp > 0 else None)
        out[f] = row
    return out


# --------------------------------------------------------------------------------------------------------------- #
# (ii) controllability                                                                                              #
# --------------------------------------------------------------------------------------------------------------- #
def default_stop_tol(d):
    """DESIGN §4: |stop distance - d| <= max(2 m, 0.1 d)."""
    return np.maximum(2.0, 0.1 * np.asarray(d, dtype=np.float64))


def _tol_vec(stop_tol, d):
    """stop_tol applied to a [W] array of distances; a scalar-only callable (``lambda d: max(2.0, 0.1 * d)``) is applied
    element-wise."""
    try:
        t = np.asarray(stop_tol(d), dtype=np.float64)
        if t.shape == d.shape:
            return t
    except (ValueError, TypeError):
        pass
    return np.array([float(stop_tol(float(x))) for x in d], dtype=np.float64)


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def controllability(paths, forced, cond_mask, pick_idx, v0, stop_d=None, stop_tol=default_stop_tol, slot_t=SLOT_T_S):
    """Force a hypothesis ``forced`` in {TURN_L, TURN_R, LANE_KEEP, STOP} on the same scene and read how much of the
    fan obeys.  ``paths`` [W, N, S, 2] is the fan generated UNDER that forcing; ``cond_mask`` [W, N] marks the
    hypothesis-conditioned candidates (None = all); ``pick_idx`` [W]; ``v0`` [W] m/s; ``stop_d`` [W] the forced stop
    distance (STOP only).

    A candidate OBEYS when its OWN path class equals ``forced``:
      TURN_L / TURN_R / LANE_KEEP   ``dir_class(terminal_heading)`` = +1 / -1 / 0 (the ROUTE definition -- primary);
                                    the ``lat3_class`` reading (1 / 2 / 0) is reported beside it (``*_alt``);
      STOP                          ``|stop_distance - d| <= stop_tol(d)`` (primary; NaN never obeys); the
                                    ``lon_class == STOP`` reading is the ``*_alt``.
    -> per-window arrays ``share_cond`` (obeying share of the CONDITIONED candidates; NaN if a window has none),
    ``share_all`` (the unconditioned base rate of the same fan), ``pick_ok`` (the pick obeys), ``n_cond``, their
    ``*_alt`` twins, and the means over windows (``mean_*``)."""
    if forced not in FORCED_CLASSES:
        raise ValueError(f"forced must be one of {FORCED_CLASSES}; got {forced!r}")
    P = _paths(paths)
    if P.ndim != 4:
        raise ValueError(f"paths must be [W, N, S, 2]; got {P.shape}")
    W, N = P.shape[:2]
    pick = np.asarray(pick_idx).astype(np.int64)
    if pick.shape != (W,) or (pick < 0).any() or (pick >= N).any():
        raise ValueError("pick_idx must be [W] indices into the N candidates")
    cond = np.ones((W, N), dtype=bool) if cond_mask is None else np.asarray(cond_mask, dtype=bool)
    if cond.shape != (W, N):
        raise ValueError(f"cond_mask must be [W, N]; got {cond.shape}")
    v0a = np.asarray(v0, dtype=np.float64)
    if forced == "STOP":
        if stop_d is None:
            raise ValueError("forced STOP needs stop_d [W]")
        d = np.asarray(stop_d, dtype=np.float64)
        sd = stop_distance(P, slot_t=slot_t)
        with np.errstate(invalid="ignore"):
            ok = np.abs(sd - d[:, None]) <= _tol_vec(stop_tol, d)[:, None]
        ok = ok & np.isfinite(sd)
        ok_alt = lon_class(P, v0a[:, None], slot_t=slot_t) == LON_NAMES.index("STOP")
    else:
        code, code3 = {"TURN_L": (1, 1), "TURN_R": (-1, 2), "LANE_KEEP": (0, 0)}[forced]
        ok = dir_class(terminal_heading(P)) == code
        ok_alt = lat3_class(P) == code3
    ar = np.arange(W)
    n_cond = cond.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(n_cond > 0, (ok & cond).sum(axis=1) / np.maximum(n_cond, 1), np.nan)
        share_alt = np.where(n_cond > 0, (ok_alt & cond).sum(axis=1) / np.maximum(n_cond, 1), np.nan)
    out = {
        "forced": forced, "n_windows": int(W), "n_candidates": int(N),
        "n_cond": n_cond, "n_windows_no_cond": int((n_cond == 0).sum()),
        "share_cond": share, "share_cond_alt": share_alt,
        "share_all": ok.mean(axis=1), "share_all_alt": ok_alt.mean(axis=1),
        "pick_ok": ok[ar, pick], "pick_ok_alt": ok_alt[ar, pick],
        "alt_reading": "lon_class==STOP" if forced == "STOP" else "lat3_class",
    }
    for k in ("share_cond", "share_cond_alt", "share_all", "share_all_alt", "pick_ok", "pick_ok_alt"):
        out["mean_" + k] = _nanmean(out[k])
    return out


def controllability_vs_shuffled(forced_arm, shuffled_arm, eid, bar=CONTROLLABILITY_BAR, n_boot=2000, seed=0):
    """Paired episode-cluster bootstrap of the PICK-follow rate, forced arm minus shuffled-condition arm, and the verdict
    against the bar.  Both arguments are `controllability` results on the SAME windows (the shuffled arm's condition
    comes from a deranged window).

    verdict["pass"] needs ALL of: the forced arm's pick-follow rate >= ``bar`` (point estimate -- the committed
    reading; ``forced_ci_lo_ge_bar`` is the stricter one beside it), the shuffled arm < ``bar``, and the forced arm >
    the shuffled arm with a SEPARATED paired CI.  Estimator: ``taniteval.ci`` (paired_episode_cluster_bootstrap)."""
    from taniteval import ci as _ci
    a = np.asarray(forced_arm["pick_ok"], dtype=np.float64)
    b = np.asarray(shuffled_arm["pick_ok"], dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"arms are not on the same windows: {a.shape} vs {b.shape}")
    paired = _ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=n_boot, seed=seed)
    ci_f = _ci.episode_cluster_bootstrap(a, eid, n_boot=n_boot, seed=seed)
    ci_s = _ci.episode_cluster_bootstrap(b, eid, n_boot=n_boot, seed=seed)
    fr, sr = float(a.mean()), float(b.mean())
    verdict = {
        "bar": float(bar), "forced_pick_rate": fr, "shuffled_pick_rate": sr,
        "forced_meets_bar": bool(fr >= bar), "forced_ci_lo_ge_bar": bool(ci_f["lo"] >= bar),
        "shuffled_below_bar": bool(sr < bar),
        "forced_above_shuffled_separated": bool(paired["separated"] and paired["delta"] > 0),
    }
    verdict["pass"] = bool(verdict["forced_meets_bar"] and verdict["shuffled_below_bar"]
                           and verdict["forced_above_shuffled_separated"])
    return {"forced": ci_f, "shuffled": ci_s, "paired_delta": paired, "verdict": verdict,
            "estimator": "paired_episode_cluster_bootstrap", "answers": "another draw of EPISODES, not of training or inference"}


# --------------------------------------------------------------------------------------------------------------- #
# (iii) consistency                                                                                                 #
# --------------------------------------------------------------------------------------------------------------- #
def _cand_class(paths, reading):
    """[W, N] candidate class as lat3 ids (0 / 1 left / 2 right), by the chosen reading."""
    if reading == "lat3":
        return lat3_class(paths)
    if reading == "dir":
        d = dir_class(terminal_heading(paths))
        return np.where(d == IGNORE_INDEX, IGNORE_INDEX, np.where(d == 1, 1, np.where(d == -1, 2, 0))).astype(np.int64)
    raise ValueError(f"reading must be 'lat3' or 'dir'; got {reading!r}")


def consistency(paths, tags_lat, pick_idx, tac_argmax_lat=None, masks=None, reading="lat3"):
    """How often the fan / the pick agree with the hypothesis they carry.

    ``paths`` [W, N, S, 2]; ``tags_lat`` [W, N] lat3 ids (0 / 1 / 2) or -1 = untagged; ``pick_idx`` [W];
    ``tac_argmax_lat`` [W] the tactical head's lat3 argmax (or None).  ``reading``: ``"lat3"`` (candidate class =
    `lat3_class`, a 30-deg excursion) or ``"dir"`` (candidate class = `dir_class(terminal_heading)` mapped to ids
    0 / 1 / 2 -- the side-collapsed route reading of DESIGN §4).  ``masks``: ``{name: [W] bool}`` (default ``all``).

    -> per-window ``cand_share`` (share of TAGGED candidates whose class equals their tag; NaN if none),
    ``pick_own_tag`` (pick's class == the pick's tag; NaN if untagged), ``pick_vs_tac`` (pick's class == the tactical
    argmax; NaN if none given / negative), and per mask the means, the pooled candidate share and the n's."""
    P = _paths(paths)
    if P.ndim != 4:
        raise ValueError(f"paths must be [W, N, S, 2]; got {P.shape}")
    W, N = P.shape[:2]
    tags = np.asarray(tags_lat).astype(np.int64)
    if tags.shape != (W, N) or ((tags < -1) | (tags > 2)).any():
        raise ValueError("tags_lat must be [W, N] with values in {-1, 0, 1, 2}")
    pick = np.asarray(pick_idx).astype(np.int64)
    if pick.shape != (W,) or (pick < 0).any() or (pick >= N).any():
        raise ValueError("pick_idx must be [W] indices into the N candidates")
    cls = _cand_class(P, reading)
    tagged = tags >= 0
    match = tagged & (cls == tags)
    n_tag = tagged.sum(axis=1)
    ar = np.arange(W)
    with np.errstate(invalid="ignore", divide="ignore"):
        cand = np.where(n_tag > 0, match.sum(axis=1) / np.maximum(n_tag, 1), np.nan)
    pt, pc = tags[ar, pick], cls[ar, pick]
    pick_own = np.where(pt >= 0, (pc == pt).astype(np.float64), np.nan)
    if tac_argmax_lat is None:
        pick_tac = np.full(W, np.nan)
    else:
        tac = np.asarray(tac_argmax_lat).astype(np.int64)
        if tac.shape != (W,):
            raise ValueError("tac_argmax_lat must be [W]")
        pick_tac = np.where(tac >= 0, (pc == tac).astype(np.float64), np.nan)
    if masks is None:
        masks = {"all": np.ones(W, dtype=bool)}
    summary = {}
    for name, m in masks.items():
        m = np.asarray(m, dtype=bool)
        if m.shape != (W,):
            raise ValueError(f"mask {name!r} must be [W]")
        nt = int(tagged[m].sum())
        summary[name] = {
            "n_windows": int(m.sum()),
            "cand_share": _nanmean(cand[m]), "n_windows_cand": int(np.isfinite(cand[m]).sum()),
            "pooled_cand_share": (float(match[m].sum() / nt) if nt else None), "n_tagged_candidates": nt,
            "pick_own_tag": _nanmean(pick_own[m]), "n_windows_pick_tagged": int(np.isfinite(pick_own[m]).sum()),
            "pick_vs_tac": _nanmean(pick_tac[m]), "n_windows_pick_vs_tac": int(np.isfinite(pick_tac[m]).sum()),
        }
    return {"reading": reading, "cand_share": cand, "pick_own_tag": pick_own, "pick_vs_tac": pick_tac,
            "cand_class": cls, "summary": summary}


# --------------------------------------------------------------------------------------------------------------- #
# (iv) route following                                                                                              #
# --------------------------------------------------------------------------------------------------------------- #
def gt_route_class(gt_paths, gt_valid):
    """-> (cls, theta): per window ``'turnL' | 'turnR' | 'straight' | 'gentle' | 'unclassified'`` from the GT plan's
    terminal heading (>= 30 deg turn, < 10 deg straight, else gentle), classified only if slot-S is valid and the path
    is >= 5 m; and the terminal heading itself."""
    th = terminal_heading(gt_paths)
    ln = path_length(gt_paths)
    ok = np.asarray(gt_valid, dtype=bool)[..., -1] & (ln >= MIN_LEN_M)
    deg = np.degrees(np.abs(th))
    cls = np.full(th.shape, "unclassified", dtype="<U12")
    cls[ok & (deg >= TURN_DEG) & (th > 0)] = "turnL"
    cls[ok & (deg >= TURN_DEG) & (th < 0)] = "turnR"
    cls[ok & (deg < STRAIGHT_DEG)] = "straight"
    cls[ok & (deg >= STRAIGHT_DEG) & (deg < TURN_DEG)] = "gentle"
    return cls, th


def route_following(pick_paths, gt_paths, gt_valid, eid=None, n_boot=2000, seed=0):
    """Route-following of the shipped pick against the GT plan (the route package's definitions, re-typed).

    ``pick_paths`` / ``gt_paths`` [W, S, 2], ``gt_valid`` [W, S].  On the classified windows:
      ``dir_ok``  = dir_class(pick terminal heading) == dir_class(GT terminal heading)   ("turn direction-correct" on turns)
      ``head15``  = |wrap(pick terminal heading - GT terminal heading)| <= 15 deg
    A non-finite pick counts as WRONG (and is tallied in ``n_pick_nonfinite``).  The bars ``ROUTE_DIR_BAR`` (0.95) and
    ``ROUTE_HEAD15_BAR`` (0.70) are read on the GT-turn windows.  With ``eid`` each rate carries an episode-cluster
    bootstrap interval (`taniteval.ci`, ratio-of-sums over ALL episodes, B = ``n_boot``)."""
    pick = _paths(pick_paths)
    gt = _paths(gt_paths)
    if pick.shape != gt.shape or pick.ndim != 3:
        raise ValueError(f"pick/gt must be matching [W, S, 2]; got {pick.shape} vs {gt.shape}")
    cls, th_gt = gt_route_class(gt, gt_valid)
    classified = cls != "unclassified"
    th_p = terminal_heading(pick)
    dir_ok = np.where(classified, (dir_class(th_p) == dir_class(th_gt)).astype(np.float64), np.nan)
    with np.errstate(invalid="ignore"):
        h15 = np.abs(wrap_angle(th_p - th_gt)) <= math.radians(HEAD_AGREE_DEG)
    head15 = np.where(classified, h15.astype(np.float64), np.nan)
    sets = {"turn": np.isin(cls, ["turnL", "turnR"]), "turnL": cls == "turnL", "turnR": cls == "turnR",
            "straight": cls == "straight", "gentle": cls == "gentle", "all_classified": classified}
    summary = {}
    for name, m in sets.items():
        row = {"n": int(m.sum())}
        for key, arr in (("dir_correct", dir_ok), ("head15", head15)):
            v = np.where(m, arr, np.nan)
            row[key] = _nanmean(v)
            row["n_" + key] = int(np.nansum(v)) if m.any() else 0
            if eid is not None and m.any():
                from taniteval import ci as _ci
                row[key + "_ci"] = _ci.episode_cluster_bootstrap(v, eid, n_boot=n_boot, seed=seed)
        summary[name] = row
    t = summary["turn"]
    bars = {
        "dir_correct_turn": {"value": t["dir_correct"], "bar": ROUTE_DIR_BAR,
                             "meets": bool(t["dir_correct"] is not None and t["dir_correct"] >= ROUTE_DIR_BAR)},
        "head15_turn": {"value": t["head15"], "bar": ROUTE_HEAD15_BAR,
                        "meets": bool(t["head15"] is not None and t["head15"] >= ROUTE_HEAD15_BAR)},
    }
    return {"gt_cls": cls, "gt_theta": th_gt, "pick_theta": th_p, "dir_ok": dir_ok, "head15": head15,
            "summary": summary, "bars": bars, "n_pick_nonfinite": int((~np.isfinite(th_p)).sum())}
