"""FOUR-FAMILY instrument for the inverse-dynamics model (IDM).

WHY THIS EXISTS
---------------
The binding rule (CLAUDE.md, Sayed 2026-08-02) is that every eval reports
LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC **in addition to** ADE, and that
"a missing metric is a work item, not an excuse". The IDM stream satisfies none
of it: **no IDM script imports ``four_families``** (two independent probes over
``stack/scripts/idm_*.py``, ``run_idm_*.py`` and every ``idm*`` hub script — all
consumers are world-model side). Its entire published validation is four scalar
R² values. So the IDM can be "good" on speed R² while labelling the wrong
manoeuvre on every turn, and nothing in the program would see it — which matters
because the IDM's whole purpose is to mint pseudo-labels for action-free video.

TWO DEFECTS THIS FIXES
----------------------
1. **No TACTICAL instrument at all.** :func:`tactical` below derives the
   manoeuvre the IDM effectively *selected* from its own predicted trajectory,
   against the manoeuvre the human *executed*, and reports the confusion.

2. **The manoeuvre class is FACTORED, not a mixed 5-way softmax.** The program's
   diagnosed root cause of longitudinal blindness is that a single 5-way
   manoeuvre softmax mixes the lateral and longitudinal axes, which is how
   "0/881 accelerate" and the speed-fan happened. Reporting one pooled class
   here would rebuild that defect inside the instrument meant to detect it, so
   :func:`manoeuvre_classes` emits a lateral class and a longitudinal class
   **separately**; the legacy mixed class is emitted too, but only so the
   collapse it causes stays visible.

CADENCE — the trap that makes a naive reuse silently wrong
----------------------------------------------------------
``taniteval.four_families`` hard-codes ``DT_S = 0.1`` (10 Hz waypoints). The IDM
emits **4** waypoints at horizons {5,10,15,20} steps — i.e. **0.5 s apart**.
Feeding IDM trajectories to that module unchanged reads every speed and yaw-rate
**5x too large**. :func:`geometry` therefore takes the cadence explicitly, and
``tests/test_idm_families.py`` asserts it reproduces
``four_families._seq_geometry`` exactly at ``dt=0.1`` — same definitions, right
cadence.

⚠️ Curvature and yaw-rate from 4 sparse waypoints have only ``H-1 = 3``
pair-valid steps per window and are correspondingly noisy. That is reported, not
hidden: every family carries its own ``n``.
"""

from __future__ import annotations

import math

import numpy as np

#: below this per-step arc length a step carries no reliable path tangent
#: (``four_families.MIN_DS_M``); scaled by cadence since it is a *distance*.
MIN_DS_M = 0.05
_EPS = 1e-8

#: IDM horizons in 10 Hz steps (``idm_head.DEFAULT_HORIZONS``) -> 0.5 s spacing
IDM_HORIZONS = (5, 10, 15, 20)
IDM_DT_S = 0.5

#: manoeuvre thresholds. Lateral: total heading change over the 2 s horizon.
#: 0.15 rad (8.6 deg) over 2 s separates a lane change / turn from lane-keeping
#: jitter; longitudinal: 1.0 m/s speed change over 2 s (0.5 m/s^2 sustained).
LAT_TURN_RAD = 0.15
LON_DV_MPS = 1.0
LATERAL_CLASSES = ("right", "straight", "left")
LONGITUDINAL_CLASSES = ("decelerate", "cruise", "accelerate")


# --------------------------------------------------------------------------- #
# geometry — four_families._seq_geometry with the cadence made explicit        #
# --------------------------------------------------------------------------- #
def geometry(wp, dt: float = IDM_DT_S) -> dict:
    """``wp`` [n, H, 2] ego-frame metres -> per-step path geometry at cadence ``dt``.

    Definitions are ``four_families._seq_geometry`` verbatim; only ``DT_S`` is a
    parameter instead of a module constant. The origin is prepended so step 0 is
    measured from the ego's own position, headings are unwrapped along the
    horizon so a +-pi crossing cannot create a fake spike, and steps shorter than
    ``MIN_DS_M`` (scaled to this cadence) are masked because a crawling vehicle
    has no meaningful tangent.
    """
    wp = np.asarray(wp, dtype=np.float64)
    if wp.ndim != 3 or wp.shape[-1] != 2:
        raise ValueError(f"waypoints must be [n, H, 2], got {wp.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    p = np.concatenate([np.zeros_like(wp[:, :1]), wp], axis=1)      # [n,H+1,2]
    d = p[:, 1:] - p[:, :-1]
    ds = np.linalg.norm(d, axis=-1)
    speed = ds / dt
    min_ds = MIN_DS_M * (dt / 0.1)          # same *speed* floor at any cadence
    valid = ds > min_ds
    heading = np.arctan2(d[..., 1], d[..., 0])
    dh = heading[:, 1:] - heading[:, :-1]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    yaw_rate = dh / dt
    ds_mid = 0.5 * (ds[:, 1:] + ds[:, :-1])
    curvature = dh / (ds_mid + _EPS)
    return {"speed": speed, "heading": heading, "valid": valid,
            "yaw_rate": yaw_rate, "curvature": curvature,
            "pair_valid": valid[:, 1:] & valid[:, :-1],
            "accel": (speed[:, 1:] - speed[:, :-1]) / dt,
            "along": p[..., 0][:, 1:], "cross": p[..., 1][:, 1:]}


def _mean(x, m):
    n = int(np.sum(m))
    return (float("nan"), 0) if n == 0 else (float(np.mean(x[m])), n)


# --------------------------------------------------------------------------- #
# manoeuvre classes — FACTORED (see module docstring)                          #
# --------------------------------------------------------------------------- #
def manoeuvre_classes(wp, dt: float = IDM_DT_S, *,
                      lat_thresh: float = LAT_TURN_RAD,
                      lon_thresh: float = LON_DV_MPS) -> dict:
    """Manoeuvre implied by a trajectory -> ``{"lateral": [n], "longitudinal": [n],
    "mixed": [n]}`` as integer class ids.

    ``lateral`` is decided on the TOTAL heading change from the first to the last
    path tangent (not the instantaneous yaw rate, which a single noisy sparse
    step can dominate). ``longitudinal`` is decided on the speed change across
    the horizon. ``mixed`` is the legacy 5-way collapse
    (left / right / accelerate / decelerate / keep) with lateral taking
    precedence — reproduced ONLY so its blindness is measurable: whenever a
    vehicle turns, ``mixed`` discards the longitudinal decision entirely.
    """
    g = geometry(wp, dt)
    h, v = g["heading"], g["valid"]
    n, H = h.shape
    lat = np.ones(n, dtype=np.int64)                 # default "straight"
    dpsi = np.zeros(n)
    for i in range(n):
        idx = np.flatnonzero(v[i])
        if idx.size >= 2:
            dpsi[i] = (h[i, idx[-1]] - h[i, idx[0]] + math.pi) % (2 * math.pi) - math.pi
    lat[dpsi > lat_thresh] = 2                       # left  (+y is left)
    lat[dpsi < -lat_thresh] = 0                      # right
    sp = g["speed"]
    dv = np.where(v[:, -1] & v[:, 0], sp[:, -1] - sp[:, 0], 0.0)
    lon = np.ones(n, dtype=np.int64)
    lon[dv > lon_thresh] = 2
    lon[dv < -lon_thresh] = 0
    mixed = np.where(lat != 1, np.where(lat == 2, 3, 4),        # left=3 right=4
                     np.where(lon == 2, 0, np.where(lon == 0, 1, 2)))
    return {"lateral": lat, "longitudinal": lon, "mixed": mixed,
            "dpsi_rad": dpsi, "dv_mps": dv}


def confusion(pred, gt, k: int) -> np.ndarray:
    """``[k, k]`` counts, rows = ground truth, cols = prediction."""
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    C = np.zeros((k, k), dtype=np.int64)
    np.add.at(C, (gt, pred), 1)
    return C


def balanced_accuracy(C: np.ndarray, require_all: bool = False) -> float:
    """Mean per-class recall over the classes that actually occur.

    Plain accuracy is unusable here: 'straight' and 'cruise' dominate, so a
    degenerate constant predictor scores ~0.9 and looks excellent. Balanced
    accuracy puts that predictor at 1/k, which is the point.

    ``require_all`` returns ``nan`` unless EVERY class has support. This matters
    inside a bootstrap: turn windows are under 1 % of a highway corpus, so many
    episode resamples contain no turn at all, and averaging over "present"
    classes then leaves a single class with recall 1.0 — i.e. a *blind constant*
    predictor's BA jumps to 1.0 and the upper CI bound becomes meaningless.
    MEASURED: without this the blind control's interval reads [0.3333, 1.0000].
    Bootstraps must pass ``require_all=True`` and report how many draws survived.
    """
    support = C.sum(1)
    present = support > 0
    if not present.any():
        return float("nan")
    if require_all and not present.all():
        return float("nan")
    return float(np.mean(np.diag(C)[present] / support[present]))


# --------------------------------------------------------------------------- #
# the four families                                                           #
# --------------------------------------------------------------------------- #
def longitudinal(pred_wp, gt_wp, dt=IDM_DT_S, *, pred_speed=None, gt_speed=None,
                 lead_available: bool = False) -> dict:
    """Is the IDM reading the RIGHT SPEED, and can it keep distance?

    ``pred_speed`` / ``gt_speed`` are the head's *scalar* speed channel and the
    CAN speed — the IDM's direct longitudinal read, which the trajectory-derived
    speed does not replace.
    """
    P, G = geometry(pred_wp, dt), geometry(gt_wp, dt)
    sp = P["speed"] - G["speed"]
    al = P["along"] - G["along"]
    ac = P["accel"] - G["accel"]
    out = {
        "traj_speed_mae_mps": round(float(np.abs(sp).mean()), 4),
        "traj_speed_bias_mps": round(float(sp.mean()), 4),
        "along_mae_m": round(float(np.abs(al).mean()), 4),
        "along_bias_m": round(float(al.mean()), 4),
        "along_final_bias_m": round(float(al[:, -1].mean()), 4),
        "accel_mae_mps2": round(float(np.abs(ac).mean()), 4),
        "n_windows": int(np.asarray(pred_wp).shape[0]),
    }
    if pred_speed is not None and gt_speed is not None:
        e = np.asarray(pred_speed, float) - np.asarray(gt_speed, float)
        out["scalar_speed_mae_mps"] = round(float(np.abs(e).mean()), 4)
        out["scalar_speed_bias_mps"] = round(float(e.mean()), 4)
    out["distance_keeping"] = {
        "status": "UNAVAILABLE",
        "reason": ("no lead-agent track on this substrate. comma2k19 ships no "
                   "object annotation at all; PhysicalAI-AV ships obstacle.offline "
                   "(97.44 % of the corpus) but the episode ingest does not read "
                   "it. Implementing it is a WORK ITEM, not a pass."),
        "n": 0,
    } if not lead_available else {"status": "AVAILABLE", "n": out["n_windows"]}
    return out


def lateral(pred_wp, gt_wp, dt=IDM_DT_S, *, pred_yaw_rate=None, gt_yaw_rate=None,
            pred_steer=None, gt_steer=None) -> dict:
    """Heading, CURVATURE and YAW-RATE error, not cross-track alone.

    "Lateral is fine" has been asserted from cross-track alone before; a path can
    be smooth and wrong — right at the waypoints, turning with the wrong
    curvature. ``n_pair_valid`` is reported because sparse IDM waypoints leave
    only ``H-1`` curvature steps per window.
    """
    P, G = geometry(pred_wp, dt), geometry(gt_wp, dt)
    both = P["valid"] & G["valid"]
    bothp = P["pair_valid"] & G["pair_valid"]
    dh = (P["heading"] - G["heading"] + math.pi) % (2 * math.pi) - math.pi
    he, n_h = _mean(np.abs(dh), both)
    ce, n_c = _mean(np.abs(P["curvature"] - G["curvature"]), bothp)
    ye, n_y = _mean(np.abs(P["yaw_rate"] - G["yaw_rate"]), bothp)
    cr = P["cross"] - G["cross"]
    out = {
        "heading_mae_rad": round(he, 5), "n_heading": n_h,
        "curvature_mae_inv_m": round(ce, 5), "n_curvature": n_c,
        "yaw_rate_mae_rad_s": round(ye, 5), "n_pair_valid": n_y,
        "cross_track_mae_m": round(float(np.abs(cr).mean()), 4),
        "cross_track_bias_m": round(float(cr.mean()), 4),
        "cross_track_final_mae_m": round(float(np.abs(cr[:, -1]).mean()), 4),
        "n_windows": int(np.asarray(pred_wp).shape[0]),
    }
    if pred_yaw_rate is not None and gt_yaw_rate is not None:
        e = np.asarray(pred_yaw_rate, float) - np.asarray(gt_yaw_rate, float)
        out["scalar_yaw_rate_mae_rad_s"] = round(float(np.abs(e).mean()), 5)
    if pred_steer is not None and gt_steer is not None:
        e = np.asarray(pred_steer, float) - np.asarray(gt_steer, float)
        out["scalar_steer_mae"] = round(float(np.abs(e).mean()), 5)
    return out


def tactical(pred_wp, gt_wp, dt=IDM_DT_S, **kw) -> dict:
    """Manoeuvre-decision quality: what the IDM would LABEL vs what was DONE.

    Reported per axis with a confusion matrix, per-class recall and balanced
    accuracy — never as one pooled score, because a pooled score hides exactly
    the lateral/longitudinal trade-off the program is trying to see.
    """
    Pm = manoeuvre_classes(pred_wp, dt, **kw)
    Gm = manoeuvre_classes(gt_wp, dt, **kw)
    out = {"n_windows": int(np.asarray(pred_wp).shape[0]),
           "thresholds": {"lat_turn_rad": kw.get("lat_thresh", LAT_TURN_RAD),
                          "lon_dv_mps": kw.get("lon_thresh", LON_DV_MPS)}}
    for axis, names in (("lateral", LATERAL_CLASSES),
                        ("longitudinal", LONGITUDINAL_CLASSES),
                        ("mixed", ("accelerate", "decelerate", "keep",
                                   "left", "right"))):
        k = len(names)
        C = confusion(Pm[axis], Gm[axis], k)
        sup = C.sum(1)
        out[axis] = {
            "classes": list(names),
            "confusion_gt_rows_pred_cols": C.tolist(),
            "support": sup.tolist(),
            "recall": [round(float(C[i, i] / sup[i]), 4) if sup[i] else None
                       for i in range(k)],
            "balanced_accuracy": round(balanced_accuracy(C), 4),
            "accuracy": round(float(np.trace(C) / max(C.sum(), 1)), 4),
            "chance_balanced_accuracy": round(1.0 / max(int((sup > 0).sum()), 1), 4),
        }
    out["goal_setting"] = {
        "status": "PARTIAL",
        "reason": ("selected-vs-executed manoeuvre IS reported above. Anchor/goal "
                   "selection is not: the IDM emits a single regressed trajectory, "
                   "it has no anchor set to select from."),
        "n": out["n_windows"],
    }
    return out


def strategic(n_windows: int, *, route_labels_available: bool = False) -> dict:
    """Route / goal-setting quality — UNAVAILABLE on every substrate the IDM has.

    Reported explicitly with the reason and n rather than dropped, per the rule.
    """
    if route_labels_available:
        raise NotImplementedError(
            "no route-labelled substrate has ever reached the IDM; wire the "
            "labels through before enabling this path")
    return {
        "status": "UNAVAILABLE",
        "reason": ("no route/goal label on the IDM's substrates. comma2k19 has "
                   "none; PhysicalAI-AV settled at five probes as carrying no map, "
                   "lane graph, junction annotation or route signal, and its "
                   "egomotion is clip-local metres with no GNSS, so map-matching "
                   "is impossible. A strategic read needs AlpaSim/NuRec map.xodr "
                   "or an external corpus — a WORK ITEM, not a pass."),
        "n": int(n_windows),
    }


def all_families(pred_wp, gt_wp, dt=IDM_DT_S, *, pred_scalars=None,
                 gt_scalars=None, **kw) -> dict:
    """All four families for one IDM arm. ``*_scalars`` are ``[n, 4]`` in
    ``idm_head.SCALAR_NAMES`` order (speed, yaw_rate, steer, long_accel)."""
    ps = None if pred_scalars is None else np.asarray(pred_scalars, float)
    gs = None if gt_scalars is None else np.asarray(gt_scalars, float)
    fam = {
        "LONGITUDINAL": longitudinal(
            pred_wp, gt_wp, dt,
            pred_speed=None if ps is None else ps[:, 0],
            gt_speed=None if gs is None else gs[:, 0]),
        "LATERAL": lateral(
            pred_wp, gt_wp, dt,
            pred_yaw_rate=None if ps is None else ps[:, 1],
            gt_yaw_rate=None if gs is None else gs[:, 1],
            pred_steer=None if ps is None else ps[:, 2],
            gt_steer=None if gs is None else gs[:, 2]),
        "TACTICAL": tactical(pred_wp, gt_wp, dt, **kw),
        "STRATEGIC": strategic(int(np.asarray(pred_wp).shape[0])),
    }
    fam["_contract"] = (
        "Four families reported per-family, never pooled, in ADDITION to ADE. "
        "A family marked UNAVAILABLE is a WORK ITEM, not a pass.")
    fam["_unavailable"] = [k for k, v in fam.items()
                           if isinstance(v, dict) and v.get("status") == "UNAVAILABLE"]
    return fam


def ade(pred_wp, gt_wp) -> float:
    """Mean L2 waypoint error (metres) — the one row, not the result."""
    p = np.asarray(pred_wp, float)
    g = np.asarray(gt_wp, float)
    return float(np.linalg.norm(p - g, axis=-1).mean())
