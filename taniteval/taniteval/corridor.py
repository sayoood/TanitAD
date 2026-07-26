"""TanitEval — CORRIDOR DEPARTURE: the horizon-capable, lateral primary.

WHAT THIS IS
------------
``corridor_departure_rate`` and its family, promoted from the E1a one-off into a
first-class library metric. The definition is **lifted from**
``TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
2026-07-25-closedloop-horizon-and-shift/e1a_horizon.py`` (``block()``, lines
302-337), not re-derived — so every E1a number stays reproducible and the two
cannot silently drift apart. :mod:`tests.test_corridor` pins that.

WHY IT HAD TO MOVE
------------------
The HPP-0 confound audit (§3.1/§3.2) found ``corridor_departure_rate`` in **zero
files** under ``taniteval/taniteval/`` — it existed only in five ``incoming/``
one-offs — while the review promotes it to **gate co-primary**, replacing the
horizon-blind ADE@2s (``01_EXECUTION_PLAN.md`` B.2 T1-1, *"the single
highest-leverage correction in the review"*).

The reason is one MEASURED table (E1a, ``e1a_horizon_heldout44_K185.json``,
paired common-start, 43 identical windows, ``episode_cluster_bootstrap``
B=2000):

===========  ====================  ====================  ==================
stratum      CDR@1.75 m, K=20      CDR@1.75 m, K=185     peak XTE 2s→18.5s
===========  ====================  ====================  ==================
overall      **0.0035**            **0.5877**            0.35 m → **38.94 m**
junction     0.0250 (n=6)          **0.8414** (n=6)      1.23 m → **46.25 m**
===========  ====================  ====================  ==================

A **168×** change in the failure rate between the standing 2 s horizon and the
event's own horizon. The standing instrument does not under-report this failure;
it cannot see it.

It is also the **lateral** axis, which is the axis that ends a drive.
``LATERAL_VS_LONGITUDINAL_ANALYSIS.md`` MEASURED that ADE is **98.6 %
longitudinal by squared-error energy**, and that lateral error compounds
**4.4-5.9× faster** than longitudinal (replicated on 2 clips). See
:mod:`taniteval.lateral` for the decomposition that makes that visible; this
module is the gate-facing consequence of it.

THE TWO SURFACES, AND WHY BOTH EXIST
------------------------------------
``corridor_block`` takes ``lat [N, K]`` — per-window, per-step **|cross-track
error|** — and is the exact E1a aggregation. Where that array comes from is the
caller's business:

* **Closed loop** (E1a's own use): at each rollout step, the lateral offset of
  the simulated ego from the nearest reference pose, in that pose's frame.
  ``e1a_horizon.rollout`` computes it during the loop; it needs a live model.
* **Open loop, from a persisted dump**: :func:`cross_track_from_paths` derives
  the same quantity from ``pred_dense`` / ``gt_dense [N, K, 2]``, which
  ``rollout.collect`` has persisted since 2026-07-25. No GPU, no re-run.

⚠️ **They are not the same number and must never be pooled.** The closed-loop
XTE accumulates control error; the open-loop XTE is a prediction residual
against the expert's own path. Every emitted block names its ``surface``.

HONEST LIMITS, STATED NOT HIDDEN
--------------------------------
* **The "corridor" is not a lane.** No lane geometry exists in this corpus
  (``driving.py`` refuses ``lane_centre_deviation`` for exactly this reason);
  the corridor is a **half-width about the reference path** and
  :data:`CORRIDOR_HALFWIDTH_M` = 1.75 m is a PROPOSED lane-relative constant,
  not a measured one. That is why the whole threshold GRID is always emitted —
  a verdict that survives only at one half-width is a knife-edge, not a result.
* **Horizon ceiling.** An episode yields a window at K only if ``T - W - K >=
  1``; PhysicalAI clips are 190-199 frames, so **K <= 190 (19.0 s) is a hard
  ceiling** and K=200 is structurally impossible on this corpus.
* **The junction stratum is a KINEMATIC SIGNATURE**, never a topology: E1a's
  ``|net heading change over the FIRST 2 s| >= 10 deg``. It must not be renamed
  "intersection" (``driving.py`` §6.3 refusal). It is held FIXED across horizons
  on purpose, so the strata stay comparable as K varies.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import ci as _ci  # noqa: E402
from taniteval import driving as _drv  # noqa: E402

BLOCK = "taniteval.corridor"
VERSION = "1.0.0"
SPEC = ("TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
        "2026-07-25-closedloop-horizon-and-shift/e1a_horizon.py::block "
        "(lines 302-337) — lifted, not re-derived")

DT = 0.1                      # MEASURED — 10 Hz, every trainer and eval path
EPS = 1e-9

# --------------------------------------------------------------------------- #
# Thresholds — E1a's, verbatim. Every one carries its evidence class.           #
# --------------------------------------------------------------------------- #
CORRIDOR_HALFWIDTH_M = 1.75   # PROPOSED — e1a_horizon.py --corridor-halfwidth;
#                               "about half a lane", NOT measured on this corpus
# ⭐ 1.391 ADDED 2026-07-26 (PI-approved). It is the ONLY MEASURED entry in this
# grid. The VectorMap instrument established that 1.75 is TWO DIFFERENT QUANTITIES
# and only one of them checks out:
#   · as HALF THE LANE WIDTH it is vindicated — measured 1.802 m [1.686, 1.939],
#     scene-cluster bootstrap over 51 AlpaSim scenes, and 1.75 sits INSIDE that CI;
#   · as a DEPARTURE THRESHOLD it is ~26% TOO PERMISSIVE — the real room to the
#     NEARER edge is 1.391 m [1.289, 1.500]. 85.7% of ego steps have less room than
#     1.75 m and 46 of 51 scenes are tighter.
# The two differ because the ego does NOT drive down the centreline, while taniteval
# measures XTE FROM THE REFERENCE PATH — so 1.391 is the ORIGIN-MATCHED threshold.
# ⚠️ 1.75 REMAINS THE ADJUDICATING VALUE and is unchanged: every published
# corridor_departure_rate (incl. E1a's 0.5877 / 0.8414) is scored at it. This entry
# is ADDITIVE — a second reported row, repricing nothing, reversible.
# ⚠️ EVIDENCE CLASS: MEASURED on ALPASIM, applied to PhysicalAI as a TRANSFER —
# PhysicalAI has no map (settled at five probes), so the CONSTANT transfers, not a
# per-timestep corridor. Label it as a transfer wherever it is quoted.
# ⚠️ `CORRIDOR_GRID_M` STAYS EXACTLY AS E1a SCORED IT. `test_corridor.py::
# test_docstring_headline_numbers_match_the_artifact` pins it to the COMMITTED E1a
# artifact's `corridor_thresholds_m`, and that guard is correct: widening this tuple
# would silently claim E1a had been scored on a grid it never saw. (I tried it; the
# test caught it. Adding alongside is the same principle applied to the void OOD
# guards — never rewrite what a historical run actually did.)
CORRIDOR_GRID_M = (1.0, 1.75, 2.5)   # PROPOSED — e1a_horizon.py --corridor-grid;
#                                      PINNED to the E1a artifact. Do NOT widen.
CORRIDOR_GRID_FORWARD_M = (1.0, 1.391, 1.75, 2.5)   # for NEW runs only
CORRIDOR_HALFWIDTH_MEASURED_M = 1.391       # [1.289, 1.500], scene-cluster bootstrap
CORRIDOR_LANE_HALFWIDTH_MEASURED_M = 1.802  # [1.686, 1.939] — vindicates 1.75 AS
#                                             HALF-LANE-WIDTH, not as a threshold
JUNCTION_DEG = 10.0           # PROPOSED — e1a_horizon.py --junction-deg (L434)
# The P1 low-OOD envelope within which the real-footage closed loop was
# MEASURED (lowood_flagship_ci.json). Beyond these the E1a mapping CLAMPS, so
# any block reporting steps outside them is EXTRAPOLATION, not measurement.
ENV_LAT_MAX = 3.0             # MEASURED envelope limit, metres (P1's FIRST sweep
                              # point of separated degradation — a real criterion)
# ⛔ NOT MEASURED — CORRECTED 2026-07-26. `ENV_YAW_MAX = 12.0` is the LAST ENTRY OF A
# COMMAND-LINE DEFAULT STRING, not a measured edge: `lowood_probe.py:228` and
# `lowood_ci.py:114` both read `--yaw-grid default="0,1,2,3,5,8,12"`. THE SWEEP
# STOPPED AT 12 BECAUSE THE STRING STOPPED AT 12. No criterion selecting a yaw edge
# exists anywhere in P1.
#
# ⚠️ AND THE TWO AXES WERE NEVER SET BY A COMMON CRITERION: P1's own report puts the
# yaw no-degradation edge at <= 2 deg, with the paired delta CI-separated from 3 deg
# onward — so 12.0 sits FOUR sweep points deep into separated degradation, while
# ENV_LAT_MAX sits at its FIRST. Evidence class is INHERITED, not MEASURED.
#
# MEASURED 2026-07-26 (881 windows / 40 clusters, episode-cluster bootstrap B=2000,
# on the edge itself): the USABLE yaw edge is 15.47 deg [12.14, 17.88] — the CI lower
# bound TOUCHES the shipped 12, i.e. 1.29x at the point estimate and NO widening at
# the lower bound. Information is fully destroyed at 26.41 deg [18.33, 29.63],
# corroborated MODEL-FREE by the FOV half-angle at 25.70 deg. At the shipped 12 deg
# the warp has already destroyed 34.7% of usable information and FABRICATED 26.4% of
# pixels.
#
# ⛔ WIDENING IT RESCUES NOTHING: even at yaw = INFINITY the LATERAL clause alone
# leaves 3.75% of K=20 windows outside (junction 18.13%), and MEASUREMENT requires
# ZERO. K=60 would need 39.25 deg (junction 60.03 deg) — 1.5x-2.3x past TOTAL
# destruction. => Closed-loop numbers are EXTRAPOLATIONS AT EVERY ADMISSIBLE HORIZON
# and must be labelled so permanently. See RETRACTION_LOG class C14.
ENV_YAW_MAX = 12.0            # ⛔ INHERITED (grid terminus) — see above
ENV_YAW_MAX_EVIDENCE = "INHERITED: last entry of --yaw-grid default, not a criterion"
ENV_YAW_USABLE_EDGE_MEASURED = 15.47      # [12.14, 17.88], episode-cluster bootstrap
ENV_YAW_DESTROYED_MEASURED = 26.41        # [18.33, 29.63]; FOV half-angle 25.70

N_BOOT = _ci.DEFAULT_N_BOOT   # 2000
DECISION_ESTIMATORS = _drv.DECISION_ESTIMATORS
DEPRECATED_ESTIMATOR = _drv.DEPRECATED_ESTIMATOR
ESTIMATOR_NOTE = _drv.ESTIMATOR_NOTE

SURFACES = ("closed_loop", "open_loop_dense")


# ========================================================================== #
# per-window components — the E1a definitions, one place                       #
# ========================================================================== #
def corridor_departure(lat_abs, threshold=CORRIDOR_HALFWIDTH_M):
    """**THE gate co-primary.** Per-window FRACTION OF STEPS outside the corridor.

    ``lat_abs`` [N, K] |cross-track error| per step -> ``[N]`` in [0, 1].
    Verbatim ``e1a_horizon.py:313`` — ``(lat > primary).mean(1)``.

    A *rate over the horizon*, not a binary: a window that leaves the corridor
    for 2 of 185 steps and recovers is not the same failure as one that leaves
    at step 30 and never returns, and ADE cannot tell them apart either."""
    return (np.asarray(lat_abs, dtype=np.float64) > float(threshold)).mean(1)


def window_departure(lat_abs, threshold=CORRIDOR_HALFWIDTH_M):
    """Per-window 0/1: did this window leave the corridor at ANY step?

    Verbatim ``e1a_horizon.py:316`` — ``(lat > primary).any(1)``. Strictly >=
    :func:`corridor_departure` window-for-window, which is a test invariant."""
    return (np.asarray(lat_abs, dtype=np.float64) > float(threshold)
            ).any(1).astype(np.float64)


def peak_xte(lat_abs):
    """Per-window max |cross-track error| (m) — ``e1a_horizon.py:319``.

    The tail statistic. ``LATERAL_VS_LONGITUDINAL_ANALYSIS.md`` §2: the mean is
    the *least* informative statistic on a safety axis (0.25 m mean vs 1.40 m
    p90 on the same windows)."""
    return np.asarray(lat_abs, dtype=np.float64).max(1)


def mean_xte(lat_abs):
    """Per-window mean |cross-track error| (m) — ``e1a_horizon.py:320``."""
    return np.asarray(lat_abs, dtype=np.float64).mean(1)


def junction_mask(head_deg_2s, junction_deg=JUNCTION_DEG):
    """E1a's junction stratum: ``|net heading change over the FIRST 2 s| >= 10 deg``.

    ``e1a_horizon.py:433-434`` — computed from ``hd2s``, the 2 s net heading
    change, and held FIXED across horizons so the strata stay comparable as K
    varies. ``rollout.collect`` persists exactly this as ``head_deg``.

    ⚠️ A **kinematic signature**, never a topology. There is no map, no lane
    graph and no junction annotation in this corpus; renaming this
    "intersection" is the specific error ``driving.py`` §6.3 refuses. It is a
    *starting point* — the v2.1/v3 curvature-relative route label is the better
    one once an arm has trained on it (HPP-0 §PC3 fix 2)."""
    return np.abs(np.asarray(head_deg_2s, dtype=np.float64)) >= float(junction_deg)


def strata(head_deg_2s, speed, junction_deg=JUNCTION_DEG):
    """E1a's three-way split (``e1a_horizon.py:433-441``) + ``overall``.

    junction = the heading signature; longitudinal = not junction AND speed >=
    the MEDIAN of this window set; other = the remainder. The median is
    recomputed per window set exactly as E1a does, so a stratum is always ~half
    the non-junction windows regardless of the arm's speed distribution."""
    hd = np.asarray(head_deg_2s, dtype=np.float64)
    spd = np.asarray(speed, dtype=np.float64)
    junc = junction_mask(hd, junction_deg)
    long_ = (~junc) & (spd >= np.median(spd))
    return {"overall": np.ones(len(hd), dtype=bool), "junction": junc,
            "longitudinal": long_, "other": (~junc) & (~long_)}


# ========================================================================== #
# open-loop cross-track from the persisted dense path                          #
# ========================================================================== #
def cross_track_from_paths(pred_dense, gt_dense):
    """``pred/gt [N, K, 2]`` ego-frame paths -> ``|XTE| [N, K]``.

    The E1a construction at the open-loop surface. E1a computes, per rollout
    step, the offset of the ego from its NEAREST reference pose expressed in
    that pose's frame (``e1a_horizon.py:244-250``)::

        m* = argmin_m ||ego - P_m||
        dlat = -sin(yaw_ref) * dx + cos(yaw_ref) * dy

    Reproduced here with the reference polyline = ``gt_dense`` (origin
    prepended, because the dense path starts one tick AFTER the ego pose, which
    is the origin — the same convention ``rollout.dense_speed_profile``
    documents), and ``yaw_ref`` taken from the reference polyline's local
    tangent.

    ⚠️ **The one documented difference from E1a.** E1a reads ``yaw_ref`` from
    the stored pose ``P_yaw[m*]``; the ego-frame dense dump does not carry
    per-step yaw, so the tangent is used. The two agree wherever the reference
    path is locally smooth and differ where consecutive GT poses are nearly
    coincident (a stopped ego) — which is why the tangent is carried forward
    across degenerate segments rather than zeroed. This is why an open-loop
    block is stamped ``surface="open_loop_dense"`` and must never be pooled
    with a closed-loop one.

    Signed convention matches ``driving.frenet``: **+ = pred is LEFT of the
    reference**. The absolute value is what the corridor metrics consume."""
    p = torch.as_tensor(pred_dense, dtype=torch.float32)
    g = torch.as_tensor(gt_dense, dtype=torch.float32)
    if p.shape != g.shape or p.ndim != 3 or p.shape[-1] != 2:
        raise ValueError(f"need matching [N,K,2] paths, got {tuple(p.shape)} "
                         f"and {tuple(g.shape)}")
    n = p.shape[0]
    ref = torch.cat([torch.zeros(n, 1, 2, dtype=g.dtype), g], dim=1)  # [N,K+1,2]
    # reference tangents, with the last valid direction carried forward so a
    # stopped segment does not produce a zero (and therefore arbitrary) normal
    d = ref[:, 1:] - ref[:, :-1]                                      # [N,K,2]
    nrm = d.norm(dim=-1, keepdim=True)
    t = torch.where(nrm > EPS, d / nrm.clamp_min(EPS), torch.zeros_like(d))
    fwd = torch.tensor([1.0, 0.0])
    for i in range(t.shape[1]):
        bad = t[:, i].norm(dim=-1) <= EPS
        if bad.any():
            t[bad, i] = t[bad, i - 1] if i > 0 else fwd
    tang = torch.cat([t[:, :1], t], dim=1)               # tangent AT each ref pt
    nvec = torch.stack([-tang[..., 1], tang[..., 0]], dim=-1)         # left normal
    # nearest reference point per predicted point
    dist = torch.cdist(p, ref)                                        # [N,K,K+1]
    mstar = dist.argmin(dim=-1)                                       # [N,K]
    idx = mstar.unsqueeze(-1).expand(-1, -1, 2)
    pref = torch.gather(ref, 1, idx)
    nref = torch.gather(nvec, 1, idx)
    return ((p - pref) * nref).sum(-1).abs().numpy()


# ========================================================================== #
# the block — E1a's `block()`, lifted                                          #
# ========================================================================== #
def _boot(x, eid, n_boot=N_BOOT, seed=0):
    return _ci.episode_cluster_bootstrap(np.asarray(x, dtype=np.float64), eid,
                                         n_boot=n_boot, seed=seed)


def corridor_block(lat_abs, eid, thresholds=CORRIDOR_GRID_M,
                   primary=CORRIDOR_HALFWIDTH_M, yaw_abs_deg=None,
                   ade2s=None, n_boot=N_BOOT, seed=0, surface="open_loop_dense",
                   min_windows=2):
    """One stratum's corridor block. ``e1a_horizon.block`` with the same keys.

    ``lat_abs`` [n, K] |XTE| per step, ``eid`` [n] episode ids. Optional
    ``yaw_abs_deg`` [n, K] and ``ade2s`` [n] add E1a's ``peak_dpsi_deg`` /
    ``closed_ade2s_m`` rows when the caller has them.

    Returns ``None`` for a stratum too small to bootstrap — E1a's own
    ``len(m) < 2`` refusal — rather than a NaN interval that reads like a pass.
    The OOD-ratio rows are deliberately absent: they need the external P1
    envelope JSON, which is not a library dependency. ``EXTRAPOLATION_*`` (which
    needs only the envelope CONSTANTS) is kept."""
    lat = np.asarray(lat_abs, dtype=np.float64)
    if lat.ndim != 2:
        raise ValueError(f"lat_abs must be [n_windows, K], got {lat.shape}")
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")
    e = [str(x) for x in eid]
    if len(e) != lat.shape[0]:
        raise ValueError(f"eid/lat length mismatch: {len(e)} vs {lat.shape[0]}")
    if lat.shape[0] < min_windows or len(set(e)) < 2:
        return None
    K = int(lat.shape[1])
    thresholds = tuple(float(t) for t in thresholds)
    if float(primary) not in thresholds:
        raise ValueError(f"primary halfwidth {primary} must be in the emitted "
                         f"grid {thresholds} — a single-threshold verdict is a "
                         f"knife-edge, not a result")
    out = {
        "block": BLOCK, "version": VERSION, "spec": SPEC,
        "surface": surface,
        "n_windows": int(lat.shape[0]), "n_episodes": int(len(set(e))),
        "horizon_K": K, "horizon_s": round(K * DT, 2),
        "corridor_primary_m": float(primary),
        "corridor_thresholds_m": list(thresholds),
        "corridor_departure_rate": _boot(
            corridor_departure(lat, primary), e, n_boot, seed),
        "corridor_departure_rate_by_threshold_m": {
            f"{t:g}": _boot(corridor_departure(lat, t), e, n_boot, seed)
            for t in thresholds},
        "window_departure_rate": _boot(
            window_departure(lat, primary), e, n_boot, seed),
        "window_departure_rate_by_threshold_m": {
            f"{t:g}": _boot(window_departure(lat, t), e, n_boot, seed)
            for t in thresholds},
        "peak_xte_m": _boot(peak_xte(lat), e, n_boot, seed),
        "mean_xte_m": _boot(mean_xte(lat), e, n_boot, seed),
        "mean_abs_xte_by_step_m": [round(float(x), 4) for x in lat.mean(0)],
        "EXTRAPOLATION_frac_steps_lat_over_3m": round(
            float((lat > ENV_LAT_MAX).mean()), 5),
        "estimator": {
            "interval": "episode_cluster_bootstrap",
            "n_boot": int(n_boot), "seed": int(seed),
            "resampling_unit": "val episode",
            "deprecated_and_refused": DEPRECATED_ESTIMATOR,
            "estimator_note": ESTIMATOR_NOTE},
    }
    if yaw_abs_deg is not None:
        yaw = np.asarray(yaw_abs_deg, dtype=np.float64)
        out["peak_dpsi_deg"] = _boot(yaw.max(1), e, n_boot, seed)
        out["EXTRAPOLATION_frac_steps_yaw_over_12deg"] = round(
            float((yaw > ENV_YAW_MAX).mean()), 5)
        out["EXTRAPOLATION_frac_windows_any_step_out_of_envelope"] = round(
            float(((lat > ENV_LAT_MAX) | (yaw > ENV_YAW_MAX)).any(1).mean()), 4)
    if ade2s is not None:
        out["closed_ade2s_m" if surface == "closed_loop" else "ade2s_m"] = _boot(
            np.asarray(ade2s, dtype=np.float64), e, n_boot, seed)
    return out


def stratified(lat_abs, eid, head_deg_2s, speed, thresholds=CORRIDOR_GRID_M,
               primary=CORRIDOR_HALFWIDTH_M, junction_deg=JUNCTION_DEG,
               yaw_abs_deg=None, ade2s=None, n_boot=N_BOOT, seed=0,
               surface="open_loop_dense"):
    """Every E1a stratum at one horizon: overall / junction / longitudinal / other.

    This is the shape ``e1a_horizon.main`` writes under ``all_windows[K]``, and
    it is what HP-2 (*"the advantage concentrates at decision points"*) reads:
    the junction stratum against the straight-cruise stratum, paired."""
    lat = np.asarray(lat_abs, dtype=np.float64)
    st = strata(head_deg_2s, speed, junction_deg)
    out = {"_stratification": (
        "junction = |net heading change over the FIRST 2 s| >= "
        f"{junction_deg:g} deg (E1a's standing definition, held FIXED across "
        "horizons so the strata stay comparable); longitudinal = not junction "
        "AND speed >= median. A KINEMATIC SIGNATURE, never a topology."),
        "junction_deg": float(junction_deg),
        "n_by_stratum": {k: int(m.sum()) for k, m in st.items()}}
    for name, m in st.items():
        idx = np.flatnonzero(m)
        out[name] = corridor_block(
            lat[idx], [str(eid[i]) for i in idx], thresholds, primary,
            yaw_abs_deg=(None if yaw_abs_deg is None
                         else np.asarray(yaw_abs_deg)[idx]),
            ade2s=(None if ade2s is None else np.asarray(ade2s)[idx]),
            n_boot=n_boot, seed=seed, surface=surface)
    return out


def paired_stratum_delta(lat_a, lat_b, eid, threshold=CORRIDOR_HALFWIDTH_M,
                         n_boot=N_BOOT, seed=0):
    """PAIRED Δ corridor-departure between two arms on the SAME windows.

    The admissible form for every HP-1…HP-6 comparison: the arms share windows,
    so a quadrature combination of two single-arm intervals is not merely weaker
    — it is invalid (the estimates are not independent). Oriented **b − a**, so
    a POSITIVE delta means ``a`` departs less, i.e. ``a`` wins."""
    a = corridor_departure(lat_a, threshold)
    b = corridor_departure(lat_b, threshold)
    d = _ci.paired_episode_cluster_bootstrap(b, a, [str(x) for x in eid],
                                             n_boot=n_boot, seed=seed)
    d["_orientation"] = ("b - a on corridor_departure_rate; POSITIVE = arm `a` "
                         "departs the corridor less often = `a` wins")
    d["threshold_m"] = float(threshold)
    return d


# ========================================================================== #
# entry point from a persisted window dump                                     #
# ========================================================================== #
def from_windows(win, thresholds=CORRIDOR_GRID_M, primary=CORRIDOR_HALFWIDTH_M,
                 junction_deg=JUNCTION_DEG, n_boot=N_BOOT, seed=0):
    """``rollout.collect`` / ``load_windows`` dict -> the stratified block.

    Requires the DENSE keys (``pred_dense`` / ``gt_dense``), persisted since
    2026-07-25. Dumps written before that — and every ``refb_eval`` /
    ``refc_eval`` dump, which never emit them — return a self-describing
    ``skipped`` node rather than a fabricated 4-waypoint approximation: at 0.5 s
    knot spacing the corridor is unmeasurable and a number computed there would
    be a different metric wearing this one's name."""
    pd_, gd = win.get("pred_dense"), win.get("gt_dense")
    if pd_ is None or gd is None:
        return {"block": BLOCK, "version": VERSION,
                "skipped": ("no dense path in this dump (pred_dense/gt_dense). "
                            "Pre-2026-07-25 dumps and refb_eval/refc_eval do "
                            "not persist it; re-run rollout.collect."),
                "dense_surface_available": False}
    lat = cross_track_from_paths(pd_, gd)
    out = stratified(lat, list(win["eid"]),
                     np.asarray(win["head_deg"], dtype=np.float64),
                     np.asarray(win["speed"], dtype=np.float64),
                     thresholds=thresholds, primary=primary,
                     junction_deg=junction_deg, n_boot=n_boot, seed=seed,
                     surface="open_loop_dense")
    out["block"] = BLOCK
    out["version"] = VERSION
    out["dense_surface_available"] = True
    out["dt_s"] = win.get("dt_s", DT)
    out["_surface_warning"] = (
        "OPEN-LOOP corridor departure (prediction residual against the "
        "expert's own path). E1a's headline numbers are CLOSED-LOOP (control "
        "error accumulates). The two are different quantities and must never "
        "be pooled or compared as if they were one series.")
    _drv.assert_no_deprecated_estimator(out, _path=BLOCK)
    return out


def horizon_ceiling(episode_T, window=8):
    """Largest K for which an episode of ``T`` frames yields any window.

    ``starts = range(0, T - W - K)`` so a window exists only if ``T - W - K >=
    1``. E1a records this as the reason **K=200 (20 s) is structurally
    impossible** on PhysicalAI (clips are 190-199 frames): the ceiling is
    ``T - W - 1``, i.e. at most 190 steps = 19.0 s."""
    return max(0, int(episode_T) - int(window) - 1)


def horizon_seconds(K, dt=DT):
    return round(float(K) * float(dt), 2)


def _headline(o):
    return (f"CDR@{o['corridor_primary_m']:g}m="
            f"{o['corridor_departure_rate']['mean']:.4f} "
            f"[{o['corridor_departure_rate']['lo']:.4f},"
            f"{o['corridor_departure_rate']['hi']:.4f}] · "
            f"winDEP={o['window_departure_rate']['mean']:.4f} · "
            f"peakXTE={o['peak_xte_m']['mean']:.3f} m · "
            f"K={o['horizon_K']} ({o['horizon_s']} s) · "
            f"n={o['n_windows']}/{o['n_episodes']} eps")


def summarise(block):
    """One line per stratum — what a gate row prints."""
    lines = []
    for name in ("overall", "junction", "longitudinal", "other"):
        o = block.get(name)
        lines.append(f"{name:14s} " + (_headline(o) if o else "n/a (too small)"))
    return "\n".join(lines)


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser("taniteval.corridor")
    ap.add_argument("--windows", required=True,
                    help="path to a results/windows_<arm>.pt dump")
    ap.add_argument("--halfwidth", type=float, default=CORRIDOR_HALFWIDTH_M)
    ap.add_argument("--junction-deg", type=float, default=JUNCTION_DEG)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from taniteval import rollout
    win = rollout.load_windows(a.windows)
    res = from_windows(win, primary=a.halfwidth, junction_deg=a.junction_deg,
                       n_boot=a.n_boot)
    if res.get("skipped"):
        print(f"[corridor] SKIPPED: {res['skipped']}")
        return
    print(summarise(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"[corridor] wrote {a.out}")


if __name__ == "__main__":
    main()
