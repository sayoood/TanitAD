"""TanitEval — M1: NO trajectory error is reported without its (lat, lon) split.

THE FINDING THAT FORCES THIS MODULE (MEASURED, not argued)
----------------------------------------------------------
``TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
2026-07-25-idm-youtube-validation/LATERAL_VS_LONGITUDINAL_ANALYSIS.md``:

* **ADE is 98.6 % LONGITUDINAL by squared-error energy.** The lateral axis
  receives **~1.4 %** of the signal. Any loss or metric built on undecomposed L2
  is numerically a *longitudinal* loss. Lateral deviation is not
  under-weighted — it is **nearly invisible**.
* **Lateral error COMPOUNDS; longitudinal does not.** Over 0.5 s → 2 s,
  longitudinal grows x3.20 while lateral grows **x14.11** — and on the second
  clip x4.43 vs **x26.14**. Lateral grows **4.4-5.9x faster**, replicated on two
  independent clips. Longitudinal error is a *bounded scale error* (~17-23 % of
  distance travelled, stable-to-improving with horizon); lateral shows no sign
  of saturating.
* **The mean is the least informative statistic on this axis.** On the same
  windows: mean |XTE| 0.25 m, **p90 1.40 m** — a lane departure at one window in
  ten, while the ADE headline reports a *speed* error. On the second clip the
  p50 alone is **3.43 m** and 73 % of windows exceed 1 m, yet longitudinal still
  carries 84.6 % of the squared error: *the aggregate ADE would still read as a
  longitudinal problem while the vehicle is metres out of its lane.*

Independently corroborated at the planner: E1a's 18.5 s failure is a **lateral**
corridor departure (59 % overall / 84 % at junctions, peak XTE 38.94 m). Two
instruments, same conclusion —

    Longitudinal error is what the 2 s metric shows you.
    Lateral error is what actually ends the drive.

WHY THIS IS HIERARCHY-PROOF WORK, NOT A SIDE QUEST
--------------------------------------------------
Route-following, junction handling and lane-keeping **are lateral phenomena**.
HP-2 (*the advantage concentrates at decision points*) and HP-3 (*same scene,
different ``nav_cmd`` ⇒ different trajectory*) are measured in the **cross-track
channel**. A strategic level that chooses a route can only demonstrate its value
on the axis that expresses route choice, so fixing lateral instrumentation is a
**prerequisite** for the proof, not a companion to it (analysis §M6).

THE AXIS CONVENTION — VERIFIED, NOT ASSUMED
-------------------------------------------
Ego frame: **axis0 = along-track (longitudinal), axis1 = cross-track (lateral)**.
Verified in the source analysis before any claim: axis0 mean |displacement| is
5.22 / 10.65 / 16.28 / **22.10 m** at 0.5 / 1 / 1.5 / 2 s against a mean speed of
10.15 m/s (≈ 20.3 m expected at 2 s). :func:`assert_axis_convention` re-checks it
on any dump so a transposed tensor cannot silently invert every number here.

TWO DECOMPOSITIONS, BOTH EMITTED, AND THEY ARE NOT THE SAME
-----------------------------------------------------------
* ``mode="ego"`` — the residual's axis0/axis1 components in the **ego frame of
  the window's last observed pose**. This is what the MEASURED finding above
  used, and it is the right frame for "did the car end up left of where it
  should be".
* ``mode="frenet"`` — the residual projected on the **GT path's local
  tangent/normal** (``driving.frenet`` generalised from 4 knots to K). This is
  the right frame for "how far off the intended path", and it is what the
  corridor metric consumes.

They coincide only where the GT path is straight and aligned with the ego x
axis; **in a turn they differ, which is exactly where lateral matters**. Every
emitted block names its mode. Pooling the two would be the same class of error
as pooling open- and closed-loop corridor departures.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
# ⛔ was: sys.path.insert(0, "/root/TanitAD/stack"[/scripts]) — that put a
# possibly PRE-v5 tree IN FRONT of the caller's PYTHONPATH and published a
# plausible wrong number instead of an error (STALE_IMPORT_GUARD.md).
from taniteval.stack_guard import ensure_stack_on_path as _ensure_stack  # noqa: E402
_ensure_stack()

from taniteval import ci as _ci  # noqa: E402
from taniteval import driving as _drv  # noqa: E402

BLOCK = "taniteval.lateral/decomposition"
VERSION = "1.0.0"
SPEC = ("TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
        "2026-07-25-idm-youtube-validation/LATERAL_VS_LONGITUDINAL_ANALYSIS.md"
        " §M1/§M2 (2026-07-25)")

DT = 0.1                      # MEASURED — 10 Hz dense path
# The two surfaces this module is handed, and why the distinction is load-bearing:
# the DENSE path is 20 consecutive 0.1 s steps (2.0 s); the SPARSE path is the 4
# gate waypoints, which sit at dense steps [5,10,15,20] — so ONE SPARSE KNOT IS
# 0.5 s, NOT 0.1 s. Treating a knot as a timestep under-reports the horizon by 5x.
# (`rollout.collect` asserts `pred == pred_dense[:, [4,9,14,19]]`.)
_DENSE_K = 20                 # knots on the dense surface
_SPARSE_K = 4                 # knots on the 4-waypoint gate surface
EPS = 1e-9
N_BOOT = _ci.DEFAULT_N_BOOT

ALONG_AXIS, CROSS_AXIS = 0, 1     # VERIFIED — see the module docstring
MODES = ("ego", "frenet")

# Tail cut-points. The MEAN is deliberately NOT the headline: §M2 —
# "gate on p90/p95/max XTE and the fraction beyond a lane-relative threshold,
#  not the mean. The mean is 0.25 m while p90 is 1.40 m."
TAIL_Q = (0.50, 0.75, 0.90, 0.95, 0.99)
# PROPOSED, never MEASURED on this corpus: no lane geometry exists here
# (``driving.py`` refuses ``lane_centre_deviation`` for that reason and calls
# ``LANE_HALF_M`` an assumed constant). 1.75 m is the E1a corridor half-width
# and is roughly half a lane; the whole GRID is always emitted so no verdict
# rests on one cut-point.
XTE_THRESHOLDS_M = (0.5, 1.0, 1.75)
LANE_HALF_M = 1.75            # PROPOSED — assumed, not measured

DECISION_ESTIMATORS = _drv.DECISION_ESTIMATORS
DEPRECATED_ESTIMATOR = _drv.DEPRECATED_ESTIMATOR
ESTIMATOR_NOTE = _drv.ESTIMATOR_NOTE


# ========================================================================== #
# the decomposition                                                            #
# ========================================================================== #
def _as3(p, name):
    t = torch.as_tensor(p, dtype=torch.float32)
    if t.ndim != 3 or t.shape[-1] != 2:
        raise ValueError(f"{name} must be [N, K, 2], got {tuple(t.shape)}")
    return t


def frenet_dense(pred, gt):
    """``driving.frenet`` generalised from 4 knots to K, on ``[N,K,2]`` paths.

    ``along + = pred is AHEAD of GT along the path``; ``cross + = pred is
    LEFT``. Orthonormal basis, so ``along² + cross² == ||pred-gt||²`` exactly —
    which :func:`energy_share` depends on and the tests pin. The origin (the
    window's last observed pose) is prepended before differencing, the same
    convention as ``rollout.dense_speed_profile``: the dense path starts one
    tick AFTER the ego pose, so the first segment is ``p[:,0]`` itself."""
    p, g = _as3(pred, "pred"), _as3(gt, "gt")
    n = g.shape[0]
    full = torch.cat([torch.zeros(n, 1, 2, dtype=g.dtype), g], dim=1)
    d = full[:, 1:] - full[:, :-1]
    nrm = d.norm(dim=-1, keepdim=True)
    t = torch.where(nrm > EPS, d / nrm.clamp_min(EPS), torch.zeros_like(d))
    fwd = torch.tensor([1.0, 0.0])
    for i in range(t.shape[1]):                  # carry last valid forward
        bad = t[:, i].norm(dim=-1) <= EPS
        if bad.any():
            t[bad, i] = t[bad, i - 1] if i > 0 else fwd
    nv = torch.stack([-t[..., 1], t[..., 0]], dim=-1)
    r = p - g
    return (r * t).sum(-1), (r * nv).sum(-1)


def decompose(pred, gt, mode="ego"):
    """``(along [N,K], cross [N,K])`` SIGNED residual components, in metres.

    ``mode="ego"`` uses the ego axes directly (axis0 along, axis1 cross) — the
    frame the MEASURED 98.6 %/x14.1 finding was computed in. ``mode="frenet"``
    projects on the GT path's local tangent. See the module docstring for why
    both exist and why they must never be pooled."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "frenet":
        return frenet_dense(pred, gt)
    p, g = _as3(pred, "pred"), _as3(gt, "gt")
    r = p - g
    return r[..., ALONG_AXIS], r[..., CROSS_AXIS]


def assert_axis_convention(gt_dense, speed=None, dt=DT, tol=0.35):
    """Fail loud if ``gt_dense`` is not (along, cross) ordered.

    The convention is *verified, not assumed* — a transposed or (y, x) tensor
    would silently invert every number in this module, turning the program's
    least-served axis into its headline. The check is the one the source
    analysis used: mean |axis0 displacement| at the final step must be
    consistent with the realised speed (``~v * K * dt``), and must dominate
    mean |axis1|. ``tol`` is the relative slack on the speed consistency.

    Returns the evidence dict; raises ``ValueError`` on a violation."""
    g = _as3(gt_dense, "gt_dense")
    K = g.shape[1]
    a_end = float(g[:, -1, ALONG_AXIS].abs().mean())
    c_end = float(g[:, -1, CROSS_AXIS].abs().mean())
    ev = {"mean_abs_along_final_m": round(a_end, 4),
          "mean_abs_cross_final_m": round(c_end, 4),
          "K": int(K), "horizon_s": round(K * dt, 2)}
    if a_end <= c_end:
        raise ValueError(
            f"axis convention violated: mean |axis0| {a_end:.3f} m does not "
            f"dominate mean |axis1| {c_end:.3f} m at {K * dt:.1f} s. axis0 must "
            f"be ALONG-track. Check for a transposed / (y,x) dump.")
    if speed is not None:
        v = float(np.mean(np.asarray(speed, dtype=np.float64)))
        expect = v * K * dt
        ev.update(mean_speed_mps=round(v, 4),
                  expected_along_m=round(expect, 4),
                  rel_err=round(abs(a_end - expect) / max(expect, EPS), 4))
        if expect > 1.0 and ev["rel_err"] > tol:
            raise ValueError(
                f"axis convention suspect: mean |axis0| {a_end:.3f} m vs "
                f"{expect:.3f} m expected from mean speed {v:.3f} m/s over "
                f"{K * dt:.1f} s (rel err {ev['rel_err']:.3f} > {tol}).")
    ev["verified"] = True
    return ev


# ========================================================================== #
# tail statistics — the point of §M2                                           #
# ========================================================================== #
def tail_stats(vals, thresholds=XTE_THRESHOLDS_M, quantiles=TAIL_Q):
    """p50/p75/p90/p95/p99/max + the fraction beyond each threshold.

    ``vals`` are per-window magnitudes (|cross-track| at a horizon). The mean is
    included ONLY so the gap to p90 is visible in the same row — it is not the
    statistic to gate on (§M2). ``frac_beyond`` is the lane-relative rate; every
    threshold in the grid is emitted so no verdict rests on a single cut-point."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size), "mean": round(float(v.mean()), 4),
           "max": round(float(v.max()), 4)}
    for q in quantiles:
        out[f"p{int(round(q * 100))}"] = round(float(np.quantile(v, q)), 4)
    out["frac_beyond_m"] = {f"{t:g}": round(float((v > t).mean()), 4)
                            for t in thresholds}
    out["mean_to_p90_ratio"] = (
        round(float(out["p90"] / out["mean"]), 3) if out["mean"] > EPS else None)
    out["_read"] = ("gate on p90/p95/max and frac_beyond, NOT the mean "
                    "(MEASURED: mean 0.25 m vs p90 1.40 m on the same windows)")
    return out


def energy_share(along, cross):
    """Longitudinal share of SQUARED error — the 98.6 % finding, recomputed.

    Returns the overall share and the per-step curve. ``along² + cross²`` is
    the squared L2 residual exactly (orthonormal basis), so the two shares sum
    to 1 by construction and the per-step curve is directly comparable to the
    analysis' table."""
    a = np.asarray(along, dtype=np.float64) ** 2
    c = np.asarray(cross, dtype=np.float64) ** 2
    tot = a.sum() + c.sum()
    per_step = (a.sum(0) / np.maximum(a.sum(0) + c.sum(0), EPS))
    return {
        "longitudinal_share_of_squared_error": round(
            float(a.sum() / max(tot, EPS)), 4),
        "lateral_share_of_squared_error": round(
            float(c.sum() / max(tot, EPS)), 4),
        "longitudinal_share_by_step": [round(float(x), 4) for x in per_step],
        "_read": ("MEASURED reference 0.986 longitudinal on the IDM "
                  "reconstruction: the lateral axis receives ~1.4 % of the "
                  "squared-error signal, so an undecomposed L2 objective is "
                  "numerically a longitudinal objective"),
    }


def growth(along, cross, ref_step=None, dt=DT):
    """The COMPOUNDING law: growth of mean |error| per axis vs a reference step.

    ``ref_step`` (1-based) defaults to the 0.5 s step, matching the analysis'
    table. Emits each axis' growth factor at every step and the **ratio of
    ratios** — lateral growth ÷ longitudinal growth, which is the replicated
    claim (x4.4 on one clip, x5.9 on the other)."""
    a = np.abs(np.asarray(along, dtype=np.float64)).mean(0)
    c = np.abs(np.asarray(cross, dtype=np.float64)).mean(0)
    K = a.size
    if ref_step is None:
        ref_step = min(K, max(1, int(round(0.5 / dt))))
    i = int(ref_step) - 1
    if not 0 <= i < K:
        raise ValueError(f"ref_step {ref_step} outside 1..{K}")
    ga = a / max(a[i], EPS)
    gc = c / max(c[i], EPS)
    return {
        "ref_step": int(ref_step), "ref_s": round(ref_step * dt, 2),
        "mean_abs_along_by_step_m": [round(float(x), 4) for x in a],
        "mean_abs_cross_by_step_m": [round(float(x), 4) for x in c],
        "along_growth_by_step": [round(float(x), 4) for x in ga],
        "cross_growth_by_step": [round(float(x), 4) for x in gc],
        "along_growth_final": round(float(ga[-1]), 4),
        "cross_growth_final": round(float(gc[-1]), 4),
        "cross_grows_faster_by": (round(float(gc[-1] / ga[-1]), 3)
                                  if ga[-1] > EPS else None),
        "_read": ("MEASURED reference: lateral grows 4.4-5.9x faster than "
                  "longitudinal over 0.5->2 s, replicated on 2 clips. "
                  "Longitudinal is a BOUNDED scale error; lateral compounds, "
                  "which is why the 18.5 s failure is a lateral corridor "
                  "departure and not a speed error."),
    }


# ========================================================================== #
# per-window components — what every trajectory metric must now also report    #
# ========================================================================== #
def per_window(pred, gt, mode="ego", steps=None):
    """``metric -> per-window values [N]``, at EVERY requested step.

    This is the M1 contract in code: for each horizon step ``k`` the L2 error is
    accompanied by its ``along`` / ``cross`` halves and their squares, so no ADE
    can be emitted without its split. ``steps`` are 1-based dense indices;
    default is every step, plus the final one always."""
    p, g = _as3(pred, "pred"), _as3(gt, "gt")
    al, cr = decompose(p, g, mode)
    de = torch.linalg.norm(p - g, dim=-1)
    K = p.shape[1]
    steps = tuple(range(1, K + 1)) if steps is None else tuple(steps)
    out = {}
    for k in steps:
        if not 1 <= k <= K:
            raise ValueError(f"step {k} outside 1..{K}")
        j, tag = k - 1, f"{k * DT:g}s"
        out[f"de@{tag}"] = de[:, j].numpy()
        out[f"along_abs@{tag}"] = al[:, j].abs().numpy()
        out[f"along_signed@{tag}"] = al[:, j].numpy()
        out[f"cross_abs@{tag}"] = cr[:, j].abs().numpy()
        out[f"cross_signed@{tag}"] = cr[:, j].numpy()
        out[f"along_sq@{tag}"] = al[:, j].pow(2).numpy()
        out[f"cross_sq@{tag}"] = cr[:, j].pow(2).numpy()
    # horizon-aggregated (mean over all steps) — the "ADE" analogues
    out["ade_dense"] = de.mean(1).numpy()
    out["along_abs_dense"] = al.abs().mean(1).numpy()
    out["cross_abs_dense"] = cr.abs().mean(1).numpy()
    out["cross_peak"] = cr.abs().max(1).values.numpy()      # the tail driver
    return out


def decompose_metric(pred, gt, eid, step=None, mode="ego", n_boot=N_BOOT,
                     seed=0, draws=None):
    """**The M1 helper**: one horizon's L2 error WITH its (lat, lon) split, CI'd.

    Returns ``{de, along_abs, cross_abs, cross_tail, energy_share_at_step}``,
    every interval an episode-cluster bootstrap. Any panel reporting an ADE can
    call this and satisfy *"no ADE is reported without its (lat, lon) split"*
    without re-deriving the geometry."""
    p, g = _as3(pred, "pred"), _as3(gt, "gt")
    K = p.shape[1]
    step = K if step is None else int(step)
    if not 1 <= step <= K:
        raise ValueError(f"step {step} outside 1..{K}")
    al, cr = decompose(p, g, mode)
    j = step - 1
    de = torch.linalg.norm(p - g, dim=-1)[:, j].numpy()
    a, c = al[:, j].abs().numpy(), cr[:, j].abs().numpy()
    d = draws or _drv._Draws(list(eid), n_boot=n_boot, seed=seed)
    return {
        "mode": mode, "step": step, "horizon_s": round(step * DT, 2),
        "de": _drv._interval(de, d),
        "along_abs": _drv._interval(a, d),
        "cross_abs": _drv._interval(c, d),
        "cross_p90": _drv._interval(c, d, reduce="p90"),
        "cross_tail": tail_stats(c),
        "energy_share_at_step": energy_share(al[:, j:j + 1].numpy(),
                                             cr[:, j:j + 1].numpy()),
    }


# ========================================================================== #
# THE BLOCK                                                                    #
# ========================================================================== #
def block(win, mode="ego", n_boot=N_BOOT, seed=0, steps=None,
          thresholds=XTE_THRESHOLDS_M, verify_axes=True, growth_ref_step=None):
    """Full lateral/longitudinal block from a ``rollout.collect`` window dump.

    Requires the dense keys ``pred_dense`` / ``gt_dense`` (persisted since
    2026-07-25). Without them the split can only be computed at 4 knots 0.5 s
    apart, which is precisely the resolution at which the compounding law is
    invisible — so a dense-less dump gets a self-describing ``skipped`` node,
    never a 4-point stand-in wearing the dense name."""
    pd_, gd = win.get("pred_dense"), win.get("gt_dense")
    if pd_ is None or gd is None:
        return {"block": BLOCK, "version": VERSION,
                "skipped": ("no dense path (pred_dense/gt_dense). The 4-knot "
                            "surface cannot express the compounding law: at "
                            "0.5 s spacing the lateral growth curve has 4 "
                            "points. Re-run rollout.collect."),
                "dense_surface_available": False}
    eid = [str(x) for x in win["eid"]]
    p, g = _as3(pd_, "pred_dense"), _as3(gd, "gt_dense")
    K = p.shape[1]
    steps = tuple(steps) if steps else tuple(
        sorted({s for s in (5, 10, 15, 20, K) if 1 <= s <= K}))
    al, cr = decompose(p, g, mode)
    al_np, cr_np = al.numpy(), cr.numpy()
    draws = _drv._Draws(eid, n_boot=n_boot, seed=seed)
    comp = per_window(p, g, mode=mode, steps=steps)

    out = {
        "block": BLOCK, "version": VERSION, "spec": SPEC,
        "mode": mode,
        "axis_convention": ("ego frame: axis0 = ALONG-track (longitudinal), "
                            "axis1 = CROSS-track (lateral)"),
        "n_windows": int(p.shape[0]), "n_episodes": draws.n_episodes,
        "horizon_K": int(K), "horizon_s": round(K * DT, 2), "dt_s": DT,
        "dense_surface_available": True,
        "steps_reported": list(steps),
        "estimator": {
            "interval": "episode_cluster_bootstrap",
            "delta": "paired_episode_cluster_bootstrap",
            "n_boot": int(n_boot), "seed": int(seed),
            "resampling_unit": "val episode",
            "deprecated_and_refused": DEPRECATED_ESTIMATOR,
            "estimator_note": ESTIMATOR_NOTE},
        "thresholds": {
            "xte_grid_m": list(thresholds),
            "lane_half_m": LANE_HALF_M,
            "lane_half_mark": ("PROPOSED — no lane geometry exists in this "
                               "corpus; driving.py refuses lane_centre_"
                               "deviation for the same reason"),
        },
    }
    if verify_axes and mode == "ego":
        out["axis_check"] = assert_axis_convention(g, win.get("speed"))

    # --- per-horizon: L2 AND its two halves, never one without the others --- #
    out["by_horizon"] = {}
    for k in steps:
        tag = f"{k * DT:g}s"
        row = {"de": _drv._interval(comp[f"de@{tag}"], draws),
               "along_abs": _drv._interval(comp[f"along_abs@{tag}"], draws),
               "along_signed": _drv._interval(comp[f"along_signed@{tag}"], draws),
               "cross_abs": _drv._interval(comp[f"cross_abs@{tag}"], draws),
               "cross_signed": _drv._interval(comp[f"cross_signed@{tag}"], draws),
               "cross_p90": _drv._interval(comp[f"cross_abs@{tag}"], draws,
                                           reduce="p90"),
               "cross_tail": tail_stats(comp[f"cross_abs@{tag}"], thresholds),
               "energy_share": energy_share(al_np[:, k - 1:k], cr_np[:, k - 1:k]),
               # the split's own sanity: along^2 + cross^2 == de^2
               "orthogonality_max_abs_residual": round(float(np.max(np.abs(
                   comp[f"along_sq@{tag}"] + comp[f"cross_sq@{tag}"]
                   - comp[f"de@{tag}"] ** 2))), 9)}
        out["by_horizon"][tag] = row

    # --- horizon-aggregated + the two structural findings ------------------- #
    out["dense_aggregate"] = {
        "ade_dense": _drv._interval(comp["ade_dense"], draws),
        "along_abs_dense": _drv._interval(comp["along_abs_dense"], draws),
        "cross_abs_dense": _drv._interval(comp["cross_abs_dense"], draws),
        "cross_peak": _drv._interval(comp["cross_peak"], draws),
        "cross_peak_p90": _drv._interval(comp["cross_peak"], draws,
                                         reduce="p90"),
        "cross_peak_tail": tail_stats(comp["cross_peak"], thresholds),
    }
    out["energy_share"] = energy_share(al_np, cr_np)
    out["growth"] = growth(al_np, cr_np, ref_step=growth_ref_step)
    out["verdict"] = _verdict(out)
    _drv.assert_no_deprecated_estimator(out, _path=BLOCK)
    return out


def from_sparse_windows(win, mode="ego", n_boot=N_BOOT, seed=0,
                        thresholds=XTE_THRESHOLDS_M):
    """M1 on the **4-waypoint** surface, for dumps written before 2026-07-25.

    Every committed ``results/windows_<arm>.pt`` predates the dense-path fix, so
    the 10 Hz block above cannot run on any archived arm without a GPU re-run.
    This path runs the identical decomposition on ``pred``/``gt [N,4,2]`` — 4
    knots 0.5 s apart — so the **98.6 % energy share** and the **cross-track
    tail** become measurable on all 40 val episodes today, which is exactly the
    follow-up ``LATERAL_VS_LONGITUDINAL_ANALYSIS.md`` §6 asks for (*"run the
    same decomposition across all 40 held-out episodes ... the natural first
    task of M1"*).

    ⚠️ **What this surface cannot do.** The growth curve has FOUR points, so the
    compounding ratio is a coarse 0.5→2.0 s estimate, not the 10 Hz law; and
    ``dt`` is 0.5 s, not 0.1. Blocks are stamped ``surface="sparse_4wp"`` and
    ``dt_s = 0.5`` and must never be pooled with a dense block."""
    if win.get("pred") is None or win.get("gt") is None:
        raise ValueError("need pred/gt [N,4,2] in the window dump")
    shim = {"eid": win["eid"], "speed": win.get("speed"),
            "pred_dense": win["pred"], "gt_dense": win["gt"]}
    out = block(shim, mode=mode, n_boot=n_boot, seed=seed,
                steps=tuple(range(1, _as3(win["pred"], "pred").shape[1] + 1)),
                thresholds=thresholds, verify_axes=False,
                # knot 1 IS 0.5 s on this surface, so it is the growth
                # reference. The dense default (step 5) would land on the LAST
                # knot here and report a growth of x1.0 for both axes — a
                # silently meaningless row.
                growth_ref_step=1)
    out.update(surface="sparse_4wp", dt_s=0.5,
               wp_steps=win.get("wp_steps"),
               dense_surface_available=False,
               _surface_warning=(
                   "4 waypoints 0.5 s apart, NOT the 10 Hz dense path. Horizon "
                   "labels in `by_horizon` are knot indices x 0.1 s and are "
                   "NOT seconds on this surface — knot j is j*0.5 s. The "
                   "energy share and the cross-track tail are exact here; the "
                   "growth curve has only 4 points and is a coarse estimate."))
    # re-stamp the horizons in the surface's real units
    out["horizon_s"] = round(out["horizon_K"] * 0.5, 2)
    out["by_horizon"] = {f"{(i + 1) * 0.5:g}s": v
                         for i, (_k, v) in enumerate(out["by_horizon"].items())}
    if mode == "ego" and win.get("speed") is not None:
        try:
            out["axis_check"] = assert_axis_convention(
                win["gt"], win["speed"], dt=0.5)
        except ValueError as e:                      # record, never silently pass
            out["axis_check"] = {"verified": False, "error": str(e)}
    return out


def _verdict(o):
    es = o["energy_share"]["longitudinal_share_of_squared_error"]
    gr = o["growth"]
    agg = o["dense_aggregate"]
    return (
        f"longitudinal carries {es:.1%} of the squared error "
        f"(lateral {1 - es:.1%}) · lateral grows "
        f"x{gr['cross_growth_final']} vs longitudinal x"
        f"{gr['along_growth_final']} over "
        f"{gr['ref_s']}->{o['horizon_s']} s "
        f"(cross faster by x{gr['cross_grows_faster_by']}) · peak |XTE| mean "
        f"{agg['cross_peak']['mean']} m, p90 {agg['cross_peak_p90']['mean']} m, "
        f"{agg['cross_peak_tail']['frac_beyond_m'][f'{LANE_HALF_M:g}']:.1%} of "
        f"windows beyond {LANE_HALF_M} m")


def paired_cross_track(pred_a, pred_b, gt, eid, step=None, mode="ego",
                       n_boot=N_BOOT, seed=0, reduce="mean", knot_dt=None):
    """PAIRED Δ cross-track between two arms on the SAME windows.

    **The channel HP-2 and HP-3 are measured in.** Oriented ``b − a``, so a
    POSITIVE delta means arm ``a`` has the smaller cross-track error, i.e. ``a``
    wins. ``reduce="p90"`` runs the same test on the tail, which is where §M2
    says the decision lives."""
    pa, pb, g = (_as3(pred_a, "pred_a"), _as3(pred_b, "pred_b"),
                 _as3(gt, "gt"))
    K = g.shape[1]
    step = K if step is None else int(step)
    _, ca = decompose(pa, g, mode)
    _, cb = decompose(pb, g, mode)
    j = step - 1
    d = _ci.paired_episode_cluster_bootstrap(
        cb[:, j].abs().numpy(), ca[:, j].abs().numpy(), [str(x) for x in eid],
        n_boot=n_boot, seed=seed, reduce=reduce)
    # HORIZON LABELLING — fixed 2026-07-26. `step` counts KNOTS, and on the
    # SPARSE surface a knot is NOT one dense timestep: the 4-knot surface sits at
    # dense steps [5,10,15,20], so the last knot is 2.0 s, not 0.4 s. The old
    # `step * DT` therefore under-reported the horizon by 5x on every sparse call
    # — and a mislabelled horizon is exactly the C9 defect (`GATE_PROTOCOL` §0.7):
    # a verdict that names the wrong timescale is not admissible.
    # `knot_dt` = seconds PER KNOT. Passed explicitly when known (rollout now
    # emits `dense_steps`/`dt_s`); otherwise inferred from the knot count against
    # the 20-step / 2.0 s dense horizon, and the inference is STAMPED so a reader
    # can see it was inferred rather than measured.
    if knot_dt is not None:
        _kdt, _src = float(knot_dt), "explicit"
    elif K in (_DENSE_K, _SPARSE_K):
        _kdt, _src = (_DENSE_K * DT) / K, "inferred_from_knot_count"
    else:
        _kdt, _src = DT, "unknown_knot_spacing_assumed_dense"
    d.update(_orientation=("b - a on |cross-track|; POSITIVE = arm `a` has the "
                           "smaller lateral error = `a` wins"),
             mode=mode, step=int(step), n_knots=int(K),
             horizon_s=round(step * _kdt, 2),
             horizon_provenance=_src)
    return d


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser("taniteval.lateral")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--mode", default="ego", choices=list(MODES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from taniteval import rollout
    res = block(rollout.load_windows(a.windows), mode=a.mode, n_boot=a.n_boot)
    if res.get("skipped"):
        print(f"[lateral] SKIPPED: {res['skipped']}")
        return
    print(res["verdict"])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"[lateral] wrote {a.out}")


if __name__ == "__main__":
    main()
