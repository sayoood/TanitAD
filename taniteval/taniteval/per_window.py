"""TanitEval -- the OPT-IN PER-WINDOW EXPORT of the trajectory-level family components.

WHY THIS EXISTS (P6-HN screen, PREREG_P6HN_SCREEN.md section 10 + amendment B4).  ``four_families.all_families`` reports each LONGITUDINAL / LATERAL / TACTICAL
component as an AGGREGATE with its own single-arm episode-cluster interval.  A two-arm difference of those is an UNPAIRED non-overlap read, which is not what the
screen registered: its seed-mean differences must be PAIRED on the same joint log draws as the PDM sub-scores.  A paired interval needs the arrays the aggregate
was made of.  This module exports exactly those arrays, one row per scored window, and nothing else.

THE DESIGN RULE (the same one ``four_families._ci_block`` states): the exported arrays are the quantities the block's OWN interval is built from, aggregated by the
SAME reduction -- mean, rms, ratio-of-sums (masked components), median.  The point estimate is therefore the block's printed number, not a second statistic.
:func:`point` is that reduction; ``tests/test_per_window_export.py`` pins ``point(component) == the block's printed value`` for EVERY component, on the dense and the
sparse grid, so a drift between this module and ``four_families`` turns a test RED rather than quietly publishing a near-miss.

OPT-IN, AND BYTE-IDENTICAL WHEN OFF.  Nothing in ``four_families.py`` is edited.  ``adapters.navsim.four_families_block(..., per_window=True)`` calls :func:`export` and
attaches the result under ``fam["_per_window"]``; with the default ``per_window=False`` the block is the one that was always produced (pinned byte-for-byte against the
tip's own function by the same test file).

COMPONENT RECORD.  ``{"kind": "mean"|"rms"|"ratio"|"median", ...}``
  mean     ``value`` [N]            point = mean(value)
  rms      ``value`` [N] (per-window mean of squares)   point = sqrt(mean(value))
  ratio    ``num`` [N], ``den`` [N], ``scale``           point = scale * sum(num) / sum(den)      (masked means; a subset mean is a ratio with den = the 0/1 membership)
  median   ``value`` [R], ``rows`` [R]                   point = median(value)                    (R = the windows that carry the component)
Every record also has ``dp`` (the block's printed decimals).  ``human_only`` marks a property of the logged human future (identical for every arm; never compared).

TACTICAL.  The class ids of the arm's and the human's EXECUTED manoeuvres (the programme's own ``refc_tactical`` labeller, imported never re-implemented): the
per-window ``gt`` / ``pred`` arrays from which accuracy (the correctness flag ``gt == pred``) and kappa are recomputed on any resample.  ``goal_setting`` carries the
goal-point components of ``four_families.tactical_goal``.

STRATEGIC is not exported: it is PAUSED (PI 2026-09-19, PI_DECISION_QUEUE.md item 25) and the caller stamps it.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import four_families as ff

SCHEMA = "taniteval.per_window/1"
_DEG = 180.0 / math.pi
#: components that are properties of the logged human future (same for every arm) -- exported, flagged, never compared (``proxy_eval.FAM_SKIP`` names the same two)
HUMAN_ONLY = ("ego_progress.gt_progress_mean_m", "ego_progress.t0_axis_gt_self_ratio")


# --------------------------------------------------------------------------- records
def _f64(t):
    return t.detach().cpu().numpy().astype(np.float64)


def _mean(value, dp, **kw):
    return {"kind": "mean", "value": np.asarray(value, np.float64).reshape(-1), "dp": dp, **kw}


def _rms(value, dp, **kw):
    return {"kind": "rms", "value": np.asarray(value, np.float64).reshape(-1), "dp": dp, **kw}


def _ratio(num, den, scale, dp, **kw):
    return {"kind": "ratio", "num": np.asarray(num, np.float64).reshape(-1), "den": np.asarray(den, np.float64).reshape(-1), "scale": float(scale), "dp": dp, **kw}


def _median(value, rows, dp, **kw):
    return {"kind": "median", "value": np.asarray(value, np.float64).reshape(-1), "rows": np.asarray(rows, np.int64).reshape(-1), "dp": dp, **kw}


def _subset_mean(full_value, valid, dp, **kw):
    """a mean over a subset of windows == a ratio of sums with den = membership (the form the masked components already use)"""
    v = np.asarray(valid, bool)
    return _ratio(np.where(v, np.asarray(full_value, np.float64), 0.0), v.astype(np.float64), 1.0, dp, **kw)


def point(c: dict) -> float:
    """the full-set point estimate of one component record -- the reduction the block itself prints (float64; the block rounds to ``dp`` decimals)"""
    k = c["kind"]
    if k == "mean":
        return float(np.mean(c["value"]))
    if k == "rms":
        return float(math.sqrt(float(np.mean(c["value"]))))
    if k == "ratio":
        d = float(np.sum(c["den"]))
        return float(c["scale"] * float(np.sum(c["num"])) / d) if d > 0 else float("nan")
    if k == "median":
        return float(np.median(c["value"])) if c["value"].size else float("nan")
    raise ValueError(f"unknown component kind {k!r}")


def _na(reason, n, **extra):
    return {"status": "UNAVAILABLE", "reason": reason, "n": int(n), **extra}


# --------------------------------------------------------------------------- the export
def _grid(win, prefer_dense):
    """the same grid choice as ``all_families`` (dense when both dense paths exist, else the sparse view with a DERIVED dt)"""
    dense = prefer_dense and win.get("pred_dense") is not None and win.get("gt_dense") is not None
    if dense:
        return (torch.as_tensor(win["pred_dense"]).float(), torch.as_tensor(win["gt_dense"]).float(), float(win.get("dt_s", ff.DT_S) or ff.DT_S), "dense")
    pred = torch.as_tensor(win["pred"]).float()
    gt = torch.as_tensor(win["gt"]).float()
    dt, _prov = ff.infer_dt(win)
    return pred, gt, dt, "sparse"


def export(win: dict, fam: dict | None = None, *, prefer_dense: bool = True) -> dict:
    """-> the per-window record of one arm.  ``win`` is the harness window dict (``pred`` / ``gt`` / ``pred_dense`` / ``gt_dense`` / ``dt_s`` / ``eid`` / ``lead`` / ``_navsim``);
    ``fam`` is the ``all_families`` block already computed for it, when the caller has it (its distance-keeping arrays are reused, never recomputed)."""
    gt_present = win.get("gt_dense") is not None or win.get("gt") is not None
    n = int((win.get("pred_dense") if win.get("pred_dense") is not None else win["pred"]).shape[0])
    ns = win.get("_navsim") or {}
    toks = ns.get("scene_tokens")
    out = {"schema": SCHEMA, "n": n, "tokens": None if toks is None else [str(t) for t in toks], "eid": None if win.get("eid") is None else [str(e) for e in win["eid"]],
           "strategic": {"status": "PAUSED", "ruling": "PI 2026-09-19, PI_DECISION_QUEUE.md item 25", "note": "not exported: reported as PAUSED, not a pass, not omitted"}}
    if not gt_present:
        out.update(status="UNAVAILABLE", reason="no ground-truth path in the window: no geometry family exists, so there is nothing to export", longitudinal={}, lateral={}, tactical={}, unavailable={})
        return out
    pred, gt, dt, grid = _grid(win, prefer_dense)
    if pred.ndim != 3 or pred.shape[-1] != 2 or pred.shape != gt.shape:
        raise ValueError(f"expected pred/gt [n,H,2] of one shape, got {tuple(pred.shape)} / {tuple(gt.shape)}")
    P, G = ff._seq_geometry(pred, dt), ff._seq_geometry(gt, dt)
    unavailable: dict = {}
    out.update(status="OK", grid=grid, dt_s=dt, horizon_steps=int(pred.shape[1]))

    # ------------------------------------------------------------------ LONGITUDINAL  (the arrays of four_families._longitudinal_ci)
    sp_err = P["speed"] - G["speed"]
    al_err = P["along"] - G["along"]
    ac_err = P["accel"] - G["accel"]
    lon: dict = {
        "speed_mae_mps": _mean(_f64(sp_err.abs().mean(1)), 4),
        "speed_bias_mps": _mean(_f64(sp_err.mean(1)), 4),
        "speed_rmse_mps": _rms(_f64((sp_err ** 2).mean(1)), 4),
    }
    for b in ff.TARGET_SPEED_BANDS_MPS:
        lon[f"target_speed_acc.within_{b}_mps"] = _mean(_f64((sp_err.abs() <= b).float().mean(1)), 4)
    lon["along_mae_m"] = _mean(_f64(al_err.abs().mean(1)), 4)
    lon["along_bias_m"] = _mean(_f64(al_err.mean(1)), 4)
    lon["along_final_bias_m"] = _mean(_f64(al_err[:, -1]), 4)
    if ac_err.shape[1] > 0:
        lon["accel_mae_mps2"] = _mean(_f64(ac_err.abs().mean(1)), 4)
    else:
        unavailable["longitudinal.accel_mae_mps2"] = _na("the horizon has fewer than 2 steps, so no acceleration exists to difference", n)
    from .progress import progress_per_window
    pn, gn = pred.detach().cpu().numpy(), gt.detach().cpu().numpy()
    w = progress_per_window(pn, gn, "human_dir")
    v = np.asarray(w["valid"], bool)
    if int(v.sum()) > 0:
        rows = np.flatnonzero(v)
        lon["ego_progress.progress_ratio_mean"] = _subset_mean(w["ratio"], v, 4)
        lon["ego_progress.progress_ratio_median"] = _median(w["ratio"][v], rows, 4)
        lon["ego_progress.progress_error_mean"] = _subset_mean(w["error"], v, 4)
        lon["ego_progress.under_progress_rate"] = _subset_mean((w["ratio"] < 1.0).astype(np.float64), v, 4)
        lon["ego_progress.gt_progress_mean_m"] = _subset_mean(w["gt_progress_m"], v, 4, human_only=True)
        gs = progress_per_window(gn, gn, "t0_axis")
        gv = np.asarray(gs["valid"], bool)
        if gv.any():
            lon["ego_progress.t0_axis_gt_self_ratio"] = _subset_mean(gs["ratio"], gv, 4, human_only=True)
    else:
        for nm in ("progress_ratio_mean", "progress_ratio_median", "progress_error_mean", "under_progress_rate"):
            unavailable[f"longitudinal.ego_progress.{nm}"] = _na("every window's logged human chord is below the progress gate: a progress ratio has no denominator", 0)
    dk = ((fam or {}).get("longitudinal") or {}).get("distance_keeping")
    if dk is None:
        dk = ff._distance_keeping(pred, dt, win.get("lead"), gt)
    dkp = dk.get("_per_window") if isinstance(dk, dict) else None
    dk_names = {"headway_min_m": "distance_keeping.mean_headway_min_m", "time_gap_min_s": "distance_keeping.mean_time_gap_min_s", "min_ttc_s": "distance_keeping.mean_min_ttc_s"}
    if not isinstance(dk, dict) or dk.get("status") != "OK" or dkp is None:
        why = (dk or {}).get("reason", "distance keeping produced no per-window arrays") if isinstance(dk, dict) else "distance keeping produced no per-window arrays"
        for nm in dk_names.values():
            unavailable[f"longitudinal.{nm}"] = _na(why, int((dk or {}).get("n", 0) or 0) if isinstance(dk, dict) else 0)
    else:
        for src, nm in dk_names.items():
            a = np.asarray(dkp[src], np.float64)
            fin = np.isfinite(a)
            if fin.any():
                lon[nm] = _subset_mean(np.where(fin, a, 0.0), fin, 4)
            else:
                unavailable[f"longitudinal.{nm}"] = _na("no window has a finite value (no lead, or every ego speed below the time-gap floor)", 0)

    # ------------------------------------------------------------------ LATERAL  (the arrays of four_families._lateral_ci; masks as in four_families.lateral)
    both = P["valid"] & G["valid"]
    both_pair = P["pair_valid"] & G["pair_valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    ct_err = P["cross"] - G["cross"]
    lat: dict = {}

    def masked(name, err, mask, scale, dp, why):
        cnt = _f64(mask.sum(1))
        s = _f64((err * mask).sum(1))
        if float(cnt.sum()) > 0:
            lat[name] = _ratio(s, cnt, scale, dp)
        else:
            unavailable[f"lateral.{name}"] = _na(why, 0)

    masked("heading_mae_deg", dh.abs(), both, _DEG, 4, "no step clears the min-displacement gate: a stopped path has no tangent, so no heading error exists")
    masked("yaw_rate_mae_degps", (P["yaw_rate"] - G["yaw_rate"]).abs(), both_pair, _DEG, 4,
           "no window has two consecutive valid steps (RETR-2026-09-26-YAWMASK: tangent-less steps are masked), so no yaw rate is defined")
    masked("curvature_mae_1pm", (P["curvature"] - G["curvature"]).abs(), both_pair, 1.0, 6, "no window has two consecutive valid steps, so no curvature is defined")
    masked("curvature_bias_1pm", P["curvature"] - G["curvature"], both_pair, 1.0, 6, "no window has two consecutive valid steps, so no curvature is defined")
    lat["cross_mae_m"] = _mean(_f64(ct_err.abs().mean(1)), 4)
    lat["cross_bias_m"] = _mean(_f64(ct_err.mean(1)), 4)
    lat["cross_final_mae_m"] = _mean(_f64(ct_err[:, -1].abs()), 4)
    ph, gh = ns.get("navsim_pred_heading_rad"), ns.get("navsim_gt_heading_rad")
    if ph is not None and gh is not None:
        d = np.asarray(ph, np.float64)[:, -1] - np.asarray(gh, np.float64)[:, -1]
        lat["last_pose_heading_err_deg"] = _mean(np.abs((d + np.pi) % (2 * np.pi) - np.pi) * _DEG, 4, extra_to_the_four_family_block=True)
    else:
        unavailable["lateral.last_pose_heading_err_deg"] = _na("the window carries no NavSim yaw channel for the pred and the gt poses", n)
    out["longitudinal"], out["lateral"] = lon, lat

    # ------------------------------------------------------------------ TACTICAL  (tactical_from_trajectory's class ids; goal setting)
    try:
        from tanitad.refs.refc_tactical import COLLAPSE_TABLE, LAT_CLASSES, LON_CLASSES, MAN5_NAMES, factor_from_kinematics
    except Exception as e:                                                  # noqa: BLE001 -- reported, never dropped
        out["tactical"] = _na(f"the canonical factored labeller `tanitad.refs.refc_tactical` is not importable ({type(e).__name__}: {e}); it is not re-implemented here", n)
    else:
        dyp, dvp, v0p, v1p, _ = ff.maneuver_kinematics(pred, dt)
        dyg, dvg, v0g, v1g, _ = ff.maneuver_kinematics(gt, dt)
        lat_p, lon_p = factor_from_kinematics(dyp, dvp, v0p, v1p)
        lat_g, lon_g = factor_from_kinematics(dyg, dvg, v0g, v1g)
        tbl = torch.as_tensor(COLLAPSE_TABLE, dtype=torch.long)
        man_p, man_g = tbl[lat_p, lon_p], tbl[lat_g, lon_g]

        def dec(g_, p_, names):
            return {"classes": list(names), "gt": g_.detach().cpu().numpy().astype(np.int64), "pred": p_.detach().cpu().numpy().astype(np.int64)}

        pe = pred[:, -1].detach().cpu().numpy().astype(np.float64)
        ge = gt[:, -1].detach().cpu().numpy().astype(np.float64)
        rp, rg = np.linalg.norm(pe, axis=-1), np.linalg.norm(ge, axis=-1)
        db = np.degrees((np.arctan2(pe[:, 1], pe[:, 0]) - np.arctan2(ge[:, 1], ge[:, 0]) + np.pi) % (2 * np.pi) - np.pi)
        move = rg > max(ff.MIN_DS_M, 0.5)
        goal = {"goal_point_error_m": _mean(np.linalg.norm(pe - ge, axis=-1), 4),
                "goal_long_bias_m": _mean(pe[:, 0] - ge[:, 0], 4),
                "goal_lat_bias_m": _mean(pe[:, 1] - ge[:, 1], 4)}
        if move.any():
            goal["goal_bearing_mae_deg"] = _subset_mean(np.abs(db), move, 4)
            goal["goal_range_ratio"] = _subset_mean(np.where(move, rp / np.maximum(rg, 1e-6), 0.0), move, 4)
        out["tactical"] = {"status": "OK", "lateral_decision": dec(lat_g, lat_p, LAT_CLASSES), "longitudinal_decision": dec(lon_g, lon_p, LON_CLASSES),
                           "maneuver_5way_collapsed": dec(man_g, man_p, MAN5_NAMES), "goal_setting": goal,
                           "source": "trajectory-derived: the EXECUTED manoeuvre of the arm's pick vs the human's (not 'selected vs executed')"}
    out["unavailable"] = unavailable
    return out
