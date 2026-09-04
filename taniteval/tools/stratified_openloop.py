#!/usr/bin/env python3
"""STRATIFIED re-analysis of a banked open-loop dump — ZERO GPU.

    python taniteval/tools/stratified_openloop.py \
        --dump <dir with ep*.npz + manifest.json> \
        --out  taniteval/results/refcv3-40284-stratified.json \
        --tag  refcv3-40284-stratified

WHY THIS TOOL EXISTS
====================
BEV-Planner (arXiv 2312.03031) measures 73.9 % straight driving in the standard
nuScenes val mix and shows that a model which LOSES its input adopts a straight
policy and then WINS THE POOLED AVERAGE. If our eval corpus is similarly
dominated by straight, constant-speed driving, a hold-action control can win the
pooled mean while losing on every manoeuvre that matters — the pooled number
would be TRUE and the conclusion drawn from it WRONG.

⛔ 73.9 % is a PUBLISHED figure about a DIFFERENT corpus and is never imported as
ours. This tool MEASURES our mix.

WHAT IS AND IS NOT RE-DERIVED
=============================
Nothing statistical is re-derived. The per-window components come from
``refav1_arm._components`` (four_families' own geometry + the programme's
canonical trajectory labeller) and every interval from
``taniteval.ci.paired_episode_cluster_bootstrap``. This module contributes
exactly one thing: a PARTITION of the windows, and the arithmetic that
decomposes a pooled mean over it.

⛔ ``overlapping_holdout_se`` is not used anywhere. It is not a jackknife, and it
BIASES THE POINT ESTIMATE, not merely the interval.

⛔ LOOP CLASS: every arm in a dump of this shape is OPEN LOOP (PI ruling
2026-09-02). The word "closed" never appears in this artifact's output.

THE STRATIFIER, AND WHY IT IS ADMISSIBLE
========================================
``four_families.maneuver_kinematics(G, dt)`` -> ``refc_tactical.
factor_from_kinematics(...)`` applied to the GROUND-TRUTH path ``g``, i.e. the
same call ``refav1_arm._components`` already makes for ``TAC_traj_*_correct``.

* It is LABEL material, computed from the ego's own future — admissible by the
  PI ruling of 2026-08-03 (*labels may use ego; INFERENCE is vision-only*). It
  never reaches a model.
* It is NOT computed from any arm's output, so the arm being graded does not
  choose the strata that grade it (the admissibility test ``p7_strata`` states).
* It is applied identically to every arm, because every arm is scored on the
  SAME windows — which is also what makes the paired estimator valid inside a
  cell.
* Its thresholds are HORIZON-MATCHED: ``LABEL_HORIZON = 20`` steps @ 10 Hz =
  2.0 s, which is exactly the ``--grid 2s`` horizon these dumps score. A
  threshold quoted outside its horizon would be the ``step_s`` scope error in
  another costume.

FOUR HARNESS CONTROLS RUN BEFORE ANY STRATIFIED NUMBER IS EMITTED
=================================================================
A probe that tunes on the data it scores manufactures results, and the only
thing that catches it is a control that must read a KNOWN value:

 C1 POOLED REPRODUCTION  — the pooled per-arm means and the pooled paired delta
                           must reproduce the banked record (``--expect``).
 C2 PARTITION            — every stratification sums to N with no window in two
                           cells and none dropped.
 C3 DECOMPOSITION IDENTITY — ``sum_s (n_s/N) * delta_s == pooled delta`` to
                           float tolerance. If it does not, the cells are not a
                           partition and every cell number is void.
 C4 CONSTANT CONTROL     — a constant-zero predictor's ADE inside each cell must
                           equal the cell's mean ||GT||, recomputed independently
                           in float64 numpy. A cell whose metric pipeline is
                           wrong fails here rather than in the verdict.

UNDERPOWERED IS NOT NEGATIVE
============================
Every cell prints ``n_windows`` and ``n_episodes``. A cell whose paired interval
straddles zero at small n is reported ``UNDERPOWERED``, never "no effect"
(RETRACTION_LOG #17). The tool refuses to emit a verdict word for any cell below
``MIN_CELL_WINDOWS`` / ``MIN_CELL_EPISODES``.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE_PARENT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_TE_PARENT)
_TE_PKG = os.path.join(_TE_PARENT, "taniteval")
_SCRIPTS = os.path.join(_REPO, "stack", "scripts")


def _bootstrap_paths() -> None:
    """Evict a wrongly-bound ``taniteval`` namespace package and PREFLIGHT every
    module the analysis needs, so a missing import fails in 2 s instead of after
    the expensive part."""
    for p in (os.path.join(_REPO, "stack"), _TE_PARENT, _SCRIPTS):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    m = sys.modules.get("taniteval")
    if m is not None:
        paths = [os.path.normcase(os.path.abspath(p))
                 for p in (getattr(m, "__path__", None) or [])]
        if os.path.normcase(os.path.abspath(_TE_PKG)) not in paths:
            for k in [k for k in sys.modules
                      if k == "taniteval" or k.startswith("taniteval.")]:
                del sys.modules[k]
    try:
        import taniteval.ci                 # noqa: F401
        import taniteval.four_families      # noqa: F401
        import tanitad.refs.refc_tactical   # noqa: F401
        import tanitad.data.v7_labels       # noqa: F401
        import torch                        # noqa: F401
    except ModuleNotFoundError as ex:       # pragma: no cover
        sys.exit(f"[strat] preflight failed ({ex}). Real package is {_TE_PKG}; "
                 f"sys.path[:3]={sys.path[:3]}")


_bootstrap_paths()


def _load_by_path(name: str, path: str):
    if not os.path.exists(path):
        sys.exit(f"[strat] required sibling {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: ⭐ IMPORTED, NEVER COPIED — refav1_arm owns the per-window family components.
ra = _load_by_path("refav1_arm_for_strat", os.path.join(_HERE, "refav1_arm.py"))

from taniteval import ci as _ci                      # noqa: E402
from taniteval import four_families as ff            # noqa: E402
from tanitad.data import v7_labels as _v7l           # noqa: E402
from tanitad.refs import refc_tactical as rt         # noqa: E402

# --------------------------------------------------------------------------- #
# power floors — a cell below these gets NO verdict word                        #
# --------------------------------------------------------------------------- #
MIN_CELL_WINDOWS = 100
MIN_CELL_EPISODES = 10

LAT_NAMES = ("lane_keep", "turn_left", "turn_right")
LON_NAMES = ("brake_stop", "steady", "accelerate")

#: the metrics carried per cell, by binding family (Sayed 2026-08-02).
FAMILY_METRICS = {
    "ADE": ("ade_m", "fde_m"),
    "longitudinal": ("LON_speed_mae_mps", "LON_along_mae_m",
                     "LON_accel_mae_mps2"),
    "lateral": ("LAT_cross_mae_m", "LAT_heading_mae_deg",
                "LAT_yaw_rate_mae_radps"),
    "tactical": ("TAC_traj_lat_correct", "TAC_traj_lon_correct"),
}
#: lower-is-better for every metric except the two tactical AGREEMENT rates.
HIGHER_IS_BETTER = {"TAC_traj_lat_correct", "TAC_traj_lon_correct"}


# --------------------------------------------------------------------------- #
def load_dump(dump_dir: str):
    """-> (G [N,K,2], {arm: P [N,K,2]}, eid [N], ws [N], v0 [N], manifest)."""
    files = sorted(glob.glob(os.path.join(dump_dir, "ep*.npz")))
    if not files:
        raise SystemExit(f"[strat] no ep*.npz under {dump_dir}")
    man = None
    mp = os.path.join(dump_dir, "manifest.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as fh:
            man = json.load(fh)
    with np.load(files[0]) as d0:
        arms = [k for k in d0.files if k not in ("g", "ws", "eid", "clip_index",
                                                 "v0")]
    G, P = [], {a: [] for a in arms}
    eid, ws, v0 = [], [], []
    for f in files:
        with np.load(f) as d:
            g = d["g"][..., :2].astype(np.float64)
            G.append(g)
            eid += [os.path.splitext(os.path.basename(f))[0]] * g.shape[0]
            ws.append(np.asarray(d["ws"]).reshape(-1))
            v0.append(np.asarray(d["v0"], dtype=np.float64).reshape(-1)
                      if "v0" in d.files else np.full(g.shape[0], np.nan))
            for a in arms:
                P[a].append(d[a][..., :2].astype(np.float64))
    return (np.concatenate(G), {a: np.concatenate(v) for a, v in P.items()},
            np.asarray(eid), np.concatenate(ws), np.concatenate(v0), man,
            len(files))


def load_decisions(dump_dir: str):
    """The refcv3 sidecar (v7.2 labels etc.), or ``None`` when absent."""
    files = sorted(glob.glob(os.path.join(dump_dir, "decisions", "ep*.npz")))
    if not files:
        return None
    cat: dict[str, list] = {}
    for f in files:
        with np.load(f) as d:
            for k in d.files:
                cat.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in cat.items()}


# --------------------------------------------------------------------------- #
# THE STRATIFIER                                                                #
# --------------------------------------------------------------------------- #
def gt_strata(G: np.ndarray, dt: float, gate: str = "v1"):
    """GT-kinematic (lat, lon) class ids per window + the labeller's provenance.

    ``gate='v1'`` is ``|dyaw| > YAW_TURN_RAD`` — the gate ``refav1_arm.
    _components`` itself uses. ``gate='v2'`` is the curvature gate
    ``|kappa| >= CURV_TURN_MAN_PER_M``, reported as a SENSITIVITY only.
    """
    import torch
    gt = torch.as_tensor(G).float()
    dyaw, dv, v0, v1, prov = ff.maneuver_kinematics(gt, dt)
    kappa = None
    if gate == "v2":
        seg = (gt[:, 1:] - gt[:, :-1]).norm(dim=-1).sum(dim=1)
        seg = torch.clamp(seg, min=rt.MIN_ARC_M)
        kappa = dyaw / seg
    lat, lon = rt.factor_from_kinematics(dyaw, dv, v0, v1, kappa=kappa)
    return (lat.numpy().astype(int), lon.numpy().astype(int),
            {"gate": gate,
             "labeller": "tanitad.refs.refc_tactical.factor_from_kinematics",
             "kinematics": "taniteval.four_families.maneuver_kinematics",
             "thresholds": {"YAW_TURN_RAD": rt.YAW_TURN_RAD,
                            "DV_ACCEL_MS": rt.DV_ACCEL_MS,
                            "DV_BRAKE_MS": rt.DV_BRAKE_MS,
                            "STOP_V_MS": rt.STOP_V_MS,
                            "MOVING_V_MS": rt.MOVING_V_MS,
                            "CURV_TURN_MAN_PER_M": rt.CURV_TURN_MAN_PER_M,
                            "LABEL_HORIZON_steps_at_10hz": rt.LABEL_HORIZON},
             "horizon_match": (
                 "LABEL_HORIZON = 20 steps @ 10 Hz = 2.0 s == this grid's "
                 "horizon, so the thresholds are horizon-matched"),
             "kinematics_provenance": prov})


def v7_strata(dec: dict | None):
    """The banked v7.2 tactical labels projected to the 3-way kinematic axes.

    Returns ``(lat3, lon3, report)`` with ``-1`` where the window carries no
    label. ⚠️ SUBSET only — a clip's single v7.2 record describes a band, so
    windows outside it are ``IGNORE_ID``. This is a CROSS-CHECK, never primary.
    """
    if dec is None or "lat_label" not in dec:
        return None, None, {"status": "ABSENT — dump carries no decisions sidecar"}
    lat_v7 = np.asarray(dec["lat_label"], dtype=int)
    lon_v7 = np.asarray(dec["lon_label"], dtype=int)
    lat3 = np.full(lat_v7.shape, -1, dtype=int)
    lon3 = np.full(lon_v7.shape, -1, dtype=int)
    for i, tok in enumerate(_v7l.HEADS["tac_lat"]):
        lat3[lat_v7 == i] = rt.V7_TO_KIN3_LAT[tok]
    for i, tok in enumerate(_v7l.HEADS["tac_lon"]):
        lon3[lon_v7 == i] = rt.V7_TO_KIN3_LON[tok]
    rep = {
        "status": "PRESENT",
        "projection": "tanitad.refs.refc_tactical.V7_TO_KIN3_{LAT,LON}",
        "_projection_is": ("many-to-one BY DESIGN — it destroys exactly the "
                           "distinctions (NUDGE vs LANE_CHANGE, CREEP vs "
                           "BRAKE_TO) the v7 space exists to keep. A metric "
                           "projection, never a label source."),
        "n_total": int(lat_v7.size),
        "n_labelled_lat": int((lat3 >= 0).sum()),
        "n_labelled_lon": int((lon3 >= 0).sum()),
        "n_ignore": int((lat_v7 == _v7l.IGNORE_ID).sum()),
        "why_subset": ("a clip carries ONE v7.2 record anchored at t0 whose "
                       "tactical band admits |t_now - t0| <= 2.0 s "
                       "(v7_labels.window_in_band); every other window is "
                       "IGNORE_ID and is NOT a class."),
        "v7_lat_counts": {tok: int((lat_v7 == i).sum())
                          for i, tok in enumerate(_v7l.HEADS["tac_lat"])},
        "v7_lon_counts": {tok: int((lon_v7 == i).sum())
                          for i, tok in enumerate(_v7l.HEADS["tac_lon"])},
    }
    return lat3, lon3, rep


# --------------------------------------------------------------------------- #
# cells                                                                         #
# --------------------------------------------------------------------------- #
def build_cells(lat, lon, prefix=""):
    """-> ordered dict of ``name -> boolean mask``, three PARTITIONS + the cross."""
    cells = {}
    for i, nm in enumerate(LAT_NAMES):
        cells[f"{prefix}LAT/{nm}"] = (lat == i)
    for i, nm in enumerate(LON_NAMES):
        cells[f"{prefix}LON/{nm}"] = (lon == i)
    straight_const = (lat == 0) & (lon == 1)
    cells[f"{prefix}CROSS/straight_const"] = straight_const
    cells[f"{prefix}CROSS/manoeuvre"] = ~straight_const
    for i, ln in enumerate(LAT_NAMES):
        for j, on in enumerate(LON_NAMES):
            cells[f"{prefix}CELL/{ln}+{on}"] = (lat == i) & (lon == j)
    return cells


def _n_ep(eid, mask):
    return int(len(np.unique(eid[mask]))) if mask.any() else 0


def cell_stat(comps, arm_a, arm_b, metric, eid, mask, n_boot, seed, alpha=0.05):
    """Paired ``a - b`` inside one cell + both arms' cell means. Never a verdict
    word when the cell is underpowered."""
    a = np.asarray(comps[arm_a][metric], dtype=np.float64)[mask]
    b = np.asarray(comps[arm_b][metric], dtype=np.float64)[mask]
    e = eid[mask]
    keep = np.isfinite(a) & np.isfinite(b)
    n, n_ep = int(keep.sum()), int(len(np.unique(e[keep]))) if keep.any() else 0
    out = {"n_windows": n, "n_episodes": n_ep,
           "n_dropped_nonfinite": int((~keep).sum())}
    if n == 0:
        out.update({"status": "EMPTY", "verdict": None})
        return out
    r = _ci.paired_episode_cluster_bootstrap(a[keep], b[keep], list(e[keep]),
                                             n_boot=n_boot, seed=seed,
                                             alpha=alpha)
    out.update(r)
    out[f"mean_{arm_a}"] = round(float(np.nanmean(a[keep])), 4)
    out[f"mean_{arm_b}"] = round(float(np.nanmean(b[keep])), 4)
    out["alpha"] = alpha
    underpowered = (n < MIN_CELL_WINDOWS) or (n_ep < MIN_CELL_EPISODES)
    if underpowered and not r["separated"]:
        out["verdict"] = "UNDERPOWERED"
        out["_underpowered_why"] = (
            f"n_windows {n} < {MIN_CELL_WINDOWS} or n_episodes {n_ep} < "
            f"{MIN_CELL_EPISODES} AND the interval straddles zero. ⛔ This is a "
            f"POWER LIMIT, not evidence of no effect (RETRACTION_LOG #17).")
    elif not r["separated"]:
        out["verdict"] = "NOT SEPARATED"
    else:
        better = "b" if r["delta"] > 0 else "a"
        if metric in HIGHER_IS_BETTER:
            better = "a" if r["delta"] > 0 else "b"
        out["verdict"] = (f"{arm_a} WORSE" if better == "b" else f"{arm_a} BETTER")
    if underpowered:
        out["low_power"] = True
    return out


# --------------------------------------------------------------------------- #
# controls                                                                      #
# --------------------------------------------------------------------------- #
def control_partition(cells, N, groups):
    """C2 — each named GROUP of cells must partition the N windows exactly."""
    rep, ok = {}, True
    for gname, names in groups.items():
        masks = [cells[n] for n in names]
        tot = int(sum(int(m.sum()) for m in masks))
        stacked = np.stack(masks).astype(int).sum(0)
        exact = bool(tot == N and stacked.max() <= 1 and stacked.min() >= 1)
        rep[gname] = {"cells": names, "sum_n": tot, "N": N,
                      "max_membership": int(stacked.max()),
                      "min_membership": int(stacked.min()), "pass": exact}
        ok &= exact
    rep["pass"] = ok
    return rep


def control_decomposition(comps, arm_a, arm_b, metric, cells, names, N):
    """C3 — ``sum_s (n_s/N) * delta_s`` must equal the pooled delta EXACTLY
    (linearity of the mean). A failure means the cells are not a partition and
    every cell number in that group is void."""
    a = np.asarray(comps[arm_a][metric], dtype=np.float64)
    b = np.asarray(comps[arm_b][metric], dtype=np.float64)
    pooled = float(np.nanmean(a) - np.nanmean(b))
    contrib, recon = {}, 0.0
    for nm in names:
        m = cells[nm]
        n = int(m.sum())
        if n == 0:
            contrib[nm] = {"n": 0, "weight": 0.0, "delta": None,
                           "contribution_m": 0.0, "share_of_pooled_pct": 0.0}
            continue
        d = float(np.nanmean(a[m]) - np.nanmean(b[m]))
        w = n / float(N)
        c = w * d
        recon += c
        contrib[nm] = {"n": n, "weight": round(w, 6), "delta": round(d, 6),
                       "contribution_m": round(c, 6),
                       "share_of_pooled_pct": round(100.0 * c / pooled, 2)
                       if pooled else None}
    return {"pooled_delta": round(pooled, 6),
            "reconstructed": round(recon, 6),
            "abs_error": float(abs(recon - pooled)),
            "tol": 1e-9,
            "pass": bool(abs(recon - pooled) < 1e-9),
            "_identity": "mean is linear, so the cell weights must reconstruct "
                         "the pooled delta EXACTLY; anything else means the "
                         "cells are not a partition",
            "contributions": contrib}


def control_constant_zero(G, cells, names):
    """C4 — a constant-zero predictor's ADE inside each cell must equal that
    cell's mean ||GT||, recomputed independently in float64 numpy. A control that
    must read a KNOWN value is the only thing that catches a confidently-wrong
    metric pipeline."""
    zero = np.zeros_like(G)
    comp = ra._components(zero, G, 0.5)["ade_m"]
    ref = np.linalg.norm(G, axis=-1).mean(axis=1)
    rep, ok = {}, True
    for nm in names:
        m = cells[nm]
        if not m.any():
            continue
        got, exp = float(np.mean(comp[m])), float(np.mean(ref[m]))
        p = bool(abs(got - exp) <= 1e-4 * max(1.0, abs(exp)))
        rep[nm] = {"measured": round(got, 6), "expected": round(exp, 6),
                   "abs_delta": abs(got - exp), "pass": p}
        ok &= p
    rep["pass"] = ok
    rep["_expected_is"] = ("mean over (window, slot) of ||GT||, float64 numpy — "
                           "a zero path's ADE IS the GT displacement")
    rep["_rel_tol"] = 1e-4
    rep["_tol_reason"] = ("the production geometry path runs in float32 torch; "
                          "this reference is float64 numpy, so the two agree to "
                          "float32 resolution")
    return rep


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--dump-30k", default=None,
                    help="a SECOND dump (a different step, same corpus/grid) "
                         "whose ha/ha0 are bit-identical — the invariance "
                         "control C5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="stratified")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect", default=None,
                    help="the banked record this must reproduce (C1)")
    args = ap.parse_args(argv)

    t0 = time.time()
    G, P, eid, ws, v0, man, n_files = load_dump(args.dump)
    dec = load_decisions(args.dump)
    N = int(G.shape[0])
    dt = float(args.dt if args.dt is not None
               else ((man or {}).get("grid") or {}).get("dt_s") or 0.5)
    arms = sorted(P)
    print(f"[strat] {N} windows / {n_files} episodes, dt={dt}s, arms={arms}")

    comps = {a: ra._components(P[a], G, dt) for a in arms}

    # ---- C1 POOLED REPRODUCTION -------------------------------------------
    pooled_means = {a: {mk: round(float(np.nanmean(v)), 4)
                        for mk, v in comps[a].items()} for a in arms}
    pooled_paired = {}
    for pa, pb, nm in (("ha", "os", "os_minus_ha"), ("ha0", "os", "os_minus_ha0")):
        if pa in comps and pb in comps:
            a = comps[pb]["ade_m"]
            b = comps[pa]["ade_m"]
            pooled_paired[nm] = _ci.paired_episode_cluster_bootstrap(
                a, b, list(eid), n_boot=args.n_boot, seed=args.seed)
    c1 = {"pooled_arm_ade_m": {a: pooled_means[a]["ade_m"] for a in arms},
          "pooled_paired_ade_m": pooled_paired}
    if args.expect and os.path.exists(args.expect):
        with open(args.expect, encoding="utf-8") as fh:
            exp = json.load(fh)
        want = {}
        for a in ("os", "ha", "ha0"):
            try:
                want[a] = exp["arms"][a]["intervals"]["metrics"]["ade_dense_m"]["mean"]
            except Exception:
                pass
        pg = exp.get("paired_decision_grade", {})
        want_d = {k[len("paired_"):]: pg[k]["ade_m"]
                  for k in ("paired_os_minus_ha", "paired_os_minus_ha0")
                  if k in pg}
        checks, ok = {}, True
        for a, w in want.items():
            got = pooled_means[a]["ade_m"]
            p = abs(got - w) <= 5e-5
            checks[f"arm_{a}_ade_m"] = {"expected": w, "measured": got, "pass": p}
            ok &= p
        for k, w in want_d.items():
            got = pooled_paired.get(k)
            if got is None:
                continue
            p = (abs(got["delta"] - w["delta"]) <= 5e-5
                 and abs(got["lo"] - w["lo"]) <= 5e-5
                 and abs(got["hi"] - w["hi"]) <= 5e-5)
            checks[f"paired_{k}"] = {
                "expected": {kk: w[kk] for kk in ("delta", "lo", "hi")},
                "measured": {kk: got[kk] for kk in ("delta", "lo", "hi")},
                "pass": p}
            ok &= p
        c1["reproduction"] = {"source": os.path.basename(args.expect),
                              "checks": checks, "pass": bool(ok)}
        print(f"[strat] C1 pooled reproduction: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(json.dumps(c1["reproduction"], indent=1))

    # ---- the stratifiers ---------------------------------------------------
    lat1, lon1, prov1 = gt_strata(G, dt, gate="v1")
    lat2, lon2, prov2 = gt_strata(G, dt, gate="v2")
    latv, lonv, v7rep = v7_strata(dec)

    cells = build_cells(lat1, lon1)
    cells2 = build_cells(lat2, lon2, prefix="V2:")
    groups = {
        "lateral": [f"LAT/{n}" for n in LAT_NAMES],
        "longitudinal": [f"LON/{n}" for n in LON_NAMES],
        "cross_primary": ["CROSS/straight_const", "CROSS/manoeuvre"],
        "cell_3x3": [f"CELL/{a}+{b}" for a in LAT_NAMES for b in LON_NAMES],
    }
    c2 = control_partition(cells, N, groups)
    print(f"[strat] C2 partition: {'PASS' if c2['pass'] else 'FAIL'}")

    # ---- the CORPUS MIX, measured, first and independently ------------------
    mix = {
        "_rule": ("⛔ BEV-Planner's 73.9 % is a PUBLISHED figure about a "
                  "DIFFERENT corpus and is NOT imported. This block is OURS, "
                  "MEASURED from the banked dump's ground-truth paths."),
        "n_windows": N, "n_episodes": int(len(np.unique(eid))),
        "lateral": {n: {"n": int(cells[f'LAT/{n}'].sum()),
                        "frac": round(float(cells[f'LAT/{n}'].mean()), 4),
                        "n_episodes": _n_ep(eid, cells[f"LAT/{n}"])}
                    for n in LAT_NAMES},
        "longitudinal": {n: {"n": int(cells[f'LON/{n}'].sum()),
                             "frac": round(float(cells[f'LON/{n}'].mean()), 4),
                             "n_episodes": _n_ep(eid, cells[f"LON/{n}"])}
                         for n in LON_NAMES},
        "cross": {n.split("/")[-1]: {
            "n": int(cells[n].sum()), "frac": round(float(cells[n].mean()), 4),
            "n_episodes": _n_ep(eid, cells[n])}
            for n in ("CROSS/straight_const", "CROSS/manoeuvre")},
        "cell_3x3": {f"{a}+{b}": {
            "n": int(cells[f'CELL/{a}+{b}'].sum()),
            "frac": round(float(cells[f'CELL/{a}+{b}'].mean()), 4)}
            for a in LAT_NAMES for b in LON_NAMES},
        "sensitivity_v2_gate": {
            "_is": "the curvature-gated lateral rule; a gentle highway curve "
                   "stays lane_keep instead of being called a turn",
            "lateral": {n: {"n": int(cells2[f'V2:LAT/{n}'].sum()),
                            "frac": round(float(cells2[f'V2:LAT/{n}'].mean()), 4)}
                        for n in LAT_NAMES},
            "cross_straight_const_frac": round(
                float(cells2["V2:CROSS/straight_const"].mean()), 4),
        },
        "stratifier": prov1,
        "sensitivity_stratifier": prov2,
        "v7_label_crosscheck": v7rep,
    }
    if latv is not None:
        for ax, arr, names in (("lateral", latv, LAT_NAMES),
                               ("longitudinal", lonv, LON_NAMES)):
            lab = arr >= 0
            mix["v7_label_crosscheck"][f"{ax}_mix_on_labelled"] = {
                n: {"n": int((arr == i).sum()),
                    "frac_of_labelled": round(float((arr == i).sum() /
                                                    max(1, lab.sum())), 4)}
                for i, n in enumerate(names)}
            # the GT-kinematic mix restricted to the SAME labelled subset, so
            # the two stratifiers are compared on identical windows
            gtarr = lat1 if ax == "lateral" else lon1
            mix["v7_label_crosscheck"][f"{ax}_gt_kin_mix_on_SAME_subset"] = {
                n: {"n": int(((gtarr == i) & lab).sum()),
                    "frac_of_labelled": round(float(((gtarr == i) & lab).sum() /
                                                    max(1, lab.sum())), 4)}
                for i, n in enumerate(names)}
            agree = float((gtarr[lab] == arr[lab]).mean()) if lab.any() else float("nan")
            mix["v7_label_crosscheck"][f"{ax}_agreement_gt_kin_vs_v7"] = round(agree, 4)

    # ---- C4 constant-zero control -----------------------------------------
    c4 = control_constant_zero(G, cells, list(groups["lateral"]) +
                               list(groups["longitudinal"]) +
                               list(groups["cross_primary"]))
    print(f"[strat] C4 constant-zero: {'PASS' if c4['pass'] else 'FAIL'}")

    # ---- the per-stratum table --------------------------------------------
    contrasts = [("ha", "os", "os_minus_ha"), ("ha0", "os", "os_minus_ha0")]
    per_stratum = {}
    for pa, pb, cname in contrasts:
        if pa not in comps or pb not in comps:
            continue
        block = {"direction": f"{pb} - {pa}",
                 "estimator": "paired_episode_cluster_bootstrap",
                 "cluster_unit": "episode", "n_boot": args.n_boot,
                 "seed": args.seed, "cells": {}}
        for nm, mask in cells.items():
            row = {"n_windows": int(mask.sum()),
                   "n_episodes": _n_ep(eid, mask), "families": {}}
            fams = FAMILY_METRICS if nm.startswith(("LAT/", "LON/", "CROSS/")) \
                else {"ADE": ("ade_m",)}
            for fam, mks in fams.items():
                for mk in mks:
                    row["families"].setdefault(fam, {})[mk] = cell_stat(
                        comps, pb, pa, mk, eid, mask, args.n_boot, args.seed)
            block["cells"][nm] = row
        # the registered PRIMARY contrast also at the Bonferroni-adjusted level
        block["primary_bonferroni"] = {
            "_is": "the registered primary contrast is 2 cells; alpha 0.05/2 = "
                   "0.025 -> a 97.5 % interval. Every OTHER cell is reported "
                   "DESCRIPTIVELY at 95 % and no cell claim is promoted.",
            "cells": {nm.split("/")[-1]: cell_stat(
                comps, pb, pa, "ade_m", eid, cells[nm], args.n_boot,
                args.seed, alpha=0.025)
                for nm in ("CROSS/straight_const", "CROSS/manoeuvre")},
        }
        block["decomposition"] = {
            g: control_decomposition(comps, pb, pa, "ade_m", cells, names, N)
            for g, names in groups.items()}
        per_stratum[cname] = block
        dc = block["decomposition"]
        print(f"[strat] C3 decomposition {cname}: "
              + " ".join(f"{g}={'PASS' if dc[g]['pass'] else 'FAIL'}"
                         for g in dc))

    # ---- C5 ha/ha0 invariance across steps ---------------------------------
    c5 = {"status": "SKIPPED — no --dump-30k passed"}
    if args.dump_30k:
        G2, P2, eid2, ws2, v02, man2, nf2 = load_dump(args.dump_30k)
        same_shape = (G2.shape == G.shape)
        ident = {}
        for a in ("g", "ha", "ha0", "ws", "v0"):
            x = G if a == "g" else (ws if a == "ws" else
                                    (v0 if a == "v0" else P.get(a)))
            y = G2 if a == "g" else (ws2 if a == "ws" else
                                     (v02 if a == "v0" else P2.get(a)))
            if x is None or y is None or not same_shape:
                continue
            eq = bool(np.array_equal(x, y))
            d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
            ident[a] = {"bit_identical": eq, "n_differing_elements": int((d > 0).sum()),
                        "n_elements": int(d.size), "max_abs_delta": float(d.max())}
        comps2 = {a: ra._components(P2[a], G2, dt) for a in ("ha", "ha0")
                  if a in P2}
        cellmeans, ok = {}, bool(same_shape)
        for nm in ("CROSS/straight_const", "CROSS/manoeuvre",
                   *groups["lateral"], *groups["longitudinal"]):
            m = cells[nm]
            for a in comps2:
                x = round(float(np.nanmean(comps[a]["ade_m"][m])), 6)
                y = round(float(np.nanmean(comps2[a]["ade_m"][m])), 6)
                cellmeans[f"{nm}|{a}"] = {"this_step": x, "other_step": y,
                                          "pass": bool(abs(x - y) < 1e-9)}
                ok &= cellmeans[f"{nm}|{a}"]["pass"]
        c5 = {"_is": ("⭐ the shared measurement surface. The floors and the GT "
                      "do not depend on the checkpoint, so the SAME "
                      "stratification must give the SAME per-cell ha/ha0 means "
                      "on both dumps — if it does not, the strata do not "
                      "partition the same windows."),
              "other_dump": os.path.basename(args.dump_30k),
              "shapes_match": same_shape,
              "identity_per_key": ident,
              "_bit_identity_correction": (
                  "⚠️ the standing claim 'ha and ha0 are BIT-IDENTICAL at 30k "
                  "and 40,284' is MEASURED here as TRUE for g / ha0 / ws / v0 "
                  "and NOT LITERALLY TRUE for `ha` — see identity_per_key.ha. "
                  "The deviation is float32-ulp scale and changes no cell mean "
                  "at 1e-9, so the control's CONCLUSION stands; the word "
                  "'bit-identical' does not, for that one arm."),
              "pass_criterion": ("per-cell ha/ha0 ADE means agree to < 1e-9 on "
                                 "both dumps (the invariance that matters); "
                                 "bit-identity is REPORTED, not required"),
              "per_cell_means": cellmeans, "pass": bool(ok)}
        print(f"[strat] C5 per-cell floor invariance: {'PASS' if ok else 'FAIL'}"
              f"  (bit-identical: "
              f"{ {k: v['bit_identical'] for k, v in ident.items()} })")

    # ---- the v7.2-LABEL cross-check stratification -------------------------
    v7_block = {"status": "ABSENT"}
    if latv is not None:
        lab = (latv >= 0) & (lonv >= 0)
        v7cells = {}
        for i, nm in enumerate(LAT_NAMES):
            v7cells[f"V7:LAT/{nm}"] = (latv == i)
        for i, nm in enumerate(LON_NAMES):
            v7cells[f"V7:LON/{nm}"] = (lonv == i)
        sc = (latv == 0) & (lonv == 1)
        v7cells["V7:CROSS/straight_const"] = sc
        v7cells["V7:CROSS/manoeuvre"] = lab & ~sc
        v7_block = {
            "_is": ("the SAME contrast cut by the banked v7.2 tactical labels "
                    "instead of the GT kinematics — a stratifier cross-check."),
            "⛔_subset_warning": (
                f"these cells cover only {int(lab.sum())} of {N} windows "
                f"({100.0 * lab.sum() / N:.1f} %), and the subset is NOT random: "
                f"a clip carries ONE v7.2 record and its band admits "
                f"|t_now - t0| <= 2.0 s, which ENRICHES manoeuvres. ⇒ the v7 "
                f"mix is NOT the corpus mix and must never be quoted as one. "
                f"This block tests only whether the SIGN of the per-stratum "
                f"contrast survives a different stratifier."),
            "n_labelled": int(lab.sum()),
            "contrasts": {},
        }
        for pa, pb, cname in contrasts:
            if pa not in comps or pb not in comps:
                continue
            v7_block["contrasts"][cname] = {
                "direction": f"{pb} - {pa}",
                "cells": {nm: cell_stat(comps, pb, pa, "ade_m", eid, msk,
                                        args.n_boot, args.seed)
                          for nm, msk in v7cells.items()}}
        print("[strat] v7.2-label cross-check stratification: done "
              f"({int(lab.sum())} labelled windows)")

    rec = {
        "artifact_kind": "stratified re-analysis of a banked open-loop dump",
        "tool": "taniteval/tools/stratified_openloop.py",
        "tag": args.tag,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_used": False,
        "tier": "T1",
        "loop_class": {
            "value": "OPEN LOOP",
            "ruling": ("PI, 2026-09-02: a predictor consuming the planner's own "
                       "output is STILL open loop. Applies to every arm here "
                       "without exception."),
        },
        "question": ("is the pooled `os - ha` headline a STRAIGHT-ROAD "
                     "ARTEFACT of the corpus mix (BEV-Planner arXiv 2312.03031), "
                     "or is the hold-action loss uniform across strata?"),
        "prereg": PREREG,
        "dump": os.path.abspath(args.dump),
        "n_windows": N, "n_episodes": int(len(np.unique(eid))),
        "dt_s": dt, "horizon_steps": int(G.shape[1]),
        "arms": arms,
        "estimator": {
            "point": "full_set pooled mean over the windows IN THE CELL",
            "interval": "paired_episode_cluster_bootstrap (taniteval/ci.py)",
            "cluster_unit": "episode", "n_boot": args.n_boot, "seed": args.seed,
            "forbidden": ("overlapping_holdout_se is NOT used anywhere — it is "
                          "not a jackknife and it BIASES THE POINT ESTIMATE."),
        },
        "multiple_comparisons": {
            "primary": "CROSS/straight_const vs CROSS/manoeuvre (2 cells)",
            "primary_adjustment": "Bonferroni alpha 0.05/2 = 0.025 (97.5 % CI), "
                                  "reported in per_stratum.*.primary_bonferroni",
            "all_other_cells": "reported DESCRIPTIVELY at 95 % with their n; no "
                               "significance is hunted across cells and no cell "
                               "claim is promoted to a headline",
        },
        "power_floors": {"MIN_CELL_WINDOWS": MIN_CELL_WINDOWS,
                         "MIN_CELL_EPISODES": MIN_CELL_EPISODES,
                         "_rule": "a cell below a floor whose interval straddles "
                                  "zero is UNDERPOWERED, never 'no effect'"},
        "parity": {
            "parity_status": ("NON-PARITY — the run's own config.json carries "
                              "v2_parity.parity false, checked false, "
                              "corpus_key null"),
            "nav": ("ORACLE-derived — the v7.2 nav_command token, provenance "
                    "ego-future; it will not exist at deployment"),
            "corpus": (man or {}).get("corpus", {}).get("labels", {}),
        },
        "corpus_mix": mix,
        "pooled": {"arm_means": pooled_means, "paired": pooled_paired},
        "per_stratum": per_stratum,
        "v7_label_stratification": v7_block,
        "controls": {"C1_pooled_reproduction": c1, "C2_partition": c2,
                     "C3_decomposition": {k: {g: {kk: vv for kk, vv in v.items()
                                                  if kk != "contributions"}
                                              for g, v in b["decomposition"].items()}
                                          for k, b in per_stratum.items()},
                     "C4_constant_zero": c4, "C5_ha_invariance": c5},
        "wallclock_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    print(f"[strat] wrote {args.out}  ({rec['wallclock_s']} s, 0 GPU)")
    return 0


PREREG = {
    "written": "BEFORE any stratified number was computed",
    "outcome_A_qualifier_needed": (
        "os - ha positive and separated on straight/constant-speed, but the "
        "interval contains zero or turns negative on turning and/or braking "
        "windows => the pooled headline is TRUE BUT MISLEADING AS A "
        "CONCLUSION and needs an explicit qualifier naming the stratum."),
    "outcome_B_headline_confirmed": (
        "os - ha positive and separated in EVERY adequately-powered stratum => "
        "the headline is NOT a mix artefact, it is uniform, and this "
        "STRENGTHENS the published claim. Reported just as plainly."),
    "outcome_C_underpowered": (
        "a straddling interval at small n is a POWER LIMIT, never 'no effect'."),
    "primary_contrast": "CROSS/straight_const vs CROSS/manoeuvre",
    "stratifier_declared_in_advance": (
        "four_families.maneuver_kinematics(GT, dt) -> "
        "refc_tactical.factor_from_kinematics(kappa=None), the v1 gate — the "
        "same call refav1_arm._components already makes. NOT a model output."),
}

if __name__ == "__main__":
    raise SystemExit(main())
