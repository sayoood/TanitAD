#!/usr/bin/env python3
"""E-GOAL-1 — shared `obstacle.offline` reader, alignment assertions, estimators.

⭐ THIS MODULE IS THE REUSABLE PIECE. It is a thin, asserting wrapper around
`stack/scripts/lead_state_gate.py`, which already contains a **proven, strictly
causal** `obstacle.offline` reader (`ego_frame` / `lead_frame`). Nothing here
re-implements those; re-implementing them would have re-derived their bugs.

What this module ADDS over the repo reader, and why each addition exists:

  1. `assert_clock_join`  -- the repo reader joins `egomotion` and
     `obstacle.offline` on a shared clock WITHOUT ever checking it. The offset is
     recoverable from the data (a world-static object has a constant world
     position iff the offset is right) and is PROVEN here, with a deliberately
     wrong offset as a failing control.
  2. `span_of`            -- the real hazard is the SPAN, not the clock:
     `egomotion` runs 20-140 s while `obstacle.offline` stops at ~20 s. A grid
     point outside the INTERSECTION silently yields "no lead vehicle", which is
     indistinguishable from a genuinely empty road. Spans are measured per clip
     and the grid is asserted inside the intersection.
  3. `lead_extra`         -- five derived lead columns (headway, required decel,
     lead absolute speed, second-nearest gap, inverse gap). All are pure
     functions of already-causal quantities, so they add no leak surface.
  4. `assert_no_oracle`   -- runtime enforcement of PRE_REGISTRATION.md S5.

⛔ LEAKAGE (retraction class C23, ORACLE-SHAPED-AS-EGO-STATE). Every admissible
column is computable at time t from data timestamped <= t. The two traps in the
neighbouring artifacts are `y_long_h*` / `dv_h*` (FUTURE horizons written into
`lead_gate_windows_h.parquet` by a post-hoc build) and `v(t+2 s)`. They are named
in `ORACLE` and abort any arm that names them.

PARITY: read-only over `labels/*.zip` and `r0/phase0_selection.parquet`.
`_epcache` is never written, no clip is re-selected, `physicalai-train-e438721ae894`
and skip-hash `f09e44db` are untouched.

PRIVACY: PhysicalAI-AV is gated-confidential. Clip UUIDs never reach an artifact;
public JSON/parquet carry `clip_<sha256[:8]>` only.
"""
from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

from taniteval.ci import (  # noqa: E402
    episode_cluster_bootstrap,
    paired_episode_cluster_bootstrap,
)

# the repo's proven, strictly causal reader -- imported, not re-implemented
from lead_state_gate import (  # noqa: E402
    CLIP_END_S,
    DENS_COLS,
    EGO_COLS,
    HIST_S,
    HORIZON_S,
    LEAD_COLS,
    TARGET_HZ,
    ego_frame,
    lead_frame,
    quaternion_yaw,
)

ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
OUT = Path(__file__).resolve().parent.parent / "raw"

N_BOOT = 2000
SEED = 0
K_FOLDS = 5

#: bars, quoted verbatim from GOAL_INPUT.md S5 (the parent's conditional spec)
BAR_BREAKEVEN_M = 0.813        # along-track RMS, given a LEARNED cross-track
BAR_HALF_M = 0.439
BAR_BREAKEVEN_MS = 0.406       # 2 s-mean speed error
BAR_HALF_MS = 0.219
REF_HEAD_ALONG_M = 1.151       # best head to date, canonical 881 (H_ridge_all_raw)
REF_HEAD_MS = 0.576

#: extra derived lead columns (pure functions of causal quantities)
XTRA_COLS = ["headway_s", "req_decel", "lead_v_abs", "gap2_m", "inv_gap"]

ORACLE = frozenset(
    {"y_long", "y_lat", "v_fut_2s", "y_long_h1", "y_long_h2", "y_long_h3",
     "y_long_h4", "y_long_h5", "dv_h1", "dv_h2", "dv_h3", "dv_h4", "dv_h5",
     "gt", "a_gt", "head_deg", "v_target", "vt_valid", "vt_lookahead"})


def assert_no_oracle(names) -> None:
    """Abort if a fitted arm names a future-derived column."""
    bad = sorted(set(map(str, names)) & ORACLE)
    if bad:
        raise RuntimeError(f"ORACLE FIELD IN ARM INPUT: {bad}")


def clip_alias(clip_id: str) -> str:
    """Opaque, stable, non-reversible alias for a gated clip UUID."""
    return "clip_" + hashlib.sha256(clip_id.encode()).hexdigest()[:8]


def r4(x) -> float:
    return round(float(x), 4)


def sep(ci: dict) -> bool:
    return (ci["lo"] > 0) or (ci["hi"] < 0)


def ci_single(v, eid, reduce="mean", seed: int = SEED) -> dict:
    return episode_cluster_bootstrap(np.asarray(v, float), list(eid),
                                     reduce=reduce, n_boot=N_BOOT, seed=seed)


def ci_paired(a, b, eid, reduce="mean", seed: int = SEED) -> dict:
    return paired_episode_cluster_bootstrap(np.asarray(a, float),
                                            np.asarray(b, float), list(eid),
                                            reduce=reduce, n_boot=N_BOOT,
                                            seed=seed)


def n_to_separate(delta: float, half_width: float, n: int) -> float:
    """Episodes needed for a CI half-width to shrink below |delta|.

    Half-widths scale as ~n^-1/2 (MODEL_REGISTRY S1.2a measures x2.8-3.9 over a
    x15 increase in n, bracketing the x3.87 a pure n^-1/2 law predicts)."""
    if delta == 0:
        return float("inf")
    return float(n * (half_width / abs(delta)) ** 2)


# --------------------------------------------------------------------------- #
# corpus access                                                                #
# --------------------------------------------------------------------------- #
def selection() -> pd.DataFrame:
    sel = pd.read_parquet(ROOT / "r0" / "phase0_selection.parquet")
    sel["clip_id"] = sel["clip_id"].astype(str)
    return sel


def available_chunks() -> list[int]:
    d = ROOT / "labels" / "obstacle.offline"
    return sorted(int(p.name.split("_")[-1].split(".")[0])
                  for p in d.glob("obstacle.offline.chunk_*.zip"))


def _zip_map(z: zipfile.ZipFile) -> dict[str, str]:
    return {n.split(".")[0]: n for n in z.namelist() if n.endswith(".parquet")}


def iter_clips(chunk: int):
    """Yield (clip_id, ego_df, obst_df) for every clip present in BOTH zips."""
    ozp = ROOT / "labels" / "obstacle.offline" / f"obstacle.offline.chunk_{chunk:04d}.zip"
    ezp = ROOT / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip"
    if not (ozp.exists() and ezp.exists()):
        return
    with zipfile.ZipFile(ozp) as oz, zipfile.ZipFile(ezp) as ez:
        om, em = _zip_map(oz), _zip_map(ez)
        for cid in sorted(set(om) & set(em)):
            ego = pd.read_parquet(io.BytesIO(ez.read(em[cid])))
            obst = pd.read_parquet(io.BytesIO(oz.read(om[cid])))
            yield cid, ego, obst


# --------------------------------------------------------------------------- #
# 1. SPAN -- the hazard the brief names                                        #
# --------------------------------------------------------------------------- #
def span_of(ego: pd.DataFrame, obst: pd.DataFrame) -> dict:
    """Per-clip time spans in SECONDS on each source's own clock."""
    et = ego["timestamp"].to_numpy(np.float64) / 1e6
    ot = (obst["timestamp_us"].to_numpy(np.float64) / 1e6
          if len(obst) else np.array([np.nan]))
    return {"ego_t0": float(et.min()), "ego_t1": float(et.max()),
            "ego_n": int(len(et)),
            "obst_t0": float(np.nanmin(ot)), "obst_t1": float(np.nanmax(ot)),
            "obst_n": int(len(obst)),
            "n_tracks": int(obst["track_id"].nunique()) if len(obst) else 0}


def grid_inside_intersection(sp: dict, t_grid: np.ndarray,
                             tol: float = 0.05) -> bool:
    """True iff EVERY grid point lies inside [max(t0), min(t1)] of both sources."""
    lo = max(sp["ego_t0"], sp["obst_t0"]) - tol
    hi = min(sp["ego_t1"], sp["obst_t1"]) + tol
    return bool(t_grid.min() >= lo and t_grid.max() <= hi)


# --------------------------------------------------------------------------- #
# 2. CLOCK JOIN -- proven, with a failing control                              #
# --------------------------------------------------------------------------- #
def _world_dispersion(ego: pd.DataFrame, obst: pd.DataFrame,
                      delta_s: float, min_obs: int = 8) -> float:
    """p10 of per-track world-xy dispersion [m] at clock offset `delta_s`.

    `obstacle.offline` boxes are `reference_frame="rig"` (ego-relative), so a
    WORLD-STATIC object has a constant world position iff `delta_s` is the true
    offset. The statistic is taken fresh at every delta, so no track is
    pre-selected at any particular delta and the test cannot be rigged to 0.
    """
    t_us = obst["timestamp_us"].to_numpy(np.float64)
    rig = np.stack([obst["center_x"].to_numpy(np.float64),
                    obst["center_y"].to_numpy(np.float64)], 1)
    uniq, inv = np.unique(obst["track_id"].to_numpy(), return_inverse=True)
    keep = np.bincount(inv, minlength=len(uniq)) >= min_obs
    if keep.sum() < 3:
        return float("nan")

    et = ego["timestamp"].to_numpy(np.float64)
    o = np.argsort(et)
    et = et[o]
    tq = t_us + delta_s * 1e6
    ex = np.interp(tq, et, ego["x"].to_numpy(np.float64)[o])
    ey = np.interp(tq, et, ego["y"].to_numpy(np.float64)[o])
    yaw_u = np.unwrap(quaternion_yaw(*(ego[c].to_numpy(np.float64)[o]
                                       for c in ("qx", "qy", "qz", "qw"))))
    yaw = np.interp(tq, et, yaw_u)
    c, s = np.cos(yaw), np.sin(yaw)
    wx = rig[:, 0] * c - rig[:, 1] * s + ex
    wy = rig[:, 0] * s + rig[:, 1] * c + ey
    disp = [0.5 * (wx[inv == k].std() + wy[inv == k].std())
            for k in np.where(keep)[0]]
    return float(np.percentile(disp, 10))


def assert_clock_join(clips, deltas=None) -> dict:
    """Sweep the clock offset over `clips` = [(cid, ego, obst), ...].

    Returns the evidence dict AND raises if delta=0 is not the minimiser or if
    the deliberately-wrong +1.0 s control is not reported worse.
    """
    if deltas is None:
        deltas = np.round(np.arange(-2.0, 2.001, 0.1), 2)
    C = np.vstack([[_world_dispersion(e, o, d) for d in deltas]
                   for _, e, o in clips])
    med = np.nanmedian(C, axis=0)
    best = float(deltas[int(np.nanargmin(med))])
    per_clip = deltas[np.nanargmin(C, axis=1)]
    z = int(np.where(deltas == 0.0)[0][0])
    p1 = int(np.where(deltas == 1.0)[0][0])
    ev = {
        "method": ("world-frame dispersion of rig-frame tracks vs a swept clock "
                   "offset; p10 of per-track xy dispersion, recomputed fresh at "
                   "every delta so no track is pre-selected"),
        "n_clips": len(clips),
        "deltas_s": [float(d) for d in deltas],
        "median_dispersion_m": [None if np.isnan(v) else r4(v) for v in med],
        "best_delta_s": best,
        "frac_clips_best_at_zero": r4((per_clip == 0.0).mean()),
        "dispersion_at_0_m": r4(med[z]),
        "dispersion_at_plus1s_m": r4(med[p1]),
        "failing_control": {
            "what": "a deliberately wrong +1.0 s offset must be reported WORSE",
            "ratio_plus1s_over_0": r4(med[p1] / med[z]),
            "passes": bool(med[p1] > med[z])},
    }
    if best != 0.0:
        raise RuntimeError(f"CLOCK JOIN FAILED: best delta {best} s, not 0")
    if not ev["failing_control"]["passes"]:
        raise RuntimeError("CLOCK JOIN INSTRUMENT HAS NO POWER: +1 s is not worse")
    return ev


# --------------------------------------------------------------------------- #
# 3. derived lead columns -- pure functions of causal quantities               #
# --------------------------------------------------------------------------- #
def lead_extra(obs: pd.DataFrame, t_grid: np.ndarray, lf: dict,
               v: np.ndarray) -> dict:
    """Five derived columns. `lf` is `lead_state_gate.lead_frame`'s output.

    `gap2_m` (second-nearest in-corridor vehicle) is the only one needing a new
    pass over the tracks; it repeats `lead_frame`'s association EXACTLY (same
    `searchsorted(side="right")-1`, same staleness bound, same corridor), so it
    inherits the same causality guarantee.
    """
    from lead_state_gate import (LEAD_LAT_M, LEAD_MAX_GAP_M, MAX_STALE_S,
                                 VEHICLE_CLASSES)
    n = t_grid.size
    gap = lf["gap_m"]
    closing = lf["closing_ms"]
    with np.errstate(divide="ignore", invalid="ignore"):
        headway = np.where(np.isfinite(gap), gap / np.maximum(v, 0.1), np.nan)
        req_dec = np.where(np.isfinite(gap) & (closing > 0),
                           closing ** 2 / (2.0 * np.maximum(gap, 0.1)), 0.0)
        req_dec = np.where(np.isfinite(gap), req_dec, np.nan)
        lead_v = np.where(np.isfinite(gap), v - closing, np.nan)
        inv_gap = np.where(np.isfinite(gap), 1.0 / np.maximum(gap, 0.5), np.nan)

    gap2 = np.full(n, np.nan)
    if obs is not None and len(obs):
        ts = obs["timestamp_us"].to_numpy(np.float64) / 1e6
        tid = obs["track_id"].to_numpy(str)
        cx = obs["center_x"].to_numpy(np.float64)
        cy = obs["center_y"].to_numpy(np.float64)
        sx = obs["size_x"].to_numpy(np.float64)
        cls = obs["label_class"].to_numpy(str)
        best = np.full(n, np.inf)
        second = np.full(n, np.inf)
        for track in np.unique(tid):
            m = tid == track
            tt = ts[m]
            o = np.argsort(tt)
            tt = tt[o]
            if cls[m][o][0] not in VEHICLE_CLASSES:
                continue
            xx, yy, ss = cx[m][o], cy[m][o], sx[m][o]
            j = np.searchsorted(tt, t_grid, side="right") - 1
            ok = j >= 0
            jj = np.clip(j, 0, len(tt) - 1)
            stale = t_grid - tt[jj]
            ok &= (stale >= 0) & (stale <= MAX_STALE_S)
            g = xx[jj] - ss[jj] / 2.0
            cand = ok & (g >= 0.0) & (g <= LEAD_MAX_GAP_M) & (np.abs(yy[jj]) < LEAD_LAT_M)
            if not cand.any():
                continue
            new_best = cand & (g < best)
            second = np.where(new_best, best, second)
            best = np.where(new_best, g, best)
            second = np.where(cand & ~new_best & (g < second), g, second)
        gap2 = np.where(np.isfinite(second), second, np.nan)

    return {"headway_s": headway, "req_decel": req_dec, "lead_v_abs": lead_v,
            "gap2_m": gap2, "inv_gap": inv_gap}


# --------------------------------------------------------------------------- #
# 4. folds                                                                     #
# --------------------------------------------------------------------------- #
def clip_folds(clip_ids: np.ndarray, k: int = K_FOLDS, seed: int = SEED):
    """k CLIP-DISJOINT folds. Yields (train_mask, test_mask) over rows."""
    uniq = np.unique(clip_ids)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % k))
    fid = np.array([fold_of[c] for c in clip_ids])
    for f in range(k):
        te = fid == f
        yield ~te, te
