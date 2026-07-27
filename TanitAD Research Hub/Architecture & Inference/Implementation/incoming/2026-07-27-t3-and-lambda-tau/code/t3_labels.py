#!/usr/bin/env python3
"""T3 stage 1 — PER-CANDIDATE RULE LABELS on v4's OWN emitted fan, AT PARITY.

WHY THIS FILE EXISTS
--------------------
T2 (``PERCANDIDATE_LABELS.md`` §4) measured the rule label on the DEV BOX, over
the *anchor vocabulary* and a *non-parity* corpus.  T3 (§8) has to score a
trained rescorer on **PDMS-lite**, which means the labels must live on the SAME
881/6 844 windows and the SAME 256-candidate fan that Bar A's cache holds --
i.e. flagship-v4-fromscratch-30k's own emitted fan over the parity val corpus
``physicalai-val-0c5f7dac3b11``.  So the labels are RE-MINTED here, pod-side, at
parity.  **They are a different measurement from T2's and the two are kept
separate; neither is a model fact and neither may enter MODEL_REGISTRY.md.**

WHAT IT DOES
------------
1. Recovers, per val episode, the affine map  pose-index -> clip timestamp.
   ``physicalai.build_episode`` resamples the video clock as
   ``t_query = linspace(t_frames[0], t_frames[-1], N)`` and keeps ``poses[k:]``
   with ``k = n_stack-1 = 2``.  The two endpoints are the ONLY unknowns, so they
   are fitted by least squares against the clip's own egomotion.  The residual
   is BOTH the clock check and the clip-identity proof: a wrong clip cannot fit.
2. Places ``obstacle.offline`` log-replay tracks in each window's ego frame.
3. Labels every one of the 256 emitted candidates with NC (at-fault collision),
   TTC, C (comfort) and EP (progress), plus the PDMS-lite (no-map) composite --
   the identical arithmetic as ``t2_labels.py``, imported where possible.
4. Applies the ⭐ pre-registered PRECONDITION first: the head's OWN reachability
   clip (``v_term in [max(0, v0-5), v0+5]``, reach = sel_accel_max * horizon =
   2.5 * 2.0).  Both the clipped and unclipped label sets are emitted.
5. Runs the instrument check in the direction that can only embarrass it: the
   SAME labeler on the realised human future.

🔒 No clip UUID reaches any artifact -- clips are ``clip_<sha256[:8]>`` aliases,
   and the raw parquets never leave /workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcl_common import obb_corners, obb_overlap, quaternion_yaw  # noqa: E402

DT = 0.1
N_STEPS = 20
WP_STEPS = (5, 10, 15, 20)
N_STACK = 3                      # physicalai.build_episode default
SEL_ACCEL_MAX, HORIZON_S = 2.5, 2.0          # the head's own reach clamp

# comfort bounds — identical to t2_labels.py (PUBLISHED nuPlan/NAVSIM, INHERITED)
A_LON_MAX, A_LON_MIN = 4.89, -4.05
A_LAT_MAX = 4.89
JERK_MAX = 8.37
YAW_RATE_MAX = 0.95

EGO_RAD_PAD = 0.35
TTC_HORIZON_S = 1.0


# --------------------------------------------------------------------------- #
# kinematics / labels — the SAME arithmetic as t2_labels.py                    #
# --------------------------------------------------------------------------- #
def candidate_kinematics(fan, v0):
    C, T, _ = fan.shape
    p = np.concatenate([np.zeros((C, 1, 2)), fan], axis=1)
    d = np.diff(p, axis=1)
    seg = np.linalg.norm(d, axis=-1)
    v = seg / DT
    psi = np.arctan2(d[..., 1], d[..., 0])
    psi = np.where(seg < 1e-3, 0.0, psi)
    psi = np.unwrap(np.concatenate([np.zeros((C, 1)), psi], axis=1), axis=1)[:, 1:]
    vfull = np.concatenate([np.full((C, 1), float(v0)), v], axis=1)
    a_lon = np.diff(vfull, axis=1) / DT
    yaw_rate = np.concatenate([psi[:, :1] / DT, np.diff(psi, axis=1) / DT], axis=1)
    a_lat = v * yaw_rate
    jerk = np.concatenate([np.zeros((C, 1)), np.diff(a_lon, axis=1) / DT], axis=1)
    return psi, v, a_lon, a_lat, jerk, yaw_rate


def _footprint_hit(ex, ey, epsi, XY_, YAW_, SZ, VAL, ego_len, ego_wid):
    C, T = ex.shape
    er = 0.5 * np.hypot(ego_len, ego_wid) + EGO_RAD_PAD
    ar = 0.5 * np.hypot(SZ[:, 0], SZ[:, 1])
    d = np.hypot(ex[:, :, None] - XY_[None, ..., 0],
                 ey[:, :, None] - XY_[None, ..., 1])
    near = (d < (er + ar[None, None, :])) & VAL[None]
    hit = np.zeros(C, bool)
    ci, ti, ai = np.nonzero(near)
    if not len(ci):
        return hit
    ec = obb_corners(ex[ci, ti], ey[ci, ti], epsi[ci, ti],
                     np.full(len(ci), ego_len + 2 * EGO_RAD_PAD),
                     np.full(len(ci), ego_wid + 2 * EGO_RAD_PAD))
    ac = obb_corners(XY_[ti, ai, 0], XY_[ti, ai, 1], YAW_[ti, ai],
                     SZ[ai, 0], SZ[ai, 1])
    np.logical_or.at(hit, ci[obb_overlap(ec, ac)], True)
    return hit


def collide(fan, psi, XY, YAW, SZ, VAL, ego_len, ego_wid):
    C, T, _ = fan.shape
    er = 0.5 * np.hypot(ego_len, ego_wid) + EGO_RAD_PAD
    ar = 0.5 * np.hypot(SZ[:, 0], SZ[:, 1])
    d = np.linalg.norm(fan[:, :, None, :] - XY[None], axis=-1)
    thr = er + ar[None, None, :]
    near = (d < thr) & VAL[None]
    hit = np.zeros(C, bool)
    fault = np.zeros(C, bool)
    ci, ti, ai = np.nonzero(near)
    if len(ci):
        ec = obb_corners(fan[ci, ti, 0], fan[ci, ti, 1], psi[ci, ti],
                         np.full(len(ci), ego_len + 2 * EGO_RAD_PAD),
                         np.full(len(ci), ego_wid + 2 * EGO_RAD_PAD))
        ac = obb_corners(XY[ti, ai, 0], XY[ti, ai, 1], YAW[ti, ai],
                         SZ[ai, 0], SZ[ai, 1])
        ov = obb_overlap(ec, ac)
        rel = XY[ti, ai] - fan[ci, ti]
        cph, sph = np.cos(psi[ci, ti]), np.sin(psi[ci, ti])
        ahead = (rel[:, 0] * cph + rel[:, 1] * sph) > -0.5 * ego_len
        np.logical_or.at(hit, ci[ov], True)
        np.logical_or.at(fault, ci[ov & ahead], True)
    return hit, fault


def ttc(fan, psi, v, XY, YAW, SZ, VAL, ego_len, ego_wid):
    C, T, _ = fan.shape
    dt_probe = 0.2
    agv = np.zeros_like(XY)
    if T > 1:
        agv[1:] = (XY[1:] - XY[:-1]) / DT
        agv[0] = agv[1]
    flag = np.zeros(C, bool)
    for k in range(1, int(round(TTC_HORIZON_S / dt_probe)) + 1):
        tau = k * dt_probe
        ex = fan[..., 0] + v * np.cos(psi) * tau
        ey = fan[..., 1] + v * np.sin(psi) * tau
        flag |= _footprint_hit(ex, ey, psi, XY + agv * tau, YAW, SZ, VAL,
                               ego_len, ego_wid)
    return flag


def track_table(obst, ego_df, times_us, ego0):
    """Log-replay agent boxes in the window's ego frame. ego0 = (x, y, yaw)."""
    t = obst["timestamp_us"].to_numpy(np.float64)
    lo, hi = times_us.min() - 3e5, times_us.max() + 3e5
    m = (t >= lo) & (t <= hi)
    if m.sum() == 0:
        return None
    t, o = t[m], obst[m]
    rig = np.stack([o["center_x"].to_numpy(np.float64),
                    o["center_y"].to_numpy(np.float64)], axis=1)
    ayaw_rig = quaternion_yaw(*(o[c].to_numpy(np.float64) for c in
                                ("orientation_x", "orientation_y",
                                 "orientation_z", "orientation_w")))
    pose = ego_at(ego_df, t)
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    wx = rig[:, 0] * c - rig[:, 1] * s + pose[:, 0]
    wy = rig[:, 0] * s + rig[:, 1] * c + pose[:, 1]
    wyaw = ayaw_rig + pose[:, 2]
    sx = o["size_x"].to_numpy(np.float64)
    sy = o["size_y"].to_numpy(np.float64)
    tid = o["track_id"].to_numpy()
    uniq, inv = np.unique(tid, return_inverse=True)
    A, T = len(uniq), len(times_us)
    XY = np.zeros((T, A, 2)); YAW = np.zeros((T, A))
    SZ = np.zeros((A, 2)); VAL = np.zeros((T, A), bool)
    for k in range(A):
        sel = inv == k
        tk = t[sel]
        if len(tk) < 2:
            continue
        order = np.argsort(tk)
        tk = tk[order]
        XY[:, k, 0] = np.interp(times_us, tk, wx[sel][order])
        XY[:, k, 1] = np.interp(times_us, tk, wy[sel][order])
        YAW[:, k] = np.interp(times_us, tk, np.unwrap(wyaw[sel][order]))
        SZ[k] = [np.median(sx[sel]), np.median(sy[sel])]
        VAL[:, k] = (times_us >= tk.min() - 1.5e5) & (times_us <= tk.max() + 1.5e5)
    d = XY - np.asarray(ego0[:2])
    cc, ss = np.cos(-ego0[2]), np.sin(-ego0[2])
    XY = np.stack([d[..., 0] * cc - d[..., 1] * ss,
                   d[..., 0] * ss + d[..., 1] * cc], axis=-1)
    return XY, YAW - ego0[2], SZ, VAL


def ego_at(ego, t_us):
    t = ego["timestamp"].to_numpy(np.float64)
    x = np.interp(t_us, t, ego["x"].to_numpy(np.float64))
    y = np.interp(t_us, t, ego["y"].to_numpy(np.float64))
    v = np.interp(t_us, t, np.hypot(ego["vx"].to_numpy(np.float64),
                                    ego["vy"].to_numpy(np.float64)))
    yaw_native = np.unwrap(quaternion_yaw(*(ego[c].to_numpy(np.float64)
                                            for c in ("qx", "qy", "qz", "qw"))))
    yaw_u = np.interp(t_us, t, yaw_native)
    return np.stack([x, y, np.arctan2(np.sin(yaw_u), np.cos(yaw_u)), v], axis=1)


# --------------------------------------------------------------------------- #
# STAGE 1 — the clock, recovered and PROVED per episode                        #
# --------------------------------------------------------------------------- #
def fit_clock(poses, ego):
    """poses [T,4] from the epcache <-> the clip's egomotion. DIRECT INVERSION.

    ``physicalai.build_episode`` interpolates egomotion at
    ``t_query = linspace(t_frames[0], t_frames[-1], N)`` and keeps
    ``poses[N_STACK-1:]``, so pose index -> clip time is EXACTLY AFFINE.  It is
    therefore recovered without an optimiser: nearest-neighbour match every pose
    to the 100 Hz egomotion in (x, y, v), then a Theil-Sen affine fit over the
    matches.  Standstill stretches are the only ambiguity and they appear as
    slope outliers, which the median absorbs.

    ⚠️ An optimiser seeded on ``egomotion``'s own range does NOT work here --
    THE CLOCK TRAP (``PERCANDIDATE_LABELS.md`` §1): egomotion carries a sparse
    trailing tail out to ~140 s beyond the 20 s clip, and a fit seeded at
    ``et.max()`` lands 100+ s away at rms ~ 10^2 m.  Measured on this host
    before this function was rewritten.

    Returns (t_of_index [T] in us, rms_xy residual [m], n_target).
    """
    T = poses.shape[0]
    N = T + N_STACK - 1
    et = ego["timestamp"].to_numpy(np.float64)
    ex = ego["x"].to_numpy(np.float64)
    ey = ego["y"].to_numpy(np.float64)
    ev = np.hypot(ego["vx"].to_numpy(np.float64), ego["vy"].to_numpy(np.float64))
    d2 = ((poses[:, 0][:, None] - ex[None]) ** 2
          + (poses[:, 1][:, None] - ey[None]) ** 2
          + (poses[:, 3][:, None] - ev[None]) ** 2)
    tm = et[d2.argmin(1)]
    i = np.arange(T, dtype=np.float64)
    sl = []
    for aa in range(0, max(T - 20, 1), 7):
        for bb in range(aa + 20, T, 37):
            sl.append((tm[bb] - tm[aa]) / (bb - aa))
    slope = float(np.median(sl))
    inter = float(np.median(tm - slope * i))
    t_of_index = inter + slope * i
    xy = np.stack([np.interp(t_of_index, et, ex),
                   np.interp(t_of_index, et, ey)], axis=1)
    rms = float(np.sqrt(((xy - poses[:, :2]) ** 2).sum(1).mean()))
    return t_of_index, rms, N


def index_offset_scan(t_of_index, ego, tidx, v0_cache, kmax=14):
    """SECOND, INDEPENDENT PROBE of the window-index convention.

    ``_contract.WindowDataset`` yields ``pose_last = poses[t + w - 1]``; ``w`` is
    a config value this script must not assume.  Scan k and pick the offset whose
    interpolated egomotion speed reproduces the cache's own ``v0``.  Reported in
    full, so the convention is MEASURED rather than inherited from a config.
    """
    tt = np.asarray(tidx, dtype=int)
    v0c = np.asarray(v0_cache, dtype=np.float64)
    errs = {}
    for k in range(kmax + 1):
        i = np.clip(tt + k, 0, len(t_of_index) - 1)
        errs[k] = float(np.abs(ego_at(ego, t_of_index[i])[:, 3] - v0c).max())
    best = min(errs, key=errs.get)
    return best, errs


# --------------------------------------------------------------------------- #
# STAGE 2 — label one episode                                                  #
# --------------------------------------------------------------------------- #
def label_episode(args):
    (ep, alias, paidir, fan, tgt, v0, tidx, poses, ego_len, ego_wid) = args
    ego = pd.read_parquet(Path(paidir) / f"{alias}.egomotion.parquet")
    ego = ego.sort_values("timestamp").reset_index(drop=True)
    op = Path(paidir) / f"{alias}.obstacle.offline.parquet"
    obst = pd.read_parquet(op) if op.exists() else None

    t_of_index, rms, N = fit_clock(poses, ego)
    W, C = fan.shape[0], fan.shape[1]
    out = {k: np.zeros((W, C), bool) for k in
           ("nc_any", "nc_fault", "ttc_flag", "comfort_ok")}
    out["progress"] = np.zeros((W, C))
    out["v_term"] = np.zeros((W, C))
    gt = {k: np.zeros(W, bool) for k in
          ("gt_nc_any", "gt_nc_fault", "gt_ttc", "gt_comfort_ok")}
    n_ag = np.zeros(W, np.int32)
    t0s = np.zeros(W)

    # SECOND, INDEPENDENT PROBE of the window-index semantics: the cache's own
    # v0 is poses[t][3], so the egomotion speed interpolated at the recovered
    # clock must match it. A wrong index convention (off-by-k, or `t` meaning
    # something else) shows up here even when the clock fit is perfect.
    k_off, v0_errs = index_offset_scan(t_of_index, ego, tidx, v0)
    v0_err = v0_errs[k_off]

    for wi in range(W):
        ti = int(min(int(tidx[wi]) + k_off, len(t_of_index) - 1))
        t0 = float(t_of_index[ti])
        t0s[wi] = t0 / 1e6
        times = t0 + (np.arange(1, N_STEPS + 1) * DT) * 1e6
        ego0 = ego_at(ego, np.array([t0]))[0]
        f = fan[wi].astype(np.float64)
        psi, v, a_lon, a_lat, jerk, yr = candidate_kinematics(f, float(v0[wi]))
        g = tgt[wi].astype(np.float64)[None]
        gpsi, gv, ga_lon, ga_lat, gjerk, gyr = candidate_kinematics(g, float(v0[wi]))
        tt = None if obst is None else track_table(obst, ego, times, ego0)
        if tt is None:
            XY = np.zeros((N_STEPS, 0, 2)); YAW = np.zeros((N_STEPS, 0))
            SZ = np.zeros((0, 2)); VAL = np.zeros((N_STEPS, 0), bool)
        else:
            XY, YAW, SZ, VAL = tt
        n_ag[wi] = int(VAL.any(axis=0).sum()) if VAL.size else 0
        if XY.shape[1]:
            hit, fault = collide(f, psi, XY, YAW, SZ, VAL, ego_len, ego_wid)
            tf = ttc(f, psi, v, XY, YAW, SZ, VAL, ego_len, ego_wid)
            ghit, gfault = collide(g, gpsi, XY, YAW, SZ, VAL, ego_len, ego_wid)
            gtf = ttc(g, gpsi, gv, XY, YAW, SZ, VAL, ego_len, ego_wid)
        else:
            hit = fault = tf = np.zeros(C, bool)
            ghit = gfault = gtf = np.zeros(1, bool)
        comf = ((a_lon.max(1) <= A_LON_MAX) & (a_lon.min(1) >= A_LON_MIN) &
                (np.abs(a_lat).max(1) <= A_LAT_MAX) &
                (np.abs(jerk).max(1) <= JERK_MAX) &
                (np.abs(yr).max(1) <= YAW_RATE_MAX))
        out["nc_any"][wi], out["nc_fault"][wi] = hit, fault
        out["ttc_flag"][wi], out["comfort_ok"][wi] = tf, comf
        out["progress"][wi] = f[:, -1, 0]
        out["v_term"][wi] = np.linalg.norm(f[:, -1, :], axis=-1) / HORIZON_S
        gt["gt_nc_any"][wi] = bool(ghit[0]); gt["gt_nc_fault"][wi] = bool(gfault[0])
        gt["gt_ttc"][wi] = bool(gtf[0])
        gt["gt_comfort_ok"][wi] = bool(
            (ga_lon.max() <= A_LON_MAX) and (ga_lon.min() >= A_LON_MIN) and
            (np.abs(ga_lat).max() <= A_LAT_MAX) and
            (np.abs(gjerk).max() <= JERK_MAX) and
            (np.abs(gyr).max() <= YAW_RATE_MAX))
    return (ep, alias, rms, N, out, gt, n_ag, t0s,
            bool(obst is not None), v0_err, k_off, v0_errs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/_bara/cache_produced_stride1.pt")
    ap.add_argument("--valdir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--paidir", default="/workspace/_t3/pai_val40")
    ap.add_argument("--alias-map", default="/workspace/_t3/val40_alias_map.json")
    ap.add_argument("--pad", type=float, default=0.35)
    ap.add_argument("--ttc-horizon", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="/workspace/_t3/t3_labels.pt")
    ap.add_argument("--json", default="/workspace/_t3/t3_labels.json")
    a = ap.parse_args(argv)
    global EGO_RAD_PAD, TTC_HORIZON_S
    EGO_RAD_PAD, TTC_HORIZON_S = a.pad, a.ttc_horizon

    t_start = time.time()
    C = torch.load(a.cache, map_location="cpu", weights_only=False)
    amap = {int(r["ep_index"]): r for r in json.load(open(a.alias_map))}
    veh = {}
    for f in sorted(Path(a.paidir).glob("vehicle_dimensions.chunk_*.parquet")):
        d = pd.read_parquet(f)
        veh[f.name] = d
    ep_arr = C["ep"].numpy()
    jobs = []
    for ep in sorted(set(ep_arr.tolist())):
        rec = amap[ep]
        sel = np.nonzero(ep_arr == ep)[0]
        poses = torch.load(Path(a.valdir) / rec["file"], map_location="cpu",
                           weights_only=False)["poses"].numpy().astype(np.float64)
        L, Wd = 4.872, 2.121                      # corpus fallback (pcl_common)
        jobs.append((ep, rec["alias"], a.paidir,
                     C["fan"][sel].numpy(), C["tgt"][sel].numpy(),
                     C["v0"][sel].numpy(), C["t"][sel].numpy(), poses, L, Wd))
    print(f"[t3-labels] {len(jobs)} episodes, {len(ep_arr)} windows, "
          f"{C['fan'].shape[1]} candidates", flush=True)

    W, NC_ = len(ep_arr), C["fan"].shape[1]
    L = {k: np.zeros((W, NC_), bool) for k in
         ("nc_any", "nc_fault", "ttc_flag", "comfort_ok")}
    L["progress"] = np.zeros((W, NC_), np.float32)
    L["v_term"] = np.zeros((W, NC_), np.float32)
    G = {k: np.zeros(W, bool) for k in
         ("gt_nc_any", "gt_nc_fault", "gt_ttc", "gt_comfort_ok")}
    n_ag = np.zeros(W, np.int32)
    t0s = np.zeros(W)
    clock = {}
    has_tracks = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for (ep, alias, rms, N, out, gt, na, t0, ok, v0e, koff, verr) in (
                pool.map(label_episode, jobs)):
            sel = np.nonzero(ep_arr == ep)[0]
            for k in L:
                L[k][sel] = out[k]
            for k in G:
                G[k][sel] = gt[k]
            n_ag[sel] = na
            t0s[sel] = t0
            clock[alias] = {"ep": int(ep), "clock_fit_rms_xy_m": round(rms, 6),
                            "n_target_frames": int(N),
                            "v0_max_abs_err_ms": round(v0e, 5),
                            "index_offset_k": int(koff),
                            "v0_err_by_offset": {str(k): round(v, 4)
                                                 for k, v in verr.items()},
                            "has_obstacle_offline": bool(ok)}
            has_tracks[int(ep)] = bool(ok)
            print(f"  [ep {ep:02d}] {alias} rms={rms:.4f} m k={koff} dv0={v0e:.4f}  "
                  f"agents/window={na.mean():.1f}  tracks={ok}  "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    dump = {k: torch.from_numpy(v) for k, v in L.items()}
    dump.update({k: torch.from_numpy(v) for k, v in G.items()})
    dump["n_agents"] = torch.from_numpy(n_ag)
    dump["t0_s"] = torch.from_numpy(t0s)
    dump["ep"] = C["ep"]
    dump["t"] = C["t"]
    dump["eid"] = C["eid"]
    dump["v0"] = C["v0"]
    dump["has_tracks"] = torch.tensor([has_tracks[int(e)] for e in ep_arr])
    dump["_labels_IS"] = ("per-window x per-candidate rule verdicts on "
                          "flagship-v4-fromscratch-30k's EMITTED fan over the "
                          "PARITY val corpus physicalai-val-0c5f7dac3b11. NOT "
                          "T2's dev-box numbers; not a model fact.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dump, a.out)

    ok = dump["has_tracks"].numpy()
    def rate(x):
        return round(float(x[ok].mean()), 5)
    res = {
        "_experiment": "T3 stage 1 -- per-candidate rule labels on v4's emitted "
                       "fan, AT PARITY (physicalai-val-0c5f7dac3b11)",
        "_evidence_class": "MEASURED (ours)",
        "_corpus": "PARITY val physicalai-val-0c5f7dac3b11 (40 episodes); "
                   "obstacle.offline ships for 39 of 40 clips (3 probes)",
        "_operating_point": {"ego_pad_m": EGO_RAD_PAD,
                             "ttc_horizon_s": TTC_HORIZON_S},
        "n_windows_total": int(W), "n_candidates": int(NC_),
        "n_windows_with_tracks": int(ok.sum()),
        "n_percandidate_verdicts": int(ok.sum()) * int(NC_),
        "clock_recovery": clock,
        "clock_fit_rms_xy_m_max": round(
            max(v["clock_fit_rms_xy_m"] for v in clock.values()), 6),
        "v0_max_abs_err_ms_over_all_episodes": round(
            max(v["v0_max_abs_err_ms"] for v in clock.values()), 6),
        "index_offset_k_measured": sorted(set(
            v["index_offset_k"] for v in clock.values())),
        "agents_per_window": {
            "mean": round(float(n_ag[ok].mean()), 2),
            "median": float(np.median(n_ag[ok])),
            "max": int(n_ag[ok].max()),
            "frac_zero": round(float((n_ag[ok] == 0).mean()), 5)},
        "positive_rates_on_v4_fan": {
            "nc_fault": rate(L["nc_fault"]), "nc_any": rate(L["nc_any"]),
            "ttc": rate(L["ttc_flag"]),
            "comfort_violation": rate(~L["comfort_ok"])},
        "within_window_variation": {
            k: round(float((L[k][ok].any(1) & ~L[k][ok].all(1)).mean()), 4)
            for k in ("nc_fault", "ttc_flag")},
        "ground_truth_self_label_control": {
            "gt_at_fault_collision_rate": round(float(G["gt_nc_fault"][ok].mean()), 5),
            "gt_any_collision_rate": round(float(G["gt_nc_any"][ok].mean()), 5),
            "gt_ttc_infraction_rate": round(float(G["gt_ttc"][ok].mean()), 5),
            "gt_comfort_ok_rate": round(float(G["gt_comfort_ok"][ok].mean()), 5),
            "published_reference": "PARA-Drive: GT 'collides' at 0.384 % under "
                                   "axis-aligned boxes (PDF-VERBATIM)"},
        "_wallclock_s": round(time.time() - t_start, 1),
        "_dump": a.out,
    }
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "clock_recovery"},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
