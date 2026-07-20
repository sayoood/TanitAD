"""TanitEval — decoupled LONGITUDINAL / LATERAL planning-quality metrics.

ADE conflates two independent competencies: *how good is the planned SPEED
profile* (longitudinal / along-track) and *how good is the planned PATH
geometry* (lateral / cross-track). This module decouples them, per horizon
(0.5/1/1.5/2 s) and per speed / curvature stratum — the TF++ / PerlAD
path-speed decomposition, plus the refbpatch fixed-distance (arc-length
resampled) path-geometry read.

It is a NEW module (adf3 owns hierarchy/loaders/registry/runner) — it only
*reads* loaders/data/registry and reuses the proven grounded-rollout geometry
(`rollout_decode`, `gt_ego_waypoints`, `baseline_waypoints`) verbatim, so its
predictions are byte-for-byte the same rollout the leaderboard/gate score.

Frame convention (from driving_diagnostic._ego): every waypoint is expressed in
the EGO frame at the last observed pose, where
    x = LONGITUDINAL (forward, along the initial heading)
    y = LATERAL      (cross-track, + = left).
Two decompositions are reported:
  * axis   — the fixed initial-ego axes (exact for straight windows).
  * frenet — project the per-point residual onto the GT path tangent/normal at
             each horizon (the honest along/cross split when the road curves).
For straight windows the two agree; the headline `long_frac` uses frenet.

Standalone:  python -m taniteval.pathspeed --model flagship-30k [--all]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from driving_diagnostic import (WP_STEPS, baseline_waypoints,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg)
from tanitad.models.metric_dynamics import rollout_decode  # noqa: E402
from taniteval import rollout as _rollout  # noqa: E402  (canonical ego append)

DT = 0.1                                 # 10 Hz
K_MAX = max(WP_STEPS)                    # 20 steps = 2 s
SPEED_SCALE = 10.0                       # matches every trainer (v0 = mps/10)
HORIZONS = tuple(k / 10.0 for k in WP_STEPS)     # (0.5, 1.0, 1.5, 2.0) s
WP_IDX = torch.tensor([k - 1 for k in WP_STEPS])  # into a full [.,K,.] path
EPS = 1e-6
N_RESAMPLE = 16                          # fixed-distance path-geometry samples


# ======================================================================== #
# Pure geometry — the testable core (no model, no I/O)                      #
# ======================================================================== #
def _prepend_origin(path: torch.Tensor) -> torch.Tensor:
    """[N,K,2] ego waypoints -> [N,K+1,2] with the observed pose (0,0) at t=0."""
    z = torch.zeros(path.shape[0], 1, 2, dtype=path.dtype, device=path.device)
    return torch.cat([z, path], dim=1)


def segment_tangents(path: torch.Tensor) -> torch.Tensor:
    """Unit direction of travel over each step. [N,K,2] -> [N,K,2].

    Tangent of step i = normalize(p_i - p_{i-1}) with p_0 = origin. Degenerate
    (near-stationary) segments carry the previous valid tangent forward; the
    first-step fallback is ego-forward (1,0)."""
    full = _prepend_origin(path)                        # [N,K+1,2]
    d = full[:, 1:] - full[:, :-1]                       # [N,K,2] per-step delta
    n = d.norm(dim=-1, keepdim=True)                     # [N,K,1]
    tang = torch.where(n > EPS, d / n.clamp_min(EPS),
                       torch.zeros_like(d))
    # carry the last valid tangent forward over degenerate segments
    N, K, _ = tang.shape
    fwd = torch.tensor([1.0, 0.0], dtype=path.dtype, device=path.device)
    for i in range(K):
        bad = tang[:, i].norm(dim=-1) <= EPS
        if bad.any():
            tang[bad, i] = tang[bad, i - 1] if i > 0 else fwd
    return tang


def frenet_residual(pred: torch.Tensor, gt: torch.Tensor):
    """Along-track / cross-track residual of pred vs GT, per horizon point.

    Projects (pred_i - gt_i) onto the GT path tangent (along) and its left
    normal (cross) at each step i. pred/gt [N,K,2] -> along,cross [N,K] (signed:
    along + = pred is AHEAD of GT along the path; cross + = pred is LEFT of GT).
    Orthonormal basis => along^2 + cross^2 == ||pred-gt||^2 exactly."""
    t = segment_tangents(gt)                             # [N,K,2] unit
    nvec = torch.stack([-t[..., 1], t[..., 0]], dim=-1)  # left normal
    r = pred - gt                                        # [N,K,2]
    along = (r * t).sum(-1)                              # [N,K]
    cross = (r * nvec).sum(-1)                           # [N,K]
    return along, cross


def axis_residual(pred: torch.Tensor, gt: torch.Tensor):
    """Fixed initial-ego-axis residual: long = Δx (forward), lat = Δy (left).
    Exact along/cross split for straight windows. [N,K,2] -> long,lat [N,K]."""
    r = pred - gt
    return r[..., 0], r[..., 1]


def step_speed(path: torch.Tensor) -> torch.Tensor:
    """Per-step planned speed (m/s). ||p_i - p_{i-1}|| / dt with p_0 = origin.
    [N,K,2] -> [N,K]. This IS 'planned speed at t = i*dt'."""
    full = _prepend_origin(path)
    return (full[:, 1:] - full[:, :-1]).norm(dim=-1) / DT


def step_accel(path: torch.Tensor) -> torch.Tensor:
    """Per-step longitudinal accel (m/s^2) = d(step_speed)/dt. [N,K,2] -> [N,K]
    (a_0 references the observed entry speed via the first speed sample)."""
    v = step_speed(path)                                 # [N,K]
    v_prev = torch.cat([v[:, :1], v[:, :-1]], dim=1)      # a_0 = 0 by convention
    return (v - v_prev) / DT


def arclength(path: torch.Tensor) -> torch.Tensor:
    """Cumulative arc length from the origin to each step. [N,K,2] -> [N,K+1]
    (leading 0)."""
    full = _prepend_origin(path)
    seg = (full[:, 1:] - full[:, :-1]).norm(dim=-1)       # [N,K]
    return torch.cat([torch.zeros(path.shape[0], 1, dtype=path.dtype,
                                  device=path.device),
                      seg.cumsum(dim=1)], dim=1)          # [N,K+1]


def heading_deg(path: torch.Tensor) -> torch.Tensor:
    """Cumulative heading (deg) of the path tangent at each step, relative to
    ego-forward. [N,K,2] -> [N,K]."""
    t = segment_tangents(path)
    return torch.atan2(t[..., 1], t[..., 0]) * (180.0 / math.pi)


def _resample_batch(poly: torch.Tensor, cum: torch.Tensor,
                    q: torch.Tensor) -> torch.Tensor:
    """Batched linear-interp on polylines at query arc lengths (row-wise).
    poly [N,P,2], cum [N,P] (monotone per row), q [N,m] -> [N,m,2]."""
    qc = torch.minimum(q, cum[:, -1:])                   # clamp to each row's len
    idx = torch.searchsorted(cum, qc, right=True).clamp(1, poly.shape[1] - 1)
    c0 = torch.gather(cum, 1, idx - 1)
    c1 = torch.gather(cum, 1, idx)
    w = ((qc - c0) / (c1 - c0).clamp_min(EPS)).clamp(0, 1).unsqueeze(-1)
    g0 = torch.gather(poly, 1, (idx - 1).unsqueeze(-1).expand(-1, -1, 2))
    g1 = torch.gather(poly, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    return g0 * (1 - w) + g1 * w


def path_geometry_crosstrack(pred: torch.Tensor, gt: torch.Tensor,
                             m: int = N_RESAMPLE):
    """SPEED-DECOUPLED lateral path error (refbpatch fixed-distance idea).

    Resample both paths at the SAME arc lengths d_j = j/m * min(L_pred, L_gt)
    (j=1..m) so the speed profile is factored out, then take the cross-track
    (perpendicular) deviation of pred from GT at each d_j. Two paths that trace
    the same GEOMETRY at different speeds => ~0. Fully vectorised over windows.
    pred/gt [N,K,2] -> [N] RMSE (m) + [N] common resample length (m)."""
    fp, fg = _prepend_origin(pred), _prepend_origin(gt)  # [N,K+1,2]
    cp, cg = arclength(pred), arclength(gt)              # [N,K+1]
    L = torch.minimum(cp[:, -1], cg[:, -1])             # [N] common arc length
    j = torch.arange(1, m + 1, dtype=pred.dtype, device=pred.device)
    q = L[:, None] * (j[None, :] / m)                    # [N,m]
    Gp = _resample_batch(fp, cp, q)                      # [N,m,2]
    Gg = _resample_batch(fg, cg, q)                      # [N,m,2]
    tg = torch.empty_like(Gg)                            # GT tangent per sample
    tg[:, 1:] = Gg[:, 1:] - Gg[:, :-1]
    tg[:, 0] = Gg[:, 0]                                  # from origin (0,0)
    tn = tg / tg.norm(dim=-1, keepdim=True).clamp_min(EPS)
    nvec = torch.stack([-tn[..., 1], tn[..., 0]], dim=-1)
    cross = ((Gp - Gg) * nvec).sum(-1)                   # [N,m] perp deviation
    pg = cross.pow(2).mean(dim=1).sqrt()                 # [N] RMSE
    pg = torch.where(L > EPS, pg, torch.zeros_like(pg))
    return pg, L


# ======================================================================== #
# Per-horizon metric block — model vs GT                                    #
# ======================================================================== #
def _rmse(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def _mean(x: torch.Tensor) -> float:
    return float(x.mean())


def metric_block(pred: torch.Tensor, gt: torch.Tensor, sel: torch.Tensor):
    """All decoupled long/lat metrics for the windows in `sel` (index tensor).

    pred/gt are FULL [N,K,2] ego paths. Returns a nested dict:
      per_horizon[t] = {longitudinal..., lateral..., decomposition...}
      trajectory     = path-geometry (arc-length) + profile RMSEs + speed bias
    Every error is model-vs-GT; speeds in m/s, distances in m, headings in deg.
    """
    p, g = pred[sel], gt[sel]
    n = p.shape[0]
    along, cross = frenet_residual(p, g)                 # [n,K] signed
    long_ax, lat_ax = axis_residual(p, g)                # [n,K] signed
    v_p, v_g = step_speed(p), step_speed(g)              # [n,K] m/s
    a_p, a_g = step_accel(p), step_accel(g)              # [n,K] m/s^2
    hd_p, hd_g = heading_deg(p), heading_deg(g)          # [n,K] deg
    de = (p - g).norm(dim=-1)                            # [n,K] point error

    per_h = {}
    for hi, t in zip(WP_IDX.tolist(), HORIZONS):
        a_i, c_i = along[:, hi], cross[:, hi]
        sq_a, sq_c = float(a_i.pow(2).mean()), float(c_i.pow(2).mean())
        per_h[f"{t:g}s"] = {
            # ---- LONGITUDINAL (speed / along-track) --------------------- #
            "long_rmse_m": round(_rmse(a_i), 4),
            "long_abs_m": round(_mean(a_i.abs()), 4),
            "long_bias_m": round(_mean(a_i), 4),      # + ahead / - short
            "along_axis_rmse_m": round(_rmse(long_ax[:, hi]), 4),
            "planned_speed_err_mps": round(_mean((v_p - v_g)[:, hi].abs()), 4),
            "planned_speed_bias_mps": round(_mean((v_p - v_g)[:, hi]), 4),
            "speed_profile_rmse_mps": round(_rmse((v_p - v_g)[:, :hi + 1]), 4),
            "accel_profile_rmse_mps2": round(_rmse((a_p - a_g)[:, :hi + 1]), 4),
            "gt_speed_mps": round(_mean(v_g[:, hi]), 3),
            "pred_speed_mps": round(_mean(v_p[:, hi]), 3),
            # ---- LATERAL (path geometry / cross-track) ------------------ #
            "lat_rmse_m": round(_rmse(c_i), 4),
            "lat_abs_m": round(_mean(c_i.abs()), 4),
            "lat_bias_m": round(_mean(c_i), 4),       # + left / - right
            "cross_axis_rmse_m": round(_rmse(lat_ax[:, hi]), 4),
            "heading_err_deg": round(_mean(
                _wrap_deg(hd_p[:, hi] - hd_g[:, hi]).abs()), 3),
            # ---- DECOMPOSITION ------------------------------------------ #
            "ade_to_h_m": round(_mean(de[:, :hi + 1]), 4),
            "de_at_h_m": round(_mean(de[:, hi]), 4),
            "long_frac_of_sqerr": round(sq_a / (sq_a + sq_c + EPS), 4),
        }

    # trajectory-level: fixed-distance (speed-decoupled) path geometry
    pg, Lc = path_geometry_crosstrack(p, g)
    speed_bias = (v_p - v_g)                              # [n,K]
    traj = {
        "path_geometry_crosstrack_rmse_m": round(_mean(pg), 4),
        "path_geometry_crosstrack_p90_m": round(
            float(torch.quantile(pg, 0.90)), 4),
        "resample_common_len_m": round(_mean(Lc), 3),
        "speed_profile_rmse_mps": round(_rmse(v_p - v_g), 4),
        "speed_bias_mps": round(_mean(speed_bias), 4),
        "underpredicts_speed": bool(_mean(speed_bias) < 0),
        "accel_profile_rmse_mps2": round(_rmse(a_p - a_g), 4),
        "along_track_progress_err_m": round(
            _mean((arclength(p)[:, -1] - arclength(g)[:, -1]).abs()), 4),
        "along_track_progress_bias_m": round(
            _mean(arclength(p)[:, -1] - arclength(g)[:, -1]), 4),
    }
    # headline: how much of the 2 s error is longitudinal vs lateral (frenet)
    sq_a2 = float(along[:, -1].pow(2).mean())
    sq_c2 = float(cross[:, -1].pow(2).mean())
    traj["long_frac_of_sqerr_2s"] = round(sq_a2 / (sq_a2 + sq_c2 + EPS), 4)
    traj["ade_2s_m"] = round(_mean(de), 4)
    return {"n": int(n), "per_horizon": per_h, "trajectory": traj}


def _wrap_deg(a: torch.Tensor) -> torch.Tensor:
    return (a + 180.0) % 360.0 - 180.0


# ======================================================================== #
# Rollout collection — full [N,K,2] path (reuses the gate rollout verbatim) #
# ======================================================================== #
@torch.no_grad()
def collect_full(model, step_readout, episodes, device, window=8, stride=8,
                 batch=8, speed_input=False, yaw_input=False, dyn_input=False):
    """Grounded operative rollout, keeping the FULL per-step path (not just the
    4 gate waypoints). Returns dict of [N,K,2] pred/gt/ctrv/cv/gs + [N] meta.
    Byte-identical rollout to taniteval.rollout.collect (same encode + decode)."""
    steps = tuple(range(1, K_MAX + 1))
    P, GT, CTRV, CV, GS, EID, SPD, HDG, V0 = ([] for _ in range(9))
    for ep in episodes:
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t + window])
                              for t in ch]).to(device)
            if fw.dtype == torch.uint8:
                fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16:
                fw = fw.float()
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + K_MAX]
                              for t in ch]).to(device)
            aw, fa = _rollout.append_ego(aw, fa, ep.poses, last, speed_input,
                                         yaw_input, dyn_input, device)
            states = model.encode_window(fw)
            wp_full, _ = rollout_decode(model.predictor, states, aw, fa,
                                        step_readout, K_MAX)      # [b,K,2]
            P.append(wp_full.cpu().float())
            GT.append(gt_ego_waypoints(ep.poses, last, steps))
            bw = baseline_waypoints(ep.poses, last, steps)
            CTRV.append(bw["constant_yaw_rate"])
            CV.append(bw["constant_velocity"])
            GS.append(bw["go_straight"])
            EID.extend([ep.episode_id] * len(ch))
            SPD.append(ep.poses[last, 3])                        # m/s
            HDG.append(net_heading_change_deg(ep.poses, last))
            V0.append(ep.poses[last, 3])
    return {"pred": torch.cat(P), "gt": torch.cat(GT).float(),
            "ctrv": torch.cat(CTRV).float(), "cv": torch.cat(CV).float(),
            "gs": torch.cat(GS).float(), "eid": EID,
            "speed": torch.cat(SPD).float(),
            "head_deg": torch.cat(HDG).float(),
            "v0": torch.cat(V0).float()}


# ======================================================================== #
# Stratification + baseline comparison + assembly                          #
# ======================================================================== #
def _strata_masks(col):
    """Speed + curvature strata (fast/sharp top-decile match generalization)."""
    speed, head = col["speed"], col["head_deg"]
    sp90 = torch.quantile(speed, 0.90)
    hd90 = torch.quantile(head, 0.90)
    ones = torch.ones_like(speed, dtype=torch.bool)
    masks = {
        "all": ones,
        "fast_top10pct_speed": speed >= sp90,
        "slow_bottom50pct_speed": speed <= torch.quantile(speed, 0.50),
        "sharp_top10pct_curvature": head >= hd90,
        "straight_lt5deg": head < 5.0,
    }
    # coarse speed bands (m/s) for the trend
    bands = [(0, 8), (8, 16), (16, 24), (24, 999)]
    for lo, hi in bands:
        masks[f"speed_{lo}-{hi if hi < 999 else 'inf'}mps"] = \
            (speed >= lo) & (speed < hi)
    return masks, {"speed_p90_mps": round(float(sp90), 3),
                   "curv_p90_deg": round(float(hd90), 3)}


def _floor_ade(col, sel):
    """Per-window best-of-3 kinematic floor ADE (cv / go_straight / ctrv)."""
    g = col["gt"][sel]
    de = lambda k: (col[k][sel] - g).norm(dim=-1).mean(dim=1)   # [n]
    stack = torch.stack([de("cv"), de("gs"), de("ctrv")], dim=0)  # [3,n]
    return round(float(stack.min(dim=0).values.mean()), 4)


def run(col, min_n=30):
    """Assemble the decoupled long/lat report from a collected full-path col."""
    masks, meta = _strata_masks(col)
    strata = {}
    for name, mask in masks.items():
        sel = mask.nonzero(as_tuple=True)[0]
        if len(sel) < 8:
            strata[name] = {"n": int(len(sel)), "note": "too few windows"}
            continue
        blk = {"low_confidence": bool(len(sel) < min_n)}
        blk["model"] = metric_block(col["pred"], col["gt"], sel)
        blk["ctrv"] = metric_block(col["ctrv"], col["gt"], sel)
        blk["cv"] = metric_block(col["cv"], col["gt"], sel)
        blk["floor_ade_2s_m"] = _floor_ade(col, sel)
        # headline read for this stratum
        m2, c2 = blk["model"]["trajectory"], blk["ctrv"]["trajectory"]
        mh = blk["model"]["per_horizon"]["2s"]
        blk["read"] = {
            "model_ade_2s_m": m2["ade_2s_m"],
            "ctrv_ade_2s_m": c2["ade_2s_m"],
            "long_frac_of_2s_sqerr": m2["long_frac_of_sqerr_2s"],
            "model_long_rmse_2s_m": mh["long_rmse_m"],
            "model_lat_rmse_2s_m": mh["lat_rmse_m"],
            "ctrv_long_rmse_2s_m": blk["ctrv"]["per_horizon"]["2s"]["long_rmse_m"],
            "ctrv_lat_rmse_2s_m": blk["ctrv"]["per_horizon"]["2s"]["lat_rmse_m"],
            "model_speed_bias_mps": m2["speed_bias_mps"],
            "path_geometry_rmse_m": m2["path_geometry_crosstrack_rmse_m"],
            "ctrv_path_geometry_rmse_m": c2["path_geometry_crosstrack_rmse_m"],
            "dominant_component": ("longitudinal"
                                   if m2["long_frac_of_sqerr_2s"] >= 0.6
                                   else "lateral"
                                   if m2["long_frac_of_sqerr_2s"] <= 0.4
                                   else "mixed"),
        }
        strata[name] = blk

    # compounding read (all-windows per-horizon growth)
    allm = strata["all"]["model"]["per_horizon"]
    growth = {t: allm[t]["de_at_h_m"] for t in allm}
    fast = strata.get("fast_top10pct_speed", {})
    return {
        "n_windows": len(col["eid"]),
        "n_episodes": len(set(col["eid"])),
        "strata_meta": meta,
        "horizons_s": list(HORIZONS),
        "per_horizon_de_all_m": growth,
        "compounding_ratio_2s_over_0p5s": round(
            growth["2s"] / max(growth["0.5s"], EPS), 3),
        "strata": strata,
        "headline": _headline(strata),
    }


def _headline(strata):
    fast = strata.get("fast_top10pct_speed", {})
    allw = strata.get("all", {})
    if "read" not in fast:
        return {"note": "fast stratum too small"}
    fr, ar = fast["read"], allw["read"]
    return {
        "high_speed_loss_is": fr["dominant_component"],
        "high_speed_long_frac_of_2s_sqerr": fr["long_frac_of_2s_sqerr"],
        "high_speed_model_ade_2s_m": fr["model_ade_2s_m"],
        "high_speed_ctrv_ade_2s_m": fr["ctrv_ade_2s_m"],
        "high_speed_model_long_rmse_m": fr["model_long_rmse_2s_m"],
        "high_speed_model_lat_rmse_m": fr["model_lat_rmse_2s_m"],
        "high_speed_speed_bias_mps": fr["model_speed_bias_mps"],
        "high_speed_underpredicts_speed": bool(fr["model_speed_bias_mps"] < 0),
        "overall_long_frac_of_2s_sqerr": ar["long_frac_of_2s_sqerr"],
        "verdict": (
            "high-speed loss is %s (%.0f%% of the 2 s sq-error is along-track); "
            "model %s speed by %.2f m/s vs GT; CTRV's along-track RMSE is "
            "%.2fm vs model %.2fm" % (
                fr["dominant_component"],
                100 * fr["long_frac_of_2s_sqerr"],
                "under-predicts" if fr["model_speed_bias_mps"] < 0
                else "over-predicts",
                abs(fr["model_speed_bias_mps"]),
                fr["ctrv_long_rmse_2s_m"], fr["model_long_rmse_2s_m"])),
    }


# ======================================================================== #
# Standalone entry (independent of the runner edit)                        #
# ======================================================================== #
def run_and_save(key, device="cuda", episodes=40,
                 out_dir="/root/taniteval/results"):
    import json
    import time
    from taniteval import data, loaders             # read-only (adf3 owns)
    from taniteval.registry import MODELS
    entry = [m for m in MODELS if m["key"] == key][0]
    t0 = time.time()
    L = loaders.load(entry, device)
    if not L["traj_capable"]:
        print(f"[pathspeed] {key}: no rollout head — skip", flush=True)
        return {"key": key, "skipped": "no rollout head"}
    val = "/root/valdata/physicalai-val-0c5f7dac3b11"
    files = data.list_val_episodes(val, episodes)
    if entry.get("train_ids"):                        # replicate the leak guard
        from tanitad.data.mixing import load_episode
        tid = set(Path(entry["train_ids"]).read_text().split())
        files = [f for f in files
                 if str(load_episode(str(f), mmap=True).episode_id) not in tid]
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    print(f"[pathspeed] {key}: rolling out {len(eps)} episodes...", flush=True)
    col = collect_full(L["model"], L["step_readout"], eps, device,
                       speed_input=bool(entry.get("speed_input")),
                       yaw_input=bool(entry.get("yaw_input")),
                       dyn_input=bool(entry.get("dyn_input")))
    print(f"[pathspeed] {key}: rollout done ({len(col['eid'])} windows, "
          f"{round(time.time() - t0, 1)}s); computing metrics...", flush=True)
    res = run(col)
    res["model"] = {k: entry.get(k) for k in
                    ("key", "name", "arch", "encoder", "speed_input")}
    res["ckpt_step"] = L["step"]
    res["wall_s"] = round(time.time() - t0, 1)
    res["method"] = ("decoupled longitudinal/lateral planning metrics — "
                     "ego-frame axis + GT-tangent frenet + arc-length-resampled "
                     "fixed-distance path geometry; grounded operative rollout")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"pathspeed_{key}.json").write_text(
        json.dumps(res, indent=2, default=str))
    h = res["headline"]
    fr = res["strata"]["fast_top10pct_speed"]["read"]
    print(f"[pathspeed] {key} step={L['step']} n={res['n_windows']}: "
          f"HIGH-SPEED loss is {h['high_speed_loss_is']} "
          f"(long_frac={h['high_speed_long_frac_of_2s_sqerr']}); "
          f"model long/lat RMSE@2s={fr['model_long_rmse_2s_m']}/"
          f"{fr['model_lat_rmse_2s_m']}m vs CTRV "
          f"{fr['ctrv_long_rmse_2s_m']}/{fr['ctrv_lat_rmse_2s_m']}m; "
          f"speed_bias={fr['model_speed_bias_mps']}m/s "
          f"-> pathspeed_{key}.json ({res['wall_s']}s)", flush=True)
    return res


def main():
    import argparse
    from taniteval.registry import MODELS
    ap = argparse.ArgumentParser("taniteval.pathspeed")
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    keys = ([m["key"] for m in MODELS] if a.all else [a.model])
    for key in keys:
        if not any(m["key"] == key for m in MODELS):
            print(f"[pathspeed] unknown model {key}", flush=True)
            continue
        try:
            run_and_save(key, a.device, a.episodes)
        except Exception as e:
            import traceback
            print(f"[pathspeed] {key} FAILED: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
