"""Camera-geometry integrity audit on the REAL cached episodes (no model).

Evidence-based, per failure mode (GEOMETRY_INTEGRITY_AUDIT.md). Every number is
a measurement on the actual comma2k19 / PhysicalAI-AV episode caches used by the
flagship run — no guesses. Runs standalone (only torch/numpy + the episode
loader); the checkpoint is NOT needed for checks 1-3.

Checks:
  1a shape/channels/dtype/range per corpus (D-015 9ch, uint8, 256).
  1b pose<->frame temporal alignment: cross-correlation lag between per-step
     visual motion and pose speed (must be 0); action-sign consistency
     (steer vs yaw-rate, accel vs d speed).
  1c action scale/units: steer[rad] & accel[m/s^2] percentiles per corpus.
  1d _ego convention: on turning windows, sign(net-yaw-change)==sign(ego-y);
     on straight windows ego-x (forward) > 0. Verifies both corpora share the
     CCW-left / forward=+x handedness.
  2  intrinsics/pixels-per-metre GROUND LAW: on straight+fast pairs, the vertical
     image speed of ground rows per metre of ego forward motion (du/dd), by image
     row, per corpus. If comma and physicalai disagree, the shared f_eff=266 is a
     lie for one of them (the action->pixel geometry is inconsistent).
  3  extrinsic horizon row per corpus = the row where ground du/dd -> 0 (vertex of
     the sqrt-linear fit). comma should sit near h/2=128; a large physicalai
     offset = pitch/height/principal-point mismatch.

Usage (pod1):
  python scripts/geom_sanity.py \
      --cache-dirs /workspace/data/comma2k19/_epcache \
                   /workspace/data/physicalai/_epcache \
      --episodes 40 --out /workspace/experiments/geom_sanity.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tanitad.data.mixing import load_episode

DT = 0.1                                   # 10 Hz contract
H_EGO = 10                                 # 1 s horizon for the _ego sign check
TURN_DEG = 10.0                            # |net yaw change| that counts as a turn


def corpus_of(cd: str) -> str:
    low = cd.lower()
    if "comma" in low:
        return "comma2k19"
    if "physical" in low:
        return "physicalai"
    return Path(cd).name


def load_val_episodes(cache_dir: str, n: int) -> list:
    val = sorted(Path(cache_dir).glob("*val*"))
    assert val, f"no *val* under {cache_dir}"
    files = sorted(val[-1].glob("ep_*.pt"))[:n]
    return [load_episode(str(p), mmap=True) for p in files]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ego(dxy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([dxy[0] * c - dxy[1] * s, dxy[0] * s + dxy[1] * c])


def cur_luma(ep) -> torch.Tensor:
    """[T,256,256] float luminance of the CURRENT frame (last 3 of the 9ch stack)."""
    f = ep.frames
    cur = f[:, -3:].float()                 # D-015: current frame is last 3 chans
    return cur.mean(1)                       # simple luma


# --------------------------------------------------------------------------- #
# 1a shapes                                                                    #
# --------------------------------------------------------------------------- #
def check_shapes(ep) -> dict:
    f = ep.frames
    return {
        "frames_shape": list(f.shape), "frames_dtype": str(f.dtype),
        "frames_min": float(f.float().min()), "frames_max": float(f.float().max()),
        "channels": int(f.shape[1]), "H": int(f.shape[2]), "W": int(f.shape[3]),
        "actions_shape": list(ep.actions.shape), "poses_shape": list(ep.poses.shape),
        "T": int(f.shape[0]),
    }


# --------------------------------------------------------------------------- #
# 1b temporal alignment  (visual motion vs speed cross-correlation)           #
# --------------------------------------------------------------------------- #
def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.std() * b.std())
    return float((a * b).mean() / d) if d > 1e-12 else 0.0


def check_temporal(eps, lags=range(-3, 4)) -> dict:
    corr_by_lag = {L: [] for L in lags}
    steer_yaw, accel_dv = [], []
    dup_frac = []
    for ep in eps:
        luma = cur_luma(ep)
        T = luma.shape[0]
        if T < 12:
            continue
        vis = (luma[1:] - luma[:-1]).abs().mean(dim=(1, 2)).numpy()   # [T-1]
        spd = ep.poses[1:, 3].numpy()                                # speed@t (t=1..)
        dup_frac.append(float((vis < 1e-4).mean()))
        for L in lags:
            # correlate vis[t] with spd[t+L]
            if L >= 0:
                a, b = vis[:len(vis) - L], spd[L:]
            else:
                a, b = vis[-L:], spd[:len(spd) + L]
            m = min(len(a), len(b))
            if m > 5:
                corr_by_lag[L].append(_pearson(a[:m], b[:m]))
        # action-sign consistency
        yaw = ep.poses[:, 2].numpy()
        yawrate = _wrap(yaw[1:] - yaw[:-1])
        steer = ep.actions[:-1, 0].numpy()
        steer_yaw.append(_pearson(steer, yawrate))
        dv = np.diff(ep.poses[:, 3].numpy())
        accel = ep.actions[:-1, 1].numpy()
        accel_dv.append(_pearson(accel, dv))
    curve = {int(L): round(float(np.mean(v)), 4) for L, v in corr_by_lag.items() if v}
    best = max(curve, key=lambda k: curve[k])
    return {
        "xcorr_visualmotion_vs_speed_by_lag": curve,
        "best_lag": best, "corr_at_lag0": curve.get(0),
        "aligned_lag0": bool(best == 0),
        "steer_vs_yawrate_corr_mean": round(float(np.mean(steer_yaw)), 4),
        "accel_vs_dspeed_corr_mean": round(float(np.mean(accel_dv)), 4),
        "duplicate_frame_frac_mean": round(float(np.mean(dup_frac)), 4),
    }


# --------------------------------------------------------------------------- #
# 1c action scale                                                             #
# --------------------------------------------------------------------------- #
def check_action_scale(eps) -> dict:
    steer = np.concatenate([ep.actions[:, 0].numpy() for ep in eps])
    accel = np.concatenate([ep.actions[:, 1].numpy() for ep in eps])
    spd = np.concatenate([ep.poses[:, 3].numpy() for ep in eps])
    pct = [1, 5, 25, 50, 75, 95, 99]

    def P(x):
        return {f"p{p}": round(float(np.percentile(x, p)), 4) for p in pct}
    return {
        "steer_rad_pct": P(steer), "abs_steer_rad_pct": P(np.abs(steer)),
        "accel_mps2_pct": P(accel), "speed_mps_pct": P(spd),
        "steer_rad_std": round(float(steer.std()), 4),
        "accel_mps2_std": round(float(accel.std()), 4),
    }


# --------------------------------------------------------------------------- #
# 1d _ego convention (handedness)                                             #
# --------------------------------------------------------------------------- #
def check_ego_convention(eps) -> dict:
    turn_ok = turn_n = 0
    straight_fwd = []
    yaw_ego_pairs = []
    for ep in eps:
        P = ep.poses.numpy()
        T = P.shape[0]
        for t in range(1, T - H_EGO):
            dyaw = _wrap(P[t + H_EGO, 2] - P[t, 2])
            e = _ego(P[t + H_EGO, :2] - P[t, :2], P[t, 2])   # [fwd, left]
            if abs(math.degrees(dyaw)) > TURN_DEG:
                turn_n += 1
                if np.sign(dyaw) == np.sign(e[1]):
                    turn_ok += 1
                yaw_ego_pairs.append((dyaw, e[1]))
            elif abs(math.degrees(dyaw)) < 2.0:
                straight_fwd.append(e[0])                    # forward comp
    corr = 0.0
    if len(yaw_ego_pairs) > 5:
        dy = np.array([p[0] for p in yaw_ego_pairs])
        el = np.array([p[1] for p in yaw_ego_pairs])
        corr = _pearson(dy, el)
    return {
        "turn_windows": turn_n,
        "frac_sign(dyaw)==sign(ego_y)": round(turn_ok / max(1, turn_n), 4),
        "corr_dyaw_vs_ego_y": round(corr, 4),
        "straight_windows": len(straight_fwd),
        "mean_ego_x_forward_on_straight": round(float(np.mean(straight_fwd)), 4)
        if straight_fwd else None,
        "frac_forward_positive": round(float(np.mean(np.array(straight_fwd) > 0)), 4)
        if straight_fwd else None,
    }


# --------------------------------------------------------------------------- #
# 2 + 3  ground-flow pixels-per-metre law & horizon (Farneback + FOE solve)    #
# --------------------------------------------------------------------------- #
# The horizon is recovered from flow DIRECTIONS (focus of expansion): for pure
# forward motion every static-point flow vector (u,v) at pixel (col,row) points
# radially from the FOE (c0,r0=horizon), so u*(r-r0)-v*(c-c0)=0. Solving that
# least-squares over textured pixels of many straight+fast pairs, with iterative
# down-weighting of outliers (dynamic vehicles), gives a horizon that is immune
# to flow MAGNITUDE and largely immune to independent motion. The scale f*h then
# comes from the flow-magnitude law du/dd = (r-r0)^2/(f*h).
def _foe_and_flow(eps, rows=range(96, 208, 8), cols=slice(72, 184),
                  straight_deg_per_step=0.5, speed_min_pct=50,
                  min_brightness=45.0, max_pairs=1500):
    import cv2
    allspd = np.concatenate([ep.poses[:, 3].numpy() for ep in eps])
    spd_thr = float(np.percentile(allspd, speed_min_pct))
    yr_thr = math.radians(straight_deg_per_step)
    # FOE normal equations  A[:, (r0,c0)] = rhs   from  u*r0 - v*c0 = u*r - v*c
    ATA = np.zeros((2, 2)); ATb = np.zeros(2)
    per_row = {int(r): [] for r in rows}       # du/dd samples per absolute row
    n_pairs = n_dark = 0
    yy, xx = np.mgrid[0:256, 0:256]
    for ep in eps:
        luma = cur_luma(ep).numpy().astype(np.uint8)
        P = ep.poses.numpy(); T = luma.shape[0]
        for t in range(1, T - 1):
            yawrate = abs(_wrap(P[t + 1, 2] - P[t, 2]))
            dd = float(np.hypot(*(P[t + 1, :2] - P[t, :2])))
            if P[t, 3] < spd_thr or yawrate > yr_thr or dd < 0.1:
                continue
            a, b = luma[t], luma[t + 1]
            if a.mean() < min_brightness:
                n_dark += 1; continue
            if np.abs(a.astype(int) - b.astype(int)).mean() < 0.2:
                continue                                    # duplicate frame
            fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            u = fl[..., 0]; v = fl[..., 1]
            mag = np.hypot(u, v)
            # textured, moving, ground-region pixels only
            gx = np.abs(np.gradient(a.astype(np.float32), axis=1))
            gy = np.abs(np.gradient(a.astype(np.float32), axis=0))
            tex = (gx + gy)
            m = (mag > 0.15) & (mag < 40) & (tex > 12) & (yy > 96) & (yy < 208) \
                & (xx > 40) & (xx < 216)
            if m.sum() < 200:
                continue
            n_pairs += 1
            uu, vv, rr, cc = u[m], v[m], yy[m].astype(float), xx[m].astype(float)
            w = np.clip(np.hypot(uu, vv), 0, 8)             # magnitude weight
            # accumulate normal equations rows [uu, -vv]·[r0,c0]=uu*rr-vv*cc
            J = np.stack([uu, -vv], 1) * w[:, None]
            rhs = (uu * rr - vv * cc) * w
            ATA += J.T @ J; ATb += J.T @ rhs
            # per-row du/dd (median downward flow over central cols / dd)
            for r in rows:
                seg = v[r - 3:r + 4, cols]
                per_row[int(r)].append(float(np.median(seg)) / dd)
            if n_pairs >= max_pairs:
                break
        if n_pairs >= max_pairs:
            break
    foe = None
    if n_pairs >= 5:
        try:
            r0, c0 = np.linalg.solve(ATA, ATb)
            foe = (float(r0), float(c0))
        except np.linalg.LinAlgError:
            pass
    dudd = {int(r): round(float(np.median(v)), 4)
            for r, v in per_row.items() if len(v) >= 10}
    return {"speed_thr_mps": round(spd_thr, 3),
            "n_straight_fast_bright_pairs": n_pairs, "n_dark_skipped": n_dark,
            "horizon_row_FOE": round(foe[0], 2) if foe else None,
            "foe_col": round(foe[1], 2) if foe else None,
            "du_per_metre_by_row": dudd}


def _fit_scale(dudd: dict, r0: float) -> dict:
    """f*h from du/dd(r) = (r-r0)^2/(f*h), over rows below the horizon."""
    rr = np.array([r for r in dudd if r > r0 + 8], float)
    yy = np.array([dudd[int(r)] for r in rr], float)
    keep = yy > 0.01
    if keep.sum() < 3:
        return {"f_times_h": None, "n": int(keep.sum())}
    rr, yy = rr[keep], yy[keep]
    fh = np.median((rr - r0) ** 2 / yy)
    return {"f_times_h": round(float(fh), 1), "n": int(len(rr))}


def check_ground_flow(eps) -> dict:
    gf = _foe_and_flow(eps)
    if gf["horizon_row_FOE"] is not None:
        gf["scale_fit"] = _fit_scale(gf["du_per_metre_by_row"],
                                     gf["horizon_row_FOE"])
    return gf


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {"exp": "geom-sanity", "episodes_per_corpus": args.episodes,
              "by_corpus": {}}
    for cd in args.cache_dirs:
        corp = corpus_of(cd)
        eps = load_val_episodes(cd, args.episodes)
        print(f"[{corp}] {len(eps)} val episodes", flush=True)
        r = {"n_episodes": len(eps),
             "1a_shapes": check_shapes(eps[0]),
             "1b_temporal": check_temporal(eps),
             "1c_action_scale": check_action_scale(eps),
             "1d_ego_convention": check_ego_convention(eps)}
        print(f"[{corp}] 1b {r['1b_temporal']['best_lag']=} "
              f"corr0={r['1b_temporal']['corr_at_lag0']} "
              f"steer/yaw={r['1b_temporal']['steer_vs_yawrate_corr_mean']}", flush=True)
        print(f"[{corp}] 1d ego_y_sign="
              f"{r['1d_ego_convention']['frac_sign(dyaw)==sign(ego_y)']} "
              f"fwd+={r['1d_ego_convention']['frac_forward_positive']}", flush=True)
        gf = check_ground_flow(eps)
        r["2_3_ground_flow"] = gf
        print(f"[{corp}] ground-flow pairs={gf['n_straight_fast_bright_pairs']} "
              f"(dark_skip={gf['n_dark_skipped']}) horizon_FOE={gf.get('horizon_row_FOE')} "
              f"f*h={gf.get('scale_fit', {}).get('f_times_h')}", flush=True)
        print(f"[{corp}] du/m by row: " + ", ".join(
            f"{k}:{v}" for k, v in gf["du_per_metre_by_row"].items()), flush=True)
        report["by_corpus"][corp] = r

    # cross-corpus ratios (the headline)
    cs = list(report["by_corpus"])
    if len(cs) == 2:
        ga = report["by_corpus"][cs[0]]["2_3_ground_flow"]
        gb = report["by_corpus"][cs[1]]["2_3_ground_flow"]
        common = sorted(set(ga["du_per_metre_by_row"]) & set(gb["du_per_metre_by_row"]))
        ratios = {r: round(gb["du_per_metre_by_row"][r] / ga["du_per_metre_by_row"][r], 3)
                  for r in common if abs(ga["du_per_metre_by_row"][r]) > 0.02}
        fha = ga.get("scale_fit", {}).get("f_times_h")
        fhb = gb.get("scale_fit", {}).get("f_times_h")
        report["ground_flow_ratio"] = {
            "corpora": cs,
            f"du_per_m_ratio_by_row_{cs[1]}_over_{cs[0]}": ratios,
            "median_row_ratio": round(float(np.median(list(ratios.values()))), 3)
            if ratios else None,
            "horizon_rows_FOE": {cs[0]: ga.get("horizon_row_FOE"),
                                 cs[1]: gb.get("horizon_row_FOE")},
            "f_times_h": {cs[0]: fha, cs[1]: fhb},
            "f_times_h_ratio": round(fhb / fha, 3) if (fha and fhb) else None,
        }
        print(f"\n[HEADLINE] horizons_FOE={report['ground_flow_ratio']['horizon_rows_FOE']} "
              f"f*h={report['ground_flow_ratio']['f_times_h']} "
              f"f*h_ratio={report['ground_flow_ratio']['f_times_h_ratio']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"[geom] report -> {args.out}\nGEOM_SANITY_DONE", flush=True)


if __name__ == "__main__":
    main()
