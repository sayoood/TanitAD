"""P2 — REAL-FOOTAGE-in-the-loop closed loop: on-policy observation-OOD.

THE GATE-1 PRE-REQ P1 could not answer. P1 measured the OBSERVATION-OOD of the
real-footage source at IMPOSED static offsets (open-loop force-GT). This measures
what a CLOSED-LOOP PLANNER actually experiences: as the ego chooses its own actions
and DEVIATES ON-POLICY from the recorded path, does the real-footage source stay in
the low-OOD envelope, or does the deviation run away?

DESIGN — a minimal edit of taniteval/closedloop.py (C6-clean):
  KEEP verbatim the deployed planner + controller: strategic_policy -> tactical_policy
  -> 0.5 s pure-pursuit waypoint -> (steer, accel) [wp_to_control], and the kinematic
  bicycle. ONLY the observation SOURCE changes.
  REPLACE closedloop.py's step (c) "imagine the next latent" with:
    (c') DRIVE the ego one bicycle step in WORLD frame under the executed control;
    (c'') project the ego onto the recorded path -> arc-length s, signed lateral
          offset dlat, heading offset dpsi;
    (c''') ARC-LENGTH RE-INDEX: show the REAL recorded window whose last frame sits
          at arc-length nearest s (== slide the window start by m* frames), WARPED by
          the residual (dlat, dpsi) homography [lowood_probe.sampling_homography];
    (c'''') RE-ENCODE the warped real window -> new latent window -> re-plan.
  So the observation is ALWAYS a real frame re-indexed by arc-length (longitudinal
  OOD ~0 by construction) + a homography for the residual lateral/heading offset
  (bounded by P1's measured envelope). The loop's OWN deviation drives the OOD.

OUTPUT: per-step on-policy deviation (dlat[t], dpsi[t]); the OOD ratio it maps to via
the P1 envelope (interpolated); closed-loop ADE@2s vs GT; stratified junction vs
longitudinal. Compared to NuRec's FLAT 3.21x (which the ego pays every step regardless
of deviation). tick-0 is on-path (dlat~dpsi~0) so tick-0 obs == the real window == the
P1 baseline (self-check).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/root/taniteval")

from lowood_probe import (build_world_from_ckpt, sampling_homography,  # noqa: E402
                          SPEED_SCALE)
from driving_diagnostic import (WP_STEPS, gt_ego_waypoints,  # noqa: E402
                                net_heading_change_deg)
from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.metric_dynamics import HierarchicalGrounding  # noqa: E402
from taniteval import ci as _ci  # noqa: E402

W = 8
K = max(WP_STEPS)                 # 20 = 2 s
DT = 0.1
WHEELBASE = 2.7
LOOKAHEAD_STEP = 5                # 0.5 s pure-pursuit target (closedloop.py)
LD2_FLOOR = 0.25
STEER_CLAMP = 0.05
ACCEL_CLAMP = 3.0
SPEED_TC = 0.5
WP_IDX = [k - 1 for k in WP_STEPS]


def wp_to_control(w_look, v):
    """closedloop.py verbatim: 0.5 s ego-frame waypoint + speed -> (steer, accel)."""
    x, y = w_look[:, 0], w_look[:, 1]
    ld2 = (x * x + y * y).clamp_min(LD2_FLOOR)
    kappa = 2.0 * y / ld2
    steer = torch.atan(WHEELBASE * kappa).clamp(-STEER_CLAMP, STEER_CLAMP)
    v_target = x / (LOOKAHEAD_STEP * DT)
    accel = ((v_target - v) / SPEED_TC).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    return steer, accel


def warp_batch(fw, Hs):
    """fw [b,W,C,Hh,Ww] in [0,1]; Hs [b,3,3] per-window cam2->cam1 sampling
    homography. grid_sample border-replicate bilinear, align_corners=True.
    Per-window H (closed loop needs a different offset per window)."""
    b, Wn, C, Hh, Ww = fw.shape
    dev = fw.device
    ys, xs = torch.meshgrid(torch.arange(Hh, dtype=torch.float64, device=dev),
                            torch.arange(Ww, dtype=torch.float64, device=dev),
                            indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T      # [3, Hh*Ww]
    src = Hs.to(dev).to(torch.float64) @ P                        # [b,3,Hh*Ww]
    su = (src[:, 0] / src[:, 2]).reshape(b, Hh, Ww)
    sv = (src[:, 1] / src[:, 2]).reshape(b, Hh, Ww)
    gx = 2.0 * su / (Ww - 1) - 1.0
    gy = 2.0 * sv / (Hh - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)                         # [b,Hh,Ww,2]
    grid = grid[:, None].expand(-1, Wn, -1, -1, -1).reshape(b * Wn, Hh, Ww, 2).float()
    out = F.grid_sample(fw.reshape(b * Wn, C, Hh, Ww), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.reshape(b, Wn, C, Hh, Ww)


def _wrap(a):
    """wrap radians to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class OODMap:
    """Interpolate the P1 MEASURED envelope: |dlat|,|dpsi| -> ADE ratio to baseline."""
    def __init__(self, ci_json):
        d = json.loads(Path(ci_json).read_text())
        self.base = d["baseline_real_frames"]["mean"]
        self.lat_x = np.array([r["amount"] for r in d["conditions"]["lat"]])
        self.lat_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["lat"]])
        self.yaw_x = np.array([r["amount"] for r in d["conditions"]["yaw"]])
        self.yaw_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["yaw"]])

    def ratio(self, dlat_abs, dpsi_abs_deg):
        """Combined OOD ratio estimate: baseline * (1 + excess_lat + excess_yaw),
        excess_* = (interp_ade - base)/base clamped >=0. Marginal-additive (the P1
        sweep varied each axis alone) -> a conservative upper estimate for the
        combined offset. Returns (ratio, ade_lat, ade_yaw)."""
        al = float(np.interp(dlat_abs, self.lat_x, self.lat_y))
        ay = float(np.interp(dpsi_abs_deg, self.yaw_x, self.yaw_y))
        ex_l = max(0.0, (al - self.base) / self.base)
        ex_y = max(0.0, (ay - self.base) / self.base)
        return 1.0 + ex_l + ex_y, al, ay


@torch.no_grad()
def cl_realfootage(world, episodes, device, speed_input, ood, stride=8, batch=16,
                   max_windows=None):
    """Real-footage-in-the-loop closed loop over every stride-window.

    Returns per-window arrays: closed-loop DE@{0.5,1,1.5,2}s vs GT, peak/mean
    on-policy |dlat| & |dpsi|, terminal arc-progress, the P1-mapped per-step OOD
    ratio (mean over the 2 s), heading-change + speed for stratification, eid."""
    navfollow_cache = {}
    rows = {k: [] for k in ("de", "peak_lat", "mean_lat", "peak_yaw", "mean_yaw",
                            "ood_mean", "ood_peak", "s_progress", "s_recorded",
                            "head_deg", "speed", "eid", "dlat_traj", "dpsi_traj",
                            "tick0_lat", "tick0_yaw")}
    n_done = 0
    for ep_i, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        poses = ep.poses.float()
        T = fr.shape[0]
        starts = list(range(0, T - W - K, stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            b = len(ch)
            t0 = torch.tensor(ch)
            last = t0 + W - 1
            # recorded future path P[m]=poses[last+m], m=0..K  (world x,y,yaw)
            idx = last[:, None] + torch.arange(0, K + 1)[None]        # [b,K+1]
            Pxy = poses[idx][..., :2]                                 # [b,K+1,2]
            Pyaw = poses[idx][..., 2]                                 # [b,K+1]
            seg = (Pxy[:, 1:] - Pxy[:, :-1]).norm(dim=-1)            # [b,K]
            cum = torch.cat([torch.zeros(b, 1), seg.cumsum(-1)], 1)   # [b,K+1]
            s_rec = cum[:, -1]                                        # recorded 2 s arc-len
            # origin ego frame = poses[last]
            oyaw = poses[last, 2]
            oxy = poses[last, :2]
            # ego world state
            ex = poses[last, 0].clone(); ey = poses[last, 1].clone()
            eyaw = poses[last, 2].clone(); ev = poses[last, 3].clone()
            nav = navfollow_cache.get(b)
            if nav is None:
                nav = torch.zeros(b, dtype=torch.long, device=device)
                navfollow_cache[b] = nav
            ego_ego = torch.zeros(b, K, 2)          # ego positions in last-pose frame
            lat_t = torch.zeros(b, K); yaw_t = torch.zeros(b, K)
            ar = torch.arange(b)
            for k in range(K):
                # (c') project ego onto recorded path -> m*, dlat, dpsi
                d = (Pxy - torch.stack([ex, ey], -1)[:, None]).norm(dim=-1)  # [b,K+1]
                mstar = d.argmin(dim=1)                               # [b]
                pref = Pxy[ar, mstar]; yref = Pyaw[ar, mstar]
                dx = ex - pref[:, 0]; dy = ey - pref[:, 1]
                dlat = -torch.sin(yref) * dx + torch.cos(yref) * dy   # left +, metres
                dpsi = _wrap(eyaw - yref)                             # rad
                lat_t[:, k] = dlat; yaw_t[:, k] = dpsi
                # (c''') arc-length re-index: window = frames[t0+m* : t0+m*+W], warped
                wins = []
                for i in range(b):
                    s = int(t0[i] + mstar[i])
                    wins.append(fr[s:s + W])
                fw = torch.stack(wins).to(device)                    # [b,W,9,H,W]
                Hs = torch.stack([
                    sampling_homography(float(dlat[i]),
                                        float(math.degrees(dpsi[i])), 1.5, 0.0)
                    for i in range(b)])
                fw = warp_batch(fw, Hs)
                # (c'''') re-encode + PLAN (deployed strategic->tactical head)
                states = world.encode_window(fw)                     # [b,W,S]
                ctx = world.strategic_policy(states, nav)["ctx"]
                wp = world.tactical_policy(states, ctx)["waypoints"]
                w_look = wp[LOOKAHEAD_STEP].cpu()                     # [b,2] ego-frame (CPU: ego integ)
                steer, accel = wp_to_control(w_look, ev)
                # DRIVE one bicycle step in WORLD frame
                ex = ex + ev * torch.cos(eyaw) * DT
                ey = ey + ev * torch.sin(eyaw) * DT
                eyaw = eyaw + ev / WHEELBASE * torch.tan(steer) * DT
                ev = (ev + accel * DT).clamp_min(0.0)
                # ego position (post-step) in the last-pose ego frame (x fwd, y left)
                wdx = ex - oxy[:, 0]; wdy = ey - oxy[:, 1]
                xf = torch.cos(oyaw) * wdx + torch.sin(oyaw) * wdy
                yl = -torch.sin(oyaw) * wdx + torch.cos(oyaw) * wdy
                ego_ego[:, k, 0] = xf.cpu(); ego_ego[:, k, 1] = yl.cpu()
            # closed-loop waypoints at {5,10,15,20} vs GT
            pred = ego_ego[:, WP_IDX]                                # [b,4,2]
            gt = gt_ego_waypoints(poses, last)                       # [b,4,2]
            de = torch.linalg.norm(pred - gt, dim=-1)                # [b,4]
            lat_abs = lat_t.abs(); yaw_abs_deg = yaw_t.abs() * 180 / math.pi
            # P1-mapped OOD per window: mean & peak over the 2 s
            ood_mean = torch.zeros(b); ood_peak = torch.zeros(b)
            for i in range(b):
                rr = [ood.ratio(float(lat_abs[i, k]), float(yaw_abs_deg[i, k]))[0]
                      for k in range(K)]
                ood_mean[i] = float(np.mean(rr)); ood_peak[i] = float(np.max(rr))
            # terminal ego arc-progress (how far the ego actually drove)
            s_prog = torch.linalg.norm(
                ego_ego[:, -1] - torch.zeros(b, 2), dim=-1)          # crude fwd dist
            hd = net_heading_change_deg(poses, last)
            rows["de"].append(de)
            rows["peak_lat"].append(lat_abs.max(1).values)
            rows["mean_lat"].append(lat_abs.mean(1))
            rows["peak_yaw"].append(yaw_abs_deg.max(1).values)
            rows["mean_yaw"].append(yaw_abs_deg.mean(1))
            rows["ood_mean"].append(ood_mean); rows["ood_peak"].append(ood_peak)
            rows["s_progress"].append(s_prog); rows["s_recorded"].append(s_rec)
            rows["head_deg"].append(hd if torch.is_tensor(hd) else torch.tensor(hd))
            rows["speed"].append(poses[last, 3])
            rows["tick0_lat"].append(lat_abs[:, 0]); rows["tick0_yaw"].append(yaw_abs_deg[:, 0])
            rows["dlat_traj"].append(lat_t); rows["dpsi_traj"].append(yaw_t * 180 / math.pi)
            rows["eid"].extend([str(ep_i)] * b)
            n_done += b
            if max_windows and n_done >= max_windows:
                break
        if max_windows and n_done >= max_windows:
            break
    out = {}
    for k, v in rows.items():
        if k == "eid":
            out[k] = v
        elif k in ("de", "dlat_traj", "dpsi_traj"):
            out[k] = torch.cat(v)
        else:
            out[k] = torch.cat(v)
    return out


def _boot(x, eid):
    return _ci.episode_cluster_bootstrap(np.asarray(x, float), eid, n_boot=2000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/models/flagship-30k/ckpt.pt")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--p1-json", default="/workspace/lowood_flagship_ci.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world, speed_input, _ = build_world_from_ckpt(flagship4b_config(), ck)
    world = world.to(device).eval()
    step = int(ck.get("step", -1))
    print(f"[p2] ckpt step {step} speed_input {speed_input} dev {device}", flush=True)
    ood = OODMap(args.p1_json)
    print(f"[p2] P1 envelope baseline={ood.base:.4f}", flush=True)

    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps]
    print(f"[p2] {len(episodes)} val episodes", flush=True)

    with strict_numerics():
        r = cl_realfootage(world, episodes, device, speed_input, ood,
                           stride=args.stride, batch=args.batch,
                           max_windows=(args.max_windows or None))

    eid = r["eid"]
    de = r["de"]                                        # [N,4]
    ade = de.mean(1).numpy()                            # [N] closed-loop ADE 0-2s
    n = de.shape[0]
    # tick-0 self-check: obs on-path -> deviation ~0
    tick0 = {"max_lat_m": round(float(r["tick0_lat"].max()), 4),
             "max_yaw_deg": round(float(r["tick0_yaw"].max()), 4)}
    hd = r["head_deg"].numpy(); spd = r["speed"].numpy()
    junc = np.abs(hd) >= args.junction_deg
    long_ = (~junc) & (spd >= np.median(spd))
    def blk(mask):
        m = np.flatnonzero(mask)
        if not len(m):
            return None
        e = [eid[i] for i in m]
        return {
            "n": int(len(m)),
            "closed_ade2s": _boot(ade[m], e),
            "peak_lat_m": _boot(r["peak_lat"].numpy()[m], e),
            "peak_yaw_deg": _boot(r["peak_yaw"].numpy()[m], e),
            "mean_lat_m": round(float(r["mean_lat"].numpy()[m].mean()), 4),
            "mean_yaw_deg": round(float(r["mean_yaw"].numpy()[m].mean()), 4),
            "ood_mean_ratio": _boot(r["ood_mean"].numpy()[m], e),
            "ood_peak_ratio": _boot(r["ood_peak"].numpy()[m], e),
            "frac_windows_ood_under_1p16": round(
                float((r["ood_peak"].numpy()[m] <= 1.16).mean()), 4),
            "frac_windows_ood_under_1p5": round(
                float((r["ood_peak"].numpy()[m] <= 1.5).mean()), 4),
        }
    res = {
        "ckpt": args.ckpt, "step": step, "n_windows": n,
        "n_episodes": len(set(eid)), "speed_input": speed_input,
        "p1_baseline_ade2s": ood.base,
        "nurec_flat_ood_ratio": round(1.5157 / ood.base, 3),
        "_design": "real-footage-in-the-loop: deployed strategic->tactical->pure-"
                   "pursuit->bicycle (closedloop.py verbatim); obs = arc-length "
                   "re-indexed REAL window warped by on-policy (dlat,dpsi); OOD "
                   "ratio via P1 envelope interp (marginal-additive, conservative).",
        "tick0_selfcheck_onpath_dev": tick0,
        "overall": {
            "closed_ade2s": _boot(ade, eid),
            "peak_lat_m": _boot(r["peak_lat"].numpy(), eid),
            "peak_yaw_deg": _boot(r["peak_yaw"].numpy(), eid),
            "ood_mean_ratio": _boot(r["ood_mean"].numpy(), eid),
            "ood_peak_ratio": _boot(r["ood_peak"].numpy(), eid),
            "frac_windows_ood_peak_under_1p16": round(
                float((r["ood_peak"].numpy() <= 1.16).mean()), 4),
            "frac_windows_ood_peak_under_1p5": round(
                float((r["ood_peak"].numpy() <= 1.5).mean()), 4),
            "mean_dlat_traj_m": [round(float(x), 4) for x in r["dlat_traj"].mean(0)],
            "mean_dpsi_traj_deg": [round(float(x), 4) for x in r["dpsi_traj"].mean(0)],
        },
        "by_scene": {"junction": blk(junc), "longitudinal": blk(long_)},
        "junction_deg_threshold": args.junction_deg,
    }
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    o = res["overall"]
    print(f"[p2] n={n} n_ep={res['n_episodes']} tick0_dev={tick0}", flush=True)
    print(f"[p2] closed_ade2s={o['closed_ade2s']['mean']:.3f}"
          f"[{o['closed_ade2s']['lo']:.3f},{o['closed_ade2s']['hi']:.3f}] "
          f"peak_lat={o['peak_lat_m']['mean']:.3f}m peak_yaw={o['peak_yaw_deg']['mean']:.2f}deg",
          flush=True)
    print(f"[p2] OOD peak ratio={o['ood_peak_ratio']['mean']:.3f}"
          f"[{o['ood_peak_ratio']['lo']:.3f},{o['ood_peak_ratio']['hi']:.3f}] "
          f"vs NuRec flat {res['nurec_flat_ood_ratio']:.2f}x | "
          f"frac windows peak-OOD<=1.16: {o['frac_windows_ood_peak_under_1p16']:.2f} "
          f"<=1.5: {o['frac_windows_ood_peak_under_1p5']:.2f}", flush=True)
    print(f"[p2] wrote {args.out}", flush=True)
    print("LOWOOD_CL_DONE", flush=True)


if __name__ == "__main__":
    main()
