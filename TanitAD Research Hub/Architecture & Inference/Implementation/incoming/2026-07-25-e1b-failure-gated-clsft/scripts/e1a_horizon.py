"""E1a — CL-HORIZON-CURVE. Departure-vs-horizon on the low-OOD real-footage
closed loop. PURE MEASUREMENT: zero training, zero renderer, REF-C base only.

PRE-REGISTERED (written before the run; see E1a_E2a_RESULTS.md §PRE-REGISTRATION):
  Sweep the closed-loop rollout horizon K and report, per horizon AND per stratum
  (junction / longitudinal / other), the corridor-departure rate, the window
  departure rate, ADE@2s (horizon-invariant), peak XTE and the P1-mapped OOD ratio,
  every interval from the EPISODE-CLUSTER BOOTSTRAP (taniteval/ci.py) --- never
  overlapping_holdout_se.
  OUTCOME A (fires)  : departure/drift grows materially and SUPER-LINEARLY with K and
                       the junction stratum separates -> the 2 s instrument hid the
                       failure -> LOWOOD-CL "BOUND" is horizon-confounded (C6 owed).
  OUTCOME B (null)   : drift flat in K -> the instrument was not the limiter; the
                       BOUND verdict stands; intervention #1 deprioritised.

DESIGN NOTE --- the confound this file removes. The window set SHRINKS with K
(starts = range(0, T - W - K, stride)), so a naive K-curve confounds horizon with
window composition. Because the start set at K_max is a strict SUBSET of every
smaller K's start set, we additionally report a COMMON-START PAIRED curve on the
identical windows at every K, with the PAIRED episode-cluster bootstrap for the
(K vs K=20) deltas. That paired curve is the decision-grade read; the all-windows
curve is the secondary (it reproduces how the standing 2 s numbers were made).

HONEST BOUND (stated, not hidden). The real-footage source re-indexes along a 1-D
manifold and warps by a ground-plane homography whose OOD envelope was MEASURED only
to |dlat| <= 3.0 m / |dyaw| <= 12 deg (lowood_flagship_ci.json). Beyond that the
np.interp mapping CLAMPS, so the reported OOD ratio is a LOWER bound there. We emit
frac_steps_outside_envelope_* per horizon; any horizon whose peak OOD ratio exceeds
~1.5x, or whose steps leave the measured envelope, is EXTRAPOLATION, not measurement.

Loop body is lowood_lanekeep.py verbatim (itself a C6-clean minimal edit of
taniteval/closedloop.py); ONLY K is parameterised and more observables are recorded.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
           "/root/taniteval", "/workspace/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from driving_diagnostic import gt_ego_waypoints  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,  # noqa: E402
                               refc_xl_config)
# pod3's /root/taniteval predates ci.py (MEASURED: the package has no `ci`
# module), so the estimator is VENDORED here as a byte-copy of the repo's
# taniteval/taniteval/ci.py (md5 verified at deploy). Nothing on the pod is mutated.
import taniteval_ci as _ci  # noqa: E402

# ---- loop constants (identical to lowood_lanekeep.py / closedloop.py) --------
W = 8
DT = 0.1
WHEELBASE = 2.7
LOOKAHEAD_STEP = 5
LD2_FLOOR = 0.25
STEER_CLAMP = 0.05
ACCEL_CLAMP = 3.0
SPEED_TC = 0.5
WP_STEPS = (5, 10, 15, 20)
WP_IDX = [k - 1 for k in WP_STEPS]

F_EFF = 266.0
CXY = 128.0
_REFC_PRESETS = {"base": refc_config, "small": refc_small_config, "xl": refc_xl_config}

# envelope measurement limits (lowood_flagship_ci.json conditions)
ENV_LAT_MAX = 3.0
ENV_YAW_MAX = 12.0


def sampling_homography(dlat_m, dyaw_deg, h_cam, pitch_deg, f=F_EFF, c=CXY):
    """lowood_probe.py / lowood_lanekeep.py VERBATIM."""
    Kk = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1.0]], dtype=torch.float64)
    Ki = torch.linalg.inv(Kk)
    p = math.radians(pitch_deg)
    n = torch.tensor([0.0, math.cos(p), math.sin(p)], dtype=torch.float64)
    d = float(h_cam)
    psi = math.radians(dyaw_deg)
    Ry = torch.tensor([[math.cos(-psi), 0, math.sin(-psi)],
                       [0, 1.0, 0],
                       [-math.sin(-psi), 0, math.cos(-psi)]], dtype=torch.float64)
    Cc = torch.tensor([dlat_m, 0.0, 0.0], dtype=torch.float64)
    t = -(Ry @ Cc)
    H_1to2 = Kk @ (Ry + torch.outer(t, n) / d) @ Ki
    return torch.linalg.inv(H_1to2)


def warp_batch(fw, Hs):
    """lowood_lanekeep.py VERBATIM."""
    b, Wn, C, Hh, Ww = fw.shape
    dev = fw.device
    ys, xs = torch.meshgrid(torch.arange(Hh, dtype=torch.float64, device=dev),
                            torch.arange(Ww, dtype=torch.float64, device=dev),
                            indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T
    src = Hs.to(dev).to(torch.float64) @ P
    su = (src[:, 0] / src[:, 2]).reshape(b, Hh, Ww)
    sv = (src[:, 1] / src[:, 2]).reshape(b, Hh, Ww)
    gx = 2.0 * su / (Ww - 1) - 1.0
    gy = 2.0 * sv / (Hh - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    grid = grid[:, None].expand(-1, Wn, -1, -1, -1).reshape(b * Wn, Hh, Ww, 2).float()
    out = F.grid_sample(fw.reshape(b * Wn, C, Hh, Ww), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.reshape(b, Wn, C, Hh, Ww)


def wp_to_control(w_look, v):
    """closedloop.py VERBATIM."""
    x, y = w_look[:, 0], w_look[:, 1]
    ld2 = (x * x + y * y).clamp_min(LD2_FLOOR)
    kappa = 2.0 * y / ld2
    steer = torch.atan(WHEELBASE * kappa).clamp(-STEER_CLAMP, STEER_CLAMP)
    v_target = x / (LOOKAHEAD_STEP * DT)
    accel = ((v_target - v) / SPEED_TC).clamp(-ACCEL_CLAMP, ACCEL_CLAMP)
    return steer, accel


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _apply_overrides(cfg, d):
    for k, v in d.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            _apply_overrides(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(cfg, k, tuple(v))
        else:
            setattr(cfg, k, v)


def load_refc(ckpt, preset, device):
    """lowood_lanekeep.load_refc VERBATIM (strict load, anchors in the ckpt buffer)."""
    cfg = _REFC_PRESETS[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
    assert not cfg.refc1, "refc1 ckpt: horizons are path checkpoints, not time"
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])                       # STRICT
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, int(ck.get("step", -1)), cfg


class OODMap:
    """P1 MEASURED envelope |dlat|,|dpsi| -> ADE ratio (lowood_lanekeep.py), but
    VECTORISED and instrumented for out-of-envelope extrapolation."""

    def __init__(self, ci_json):
        d = json.loads(Path(ci_json).read_text())
        self.base = d["baseline_real_frames"]["mean"]
        self.lat_x = np.array([r["amount"] for r in d["conditions"]["lat"]])
        self.lat_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["lat"]])
        self.yaw_x = np.array([r["amount"] for r in d["conditions"]["yaw"]])
        self.yaw_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["yaw"]])

    def ratio_arr(self, lat_abs, yaw_abs_deg):
        al = np.interp(lat_abs, self.lat_x, self.lat_y)      # CLAMPS beyond 3.0 m
        ay = np.interp(yaw_abs_deg, self.yaw_x, self.yaw_y)  # CLAMPS beyond 12 deg
        ex_l = np.clip((al - self.base) / self.base, 0.0, None)
        ex_y = np.clip((ay - self.base) / self.base, 0.0, None)
        return 1.0 + ex_l + ex_y


@torch.no_grad()
def openloop_canary(model, episodes, device, stride=8, batch=16):
    """REF-C open-loop ADE@2s on this corpus. Registry canary: the canonical-val
    number is 0.4728 [0.3835, 0.5699] (MODEL_REGISTRY §4.3). A wildly different
    value here means the corpus/input convention differs, and the closed-loop
    numbers below would not be comparable to anything standing."""
    ades, eids = [], []
    for ei, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        poses = ep.poses.float()
        T = fr.shape[0]
        starts = list(range(0, T - W - max(WP_STEPS), stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            frames = torch.stack([fr[t0:t0 + W] for t0 in ch]).to(device)
            last = torch.tensor([t0 + W - 1 for t0 in ch])
            v0 = poses[last, 3].to(device)
            traj = model(frames, nav_cmd=None, v0=v0, steps=2)["traj"].cpu()
            gt = gt_ego_waypoints(poses, last)
            ades.append(torch.linalg.norm(traj - gt, dim=-1).mean(1))
            eids += [str(ei)] * len(ch)
    a = torch.cat(ades).numpy()
    return _ci.episode_cluster_bootstrap(a, eids, n_boot=2000)


@torch.no_grad()
def rollout(model, episodes, device, K, stride=8, batch=16):
    """Real-footage-in-the-loop closed loop at horizon K. Loop body VERBATIM from
    lowood_lanekeep.cl_realfootage; K is the only parameter changed."""
    rows = {k: [] for k in ("ade2s", "peak_lat", "mean_lat", "peak_yaw",
                            "hd2s", "hdK", "speed", "eid", "t0", "epi")}
    lat_all, yaw_all, de_fixed = [], [], []
    fixed_steps = [s for s in (20, 40, 80, 120, 160, 190) if s <= K]
    for ep_i, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        poses = ep.poses.float()
        T = fr.shape[0]
        starts = list(range(0, T - W - K, stride))
        if not starts:
            continue
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            b = len(ch)
            t0 = torch.tensor(ch)
            last = t0 + W - 1
            idx = last[:, None] + torch.arange(0, K + 1)[None]
            Pxy = poses[idx][..., :2]
            Pyaw = poses[idx][..., 2]
            oyaw = poses[last, 2]
            oxy = poses[last, :2]
            ex = poses[last, 0].clone(); ey = poses[last, 1].clone()
            eyaw = poses[last, 2].clone(); ev = poses[last, 3].clone()
            ego_ego = torch.zeros(b, K, 2)
            lat_t = torch.zeros(b, K); yaw_t = torch.zeros(b, K)
            ar = torch.arange(b)
            for k in range(K):
                d = (Pxy - torch.stack([ex, ey], -1)[:, None]).norm(dim=-1)
                mstar = d.argmin(dim=1)
                pref = Pxy[ar, mstar]; yref = Pyaw[ar, mstar]
                dx = ex - pref[:, 0]; dy = ey - pref[:, 1]
                dlat = -torch.sin(yref) * dx + torch.cos(yref) * dy
                dpsi = _wrap(eyaw - yref)
                lat_t[:, k] = dlat; yaw_t[:, k] = dpsi
                wins = [fr[int(t0[i] + mstar[i]):int(t0[i] + mstar[i]) + W]
                        for i in range(b)]
                fw = torch.stack(wins).to(device)
                Hs = torch.stack([
                    sampling_homography(float(dlat[i]),
                                        float(math.degrees(dpsi[i])), 1.5, 0.0)
                    for i in range(b)])
                fw = warp_batch(fw, Hs)
                w_look = model(fw, nav_cmd=None, v0=ev.to(device),
                               steps=2)["traj"][:, 0].cpu()
                steer, accel = wp_to_control(w_look, ev)
                ex = ex + ev * torch.cos(eyaw) * DT
                ey = ey + ev * torch.sin(eyaw) * DT
                eyaw = eyaw + ev / WHEELBASE * torch.tan(steer) * DT
                ev = (ev + accel * DT).clamp_min(0.0)
                wdx = ex - oxy[:, 0]; wdy = ey - oxy[:, 1]
                ego_ego[:, k, 0] = (torch.cos(oyaw) * wdx + torch.sin(oyaw) * wdy).cpu()
                ego_ego[:, k, 1] = (-torch.sin(oyaw) * wdx + torch.cos(oyaw) * wdy).cpu()
            # horizon-INVARIANT closed-loop ADE@2s (steps 5,10,15,20)
            gt2 = gt_ego_waypoints(poses, last, wp_steps=WP_STEPS)
            rows["ade2s"].append(
                torch.linalg.norm(ego_ego[:, WP_IDX] - gt2, dim=-1).mean(1))
            # DE at each fixed elapsed time available at this K
            gtf = gt_ego_waypoints(poses, last, wp_steps=tuple(fixed_steps))
            de_fixed.append(torch.linalg.norm(
                ego_ego[:, [s - 1 for s in fixed_steps]] - gtf, dim=-1))
            lat_abs = lat_t.abs(); yaw_abs_deg = yaw_t.abs() * 180 / math.pi
            lat_all.append(lat_abs); yaw_all.append(yaw_abs_deg)
            rows["peak_lat"].append(lat_abs.max(1).values)
            rows["mean_lat"].append(lat_abs.mean(1))
            rows["peak_yaw"].append(yaw_abs_deg.max(1).values)
            rows["hd2s"].append(
                _wrap(poses[last + 20, 2] - poses[last, 2]).abs() * 180 / math.pi)
            rows["hdK"].append(
                _wrap(poses[last + K, 2] - poses[last, 2]).abs() * 180 / math.pi)
            rows["speed"].append(poses[last, 3])
            rows["eid"] += [str(ep_i)] * b
            rows["t0"].append(t0.clone())
            rows["epi"].append(torch.full((b,), ep_i))
    out = {k: (v if k == "eid" else torch.cat(v)) for k, v in rows.items()}
    out["lat"] = torch.cat(lat_all)              # [N,K] |XTE| per step
    out["yaw"] = torch.cat(yaw_all)              # [N,K] |dpsi| deg per step
    out["de_fixed"] = torch.cat(de_fixed)        # [N, len(fixed_steps)]
    out["fixed_steps"] = fixed_steps
    return out


def _boot(x, eid):
    return _ci.episode_cluster_bootstrap(np.asarray(x, float), eid, n_boot=2000)


def block(pw, mask, thresholds, primary, ood, K):
    m = np.flatnonzero(mask)
    if len(m) < 2:
        return None
    e = [pw["eid"][i] for i in m]
    lat = pw["lat"].numpy()[m]                    # [n,K]
    yaw = pw["yaw"].numpy()[m]
    ratio = ood.ratio_arr(lat, yaw)               # [n,K]
    out = {
        "n_windows": int(len(m)), "n_episodes": int(len(set(e))),
        "horizon_K": K, "horizon_s": round(K * DT, 2),
        "corridor_departure_rate": _boot((lat > primary).mean(1), e),
        "corridor_departure_rate_by_threshold_m": {
            f"{t:g}": _boot((lat > t).mean(1), e) for t in thresholds},
        "window_departure_rate": _boot((lat > primary).any(1).astype(float), e),
        "window_departure_rate_by_threshold_m": {
            f"{t:g}": _boot((lat > t).any(1).astype(float), e) for t in thresholds},
        "peak_xte_m": _boot(lat.max(1), e),
        "mean_xte_m": _boot(lat.mean(1), e),
        "peak_dpsi_deg": _boot(yaw.max(1), e),
        "closed_ade2s_m": _boot(pw["ade2s"].numpy()[m], e),
        "ood_peak_ratio": _boot(ratio.max(1), e),
        "ood_mean_ratio": _boot(ratio.mean(1), e),
        "frac_windows_ood_peak_under_1p16": round(float((ratio.max(1) <= 1.16).mean()), 4),
        "frac_windows_ood_peak_under_1p5": round(float((ratio.max(1) <= 1.5).mean()), 4),
        "EXTRAPOLATION_frac_steps_lat_over_3m": round(float((lat > ENV_LAT_MAX).mean()), 5),
        "EXTRAPOLATION_frac_steps_yaw_over_12deg": round(float((yaw > ENV_YAW_MAX).mean()), 5),
        "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": round(
            float(((lat > ENV_LAT_MAX) | (yaw > ENV_YAW_MAX)).any(1).mean()), 4),
        "mean_abs_xte_by_step_m": [round(float(x), 4) for x in lat.mean(0)],
    }
    fs = pw["fixed_steps"]
    dfx = pw["de_fixed"].numpy()[m]
    out["de_at_elapsed_s"] = {f"{s * DT:g}": _boot(dfx[:, i], e)
                              for i, s in enumerate(fs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refc-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--val-files", default="", help="comma list of ep_*.pt basenames")
    ap.add_argument("--p1-json", default="/workspace/e1a_e2a/lowood_flagship_ci.json")
    ap.add_argument("--horizons", default="20,40,80,120,160")
    ap.add_argument("--episodes", type=int, default=999)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corridor-grid", default="1.0,1.75,2.5")
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--skip-canary", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ks = [int(x) for x in args.horizons.split(",")]
    thresholds = [float(x) for x in args.corridor_grid.split(",")]
    primary = args.corridor_halfwidth
    assert primary in thresholds
    ood = OODMap(args.p1_json)

    if args.val_files:
        eps = [Path(args.val_dir) / f for f in args.val_files.split(",")]
    else:
        eps = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps]
    Ts = [int(e.poses.shape[0]) for e in episodes]
    print(f"[e1a] {len(episodes)} eps from {args.val_dir} | T in "
          f"[{min(Ts)},{max(Ts)}] | K sweep {Ks} | dev {device}", flush=True)
    # An episode contributes a window at horizon K only if T - W - K >= 1. Episodes
    # SHORTER than that are silently dropped by `rollout`, so report the survivors
    # per horizon instead of asserting a uniform T (they are 190-199 frames).
    surv = {K: int(sum(1 for T in Ts if T - W - K >= 1)) for K in Ks}
    print(f"[e1a] episodes surviving each horizon: {surv}", flush=True)
    assert surv[max(Ks)] > 0, (
        f"K_max {max(Ks)} leaves no window in any episode "
        f"(shortest {min(Ts)} frames; the ceiling is T-W-1 = {min(Ts) - W - 1})")

    model, step, cfg = load_refc(args.refc_ckpt, args.refc_preset, device)
    print(f"[e1a] REF-C {args.refc_preset} step {step} | anchors "
          f"{tuple(model.decoder.anchors.shape)} | horizons {cfg.trajectory.horizons}",
          flush=True)

    res = {
        "_experiment": "E1a CL-HORIZON-CURVE",
        "_design": "real-footage-in-the-loop low-OOD closed loop (lowood_lanekeep.py "
                   "loop body verbatim); ONLY the rollout horizon K varies. Zero "
                   "training, zero renderer. REF-C base, 2 truncated-denoise steps.",
        "_estimator": "episode_cluster_bootstrap (taniteval/ci.py), B=2000, over "
                      "val EPISODES; paired form for the (K vs K=20) deltas. "
                      "NEVER overlapping_holdout_se.",
        "_honest_bound": "the P1 OOD envelope was MEASURED only to |dlat|<=3.0 m / "
                         "|dyaw|<=12 deg; np.interp CLAMPS beyond that, so the "
                         "reported OOD ratio is a LOWER bound at long horizons. "
                         "EXTRAPOLATION_* fields carry the out-of-envelope fraction.",
        "refc_ckpt": args.refc_ckpt, "refc_step": step, "refc_preset": args.refc_preset,
        "n_anchors": int(model.decoder.anchors.shape[0]),
        "denoise_steps": 2,
        "val_dir": args.val_dir, "n_episodes": len(episodes),
        "episode_T_min": min(Ts), "episode_T_max": max(Ts),
        "episodes_surviving_each_horizon": surv,
        "_horizon_ceiling_note": f"an episode yields a window at K only if T-W-K>=1; "
                                 f"T ranges {min(Ts)}-{max(Ts)} frames so the absolute "
                                 f"ceiling is K = {max(Ts) - W - 1} ({(max(Ts) - W - 1) * DT:.1f} s). "
                                 f"K=200 (20 s) is STRUCTURALLY IMPOSSIBLE on this corpus.",
        "stride": args.stride, "corridor_thresholds_m": thresholds,
        "corridor_primary_m": primary, "junction_deg": args.junction_deg,
        "horizons_K": Ks, "horizons_s": [round(k * DT, 2) for k in Ks],
        "p1_envelope": args.p1_json, "p1_baseline_ade2s": ood.base,
    }

    with strict_numerics():
        if not args.skip_canary:
            t = time.time()
            res["openloop_ade2s_canary"] = openloop_canary(
                model, episodes, device, args.stride, args.batch)
            res["openloop_ade2s_canary"]["_registry_reference_canonical_val"] = \
                "0.4728 [0.3835, 0.5699] (MODEL_REGISTRY 4.3, physicalai-val-0c5f7dac3b11)"
            c = res["openloop_ade2s_canary"]
            print(f"[e1a] CANARY open-loop ADE@2s = {c['mean']:.4f} "
                  f"[{c['lo']:.4f},{c['hi']:.4f}] n={c['n_windows']} "
                  f"({time.time() - t:.0f}s)", flush=True)

        per_K = {}
        for K in Ks:
            t = time.time()
            pw = rollout(model, episodes, device, K, args.stride, args.batch)
            per_K[K] = pw
            hd = pw["hd2s"].numpy(); spd = pw["speed"].numpy()
            junc = hd >= args.junction_deg
            long_ = (~junc) & (spd >= np.median(spd))
            other = (~junc) & (~long_)
            res.setdefault("all_windows", {})[str(K)] = {
                "overall": block(pw, np.ones(len(hd), bool), thresholds, primary, ood, K),
                "junction": block(pw, junc, thresholds, primary, ood, K),
                "longitudinal": block(pw, long_, thresholds, primary, ood, K),
                "other": block(pw, other, thresholds, primary, ood, K),
                "_stratification": "junction = |net heading change over the FIRST 2 s| "
                                   ">= 10 deg (the standing definition, held FIXED "
                                   "across K so the strata are comparable); "
                                   "longitudinal = not junction AND speed >= median.",
            }
            o = res["all_windows"][str(K)]["overall"]
            print(f"[e1a] K={K:4d} ({K * DT:4.1f}s) n={o['n_windows']:5d} "
                  f"CDR@{primary}={o['corridor_departure_rate']['mean']:.4f} "
                  f"winDEP={o['window_departure_rate']['mean']:.4f} "
                  f"peakXTE={o['peak_xte_m']['mean']:.3f} "
                  f"OODpeak={o['ood_peak_ratio']['mean']:.3f} "
                  f"outEnv={o['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']:.3f} "
                  f"({time.time() - t:.0f}s)", flush=True)
            Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    # ---- COMMON-START PAIRED curve (decision-grade) -------------------------
    key = lambda pw: [(int(a), int(b)) for a, b in zip(pw["epi"], pw["t0"])]  # noqa: E731
    keysets = [set(key(per_K[K])) for K in Ks]
    common = set.intersection(*keysets)
    res["paired_common_start"] = {
        "_note": "IDENTICAL windows at every horizon (the start set at K_max is a "
                 "subset of every smaller K's). Removes the window-composition "
                 "confound; deltas use the PAIRED episode-cluster bootstrap.",
        "n_common_windows": len(common)}
    if len(common) >= 8:
        sel = {}
        for K in Ks:
            kk = key(per_K[K])
            order = {c: i for i, c in enumerate(kk)}
            sel[K] = np.array([order[c] for c in sorted(common)])
        base_pw = per_K[Ks[0]]
        s0 = sel[Ks[0]]
        hd = base_pw["hd2s"].numpy()[s0]; spd = base_pw["speed"].numpy()[s0]
        junc = hd >= args.junction_deg
        long_ = (~junc) & (spd >= np.median(spd))
        other = (~junc) & (~long_)
        eidc = [base_pw["eid"][i] for i in s0]
        strata = {"overall": np.ones(len(hd), bool), "junction": junc,
                  "longitudinal": long_, "other": other}
        for K in Ks:
            pw = per_K[K]
            s = sel[K]
            sub = {"eid": [pw["eid"][i] for i in s],
                   "lat": pw["lat"][s], "yaw": pw["yaw"][s],
                   "ade2s": pw["ade2s"][s], "de_fixed": pw["de_fixed"][s],
                   "fixed_steps": pw["fixed_steps"]}
            res["paired_common_start"][str(K)] = {
                nm: block(sub, mk, thresholds, primary, ood, K)
                for nm, mk in strata.items()}
        # paired deltas vs K = Ks[0]
        P = _ci.paired_episode_cluster_bootstrap
        ref = Ks[0]
        deltas = {}
        for nm, mk in strata.items():
            m = np.flatnonzero(mk)
            if len(m) < 2:
                continue
            e = [eidc[i] for i in m]
            r_lat = per_K[ref]["lat"][sel[ref]].numpy()
            dd = {}
            for K in Ks[1:]:
                k_lat = per_K[K]["lat"][sel[K]].numpy()
                dd[str(K)] = {
                    "d_corridor_departure_rate": P((k_lat > primary).mean(1)[m],
                                                   (r_lat > primary).mean(1)[m], e),
                    "d_window_departure_rate": P(
                        (k_lat > primary).any(1).astype(float)[m],
                        (r_lat > primary).any(1).astype(float)[m], e),
                    "d_peak_xte_m": P(k_lat.max(1)[m], r_lat.max(1)[m], e),
                    "d_closed_ade2s_m": P(per_K[K]["ade2s"][sel[K]].numpy()[m],
                                          per_K[ref]["ade2s"][sel[ref]].numpy()[m], e),
                }
            deltas[nm] = dd
        res["paired_common_start"]["deltas_vs_K20"] = {
            "_note": f"positive => the LONGER horizon is WORSE (higher). Reference "
                     f"K={ref}. Paired episode-cluster bootstrap, B=2000.",
            **deltas}
    else:
        res["paired_common_start"]["_skipped"] = (
            f"only {len(common)} common windows; paired curve not emitted")

    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"[e1a] wrote {args.out}", flush=True)
    print("E1A_DONE", flush=True)


if __name__ == "__main__":
    main()
