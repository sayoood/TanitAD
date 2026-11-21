"""Lower-OOD closed-loop eval SOURCE — real-footage log-replay OOD characterization.

WHY: every closed-loop number we have is confounded by NuRec/AlpaSim reconstruction
OOD — REF-C's open-loop ADE on NuRec reconstructions is 1.5157 vs 0.4728 on real
footage (3.21x; REFC_openloop_diagnostic.json). This measures the OOD of a
*real-footage log-replay* source: the observation is the REAL recorded frame
(reconstruction-OOD = 0 by construction); the only new error is the POSE-MISMATCH
between where the ego actually is (under a planner that deviates from the recorded
path) and where the real frame was captured. We quantify how open-loop ADE degrades
as a function of that imposed ego-vs-frame deviation (lateral metres / yaw degrees),
to bound the usable deviation envelope of a real-footage closed-loop source.

METHOD (Delta=0 is byte-for-byte scripts/eval_grounded_rollout_4b.py):
  encode each REAL val window -> roll the OPERATIVE predictor 20 steps under the
  TRUE action sequence (intent-free) -> decode per-step metric dpose with the
  trained grounding.step['op'] readout -> SE(2) accumulate to waypoints
  {5,10,15,20}={0.5,1,1.5,2}s -> ADE vs GT. The ONLY change per condition is an
  OBSERVATION-ONLY warp of the input frames simulating the ego being offset from
  the frame's capture pose (the speed channel v0 and the GT stay the true ego).

  lateral Delta: ground-plane (flat-road) homography, horizon-fixed shear — models
    only the ROAD-SURFACE parallax, so it UNDER-models 3D-structure parallax =>
    the resulting envelope is an OPTIMISTIC (upper-bound) usable deviation.
  yaw Delta: exact rotation homography about the down axis (depth-independent).
  pixshift: literal integer column roll, calibration-free cross-check.

Canonical pinhole intrinsics of the phase-0 cache: f_eff=266 px, 256x256,
principal point centered (build_pai_cache asserts |f_eff-266|<8; pai_calib_probe).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/TanitAD/stack")

from driving_diagnostic import (WP_STEPS, baseline_waypoints, de_of,  # noqa: E402
                                gt_ego_waypoints, net_heading_change_deg,
                                scalar_metrics)
from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.metric_dynamics import (HierarchicalGrounding,  # noqa: E402
                                            rollout_decode)

K_MAX = max(WP_STEPS)
F_EFF = 266.0            # canonical pinhole focal (px) of the 256x256 cache
CXY = 128.0             # principal point (centered)
SPEED_SCALE = 10.0      # tanitad/train/flagship_losses.py: v0 = pose_last[:,3]/10


def append_speed_channel(actions: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """Append v0 [B,1] (already /SPEED_SCALE), constant over time: [B,K,2]->[B,K,3]."""
    return torch.cat([actions, v0.unsqueeze(1).expand(-1, actions.shape[1], -1)],
                     dim=-1)


def build_world_from_ckpt(cfg, ck, ckpt_path=None):
    """Inlined (pod stack 0f93b98 predates tanitad.eval.ckpt_compat). Infer the
    trained operative action_dim from the act_emb weight (3 = speed-input), wire
    the predictor/tactical_pred to it, build + STRICT-load. Mirrors taniteval
    loaders.py + eval/ckpt_compat.build_world_from_ckpt."""
    import dataclasses
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    w = sd.get("predictor.act_emb.0.weight")
    action_dim = int(w.shape[1]) if torch.is_tensor(w) and w.ndim == 2 else 2
    if cfg.predictor.action_dim != action_dim:
        cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=action_dim)
        if getattr(cfg, "tactical_pred", None) is not None:
            cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred,
                                                    action_dim=action_dim)
        if hasattr(cfg, "speed_input"):
            cfg.speed_input = (action_dim == 3)
    world = WorldModel(cfg)
    world.load_state_dict(sd)
    return world, action_dim >= 3, "weights"


# --------------------------------------------------------------------------- #
# Observation warp — synthesize the view an ego OFFSET from the frame's capture #
# pose would see, by sampling the REAL frame (plane homography + rotation).     #
# --------------------------------------------------------------------------- #
def sampling_homography(dlat_m: float, dyaw_deg: float, h_cam: float,
                        pitch_deg: float, f: float = F_EFF, c: float = CXY):
    """3x3 homography mapping OFFSET-view (cam2) pixels -> REAL-frame (cam1)
    pixels, for grid_sample. cam2 = cam1 translated +dlat_m to the RIGHT and
    yawed +dyaw_deg to the LEFT. Camera frame X-right Y-down Z-forward; road
    plane normal n=(0,cos p, sin p), distance d=h_cam. Delta=0 -> identity."""
    K = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1.0]], dtype=torch.float64)
    Ki = torch.linalg.inv(K)
    p = math.radians(pitch_deg)
    n = torch.tensor([0.0, math.cos(p), math.sin(p)], dtype=torch.float64)
    d = float(h_cam)
    psi = math.radians(dyaw_deg)
    # world->cam2 rotation R = Ry(-psi); translation t = -R @ Cc, Cc=(dlat,0,0)
    Ry = torch.tensor([[math.cos(-psi), 0, math.sin(-psi)],
                       [0, 1.0, 0],
                       [-math.sin(-psi), 0, math.cos(-psi)]], dtype=torch.float64)
    R = Ry
    Cc = torch.tensor([dlat_m, 0.0, 0.0], dtype=torch.float64)
    t = -(R @ Cc)
    H_1to2 = K @ (R + torch.outer(t, n) / d) @ Ki      # cam1 px -> cam2 px
    return torch.linalg.inv(H_1to2)                    # cam2 px -> cam1 px (sample)


def warp_frames(fw: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """fw [N,C,Hh,Ww] float in [0,1]; H [3,3] cam2->cam1 sampling homography.
    Returns the warped frames (border-replicate, bilinear). Identity H -> fw."""
    N, C, Hh, Ww = fw.shape
    dev = fw.device
    ys, xs = torch.meshgrid(torch.arange(Hh, dtype=torch.float64, device=dev),
                            torch.arange(Ww, dtype=torch.float64, device=dev),
                            indexing="ij")
    ones = torch.ones_like(xs)
    P = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T          # [3, Hh*Ww]
    Hd = H.to(dev)
    src = Hd @ P                                                       # [3, Hh*Ww]
    su = (src[0] / src[2]).reshape(Hh, Ww)
    sv = (src[1] / src[2]).reshape(Hh, Ww)
    gx = 2.0 * su / (Ww - 1) - 1.0                                     # align_corners=True
    gy = 2.0 * sv / (Hh - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None].expand(N, -1, -1, -1).float()
    return torch.nn.functional.grid_sample(fw, grid, mode="bilinear",
                                           padding_mode="border",
                                           align_corners=True)


def pix_roll(fw: torch.Tensor, px: int) -> torch.Tensor:
    """Calibration-free horizontal column roll by px (replicate edge)."""
    if px == 0:
        return fw
    out = torch.roll(fw, shifts=px, dims=-1)
    if px > 0:
        out[..., :px] = fw[..., :1]
    else:
        out[..., px:] = fw[..., -1:]
    return out


@torch.no_grad()
def run_condition(world, step_readout, episodes, device, window, speed_input,
                  warp_kind, amount, h_cam, pitch_deg, stride, batch):
    """One ADE pass over all windows with a fixed observation warp."""
    H = None
    if warp_kind == "lat":
        H = sampling_homography(amount, 0.0, h_cam, pitch_deg)
    elif warp_kind == "yaw":
        H = sampling_homography(0.0, amount, h_cam, pitch_deg)
    DE, HDG, SPD = [], [], []
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    for ep in episodes:
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        T = fr.shape[0]
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)   # [b,W,9,256,256]
            b, W = fw.shape[0], fw.shape[1]
            if warp_kind in ("lat", "yaw") and abs(amount) > 0:
                flat = fw.reshape(b * W, fw.shape[2], fw.shape[3], fw.shape[4])
                flat = warp_frames(flat, H)
                fw = flat.reshape(b, W, fw.shape[2], fw.shape[3], fw.shape[4])
            elif warp_kind == "pixshift" and int(amount) != 0:
                flat = fw.reshape(b * W, fw.shape[2], fw.shape[3], fw.shape[4])
                flat = pix_roll(flat, int(amount))
                fw = flat.reshape(b, W, fw.shape[2], fw.shape[3], fw.shape[4])
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + K_MAX]
                              for t in ch]).to(device)
            if speed_input:
                v0 = (ep.poses[last, 3:4] / SPEED_SCALE).to(device)
                aw = append_speed_channel(aw, v0)
                fa = append_speed_channel(fa, v0)
            states = world.encode_window(fw)
            wp_full, _ = rollout_decode(world.predictor, states, aw, fa,
                                        step_readout, K_MAX)
            pred = wp_full.index_select(1, wp_idx.to(device)).cpu().float()
            gt = gt_ego_waypoints(ep.poses, last)
            DE.append(de_of(pred, gt))
            HDG.append(net_heading_change_deg(ep.poses, last))
            SPD.append(ep.poses[last, 3])
    de = torch.cat(DE)
    m = scalar_metrics(de)
    m["n_windows"] = int(de.shape[0])
    m["de@Ts"] = [float(de[:, i].mean()) for i in range(len(WP_STEPS))]
    return m


def selfcheck(device):
    """Delta=0 warp is identity; horizon row (v=128) is invariant under lateral
    shear; lateral shift moves a ground point in the correct direction."""
    torch.manual_seed(0)
    img = torch.rand(1, 9, 256, 256, device=device)
    H0 = sampling_homography(0.0, 0.0, 1.5, 0.0).to(device)
    out0 = warp_frames(img, H0)
    id_err = float((out0 - img).abs().max())
    H1 = sampling_homography(0.5, 0.0, 1.5, 0.0)         # +0.5 m right
    # a bright ground marker dead-ahead low in the frame should move LEFT in cam2
    marker = torch.zeros(1, 1, 256, 256, device=device)
    marker[0, 0, 210, 128] = 1.0
    w = warp_frames(marker, H1.to(device))
    col = int(w[0, 0, 210].argmax())
    return {"identity_max_abs_err": id_err, "marker_col_after_+0.5m": col,
            "marker_moved_left": bool(col < 128)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/models/flagship-30k/ckpt.pt")
    ap.add_argument("--val-dir",
                    default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--h-cam", type=float, default=1.5)     # nominal camera height (m)
    ap.add_argument("--pitch-deg", type=float, default=0.0)  # nominal level camera
    ap.add_argument("--lat-grid", default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--yaw-grid", default="0,1,2,3,5,8,12")
    ap.add_argument("--pix-grid", default="0,2,4,8,16,32")
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = selfcheck(device)
    print(f"[selfcheck] {sc}", flush=True)
    if args.selfcheck_only:
        Path(args.out).write_text(json.dumps({"selfcheck": sc}, indent=2))
        return
    assert sc["identity_max_abs_err"] < 1e-3, "Delta=0 warp is not identity"
    assert sc["marker_moved_left"], "lateral +right warp moved marker wrong way"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world, speed_input, act_src = build_world_from_ckpt(flagship4b_config(), ck,
                                                        ckpt_path=args.ckpt)
    world = world.to(device).eval()
    grounding = HierarchicalGrounding(world.state_dim).to(device).eval()
    grounding.load_state_dict(ck["grounding"])
    step_readout = grounding.step["op"]
    window = world.predictor.cfg.window
    step = int(ck.get("step", -1))
    print(f"[lowood] ckpt step {step} speed_input {speed_input}({act_src}) "
          f"window {window} dev {device}", flush=True)

    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps]
    assert episodes, f"no ep_*.pt under {args.val_dir}"
    print(f"[lowood] {len(episodes)} val episodes from {args.val_dir}", flush=True)

    def grid(s):
        return [float(x) for x in s.split(",") if x != ""]

    results = {"ckpt": args.ckpt, "step": step, "val_dir": args.val_dir,
               "n_episodes": len(episodes), "speed_input": speed_input,
               "intrinsics": {"f_eff_px": F_EFF, "principal": CXY,
                              "h_cam_m": args.h_cam, "pitch_deg": args.pitch_deg},
               "selfcheck": sc,
               "nurec_refc_openloop_ade": 1.5157,     # REFC_openloop_diagnostic.json
               "real_refc_openloop_ade": 0.4728,      # taniteval REF-C base
               "conditions": {}}

    with strict_numerics():
        base = run_condition(world, step_readout, episodes, device, window,
                             speed_input, "none", 0.0, args.h_cam,
                             args.pitch_deg, args.stride, args.batch)
        results["baseline_real_frames"] = base
        print(f"[baseline] real-frame open-loop ade_0_2s={base['ade_0_2s']:.4f} "
              f"n={base['n_windows']}  (recon-OOD = 0)", flush=True)

        for kind, gvals in (("lat", grid(args.lat_grid)),
                            ("yaw", grid(args.yaw_grid)),
                            ("pixshift", grid(args.pix_grid))):
            results["conditions"][kind] = []
            for a in gvals:
                m = run_condition(world, step_readout, episodes, device, window,
                                  speed_input, kind, a, args.h_cam,
                                  args.pitch_deg, args.stride, args.batch)
                m["amount"] = a
                results["conditions"][kind].append(m)
                print(f"[{kind}] amt={a:<5} ade_0_2s={m['ade_0_2s']:.4f} "
                      f"de@2s={m['de@Ts'][-1]:.4f}", flush=True)
                Path(args.out).write_text(json.dumps(results, indent=2))  # bank each

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[lowood] wrote {args.out}", flush=True)
    print("LOWOOD_DONE", flush=True)


if __name__ == "__main__":
    main()
