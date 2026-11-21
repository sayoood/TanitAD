"""C-ADE -- P1's OWN criterion, extended on the YAW arm past the 12 deg grid end.

This is `lowood_ci.py` with four changes and nothing else:

  1. the yaw grid runs to 30 deg instead of stopping at 12 (P1's grid end);
  2. **DESTROYED-OBSERVATION CONTROLS** are run in the same pass -- the
     pre-registered C13 dynamic-range check. Without them no C-ADE number is
     admissible, because P1's criterion is model-mediated and could saturate:
     an arm that largely ignores the image would show flat ADE under a
     destroyed observation, and the whole envelope would sit inside its noise;
  3. the per-window error is DECOMPOSED into along-track and cross-track
     (`gt_ego_waypoints` is already ego-frame, so component 0 is along and
     component 1 is cross) -- this finding exists because an undecomposed
     statistic hid which axis was failing;
  4. `sys.path` points at the trees that actually exist on `tanitad-eval`
     (P1's scripts hardcode `/workspace/TanitAD/stack`, which does NOT exist
     on this host).

The warp geometry is IMPORTED VERBATIM from `lowood_probe.py` -- never
re-implemented -- so the 0..12 deg points are a strict REPRODUCTION of P1 and
the extension shares its observation model exactly.

Estimator: `taniteval.ci.episode_cluster_bootstrap` B=2000, unit = val episode,
plus the PAIRED form vs Delta=0 on the same windows. `overlapping_holdout_se`
appears nowhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lowood_probe import (append_speed_channel, build_world_from_ckpt,  # noqa: E402
                          sampling_homography, selfcheck, warp_frames,
                          K_MAX, SPEED_SCALE)
from driving_diagnostic import WP_STEPS, gt_ego_waypoints  # noqa: E402
from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.metric_dynamics import (HierarchicalGrounding,  # noqa: E402
                                            rollout_decode)
from taniteval import ci as _ci  # noqa: E402

# The DESTROYED-OBSERVATION controls. Each must land FAR above the 12 deg value
# or C-ADE has no dynamic range and outcome C is declared.
DEAD_KINDS = ("dead_black", "dead_noise", "dead_shuffle")


def destroy(fw, kind, seed=0):
    """Replace the observation with one carrying NO scene information.

    `dead_black`   -- zeros: no content at all.
    `dead_noise`   -- uniform noise: content with no structure.
    `dead_shuffle` -- a fixed spatial permutation: destroys all structure while
                      preserving the exact per-frame intensity histogram, so it
                      isolates STRUCTURE from BRIGHTNESS statistics.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "dead_black":
        return torch.zeros_like(fw)
    if kind == "dead_noise":
        return torch.rand(fw.shape, generator=g).to(fw.device, fw.dtype)
    if kind == "dead_shuffle":
        n = fw.shape[-1] * fw.shape[-2]
        perm = torch.randperm(n, generator=g).to(fw.device)
        flat = fw.reshape(*fw.shape[:-2], n)
        return flat.index_select(-1, perm).reshape(fw.shape)
    raise ValueError(kind)


@torch.no_grad()
def run_condition_pw(world, step_readout, episodes, device, window, speed_input,
                     warp_kind, amount, h_cam, pitch_deg, stride, batch):
    """One pass. Returns per-window ADE_0_2s, DE@2s, |along|@2s, |cross|@2s, eid.

    Window order is IDENTICAL across conditions (same eps/stride) -> aligned for
    the paired bootstrap. eid = episode loop index (one ep_*.pt == one cluster).
    """
    H = None
    if warp_kind == "yaw":
        H = sampling_homography(0.0, amount, h_cam, pitch_deg)
    elif warp_kind == "lat":
        H = sampling_homography(amount, 0.0, h_cam, pitch_deg)
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    DE, ALONG, CROSS, EID = [], [], [], []
    for ep_i, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        T = fr.shape[0]
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)
            b, W = fw.shape[0], fw.shape[1]
            if warp_kind in ("lat", "yaw") and abs(amount) > 0:
                flat = fw.reshape(b * W, *fw.shape[2:])
                flat = warp_frames(flat, H)
                fw = flat.reshape(b, W, *flat.shape[1:])
            elif warp_kind in DEAD_KINDS:
                fw = destroy(fw, warp_kind, seed=0)
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
            d = pred - gt                                   # [b, H, 2] ego frame
            DE.append(d.norm(dim=-1))                       # [b, H]
            ALONG.append(d[..., 0].abs())                   # forward  = along
            CROSS.append(d[..., 1].abs())                   # lateral  = cross
            EID.extend([str(ep_i)] * len(ch))
    de = torch.cat(DE)
    return (de.mean(1).numpy(), de[:, -1].numpy(),
            torch.cat(ALONG)[:, -1].numpy(), torch.cat(CROSS)[:, -1].numpy(), EID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/models/flagship-30k/ckpt.pt")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--h-cam", type=float, default=1.5)
    ap.add_argument("--pitch-deg", type=float, default=0.0)
    # P1's grid 0,1,2,3,5,8,12 EXTENDED past its end, + a 90 deg wrecking ball
    ap.add_argument("--yaw-grid", default="0,1,2,3,5,8,12,14,16,18,20,22,25,28,30,90")
    ap.add_argument("--lat-grid", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--perwindow-out", default="",
                    help="npz path for per-window arrays. REQUIRED to locate "
                         "the information-destruction crossing, which is a "
                         "PAIRED test of yaw(psi) vs the dead controls on the "
                         "same windows -- it cannot be done from summary CIs.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = selfcheck(device)
    print(f"[selfcheck] {sc}", flush=True)
    assert sc["identity_max_abs_err"] < 1e-3, "Delta=0 warp is not identity"
    assert sc["marker_moved_left"], "lateral +right warp moved marker wrong way"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world, speed_input, _ = build_world_from_ckpt(flagship4b_config(), ck)
    world = world.to(device).eval()
    grounding = HierarchicalGrounding(world.state_dim).to(device).eval()
    grounding.load_state_dict(ck["grounding"])
    step_readout = grounding.step["op"]
    window = world.predictor.cfg.window
    step = int(ck.get("step", -1))
    print(f"[yawext] ckpt step {step} speed_input {speed_input} window {window} "
          f"dev {device}", flush=True)

    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps]
    assert episodes, f"no ep_*.pt under {args.val_dir}"
    print(f"[yawext] {len(episodes)} val episodes from {args.val_dir}", flush=True)

    def grid(s):
        return [float(x) for x in s.split(",") if x != ""]

    res = {
        "_what": "P1's ADE criterion, extended on the YAW arm past its grid end, "
                 "with destroyed-observation dynamic-range controls",
        "_evidence_class": "MEASURED (ours)",
        "ckpt": args.ckpt, "step": step, "val_dir": args.val_dir,
        "n_episodes": len(episodes), "speed_input": speed_input,
        "stride": args.stride,
        "estimator": "episode_cluster_bootstrap (taniteval/ci.py), paired vs "
                     "Delta=0 on the same windows; B=%d; unit=val episode"
                     % args.n_boot,
        "_refused_estimator": "overlapping_holdout_se (biases point AND interval)",
        "n_boot": args.n_boot,
        "intrinsics": {"f_eff_px": 266.0, "principal": 128.0,
                       "h_cam_m": args.h_cam, "pitch_deg": args.pitch_deg},
        "selfcheck": sc,
        "_p1_grid_end_deg": 12.0,
        "conditions": {},
    }

    # per-window retention: the paired yaw-vs-dead test needs the raw arrays
    PW: dict[str, np.ndarray] = {}

    def bank():
        Path(args.out).write_text(json.dumps(res, indent=2))
        if args.perwindow_out and PW:
            np.savez_compressed(args.perwindow_out, **PW)

    with strict_numerics():
        t0 = time.time()
        a0, d20, al0, cr0, eid0 = run_condition_pw(
            world, step_readout, episodes, device, window, speed_input,
            "none", 0.0, args.h_cam, args.pitch_deg, args.stride, args.batch)
        base = _ci.episode_cluster_bootstrap(a0, eid0, n_boot=args.n_boot)
        res["baseline_real_frames"] = base
        res["baseline_along_2s"] = _ci.episode_cluster_bootstrap(
            al0, eid0, n_boot=args.n_boot)
        res["baseline_cross_2s"] = _ci.episode_cluster_bootstrap(
            cr0, eid0, n_boot=args.n_boot)
        PW["eid"] = np.asarray(eid0)
        PW["ade__baseline"] = a0
        PW["along__baseline"] = al0
        PW["cross__baseline"] = cr0
        print(f"[baseline] ADE_0_2s={base['mean']:.4f} "
              f"[{base['lo']:.4f},{base['hi']:.4f}] n_win={base['n_windows']} "
              f"n_ep={base['n_episodes']} ({time.time()-t0:.0f}s)", flush=True)
        bank()

        # Priority order, so a killed run still yields value: the YAW arm (the
        # binding axis) -> the DEAD controls (which gate the admissibility of
        # every yaw number above) -> the LATERAL arm (explicitly "only if time
        # permits").
        todo = [("yaw", a) for a in grid(args.yaw_grid)]
        todo += [(k, 0.0) for k in DEAD_KINDS]
        todo += [("lat", a) for a in grid(args.lat_grid)]

        for kind, a in todo:
            t1 = time.time()
            ade, de2, al, cr, eid = run_condition_pw(
                world, step_readout, episodes, device, window, speed_input,
                kind, a, args.h_cam, args.pitch_deg, args.stride, args.batch)
            ci = _ci.episode_cluster_bootstrap(ade, eid, n_boot=args.n_boot)
            paired = _ci.paired_episode_cluster_bootstrap(
                ade, a0, eid, n_boot=args.n_boot)
            row = {
                "kind": kind, "amount": a, "ade2s_ci": ci,
                "paired_vs_baseline": paired,
                "pct_vs_baseline": round(
                    100.0 * (ci["mean"] - base["mean"]) / base["mean"], 2),
                "along_2s": _ci.episode_cluster_bootstrap(
                    al, eid, n_boot=args.n_boot),
                "cross_2s": _ci.episode_cluster_bootstrap(
                    cr, eid, n_boot=args.n_boot),
                "paired_along_2s": _ci.paired_episode_cluster_bootstrap(
                    al, al0, eid, n_boot=args.n_boot),
                "paired_cross_2s": _ci.paired_episode_cluster_bootstrap(
                    cr, cr0, eid, n_boot=args.n_boot),
                "_beyond_p1_grid": bool(kind == "yaw" and a > 12.0),
                "_is_dynamic_range_control": bool(kind in DEAD_KINDS
                                                  or (kind == "yaw" and a >= 90)),
                "secs": round(time.time() - t1, 1),
            }
            res["conditions"].setdefault(kind, []).append(row)
            tag = kind if kind in DEAD_KINDS else f"{kind}{a:g}"
            PW[f"ade__{tag}"] = ade
            PW[f"along__{tag}"] = al
            PW[f"cross__{tag}"] = cr
            sep = "SEP" if paired["separated"] else "n.s."
            flag = " *BEYOND-P1*" if row["_beyond_p1_grid"] else ""
            flag += " *DEAD-CONTROL*" if row["_is_dynamic_range_control"] else ""
            print(f"[{kind}] amt={a:<6g} ADE={ci['mean']:.4f} "
                  f"[{ci['lo']:.4f},{ci['hi']:.4f}] "
                  f"d={paired['delta']:+.4f} [{paired['lo']:+.4f},"
                  f"{paired['hi']:+.4f}] {sep} "
                  f"({row['secs']:.0f}s){flag}", flush=True)
            bank()

    bank()
    print(f"[yawext] wrote {args.out}", flush=True)
    print("YAWEXT_DONE", flush=True)


if __name__ == "__main__":
    main()
