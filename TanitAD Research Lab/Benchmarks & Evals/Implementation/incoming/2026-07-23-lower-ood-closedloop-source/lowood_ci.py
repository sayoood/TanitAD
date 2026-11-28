"""P1 — DECISION-GRADE the real-footage observation-OOD envelope (flagship v1).

Extends the prior prototype (lowood_probe.py, n=12 -> 265 windows, BARE point
estimates) with the program's mandated decision-grade estimator: the
EPISODE-CLUSTER BOOTSTRAP (taniteval/ci.py). For every deviation condition we
now report a percentile CI over the val episodes AND a PAIRED bootstrap vs the
Delta=0 baseline (same windows) so the envelope's rise is CI-resolved from noise
rather than asserted from a single point.

WHY: CLAUDE.md — "Never quote an interval without its estimator"; the decision-
grade interval is the episode-cluster bootstrap; for two arms on the same windows
use the PAIRED version, never quadrature. The prototype quoted "+6.3% at 2 m
lateral" etc. with no interval; this closes that gap for the one arm reachable on
pod1 (flagship v1). The 40-ep tightening and the REF-C 2nd arm are BLOCKED
(eval-pod-resident / HF-storage-locked) and reported as such, not faked.

Delta=0 here is byte-identical to lowood_probe.py's Delta=0 (== the gate rollout).
Reuses lowood_probe's warp geometry verbatim (imported), so this adds ONLY the
per-window retention + bootstrap — no new observation model, no new decode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/root/taniteval")

from lowood_probe import (append_speed_channel, build_world_from_ckpt,  # noqa: E402
                          pix_roll, sampling_homography, selfcheck, warp_frames,
                          K_MAX, SPEED_SCALE)
from driving_diagnostic import WP_STEPS, de_of, gt_ego_waypoints  # noqa: E402
from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.metric_dynamics import (HierarchicalGrounding,  # noqa: E402
                                            rollout_decode)
from taniteval import ci as _ci  # noqa: E402

# REF-C base reference numbers (INHERITED, registry 4.3 / REFC_openloop_diagnostic.json)
NUREC_REFC_OPENLOOP_ADE = 1.5157      # REF-C base open-loop ADE@2s on NuRec recon
REAL_REFC_OPENLOOP_ADE = 0.4728       # REF-C base open-loop ADE@2s on real footage


@torch.no_grad()
def run_condition_pw(world, step_readout, episodes, device, window, speed_input,
                     warp_kind, amount, h_cam, pitch_deg, stride, batch):
    """One pass; returns per-window ADE_0_2s [N] and per-window episode ids [N].
    Window order is IDENTICAL across conditions (same eps/stride) -> aligned for
    the paired bootstrap. eid = episode LOOP INDEX (one ep_*.pt == one cluster)."""
    H = None
    if warp_kind == "lat":
        H = sampling_homography(amount, 0.0, h_cam, pitch_deg)
    elif warp_kind == "yaw":
        H = sampling_homography(0.0, amount, h_cam, pitch_deg)
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    DE, EID = [], []
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
            DE.append(de_of(pred, gt))                     # [b,4] per-window/horizon
            EID.extend([str(ep_i)] * len(ch))
    de = torch.cat(DE)                                     # [N,4]
    ade_pw = de.mean(dim=1).numpy()                        # [N] ADE_0_2s per window
    de2_pw = de[:, -1].numpy()                             # [N] DE@2s per window
    return ade_pw, de2_pw, EID


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
    ap.add_argument("--lat-grid", default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--yaw-grid", default="0,1,2,3,5,8,12")
    ap.add_argument("--pix-grid", default="0,2,4,8,16,32")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = selfcheck(device)
    print(f"[selfcheck] {sc}", flush=True)
    assert sc["identity_max_abs_err"] < 1e-3
    assert sc["marker_moved_left"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world, speed_input, _ = build_world_from_ckpt(flagship4b_config(), ck)
    world = world.to(device).eval()
    grounding = HierarchicalGrounding(world.state_dim).to(device).eval()
    grounding.load_state_dict(ck["grounding"])
    step_readout = grounding.step["op"]
    window = world.predictor.cfg.window
    step = int(ck.get("step", -1))
    print(f"[lowood-ci] ckpt step {step} speed_input {speed_input} window {window} "
          f"dev {device}", flush=True)

    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps]
    assert episodes, f"no ep_*.pt under {args.val_dir}"
    print(f"[lowood-ci] {len(episodes)} val episodes", flush=True)

    def grid(s):
        return [float(x) for x in s.split(",") if x != ""]

    def boot(ade_pw, eid):
        return _ci.episode_cluster_bootstrap(ade_pw, eid, n_boot=args.n_boot)

    results = {"ckpt": args.ckpt, "step": step, "val_dir": args.val_dir,
               "n_episodes": len(episodes), "speed_input": speed_input,
               "estimator": "episode_cluster_bootstrap (taniteval/ci.py), "
                            "paired vs Delta=0 on the same windows",
               "n_boot": args.n_boot,
               "intrinsics": {"f_eff_px": 266.0, "principal": 128.0,
                              "h_cam_m": args.h_cam, "pitch_deg": args.pitch_deg},
               "selfcheck": sc,
               "nurec_refc_openloop_ade": NUREC_REFC_OPENLOOP_ADE,
               "real_refc_openloop_ade": REAL_REFC_OPENLOOP_ADE,
               "conditions": {}}

    with strict_numerics():
        ade0, de20, eid0 = run_condition_pw(world, step_readout, episodes, device,
                                            window, speed_input, "none", 0.0,
                                            args.h_cam, args.pitch_deg,
                                            args.stride, args.batch)
        base_ci = boot(ade0, eid0)
        results["baseline_real_frames"] = base_ci
        # gap to NuRec recon-OOD (REF-C base scalar, INHERITED — non-paired, labelled)
        results["gap_to_nurec"] = {
            "_def": "real-footage baseline (flagship, MEASURED here) vs NuRec "
                    "recon-OOD (REF-C base scalar, INHERITED) — non-paired: "
                    "different model+source, so a labelled ratio not a paired CI",
            "flagship_real_ade2s": base_ci["mean"],
            "flagship_real_ade2s_ci": [base_ci["lo"], base_ci["hi"]],
            "nurec_refc_ade2s": NUREC_REFC_OPENLOOP_ADE,
            "ratio_nurec_over_flagship_real":
                round(NUREC_REFC_OPENLOOP_ADE / base_ci["mean"], 3)}
        print(f"[baseline] real-frame ADE_0_2s={base_ci['mean']:.4f} "
              f"[{base_ci['lo']:.4f},{base_ci['hi']:.4f}] "
              f"n_ep={base_ci['n_episodes']} n_win={base_ci['n_windows']}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))

        for kind, gvals in (("lat", grid(args.lat_grid)),
                            ("yaw", grid(args.yaw_grid)),
                            ("pixshift", grid(args.pix_grid))):
            results["conditions"][kind] = []
            for a in gvals:
                ade, de2, eid = run_condition_pw(world, step_readout, episodes,
                                                 device, window, speed_input,
                                                 kind, a, args.h_cam,
                                                 args.pitch_deg, args.stride,
                                                 args.batch)
                ci = boot(ade, eid)
                paired = _ci.paired_episode_cluster_bootstrap(
                    ade, ade0, eid, n_boot=args.n_boot)   # condition - baseline
                row = {"amount": a, "ade2s_ci": ci,
                       "paired_vs_baseline": paired,
                       "pct_vs_baseline":
                           round(100.0 * (ci["mean"] - base_ci["mean"])
                                 / base_ci["mean"], 2)}
                results["conditions"][kind].append(row)
                sep = "SEP" if paired["separated"] else "n.s."
                print(f"[{kind}] amt={a:<5} ADE={ci['mean']:.4f} "
                      f"[{ci['lo']:.4f},{ci['hi']:.4f}] "
                      f"dvs_base={paired['delta']:+.4f} "
                      f"[{paired['lo']:+.4f},{paired['hi']:+.4f}] {sep}", flush=True)
                Path(args.out).write_text(json.dumps(results, indent=2))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[lowood-ci] wrote {args.out}", flush=True)
    print("LOWOOD_CI_DONE", flush=True)


if __name__ == "__main__":
    main()
