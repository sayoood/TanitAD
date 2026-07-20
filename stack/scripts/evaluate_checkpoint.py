"""One-command checkpoint evaluation: D1-D3 gates + custom metrics + spectrum.

Chains the integrated Phase-0 measurement machinery over a trained checkpoint
and cached validation episodes:

    D1  encoder state -> ego waypoints @0.5/1/1.5/2 s (frozen probe, vs pool)
    D2  imagined next-latent action ranking (P1 calibrated + P4 forward-dynamics)
    D3  imagined-vs-oracle trajectory decode at the predictor's max horizon
    + trajectory_extra_metrics merged into each gate report
    + transition-spectrum analysis (sizing evidence on the trained model)

HONEST HORIZON NOTE (P8): the current predictor imagines k in {1,2,4} steps
(0.1-0.4 s @ 10 Hz). D3 therefore runs at 0.4 s, not the plan's 2 s — reported
as `d3_horizon_s` in the output; extending imagination horizons is a config
decision for the next run. D1 decodes up to 2 s from the ENCODER state (probe
horizons are independent of the predictor).

Usage (pod):
  python scripts/evaluate_checkpoint.py \
      --ckpt /workspace/experiments/p0-sB01-realmix/ckpt.pt \
      --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache \
      --out /workspace/experiments/p0-sB01-realmix
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tanitad.config import base250cam_config, smoke_config  # noqa: F401
from tanitad.data.mixing import load_episode
from tanitad.eval.ckpt_compat import (SPEED_SCALE, append_speed_channel,
                                      build_world_from_ckpt)
from tanitad.eval.gates import (I2Input, gates_metrics_json, run_d1, run_d2,
                                run_d3)
from tanitad.eval.metrics import trajectory_extra_metrics
from tanitad.eval.spectral import estimate_transition_spectrum
from tanitad.instruments.numerics import strict_numerics

WAYPOINT_STEPS = (5, 10, 15, 20)          # 0.5/1/1.5/2 s @ 10 Hz (D1 targets)


def _ego_frame(dxy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate ENU displacements into the ego frame at departure yaw."""
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    x = dxy[..., 0] * c - dxy[..., 1] * s
    y = dxy[..., 0] * s + dxy[..., 1] * c
    return torch.stack([x, y], dim=-1)


@torch.no_grad()
def build_eval_tensors(world, episodes, device, window: int, k_max: int,
                       stride: int = 6, batch: int = 8,
                       speed_input: bool = False) -> dict:
    """Collect everything the three gates need from contract episodes.

    ``speed_input`` (3-ch operative ckpts): append v0 = poses[last,3]/
    SPEED_SCALE as the constant 3rd action channel wherever actions feed the
    MODEL (``world.imagine``), exactly as flagship_losses does in training —
    t=0 speed only, never a future speed. The returned ``actions`` tensor
    stays the RAW recorded 2-ch actions: it is probe DATA for the D2
    calibration / forward-dynamics probes and the spectral estimate, so it is
    kept comparable across checkpoints."""
    out = {k: [] for k in ("states", "waypoints", "eps", "z_prev", "z_true1",
                           "z_imag1", "z_imag_k", "z_true_k", "disp1",
                           "disp_k", "actions", "prev_state")}
    need_ahead = max(max(WAYPOINT_STEPS), k_max)
    for ep in episodes:
        frames = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = frames.shape[0]
        t0s = list(range(0, T - window - need_ahead, stride))
        for i in range(0, len(t0s), batch):
            chunk = t0s[i:i + batch]
            last = torch.tensor([t + window - 1 for t in chunk])
            fw = torch.stack([frames[t:t + window] for t in chunk]).to(device)
            aw = torch.stack([ep.actions[t:t + window] for t in chunk]).to(device)
            if speed_input:
                v0 = (ep.poses[last, 3:4] / SPEED_SCALE).to(device)   # [b, 1]
                aw = append_speed_channel(aw, v0)
            states = world.encode_window(fw)                     # [b, W, S]
            preds = world.imagine(states, aw)
            yaw0 = ep.poses[last, 2]
            p0 = ep.poses[last, :2]
            wp = torch.stack([_ego_frame(ep.poses[last + k, :2] - p0, yaw0)
                              for k in WAYPOINT_STEPS], dim=1)   # [b, 4, 2]
            z_true1 = world.encode(
                torch.stack([frames[t + window] for t in chunk]).to(device))
            z_true_k = world.encode(
                torch.stack([frames[t + window + k_max - 1]
                             for t in chunk]).to(device))
            dyaw = ep.poses[last, 2] - ep.poses[last - 1, 2]
            out["states"].append(states[:, -1].cpu())
            out["waypoints"].append(wp)
            out["eps"].extend([ep.episode_id] * len(chunk))
            out["z_prev"].append(states[:, -1].cpu())
            out["z_true1"].append(z_true1.cpu())
            out["z_imag1"].append(preds[1].cpu())
            out["z_imag_k"].append(preds[k_max].cpu())
            out["z_true_k"].append(z_true_k.cpu())
            out["disp1"].append(_ego_frame(
                ep.poses[last + 1, :2] - p0, yaw0))
            out["disp_k"].append(_ego_frame(
                ep.poses[last + k_max, :2] - p0, yaw0))
            out["actions"].append(ep.actions[last])
            out["prev_state"].append(
                torch.stack([ep.poses[last, 3], dyaw], dim=-1))
    return {k: (torch.cat(v) if k != "eps" else v) for k, v in out.items()}


def evaluate(world, episodes, device, exp_id: str, git_hash: str,
             corpus_meta: dict | None = None, speed_input: bool = False) -> dict:
    world = world.to(device).eval()
    cfg_w = world.predictor.cfg
    k_max = max(cfg_w.horizons)
    with strict_numerics():
        t = build_eval_tensors(world, episodes, device, cfg_w.window, k_max,
                               speed_input=speed_input)
        frames_i2 = (episodes[0].frames[:16].float().div(255.0)
                     if episodes[0].frames.dtype == torch.uint8
                     else episodes[0].frames[:16]).to(device)
        i2 = I2Input(encode_fn=lambda x: world.encode(x), frames=frames_i2,
                     batch_size=8)
        extras = trajectory_extra_metrics()
        pooled = t["states"].mean(dim=1, keepdim=True).expand_as(
            t["states"]).contiguous()

        d1 = run_d1(t["states"], t["waypoints"], t["eps"], unit="camera",
                    i2=i2, pooled_states=pooled, extra_metrics=extras)
        d2 = run_d2(t["z_prev"], t["z_true1"], t["z_imag1"], t["disp1"],
                    t["eps"], i2=i2, actions=t["actions"],
                    prev_state=t["prev_state"], fit_meta=corpus_meta,
                    run_meta=corpus_meta, extra_metrics=extras)
        d3 = run_d3(t["z_prev"], t["z_true_k"], t["z_imag_k"], t["disp_k"],
                    t["eps"], i2=i2, extra_metrics=extras)
        spec = estimate_transition_spectrum(t["z_prev"], t["actions"],
                                            t["z_true1"])
    spec_d = spec.to_dict() if hasattr(spec, "to_dict") else vars(spec)
    spec_d = {k: (v.tolist()[:32] if isinstance(v, torch.Tensor) else v)
              for k, v in spec_d.items()}
    report = gates_metrics_json(exp_id, git_hash, [d1, d2, d3], extra={
        "d3_horizon_s": k_max / 10.0,
        "n_eval_windows": int(t["states"].shape[0]),
        "speed_input": speed_input,
        "spectral": spec_d,
    })
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=24,
                    help="val episodes per cache dir")
    ap.add_argument("--git-hash", default="unknown")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tanitad.data.comma2k19 import CORPUS_META
    # Self-describing ckpt: build at the TRAINED action_dim (speed-input ckpts
    # are 3-ch) so the load stays strict — never strict=False (that silently
    # leaves act_emb/inv_dyn random-init).
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world, speed_input, _src = build_world_from_ckpt(base250cam_config(), ck,
                                                     ckpt_path=args.ckpt)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1

    episodes = []
    for cd in args.cache_dirs:
        val_dirs = sorted(Path(cd).glob("*val*"))
        if val_dirs:
            files = sorted(val_dirs[-1].glob("ep_*.pt"))[:args.episodes]
            episodes += [load_episode(str(p), mmap=True) for p in files]
    print(f"[eval] {len(episodes)} val episodes, checkpoint step {step}, "
          f"speed_input {speed_input}")

    report = evaluate(world, episodes, device,
                      exp_id=f"p0-sB01-gates-step{step}",
                      git_hash=args.git_hash, corpus_meta=CORPUS_META,
                      speed_input=speed_input)
    out = Path(args.out) / f"gates_step{step}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["summary"], indent=2))
    for g in report["gates"]:
        print(f"  {g['gate']}: {g['verdict']}")
    print(f"[eval] full report -> {out}")


if __name__ == "__main__":
    main()
