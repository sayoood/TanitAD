"""Run the transition-spectrum analysis on a checkpoint (L2 / p0-spectral-sizing).

Loads a training checkpoint + cached validation episodes, collects
(z_t, a_t, z_{t+1}) pairs through the CURRENT encoder, and prints the
SpectrumReport. Per the intake caveat: on an early/undertrained checkpoint
this is a DIAGNOSTIC (spectrum shape / effective rank), NOT a sizing claim —
decision-grade sizing needs the trained Stage-B checkpoint.

Usage (pod):
  python scripts/run_spectral.py --ckpt /workspace/experiments/p0-sB01-realmix/ckpt.pt \
      --cache-dir /workspace/data/comma2k19/_epcache --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.eval.spectral import estimate_transition_spectrum, pairs_from_states
from tanitad.instruments.numerics import strict_numerics
from tanitad.models.fourbrain import WorldModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="an _epcache dir; the newest val cache inside is used")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = base250cam_config()
    world = WorldModel(cfg).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    state = ck["model"] if "model" in ck else ck
    world.load_state_dict(state)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    print(f"[spectral] checkpoint loaded (step {step})")

    val_dirs = sorted(Path(args.cache_dir).glob("*val*"))
    assert val_dirs, f"no val cache under {args.cache_dir}"
    eps = [load_episode(str(p), mmap=True)
           for p in sorted(val_dirs[-1].glob("ep_*.pt"))[:args.episodes]]
    print(f"[spectral] {len(eps)} val episodes from {val_dirs[-1].name}")

    zs, acts = [], []
    with torch.no_grad(), strict_numerics():
        for ep in eps:
            frames = (ep.frames.float() / 255.0 if ep.frames.dtype == torch.uint8
                      else ep.frames).to(device)
            z = torch.cat([world.encode(frames[i:i + 32])
                           for i in range(0, frames.shape[0], 32)])
            zs.append(z.cpu())
            acts.append(ep.actions)
    # per-episode pairs, concatenated (no cross-episode transitions)
    z_t, a_t, z_next = [], [], []
    for z, a in zip(zs, acts):
        zt, at, zn = pairs_from_states(z.unsqueeze(0), a.unsqueeze(0))
        z_t.append(zt); a_t.append(at); z_next.append(zn)
    z_t, a_t, z_next = torch.cat(z_t), torch.cat(a_t), torch.cat(z_next)
    print(f"[spectral] {z_t.shape[0]} transition pairs, dim {z_t.shape[1]}")

    report = estimate_transition_spectrum(z_t, a_t, z_next)
    d = report.to_dict() if hasattr(report, "to_dict") else vars(report)
    d = {k: (v.tolist()[:64] if isinstance(v, torch.Tensor) else v)
         for k, v in d.items()}
    d["checkpoint_step"] = step
    d["DIAGNOSTIC_NOTE"] = ("early-checkpoint spectrum = diagnostic only; "
                            "sizing claims need the trained checkpoint")
    out = args.out or (Path(args.ckpt).parent / f"spectral_step{step}.json")
    Path(out).write_text(json.dumps(d, indent=2, default=str))
    for k in ("fit_r2", "operator_effective_rank", "energy_knee_k",
              "recommendation"):
        if k in d:
            print(f"  {k}: {d[k]}")
    print(f"[spectral] full report -> {out}")


if __name__ == "__main__":
    main()
