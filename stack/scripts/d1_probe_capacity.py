"""D1-regression discriminator: is position info LOST or just less LINEAR?

Fits probes of increasing capacity (ridge at 3 regularizations + a small MLP)
on latents from two checkpoints (14k vs current) and compares held-out route
ADE@1s. Interpretation (pre-registered):
  - MLP recovers 14k-level ADE on the later ckpt  -> info intact, linear
    readout mismatch (SigReg isotropization redistributes dims) -> remedy is
    readout/schedule, not a training crisis.
  - MLP also degrades                              -> true information loss ->
    escalate with schedule options.

Usage (pod1):
  python scripts/d1_probe_capacity.py --ckpts /workspace/ckpt14k_frozen.pt \
      /workspace/experiments/p0-sB01-realmix/ckpt.pt \
      --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache \
      --out /workspace/experiments/d1_probe_capacity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.instruments.numerics import strict_numerics

STEPS_1S = 10                     # ADE@1s target (10 steps @10Hz)


def _ego(dxy, yaw):
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


@torch.no_grad()
def collect(world, episodes, device, window, stride=8, batch=8):
    S, Y, E = [], [], []
    for ei, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - window - STEPS_1S, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)
            st = world.encode_window(fw)[:, -1].cpu()
            last = torch.tensor([t + window - 1 for t in ch])
            wp = _ego(ep.poses[last + STEPS_1S, :2] - ep.poses[last, :2],
                      ep.poses[last, 2])
            S.append(st); Y.append(wp); E.extend([ei] * len(ch))
    return torch.cat(S), torch.cat(Y), torch.tensor(E)


def fit_eval(S, Y, E, kind, alpha=10.0):
    """Route-level split by episode index parity (I3-style, deterministic)."""
    tr = (E % 2 == 0); va = ~tr
    Xtr, Ytr, Xva, Yva = S[tr], Y[tr], S[va], Y[va]
    if kind == "mlp":
        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.LayerNorm(S.shape[1]), torch.nn.Linear(S.shape[1], 256),
            torch.nn.GELU(), torch.nn.Linear(256, 2))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep_i in range(60):
            perm = torch.randperm(Xtr.shape[0])
            for j in range(0, len(perm), 512):
                b = perm[j:j + 512]
                loss = (net(Xtr[b]) - Ytr[b]).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = net(Xva)
    else:
        X = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], 1)
        W = torch.linalg.solve(X.T @ X + alpha * torch.eye(X.shape[1]),
                               X.T @ Ytr)
        Xv = torch.cat([Xva, torch.ones(Xva.shape[0], 1)], 1)
        pred = Xv @ W
    return round(float((pred - Yva).norm(dim=-1).mean()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs=2, required=True)
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    from tanitad.models.fourbrain import WorldModel

    eps = []
    for cd in args.cache_dirs:
        val = sorted(Path(cd).glob("*val*"))[-1]
        eps += [load_episode(str(p), mmap=True)
                for p in sorted(val.glob("ep_*.pt"))[:12]]

    report = {"exp": "d1-probe-capacity", "ade_1s_by_probe": {}}
    for ck_path in args.ckpts:
        world = WorldModel(base250cam_config())
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        world.load_state_dict(ck["model"] if "model" in ck else ck)
        step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
        world = world.to(device).eval()
        with strict_numerics():
            S, Y, E = collect(world, eps, device, world.predictor.cfg.window)
        row = {}
        for kind, a in (("ridge", 1.0), ("ridge", 10.0), ("ridge", 100.0),
                        ("mlp", None)):
            key = f"{kind}{'' if a is None else f'_a{a:g}'}"
            row[key] = fit_eval(S.float(), Y.float(), E, kind,
                                alpha=a or 10.0)
        report["ade_1s_by_probe"][f"step{step}"] = row
        print(f"step{step}: {row}", flush=True)
        del world
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("PROBE_CAPACITY_DONE", flush=True)


if __name__ == "__main__":
    main()
