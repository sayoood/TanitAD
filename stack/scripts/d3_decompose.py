"""D3/I4 decomposition — WHY is multi-step imagination blocked? (Sayed, 2026-07-09)

Three questions on a mid-training checkpoint, each with a decision attached:

  Q1  How does imag-rel grow with horizon k in {1,2,4}?
      superlinear -> compounding/consistency -> K-step training is the medicine
  Q2  Does recursively composing the 1-STEP head (with true future actions)
      beat the direct 4-step head?
      yes -> the direct multi-horizon head is the weak part (recursion or
             K-step fixes it);  no -> errors compound, consistency confirmed
  Q3  Is one corpus (comma highway vs physicalai urban) driving the failure?

Usage (pod1):
  python scripts/d3_decompose.py --ckpt .../ckpt.pt \
      --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache \
      --out /workspace/experiments/d3_decompose.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.instruments.numerics import strict_numerics


@torch.no_grad()
def collect(world, episodes, device, window, kmax=4, stride=8, batch=4,
            max_windows=300):
    out = {"z": [], "fut": [], "aw": [], "fa": []}
    n = 0
    for ep in episodes:
        frames = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = frames.shape[0]
        for i in range(0, T - window - kmax, stride):
            out["aw"].append(ep.actions[i:i + window])
            out["fa"].append(ep.actions[i + window:i + window + kmax])
            n += 1
        for i0 in range(0, T - window - kmax, stride):
            pass
        # frames handled in batched second pass below
    # second pass: batched encode (memory-light)
    idx = 0
    zs, futs = [], []
    for ep in episodes:
        frames = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = frames.shape[0]
        starts = list(range(0, T - window - kmax, stride))
        for j in range(0, len(starts), batch):
            chunk = starts[j:j + batch]
            fw = torch.stack([frames[t:t + window] for t in chunk]).to(device)
            zs.append(world.encode_window(fw).cpu())
            ff = torch.stack([frames[t + window:t + window + kmax]
                              for t in chunk]).to(device)
            futs.append(world.encode_window(ff).cpu())
            idx += len(chunk)
        if idx >= max_windows:
            break
    m = min(idx, max_windows, len(out["aw"]))
    return (torch.cat(zs)[:m], torch.cat(futs)[:m],
            torch.stack(out["aw"][:m]), torch.stack(out["fa"][:m]))


@torch.no_grad()
def analyze(world, states, futs, aw, fa, device):
    """imag-rel per horizon (direct heads) + recursive 1-step composition."""
    states, futs = states.to(device), futs.to(device)
    aw, fa = aw.to(device), fa.to(device)
    z_t = states[:, -1]
    preds = world.imagine(states, aw)
    res = {}
    for k in sorted(preds):
        true_k = futs[:, k - 1]
        rel = ((preds[k] - true_k).norm(dim=-1)
               / (true_k - z_t).norm(dim=-1).clamp_min(1e-8))
        res[f"direct_k{k}"] = round(float(rel.median()), 3)
    # recursive: roll the 1-step head with TRUE future actions
    win_s, win_a = states, aw
    for j in range(4):
        z_hat = world.predictor(win_s, win_a)[1]
        if j < 3:
            win_s = torch.cat([win_s[:, 1:], z_hat.unsqueeze(1)], dim=1)
            win_a = torch.cat([win_a[:, 1:], fa[:, j].unsqueeze(1)], dim=1)
    true_4 = futs[:, 3]
    rel4 = ((z_hat - true_4).norm(dim=-1)
            / (true_4 - z_t).norm(dim=-1).clamp_min(1e-8))
    res["recursive_1step_x4"] = round(float(rel4.median()), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tanitad.models.fourbrain import WorldModel
    world = WorldModel(base250cam_config())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.to(device).eval()
    window = world.predictor.cfg.window

    report = {"exp": "d3-decompose", "ckpt_step": step}
    with strict_numerics():
        for cd in args.cache_dirs:
            name = Path(cd).parent.name                      # corpus dir name
            val = sorted(Path(cd).glob("*val*"))
            eps = [load_episode(str(p), mmap=True)
                   for p in sorted(val[-1].glob("ep_*.pt"))[:16]]
            s, f, aw, fa = collect(world, eps, device, window)
            report[name] = analyze(world, s, f, aw, fa, device)
            report[name]["n_windows"] = int(s.shape[0])
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
