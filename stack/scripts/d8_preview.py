"""SC-05 / D8 preview: imagination-error separation, in-domain vs degraded weather.

First measured numbers for the self-knowledge edge (H11/A9) on a PARTIAL
checkpoint: does the free familiarity signal e = |z_imag - z_true| / |dz_true|
already separate the training domain (comma val) from degraded-visibility
synthetic driving (Cosmos foggy/rainy/snowy/night)?

Three groups, two axes (P8 — cosmos is synthetic, so domain shift confounds
weather; the clear-weather cosmos group isolates the weather axis):

    comma-val        in-domain baseline (route-level held-out)
    cosmos-clear     domain shift only        (sunny / morning / golden_hour)
    cosmos-degraded  domain + weather shift   (foggy / rainy / snowy / night)

Reports window-level imag_rel stats per group and rank AUROCs. This is a
DIAGNOSTIC preview for SCENARIO_DATABASE SC-05 — gate D8 proper runs on the
trained checkpoint against real OOD probes (nuScenes, never trained).

Usage (local 4060):
  python stack/scripts/d8_preview.py --ckpt .../ckpt_full.pt \
      --comma-cache .../comma2k19-val-<hash> --cosmos-root .../extracted \
      --out .../d8_preview.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tanitad.config import base250cam_config
from tanitad.data.cosmos_drive import build_episode, discover_clips
from tanitad.data.mixing import load_episode
from tanitad.instruments.numerics import strict_numerics

CLEAR = {"sunny", "morning", "golden_hour"}
DEGRADED = {"foggy", "rainy", "snowy", "night"}


@torch.no_grad()
def imag_rel_scores(world, episodes, device, window: int,
                    stride: int = 6, batch: int = 4,
                    max_windows: int = 400) -> torch.Tensor:
    """Window-level 1-step relative imagination error (the A9 signal)."""
    scores = []
    for ep in episodes:
        frames = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = frames.shape[0]
        t0s = list(range(0, T - window - 1, stride))
        for i in range(0, len(t0s), batch):
            chunk = t0s[i:i + batch]
            fw = torch.stack([frames[t:t + window] for t in chunk]).to(device)
            aw = torch.stack([ep.actions[t:t + window] for t in chunk]).to(device)
            states = world.encode_window(fw)
            z_prev = states[:, -1]
            z_imag1 = world.imagine(states, aw)[1]
            z_true1 = world.encode(
                torch.stack([frames[t + window] for t in chunk]).to(device))
            err = (z_imag1 - z_true1).norm(dim=-1)
            step = (z_true1 - z_prev).norm(dim=-1).clamp_min(1e-8)
            scores.append((err / step).cpu())
        if sum(s.numel() for s in scores) >= max_windows:
            break
    out = torch.cat(scores) if scores else torch.empty(0)
    return out[:max_windows]


def rank_auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(score_pos > score_neg) via Mann-Whitney U (ties get half credit)."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    all_s = torch.cat([pos, neg])
    order = all_s.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, all_s.numel() + 1, dtype=torch.float64)
    # average ranks over ties
    uniq, inv, counts = torch.unique(all_s, return_inverse=True,
                                     return_counts=True)
    if (counts > 1).any():
        sums = torch.zeros_like(uniq, dtype=torch.float64)
        sums.scatter_add_(0, inv, ranks)
        ranks = (sums / counts.to(torch.float64))[inv]
    r_pos = ranks[:pos.numel()].sum().item()
    n_p, n_n = pos.numel(), neg.numel()
    return (r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n)


def _stats(x: torch.Tensor) -> dict:
    return {"n": int(x.numel()), "mean": round(float(x.mean()), 4),
            "median": round(float(x.median()), 4),
            "p90": round(float(x.quantile(0.9)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--cosmos-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=12)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tanitad.models.fourbrain import WorldModel
    cfg = base250cam_config()
    world = WorldModel(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.to(device).eval()
    window = world.predictor.cfg.window

    comma = [load_episode(str(p), mmap=True)
             for p in sorted(Path(args.comma_cache).glob("ep_*.pt"))[:args.max_eps]]

    root = Path(args.cosmos_root)
    mp4s = list(root.rglob("*.mp4"))
    cam = str(mp4s[0].parent.relative_to(root)) if mp4s else None
    clips = discover_clips(root, camera_subdir=cam)
    groups = {"cosmos_clear": [c for c in clips if c["weather"] in CLEAR],
              "cosmos_degraded": [c for c in clips if c["weather"] in DEGRADED]}
    print(f"[d8] ckpt step {step}; comma eps {len(comma)}; "
          f"clear {len(groups['cosmos_clear'])} / degraded "
          f"{len(groups['cosmos_degraded'])} clips")

    with strict_numerics():
        s = {"comma_val": imag_rel_scores(world, comma, device, window)}
        for name, cl in groups.items():
            eps = []
            for c in cl:
                try:
                    eps.append(build_episode(c, size=cfg.encoder.image_size))
                except Exception as e:
                    print(f"[d8] skip {c['clip_id']}: {type(e).__name__}")
            s[name] = imag_rel_scores(world, eps, device, window)

    report = {
        "exp": "d8-preview-sc05", "ckpt_step": step,
        "caveat": ("partial checkpoint (~{:.0%} of 30k); cosmos is synthetic "
                   "=> domain shift confounds weather; diagnostic, not gate D8"
                   ).format(step / 30000 if step > 0 else 0),
        "groups": {k: _stats(v) for k, v in s.items()},
        "auroc": {
            "comma_vs_degraded": round(rank_auroc(s["cosmos_degraded"],
                                                  s["comma_val"]), 4),
            "comma_vs_clear": round(rank_auroc(s["cosmos_clear"],
                                               s["comma_val"]), 4),
            "clear_vs_degraded_weather_axis": round(
                rank_auroc(s["cosmos_degraded"], s["cosmos_clear"]), 4),
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
