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

Reports window-level stats per group and rank AUROCs for THREE scores (the
2026-07-08 redesign after the first preview showed the naive ratio is
confounded by scene-change rate):

    rel   |z_imag - z_true| / |dz_true|   (the original A9 ratio; kept for trend)
    abs   |z_imag - z_true|               (no step-size normalization)
    maha  diagonal Mahalanobis of z_true vs a reference fitted on HALF the
          comma windows (the other half is scored -> honest in-domain baseline)

Optional matched-pairs mode (--pairs-root, from scripts/cosmos_pairs.py): the
SAME scene under clear vs degraded weather — isolates the weather axis from
scene content. This is a DIAGNOSTIC preview for SCENARIO_DATABASE SC-05 — gate
D8 proper runs on the trained checkpoint against real OOD probes (nuScenes,
never trained).

Usage (local 4060):
  python stack/scripts/d8_preview.py --ckpt .../ckpt_full.pt \
      --comma-cache .../comma2k19-val-<hash> --cosmos-root .../extracted \
      [--pairs-root .../pairs] --out .../d8_preview.json
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
def collect(world, episodes, device, window: int, stride: int = 6,
            batch: int = 4, max_windows: int = 400) -> dict:
    """Per-window latents: z_prev, z_imag1, z_true1, plus the episode index."""
    out = {"z_prev": [], "z_imag1": [], "z_true1": [], "ep": []}
    n = 0
    for ei, ep in enumerate(episodes):
        frames = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = frames.shape[0]
        t0s = list(range(0, T - window - 1, stride))
        for i in range(0, len(t0s), batch):
            chunk = t0s[i:i + batch]
            fw = torch.stack([frames[t:t + window] for t in chunk]).to(device)
            aw = torch.stack([ep.actions[t:t + window] for t in chunk]).to(device)
            states = world.encode_window(fw)
            out["z_prev"].append(states[:, -1].cpu())
            out["z_imag1"].append(world.imagine(states, aw)[1].cpu())
            out["z_true1"].append(world.encode(
                torch.stack([frames[t + window] for t in chunk]).to(device)).cpu())
            out["ep"].extend([ei] * len(chunk))
            n += len(chunk)
        if n >= max_windows:
            break
    if not out["z_prev"]:
        return {k: torch.empty(0) for k in out}
    return {"z_prev": torch.cat(out["z_prev"])[:max_windows],
            "z_imag1": torch.cat(out["z_imag1"])[:max_windows],
            "z_true1": torch.cat(out["z_true1"])[:max_windows],
            "ep": torch.tensor(out["ep"])[:max_windows]}


def rel_score(c: dict) -> torch.Tensor:
    err = (c["z_imag1"] - c["z_true1"]).norm(dim=-1)
    step = (c["z_true1"] - c["z_prev"]).norm(dim=-1).clamp_min(1e-8)
    return err / step


def abs_score(c: dict) -> torch.Tensor:
    return (c["z_imag1"] - c["z_true1"]).norm(dim=-1)


def fit_diag_gauss(z: torch.Tensor, shrink: float = 0.1) -> tuple:
    """Diagonal Gaussian reference with variance shrinkage toward the mean var."""
    mu = z.mean(dim=0)
    var = z.var(dim=0, unbiased=True)
    var = (1 - shrink) * var + shrink * var.mean()
    return mu, var.clamp_min(1e-8)


def maha_score(z: torch.Tensor, mu: torch.Tensor,
               var: torch.Tensor) -> torch.Tensor:
    return ((z - mu) ** 2 / var).mean(dim=-1)


def imag_rel_scores(world, episodes, device, window: int, stride: int = 6,
                    batch: int = 4, max_windows: int = 400) -> torch.Tensor:
    """Window-level 1-step relative imagination error (the A9 signal)."""
    return rel_score(collect(world, episodes, device, window, stride, batch,
                             max_windows))


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


def _build_clips(clips, size):
    eps, keys = [], []
    for c in clips:
        try:
            eps.append(build_episode(c, size=size))
            keys.append((c["clip_id"], int(c.get("chunk", 0)), c["weather"]))
        except Exception as e:
            print(f"[d8] skip {c['clip_id']}: {type(e).__name__}")
    return eps, keys


def matched_pairs_report(world, pairs_root: Path, device, window: int,
                         size: int) -> dict:
    """Per-scene clear-vs-degraded comparison: same (base,chunk), two weathers."""
    mp4s = list(pairs_root.rglob("*.mp4"))
    cam = str(mp4s[0].parent.relative_to(pairs_root)) if mp4s else None
    clips = discover_clips(pairs_root, camera_subdir=cam)
    eps, keys = _build_clips(clips, size)
    per_clip = {}
    for ep, key in zip(eps, keys):
        c = collect(world, [ep], device, window, stride=4, max_windows=64)
        if c["z_prev"].numel() == 0:
            continue
        per_clip[key] = {"abs": float(abs_score(c).mean()),
                         "rel": float(rel_score(c).mean())}
    pairs = []
    scenes = {}
    for (base, chunk, weather), v in per_clip.items():
        scenes.setdefault((base, chunk), {})[weather] = v
    for (base, chunk), wmap in scenes.items():
        cl = [w for w in wmap if w in CLEAR]
        dg = [w for w in wmap if w in DEGRADED]
        if cl and dg:
            pairs.append({"scene": f"{base}_{chunk}",
                          "clear_w": cl[0], "degraded_w": dg[0],
                          "abs_diff": round(wmap[dg[0]]["abs"]
                                            - wmap[cl[0]]["abs"], 4),
                          "rel_diff": round(wmap[dg[0]]["rel"]
                                            - wmap[cl[0]]["rel"], 4)})
    n = len(pairs)
    frac = (sum(p["abs_diff"] > 0 for p in pairs) / n) if n else float("nan")
    diffs = sorted(p["abs_diff"] for p in pairs)
    return {"n_pairs": n,
            "frac_degraded_higher_abs": round(frac, 4) if n else None,
            "median_abs_diff": diffs[n // 2] if n else None,
            "pairs": pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--cosmos-root", required=True)
    ap.add_argument("--pairs-root", default=None)
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
    size = cfg.encoder.image_size

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
        col = {"comma_val": collect(world, comma, device, window)}
        for name, cl in groups.items():
            eps, _ = _build_clips(cl, size)
            col[name] = collect(world, eps, device, window)

        # split comma windows: fit half -> reference, score half -> baseline
        cc = col["comma_val"]
        half = cc["z_true1"].shape[0] // 2
        mu, var = fit_diag_gauss(cc["z_true1"][:half])
        scores = {}
        for name, c in col.items():
            z = c["z_true1"][half:] if name == "comma_val" else c["z_true1"]
            sub = {k: (v[half:] if name == "comma_val" else v)
                   for k, v in c.items() if k != "ep"}
            scores[name] = {"rel": rel_score(sub), "abs": abs_score(sub),
                            "maha": maha_score(z, mu, var)}

        pairs = None
        if args.pairs_root:
            pairs = matched_pairs_report(world, Path(args.pairs_root), device,
                                         window, size)

    report = {
        "exp": "d8-preview-sc05-v2", "ckpt_step": step,
        "caveat": ("partial checkpoint (~{:.0%} of 30k); cosmos is synthetic "
                   "=> domain shift confounds weather; diagnostic, not gate D8"
                   ).format(step / 30000 if step > 0 else 0),
        "groups": {name: {k: _stats(v) for k, v in sc.items()}
                   for name, sc in scores.items()},
        "auroc": {k: {
            "comma_vs_degraded": round(rank_auroc(
                scores["cosmos_degraded"][k], scores["comma_val"][k]), 4),
            "comma_vs_clear": round(rank_auroc(
                scores["cosmos_clear"][k], scores["comma_val"][k]), 4),
            "clear_vs_degraded_weather_axis": round(rank_auroc(
                scores["cosmos_degraded"][k], scores["cosmos_clear"][k]), 4),
        } for k in ("rel", "abs", "maha")},
        "matched_pairs": pairs,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
