"""fix-B1 PRIMARY eval: the EMERGENT rollout trajectory of the grounded model.

Unlike the post-hoc probe ladder in ``driving_diagnostic.py`` (which fits a ridge/
MLP on the frozen latent), this decodes a trajectory the way the model would drive
it: roll the operative predictor forward under the TRUE action sequence, decode
each PREDICTED latent's per-step metric Δpose with the trained
``StepDisplacementReadout``, and ACCUMULATE (SE(2) dead-reckoning) into ego
waypoints at {5,10,15,20} = {0.5,1,1.5,2}s. No fitting at eval — the readout was
trained during the fine-tune. This is arm (c) ("model trajectory head"), which was
N/A for the pre-fix checkpoint.

Reported (route-resampled, fp32, strict numerics), against the pre-registered
baselines held-out=3.89 / oracle=1.65 / CV=0.28:
  - rollout ADE@1s (== ade_0_2s, the 4-waypoint mean the plan keys "ADE@1s") and
    per-horizon de@Ts, mean ± 95% CI over route splits;
  - straight/gentle/sharp stratified rollout ADE vs constant-velocity (does the
    model beat CV on the straight stratum?);
  - by-corpus breakdown.

Conventions (GT waypoints, CV baseline, curvature strata, metric helpers) are
imported verbatim from ``driving_diagnostic.py`` so this number is comparable to
the diagnostic's probe numbers on the same windows.

Usage (pod1):
  python scripts/eval_metric_rollout.py \
     --ckpt /workspace/experiments/finetune_traj/ckpt.pt \
     --cache-dirs /workspace/data/comma2k19/_epcache /workspace/data/physicalai/_epcache \
     --out /workspace/experiments/finetune_traj/rollout_eval.json --episodes 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from driving_diagnostic import (BASELINES, WP_STEPS, agg_metric_dicts,  # noqa: E402
                                baseline_waypoints, curvature_bucket, de_of,
                                gt_ego_waypoints, mean_ci,
                                net_heading_change_deg, scalar_metrics, _strat)

from tanitad.config import base250cam_config, smoke_config  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.eval.gates import split_by_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.models.metric_dynamics import (StepDisplacementReadout,  # noqa: E402
                                            rollout_decode)

K_MAX = max(WP_STEPS)


@torch.no_grad()
def collect(world, step_readout, episodes, corpora, device, window, fwd_k,
            stride, batch):
    """Encode each window, roll ``fwd_k`` steps under TRUE actions, decode+
    accumulate -> rollout waypoints at WP_STEPS. Also GT + CV baseline + meta."""
    S_wp, GT, CV, EID, COR, SPD, HDG = [], [], [], [], [], [], []
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    need_ahead = max(K_MAX, fwd_k)                        # frames consumed ahead
    for ep, corp in zip(episodes, corpora):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - window - need_ahead, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + fwd_k]
                              for t in ch]).to(device)
            states = world.encode_window(fw)                     # [b, W, S]
            wp_full, _ = rollout_decode(world.predictor, states, aw, fa,
                                        step_readout, fwd_k)      # [b, fwd_k, 2]
            pred_wp = wp_full.index_select(1, wp_idx.to(device)).cpu()  # [b,4,2]
            last = torch.tensor([t + window - 1 for t in ch])
            S_wp.append(pred_wp.float())
            GT.append(gt_ego_waypoints(ep.poses, last))
            CV.append(baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            COR.extend([corp] * len(ch))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))
    return {"pred": torch.cat(S_wp).float(), "gt": torch.cat(GT).float(),
            "cv": torch.cat(CV).float(), "eid": EID, "corpus": COR,
            "speed": torch.cat(SPD).float(), "head_deg": torch.cat(HDG).float()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fine-tune ckpt (has step_readout)")
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--fwd-k", type=int, default=K_MAX)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--git-hash", default="unknown")
    ap.add_argument("--config", choices=["base250cam", "smoke"],
                    default="base250cam")
    args = ap.parse_args()

    assert args.fwd_k >= K_MAX, (
        f"--fwd-k {args.fwd_k} < {K_MAX}: the rollout must reach the 2 s waypoint "
        f"(step {K_MAX}) to decode waypoints at {WP_STEPS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = smoke_config() if args.config == "smoke" else base250cam_config()

    def corpus_of(cd: str) -> str:
        low = cd.lower()
        return ("comma2k19" if "comma" in low
                else "physicalai" if "physical" in low else Path(cd).name)

    episodes, corpora = [], []
    for cd in args.cache_dirs:
        val_dirs = sorted(Path(cd).glob("*val*"))
        if not val_dirs:
            print(f"[rollout] WARNING no *val* under {cd}", flush=True)
            continue
        for p in sorted(val_dirs[-1].glob("ep_*.pt"))[:args.episodes]:
            episodes.append(load_episode(str(p), mmap=True))
            corpora.append(corpus_of(cd))
    assert episodes, "no val episodes loaded"

    world = WorldModel(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    world = world.to(device).eval()
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    step_readout = StepDisplacementReadout(world.state_dim).to(device).eval()
    assert "step_readout" in ck, (
        "ckpt has no 'step_readout' — was this produced by finetune_traj "
        "--mode dynamics? (rollout eval needs the trained step readout)")
    step_readout.load_state_dict(ck["step_readout"])
    window = world.predictor.cfg.window
    print(f"[rollout] {len(episodes)} val episodes, ckpt step {step}, "
          f"window {window}, fwd_k {args.fwd_k}, device {device}", flush=True)

    with strict_numerics():
        data = collect(world, step_readout, episodes, corpora, device, window,
                       args.fwd_k, args.stride, args.batch)
        n = data["pred"].shape[0]
        splits = [split_by_episode(data["eid"], args.val_frac, s)
                  for s in range(args.seed, args.seed + args.n_splits)]

        pred, gt, cv = data["pred"], data["gt"], data["cv"]
        model_de = de_of(pred, gt)
        cv_de = de_of(cv, gt)

        # route-resampled held-out (predictions fixed; CI = route sampling)
        roll_split = [scalar_metrics(model_de[va]) for _tr, va in splits]
        cv_split = [scalar_metrics(cv_de[va]) for _tr, va in splits]
        rollout_metrics = agg_metric_dicts(roll_split)
        cv_metrics = agg_metric_dicts(cv_split)
        full = scalar_metrics(model_de)

        # stratify (seed-0 val split) by curvature / speed / corpus vs CV
        _tr0, va0 = splits[0]
        curv = [curvature_bucket(float(h)) for h in data["head_deg"][va0]]
        q = torch.quantile(data["speed"], torch.tensor([1 / 3, 2 / 3]))
        t1, t2 = float(q[0]), float(q[1])
        spd = ["low" if float(s) < t1 else "high" if float(s) >= t2 else "med"
               for s in data["speed"][va0]]
        cor = [data["corpus"][i] for i in va0]
        strat = {
            "n_val_windows": len(va0),
            "by_curvature": _strat(curv, model_de[va0], cv_de[va0]),
            "by_speed": _strat(spd, model_de[va0], cv_de[va0]),
            "by_corpus": _strat(cor, model_de[va0], cv_de[va0]),
        }

    beats_cv_straight = None
    straight = strat["by_curvature"].get("straight")
    if straight is not None:
        beats_cv_straight = bool(straight["model_ade@1s"] < straight["cv_ade@1s"])

    report = {
        "exp": "fix-b1-metric-rollout-eval",
        "ckpt": args.ckpt, "step": step, "git_hash": args.git_hash,
        "method": ("predictor rollout under TRUE actions -> per-step Δpose via "
                   "StepDisplacementReadout -> SE(2) accumulation; NO fit at eval"),
        "n_windows_total": n,
        "config": {"episodes_per_dir": args.episodes, "fwd_k": args.fwd_k,
                   "stride": args.stride, "window": window,
                   "n_splits": args.n_splits, "waypoint_steps": list(WP_STEPS),
                   "fp32": True, "strict_numerics": True},
        "preregistered_baselines": {"held_out_mlp_ade@1s": 3.89,
                                    "oracle_ceiling_ade@1s": 1.65,
                                    "constant_velocity_ade@1s": 0.28},
        "rollout_heldout": rollout_metrics,
        "rollout_full_set": full,
        "constant_velocity": cv_metrics,
        "straight_stratum_beats_cv": beats_cv_straight,
        "error_localization": strat,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print("\n=== METRIC ROLLOUT EVAL ===", flush=True)
    print(f"  rollout  ade_0_2s={rollout_metrics['ade_0_2s']['mean']:.3f}"
          f"±{rollout_metrics['ade_0_2s']['ci95']:.3f}  "
          f"de@1s={rollout_metrics['de@1s']['mean']:.3f}  "
          f"de@2s={rollout_metrics['de@2s']['mean']:.3f}", flush=True)
    print(f"  CV       ade_0_2s={cv_metrics['ade_0_2s']['mean']:.3f}  "
          f"de@1s={cv_metrics['de@1s']['mean']:.3f}", flush=True)
    for lab, v in strat["by_curvature"].items():
        print(f"  [{lab:8s}] model_ade@1s={v['model_ade@1s']:.3f} "
              f"cv_ade@1s={v['cv_ade@1s']:.3f} n={v['n']}", flush=True)
    print(f"  straight_beats_cv={beats_cv_straight}", flush=True)
    print(f"[rollout] report -> {args.out}", flush=True)
    print("METRIC_ROLLOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
