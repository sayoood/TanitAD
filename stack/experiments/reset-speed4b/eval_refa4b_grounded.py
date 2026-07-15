"""REF-A 4-BRAIN grounded-rollout eval — the 4-brain/speed/temporal twin of
eval_refa_grounded.py (which builds a vanilla RefAModel and cannot load a
4-brain checkpoint: extra policy keys + intent_proj + action_dim 3 + temporal
adapter all mismatch a strict load).

Only differences vs eval_refa_grounded.py (metric definition, CV baseline,
stratification, 8-split aggregation all reused VERBATIM — same imported
functions, so the number is apples-to-apples with the operative-only gate):
  * model: RefAModelPlus.from_stack_config(flagship4b_config()) with adapter
    'temporal' and (speed-input) action_dim=3 — the SAME build path as the
    trainer, so ck['model'] loads strict.
  * actions: current ego-speed v0 = poses[last,3]/SPEED_SCALE appended as the
    3rd action channel to BOTH the window and future actions — identical to the
    trainer's compute_losses_plus, so the predictor sees the inputs it trained on.
  * the operative rollout runs intent=None (operative-alone) — the conservative
    grounded metric, matching how the operative-only run was evaluated.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/tmp/refa_plus")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/TanitAD/stack")

from driving_diagnostic import (WP_STEPS, agg_metric_dicts, baseline_waypoints,
                                curvature_bucket, de_of, gt_ego_waypoints,
                                net_heading_change_deg, scalar_metrics, _strat)
from refa_plus import RefAModelPlus
from tanitad.config import flagship4b_config
from tanitad.eval.gates import split_by_episode
from tanitad.instruments.numerics import strict_numerics
from tanitad.models.metric_dynamics import StepDisplacementReadout, rollout_decode
from tanitad.refs.refa import refa_predictor_config

K_MAX = max(WP_STEPS)
SPEED_SCALE = 10.0        # MUST match refa_train_plus.SPEED_SCALE


class EpWrap:
    """Expose the fields the driving_diagnostic helpers read, over a
    feature-cache episode dict."""

    def __init__(self, d, eid):
        self.feats = d["feats_fp16"]            # [T, N, D] fp16 (mmap)
        self.actions = d["actions"].float()     # [T, 2]
        self.poses = d["poses"].float()         # [T, 4] = (x, y, yaw, v)
        self.episode_id = eid


@torch.no_grad()
def collect(model, step_readout, episodes, corpora, device, window, fwd_k,
            stride, batch, speed_input=False):
    S_wp, GT, CV, EID, COR, SPD, HDG = [], [], [], [], [], [], []
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    need_ahead = max(K_MAX, fwd_k)
    for ep, corp in zip(episodes, corpora):
        feats = ep.feats
        T = feats.shape[0]
        starts = list(range(0, T - window - need_ahead, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([feats[t:t + window] for t in ch]).to(device)
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + fwd_k]
                              for t in ch]).to(device)
            if speed_input:
                # v0 = current ego-speed at the last observed frame (leakage-safe:
                # it is an OBSERVED quantity, the same one training appended).
                v0 = (ep.poses[last, 3:4] / SPEED_SCALE).to(device)   # [b, 1]
                aw = torch.cat([aw, v0.unsqueeze(1).expand(-1, aw.shape[1], -1)],
                               dim=-1)
                fa = torch.cat([fa, v0.unsqueeze(1).expand(-1, fa.shape[1], -1)],
                               dim=-1)
            states = model.encode_window(fw)                       # [b, W, S]
            wp_full, _ = rollout_decode(model.predictor, states, aw, fa,
                                        step_readout, fwd_k)        # [b, fwd_k, 2]
            pred_wp = wp_full.index_select(1, wp_idx.to(device)).cpu()
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


def build_model(args):
    """Match the trainer's build so ck['model'] loads strict."""
    if args.four_brain:
        cfg = flagship4b_config()
        if args.speed_input:
            object.__setattr__(cfg.predictor, "action_dim", 3)
            if cfg.tactical_pred is not None:
                object.__setattr__(cfg.tactical_pred, "action_dim", 3)
        return RefAModelPlus.from_stack_config(cfg, n_tokens=256,
                                               adapter_kind=args.adapter)
    pc = refa_predictor_config()
    if args.speed_input:
        pc = dataclasses.replace(pc, action_dim=3)
    return RefAModelPlus(pc, adapter_kind=args.adapter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--fwd-k", type=int, default=K_MAX)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default="temporal")
    ap.add_argument("--four-brain", action="store_true")
    ap.add_argument("--speed-input", action="store_true")
    args = ap.parse_args()
    assert args.fwd_k >= K_MAX
    device = "cuda" if torch.cuda.is_available() else "cpu"

    episodes, corpora = [], []
    for cd in args.cache_dirs:
        val_dirs = sorted(Path(cd).glob("*val*"))
        if not val_dirs:
            print(f"[refa4b-grounded] WARNING no *val* under {cd}", flush=True)
            continue
        files = sorted(val_dirs[-1].glob("ep_*.pt"))[:args.episodes]
        for i, p in enumerate(files):
            d = torch.load(str(p), map_location="cpu", weights_only=True,
                           mmap=True)
            episodes.append(EpWrap(d, i))
            corpora.append("physicalai")
    assert episodes, "no val episodes"

    model = build_model(args)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    step = int(ck.get("step", -1))
    step_readout = StepDisplacementReadout(model.state_dim).to(device).eval()
    step_readout.load_state_dict(ck["step_readout"])
    window = model.pred_cfg.window
    print(f"[refa4b-grounded] {len(episodes)} val eps, step {step}, window "
          f"{window}, fwd_k {args.fwd_k}, state_dim {model.state_dim}, "
          f"four_brain={args.four_brain} speed_input={args.speed_input}, "
          f"dev {device}", flush=True)

    with strict_numerics():
        data = collect(model, step_readout, episodes, corpora, device, window,
                       args.fwd_k, args.stride, args.batch,
                       speed_input=args.speed_input)
        n = data["pred"].shape[0]
        splits = [split_by_episode(data["eid"], args.val_frac, s)
                  for s in range(args.seed, args.seed + args.n_splits)]
        pred, gt, cv = data["pred"], data["gt"], data["cv"]
        model_de = de_of(pred, gt)
        cv_de = de_of(cv, gt)
        roll_split = [scalar_metrics(model_de[va]) for _tr, va in splits]
        cv_split = [scalar_metrics(cv_de[va]) for _tr, va in splits]
        rollout_metrics = agg_metric_dicts(roll_split)
        cv_metrics = agg_metric_dicts(cv_split)
        full = scalar_metrics(model_de)
        cv_full = scalar_metrics(cv_de)

        _tr0, va0 = splits[0]
        curv = [curvature_bucket(float(h)) for h in data["head_deg"][va0]]
        q = torch.quantile(data["speed"], torch.tensor([1 / 3, 2 / 3]))
        t1, t2 = float(q[0]), float(q[1])
        spd = ["low" if float(s) < t1 else "high" if float(s) >= t2 else "med"
               for s in data["speed"][va0]]
        cor = [data["corpus"][i] for i in va0]
        strat = {"n_val_windows": len(va0),
                 "by_curvature": _strat(curv, model_de[va0], cv_de[va0]),
                 "by_speed": _strat(spd, model_de[va0], cv_de[va0]),
                 "by_corpus": _strat(cor, model_de[va0], cv_de[va0])}

    straight = strat["by_curvature"].get("straight")
    grounded_beats_cv_straight = (
        bool(straight["model_ade@1s"] < straight["cv_ade@1s"])
        if straight is not None else None)
    grounded_beats_cv_overall = bool(
        rollout_metrics["ade_0_2s"]["mean"] < cv_metrics["ade_0_2s"]["mean"])

    report = {
        "exp": "refa-4brain-phase0-grounded-rollout", "ckpt": args.ckpt,
        "step": step,
        "arch": ("REF-A 4-brain (frozen-DINOv2 -> temporal adapter -> shared "
                 "operative predictor + tactical/strategic brains; speed-input)"),
        "method": ("operative predictor rollout (intent=None) under TRUE actions "
                   "+v0 -> per-step dpose via ck['step_readout'] -> SE(2) "
                   "accumulate; NO fit at eval (same protocol as "
                   "eval_refa_grounded / eval_grounded_rollout_4b)"),
        "n_windows_total": n, "n_val_episodes": len(episodes),
        "eval_config": {"episodes_per_dir": args.episodes, "fwd_k": args.fwd_k,
                        "stride": args.stride, "window": window,
                        "n_splits": args.n_splits, "val_frac": args.val_frac,
                        "waypoint_steps": list(WP_STEPS),
                        "four_brain": args.four_brain,
                        "speed_input": args.speed_input},
        "grounded_rollout_heldout": rollout_metrics,
        "grounded_rollout_full_set": full,
        "constant_velocity_heldout": cv_metrics,
        "constant_velocity_full_set": cv_full,
        "grounded_beats_cv_overall_ade_0_2s": grounded_beats_cv_overall,
        "straight_stratum": {
            "grounded_rollout_ade@1s":
                (straight["model_ade@1s"] if straight else None),
            "cv_ade@1s": (straight["cv_ade@1s"] if straight else None),
            "n": (straight["n"] if straight else 0),
            "grounded_beats_cv": grounded_beats_cv_straight},
        "error_localization": strat,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))

    rm = rollout_metrics
    cm = cv_metrics
    print("\n=== REF-A 4-BRAIN GROUNDED ROLLOUT ===", flush=True)
    print(f"  grounded ade_0_2s={rm['ade_0_2s']['mean']:.3f}"
          f"+/-{rm['ade_0_2s']['ci95']:.3f} (std {rm['ade_0_2s']['std']:.3f})  "
          f"de@1s={rm['de@1s']['mean']:.3f}  de@2s={rm['de@2s']['mean']:.3f}",
          flush=True)
    print(f"  CV       ade_0_2s={cm['ade_0_2s']['mean']:.3f}  "
          f"de@1s={cm['de@1s']['mean']:.3f}  de@2s={cm['de@2s']['mean']:.3f}",
          flush=True)
    print(f"  grounded_beats_cv_overall(ade_0_2s)={grounded_beats_cv_overall}",
          flush=True)
    for lab, v in strat["by_curvature"].items():
        lc = " LOWCONF" if v["low_confidence"] else ""
        print(f"  [{lab:8s}] grounded_ade@1s={v['model_ade@1s']:.3f} "
              f"cv_ade@1s={v['cv_ade@1s']:.3f} grounded_ade@2s={v['model_ade@2s']:.3f} "
              f"cv_ade@2s={v['cv_ade@2s']:.3f} n={v['n']}{lc}", flush=True)
    print(f"  straight grounded_beats_cv={grounded_beats_cv_straight}",
          flush=True)
    print(f"[refa4b-grounded] report -> {args.out}", flush=True)
    print("REFA4B_GROUNDED_DONE", flush=True)


if __name__ == "__main__":
    main()
