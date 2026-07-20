"""Axis-6 pre-check 3: grounded-rollout sanity for a 4-brain flagship ckpt.

The full-flagship trainer (``scripts/train_flagship4b.py``) saves the metric
grounding under ``ck["grounding"]`` — a :class:`HierarchicalGrounding` whose
``step["op"]`` is the operative :class:`StepDisplacementReadout`. That is a
DIFFERENT container from the single ``ck["step_readout"]`` that
``eval_metric_rollout.py`` expects, and the 4b encoder is a different config, so
that script is not directly reusable here. This purpose-built eval decodes the
trajectory the way the model would drive it, using the SAME geometry helpers as
``driving_diagnostic.py`` so the numbers are comparable to the probe ladder.

Method (no fitting at eval):
  encode each val window -> roll the OPERATIVE predictor ``fwd_k`` steps under
  the TRUE action sequence (intent-free, exactly as the forward-consistency
  grounding was TRAINED) -> decode each transition's per-step metric Δpose with
  the trained operative ``step["op"]`` readout -> SE(2) accumulate to ego
  waypoints at {5,10,15,20}={0.5,1,1.5,2}s.

Reports (route-resampled, fp32, strict numerics):
  - grounded-rollout ADE (== ade_0_2s) + per-horizon de@Ts vs constant-velocity;
  - straight/gentle/sharp stratified grounded-rollout ADE vs CV (does the grounded
    readout beat CV on the straight stratum?);
  - if ``--diagnostic-json`` is given, pulls that run's best RAW PROBE straight
    ADE so the report answers "grounded readout vs raw probe vs CV" in one place.

Usage (pod1):
  python scripts/eval_grounded_rollout_4b.py \
     --ckpt /workspace/experiments/axis6-relaxed/ckpt.pt \
     --cache-dirs /workspace/data/comma2k19/_epcache \
     --config flagship4b_reduced \
     --diagnostic-json /workspace/experiments/axis6-relaxed/diag.json \
     --out /workspace/experiments/axis6-relaxed/grounded_rollout.json --episodes 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from driving_diagnostic import (WP_STEPS, agg_metric_dicts,  # noqa: E402
                                baseline_waypoints, curvature_bucket, de_of,
                                gt_ego_waypoints, net_heading_change_deg,
                                scalar_metrics, _strat)

from tanitad.config import (base250cam_config, flagship4b_config,  # noqa: E402
                            flagship4b_reduced_config)
from tanitad.data.mixing import load_episode  # noqa: E402
from tanitad.eval.ckpt_compat import (SPEED_SCALE,  # noqa: E402
                                      append_speed_channel,
                                      build_world_from_ckpt)
from tanitad.eval.gates import split_by_episode  # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.metric_dynamics import (HierarchicalGrounding,  # noqa: E402
                                            rollout_decode)

K_MAX = max(WP_STEPS)
_CFG = {"base250cam": base250cam_config, "flagship4b": flagship4b_config,
        "flagship4b_reduced": flagship4b_reduced_config}


@torch.no_grad()
def collect(world, step_readout, episodes, corpora, device, window, fwd_k,
            stride, batch, speed_input=False):
    """Encode each window, roll ``fwd_k`` steps under TRUE actions, decode each
    transition with the grounded operative step-readout + SE(2) accumulate.
    ``speed_input`` appends v0 = poses[last,3]/SPEED_SCALE as the 3rd action
    channel, EXACTLY as flagship_losses does in training: the t=0 speed only,
    constant over the window AND the future actions — never a future speed."""
    S_wp, GT, CV, EID, COR, SPD, HDG = [], [], [], [], [], [], []
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    need_ahead = max(K_MAX, fwd_k)
    for ep, corp in zip(episodes, corpora):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - window - need_ahead, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)
            aw = torch.stack([ep.actions[t:t + window] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + window:t + window + fwd_k]
                              for t in ch]).to(device)
            if speed_input:
                v0 = (ep.poses[last, 3:4] / SPEED_SCALE).to(device)   # [b, 1]
                aw = append_speed_channel(aw, v0)
                fa = append_speed_channel(fa, v0)
            states = world.encode_window(fw)                       # [b, W, S]
            wp_full, _ = rollout_decode(world.predictor, states, aw, fa,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="4b ckpt (has 'grounding')")
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", choices=list(_CFG), default="flagship4b_reduced")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--fwd-k", type=int, default=K_MAX)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--git-hash", default="unknown")
    ap.add_argument("--diagnostic-json", default=None,
                    help="optional: driving_diagnostic.py JSON to pull the best "
                         "RAW PROBE straight-stratum ADE for a 3-way compare")
    args = ap.parse_args()

    assert args.fwd_k >= K_MAX, f"--fwd-k {args.fwd_k} < {K_MAX} (need 2 s reach)"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def corpus_of(cd: str) -> str:
        low = cd.lower()
        return ("comma2k19" if "comma" in low
                else "physicalai" if "physical" in low else Path(cd).name)

    episodes, corpora = [], []
    for cd in args.cache_dirs:
        val_dirs = sorted(Path(cd).glob("*val*"))
        if not val_dirs:
            print(f"[grounded] WARNING no *val* under {cd}", flush=True)
            continue
        for p in sorted(val_dirs[-1].glob("ep_*.pt"))[:args.episodes]:
            episodes.append(load_episode(str(p), mmap=True))
            corpora.append(corpus_of(cd))
    assert episodes, "no val episodes loaded"

    # Self-describing ckpt: build at the TRAINED action_dim (speed-input ckpts
    # are 3-ch), keep the load strict — never relax strictness to paper over a
    # shape mismatch (strict=False would leave act_emb/inv_dyn random-init).
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world, speed_input, act_src = build_world_from_ckpt(_CFG[args.config](), ck,
                                                        ckpt_path=args.ckpt)
    world = world.to(device).eval()
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    assert "grounding" in ck, (
        "ckpt has no 'grounding' key — expected a train_flagship4b.py ckpt "
        "(the grounded operative step-readout lives in ck['grounding'].step['op'])")
    grounding = HierarchicalGrounding(world.state_dim).to(device).eval()
    grounding.load_state_dict(ck["grounding"])
    step_readout = grounding.step["op"]          # operative per-step Δpose readout
    window = world.predictor.cfg.window
    print(f"[grounded] {len(episodes)} val episodes, ckpt step {step}, window "
          f"{window}, fwd_k {args.fwd_k}, config {args.config}, "
          f"speed_input {speed_input}, dev {device}", flush=True)

    with strict_numerics():
        data = collect(world, step_readout, episodes, corpora, device, window,
                       args.fwd_k, args.stride, args.batch,
                       speed_input=speed_input)
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

    straight = strat["by_curvature"].get("straight")
    grounded_beats_cv_straight = (
        bool(straight["model_ade@1s"] < straight["cv_ade@1s"])
        if straight is not None else None)

    raw_probe_straight = beats_raw_probe = None
    if args.diagnostic_json and Path(args.diagnostic_json).exists():
        dj = json.loads(Path(args.diagnostic_json).read_text())
        ds = (dj.get("section3_error_localization", {})
              .get("by_curvature", {}).get("straight"))
        if ds is not None and straight is not None:
            raw_probe_straight = ds["model_ade@1s"]
            beats_raw_probe = bool(straight["model_ade@1s"] < raw_probe_straight)

    report = {
        "exp": "axis6-grounded-rollout-4b",
        "ckpt": args.ckpt, "step": step, "git_hash": args.git_hash,
        "config": args.config,
        "speed_input": speed_input, "action_dim_source": act_src,
        "method": ("operative predictor rollout under TRUE actions (intent-free) "
                   "-> per-step Δpose via grounding.step['op'] -> SE(2) accumulate; "
                   "NO fit at eval"),
        "n_windows_total": n,
        "eval_config": {"episodes_per_dir": args.episodes, "fwd_k": args.fwd_k,
                        "stride": args.stride, "window": window,
                        "n_splits": args.n_splits,
                        "waypoint_steps": list(WP_STEPS)},
        "grounded_rollout_heldout": rollout_metrics,
        "grounded_rollout_full_set": full,
        "constant_velocity": cv_metrics,
        "straight_stratum": {
            "grounded_rollout_ade@1s":
                (straight["model_ade@1s"] if straight else None),
            "cv_ade@1s": (straight["cv_ade@1s"] if straight else None),
            "raw_probe_ade@1s": raw_probe_straight,
            "n": (straight["n"] if straight else 0),
            "grounded_beats_cv": grounded_beats_cv_straight,
            "grounded_beats_raw_probe": beats_raw_probe,
        },
        "error_localization": strat,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print("\n=== GROUNDED ROLLOUT (op step-readout) ===", flush=True)
    print(f"  grounded ade_0_2s={rollout_metrics['ade_0_2s']['mean']:.3f}"
          f"±{rollout_metrics['ade_0_2s']['ci95']:.3f}  "
          f"de@1s={rollout_metrics['de@1s']['mean']:.3f}  "
          f"de@2s={rollout_metrics['de@2s']['mean']:.3f}", flush=True)
    print(f"  CV       ade_0_2s={cv_metrics['ade_0_2s']['mean']:.3f}  "
          f"de@1s={cv_metrics['de@1s']['mean']:.3f}", flush=True)
    for lab, v in strat["by_curvature"].items():
        print(f"  [{lab:8s}] grounded_ade@1s={v['model_ade@1s']:.3f} "
              f"cv_ade@1s={v['cv_ade@1s']:.3f} n={v['n']}", flush=True)
    print(f"  straight: grounded_beats_cv={grounded_beats_cv_straight} "
          f"grounded_beats_raw_probe={beats_raw_probe} "
          f"(raw_probe={raw_probe_straight})", flush=True)
    print(f"[grounded] report -> {args.out}", flush=True)
    print("GROUNDED_ROLLOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
