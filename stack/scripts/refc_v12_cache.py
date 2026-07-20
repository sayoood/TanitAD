"""REF-C v1.2 — frozen-decoder feature cache.

The re-scorer trains on the OUTPUT of a frozen refc-xl-30k, so the 252 M
encoder+decoder forward is pure constant work: run it ONCE per window, store
what the head consumes, and train the head for many epochs at ~0 GPU cost.

WHAT IS CACHED, AND WHY IT MATCHES DEPLOYMENT EXACTLY
-----------------------------------------------------
The forward is run in the TanitEval decode condition, not the training one:
``model.eval()`` (no denoise noise, no ego-dropout), ``nav_cmd=None`` -> the
`follow` command, ``v0 = poses[last, 3]`` fed, ``steps = 2`` truncated denoise,
window 8, **stride 8** — i.e. exactly ``taniteval.refc_eval.collect``. The cache
is therefore a recording of the deployed decoder, not of a training-mode
variant, and a head trained on it transfers to the harness without a shift.

Per window (fp16 unless noted):
  q            [N, d]     FINAL-denoise-pass anchor query embedding (256x512)
  q0           [N, d]     t=0 classifier-pass query embedding — the one whose
                          frozen LINEAR readout IS the 0.907-Spearman selection
                          score. Both are stored because which one a learned
                          re-scorer should consume is an empirical question and
                          re-running the 252 M forward to find out is the one
                          thing this cache exists to avoid.
  base_logit   [N]        the frozen SELECTION score (t=0 conf + H19 reweight)
  refined_conf [N]        the DISCARDED refined-pass confidence, kept as a
                          free-of-charge baseline: "what if REF-C had simply
                          selected on the refined logits, untrained?"
  fan          [N, S, 2]  the refined trajectories — what selection ranks
  pooled       [F]        encoder pooled latent
  cond         [d]        decoder condition (measurement + hierarchy graft)
  tgt          [S, 2]     GT ego-frame waypoints (refb_labels.waypoint_targets,
                          the frame TanitEval's gt_ego_waypoints uses)
  v0           []         ego speed at the window's last pose
  eid          []         episode id — the jackknife/dev split must be
                          EPISODE-disjoint, never window-disjoint

~527 KB/window at N=256, d=512, F=992 (two embeddings).

SPLIT DISCIPLINE. Only ``physicalai-train-e438721ae894`` is read. The first
``--dev-episodes`` episodes of the requested slice become the DEV cache (used
for the temperature sweep's model selection) and are episode-disjoint from the
train cache. The 881-window TanitEval val set is never touched here — it is
seen once, at the end, through the harness.

Usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack python3 scripts/refc_v12_cache.py \
      --data-root /workspace/pai_epcache \
      --ckpt /workspace/experiments/refc-diffusion-xl-30k/ckpt.pt \
      --anchors /workspace/experiments/refc_anchors_full.pt \
      --out /root/refc_v12_cache --episodes 1400 --dev-episodes 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import refb_labels
from refb_train import load_cached_episodes
from tanitad.models.refc_rescorer import refc_forward_fan
from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                               refc_smoke_config, refc_xl_config)

PRESETS = {"small": refc_small_config, "base": refc_config,
           "xl": refc_xl_config, "smoke": refc_smoke_config}


def _apply_overrides(cfg, d: dict) -> None:
    """Push a run's config.json cfg dump onto a preset dataclass (loaders.py's
    convention) so every gated graft is constructed at the trained shape and
    the state_dict loads STRICT."""
    for k, v in d.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            _apply_overrides(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(cfg, k, tuple(v))
        else:
            setattr(cfg, k, v)


def load_frozen(ckpt: str, preset: str, anchors: str | None, device: str):
    cfg = PRESETS[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
    assert not cfg.refc1, "refc1 ckpt: horizons are path checkpoints, not time"
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])                          # STRICT
    if anchors:
        # The trained anchors travel in the ckpt buffer; installing the file
        # again is a fail-loud CROSS-CHECK, not a mutation.
        anc = torch.load(anchors, map_location="cpu", weights_only=True)
        anc = anc["anchors"] if isinstance(anc, dict) else anc
        delta = (model.decoder.anchors.float() - anc.float()).abs().max()
        assert float(delta) < 1e-5, (
            f"anchor mismatch: ckpt buffer vs {anchors} max|d| = {float(delta)}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, int(ck.get("step", -1))


@torch.no_grad()
def build(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, step = load_frozen(args.ckpt, args.config, args.anchors, device)
    horizons = tuple(cfg.trajectory.horizons)
    window, k_max = int(cfg.window), max(horizons)
    n_anchors = model.decoder.anchors.shape[0]
    print(f"[v12cache] frozen refc step={step} n_anchors={n_anchors} "
          f"horizons={horizons} window={window} device={device}", flush=True)

    eps, src = load_cached_episodes(args.data_root, args.pattern, args.episodes)
    print(f"[v12cache] {len(eps)} episodes from {src}", flush=True)
    out_root = Path(args.out)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "dev").mkdir(parents=True, exist_ok=True)

    n_dev = args.dev_episodes
    manifest = {"ckpt": args.ckpt, "ckpt_step": step, "src": str(src),
                "n_anchors": n_anchors, "horizons": list(horizons),
                "window": window, "stride": args.stride,
                "diffusion_steps": cfg.decoder.diffusion_steps,
                "d_q": cfg.decoder.d, "d_pooled": model.encoder.feat_dim,
                "episodes": len(eps), "dev_episodes": n_dev,
                "decode": "eval-mode, nav=follow, v0 fed, deterministic denoise",
                "shards": []}
    t0 = time.time()
    n_win_total = 0
    for e_i, ep in enumerate(eps):
        split = "dev" if e_i < n_dev else "train"
        dst = out_root / split / f"sh_{e_i:05d}.pt"
        if dst.exists() and not args.overwrite:
            continue
        fr = ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - window - k_max, args.stride))
        if not starts:
            continue
        Q, Q0, BL, RC, FAN, PL, CD, TG, V0 = ([] for _ in range(9))
        for i in range(0, len(starts), args.batch):
            ch = starts[i:i + args.batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + window])
                              for t in ch]).to(device).float().div_(255.0)
            pose_last = ep.poses[last].to(device).float()
            fut = torch.stack([ep.poses[t + window:t + window + k_max]
                               for t in ch]).to(device).float()
            v0 = pose_last[:, 3]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda" and args.amp)):
                o = refc_forward_fan(model, fw, nav_cmd=None, v0=v0)
            tgt = refb_labels.waypoint_targets(pose_last, fut, horizons)
            Q.append(o["q"].half().cpu())
            Q0.append(o["q0"].half().cpu())
            BL.append(o["anchor_logits"].float().half().cpu())
            RC.append(o["refined_conf"].float().half().cpu())
            FAN.append(o["anchor_traj"].float().half().cpu())
            PL.append(o["pooled"].float().half().cpu())
            CD.append(o["cond"].float().half().cpu())
            TG.append(tgt.float().half().cpu())
            V0.append(v0.float().cpu())
        rec = {"q": torch.cat(Q), "q0": torch.cat(Q0),
               "base_logit": torch.cat(BL),
               "refined_conf": torch.cat(RC), "fan": torch.cat(FAN),
               "pooled": torch.cat(PL), "cond": torch.cat(CD),
               "tgt": torch.cat(TG), "v0": torch.cat(V0),
               "eid": str(ep.episode_id)}
        tmp = dst.with_suffix(".tmp")
        torch.save(rec, tmp)
        tmp.replace(dst)
        n_win_total += rec["v0"].shape[0]
        manifest["shards"].append({"split": split, "file": dst.name,
                                   "eid": str(ep.episode_id),
                                   "n": int(rec["v0"].shape[0])})
        if e_i % 25 == 0:
            el = time.time() - t0
            print(json.dumps({"ep": e_i, "split": split,
                              "win": n_win_total, "elapsed_s": round(el, 1),
                              "win_per_s": round(n_win_total / max(el, 1e-9), 2)
                              }), flush=True)
    manifest["windows"] = n_win_total
    manifest["wall_s"] = round(time.time() - t0, 1)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"done": True, "windows": n_win_total,
                      "wall_s": manifest["wall_s"], "out": str(out_root)}),
          flush=True)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pattern", default="*train*")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="xl", choices=tuple(PRESETS))
    ap.add_argument("--anchors", default=None,
                    help="cross-check the ckpt's anchor buffer against this file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1400, help="0 = all")
    ap.add_argument("--dev-episodes", type=int, default=200,
                    help="first N episodes -> the DEV cache (episode-disjoint)")
    ap.add_argument("--stride", type=int, default=8,
                    help="window stride (8 = the TanitEval protocol)")
    ap.add_argument("--batch", type=int, default=8)
    # fp32 by DEFAULT: TanitEval decodes in fp32, and the cache must record the
    # same numbers the harness will later reproduce, not a bf16 approximation.
    ap.add_argument("--amp", action="store_true", default=False)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
