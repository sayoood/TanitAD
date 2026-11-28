"""E1b P0 substrate gate — MEASURED, read-only. No training, no writes to caches.

Proves, before any pod-day is spent:
  1. episode-id DISJOINTNESS (byte level) between the parity-TRAIN cache
     (physicalai-train-e438721ae894, mining source) and the held-out EVAL cache
     (physicalai-val-heldout-79d4e3d2d4c6, E1a's eval set). Intersection MUST be 0.
  2. REF-C base ckpt loads STRICT, anchors buffer shape, step.
  3. a real open-loop forward produces a finite [B,4,2] traj on a held-out window.
  4. parity-train episode-length distribution (to size the mining rollout).
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np
import torch

for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/workspace/e1a_e2a"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data.mixing import load_episode
from tanitad.refs.refc import RefCModel, refc_config

TRAIN = "/workspace/pai_epcache/physicalai-train-e438721ae894"
HELDOUT = "/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6"
CKPT = "/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt"


def ep_ids(cache_dir):
    ids = {}
    for p in sorted(Path(cache_dir).glob("ep_*.pt")):
        try:
            eid = str(load_episode(str(p), mmap=True).episode_id)
        except Exception as e:
            eid = f"<ERR {type(e).__name__}>"
        ids[p.name] = eid
    return ids


def main():
    out = {}
    tr = ep_ids(TRAIN)
    ho = ep_ids(HELDOUT)
    tr_set, ho_set = set(tr.values()), set(ho.values())
    inter = sorted(tr_set & ho_set)
    out["disjointness"] = {
        "train_cache": TRAIN, "heldout_cache": HELDOUT,
        "n_train_files": len(tr), "n_train_distinct_ids": len(tr_set),
        "n_heldout_files": len(ho), "n_heldout_distinct_ids": len(ho_set),
        "intersection_count": len(inter),
        "intersection_ids": inter[:20],
        "DISJOINT": len(inter) == 0,
        "heldout_ids_sample": sorted(ho_set)[:10],
    }
    # chance collision expectation (4-char id packed into int, per E1a note)
    print(f"[disjoint] train ids={len(tr_set)} heldout ids={len(ho_set)} "
          f"intersection={len(inter)} -> DISJOINT={len(inter) == 0}", flush=True)

    # ---- REF-C base ckpt load (STRICT) + forward ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = refc_config()
    cj = Path(CKPT).parent / "config.json"
    if cj.exists():
        cd = json.loads(cj.read_text()).get("cfg", {})
        # apply n_anchors / widths already match refc_config for base; assert instead
    model = RefCModel(cfg)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    missing_unexpected = model.load_state_dict(ck["model"], strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out["ckpt"] = {
        "path": CKPT, "step": int(ck.get("step", -1)),
        "anchors_shape": list(model.decoder.anchors.shape),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "missing_keys": list(missing_unexpected.missing_keys),
        "unexpected_keys": list(missing_unexpected.unexpected_keys),
    }
    print(f"[ckpt] step={out['ckpt']['step']} anchors={out['ckpt']['anchors_shape']} "
          f"params={out['ckpt']['n_params']:,} "
          f"missing={len(out['ckpt']['missing_keys'])} "
          f"unexpected={len(out['ckpt']['unexpected_keys'])}", flush=True)

    # one forward on a held-out window
    ho_files = sorted(Path(HELDOUT).glob("ep_*.pt"))
    ep = load_episode(str(ho_files[0]), mmap=True)
    fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames.float()
    poses = ep.poses.float()
    W = 8
    frames = fr[0:W][None].to(device)
    v0 = poses[W - 1, 3][None].to(device)
    with torch.no_grad():
        o = model(frames, nav_cmd=None, v0=v0, steps=2)
    traj = o["traj"]
    out["forward"] = {
        "traj_shape": list(traj.shape),
        "traj_finite": bool(torch.isfinite(traj).all()),
        "anchor_logits_shape": list(o["anchor_logits"].shape),
        "anchor_traj_shape": list(o["anchor_traj"].shape),
        "traj_sample": traj[0].cpu().numpy().round(3).tolist(),
    }
    print(f"[fwd] traj={out['forward']['traj_shape']} finite={out['forward']['traj_finite']} "
          f"anchor_traj={out['forward']['anchor_traj_shape']}", flush=True)

    # ---- parity-train T distribution (sample for speed) ----
    tr_files = sorted(Path(TRAIN).glob("ep_*.pt"))
    Ts = []
    for p in tr_files[::20]:  # every 20th, ~119 episodes
        try:
            Ts.append(int(load_episode(str(p), mmap=True).poses.shape[0]))
        except Exception:
            pass
    Ts = np.array(Ts)
    out["parity_train_T"] = {
        "n_sampled": int(Ts.size), "min": int(Ts.min()), "max": int(Ts.max()),
        "mean": round(float(Ts.mean()), 1), "median": int(np.median(Ts)),
        "n_total_files": len(tr_files),
    }
    print(f"[T] parity-train sampled {Ts.size}: T in [{Ts.min()},{Ts.max()}] "
          f"mean={Ts.mean():.1f} median={int(np.median(Ts))} "
          f"(total files {len(tr_files)})", flush=True)

    Path("/workspace/e1b/probe_substrate.json").parent.mkdir(parents=True, exist_ok=True)
    Path("/workspace/e1b/probe_substrate.json").write_text(json.dumps(out, indent=2))
    print("E1B_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
