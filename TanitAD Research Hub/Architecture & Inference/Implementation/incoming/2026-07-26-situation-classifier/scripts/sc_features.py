"""Situation classifier — STEP 2 (pod3, GPU): extract FROZEN v1 encoder features.

⭐ **There is no alignment step here, and that is a substantive difference from H2.** The labels are
built on the episode's own `poses`, which `build_episode` produced at exactly the frames it stored —
so label index `t` and feature index `t` are the same index by construction. H2 had to recover an
integer lag from a speed correlation and lost 10.65 % of its clips to a guard that tripped on its
own degeneracy; that failure mode does not exist in this stream.

The feature is `WorldModel.encode(frames)` — the 2048-d `SpatialGridReadout` state the world model
itself consumes. Encoder + readout are the deployed v1's, **FROZEN** and **STRICT-loaded** from the
trunk payload written by `sc_extract_trunk.py` (`load_state_dict(..., strict=True)` on each of the
two modules, so a missing or renamed key is a hard failure, never a silent partial load).

C-FID (the fidelity half of the bi-directional harness check) is enforced here: the per-clip row
count of the label bundle must equal the episode's frame count, or the clip is refused.

usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack python3 sc_features.py \
      --bundle /workspace/sitclf/bundle --out /workspace/sitclf/feats \
      --trunk /workspace/sitclf/v1_trunk.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

CACHE = "/workspace/pai_epcache/physicalai-train-e438721ae894"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from tanitad.config import flagship4b_config
    from tanitad.models.fourbrain import WorldModel

    meta = json.load(open(os.path.join(args.bundle, "sc_meta.json")))
    if args.limit:
        meta = meta[:args.limit]

    t0 = time.time()
    pay = torch.load(args.trunk, map_location="cpu", weights_only=False)
    trunk = pay["trunk"]
    cfg = flagship4b_config()
    world = WorldModel(cfg)
    enc_sd = {k[len("encoder."):]: v for k, v in trunk.items() if k.startswith("encoder.")}
    ro_sd = {k[len("readout."):]: v for k, v in trunk.items() if k.startswith("readout.")}
    world.encoder.load_state_dict(enc_sd, strict=True)     # STRICT -> the weights ARE v1's
    world.readout.load_state_dict(ro_sd, strict=True)
    world = world.to(args.device).eval()
    for p in world.parameters():
        p.requires_grad_(False)
    n_enc = sum(p.numel() for p in world.encoder.parameters())
    n_ro = sum(p.numel() for p in world.readout.parameters())
    assert n_enc == pay["encoder_params"] and n_ro == pay["readout_params"], "param-count mismatch"
    print(f"[feat] STRICT-loaded v1 trunk (step {pay['step']}) in {time.time()-t0:.1f}s; "
          f"encoder {n_enc:,} + readout {n_ro:,}; state_dim={world.state_dim}", flush=True)

    kept, refused, tstart = 0, [], time.time()
    for m in meta:
        k = m["k"]
        dst = os.path.join(args.out, f"clip_{k:05d}.npy")
        if os.path.exists(dst):
            kept += 1
            continue
        ep = torch.load(os.path.join(args.cache, m["file"]), map_location="cpu",
                        weights_only=True, mmap=True)
        frames = ep["frames_u8"]
        # ---- C-FID: the label row count MUST equal the episode frame count ----
        if int(frames.shape[0]) != int(m["T"]):
            refused.append({"k": k, "T_label": int(m["T"]), "T_episode": int(frames.shape[0])})
            continue
        feats = np.empty((frames.shape[0], world.state_dim), dtype=np.float16)
        with torch.no_grad():
            for b in range(0, frames.shape[0], args.batch):
                x = frames[b:b + args.batch].to(args.device, non_blocking=True)
                x = x.float().div_(255.0)
                feats[b:b + args.batch] = world.encode(x).to(torch.float16).cpu().numpy()
        np.save(dst, feats)
        kept += 1
        if kept % 100 == 0:
            el = time.time() - tstart
            print(f"[feat] {kept}/{len(meta)} ({el:.0f}s, {el/max(kept,1):.2f}s per clip)",
                  flush=True)

    summary = {"n_clips": len(meta), "kept": kept, "refused_C_FID": refused,
               "cache": args.cache, "state_dim": int(world.state_dim),
               "trunk": {"step": pay["step"], "encoder_params": n_enc, "readout_params": n_ro,
                         "frozen": True, "strict_load": True,
                         "preprocess": "uint8 -> float/255 (to_float_frames contract)"},
               "wallclock_s": round(time.time() - tstart, 1),
               "alignment": "NONE REQUIRED — labels are built on the episode's own poses index"}
    json.dump(summary, open(os.path.join(args.out, "feat_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2)[:1200])
    if refused:
        print(f"[feat] ⛔ C-FID refused {len(refused)} clips on a frame-count mismatch", flush=True)


if __name__ == "__main__":
    main()
