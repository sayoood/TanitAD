"""H2 classifier — STEP 5 (pod2, GPU): MEASURE the compute the gate is supposed to save.

The efficiency claim is arithmetic over two costs, and both are measured here rather than assumed:

  * `enc`  — one frozen v1 encoder+readout pass over ONE camera frame (9-ch 256 px).
  * `head` — one forward of the sensor-need head over its 8-step window.

⚠️ The head's own cost is part of the ledger. A gate that costs what it saves is worthless, so the
saving is reported NET of the head, and the head runs on EVERY frame while the extra cameras run
only when it fires.

    always-on-K   :  K * enc                       per frame
    gated         :  1 * enc + head + B * enc      per frame     (B = extra cams/frame)
    saving(K, B)  :  1 - (1 + B + head/enc) / K

FLOPs are reported as analytic MAC counts alongside the wall-clock, because wall-clock on an A40 at
batch 32 is not the Orin/Thor number and must not be quoted as one.

usage (pod2):
  PYTHONPATH=/workspace/TanitAD/stack python3 h2c_cost.py \
      --ckpt /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt --out /workspace/h2clf/run
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch


def bench(fn, warm=10, iters=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def vit_macs(n_tokens, d, depth, in_ch, patch):
    patch_macs = n_tokens * d * in_ch * patch * patch
    per_block = 4 * n_tokens * d * d + 2 * n_tokens * n_tokens * d + 8 * n_tokens * d * d
    return patch_macs + depth * per_block


def head_macs(w, d_in, d, layers):
    proj = w * d_in * d
    per_block = 4 * w * d * d + 2 * w * w * d + 8 * w * d * d
    return proj + layers * per_block + d * d + d * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from h2c_train import SensorNeedHead
    from tanitad.config import flagship4b_config
    from tanitad.eval.ckpt_compat import build_world_from_ckpt

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    world, _si, _src = build_world_from_ckpt(flagship4b_config(), ck, args.ckpt)
    world = world.to(dev).eval()
    for p in world.parameters():
        p.requires_grad_(False)
    head = SensorNeedHead(d_img=2048, d_ego=2).to(dev).eval()

    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "encoder": {"depth": world.cfg.encoder.depth, "d_model": world.cfg.encoder.d_model,
                       "n_tokens": world.encoder.n_tokens, "patch": world.cfg.encoder.patch_size,
                       "in_channels": world.cfg.encoder.in_channels,
                       "params": sum(p.numel() for p in world.encoder.parameters())
                                 + sum(p.numel() for p in world.readout.parameters())},
           "head": {"params": sum(p.numel() for p in head.parameters())},
           "wallclock_s_per_item": {}, "analytic_macs": {}}
    res["analytic_macs"]["encoder_pass_per_camera_frame"] = int(
        vit_macs(world.encoder.n_tokens, world.cfg.encoder.d_model, world.cfg.encoder.depth,
                 world.cfg.encoder.in_channels, world.cfg.encoder.patch_size))
    res["analytic_macs"]["head_pass_per_frame"] = int(head_macs(8, 2050, 256, 2))
    res["analytic_macs"]["head_over_encoder"] = (
        res["analytic_macs"]["head_pass_per_frame"]
        / res["analytic_macs"]["encoder_pass_per_camera_frame"])

    with torch.no_grad():
        for b in args.batches:
            x = torch.randn(b, 9, 256, 256, device=dev)
            h = torch.randn(b, 8, 2050, device=dev)
            res["wallclock_s_per_item"][f"encoder_b{b}"] = bench(lambda: world.encode(x)) / b
            res["wallclock_s_per_item"][f"head_b{b}"] = bench(lambda: head(h)) / b

    e = res["wallclock_s_per_item"]["encoder_b32"]
    hd = res["wallclock_s_per_item"]["head_b32"]
    res["head_over_encoder_wallclock"] = hd / e
    res["saving"] = {}
    for K in (3, 7):
        res["saving"][f"always_on_{K}"] = {
            B: {"gated_cost_in_encoder_passes": 1 + B + hd / e,
                "always_on_cost": K,
                "saving_frac_wallclock": 1 - (1 + B + hd / e) / K,
                "saving_frac_macs": 1 - (1 + B + res["analytic_macs"]["head_over_encoder"]) / K}
            for B in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0)}
    json.dump(res, open(os.path.join(args.out, "cost_model.json"), "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
