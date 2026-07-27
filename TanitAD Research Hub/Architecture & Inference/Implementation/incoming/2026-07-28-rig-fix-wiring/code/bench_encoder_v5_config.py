"""The compute delta at the V5 CONFIG — re-measured, not inherited.

⚠️ WHY RE-MEASURE. The rig-clean stream's `token_cost.json` is real, but it is
`base250cam_config()` at image-batch 8 and no autocast. v5 trains
`flagship4b_config()` under AMP, and its encoder sees `batch x window` images per
micro-batch (16 x 8 = 128), not `batch`. A ratio quoted from the wrong config and
the wrong batch is exactly the kind of INHERITED number this program retracts.

Reported: seconds per forward+backward, PER IMAGE, at several image batches, with
the ratio to the 256x640 parent. The ratio is the decision quantity and is the
part that is batch-robust; the absolute seconds are for the host they were taken
on and nothing else.

⛔ Encoder only. This is NOT a whole-step number: the predictor, the four-brain
heads and the planner do not shrink with the token count, so the run-level saving
is strictly SMALLER than the ratio below. Said here so it cannot be rounded up in
prose.

Usage: PYTHONPATH=<stack> python3 bench_encoder_v5_config.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tanitad.config import flagship4b_config
    from tanitad.data.calib import (PHYSICALAI_RIG_CLEAN_128x576,
                                    PHYSICALAI_RIG_CLEAN_176x624,
                                    PHYSICALAI_WIDE120_256x640)
    from tanitad.geometry import apply_frame, geometry_report
    from tanitad.models.encoder import ViTEncoder

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg0 = flagship4b_config()
    out = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
           "config": "flagship4b_config", "amp": bool(a.amp and dev == "cuda"),
           "reps": a.reps, "window": int(cfg0.predictor.window),
           "in_channels": int(cfg0.encoder.in_channels),
           "images_per_microbatch_at_batch16": 16 * int(cfg0.predictor.window),
           "scope": ("ENCODER forward+backward only — the predictor, the "
                     "four-brain heads and the planner do not shrink with the "
                     "token count, so the run-level saving is SMALLER."),
           "frames": []}

    for frame in (PHYSICALAI_WIDE120_256x640, PHYSICALAI_RIG_CLEAN_176x624,
                  PHYSICALAI_RIG_CLEAN_128x576):
        cfg = flagship4b_config()
        apply_frame(cfg, frame)
        rep = geometry_report(cfg)
        enc = ViTEncoder(cfg.encoder).to(dev)
        row = {"tag": frame.tag(), "hw": [frame.height, frame.width],
               "n_tokens": rep["n_tokens"], "token_grid": rep["token_grid"],
               "state_dim": rep["state_dim"],
               "params_M": round(sum(p.numel() for p in enc.parameters()) / 1e6, 3),
               "by_image_batch": {}}
        for B in a.batches:
            try:
                x = torch.randn(B, cfg.encoder.in_channels, frame.height,
                                frame.width, device=dev)
                if dev == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if out["amp"] else torch.autocast("cpu", enabled=False))
                for _ in range(3):
                    with ctx:
                        y = enc(x)
                    y.float().sum().backward()
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(a.reps):
                    with ctx:
                        y = enc(x)
                    y.float().sum().backward()
                if dev == "cuda":
                    torch.cuda.synchronize()
                dt = (time.time() - t0) / a.reps
                row["by_image_batch"][str(B)] = {
                    "s_per_fwdbwd": round(dt, 5),
                    "s_per_image": round(dt / B, 6),
                    "peak_MiB": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                 if dev == "cuda" else None)}
                del x, y
            except torch.cuda.OutOfMemoryError as e:      # noqa: PERF203
                row["by_image_batch"][str(B)] = {"OOM": str(e)[:120]}
            if dev == "cuda":
                torch.cuda.empty_cache()
        out["frames"].append(row)
        del enc
        if dev == "cuda":
            torch.cuda.empty_cache()

    base = out["frames"][0]
    for f in out["frames"]:
        f["tokens_vs_parent"] = round(f["n_tokens"] / base["n_tokens"], 4)
        f["ratio_vs_parent"] = {
            B: round(v["s_per_image"] / base["by_image_batch"][B]["s_per_image"], 4)
            for B, v in f["by_image_batch"].items()
            if "s_per_image" in v and "s_per_image" in base["by_image_batch"][B]}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
