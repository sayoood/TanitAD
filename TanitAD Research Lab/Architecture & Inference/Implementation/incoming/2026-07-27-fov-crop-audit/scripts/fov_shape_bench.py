"""FOV crop audit — the PRICE of each input shape, MEASURED on the dev box (RTX 4060, 8.6 GB).

The PI's ">= 100 deg" and "review the 256 px resolution" asks both cost compute, and the cost table
that has been circulating is ESTIMATED. This measures it.

⚠️ **WDDM SPILL FILTER, applied to every row.** On Windows/WDDM a CUDA allocation that exceeds the
device pool does NOT raise OOM — it silently spills to host RAM and keeps running, ~20x slower, while
still reporting a plausible `max_memory_allocated`. So a memory number alone is not a capacity claim.
Each shape is timed at TWO batch sizes; if ms/frame rises by more than `SPILL_FACTOR` when the batch
grows (work per frame is constant, so it must not), the row is marked `SPILLED` and its memory
figure is refused as a capacity claim.

Two regimes are timed, because they have different prices:
  * `infer`  — forward only, no_grad (the deployment / feature-extraction price)
  * `train`  — forward + backward on a scalar loss (the training price, which is what a re-cache
               plus a re-train would actually cost)

⚠️ Non-square and non-256 shapes need TWO small relaxations of a square assumption
(`encoder.ViTEncoder.n_tokens`/`pos`, `readout.SpatialGridReadout`'s `hw*hw == n_tokens`). They are
applied here as a LOCAL SHIM (`shape_shim.py`) and the training path in `stack/` is NOT modified.

usage:  python fov_shape_bench.py <out_json>
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get(
    "TANITAD_STACK", r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shape_shim import build_trunk                              # noqa: E402

SPILL_FACTOR = 1.35      # ms/frame growth from batch b1->b2 above which the row is a SPILL
PATCH = 16
F_REF_TODAY, SIZE_TODAY = 266.0, 256
HALF_TODAY = math.degrees(math.atan((SIZE_TODAY / 2) / F_REF_TODAY))

# (H, W, hfov_deg) — hfov is what the WIDTH is asked to span; the vfov follows from H at the same
# angular scale (that is the whole point of the letterbox row).
SHAPES = [
    (256, 256, 2 * HALF_TODAY),      # today
    (256, 256, 100.0),               # the PI's ask, FREE (same tokens) but 2.5x coarser
    (320, 320, 100.0),
    (384, 384, 100.0),
    (512, 512, 100.0),
    (640, 640, 100.0),               # 100 deg at TODAY's angular resolution, square
    (256, 640, 100.0),               # the letterbox: 100 deg wide, today's angular resolution
    (256, 512, 80.0),
    (256, 384, 60.0),
]


def bench(world, h, w, regime, batch, iters=6, warmup=2):
    dev = "cuda"
    # the shim freezes every parameter (it is a frozen probe); the TRAIN regime has to un-freeze
    # them or the backward has nothing to accumulate into and the price would be understated.
    want_grad = regime == "train"
    for p in world.parameters():
        p.requires_grad_(want_grad)
    x = torch.randint(0, 255, (batch, 9, h, w), dtype=torch.uint8, device=dev)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    def step():
        if regime == "infer":
            with torch.no_grad():
                world.encode(x.float().div(255.0))
        else:
            world.zero_grad(set_to_none=True)
            world.encode(x.float().div(255.0)).square().mean().backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return {"ms_per_frame": round(dt * 1000 / batch, 3),
            "frames_per_s": round(batch / dt, 1),
            "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
            "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
            "batch": batch}


def main():
    out_json = sys.argv[1]
    props = torch.cuda.get_device_properties(0)
    rows = []
    for h, w, hfov in SHAPES:
        gh, gw = h // PATCH, w // PATCH
        tokens = gh * gw
        half = math.radians(hfov / 2.0)
        f_eff = (w / 2.0) / math.tan(half)
        row = {"h": h, "w": w, "tokens": tokens, "hfov_deg": round(hfov, 2),
               "vfov_deg": round(math.degrees(2 * math.atan((h / 2.0) / f_eff)), 2),
               "f_eff_px": round(f_eff, 2),
               "px_per_deg": round((w / 2.0) / (hfov / 2.0), 3)}
        try:
            world = build_trunk(image_h=h, image_w=w, device="cuda")
        except Exception as exc:                                    # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        for regime in ("infer", "train"):
            try:
                b1, b2 = (4, 8) if tokens > 900 else (8, 16)
                r1 = bench(world, h, w, regime, b1)
                r2 = bench(world, h, w, regime, b2)
                spilled = r2["ms_per_frame"] > SPILL_FACTOR * r1["ms_per_frame"]
                row[regime] = {**r2, "ms_per_frame_small_batch": r1["ms_per_frame"],
                               "SPILLED": bool(spilled),
                               "capacity_claim_admissible": not spilled}
            except torch.cuda.OutOfMemoryError as exc:              # pragma: no cover
                row[regime] = {"OOM": str(exc)[:160]}
            torch.cuda.empty_cache()
        del world
        torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)

    base = next(r for r in rows if r["h"] == 256 and r["w"] == 256
                and abs(r["hfov_deg"] - 2 * HALF_TODAY) < 0.1)
    for r in rows:
        for regime in ("infer", "train"):
            if regime in r and "ms_per_frame" in r[regime] and regime in base:
                r[regime]["x_vs_today"] = round(
                    r[regime]["ms_per_frame"] / base[regime]["ms_per_frame"], 2)
    res = {"device": props.name, "total_mem_GiB": round(props.total_memory / 2 ** 30, 2),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "spill_filter": {"rule": "ms/frame must not grow >%.2fx when the batch doubles; "
                                    "work per frame is constant, so growth == host-RAM spill "
                                    "(WDDM does not OOM)" % SPILL_FACTOR,
                            "SPILL_FACTOR": SPILL_FACTOR},
           "today": {"h": 256, "w": 256, "hfov_deg": round(2 * HALF_TODAY, 3),
                     "tokens": 256, "F_REF": F_REF_TODAY},
           "shapes": rows}
    json.dump(res, open(out_json, "w"), indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
