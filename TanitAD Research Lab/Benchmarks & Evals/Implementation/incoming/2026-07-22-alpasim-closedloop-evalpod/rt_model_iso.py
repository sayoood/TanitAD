#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated per-step MODEL inference tick for flagship-v1 + REF-C-base on the A40.
Times the FULL driver plan() (frame canonicalization + model forward) on fixed inputs,
at a given render frame size. Renderer-independent -> clean model component of the loop.
Run: PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts <venv>/python rt_model_iso.py
"""
import sys, time, json, statistics as st
import numpy as np, torch
sys.path.insert(0, "/workspace")
from refc_driver import RefCPolicy
from flagship_v1_driver import FlagshipV1Policy
from tanitad.data.calib import FThetaIntrinsics

# scene ftheta calib (from the live sessions: cx~959 cy~755 native 1920x1080, fwd poly1~944)
INTR = FThetaIntrinsics(poly=(0.0, 944.49, -10.98, 32.70, -77.40, 32.52),
                        cx=959.9, cy=746.6, width=1920, height=1080, per_clip=True)

def _summ(ts):
    ts = sorted(ts)
    return {"n": len(ts), "median_ms": st.median(ts) * 1e3, "mean_ms": sum(ts) / len(ts) * 1e3,
            "p10_ms": ts[int(0.1 * len(ts))] * 1e3, "p90_ms": ts[int(0.9 * len(ts))] * 1e3}


def bench(policy, kind, H, W, n_frames=10, n=30, warm=5):
    """Decompose the driver's per-step plan() into CANON (CPU frame preprocessing) vs
    MODEL forward (GPU). n_frames = frames re-canonicalized per drive (deque grows to 24)."""
    from tanitad.data.calib import ftheta_crop_resize
    from tanitad.data.comma2k19 import stack_frames
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n_frames)]
    dev = policy.device
    win = policy.window
    tot, canon_t, model_t = [], [], []
    for i in range(n + warm):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        # --- CANON stage (exactly the drivers' preprocessing) ---
        vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        canon = ftheta_crop_resize(vid, INTR, 256, center="principal")
        stacked = stack_frames(canon, 3)
        fw = stacked[-win:][None].to(dev).float().div_(255.0)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        # --- MODEL stage (GPU forward) ---
        with torch.no_grad():
            if kind == "flagship":
                nav = torch.tensor([3], dtype=torch.long, device=dev)
                stt = policy.model.encode_window(fw)
                ctx = policy.model.strategic_policy(stt, nav)["ctx"]
                policy.model.tactical_policy(stt, ctx)["waypoints"]
            else:
                v0t = torch.tensor([12.0], device=dev); navt = torch.tensor([3], dtype=torch.long, device=dev)
                policy.model(fw, nav_cmd=navt, v0=v0t, steps=2)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        if i >= warm:
            tot.append(t2 - t0); canon_t.append(t1 - t0); model_t.append(t2 - t1)
    return {"plan_total": _summ(tot), "canon_cpu": _summ(canon_t), "model_gpu": _summ(model_t),
            "n_frames_canon": n_frames}

def main():
    out = {}
    fp = FlagshipV1Policy(ckpt="/root/models/flagship-30k/ckpt.pt")
    rp = RefCPolicy(ckpt="/root/models/refc-base-30k/ckpt.pt", preset="base")
    for tag, (H, W) in {"480x854": (480, 854), "1080x1920": (1080, 1920)}.items():
        out[tag] = {}
        for nf in (10, 24):   # 10 = min window; 24 = steady-state deque (driver re-canons all)
            out[tag]["nframes_%d" % nf] = {
                "flagship_v1": bench(fp, "flagship", H, W, n_frames=nf),
                "refc_base": bench(rp, "refc", H, W, n_frames=nf)}
        f10 = out[tag]["nframes_10"]["flagship_v1"]
        print(tag, "| flag plan %.0f (canon %.0f + model %.0f) ms | refc plan %.0f ms" % (
            f10["plan_total"]["median_ms"], f10["canon_cpu"]["median_ms"], f10["model_gpu"]["median_ms"],
            out[tag]["nframes_10"]["refc_base"]["plan_total"]["median_ms"]))
    out["device"] = torch.cuda.get_device_name(0)
    out["note"] = ("plan_total = canon_cpu (ftheta_crop_resize+stack of the frame history, CPU) + "
                   "model_gpu (256x256 forward). The driver re-canonicalizes the WHOLE deque every "
                   "step; nframes_24 is the steady-state cost, nframes_10 the minimum needed.")
    json.dump(out, open("/workspace/rt_model_iso.json", "w"), indent=2)
    print("wrote /workspace/rt_model_iso.json")

if __name__ == "__main__":
    main()
