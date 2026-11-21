"""Deep deployment profile of the TanitAD stack on Jetson Thor (PI request 2026-08-02).

⛔ METHOD, stated because a latency number without it is worthless:
  * **WARMUP then SYNC.** CUDA is async — timing without `torch.cuda.synchronize()` measures
    kernel LAUNCH, not execution. Every stage below warms up (cuDNN/TF32 autotune picks its
    algorithm on first call) and synchronises around each timed region.
  * **p50 AND p99**, not a mean. The budget is a real-time deadline: the tail is the spec.
    10 Hz planning => **100 ms** budget (the programme's stated bar).
  * **The DEPLOYED geometry**, 176x624 sub-frame at 120 deg HFOV cylindrical — v5f's actual
    input, not a convenient square.
  * **Unified memory.** Thor's 122 GB is SHARED between CPU and GPU; `torch.cuda.max_memory_
    allocated` is the allocator's view and is NOT the system footprint. Both are reported.
  * ⚠️ This profiles the ARCHITECTURE at fp32/bf16 eager. It is a BASELINE, not a deployment
    claim: TensorRT/INT8/CUDA-graph capture are the optimisation stream this baseline exists
    to be measured against (the v1-class A40 result was 138 ms p50 eager -> 18.75 ms after a
    sequenced 4-lever pass, so eager numbers here should be read as the STARTING point).
"""
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.expanduser("~/TanitAD/stack"))
sys.path.insert(0, os.path.expanduser("~/TanitAD/taniteval"))

import torch

from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel

DEV = "cuda"
BUDGET_MS = 100.0
H, W = 176, 624          # v5f deployed sub-frame
WARMUP, ITERS = 10, 50


def sync():
    torch.cuda.synchronize()


def timeit(fn, warmup=WARMUP, iters=ITERS):
    """-> dict of ms percentiles. Warms up, then times each call with a sync barrier."""
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 2),
            "p90_ms": round(ts[int(len(ts) * 0.9)], 2),
            "p99_ms": round(ts[min(len(ts) - 1, int(len(ts) * 0.99))], 2),
            "min_ms": round(ts[0], 2), "max_ms": round(ts[-1], 2), "n": iters}


def mem_mb():
    return {"alloc_MB": round(torch.cuda.memory_allocated() / 1e6, 1),
            "peak_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}


def rss_gb():
    """System-wide used RAM — on unified memory this is the number that actually bounds us."""
    try:
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
        return round((mi["MemTotal"] - mi["MemAvailable"]) / 1e6, 2)
    except Exception:
        return None


out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "arch": platform.machine(), "geometry": f"{H}x{W}",
       "budget_ms": BUDGET_MS, "method": "warmup+cuda_synchronize, p50/p99 over 50 iters"}

cfg = flagship4b_config()
# ⛔ Apply v5f's DEPLOYED geometry through the trainer's own seam. The encoder raises if the
# input does not match the declared frame — its positional embedding is sized for it. Building
# the frame by hand would risk a lookalike; resolve_v2_frames is what the real run calls.
from types import SimpleNamespace
sys.path.insert(0, os.path.expanduser('~/TanitAD/stack/scripts'))
from train_flagship_v4 import resolve_v2_frames
_ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0, projection='cylindrical',
                      v2_subframe='176x624', f_ref=None)
_cache_frame, _train_frame = resolve_v2_frames(_ns, cfg, label='thor_profile')
out['frame'] = f'{_train_frame.height}x{_train_frame.width} hfov {_train_frame.hfov_deg:.1f}'
import dataclasses
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, 'tactical_pred', None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
model = WorldModel(cfg).to(DEV).eval()
Wn, C = cfg.predictor.window, cfg.encoder.in_channels
out["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
out["rss_after_load_GB"] = rss_gb()

torch.cuda.reset_peak_memory_stats()
frames = torch.randn(1, Wn, C, H, W, device=DEV)
acts = torch.randn(1, Wn, cfg.predictor.action_dim, device=DEV)

with torch.no_grad():
    # ---- stage 1: encoder (the window encode — dominates on any wide input) ----
    out["encode_window_fp32"] = timeit(lambda: model.encode_window(frames))
    states = model.encode_window(frames)
    out["state_shape"] = tuple(states.shape)

    # ---- stage 2: one predictor step (the imagination unit cost) ----
    out["predictor_1step_fp32"] = timeit(lambda: model.predictor(states, acts))

    # ---- stage 3: a 20-step recursive roll (what imagination conditioning actually costs) ----
    def roll20():
        ws, wa = states, acts
        for _ in range(20):
            z = model.predictor(ws, wa)[1]
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
        return ws
    out["predictor_roll20_fp32"] = timeit(roll20, warmup=3, iters=15)
    out["mem_fp32"] = mem_mb()

    # ---- stage 4: bf16 autocast — the cheapest real deployment lever ----
    torch.cuda.reset_peak_memory_stats()
    def enc_bf16():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.encode_window(frames)
    out["encode_window_bf16"] = timeit(enc_bf16)
    out["mem_bf16"] = mem_mb()

    # ---- stage 5: batch scaling — does the 122 GB buy throughput? ----
    batch = {}
    for b in (1, 2, 4, 8):
        try:
            torch.cuda.reset_peak_memory_stats()
            fb = torch.randn(b, Wn, C, H, W, device=DEV)
            r = timeit(lambda: model.encode_window(fb), warmup=3, iters=10)
            batch[f"b{b}"] = {"p50_ms": r["p50_ms"],
                              "ms_per_sample": round(r["p50_ms"] / b, 2),
                              "peak_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
            del fb
        except RuntimeError as e:
            batch[f"b{b}"] = {"error": type(e).__name__}
            break
    out["batch_scaling_encode"] = batch

e50 = out["encode_window_fp32"]["p50_ms"]
p50 = out["predictor_1step_fp32"]["p50_ms"]
out["verdict"] = {
    "encode_p50_vs_budget": f"{e50:.1f} ms = {100*e50/BUDGET_MS:.0f}% of the {BUDGET_MS:.0f} ms budget",
    "bf16_speedup_x": round(e50 / max(out["encode_window_bf16"]["p50_ms"], 1e-9), 2),
    "imagination_roll20_ms": out["predictor_roll20_fp32"]["p50_ms"],
    "note": ("EAGER fp32/bf16 baseline — TensorRT/INT8/CUDA-graph capture not applied. "
             "The A40 precedent moved a v1-class plan step 138 ms -> 18.75 ms with a "
             "sequenced 4-lever pass, so read these as the STARTING point."),
}
out["rss_end_GB"] = rss_gb()
print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser("~/thor_profile.json"), "w") as f:
    json.dump(out, f, indent=1)
