"""Thor sustained-load + optimisation-lever profile (2026-08-02, follow-up).

⚠️ **WHY THIS EXISTS: it is the risk check on the profile already published.** Those p50s came
from 50 iterations (~10 s of GPU work). An edge SoC that holds 27.8 ms for ten seconds and then
thermally throttles has NOT met a deployment budget, and publishing the burst number as if it had
would be exactly the class of over-claim the programme keeps retracting. This runs the encoder
for ~3 minutes and reports the LAST-decile p50 against the FIRST-decile p50.

Also measures the two cheapest ranked levers on the encoder (the 44x-dominant stage):
  * CUDA-graph capture — the A40 precedent's proven FIRST lever (capture before the rest;
    the programme's own finding was that the levers are SEQUENCED, not additive).
  * torch.compile — timeboxed; on a fresh aarch64 stack it may fail, and a failure is a RESULT
    (it tells the deployment stream which path to invest in), not an error to hide.
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.expanduser("~/TanitAD/stack"))
sys.path.insert(0, os.path.expanduser("~/TanitAD/stack/scripts"))
sys.path.insert(0, os.path.expanduser("~/TanitAD/taniteval"))

import dataclasses
from types import SimpleNamespace

import torch

from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel
from train_flagship_v4 import resolve_v2_frames

DEV, H, W = "cuda", 176, 624
out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}

cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection="cylindrical", v2_subframe="176x624", f_ref=None)
resolve_v2_frames(ns, cfg, label="thor_sustained")
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, "tactical_pred", None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)

model = WorldModel(cfg).to(DEV).eval()
Wn, C = cfg.predictor.window, cfg.encoder.in_channels
frames = torch.randn(1, Wn, C, H, W, device=DEV)


def enc_bf16():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.encode_window(frames)


# ---------------------------------------------------------------- 1. sustained
DURATION_S = 180
with torch.no_grad():
    for _ in range(10):
        enc_bf16()
    torch.cuda.synchronize()
    ts, t_start = [], time.perf_counter()
    while time.perf_counter() - t_start < DURATION_S:
        t0 = time.perf_counter()
        enc_bf16()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)

n = len(ts)
d = max(1, n // 10)
first, last = ts[:d], ts[-d:]
out["sustained"] = {
    "duration_s": DURATION_S, "iters": n,
    "first_decile_p50_ms": round(statistics.median(first), 2),
    "last_decile_p50_ms": round(statistics.median(last), 2),
    "overall_p50_ms": round(statistics.median(ts), 2),
    "overall_p99_ms": round(sorted(ts)[int(n * 0.99)], 2),
    "max_ms": round(max(ts), 2),
}
drift = out["sustained"]["last_decile_p50_ms"] / out["sustained"]["first_decile_p50_ms"]
out["sustained"]["throttle_ratio_last_over_first"] = round(drift, 3)
out["sustained"]["verdict"] = (
    "NO THROTTLING — the burst number holds under sustained load" if drift < 1.05
    else f"⛔ THROTTLES {drift:.2f}x — the published burst p50 is NOT a deployment number")

# ---------------------------------------------------------------- 2. CUDA graph
try:
    with torch.no_grad():
        static_in = frames.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.encode_window(static_in)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                static_out = model.encode_window(static_in)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        gts = []
        for _ in range(50):
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            gts.append((time.perf_counter() - t0) * 1e3)
        gts.sort()
        out["cuda_graph"] = {"p50_ms": round(gts[len(gts) // 2], 2),
                             "p99_ms": round(gts[int(len(gts) * 0.99)], 2),
                             "out_shape": tuple(static_out.shape)}
        out["cuda_graph"]["speedup_vs_eager_bf16_x"] = round(
            out["sustained"]["overall_p50_ms"] / out["cuda_graph"]["p50_ms"], 2)
except Exception as e:
    out["cuda_graph"] = {"FAILED": f"{type(e).__name__}: {str(e)[:220]}"}

# ---------------------------------------------------------------- 3. torch.compile
try:
    t0 = time.perf_counter()
    cmodel = torch.compile(model.encode_window, mode="max-autotune-no-cudagraphs")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cmodel(frames)
        torch.cuda.synchronize()
        compile_s = round(time.perf_counter() - t0, 1)
        cts = []
        for _ in range(30):
            t1 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cmodel(frames)
            torch.cuda.synchronize()
            cts.append((time.perf_counter() - t1) * 1e3)
    cts.sort()
    out["torch_compile"] = {"compile_s": compile_s,
                            "p50_ms": round(cts[len(cts) // 2], 2),
                            "speedup_vs_eager_bf16_x": round(
                                out["sustained"]["overall_p50_ms"] / cts[len(cts) // 2], 2)}
except Exception as e:
    out["torch_compile"] = {"FAILED": f"{type(e).__name__}: {str(e)[:220]}",
                            "note": "a failure here is a RESULT — it routes the deployment "
                                    "stream to TensorRT/graph capture instead"}

print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser("~/thor_sustained.json"), "w") as f:
    json.dump(out, f, indent=1)
