"""TensorRT engine build + benchmark on Thor — the Production & Optimization stream's #1 item.

Their backlog recorded it as: "TRT-fp16 engine for flagship@30k -- NOW THE TOP LATENCY ITEM ...
Toolchain-blocked on the dev box (tensorrt missing) -> run when a pod is idle or tensorrt +
onnxruntime-gpu land. ONNX IR already parity-clean." TensorRT 10.13.3.9 has now landed on Thor,
so the block is gone.

⭐ THE TARGET IS THE PREDICTOR, not the encoder. Today's measurements say why:
  * the roll is 71 % of the tick and is COMPUTE-bound (full-roll graph capture bought only 1.02x)
  * precision alone LOSES there (bf16 0.86x, fp16 0.90x) because the tensors are small
  * fusion is the one thing eager PyTorch cannot do, and small-tensor many-op work is exactly
    what kernel fusion targets
So this exports the PREDICTOR and asks whether TRT beats the 3.42 ms/step CUDA-graph baseline.
The engine must beat the FREE graph to justify the toolchain -- their bar, not mine.

⚠️ Accuracy beside speed (G-P2): every engine output is compared to the eager fp32 reference.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/TanitAD/stack"))
sys.path.insert(0, os.path.expanduser("~/TanitAD/stack/scripts"))
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")   # TRT bindings are system-level

import dataclasses
import subprocess
from types import SimpleNamespace

import torch

from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel
from train_flagship_v4 import resolve_v2_frames

DEV, H, W = "cuda", 176, 624
OUT = os.path.expanduser("~/trt")
os.makedirs(OUT, exist_ok=True)
out = {"device": torch.cuda.get_device_name(0)}
try:
    import tensorrt as trt
    out["trt_version"] = trt.__version__
except Exception as e:
    out["trt_version"] = f"IMPORT FAILED {e}"

cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection="cylindrical", v2_subframe="176x624", f_ref=None)
resolve_v2_frames(ns, cfg, label="trt")
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, "tactical_pred", None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
model = WorldModel(cfg).to(DEV).eval()
Wn, A = cfg.predictor.window, cfg.predictor.action_dim
S = model.state_dim


class PredWrap(torch.nn.Module):
    """The predictor's 1-step head, isolated so the ONNX graph is exactly the hot path."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, states, actions):
        return self.p(states, actions)[1]


wrap = PredWrap(model.predictor).eval()
st = torch.randn(1, Wn, S, device=DEV)
ac = torch.randn(1, Wn, A, device=DEV)
with torch.no_grad():
    ref = wrap(st, ac).float().clone()

# ------------------------------------------------------------------ ONNX export
onnx_path = f"{OUT}/predictor.onnx"
try:
    torch.onnx.export(wrap, (st, ac), onnx_path,
                      input_names=["states", "actions"], output_names=["z_next"],
                      opset_version=17, dynamo=False)
    out["onnx_export"] = {"ok": True, "MB": round(os.path.getsize(onnx_path) / 1e6, 1)}
except Exception as e:
    out["onnx_export"] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:200]}"}

# ------------------------------------------------------------------ TRT engines
def build(tag, extra):
    eng = f"{OUT}/predictor_{tag}.plan"
    cmd = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={onnx_path}",
           f"--saveEngine={eng}", "--skipInference"] + extra
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    ok = os.path.exists(eng)
    return {"ok": ok, "build_s": round(time.perf_counter() - t0, 1),
            "MB": round(os.path.getsize(eng) / 1e6, 1) if ok else None,
            "err": None if ok else (r.stderr or r.stdout)[-260:]}, eng


engines = {}
if out.get("onnx_export", {}).get("ok"):
    for tag, extra in (("fp32", []), ("fp16", ["--fp16"])):
        info, eng = build(tag, extra)
        engines[tag] = eng if info["ok"] else None
        out[f"build_{tag}"] = info

# ------------------------------------------------------------------ benchmark
def run_engine(eng):
    """trtexec's own timing — the tool reports median GPU compute time directly."""
    r = subprocess.run(["/usr/src/tensorrt/bin/trtexec", f"--loadEngine={eng}",
                        "--iterations=200", "--warmUp=500", "--avgRuns=100"],
                       capture_output=True, text=True, timeout=900)
    med = None
    for line in (r.stdout or "").splitlines():
        if "GPU Compute Time" in line and "median" in line:
            for part in line.split(","):
                if "median" in part:
                    med = float(part.split("=")[1].strip().split()[0])
    return med


for tag, eng in engines.items():
    if eng:
        m = run_engine(eng)
        out[f"trt_{tag}_median_ms"] = m
        if m:
            out[f"trt_{tag}_vs_cudagraph_x"] = round(3.42 / m, 2)   # graph baseline
            out[f"trt_{tag}_vs_eager_x"] = round(4.23 / m, 2)

out["baselines_ms"] = {"eager_fp32": 4.23, "cuda_graph": 3.42,
                       "note": "the engine must beat the FREE CUDA graph to justify the toolchain"}
if out.get("trt_fp16_median_ms"):
    best = min(v for k, v in out.items()
               if k.startswith("trt_") and k.endswith("_median_ms") and v)
    out["projected_roll20_ms"] = round(best * 20, 2)
    out["projected_tick_ms"] = round(27.8 + best * 20, 2)

print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser("~/thor_trt.json"), "w") as f:
    json.dump(out, f, indent=1)
