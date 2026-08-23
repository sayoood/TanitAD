#!/usr/bin/env python3
"""D1 — REBUILD the predictor engine at batch 9 (dynamic 1..9) and re-measure.

The runbook's 2026-08-03 pricing annotation: the shipped ``predictor_fp16.plan``
is **batch-1 static** while the deployed ``TacticalSelector`` fans over **9**
candidates (``config.py:95``, ``fourbrain.py:571``) — 244 % of the 100 ms budget
serialised, 56 % batched. This script builds the engine that fixes it, verifies
it BY LOADING AND EXECUTING (never by exit code), and measures:

  * per-batch engine latency 1..9 — **in-process** (what deployment pays,
    including the python binding + set_input_shape) AND ``trtexec`` (kernel-only),
    because the published 1.168/1.294 ms are trtexec medians and mixing the two
    would overstate the deployed headroom;
  * the FAN as executed: 9 candidates x K=20 steps, serialised through a batch-1
    engine vs batched through the dynamic engine vs eager — MEASURED end to end,
    not composed arithmetically;
  * p50 AND p95 (the brief's requirement; p99 kept for continuity with the
    runbook's standing rule).

⛔ Geometry: v1 = 256x256 SQUARE (its trained raster); v5f = 176x624 cylindrical.
Never cross them. The predictor itself consumes 2048-d latents and is
geometry-independent — that is asserted here, not assumed.

Run:  setsid nohup python thor_d1_batch9_engine.py > /tmp/thor_d1.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import dataclasses
import json
import os
import statistics as st
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

DEV = "cuda"
HOME = os.path.expanduser("~")
WORK = f"{HOME}/trt_d1"
OUTJ = f"{HOME}/thor_d1_batch9_engine.json"
V1_CKPT = f"{HOME}/models/flagship-v1-speedjerk/ckpt.pt"
V5F_CKPT = f"{HOME}/models/v5f/ckpt.pt"
WIN, K_ROLL, N_CAND = 8, 20, 9
sys.path.insert(0, f"{HOME}/TanitAD/stack")
sys.path.insert(0, f"{HOME}/TanitAD/stack/scripts")
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")
os.makedirs(WORK, exist_ok=True)

from build_predictor_trt import (TRTPredictor, build_engine,  # noqa: E402
                                 export_onnx, verify_engine)
from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402

OUT = {"script": os.path.abspath(__file__), "host": os.uname().nodename,
       "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "budget_ms": 100.0, "n_candidates": N_CAND, "k_roll": K_ROLL}
import tensorrt as trt  # noqa: E402

OUT["tensorrt"] = trt.__version__


def dump():
    OUT["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(OUTJ, "w") as f:
        json.dump(OUT, f, indent=2)


def stats(samples_ms):
    s = sorted(samples_ms)
    n = len(s)
    return {"n": n, "p50": round(st.median(s), 4),
            "p95": round(s[min(n - 1, int(0.95 * n))], 4),
            "p99": round(s[min(n - 1, int(0.99 * n))], 4),
            "mean": round(st.fmean(s), 4), "min": round(s[0], 4)}


def bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return stats(out)


# ============================================================= models (REAL weights)
def build_v1():
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)
    if getattr(cfg, "tactical_pred", None) is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    m = WorldModel(cfg)
    ck = torch.load(V1_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])                      # STRICT
    meta = {"ckpt": V1_CKPT, "step": int(ck.get("step", -1)), "strict_load": True,
            "raster": f"{cfg.encoder.image_size}x{cfg.encoder.image_size} SQUARE",
            "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
            "state_dim": m.state_dim, "action_dim": cfg.predictor.action_dim}
    assert cfg.encoder.image_size == 256, "v1 must stay at its TRAINED 256px raster"
    return m.to(DEV).eval(), cfg, meta


def build_v5f():
    from train_flagship_v4 import resolve_v2_frames
    cfg = flagship4b_config()
    ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                         projection="cylindrical", v2_subframe="176x624", f_ref=None)
    resolve_v2_frames(ns, cfg, label="d1")
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    m = WorldModel(cfg)
    ck = torch.load(V5F_CKPT, map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    meta = {"ckpt": V5F_CKPT, "step": int(ck.get("step", -1)),
            "strict_load": False, "sd_missing": len(miss), "sd_unexpected": len(unexp),
            "raster": f"{cfg.encoder.frame_h}x{cfg.encoder.frame_w} CYLINDRICAL"
                      if hasattr(cfg.encoder, "frame_h") else "176x624 CYLINDRICAL",
            "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
            "state_dim": m.state_dim, "action_dim": cfg.predictor.action_dim}
    return m.to(DEV).eval(), cfg, meta


print("=== STAGE 1: REAL weights ===", flush=True)
v1, v1cfg, v1meta = build_v1()
OUT["v1"] = v1meta
print("v1", v1meta, flush=True)
v5f, v5cfg, v5meta = build_v5f()
OUT["v5f"] = v5meta
print("v5f", v5meta, flush=True)
S, A = v1.state_dim, 3
assert v5f.state_dim == S, "predictor engines are only shared if state_dim matches"
OUT["predictor_geometry_independent"] = {
    "v1_state_dim": v1.state_dim, "v5f_state_dim": v5f.state_dim,
    "note": "the predictor consumes 2048-d latents, not pixels — asserted, not assumed"}
dump()

# ============================================================= STAGE 2: build engines
print("=== STAGE 2: export + build ===", flush=True)
ENGINES = {}
plans = {}
for tag, model, max_b, static in [
        ("v1_dyn1-9_fp16", v1, 9, False),      # ⭐ THE DEPLOYMENT ARTIFACT
        ("v1_static_b1_fp16", v1, 1, True),    # the SHIPPED shape — the control
        ("v5f_dyn1-9_fp16", v5f, 9, False)]:
    onnx_p, plan_p = f"{WORK}/{tag}.onnx", f"{WORK}/{tag}.plan"
    if os.environ.get("REUSE_PLANS") == "1" and os.path.exists(plan_p):
        # Re-verify the SAME plan bytes rather than rebuilding — a rebuild would
        # invalidate the verification already banked against this file.
        rec = {"onnx": {"path": onnx_p, "reused": True},
               "build": {"ok": True, "reused_plan": plan_p,
                         "MB": round(os.path.getsize(plan_p) / 1e6, 1)}}
    else:
        rec = {"onnx": export_onnx(model.predictor, onnx_p, batch=max_b, window=WIN,
                                   state_dim=S, action_dim=A, device=DEV, opset=17,
                                   dynamic=not static)}
        rec["build"] = build_engine(onnx_p, plan_p, window=WIN, state_dim=S,
                                    action_dim=A, min_batch=1, opt_batch=max_b,
                                    max_batch=max_b, fp16=True, dynamic=not static)
    print(tag, rec["build"]["ok"], rec["build"].get("build_s"), flush=True)
    assert rec["build"]["ok"], f"{tag} build failed: {rec['build'].get('tail')}"
    # ⛔ VERIFY BY LOADING AND EXECUTING, never by exit code.
    eng, rec["verify"] = verify_engine(plan_p, model.predictor, window=WIN,
                                       state_dim=S, action_dim=A,
                                       batches=(1,) if static else (1, 9), device=DEV)
    ENGINES[tag] = eng
    plans[tag] = plan_p
    ENGINES[tag] = eng
    OUT.setdefault("engines", {})[tag] = rec
    print(tag, "profile", rec["verify"]["profile_states"],
          "rel_err", {k: v["rel_err"] for k, v in rec["verify"]["per_batch"].items()},
          flush=True)
    dump()

# The claim that decides whether ONE engine can serve the fleet's two arms.
_st = torch.randn(9, WIN, S, device=DEV)
_ac = torch.randn(9, WIN, A, device=DEV)
with torch.no_grad():
    _a = ENGINES["v1_dyn1-9_fp16"](_st, _ac)[1]
    _b = ENGINES["v5f_dyn1-9_fp16"](_st, _ac)[1]
OUT["two_arms_are_different_engines"] = {
    "rel_err_v1_vs_v5f_same_input": round(float((_a - _b).norm() / _a.norm()), 6),
    "why": ("engines carry WEIGHTS: the same input must give different outputs, "
            "or one of them was built from the wrong checkpoint")}
dump()

# ============================================================= STAGE 3: batch sweep
print("=== STAGE 3: per-batch latency ===", flush=True)
sweep = {}
dyn = ENGINES["v1_dyn1-9_fp16"]
b1 = ENGINES["v1_static_b1_fp16"]
for b in range(1, N_CAND + 1):
    stt = torch.randn(b, WIN, S, device=DEV)
    act = torch.randn(b, WIN, A, device=DEV)
    row = {"engine_inproc": bench(lambda: dyn(stt, act))}
    with torch.no_grad():
        row["eager"] = bench(lambda: v1.predictor(stt, act), warmup=10, iters=60)
    row["per_candidate_engine_p50"] = round(row["engine_inproc"]["p50"] / b, 4)
    row["per_candidate_eager_p50"] = round(row["eager"]["p50"] / b, 4)
    sweep[b] = row
    print("batch", b, row["engine_inproc"]["p50"], row["eager"]["p50"], flush=True)
OUT["batch_sweep_v1_dyn"] = sweep
stt1 = torch.randn(1, WIN, S, device=DEV)
act1 = torch.randn(1, WIN, A, device=DEV)
OUT["batch_sweep_v1_static_b1"] = {1: bench(lambda: b1(stt1, act1))}
dump()

# trtexec kernel-only medians — the metric the published 1.168/1.294 ms used.
print("=== STAGE 3b: trtexec medians (kernel-only, published metric) ===", flush=True)
tex = {}
for tag, b in [("v1_dyn1-9_fp16", 1), ("v1_dyn1-9_fp16", 9),
               ("v1_static_b1_fp16", 1)]:
    cmd = ["/usr/src/tensorrt/bin/trtexec", f"--loadEngine={plans[tag]}",
           "--iterations=200", "--warmUp=500", "--avgRuns=200"]
    if "dyn" in tag:
        cmd.append(f"--shapes=states:{b}x{WIN}x{S},actions:{b}x{WIN}x{A}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    med = None
    for ln in (r.stdout or "").splitlines():
        if "GPU Compute Time" in ln and "median" in ln:
            for part in ln.split(","):
                if "median" in part:
                    med = float(part.split("=")[1].strip().split(" ")[0])
    tex[f"{tag}_b{b}"] = {"median_ms": med}
    print("trtexec", tag, b, med, flush=True)
OUT["trtexec_medians"] = tex
dump()

# ============================================================= STAGE 4: the FAN
print("=== STAGE 4: the 9-candidate fan, MEASURED ===", flush=True)
states0 = torch.randn(1, WIN, S, device=DEV)
acts0 = torch.randn(1, WIN, A, device=DEV)
prim = torch.randn(N_CAND, K_ROLL, A, device=DEV) * 0.3


def roll_serialised(pred, n=N_CAND, k=K_ROLL):
    """What the deployed selector does TODAY: one candidate at a time."""
    for i in range(n):
        s, a = states0.clone(), acts0.clone()
        for j in range(k):
            a = torch.roll(a, -1, dims=1)
            a[:, -1] = prim[i, j]
            z = pred(s, a)[1]
            s = torch.roll(s, -1, dims=1)
            s[:, -1] = z


def roll_batched(pred, n=N_CAND, k=K_ROLL):
    """What ``propose_and_score(batch_fan=True)`` does: N rows, K steps."""
    s = states0.expand(n, -1, -1).contiguous()
    a = acts0.expand(n, -1, -1).contiguous()
    for j in range(k):
        a = torch.roll(a, -1, dims=1)
        a[:, -1] = prim[:n, j]
        z = pred(s, a)[1]
        s = torch.roll(s, -1, dims=1)
        s[:, -1] = z


fan = {}
OUT["fan_measured"] = fan          # bind first: every row banks as it lands


def measure(name, fn, warm, iters, grad_off=False):
    if grad_off:
        with torch.no_grad():
            fan[name] = bench(fn, warm, iters)
    else:
        fan[name] = bench(fn, warm, iters)
    print("fan", name, "p50", fan[name]["p50"], "p95", fan[name]["p95"], flush=True)
    dump()


measure("eager_serialised", lambda: roll_serialised(v1.predictor), 3, 10, True)
measure("eager_batched", lambda: roll_batched(v1.predictor), 3, 10, True)
measure("trt_b1_serialised_SHIPPED", lambda: roll_serialised(b1), 2, 10)
measure("trt_dyn_serialised", lambda: roll_serialised(dyn), 2, 10)
measure("trt_dyn_batched_FIX", lambda: roll_batched(dyn), 3, 20)
measure("single_candidate_trt_dyn", lambda: roll_batched(dyn, n=1), 3, 20)
OUT["fan_speedup_shipped_to_fixed"] = round(
    fan["trt_b1_serialised_SHIPPED"]["p50"] / fan["trt_dyn_batched_FIX"]["p50"], 3)
dump()
print("=== DONE ===", OUTJ, flush=True)
