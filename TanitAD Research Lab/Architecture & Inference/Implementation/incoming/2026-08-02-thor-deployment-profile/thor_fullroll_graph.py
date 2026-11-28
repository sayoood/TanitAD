"""Lever #2: capture the ENTIRE 20-step imagination roll in ONE CUDA graph.

WHY THIS IS NOW THE TARGET: after bf16, the combined tick is 98.63 ms of which the 20-step roll
is ~69.6 ms = **71 %**. It used to be the minority cost. Per-step graph replay already removed the
launch overhead *inside* each step, but 20 separate `g.replay()` calls still pay 20 launch
boundaries plus 20 `torch.cat` window slides in Python. One capture removes all of it.

⭐ The whole roll is capturable because every shape is STATIC: the window slide is a fixed-size
tensor shuffle, not a data-dependent branch. Nothing in the loop depends on a value.

⚠️ ACCURACY IS THE POINT, NOT AN AFTERTHOUGHT (their G-P2 rule). Per-step capture was measured
BIT-EXACT (rel-err 0.0). A full-roll capture reuses one set of static buffers across 20 steps, so
an aliasing mistake would silently corrupt the rollout while still producing plausible numbers —
which is exactly the failure class that is hard to catch later. Every variant below is checked
against the eager fp32 reference, and a variant that drifts is REPORTED AS FAILED, not shipped.
"""
import json
import os
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

DEV, H, W, K = "cuda", 176, 624, 20
out = {"device": torch.cuda.get_device_name(0), "K": K,
       "context": "roll is 71% of the 98.63 ms tick after bf16 — this is the dominant stage"}

cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection="cylindrical", v2_subframe="176x624", f_ref=None)
resolve_v2_frames(ns, cfg, label="thor_fullroll")
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, "tactical_pred", None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)

model = WorldModel(cfg).to(DEV).eval()
Wn, C, A = cfg.predictor.window, cfg.encoder.in_channels, cfg.predictor.action_dim
frames = torch.randn(1, Wn, C, H, W, device=DEV)


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 2),
            "p99_ms": round(ts[min(len(ts) - 1, int(len(ts) * 0.99))], 2)}


with torch.no_grad():
    states0 = model.encode_window(frames)
    acts = torch.randn(1, Wn, A, device=DEV)

    # ---------------- reference: eager fp32 roll ----------------
    def roll_eager():
        ws = states0
        for _ in range(K):
            z = model.predictor(ws, acts)[1]
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
        return ws
    ref = roll_eager().float().clone()
    out["roll_eager"] = bench(roll_eager)

    # ---------------- variant A: per-step graph (today's baseline) ----------------
    st_s, st_a = states0.clone(), acts.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model.predictor(st_s, st_a)
    torch.cuda.current_stream().wait_stream(s)
    g_step = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_step):
        z_step = model.predictor(st_s, st_a)[1]

    def roll_perstep():
        ws = states0
        for _ in range(K):
            st_s.copy_(ws); st_a.copy_(acts)
            g_step.replay()
            ws = torch.cat([ws[:, 1:], z_step.unsqueeze(1)], dim=1)
        return ws
    a_out = roll_perstep().float()
    out["roll_perstep_graph"] = bench(roll_perstep)
    out["perstep_rel_err"] = round(float((a_out - ref).norm() / ref.norm()), 9)

    # ---------------- variant B: the WHOLE roll in ONE graph ----------------
    # Static buffers: the loop mutates `buf` in place, so capture sees a fixed dataflow.
    buf = states0.clone()
    buf_a = acts.clone()
    try:
        s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            for _ in range(3):
                w = buf.clone()
                for _ in range(K):
                    z = model.predictor(w, buf_a)[1]
                    w = torch.cat([w[:, 1:], z.unsqueeze(1)], dim=1)
        torch.cuda.current_stream().wait_stream(s2)

        g_full = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_full):
            w_cap = buf.clone()
            for _ in range(K):
                z_c = model.predictor(w_cap, buf_a)[1]
                w_cap = torch.cat([w_cap[:, 1:], z_c.unsqueeze(1)], dim=1)
            full_out = w_cap

        def roll_full():
            buf.copy_(states0); buf_a.copy_(acts)
            g_full.replay()
            return full_out

        b_out = roll_full().float()
        rel_b = float((b_out - ref).norm() / ref.norm())
        out["roll_full_graph"] = bench(roll_full)
        out["fullgraph_rel_err"] = round(rel_b, 9)
        if rel_b > 1e-4:
            out["roll_full_graph"]["⛔ REJECTED"] = (
                f"rel-err {rel_b:.2e} vs eager — a full-roll capture that DRIFTS is a "
                "buffer-aliasing bug, not a speedup. Not shippable.")
        else:
            out["fullgraph_speedup_vs_perstep_x"] = round(
                out["roll_perstep_graph"]["p50_ms"] / out["roll_full_graph"]["p50_ms"], 2)
            out["fullgraph_speedup_vs_eager_x"] = round(
                out["roll_eager"]["p50_ms"] / out["roll_full_graph"]["p50_ms"], 2)
    except Exception as e:
        out["roll_full_graph"] = {"FAILED": f"{type(e).__name__}: {str(e)[:250]}",
                                  "note": "a failure is a RESULT — it means the roll must stay "
                                          "per-step and the next lever is TensorRT"}

    # ---------------- projected new tick ----------------
    best_roll = None
    if isinstance(out.get("roll_full_graph"), dict) and "p50_ms" in out["roll_full_graph"] \
            and "⛔ REJECTED" not in out["roll_full_graph"]:
        best_roll = out["roll_full_graph"]["p50_ms"]
    else:
        best_roll = out["roll_perstep_graph"]["p50_ms"]
    out["projected_tick_ms"] = round(27.8 + best_roll, 2)   # bf16 encoder + best roll
    out["projected_vs_budget_pct"] = round(out["projected_tick_ms"], 1)

print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser("~/thor_fullroll_graph.json"), "w") as f:
    json.dump(out, f, indent=1)
