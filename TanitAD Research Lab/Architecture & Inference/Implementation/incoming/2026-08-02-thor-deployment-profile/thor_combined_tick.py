"""Thor combined-tick optimisation — REPLICATING the prod-optimization agent's measured playbook.

⭐ NOT a fresh search. The Production & Optimization stream already MEASURED, on a 4060:

    * "the whole win is the ViT; the predictor is LAUNCH-BOUND"        (their fp16 diagnosis)
    * manual torch.cuda.CUDAGraph on the PREDICTOR = **2.57x** (predict-1) / 1.33x (K9),
      rel-err 2.8e-7, decision-agreement **100 %**, wp-shift 0.00 m
    * fp16 encoder + CUDA-graph predictor combined tick = **17.75 -> 11.16 ms, 1.59x**,
      agreement 96.9 %, and the measured value matched the additive projection to 0.4 %
      => on that silicon THE LEVERS COMPOSED
    * torch.compile NOT viable (Triton missing -> inductor fails) => deploy via MANUAL capture

My first Thor pass tested graph capture on the ENCODER and got only 1.09x — which is not a
contradiction, it is their diagnosis reproducing: a COMPUTE-bound stage cannot be helped by
removing launch overhead. This run puts each lever on the stage their measurements assign it:

    encoder   -> precision (bf16 measured 6.76x on Thor)
    predictor -> manual CUDA-graph capture (launch-bound)

⚠️ G-P2 (their own rule): an accuracy delta next to every speed delta. Every timing below is
paired with a numerical-agreement check against the fp32 eager reference — a speedup whose
output drifted is not a speedup, and their bar was 95.3 % decision-agreement.
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

DEV, H, W = "cuda", 176, 624
K_ROLL = 20
out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "replicates": "Production&Optimization run #4/#5 (4060): predictor graph 2.57x, "
                     "combined tick 1.59x, torch.compile not viable"}

cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection="cylindrical", v2_subframe="176x624", f_ref=None)
resolve_v2_frames(ns, cfg, label="thor_tick")
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, "tactical_pred", None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)

model = WorldModel(cfg).to(DEV).eval()
Wn, C, A = cfg.predictor.window, cfg.encoder.in_channels, cfg.predictor.action_dim
frames = torch.randn(1, Wn, C, H, W, device=DEV)


def bench(fn, warmup=10, iters=50):
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
    # ============================ reference: fp32 eager ============================
    states32 = model.encode_window(frames)
    acts = torch.randn(1, Wn, A, device=DEV)
    ref_z = model.predictor(states32, acts)[1].float().clone()

    out["encoder_fp32"] = bench(lambda: model.encode_window(frames))

    def enc_bf16():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.encode_window(frames)
    out["encoder_bf16"] = bench(enc_bf16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z_bf16 = model.encode_window(frames).float()
    out["encoder_bf16_accuracy"] = {
        "rel_err_vs_fp32": round(float((z_bf16 - states32).norm() / states32.norm()), 6),
        "max_abs_dz": round(float((z_bf16 - states32).abs().max()), 6)}

    # ==================== predictor: eager vs MANUAL CUDA GRAPH ====================
    out["predictor_1step_eager"] = bench(lambda: model.predictor(states32, acts))

    st_s, st_a = states32.clone(), acts.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model.predictor(st_s, st_a)
    torch.cuda.current_stream().wait_stream(s)
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        g_out = model.predictor(st_s, st_a)[1]
    st_s.copy_(states32); st_a.copy_(acts)
    g1.replay(); torch.cuda.synchronize()
    out["predictor_1step_graph"] = bench(lambda: g1.replay())
    out["predictor_graph_speedup_x"] = round(
        out["predictor_1step_eager"]["p50_ms"] / out["predictor_1step_graph"]["p50_ms"], 2)
    out["predictor_graph_accuracy"] = {
        "rel_err_vs_eager": round(float((g_out.float() - ref_z).norm() /
                                        ref_z.norm().clamp_min(1e-12)), 9),
        "note": "their 4060 bar: rel-err 2.8e-7, agreement 100 %"}

    # ================ the K-step roll: eager vs graph-replayed loop ================
    def roll_eager():
        ws = states32
        for _ in range(K_ROLL):
            z = model.predictor(ws, acts)[1]
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
        return ws
    out["roll20_eager"] = bench(roll_eager, warmup=3, iters=15)

    def roll_graph():
        ws = states32
        for _ in range(K_ROLL):
            st_s.copy_(ws); st_a.copy_(acts)
            g1.replay()
            ws = torch.cat([ws[:, 1:], g_out.unsqueeze(1)], dim=1)
        return ws
    out["roll20_graph"] = bench(roll_graph, warmup=3, iters=15)
    out["roll20_speedup_x"] = round(
        out["roll20_eager"]["p50_ms"] / out["roll20_graph"]["p50_ms"], 2)

    # ============================ the COMBINED TICK ============================
    def tick_baseline():
        st = model.encode_window(frames)
        ws = st
        for _ in range(K_ROLL):
            z = model.predictor(ws, acts)[1]
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
        return ws
    out["tick_fp32_eager"] = bench(tick_baseline, warmup=3, iters=15)

    def tick_optimised():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            st = model.encode_window(frames)
        ws = st.float()
        for _ in range(K_ROLL):
            st_s.copy_(ws); st_a.copy_(acts)
            g1.replay()
            ws = torch.cat([ws[:, 1:], g_out.unsqueeze(1)], dim=1)
        return ws
    out["tick_bf16enc_graphpred"] = bench(tick_optimised, warmup=3, iters=15)
    out["TICK_SPEEDUP_x"] = round(
        out["tick_fp32_eager"]["p50_ms"] / out["tick_bf16enc_graphpred"]["p50_ms"], 2)

    # do the levers COMPOSE, as they did on the 4060 (to 0.4 %)?
    enc_gain = out["encoder_fp32"]["p50_ms"] - out["encoder_bf16"]["p50_ms"]
    pred_gain = out["roll20_eager"]["p50_ms"] - out["roll20_graph"]["p50_ms"]
    projected = out["tick_fp32_eager"]["p50_ms"] - enc_gain - pred_gain
    measured = out["tick_bf16enc_graphpred"]["p50_ms"]
    out["composition_check"] = {
        "additive_projection_ms": round(projected, 2),
        "measured_ms": round(measured, 2),
        "delta_pct": round(100 * (measured - projected) / max(projected, 1e-9), 1),
        "verdict": ("LEVERS COMPOSE (within 10 %) — matches the 4060 finding"
                    if abs(measured - projected) / max(projected, 1e-9) < 0.10
                    else "LEVERS DO NOT COMPOSE on Thor — sequence them, do not add them")}
    out["budget_100ms"] = {
        "baseline_pct": round(100 * out["tick_fp32_eager"]["p50_ms"] / 100.0, 1),
        "optimised_pct": round(100 * measured / 100.0, 1)}

print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser("~/thor_combined_tick.json"), "w") as f:
    json.dump(out, f, indent=1)
