"""Batch-1 latency baseline (I8, Production P0.1) + first real TMS/CNCE numbers.

Two backlog items in one measured experiment (D-020 G-H):

1. **I8 latency** — batch-1 streaming decision tick on the local RTX 4060 (the
   declared Orin proxy): encoder forward, predictor multi-horizon pass, and the
   K-candidate imagine-and-select tactical pass. CUDA-event timing, p50/p95
   over N reps after warmup, peak VRAM, one nvidia-smi power sample mid-loop.

2. **TMS / CNCE on real telemetry** — the two Deep-Think-14 metrics that do NOT
   need closed-loop occlusion geometry. TMS over comma2k19 val episodes scores
   the EXPERT LOG's smoothness (an honest reference band for later closed-loop
   comparison — not a claim about our policy). CNCE uses real log progress
   (D = integral v dt), the measured decision-tick latency, and the true active
   parameter count — the first real number for the efficiency-moat metric.
   Collisions = 0 by construction in log replay (footnoted).

LAL / OKRI / LOPS still require scripted occluder scenarios (CARLA-on-pod,
W31-32) and are NOT computed here (P8: no telemetry, no number).

Usage:
  python stack/scripts/latency_cnce_baseline.py --ckpt .../ckpt_full.pt \
      --comma-cache .../comma2k19-val-<hash> --out .../latency_cnce.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.eval.metrics import (ScenarioTelemetry, compute_cnce, compute_tms)
from tanitad.instruments.numerics import strict_numerics

K_CANDIDATES = 9            # tactical maneuver vocabulary size (Phase 0)
WARMUP, REPS = 10, 100
PARAMS_BILLIONS = None      # measured from the model, not hardcoded


def _time_cuda(fn, warmup: int = WARMUP, reps: int = REPS) -> dict:
    """p50/p95 milliseconds of fn() on the current CUDA stream."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    t = np.array(times)
    return {"p50_ms": round(float(np.percentile(t, 50)), 3),
            "p95_ms": round(float(np.percentile(t, 95)), 3)}


def measure_latency(world, device) -> dict:
    cfg_w = world.predictor.cfg
    W = cfg_w.window
    frames1 = torch.rand(1, 9, 256, 256, device=device)
    states = torch.rand(1, W, world.encode(frames1).shape[-1], device=device)
    actions = torch.rand(1, W, 2, device=device)
    states_k = states.expand(K_CANDIDATES, -1, -1).contiguous()
    actions_k = torch.rand(K_CANDIDATES, W, 2, device=device)

    torch.cuda.reset_peak_memory_stats()
    out = {}
    with torch.no_grad():
        out["encode_1frame"] = _time_cuda(lambda: world.encode(frames1))
        out["predict_1pass"] = _time_cuda(lambda: world.imagine(states, actions))
        out[f"select_K{K_CANDIDATES}"] = _time_cuda(
            lambda: world.imagine(states_k, actions_k))
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,name",
                 "--format=csv,noheader"], capture_output=True, text=True,
                timeout=10)
            out["gpu_power_sample"] = p.stdout.strip()
        except Exception:
            out["gpu_power_sample"] = "unavailable"
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    out["decision_tick_p50_ms"] = round(
        out["encode_1frame"]["p50_ms"] + out[f"select_K{K_CANDIDATES}"]["p50_ms"], 3)
    return out


def episode_telemetry(ep, latency_ms: float, params_b: float) -> ScenarioTelemetry:
    """Real-log telemetry: v from poses, jerk/steer-rate from CAN actions."""
    dt = 0.1
    v = ep.poses[:, 3].numpy()
    accel = ep.actions[:, 1].numpy()
    steer = ep.actions[:, 0].numpy()
    jerk = np.gradient(accel, dt)
    steer_rate = np.abs(np.gradient(steer, dt))
    T = len(v)
    nan2 = np.full((T, 2), np.nan)
    return ScenarioTelemetry(
        ego_v=v, ego_jerk=jerk, steer_rate=steer_rate,
        latency_ms=np.full(T, latency_ms),
        hazard_los_flag=np.zeros(T, bool),
        dist_to_blind_spot=np.full(T, 1e9),
        is_occluded_flag=np.zeros(T, bool),
        wm_hazard_xy=nan2, gt_hazard_xy=nan2,
        dt=dt, collisions=0, params_billions=params_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=12)
    args = ap.parse_args()

    device = "cuda"
    from tanitad.models.fourbrain import WorldModel
    world = WorldModel(base250cam_config())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.to(device).eval()
    params_b = sum(p.numel() for p in world.parameters()) / 1e9

    with strict_numerics():
        lat = measure_latency(world, device)

    tick_ms = lat["decision_tick_p50_ms"]
    eps = [load_episode(str(p), mmap=True)
           for p in sorted(Path(args.comma_cache).glob("ep_*.pt"))[:args.max_eps]]
    rows = []
    for ep in eps:
        tel = episode_telemetry(ep, tick_ms, params_b)
        rows.append({"episode_id": int(ep.episode_id),
                     "T": int(ep.frames.shape[0]),
                     "TMS_expert_log": round(compute_tms(tel), 4),
                     "CNCE": round(compute_cnce(tel), 2),
                     "progress_m": round(float(np.trapezoid(
                         tel.ego_v, dx=tel.dt)) if hasattr(np, "trapezoid")
                         else float(np.trapz(tel.ego_v, dx=tel.dt)), 1)})

    tms = [r["TMS_expert_log"] for r in rows]
    cnce = [r["CNCE"] for r in rows]
    report = {
        "exp": "i8-latency+tms-cnce-baseline", "ckpt_step": step,
        "hardware": "RTX 4060 (Orin proxy, I8), fp32, strict numerics, batch 1",
        "params_billions": round(params_b, 4),
        "latency": lat,
        "notes": [
            "TMS scores the EXPERT LOG smoothness (reference band), not our policy",
            "CNCE: collisions=0 by construction in log replay; latency = measured "
            "decision tick (encode + K9 select) on the 4060",
            "LAL/OKRI/LOPS need closed-loop occluder telemetry (CARLA W31-32) - "
            "not computed (P8)"],
        "TMS_expert_log": {"median": round(float(np.median(tms)), 4),
                           "min": min(tms), "max": max(tms)},
        "CNCE": {"median": round(float(np.median(cnce)), 2),
                 "min": min(cnce), "max": max(cnce)},
        "episodes": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("latency", "TMS_expert_log", "CNCE", "params_billions")},
                     indent=2))


if __name__ == "__main__":
    main()
