"""P2 — the FIRST real beyond-ADE numbers computable without the renderer.

The occluder / work-zone / traffic-light closed-loop scenarios need a rendered signal/occlusion
geometry (MetaDrive/CARLA), which is NOT installed on the dev box (probed 2026-07-24: absent in the
`tanitad` and `carla312` venvs and in pip metadata; `metadrive_frontcam.py` lazily imports it and
its live-rollout test skips). So LAL / OKRI / LOPS / TLC stay renderer/label-gated here.

But TWO of the five (+TLC) DO run on real log-replay telemetry we already have:

  * **TMS** — scores the EXPERT LOG's tactical smoothness over real comma2k19 val episodes. This is a
    reference band for later closed-loop comparison, NOT a claim about our policy (the log is the
    human expert). Model-independent.
  * **CNCE** — D_progress from the real log, the measured decision-tick latency of the real
    base250cam WorldModel architecture on this GPU, the real active-parameter count, and collisions=0
    (log replay, by construction). Latency + params are weight-value-independent, so this is a real
    ARCHITECTURE efficiency number (identical for trained or freshly-initialised weights).

This mirrors the accepted `stack/scripts/latency_cnce_baseline.py`; it is self-contained so it needs
no trained checkpoint (the deployed weights are on HF/pods; pulling them needs sign-off and the pods
are training). Run:

    PYTHONPATH=stack python "<this file>" \
        --comma-cache C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f \
        --out real_tms_cnce.json [--max-eps 30] [--reps 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.eval.metrics import ScenarioTelemetry, compute_cnce, compute_tms
from tanitad.models.fourbrain import WorldModel

K_CANDIDATES = 9          # tactical maneuver vocabulary size (Phase 0) — matches latency_cnce_baseline
DT = 0.1                  # comma2k19 physics step


def _time_cuda(fn, warmup: int, reps: int) -> float:
    """p50 milliseconds of fn() on CUDA (event timing), or wall-clock ms on CPU."""
    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return float(np.percentile(ts, 50))
    import time
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(ts, 50))


def measure_decision_tick_ms(world, device, warmup: int, reps: int) -> dict:
    """Decision tick = encode(1 frame) + K-candidate imagine-and-select pass (latency_cnce_baseline)."""
    cfg_w = world.predictor.cfg
    W = cfg_w.window
    frames1 = torch.rand(1, 9, 256, 256, device=device)
    latent = world.encode(frames1).shape[-1]
    states = torch.rand(1, W, latent, device=device)
    states_k = states.expand(K_CANDIDATES, -1, -1).contiguous()
    actions_k = torch.rand(K_CANDIDATES, W, 2, device=device)
    with torch.no_grad():
        enc = _time_cuda(lambda: world.encode(frames1), warmup, reps)
        sel = _time_cuda(lambda: world.imagine(states_k, actions_k), warmup, reps)
    return {"encode_1frame_p50_ms": round(enc, 3),
            f"select_K{K_CANDIDATES}_p50_ms": round(sel, 3),
            "decision_tick_p50_ms": round(enc + sel, 3)}


def episode_telemetry(ep, latency_ms: float, params_b: float) -> ScenarioTelemetry:
    """Real-log telemetry: v from poses, jerk/steer-rate from CAN actions (as latency_cnce_baseline)."""
    v = np.asarray(ep.poses)[:, 3]
    accel = np.asarray(ep.actions)[:, 1]
    steer = np.asarray(ep.actions)[:, 0]
    jerk = np.gradient(accel, DT)
    steer_rate = np.abs(np.gradient(steer, DT))
    T = len(v)
    nan2 = np.full((T, 2), np.nan)
    return ScenarioTelemetry(
        ego_v=v, ego_jerk=jerk, steer_rate=steer_rate,
        latency_ms=np.full(T, latency_ms),
        hazard_los_flag=np.zeros(T, bool), dist_to_blind_spot=np.full(T, 1e9),
        is_occluded_flag=np.zeros(T, bool), wm_hazard_xy=nan2, gt_hazard_xy=nan2,
        dt=DT, collisions=0, params_billions=params_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=30)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    world = WorldModel(base250cam_config()).to(device).eval()
    params_b = sum(p.numel() for p in world.parameters()) / 1e9

    lat = measure_decision_tick_ms(world, device, warmup=10, reps=args.reps)
    tick_ms = lat["decision_tick_p50_ms"]

    eps_paths = sorted(Path(args.comma_cache).glob("ep_*.pt"))[:args.max_eps]
    rows = []
    for p in eps_paths:
        ep = load_episode(str(p), mmap=True)
        tel = episode_telemetry(ep, tick_ms, params_b)
        rows.append({
            "episode_id": int(ep.episode_id),
            "T": int(np.asarray(ep.poses).shape[0]),
            "TMS_expert_log": round(compute_tms(tel), 4),
            "CNCE": round(compute_cnce(tel), 2),
            "progress_m": round(float(np.trapezoid(tel.ego_v, dx=DT)), 1),
        })

    tms = [r["TMS_expert_log"] for r in rows]
    cnce = [r["CNCE"] for r in rows]
    report = {
        "exp": "sc14-P2-real-tms-cnce-log-replay",
        "evidence_class": "MEASURED (real comma2k19 val telemetry + real base250cam architecture)",
        "device": (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
        "params_billions": round(params_b, 4),
        "n_episodes": len(rows),
        "comma_cache": str(args.comma_cache),
        "latency": lat,
        "TMS_expert_log": {"median": round(float(np.median(tms)), 4),
                           "min": float(np.min(tms)), "max": float(np.max(tms))},
        "CNCE": {"median": round(float(np.median(cnce)), 2),
                 "min": float(np.min(cnce)), "max": float(np.max(cnce))},
        "notes": [
            "TMS scores the EXPERT LOG smoothness (reference band), NOT our policy (P8).",
            "CNCE: collisions=0 by log-replay construction; latency+params are weight-independent "
            "(so identical for trained vs random init) -> a real ARCHITECTURE efficiency number.",
            "LAL/OKRI/LOPS/TLC need rendered occlusion/signal geometry (MetaDrive/CARLA), absent on "
            "the dev box -> renderer-gated, NOT computed here (no telemetry, no number).",
        ],
        "episodes": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("device", "params_billions", "n_episodes", "latency",
                       "TMS_expert_log", "CNCE")}, indent=2))


if __name__ == "__main__":
    main()
