"""Half-precision (FP16 / BF16) batch-1 latency + accuracy delta on the RTX 4060.

Production & Optimization backlog P1.4 (measured optimization experiment, D-020
G-H / G-P2). TensorRT-proper needs a toolchain that is NOT installed on this dev
box (`import tensorrt` -> ModuleNotFoundError; onnxruntime has CPU EP only). A
TRT-fp16 engine's FIRST question is nonetheless answerable *today* on the 4060,
and it is the honest precursor: **if plain fp16/bf16 already moves the latent
past tolerance or flips the driving decision, TRT-fp16 will too** (TRT-fp16 casts
the same weights). So this experiment measures, under one controlled session:

  speed delta  : batch-1 decision-tick latency (encode + K9 imagine-and-select)
                 and per-stage p50/p95 + peak VRAM, for fp32 / fp16 / bf16.
  accuracy delta (G-P2, ALWAYS next to the speed number):
    (a) encoder latent   : max|Δ|, mean|Δ|, cosine, rel-err vs fp32, on REAL
                           comma2k19 val windows (not random tensors).
    (b) predictor heads  : same deltas per horizon.
    (c) DECISION metric  : does reduced precision change which maneuver
                           imagine-and-select picks?  A single fp64 RidgeProbe
                           (probe_imag, precision-invariant reference) decodes
                           the K=9 sustained-steer fan under each precision; we
                           report selection-agreement %, the decoded-waypoint
                           shift in METRES, and the score-vector shift. This is
                           the deployment-critical accuracy number.

Numerics are pinned for the reference: the fp32 latents are computed under
`strict_numerics()`; fp16/bf16 outputs are compared against them. Latency is
measured for all three precisions identically (no strict_numerics, so the timing
reflects real inference incl. tensor-core paths). Any NaN/Inf under a precision
is itself the finding (ViT overflow => keep that tower higher precision, exactly
the "native-TRT ViT INT8 trap / vision tower stays FP16" lesson in the KB).

Usage (local 4060):
  python half_precision_latency_accuracy.py \
      --ckpt C:/Users/Admin/tanitad-data/eval/ckpt_full.pt \
      --comma-cache C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f \
      --out .../half_precision.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.instruments.numerics import strict_numerics
from tanitad.models.fourbrain import WorldModel
from tanitad.models.readout import RidgeProbe

K_FAN = 9
STEER_SWEEP = np.linspace(-0.12, 0.12, K_FAN)      # rad, sustained-steer fan
SUSTAIN = 4
WARMUP, REPS = 10, 100
PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# --------------------------------------------------------------------------- #
#  latency
# --------------------------------------------------------------------------- #
def _time_cuda(fn, warmup=WARMUP, reps=REPS) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    t = np.array(ts)
    return {"p50_ms": round(float(np.percentile(t, 50)), 3),
            "p95_ms": round(float(np.percentile(t, 95)), 3)}


def measure_latency(world, dtype, device) -> dict:
    W = world.predictor.cfg.window
    S = world.state_dim
    frames1 = torch.rand(1, 9, 256, 256, device=device, dtype=dtype)
    states = torch.rand(1, W, S, device=device, dtype=dtype)
    actions = torch.rand(1, W, 2, device=device, dtype=dtype)
    states_k = states.expand(K_FAN, -1, -1).contiguous()
    actions_k = torch.rand(K_FAN, W, 2, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats()
    out = {}
    with torch.no_grad():
        out["encode_1frame"] = _time_cuda(lambda: world.encode(frames1))
        out["predict_1pass"] = _time_cuda(lambda: world.imagine(states, actions))
        out[f"select_K{K_FAN}"] = _time_cuda(lambda: world.imagine(states_k, actions_k))
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    out["decision_tick_p50_ms"] = round(
        out["encode_1frame"]["p50_ms"] + out[f"select_K{K_FAN}"]["p50_ms"], 3)
    return out


# --------------------------------------------------------------------------- #
#  real-window collection (self-contained; mirrors viz_trajectory_fan.collect)
# --------------------------------------------------------------------------- #
def _ego_frame(dxy, yaw):
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    x = dxy[..., 0] * c - dxy[..., 1] * s
    y = dxy[..., 0] * s + dxy[..., 1] * c
    return torch.stack([x, y], dim=-1)


def _tac_horizon(world):
    if world.tactical_pred is not None:
        return max(world.tactical_pred.cfg.horizons)
    return max(world.predictor.cfg.horizons)


def _imagine_h(world, states, actions, h):
    pred = world.tactical_pred if world.tactical_pred is not None else world.predictor
    return pred(states, actions)[h]


def _frames(ep, t, window):
    f = ep.frames[t:t + window]
    return f.float().div(255.0) if f.dtype == torch.uint8 else f.float()


@torch.no_grad()
def collect_windows(world, episodes, device, window, stride=8):
    """Per window: fp32 state, imagined tactical latent (true actions), GT
    displacement at the tactical horizon. Used to fit probe_imag and as the
    fp32 accuracy reference."""
    h = _tac_horizon(world)
    idx = []
    for ep in episodes:
        T = ep.frames.shape[0]
        for i0 in range(0, T - window - h - 1, stride):
            idx.append((ep, i0))
    rows = {"state": [], "z_imag_h": [], "disp_h": [], "meta": []}
    with strict_numerics():
        for ep, t in idx:
            fw = _frames(ep, t, window).unsqueeze(0).to(device)
            aw = ep.actions[t:t + window].unsqueeze(0).float().to(device)
            states = world.encode_window(fw)
            z_h = _imagine_h(world, states, aw, h)
            last = t + window - 1
            yaw0, p0 = ep.poses[last, 2], ep.poses[last, :2]
            rows["state"].append(states[0, -1].cpu())
            rows["z_imag_h"].append(z_h[0].cpu())
            rows["disp_h"].append(_ego_frame(ep.poses[last + h, :2] - p0, yaw0))
            rows["meta"].append((ep, t))
    for k in ("state", "z_imag_h", "disp_h"):
        rows[k] = torch.stack(rows[k])
    return rows, h


@torch.no_grad()
def fan_decode(world, ep, t0, window, probe, device, dtype):
    """K=9 sustained-steer maneuvers -> imagined tactical latent (under `dtype`)
    -> decode with the FIXED fp64 probe -> K decoded ego-xy (metres)."""
    h = _tac_horizon(world)
    fw = _frames(ep, t0, window).unsqueeze(0).expand(K_FAN, -1, -1, -1, -1).to(device, dtype)
    aw = ep.actions[t0:t0 + window].unsqueeze(0).repeat(K_FAN, 1, 1).to(device, dtype)
    sweep = torch.tensor(STEER_SWEEP, dtype=dtype, device=device)
    aw[:, -SUSTAIN:, 0] = aw[:, -SUSTAIN:, 0] + sweep[:, None]
    states = world.encode_window(fw)
    z_h = _imagine_h(world, states, aw, h).float().cpu()   # up-cast for the probe
    return probe.predict(z_h).numpy().reshape(K_FAN, 2)     # [K,2] metres


# --------------------------------------------------------------------------- #
#  accuracy deltas
# --------------------------------------------------------------------------- #
def _delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a = reference (fp32), b = precision output. Both [N, D] float32."""
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)
    finite = bool(torch.isfinite(b).all())
    return {"max_abs": round(float(d.max()), 6),
            "mean_abs": round(float(d.mean()), 6),
            "cosine_min": round(float(cos.min()), 6),
            "cosine_mean": round(float(cos.mean()), 6),
            "rel_err_mean": round(float(rel.mean()), 6),
            "rel_err_max": round(float(rel.max()), 6),
            "finite": finite}


def load_world(cfg, ckpt, dtype, device):
    world = WorldModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    return world.to(device=device, dtype=dtype).eval(), step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=64)
    args = ap.parse_args()

    device = "cuda"
    cfg = base250cam_config()
    W = cfg.predictor.window

    # fp32 baseline model + reference latents on REAL windows -----------------
    world32, step = load_world(cfg, args.ckpt, torch.float32, device)
    params_b = sum(p.numel() for p in world32.parameters()) / 1e9
    eps = [load_episode(str(p), mmap=True)
           for p in sorted(Path(args.comma_cache).glob("ep_*.pt"))[:args.max_eps]]
    rows, h = collect_windows(world32, eps, device, W)
    n_all = rows["state"].shape[0]

    # fit probe_imag (fp64) on a disjoint fit-half; showcase = other half -----
    perm = torch.randperm(n_all, generator=torch.Generator().manual_seed(0))
    half = n_all // 2
    fit_i, show_i = perm[:half], perm[half:half + args.max_windows]
    probe = RidgeProbe(alpha=1e-3).fit(rows["z_imag_h"][fit_i], rows["disp_h"][fit_i])
    probe_r2 = probe.r2(rows["z_imag_h"][show_i], rows["disp_h"][show_i])

    # fp32 reference encoder/predictor latents on the showcase windows --------
    show_meta = [rows["meta"][i] for i in show_i.tolist()]
    with strict_numerics(), torch.no_grad():
        ref_state, ref_pred = [], {hh: [] for hh in cfg.predictor.horizons}
        for ep, t in show_meta:
            fw = _frames(ep, t, W).unsqueeze(0).to(device)
            aw = ep.actions[t:t + W].unsqueeze(0).float().to(device)
            st = world32.encode_window(fw)
            ref_state.append(st[0, -1].cpu())
            pr = world32.imagine(st, aw)
            for hh in cfg.predictor.horizons:
                ref_pred[hh].append(pr[hh][0].cpu())
    ref_state = torch.stack(ref_state)
    ref_pred = {hh: torch.stack(v) for hh, v in ref_pred.items()}
    ref_fan = np.stack([fan_decode(world32, ep, t, W, probe, device, torch.float32)
                        for ep, t in show_meta])          # [N,K,2] metres
    ref_sel = ref_fan_argmin = None

    # subgoal per window = GT displacement at the tactical horizon (fan target)
    subgoal = torch.stack([rows["disp_h"][i] for i in show_i]).numpy()   # [N,2]
    def _pick(fan):    # imagine-and-select: nearest imagined waypoint to subgoal
        return np.argmin(np.linalg.norm(fan - subgoal[:, None, :], axis=-1), axis=1)
    ref_pick = _pick(ref_fan)

    report = {
        "exp": "half-precision-latency-accuracy",
        "backlog": "Production&Optimization P1.4 (TRT-fp16 precursor; TRT toolchain not installed)",
        "ckpt_step": step, "params_billions": round(params_b, 4),
        "hardware": "RTX 4060 (Orin proxy, I8), batch 1",
        "n_showcase_windows": int(len(show_i)), "n_fit_windows": int(half),
        "tactical_horizon_steps": int(h), "probe_imag_r2": round(probe_r2, 4),
        "notes": [
            "fp32 latents computed under strict_numerics (pinned reference).",
            "Latency measured identically for all precisions (tensor-core paths on).",
            "probe_imag is a single fp64 ridge map, IDENTICAL across precisions -> "
            "any selection flip is attributable to WM precision, not the probe.",
            "imagine-and-select pick = argmin_K ||decoded_fan - subgoal|| (subgoal = "
            "GT ego displacement at the 1.6 s tactical horizon).",
            "TensorRT-proper: `import tensorrt` -> ModuleNotFoundError; ORT CPU-only. "
            "TRT-fp16 engine build needs the NVIDIA TRT toolchain (propose install).",
        ],
        "precisions": {},
    }

    for name, dtype in PRECISIONS.items():
        world = world32 if name == "fp32" else load_world(cfg, args.ckpt, dtype, device)[0]
        lat = measure_latency(world, dtype, device)
        block = {"latency": lat}
        if name == "fp32":
            block["accuracy_vs_fp32"] = "reference"
            block["decision"] = {"selection_agreement": 1.0,
                                 "note": "self (reference)"}
        else:
            with torch.no_grad():
                enc_p, pred_p = [], {hh: [] for hh in cfg.predictor.horizons}
                for ep, t in show_meta:
                    fw = _frames(ep, t, W).unsqueeze(0).to(device, dtype)
                    aw = ep.actions[t:t + W].unsqueeze(0).to(device, dtype)
                    st = world.encode_window(fw)
                    enc_p.append(st[0, -1].float().cpu())
                    pr = world.imagine(st, aw)
                    for hh in cfg.predictor.horizons:
                        pred_p[hh].append(pr[hh][0].float().cpu())
                enc_p = torch.stack(enc_p)
                pred_p = {hh: torch.stack(v) for hh, v in pred_p.items()}
                fan_p = np.stack([fan_decode(world, ep, t, W, probe, device, dtype)
                                  for ep, t in show_meta])
            pick_p = _pick(fan_p)
            agree = float((pick_p == ref_pick).mean())
            wp_shift = np.linalg.norm(fan_p - ref_fan, axis=-1)   # metres, [N,K]
            sel_wp_shift = np.linalg.norm(
                fan_p[np.arange(len(pick_p)), ref_pick]
                - ref_fan[np.arange(len(ref_pick)), ref_pick], axis=-1)
            block["accuracy_vs_fp32"] = {
                "encoder_state": _delta(ref_state, enc_p),
                "predictor": {f"h{hh}": _delta(ref_pred[hh], pred_p[hh])
                              for hh in cfg.predictor.horizons},
            }
            block["decision"] = {
                "selection_agreement": round(agree, 4),
                "n_flips": int((pick_p != ref_pick).sum()),
                "decoded_wp_shift_m_mean": round(float(wp_shift.mean()), 4),
                "decoded_wp_shift_m_max": round(float(wp_shift.max()), 4),
                "selected_wp_shift_m_mean": round(float(sel_wp_shift.mean()), 4),
                "fan_finite": bool(np.isfinite(fan_p).all()),
            }
        report["precisions"][name] = block
        if name != "fp32":
            del world
            torch.cuda.empty_cache()

    # tidy speed-vs-accuracy summary -----------------------------------------
    base_tick = report["precisions"]["fp32"]["latency"]["decision_tick_p50_ms"]
    summary = {}
    for name in PRECISIONS:
        tick = report["precisions"][name]["latency"]["decision_tick_p50_ms"]
        row = {"decision_tick_ms": tick, "speedup_vs_fp32": round(base_tick / tick, 3),
               "hz": round(1000.0 / tick, 1),
               "peak_vram_gb": report["precisions"][name]["latency"]["peak_vram_gb"]}
        if name != "fp32":
            dec = report["precisions"][name]["decision"]
            enc = report["precisions"][name]["accuracy_vs_fp32"]["encoder_state"]
            row["selection_agreement"] = dec["selection_agreement"]
            row["encoder_rel_err_mean"] = enc["rel_err_mean"]
            row["encoder_finite"] = enc["finite"]
            row["decoded_wp_shift_m_mean"] = dec["decoded_wp_shift_m_mean"]
        summary[name] = row
    report["SUMMARY_speed_vs_accuracy"] = summary

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"SUMMARY_speed_vs_accuracy": summary,
                      "probe_imag_r2": report["probe_imag_r2"],
                      "n_showcase_windows": report["n_showcase_windows"]}, indent=2))


if __name__ == "__main__":
    main()
