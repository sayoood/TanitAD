"""Predictor batch-1 latency attack (CUDA graphs / torch.compile) on the RTX 4060.

Production & Optimization backlog P0 #3 (fleet directive 2026-07-17): the
2026-07-17 clean-GPU profile found the fp16 win is ALL the ViT encoder; the
predictor + K9-select passes are batch-1 *launch-bound*, not compute-bound
(fp16 barely moved them: 5.81 -> 5.99 ms). Launch-bound == kernel-launch
overhead dominates. The right lever for that is **CUDA-graph capture** (replay a
whole recorded launch sequence in one shot) or torch.compile's cudagraphs
backend -- NOT precision, NOT quantization.

Target = the OPERATIVE predictor pass (``world.imagine`` == ``world.predictor``),
i.e. exactly the ``predict_1pass`` / ``select_K9`` passes the prior profile timed.

Measured, on the 4060 (declared Orin proxy, I8), under one controlled fp32 session:

  speed delta   : batch-1 p50/p95 latency for the predictor pass and the K=9
                  imagine-and-select pass, eager vs each optimization.
  accuracy delta (G-P2, reported next to every speed number):
    - predictor latent : max|delta|, rel-err, cosine vs eager, per horizon, on
                         REAL comma2k19 windows (not random tensors).
    - DECISION metric  : does the optimization change which maneuver
                         imagine-and-select picks?  The K=9 sustained-steer fan
                         is decoded by one FIXED fp64 RidgeProbe (identical
                         across variants) -> selection-agreement %, decoded
                         waypoint shift in METRES. A CUDA graph replays the SAME
                         fp32 kernels, so the expectation is ~bitwise identity
                         (rel-err ~0, agreement 1.0): a *free* latency win.

Pre-registered FALSIFIER (backlog): if the predictor speedup is < 10 %, the
tick is memory-bound (not launch-bound) -> record the roofline number and close
the item; CUDA graphs cannot help a memory-bound kernel.

Optimizations attempted:
  A. manual torch.cuda.CUDAGraph capture of the predictor pass (Triton-free).
  B. torch.compile(backend="cudagraphs")  (dynamo cudagraphs, Triton-free).
  C. torch.compile(mode="reduce-overhead") (inductor+cudagraphs; needs Triton).
     On this Windows dev box Triton is NOT installed -> C is expected to fall
     back / error; that outcome is itself a documented deployment finding.

Usage (local 4060):
  python predictor_latency_attack.py \
      --ckpt C:/Users/Admin/tanitad-data/eval/ckpt_full.pt \
      --comma-cache C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f \
      --out .../predictor_latency.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.instruments.numerics import strict_numerics
from tanitad.models.fourbrain import WorldModel
from tanitad.models.readout import RidgeProbe

K_FAN = 9
STEER_SWEEP = np.linspace(-0.12, 0.12, K_FAN)   # rad, sustained-steer fan
SUSTAIN = 4
WARMUP, REPS = 20, 200


# --------------------------------------------------------------------------- #
#  timing
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
    return {"p50_ms": round(float(np.percentile(t, 50)), 4),
            "p95_ms": round(float(np.percentile(t, 95)), 4)}


# --------------------------------------------------------------------------- #
#  CUDA-graph wrapper of the predictor pass (static IO)
# --------------------------------------------------------------------------- #
class GraphedPredictor:
    """Capture predictor(static_states, static_actions) as one CUDA graph.

    run(states, actions) copies inputs into the static buffers, replays, and
    returns the static output tensors (updated in place). Timing run() reflects
    the deployed cost: input copy + graph replay.
    """

    def __init__(self, predictor, states, actions):
        self.predictor = predictor
        self.static_states = states.clone()
        self.static_actions = actions.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self.predictor(self.static_states, self.static_actions)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.static_out = self.predictor(self.static_states, self.static_actions)

    def run(self, states, actions):
        self.static_states.copy_(states)
        self.static_actions.copy_(actions)
        self.graph.replay()
        return self.static_out


# --------------------------------------------------------------------------- #
#  real-window collection
# --------------------------------------------------------------------------- #
def _ego_frame(dxy, yaw):
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    x = dxy[..., 0] * c - dxy[..., 1] * s
    y = dxy[..., 0] * s + dxy[..., 1] * c
    return torch.stack([x, y], dim=-1)


def _frames(ep, t, window):
    f = ep.frames[t:t + window]
    return f.float().div(255.0) if f.dtype == torch.uint8 else f.float()


@torch.no_grad()
def collect_windows(world, episodes, device, window, h, stride=8):
    idx = []
    for ep in episodes:
        T = ep.frames.shape[0]
        for i0 in range(0, T - window - h - 1, stride):
            idx.append((ep, i0))
    rows = {"state": [], "action": [], "z_imag_h": [], "disp_h": []}
    with strict_numerics():
        for ep, t in idx:
            fw = _frames(ep, t, window).unsqueeze(0).to(device)
            aw = ep.actions[t:t + window].unsqueeze(0).float().to(device)
            states = world.encode_window(fw)
            z_h = world.imagine(states, aw)[h]
            last = t + window - 1
            yaw0, p0 = ep.poses[last, 2], ep.poses[last, :2]
            rows["state"].append(states[0].cpu())            # [W, S]
            rows["action"].append(aw[0].cpu())               # [W, 2]
            rows["z_imag_h"].append(z_h[0].cpu())
            rows["disp_h"].append(_ego_frame(ep.poses[last + h, :2] - p0, yaw0))
    for k in rows:
        rows[k] = torch.stack(rows[k])
    return rows


def _delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)
    return {"max_abs": float(f"{float(d.max()):.3e}"),
            "mean_abs": float(f"{float(d.mean()):.3e}"),
            "cosine_min": round(float(cos.min()), 6),
            "rel_err_mean": float(f"{float(rel.mean()):.3e}"),
            "rel_err_max": float(f"{float(rel.max()):.3e}"),
            "finite": bool(torch.isfinite(b).all())}


def load_world(cfg, ckpt, device):
    world = WorldModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    return world.to(device=device, dtype=torch.float32).eval(), step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=64)
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False   # honest fp32 reference
    cfg = base250cam_config()
    W = cfg.predictor.window
    world, step = load_world(cfg, args.ckpt, device)
    params_b = sum(p.numel() for p in world.parameters()) / 1e9
    pred = world.predictor                          # the launch-bound target
    horizons = list(pred.cfg.horizons)
    h = max(horizons)

    eps = [load_episode(str(p), mmap=True)
           for p in sorted(Path(args.comma_cache).glob("ep_*.pt"))[:args.max_eps]]
    rows = collect_windows(world, eps, device, W, h)
    n_all = rows["state"].shape[0]
    n = min(args.max_windows, n_all)
    states_all = rows["state"][:n].to(device)          # [N, W, S]
    actions_all = rows["action"][:n].to(device)        # [N, W, 2]

    # fp64 probe on eager imagined latent(h) -> GT displacement (fixed ref) ----
    probe = RidgeProbe(alpha=1e-3).fit(rows["z_imag_h"][:n], rows["disp_h"][:n])
    probe_r2 = probe.r2(rows["z_imag_h"][:n], rows["disp_h"][:n])
    subgoal = rows["disp_h"][:n].numpy()               # [N, 2]

    # representative batch-1 window + K=9 fan for capture/timing --------------
    s1 = states_all[:1].contiguous()
    a1 = actions_all[:1].contiguous()
    sK = states_all[:1].expand(K_FAN, -1, -1).contiguous()
    aK = actions_all[:1].repeat(K_FAN, 1, 1).contiguous()
    sweep = torch.tensor(STEER_SWEEP, dtype=torch.float32, device=device)
    aK[:, -SUSTAIN:, 0] = aK[:, -SUSTAIN:, 0] + sweep[:, None]

    report = {
        "exp": "predictor-batch1-latency-attack",
        "backlog": "Production&Optimization P0#3 (fleet 2026-07-17): launch-bound predictor",
        "ckpt_step": step, "params_billions": round(params_b, 4),
        "hardware": "RTX 4060 (Orin proxy, I8), fp32 (tf32 off), batch 1",
        "tactical_horizon_steps": int(h),
        "n_accuracy_windows": int(n),
        "predictor_module": type(pred).__name__,
        "predictor_depth": int(pred.cfg.depth),
        "probe_imag_r2": round(float(probe_r2), 4),
        "reps": REPS,
        "notes": [
            "predictor pass = the operative transformer forward (world.imagine).",
            "CUDA graph replays the SAME fp32 kernels -> accuracy delta ~0 expected "
            "(the win is pure kernel-launch elimination).",
            "run()/replay timing includes the input copy_ (deployed cost).",
            "falsifier: <10% predictor speedup => memory-bound, not launch-bound.",
        ],
        "variants": {},
    }

    @torch.no_grad()
    def eager_predict(s, a):
        return world.imagine(s, a)

    # decision-agreement: decode K=9 fan latent(h) with the fixed probe -> pick
    def _pick_from_fan_latent(zK):     # zK: [K, S] tensor -> argmin index int
        xy = probe.predict(zK.float().cpu()).numpy().reshape(K_FAN, 2)
        return int(np.argmin(np.linalg.norm(xy - subgoal[0][None, :], axis=-1))), xy

    ref_pick, ref_xy = _pick_from_fan_latent(eager_predict(sK, aK)[h][:, :])

    # ---- eager baseline -----------------------------------------------------
    lat_eager = {
        "predict_1pass": _time_cuda(lambda: eager_predict(s1, a1)),
        f"select_K{K_FAN}": _time_cuda(lambda: eager_predict(sK, aK)),
    }
    report["variants"]["eager_fp32"] = {
        "latency": lat_eager, "accuracy_vs_eager": "reference",
        "decision": {"selection_agreement": 1.0, "n_flips": 0,
                     "decoded_wp_shift_m_mean": 0.0}}
    ref_all = {k: v.detach().cpu() for k, v in eager_predict(states_all, actions_all).items()}

    def _decision_block(fan_latent):
        pick, xy = _pick_from_fan_latent(fan_latent)
        shift = float(np.linalg.norm(xy - ref_xy, axis=-1).mean())
        return {"selection_agreement": float(pick == ref_pick),
                "n_flips": int(pick != ref_pick),
                "decoded_wp_shift_m_mean": round(shift, 5),
                "fan_finite": bool(torch.isfinite(fan_latent).all())}

    # ---- A. manual CUDA graph ----------------------------------------------
    try:
        g1 = GraphedPredictor(pred, s1, a1)
        gK = GraphedPredictor(pred, sK, aK)
        graph_all = {k: [] for k in horizons}
        for i in range(n):
            o = g1.run(states_all[i:i + 1], actions_all[i:i + 1])
            for k in horizons:
                graph_all[k].append(o[k].detach().cpu().clone())
        graph_all = {k: torch.cat(v) for k, v in graph_all.items()}
        acc = {f"h{k}": _delta(ref_all[k], graph_all[k]) for k in horizons}
        dec = _decision_block(gK.run(sK, aK)[h].clone())
        lat_g = {
            "predict_1pass": _time_cuda(lambda: g1.run(s1, a1)),
            f"select_K{K_FAN}": _time_cuda(lambda: gK.run(sK, aK)),
        }
        report["variants"]["cuda_graph_manual"] = {
            "latency": lat_g, "accuracy_vs_eager": acc, "decision": dec}
    except Exception as ex:
        report["variants"]["cuda_graph_manual"] = {
            "error": f"{type(ex).__name__}: {ex}",
            "traceback": traceback.format_exc().splitlines()[-4:]}

    # ---- B/C. torch.compile variants ---------------------------------------
    for label, kwargs in [("compile_cudagraphs", {"backend": "cudagraphs"}),
                          ("compile_reduce_overhead", {"mode": "reduce-overhead"})]:
        try:
            torch._dynamo.reset()
            cpred = torch.compile(lambda s, a: world.imagine(s, a), **kwargs)
            with torch.no_grad():
                for _ in range(5):
                    cpred(s1, a1)
                co = {k: v.detach().cpu() for k, v in cpred(states_all, actions_all).items()}
                dec = _decision_block(cpred(sK, aK)[h].clone())
            acc = {f"h{k}": _delta(ref_all[k], co[k]) for k in horizons}
            lat_c = {
                "predict_1pass": _time_cuda(lambda: cpred(s1, a1)),
                f"select_K{K_FAN}": _time_cuda(lambda: cpred(sK, aK)),
            }
            report["variants"][label] = {
                "latency": lat_c, "accuracy_vs_eager": acc, "decision": dec}
        except Exception as ex:
            report["variants"][label] = {
                "error": f"{type(ex).__name__}: {ex}",
                "traceback": traceback.format_exc().splitlines()[-4:]}

    # ---- summary ------------------------------------------------------------
    base = {k: lat_eager[k]["p50_ms"] for k in lat_eager}
    summary = {}
    for name, blk in report["variants"].items():
        if "latency" not in blk:
            summary[name] = {"status": "FAILED", "error": blk.get("error", "?")}
            continue
        row = {}
        for stage in base:
            t = blk["latency"][stage]["p50_ms"]
            row[stage] = {"p50_ms": t, "speedup": round(base[stage] / t, 3)}
        if isinstance(blk["accuracy_vs_eager"], dict):
            rel = max(v["rel_err_max"] for v in blk["accuracy_vs_eager"].values())
            fin = all(v["finite"] for v in blk["accuracy_vs_eager"].values())
            row["accuracy"] = {"worst_rel_err_max": rel, "all_finite": fin}
        if "decision" in blk:
            row["decision"] = {k: blk["decision"][k] for k in
                               ("selection_agreement", "decoded_wp_shift_m_mean")}
        summary[name] = row
    report["SUMMARY"] = summary
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"SUMMARY": summary, "n_windows": int(n),
                      "predictor": report["predictor_module"],
                      "depth": report["predictor_depth"],
                      "probe_r2": report["probe_imag_r2"]}, indent=2))


if __name__ == "__main__":
    main()
