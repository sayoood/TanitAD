"""Combined deploy-tick harness: fp16 encoder + CUDA-graph predictor, measured.

Production & Optimization backlog A3 (run #4, 2026-07-18): run #4 measured the two
orthogonal latency levers SEPARATELY and then *projected* the deployed tick:

  run #3/#4 fp32 tick 14.79 ms = encode 8.98 + select 5.81.
  - encoder is COMPUTE-bound -> fp16 lever (encode 8.98 -> 4.69 ms, run #3).
  - predictor/select is LAUNCH-bound -> CUDA-graph lever (select 5.94 -> 4.45 ms, run #4).
  additive projection: fp16-encode 4.69 + graph-select 4.45 = ~9.1 ms ~= 109 Hz.

That 9.1 ms / 109 Hz is an ADDITIVE projection from two separately-measured stages.
This harness replaces it with a MEASURED combined tick: run the two levers together
in one decision tick and time it end-to-end, on real comma2k19 windows, reporting the
true deployed decision cost (latency AND selection agreement, G-P2).

Tick definition (identical to run #3/#4): decision_tick = encode(1 frame) + select_K9.
  - reference : fp32 eager encode + fp32 eager select_K9.
  - fp16_eager: fp16 encode + fp16 eager select_K9 (isolates the encoder lever).
  - fp16_graph: fp16 encode + CUDA-graph fp32 select_K9  <-- THE DEPLOY RECIPE.

Accuracy (G-P2): the full-tick decision is decoded by ONE fixed fp64 RidgeProbe
(precision/graph invariant) over the K=9 sustained-steer fan -> selection agreement %
and decoded-waypoint shift in METRES vs the fp32-eager reference tick. This is the
end-to-end deployed decision cost of the combined recipe (not a per-stage delta).

Two modes (VRAM must be isolated -- co-residency inflates it, run #3/#4 P1.4c finding):
  --mode tick                     latency + accuracy of the 3 tick variants (fp32 ref
                                  co-resident with fp16 -> VRAM here is NOT trusted).
  --mode vram --precision fp32    isolated one-process clean standalone peak VRAM
  --mode vram --precision fp16    (loads exactly one model -> the true footprint; P1.4c).

Usage (local 4060, run each in its own process for clean VRAM):
  python combined_tick_harness.py --mode tick  --ckpt <ckpt> --comma-cache <dir> --out tick.json
  python combined_tick_harness.py --mode vram --precision fp32 --ckpt <ckpt> --out vram_fp32.json
  python combined_tick_harness.py --mode vram --precision fp16 --ckpt <ckpt> --out vram_fp16.json
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
#  CUDA-graph wrapper of the predictor select pass (static IO) -- from run #4
# --------------------------------------------------------------------------- #
class GraphedPredictor:
    """Capture predictor(static_states, static_actions) as one CUDA graph.

    run(states, actions) copies inputs into the static buffers, replays, returns
    the static output dict (updated in place). Timing run() = deployed cost
    (input copy_ + graph replay)."""

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
#  real-window collection (mirrors run #3/#4)
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
    rows = {"z_imag_h": [], "disp_h": [], "meta": []}
    with strict_numerics():
        for ep, t in idx:
            fw = _frames(ep, t, window).unsqueeze(0).to(device)
            aw = ep.actions[t:t + window].unsqueeze(0).float().to(device)
            states = world.encode_window(fw)
            z_h = world.imagine(states, aw)[h]
            last = t + window - 1
            yaw0, p0 = ep.poses[last, 2], ep.poses[last, :2]
            rows["z_imag_h"].append(z_h[0].cpu())
            rows["disp_h"].append(_ego_frame(ep.poses[last + h, :2] - p0, yaw0))
            rows["meta"].append((ep, t))
    for k in ("z_imag_h", "disp_h"):
        rows[k] = torch.stack(rows[k])
    return rows


def load_world(cfg, ckpt, dtype, device):
    world = WorldModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    return world.to(device=device, dtype=dtype).eval(), step


# --------------------------------------------------------------------------- #
#  fan decode under a chosen encode dtype -> fixed probe -> decoded xy (metres)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def fan_decode(enc_world, predict_fn, ep, t0, window, h, probe, device, enc_dtype):
    """K=9 sustained-steer maneuvers. encode under enc_dtype (encoder lever);
    predict via predict_fn (eager fp32 / eager fp16 / graph fp32) -> latent(h)
    -> fixed fp64 probe -> [K,2] metres."""
    fw = _frames(ep, t0, window).unsqueeze(0).expand(K_FAN, -1, -1, -1, -1).to(device, enc_dtype)
    aw = ep.actions[t0:t0 + window].unsqueeze(0).repeat(K_FAN, 1, 1).to(device, enc_dtype)
    sweep = torch.tensor(STEER_SWEEP, device=device, dtype=enc_dtype)
    aw[:, -SUSTAIN:, 0] = aw[:, -SUSTAIN:, 0] + sweep[:, None]
    states = enc_world.encode_window(fw)              # [K, W, S] under enc_dtype
    z_h = predict_fn(states, aw)[h].float().cpu()     # up-cast for the probe
    return probe.predict(z_h).numpy().reshape(K_FAN, 2)


# =========================================================================== #
#  MODE: vram  (isolated one-process clean standalone footprint, P1.4c)
# =========================================================================== #
def run_vram(args):
    device = "cuda"
    dtype = {"fp32": torch.float32, "fp16": torch.float16}[args.precision]
    cfg = base250cam_config()
    W, S = cfg.predictor.window, None
    world, step = load_world(cfg, args.ckpt, dtype, device)
    S = world.state_dim
    params_b = sum(p.numel() for p in world.parameters()) / 1e9
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # one full decision tick on random-but-shaped inputs (weights dominate VRAM;
    # this is the standalone footprint, no fp32 reference co-resident).
    frame1 = torch.rand(1, 9, 256, 256, device=device, dtype=dtype)
    with torch.no_grad():
        st = world.encode(frame1)                     # [1, S]
        states = st.unsqueeze(1).expand(K_FAN, W, -1).contiguous()
        actions = torch.rand(K_FAN, W, 2, device=device, dtype=dtype)
        for _ in range(5):
            world.imagine(states, actions)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    resv = torch.cuda.max_memory_reserved() / 1e9
    report = {
        "exp": "combined-tick/vram-isolated", "mode": "vram",
        "precision": args.precision, "ckpt_step": step,
        "params_billions": round(params_b, 4),
        "hardware": "RTX 4060 (Orin proxy, I8), isolated one-process",
        "weights_gb_theoretical": round(params_b * 1e9 * (4 if dtype == torch.float32 else 2) / 1e9, 4),
        "peak_vram_alloc_gb": round(peak, 4),
        "peak_vram_reserved_gb": round(resv, 4),
        "note": ("standalone footprint: exactly one model resident, no fp32 reference "
                 "co-resident -> trustworthy (unlike the run #3/#4 accuracy-harness rows)."),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# =========================================================================== #
#  MODE: tick  (combined-lever decision tick, latency + accuracy)
# =========================================================================== #
def run_tick(args):
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False    # honest fp32 reference
    cfg = base250cam_config()
    W = cfg.predictor.window
    world32, step = load_world(cfg, args.ckpt, torch.float32, device)
    world16, _ = load_world(cfg, args.ckpt, torch.float16, device)
    params_b = sum(p.numel() for p in world32.parameters()) / 1e9
    horizons = list(world32.predictor.cfg.horizons)
    h = max(horizons)

    eps = [load_episode(str(p), mmap=True)
           for p in sorted(Path(args.comma_cache).glob("ep_*.pt"))[:args.max_eps]]
    rows = collect_windows(world32, eps, device, W, h)
    n_all = rows["z_imag_h"].shape[0]

    # fixed fp64 probe on a disjoint fit-half; showcase = other half ----------
    perm = torch.randperm(n_all, generator=torch.Generator().manual_seed(0))
    half = n_all // 2
    fit_i = perm[:half]
    show_i = perm[half:half + args.max_windows]
    probe = RidgeProbe(alpha=1e-3).fit(rows["z_imag_h"][fit_i], rows["disp_h"][fit_i])
    probe_r2 = probe.r2(rows["z_imag_h"][show_i], rows["disp_h"][show_i])
    show_meta = [rows["meta"][i] for i in show_i.tolist()]
    subgoal = torch.stack([rows["disp_h"][i] for i in show_i]).numpy()   # [N,2]

    def _pick(fan):    # imagine-and-select: nearest imagined waypoint to subgoal
        return np.argmin(np.linalg.norm(fan - subgoal[:, None, :], axis=-1), axis=1)

    # representative window for capture/timing --------------------------------
    ep0, t0 = show_meta[0]
    fw1_32 = _frames(ep0, t0, W)[-1:].to(device, torch.float32)   # [1, C, H, W] single frame
    fw1_16 = fw1_32.to(torch.float16)
    with torch.no_grad():
        stK_32 = world32.encode(fw1_32).unsqueeze(1).expand(K_FAN, W, -1).contiguous()
        stK_16src = world16.encode(fw1_16).float().unsqueeze(1).expand(K_FAN, W, -1).contiguous()
        stK_16 = world16.encode(fw1_16).unsqueeze(1).expand(K_FAN, W, -1).contiguous()
    aK_32 = torch.rand(K_FAN, W, 2, device=device)
    aK_16 = aK_32.to(torch.float16)

    # predict_fns -------------------------------------------------------------
    def pred_eager32(s, a): return world32.imagine(s, a)
    def pred_eager16(s, a): return world16.imagine(s, a)
    graph32 = GraphedPredictor(world32.predictor, stK_32, aK_32)  # fp32 static buffers
    def pred_graph32(s, a): return graph32.run(s.float(), a.float())

    report = {
        "exp": "combined-deploy-tick", "mode": "tick",
        "backlog": "Production&Optimization A3 (run #4): measure the combined fp16+graph tick",
        "ckpt_step": step, "params_billions": round(params_b, 4),
        "hardware": "RTX 4060 (Orin proxy, I8), batch 1, tf32 off",
        "tick_def": "decision_tick = encode(1 frame) + select_K9 (identical to run #3/#4)",
        "tactical_horizon_steps": int(h), "n_accuracy_windows": int(len(show_i)),
        "n_fit_windows": int(half), "probe_imag_r2": round(float(probe_r2), 4),
        "reps": REPS,
        "notes": [
            "reference = fp32 eager tick (encode fp32 + select_K9 fp32).",
            "fp16_graph = the DEPLOY recipe: fp16 encode + CUDA-graph fp32 select "
            "(fp16 state up-cast to fp32 into the graph's static buffers).",
            "accuracy = END-TO-END decision cost: fp64-probe-decoded K=9 fan under the "
            "full recipe vs the fp32-eager reference fan -> agreement %, wp shift (m).",
            "VRAM here is co-resident (fp32+fp16 worlds) -> NOT trusted; see --mode vram.",
        ],
        "variants": {},
    }

    # reference fan (fp32 eager encode + fp32 eager predict) -------------------
    with torch.no_grad():
        ref_fan = np.stack([fan_decode(world32, pred_eager32, ep, t, W, h, probe, device,
                                       torch.float32) for ep, t in show_meta])   # [N,K,2]
    ref_pick = _pick(ref_fan)

    def decision_block(enc_world, predict_fn, enc_dtype):
        with torch.no_grad():
            fan = np.stack([fan_decode(enc_world, predict_fn, ep, t, W, h, probe, device,
                                       enc_dtype) for ep, t in show_meta])
        pick = _pick(fan)
        wp = np.linalg.norm(fan - ref_fan, axis=-1)                     # [N,K]
        sel = np.linalg.norm(
            fan[np.arange(len(pick)), ref_pick] - ref_fan[np.arange(len(ref_pick)), ref_pick],
            axis=-1)
        return {"selection_agreement": round(float((pick == ref_pick).mean()), 4),
                "n_flips": int((pick != ref_pick).sum()),
                "decoded_wp_shift_m_mean": round(float(wp.mean()), 4),
                "decoded_wp_shift_m_max": round(float(wp.max()), 4),
                "selected_wp_shift_m_mean": round(float(sel.mean()), 4),
                "fan_finite": bool(np.isfinite(fan).all())}

    # ---- variant table: (encode dtype, select fn, encode inputs) ------------
    variants = [
        ("fp32_eager", world32, pred_eager32, torch.float32, fw1_32, stK_32, aK_32),
        ("fp16_eager", world16, pred_eager16, torch.float16, fw1_16, stK_16, aK_16),
        ("fp16_graph", world16, pred_graph32, torch.float16, fw1_16, stK_16src, aK_32),
    ]
    for name, encw, pfn, edt, fw1, stK, aK in variants:
        try:
            enc_lat = _time_cuda(lambda: encw.encode(fw1))
            sel_lat = _time_cuda(lambda: pfn(stK, aK))
            tick = round(enc_lat["p50_ms"] + sel_lat["p50_ms"], 4)
            dec = ("reference" if name == "fp32_eager"
                   else decision_block(encw, pfn, edt))
            if name == "fp32_eager":
                dec = {"selection_agreement": 1.0, "n_flips": 0,
                       "decoded_wp_shift_m_mean": 0.0, "note": "self (reference)"}
            report["variants"][name] = {
                "encode_1frame": enc_lat, f"select_K{K_FAN}": sel_lat,
                "decision_tick_p50_ms": tick, "hz": round(1000.0 / tick, 1),
                "decision": dec}
        except Exception as ex:
            report["variants"][name] = {
                "error": f"{type(ex).__name__}: {ex}",
                "traceback": traceback.format_exc().splitlines()[-4:]}

    # ---- summary: validate run #4's additive projection ---------------------
    base = report["variants"]["fp32_eager"].get("decision_tick_p50_ms")
    summary = {}
    for name, blk in report["variants"].items():
        if "decision_tick_p50_ms" not in blk:
            summary[name] = {"status": "FAILED", "error": blk.get("error", "?")}
            continue
        tick = blk["decision_tick_p50_ms"]
        row = {"decision_tick_ms": tick, "hz": blk["hz"],
               "speedup_vs_fp32": round(base / tick, 3) if base else None,
               "selection_agreement": blk["decision"].get("selection_agreement"),
               "decoded_wp_shift_m_mean": blk["decision"].get("decoded_wp_shift_m_mean")}
        summary[name] = row
    # additive projection cross-check (run #4)
    if all(k in report["variants"] and "encode_1frame" in report["variants"][k]
           for k in ("fp16_eager", "fp32_eager")):
        proj = round(report["variants"]["fp16_eager"]["encode_1frame"]["p50_ms"]
                     + report["variants"]["fp16_graph"][f"select_K{K_FAN}"]["p50_ms"], 4) \
            if "encode_1frame" in report["variants"]["fp16_graph"] else None
        summary["_run4_additive_projection_ms"] = proj
        summary["_measured_fp16_graph_ms"] = report["variants"]["fp16_graph"].get(
            "decision_tick_p50_ms")
    report["SUMMARY"] = summary
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"SUMMARY": summary, "probe_imag_r2": report["probe_imag_r2"],
                      "n_windows": report["n_accuracy_windows"]}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tick", "vram"], default="tick")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--comma-cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=64)
    args = ap.parse_args()
    if args.mode == "vram":
        run_vram(args)
    else:
        assert args.comma_cache, "--comma-cache required for --mode tick"
        run_tick(args)


if __name__ == "__main__":
    main()
