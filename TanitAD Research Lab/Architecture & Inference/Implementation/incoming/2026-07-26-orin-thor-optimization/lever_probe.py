#!/usr/bin/env python3
"""Quantify the candidate optimization levers WITHOUT Orin/Thor hardware.

Four measurements, all device-independent or format-intrinsic:

  A. ENCODER SCALING -- FLOPs/params vs input resolution (token count).  This is
     the crop lever, and it is the number the "widen the 51.4 deg crop of a
     120.5 deg camera" stream needs to know the deployment cost of.
  B. ROLLOUT STRUCTURE -- per-step predictor FLOPs vs window and depth, plus the
     KV-cache CEILING (what a perfect incremental-decode cache could save).
  C. FAN BATCHING -- arithmetic intensity vs candidate count K.  Explains the
     already-MEASURED "marginal candidate ~= 0.3 ms" and prices the hierarchy.
  D. QUANTIZATION FORMAT NUMERICS -- weight round-trip error of INT8-per-channel
     (the format that PASSED the 2026-07-23 Gate B, +0.0065 m), FP8-E4M3,
     NVFP4 (E2M1 + per-16 E4M3 block scale), MXFP4, INT4-per-channel.
     Format-intrinsic; establishes the ORDERING that decides whether Thor's
     only real lever (weight compression) is worth a real-weight Gate-B run.

  E. ROOFLINE -- Orin / Thor envelopes from PUBLISHED vendor specs, applied to
     the MEASURED per-component budget from budget_probe.py.

Run: python lever_probe.py --budget artifacts/budget_report.json \
                           --out artifacts/lever_report.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.config import (EncoderConfig, PredictorConfig,           # noqa: E402
                            flagship4b_config)
from tanitad.eval.ckpt_compat import adapt_config_action_dim          # noqa: E402
from tanitad.models.encoder import ViTEncoder                          # noqa: E402
from tanitad.models.predictor import OperativePredictor                # noqa: E402
from tanitad.models.readout import SpatialGridReadout                  # noqa: E402

K_ROLLOUT = 20
WINDOW = 8


# --------------------------------------------------------------------------- #
# PUBLISHED device envelopes.  Everything marked DERIVED is arithmetic on a
# PUBLISHED figure with the operation stated -- never a second vendor claim.
# --------------------------------------------------------------------------- #
DEVICES = {
    "jetson_agx_orin_64gb": {
        "arch": "Ampere GA10B, SM 8.7",
        "ai_perf_published": "275 TOPS INT8 (sparse), whole module",
        "gpu_int8_sparse_tops": 170.0,        # PUBLISHED (nvidia.com Orin page)
        "dla_int8_sparse_tops": 105.0,        # DERIVED 275 - 170; DLA CANNOT run
                                              # our attention (PUBLISHED refutation)
        "gpu_int8_dense_tops": 85.0,          # DERIVED /2 (sparse->dense)
        "gpu_fp16_dense_tflops": 42.5,        # DERIVED /2 (int8->fp16)
        "gpu_fp8_dense_tflops": None,         # NO FP8 datapath on Ampere
        "gpu_fp4_dense_tflops": None,
        "mem_gb": 64, "mem_type": "256-bit LPDDR5",
        "bw_gb_s": 204.8,                     # PUBLISHED
        "l2_mb": 4.0,
        "power_w": "15-60",                   # PUBLISHED
        "src": "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/",
    },
    "jetson_agx_thor_t5000": {
        "arch": "Blackwell + Transformer Engine",
        "ai_perf_published": "2070 TFLOPS FP4 (sparse)",
        "gpu_fp4_sparse_tflops": 2070.0,      # PUBLISHED (nvidia.com Thor page)
        "gpu_fp4_dense_tflops": 1035.0,       # DERIVED /2
        "gpu_fp8_dense_tflops": 517.5,        # DERIVED /2
        "gpu_fp16_dense_tflops": 258.75,      # DERIVED /2
        "gpu_int8_dense_tops": 517.5,         # DERIVED (int8 == fp8 rate)
        "mem_gb": 128, "mem_type": "256-bit LPDDR5X",
        "bw_gb_s": 273.0,                     # PUBLISHED
        "l2_mb": None,
        "power_w": "40-130",                  # PUBLISHED
        "src": "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/",
    },
}
PRECISION_BYTES = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "fp8": 1.0, "nvfp4": 0.5}


# --------------------------------------------------------------------------- #
def count_flops(fn) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    off = False
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        off = True
    except Exception:
        pass
    try:
        with torch.no_grad():
            fn()
        c = FlopCounterMode(display=False)
        with c, torch.no_grad():
            fn()
        return float(c.get_total_flops())
    finally:
        if off:
            torch.backends.mha.set_fastpath_enabled(True)


def nparams(m) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# D. format quantizers                                                        #
# --------------------------------------------------------------------------- #
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _q_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest FP4-E2M1 magnitude (NVFP4 element format)."""
    s = torch.sign(x)
    a = x.abs()
    grid = _E2M1.to(x.device, x.dtype)
    mid = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(a, mid)
    return s * grid[idx]


def _q_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through OCP FP8-E4M3 (max normal 448)."""
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def fq_int_per_channel(w: torch.Tensor, bits: int) -> torch.Tensor:
    """The EXACT scheme the 2026-07-23 Gate-B run used (per-output-channel
    symmetric), so `bits=8` here is the format that MEASURED +0.0065 m."""
    wf = w.float().reshape(w.shape[0], -1)
    amax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    qmax = 2 ** (bits - 1) - 1
    scale = amax / qmax
    q = torch.clamp(torch.round(wf / scale), -qmax - 1, qmax)
    return (q * scale).reshape(w.shape)


def fq_fp8_per_channel(w: torch.Tensor) -> torch.Tensor:
    wf = w.float().reshape(w.shape[0], -1)
    amax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / 448.0
    return (_q_e4m3(wf / scale) * scale).reshape(w.shape)


def fq_block_fp4(w: torch.Tensor, block: int, scale_fmt: str) -> torch.Tensor:
    """NVFP4 (block=16, E4M3 block scale) / MXFP4 (block=32, E8M0 power-of-two
    block scale).  Both use the E2M1 element format."""
    wf = w.float().reshape(-1)
    pad = (-wf.numel()) % block
    if pad:
        wf = torch.cat([wf, wf.new_zeros(pad)])
    wb = wf.reshape(-1, block)
    amax = wb.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    s = amax / 6.0                                    # 6.0 = E2M1 max magnitude
    if scale_fmt == "e4m3":
        s = _q_e4m3(s).clamp_min(1e-12)
    elif scale_fmt == "e8m0":
        s = torch.pow(2.0, torch.ceil(torch.log2(s)))
    out = (_q_e2m1(wb / s) * s).reshape(-1)
    return out[: w.numel()].reshape(w.shape).to(w.dtype)


def fmt_error(w: torch.Tensor, wq: torch.Tensor) -> dict:
    a, b = w.float().reshape(w.shape[0], -1), wq.float().reshape(w.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    rel = ((a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12))
    snr = 20 * math.log10(a.norm().item()
                          / max((a - b).norm().item(), 1e-30))
    return {"cos_mean": round(cos.mean().item(), 8),
            "cos_min": round(cos.min().item(), 8),
            "rel_l2_mean": round(rel.mean().item(), 6),
            "snr_db": round(snr, 2)}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="artifacts/budget_report.json")
    ap.add_argument("--out", default="artifacts/lever_report.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    budget = json.loads(Path(args.budget).read_text(encoding="utf-8"))

    base = adapt_config_action_dim(flagship4b_config(), 3)
    rep: dict = {"what": "optimization-lever quantification without Orin/Thor "
                         "silicon (device-independent + format-intrinsic)",
                 "env": {"torch": torch.__version__, "device": str(dev)},
                 "devices_published": DEVICES}

    # ---------- A. encoder scaling / the crop lever -------------------------
    print("[A] encoder scaling sweep")
    ec = base.encoder
    rows = []
    # The ViT is parameterised by TOKEN COUNT (its `pos` table is square), so
    # sweep square token grids and read the FLOP(N) curve; any crop/resolution
    # scenario is then evaluated at its own N by the fitted curve below.
    for g in (8, 12, 16, 20, 24, 28, 32):
        px = g * ec.patch_size
        cfg = dataclasses.replace(ec, image_size=px)
        enc = ViTEncoder(cfg).to(dev).eval()
        x = torch.randn(1, ec.in_channels, px, px, device=dev)
        f = count_flops(lambda: enc(x))
        act = None
        if dev.type == "cuda":
            torch.cuda.synchronize(dev); torch.cuda.empty_cache()
            b0 = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            with torch.no_grad():
                enc(x)
            torch.cuda.synchronize(dev)
            act = round((torch.cuda.max_memory_allocated(dev) - b0) / 1e6, 3)
        rows.append({"token_grid": f"{g}x{g}", "n_tokens": g * g, "px": px,
                     "gflops": round(f / 1e9, 3), "params": nparams(enc),
                     "activation_mb": act})
        del enc
    ref = next(r["gflops"] for r in rows if r["n_tokens"] == 256)
    for r in rows:
        r["vs_deployed_x"] = round(r["gflops"] / ref, 3)

    # least-squares fit  GFLOP(N) = a*N + b*N^2  (linear = projections/MLP,
    # quadratic = attention score+context matmuls)
    Nv = torch.tensor([float(r["n_tokens"]) for r in rows])
    Yv = torch.tensor([r["gflops"] for r in rows])
    A = torch.stack([Nv, Nv ** 2], 1)
    coef = torch.linalg.lstsq(A, Yv.unsqueeze(1)).solution.squeeze(1)
    a, b = coef[0].item(), coef[1].item()
    pred_y = A @ coef
    r2 = 1 - ((Yv - pred_y) ** 2).sum().item() / (
        ((Yv - Yv.mean()) ** 2).sum().item())

    def gflop_at(n):
        return a * n + b * n * n

    # 120.5 / 51.4 = 2.344x wider FOV.  Two ways to widen, very different cost.
    scen = [
        ("DEPLOYED 51.4 deg crop, 256 px, 16x16", 256),
        ("HALVE the crop to 25.7 deg at same angular res (8x16 tokens)", 128),
        ("0.75x linear downscale of the deployed crop (12x12)", 144),
        ("WIDEN to full 120.5 deg at SAME angular res (37.5x16 tokens)", 600),
        ("WIDEN to full 120.5 deg, TOKEN-BUDGET-MATCHED (rescale to 256)", 256),
        ("WIDEN to full 120.5 deg at 0.75x angular res (28x12)", 336),
    ]
    scen_rows = [{"scenario": s, "n_tokens": n,
                  "gflops_fitted": round(gflop_at(n), 3),
                  "vs_deployed_x": round(gflop_at(n) / gflop_at(256), 3)}
                 for s, n in scen]

    rep["A_encoder_scaling"] = {
        "note": "the ViT trunk's PARAMS are token-count INDEPENDENT, so a crop "
                "change moves DRAM WEIGHT traffic by EXACTLY ZERO. It is a "
                "pure FLOP + activation lever -- and the encoder is the "
                "compute-bound half of the tick, so this is the only place "
                "where FLOPs are the right currency.",
        "measured_rows": rows,
        "fit": {"model": "GFLOP(N) = a*N + b*N^2",
                "a_linear": round(a, 6), "b_quadratic": round(b, 9),
                "r2": round(r2, 6), "n_points": len(rows),
                "fit_window_tokens": [64, 1024],
                "quadratic_share_at_N256": round(
                    b * 256 ** 2 / gflop_at(256) * 100, 1)},
        "crop_scenarios_fitted": scen_rows}
    for r in rows:
        print("   ", r["token_grid"], r["n_tokens"], r["gflops"], "GFLOP")
    for r in scen_rows:
        print("    scen", r["vs_deployed_x"], "x ", r["scenario"][:56])

    # ---------- B. rollout structure + KV-cache ceiling ---------------------
    print("[B] rollout structure")
    S = 2048
    pc = base.predictor
    pred = OperativePredictor(pc, S).to(dev).eval()
    st = torch.randn(1, WINDOW, S, device=dev)
    ac = torch.randn(1, WINDOW, pc.action_dim, device=dev)
    f_full = count_flops(lambda: pred(st, ac))

    # KV-cache CEILING: a perfect incremental decode computes the transformer
    # trunk for ONE new position instead of all 8.  Emulated by a window-1
    # predictor of identical width/depth (attention over 1 key is the lower
    # bound; a real sliding cache attends 8 keys, so this brackets the win).
    pc1 = dataclasses.replace(pc, window=1)
    pred1 = OperativePredictor(pc1, S).to(dev).eval()
    f_incr = count_flops(lambda: pred1(st[:, -1:], ac[:, -1:]))

    depth_rows = []
    for d in (10, 8, 6, 4):
        pcd = dataclasses.replace(pc, depth=d)
        pm = OperativePredictor(pcd, S).to(dev).eval()
        depth_rows.append({"depth": d, "params": nparams(pm),
                           "gflops_per_step": round(count_flops(
                               lambda: pm(st, ac)) / 1e9, 4)})
        del pm
    width_rows = []
    for dm in (768, 640, 512, 384):
        pcw = dataclasses.replace(pc, d_model=dm,
                                  n_heads=max(1, dm // 64))
        pm = OperativePredictor(pcw, S).to(dev).eval()
        width_rows.append({"d_model": dm, "params": nparams(pm),
                           "gflops_per_step": round(count_flops(
                               lambda: pm(st, ac)) / 1e9, 4)})
        del pm
    win_rows = []
    for wn in (8, 6, 4, 2):
        pcw = dataclasses.replace(pc, window=wn)
        pm = OperativePredictor(pcw, S).to(dev).eval()
        win_rows.append({"window": wn, "params": nparams(pm),
                         "gflops_per_step": round(count_flops(
                             lambda: pm(st[:, :wn], ac[:, :wn])) / 1e9, 4)})
        del pm

    p_pred = nparams(pred)
    rep["B_rollout_structure"] = {
        "predictor_params": p_pred,
        "gflops_per_step_deployed": round(f_full / 1e9, 4),
        "kv_cache_ceiling": {
            "gflops_per_step_incremental": round(f_incr / 1e9, 4),
            "flop_saving_x": round(f_full / max(f_incr, 1), 2),
            "weight_bytes_saved_fp16_mb": 0.0,
            "BLOCKER": "OperativePredictor.forward adds an ABSOLUTE window "
                       "position embedding (`self.pos[:, :w]`, predictor.py:"
                       "in_proj line) BEFORE the causal blocks. The deploy "
                       "rollout SLIDES the window every step, so every "
                       "retained state's positional index shifts by one and "
                       "its K/V change -- a KV cache is INVALID without a "
                       "relative/rotary rewrite, i.e. a retrain.",
            "why_it_would_not_help_anyway":
                "it removes FLOPs, and the rollout is bandwidth-bound (see E); "
                "weight traffic is unchanged, so the binding term does not move.",
        },
        "depth_scan": depth_rows,
        "width_scan": width_rows,
        "window_scan": win_rows,
        "strided_head_lever": {
            "note": "the k=2 / k=4 horizon heads are ALREADY TRAINED and in "
                    "the deployed ckpt; a strided roll reaches 2 s in 10 / 5 "
                    "predictor calls instead of 20 -- a pure BYTES lever "
                    "needing no retrain. Latency MEASURED "
                    "(eff_levers_flagship-30k.json strided_head_latency); "
                    "accuracy explicitly UNMEASURED there.",
            "predictor_calls": {"k1": 20, "k2": 10, "k4": 5},
            "weight_traffic_fp16_mb": {
                "k1": round(p_pred * 2 * 20 / 1e6, 1),
                "k2": round(p_pred * 2 * 10 / 1e6, 1),
                "k4": round(p_pred * 2 * 5 / 1e6, 1)},
        },
    }

    # ---------- C. fan batching --------------------------------------------
    print("[C] fan batching")
    fan = []
    for K in (1, 2, 4, 8, 16, 32):
        stK = torch.randn(K, WINDOW, S, device=dev)
        acK = torch.randn(K, WINDOW, pc.action_dim, device=dev)
        f = count_flops(lambda: pred(stK, acK))
        wb = p_pred * 2                       # weights read ONCE for the batch
        act = K * WINDOW * pc.d_model * 2 * 2  # in+out activations, fp16
        fan.append({"K": K, "gflops": round(f / 1e9, 4),
                    "weight_bytes_fp16_mb": round(wb / 1e6, 2),
                    "intensity_flop_per_byte": round(f / (wb + act), 2),
                    "gflops_per_candidate": round(f / 1e9 / K, 4)})
        del stK, acK
    rep["C_fan_batching"] = {
        "note": "arithmetic intensity rises ~linearly in K because the SAME "
                "182.7 MB of fp16 predictor weights serves every candidate. "
                "This is the mechanism behind the already-MEASURED 'marginal "
                "candidate ~= 0.3 ms' (eff_levers, K=8 fan 20.82 ms p50): on a "
                "bandwidth-bound rollout the fan is nearly FREE capacity we "
                "are already paying for.",
        "rows": fan}
    del pred, pred1

    # ---------- D. quantization format numerics ----------------------------
    print("[D] format numerics")
    torch.manual_seed(0)
    dists = {
        "gaussian_d768x768": torch.randn(768, 768),
        "gaussian_d3072x768_mlp": torch.randn(3072, 768),
        "heavytail_t3_d768x768": torch.distributions.StudentT(3.0).sample(
            (768, 768)),
        "outlier_1pct_d768x768": (lambda t: (t.mul_(
            torch.where(torch.rand_like(t) < 0.01, 8.0, 1.0)), t)[1])(
            torch.randn(768, 768)),
    }
    fmts = {
        "int8_per_channel  (MEASURED-PASS reference)":
            lambda w: fq_int_per_channel(w, 8),
        "fp8_e4m3_per_channel": fq_fp8_per_channel,
        "nvfp4_block16_e4m3scale": lambda w: fq_block_fp4(w, 16, "e4m3"),
        "mxfp4_block32_e8m0scale": lambda w: fq_block_fp4(w, 32, "e8m0"),
        "int4_per_channel  (the DINO-WM collapse format)":
            lambda w: fq_int_per_channel(w, 4),
    }
    dres = {}
    for dn, w in dists.items():
        dres[dn] = {fn: fmt_error(w, f(w)) for fn, f in fmts.items()}
        print("   ", dn)
        for fn, r in dres[dn].items():
            print("      %-46s cos=%.7f snr=%6.2f dB"
                  % (fn, r["cos_mean"], r["snr_db"]))
    rep["D_format_numerics"] = {
        "scope": "FORMAT-INTRINSIC round-trip error on synthetic weight "
                 "distributions. This establishes the ORDERING of the formats, "
                 "NOT an accuracy claim for our checkpoint -- the trained "
                 "weight distribution is not measured here (no ckpt on this "
                 "host). The real-weight run is the pre-registered experiment.",
        "anchor": "int8_per_channel is the SAME scheme as bench_p1_accuracy.py "
                  "_fakequant_(bits=8, dim=0), which MEASURED blanket "
                  "weight-only cos 0.99947 (encoder) / 0.9999997 (predictor) "
                  "and a downstream +0.0065 m ADE@2s -- a PASS against the "
                  "0.02 m falsifier. Any format at or above that cosine on the "
                  "same distribution is a candidate; any format below it is not.",
        "results": dres}

    # ---------- E. roofline -------------------------------------------------
    print("[E] roofline")
    tb = budget["tick_budget"]
    gf_enc = tb["gflops"]["encoder_1frame"]
    gf_roll = tb["gflops"]["rollout_decode_k20_measured"]
    gf_tick = tb["gflops"]["deployed_tick_total"]
    traffic = tb["weight_traffic_mb_per_tick"]

    roof = {}
    for dname, d in DEVICES.items():
        rows = {}
        for prec in ("fp32", "fp16", "int8", "fp8", "nvfp4"):
            tflops = {"fp32": None,
                      "fp16": d.get("gpu_fp16_dense_tflops"),
                      "int8": d.get("gpu_int8_dense_tops"),
                      "fp8": d.get("gpu_fp8_dense_tflops"),
                      "nvfp4": d.get("gpu_fp4_dense_tflops")}[prec]
            if tflops is None:
                rows[prec] = {"supported": False,
                              "why": "no tensor-core datapath for this "
                                     "precision on this architecture"}
                continue
            bw = d["bw_gb_s"]
            t_mb = traffic[prec]["total_tick_mb"]
            enc_mb = traffic[prec]["encoder_once_mb"]
            roll_mb = t_mb - enc_mb
            t_compute_ms = gf_tick / tflops                    # GFLOP / TFLOPS
            t_bw_ms = t_mb / bw                                # MB / (GB/s)
            rows[prec] = {
                "supported": True,
                "peak_dense_tflops_or_tops": tflops,
                "bw_gb_s": bw,
                "machine_balance_flop_per_byte": round(tflops * 1e12
                                                       / (bw * 1e9), 1),
                "tick_compute_floor_ms": round(t_compute_ms, 3),
                "tick_bandwidth_floor_ms": round(t_bw_ms, 3),
                "binding_term": ("bandwidth" if t_bw_ms > t_compute_ms
                                 else "compute"),
                "bound_by_x": round(max(t_bw_ms, t_compute_ms)
                                    / max(min(t_bw_ms, t_compute_ms), 1e-9), 1),
                "tick_floor_ms": round(max(t_bw_ms, t_compute_ms), 3),
                "max_hz_at_100pct_of_peak": round(
                    1000.0 / max(t_bw_ms, t_compute_ms), 1),
                "hz_at_60pct_dram_efficiency": round(
                    1000.0 / (max(t_bw_ms, t_compute_ms) / 0.6), 1),
                "component_floors_ms": {
                    "encoder_compute": round(gf_enc / tflops, 3),
                    "encoder_bandwidth": round(enc_mb / bw, 3),
                    "rollout_compute": round(gf_roll / tflops, 3),
                    "rollout_bandwidth": round(roll_mb / bw, 3)},
            }
        # weight-resident capacity
        rows["capacity"] = {
            "device_mem_gb": d["mem_gb"],
            "deployed_path_params": budget["params"]["deployed_path_params"],
            "full_model_params": budget["params"]["model_total"],
            "deployed_path_mb": {p: round(
                budget["params"]["deployed_path_params"] * b / 1e6, 1)
                for p, b in PRECISION_BYTES.items()},
            "full_model_mb": {p: round(
                budget["params"]["model_total"] * b / 1e6, 1)
                for p, b in PRECISION_BYTES.items()},
            "peak_activation_mb_deployed_tick": round(
                budget["stages"]["encode_1frame_to_state"]["memory"]
                ["activation_mb"], 2),
            "verdict": "CAPACITY IS A NON-ISSUE on both devices by 2 orders of "
                       "magnitude; the constraint is BANDWIDTH, not capacity.",
        }
        roof[dname] = rows
    rep["E_roofline"] = {
        "inputs_measured": {"tick_gflops": gf_tick,
                            "encoder_gflops": gf_enc,
                            "rollout_gflops": gf_roll,
                            "weight_traffic_mb": traffic},
        "method": "floor = max(FLOPs/peak_dense, weight_bytes/peak_bw). Both "
                  "are UNREACHABLE lower bounds at 100% of peak; the ratio "
                  "between them (binding term) is the decision-grade output, "
                  "not the absolute ms. ESTIMATED -- no Orin/Thor silicon.",
        "by_device": roof}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    for dn, rows in roof.items():
        print("==", dn)
        for prec, r in rows.items():
            if prec == "capacity" or not r.get("supported"):
                continue
            print("   %-6s floor=%7.2f ms  bound=%s (%.1fx)  balance=%s F/B"
                  % (prec, r["tick_floor_ms"], r["binding_term"],
                     r["bound_by_x"], r["machine_balance_flop_per_byte"]))


if __name__ == "__main__":
    main()
