"""INT8 weight-only quantization accuracy curve (per module) + clean-GPU
fp16/fp32 latency, on the RTX 4060.

Production & Optimization backlog **P1.6** (INT8/FP8 quantization curves) plus
the **P1.4b** clean-GPU latency closure. Measured optimization experiment
(D-020 G-H / G-P2: accuracy delta ALWAYS next to the speed/efficiency delta).

Why weight-only fake-quant and not a real TensorRT/ModelOpt INT8 engine: the TRT
toolchain is NOT installed on this dev box (`import tensorrt` -> ModuleNotFound;
onnxruntime exposes CPU/Azure EP only; `modelopt` absent). But the FIRST question
a ModelOpt INT8 PTQ answers is answerable today and is the honest precursor:
**does rounding a module's weights to per-output-channel symmetric int8 move the
driving decision?** If yes for a module, no engine will save it; if no, that
module is an INT8 candidate. This isolates the *weight-quant* accuracy term
(activations kept fp32) so it composes with the separately-measured fp16
activation term (`half_precision_step6500.json`) to bound a full INT8 engine:
    INT8-engine error  ~=  fp16-activation error  (+)  int8-weight error.
INT8 *latency* needs the fused int8 kernels (TRT) and is therefore NOT claimed
here; the measurable INT8 efficiency delta today is the **weight-memory footprint
reduction** (bytes), reported next to the accuracy delta (G-P2).

Accuracy is scored in the DECISION space, per the 2026-07-09 fp16/bf16 precedent
(cosine ~1.0 is too coarse): one FIXED fp64 ridge probe decodes a K=9
sustained-steer imagine-and-select fan; we report selection-agreement %, decoded-
waypoint shift in METRES, and encoder/predictor rel-err vs the fp32 reference,
on the SAME 64 real comma2k19 windows as the half-precision run (comparable).

Module groups (the KB thesis under test: "keep the ViT tower FP16, quantize
predictor/heads first"):
  int8_predictor : operative + tactical predictors only
  int8_heads     : everything EXCEPT the ViT encoder (readout, predictors,
                   inv-dyn, imagination) — the non-vision compute
  int8_encoder   : the ViT tower only (expected most sensitive)
  int8_all       : the whole model

Usage (local 4060, GPU idle):
  PYTHONUTF8=1 python int8_quant_curve.py \
    --ckpt C:/Users/Admin/tanitad-data/eval/ckpt_full.pt \
    --comma-cache C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f \
    --out .../int8_quant_step6500.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# --- reuse the half-precision harness (identical methodology / windows) ------ #
_HP = (Path(__file__).resolve().parents[1]
       / "half_precision" / "half_precision_latency_accuracy.py")
_spec = importlib.util.spec_from_file_location("hp", _HP)
hp = importlib.util.module_from_spec(_spec)
sys.modules["hp"] = hp
_spec.loader.exec_module(hp)

from tanitad.config import base250cam_config          # noqa: E402
from tanitad.data.mixing import load_episode           # noqa: E402
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.readout import RidgeProbe          # noqa: E402


# --------------------------------------------------------------------------- #
#  int8 weight-only fake-quant (per-output-channel symmetric)
# --------------------------------------------------------------------------- #
def _quantize_weight_(w: torch.Tensor) -> int:
    """In-place per-output-channel symmetric int8 fake-quant of a weight tensor.

    Row 0 (output channel) keeps its own scale = max|w_row| / 127. Returns the
    number of weight ELEMENTS quantized (for the byte-footprint accounting).
    Mirrors what NVIDIA ModelOpt INT8 weight PTQ does before TRT fuses the QDQ.
    """
    out_ch = w.shape[0]
    flat = w.reshape(out_ch, -1)
    scale = flat.abs().amax(dim=1, keepdim=True) / 127.0
    scale = scale.clamp_min(1e-12)
    q = torch.clamp(torch.round(flat / scale), -127, 127)
    flat.copy_((q * scale).to(flat.dtype))
    return int(w.numel())


def quantize_module_(module: nn.Module) -> int:
    """Quantize every Linear/Conv weight in `module` (in place). Returns #elems."""
    n = 0
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            with torch.no_grad():
                n += _quantize_weight_(m.weight)
    return n


def _linear_conv_bytes(module: nn.Module) -> int:
    """fp32 weight bytes of the Linear/Conv leaves under `module`."""
    b = 0
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            b += m.weight.numel() * 4
    return b


def _target_modules(world) -> dict:
    """Named module subtrees for the per-group INT8 sweep."""
    groups = {
        "int8_predictor": [world.predictor]
        + ([world.tactical_pred] if world.tactical_pred is not None else []),
        "int8_encoder": [world.encoder],
        "int8_all": [world],
    }
    # "everything except the ViT encoder": readout + predictors + inv_dyn +
    # imagination (the non-vision compute path).
    heads = [world.readout, world.predictor, world.inv_dyn]
    if world.tactical_pred is not None:
        heads.append(world.tactical_pred)
    if getattr(world, "imagination", None) is not None:
        heads.append(world.imagination)
    groups["int8_heads"] = heads
    return groups


# --------------------------------------------------------------------------- #
#  accuracy of a (possibly quantized) world model vs the fp32 reference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decision_accuracy(world, cfg, show_meta, probe, device, W,
                      ref_state, ref_pred, ref_fan, ref_pick, subgoal) -> dict:
    """Encoder/predictor rel-err + imagine-and-select decision metric, fp32
    compute (weights already quantized in `world`)."""
    enc_p, pred_p = [], {hh: [] for hh in cfg.predictor.horizons}
    for ep, t in show_meta:
        fw = hp._frames(ep, t, W).unsqueeze(0).to(device)
        aw = ep.actions[t:t + W].unsqueeze(0).float().to(device)
        st = world.encode_window(fw)
        enc_p.append(st[0, -1].float().cpu())
        pr = world.imagine(st, aw)
        for hh in cfg.predictor.horizons:
            pred_p[hh].append(pr[hh][0].float().cpu())
    enc_p = torch.stack(enc_p)
    pred_p = {hh: torch.stack(v) for hh, v in pred_p.items()}
    fan_p = np.stack([hp.fan_decode(world, ep, t, W, probe, device, torch.float32)
                      for ep, t in show_meta])
    pick_p = np.argmin(np.linalg.norm(fan_p - subgoal[:, None, :], axis=-1), axis=1)
    agree = float((pick_p == ref_pick).mean())
    wp_shift = np.linalg.norm(fan_p - ref_fan, axis=-1)
    return {
        "encoder_state": hp._delta(ref_state, enc_p),
        "predictor": {f"h{hh}": hp._delta(ref_pred[hh], pred_p[hh])
                      for hh in cfg.predictor.horizons},
        "decision": {
            "selection_agreement": round(agree, 4),
            "n_flips": int((pick_p != ref_pick).sum()),
            "decoded_wp_shift_m_mean": round(float(wp_shift.mean()), 4),
            "decoded_wp_shift_m_max": round(float(wp_shift.max()), 4),
            "fan_finite": bool(np.isfinite(fan_p).all()),
        },
    }


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

    # fp32 reference model + real windows (same pipeline as half_precision) ----
    world32, step = hp.load_world(cfg, args.ckpt, torch.float32, device)
    params_b = sum(p.numel() for p in world32.parameters()) / 1e9
    eps = [load_episode(str(p), mmap=True)
           for p in sorted(Path(args.comma_cache).glob("ep_*.pt"))[:args.max_eps]]
    rows, h = hp.collect_windows(world32, eps, device, W)
    n_all = rows["state"].shape[0]

    perm = torch.randperm(n_all, generator=torch.Generator().manual_seed(0))
    half = n_all // 2
    fit_i, show_i = perm[:half], perm[half:half + args.max_windows]
    probe = RidgeProbe(alpha=1e-3).fit(rows["z_imag_h"][fit_i], rows["disp_h"][fit_i])
    probe_r2 = probe.r2(rows["z_imag_h"][show_i], rows["disp_h"][show_i])

    show_meta = [rows["meta"][i] for i in show_i.tolist()]
    with strict_numerics(), torch.no_grad():
        ref_state, ref_pred = [], {hh: [] for hh in cfg.predictor.horizons}
        for ep, t in show_meta:
            fw = hp._frames(ep, t, W).unsqueeze(0).to(device)
            aw = ep.actions[t:t + W].unsqueeze(0).float().to(device)
            st = world32.encode_window(fw)
            ref_state.append(st[0, -1].cpu())
            pr = world32.imagine(st, aw)
            for hh in cfg.predictor.horizons:
                ref_pred[hh].append(pr[hh][0].cpu())
    ref_state = torch.stack(ref_state)
    ref_pred = {hh: torch.stack(v) for hh, v in ref_pred.items()}
    ref_fan = np.stack([hp.fan_decode(world32, ep, t, W, probe, device, torch.float32)
                        for ep, t in show_meta])
    subgoal = torch.stack([rows["disp_h"][i] for i in show_i]).numpy()
    ref_pick = np.argmin(np.linalg.norm(ref_fan - subgoal[:, None, :], axis=-1), axis=1)

    total_lc_bytes = _linear_conv_bytes(world32)

    report = {
        "exp": "int8-weight-only-quant-curve + clean-gpu-latency",
        "backlog": "Production&Optimization P1.6 (INT8 curves) + P1.4b (clean latency)",
        "ckpt_step": step, "params_billions": round(params_b, 4),
        "hardware": "RTX 4060 (Orin proxy, I8), batch 1, GPU idle (clean)",
        "n_showcase_windows": int(len(show_i)), "n_fit_windows": int(half),
        "tactical_horizon_steps": int(h), "probe_imag_r2": round(probe_r2, 4),
        "total_linear_conv_weight_bytes_fp32": total_lc_bytes,
        "method": [
            "INT8 = per-output-channel SYMMETRIC weight fake-quant (scale=max|w|/127), "
            "activations kept fp32 -> isolates the weight-quant accuracy term.",
            "Compose with the fp16 activation term (half_precision_step6500.json) for "
            "a full-engine bound: INT8-engine err ~= fp16-act err (+) int8-weight err.",
            "Decision metric = argmin_K||decoded_fan - subgoal|| over a K=9 steer fan, "
            "one FIXED fp64 probe across all arms -> flips attributable to weight-quant.",
            "INT8 LATENCY not claimed (needs TRT int8 kernels; toolchain absent). "
            "INT8 efficiency delta reported today = weight-memory footprint reduction.",
            "Latency arms (fp32/fp16) measured on an IDLE 4060 -> clean absolutes "
            "(closes P1.4b; the 2026-07-09 run was CarlaUE4-contended).",
        ],
        "latency_clean": {},
        "int8_groups": {},
    }

    # --- clean-GPU latency: fp32 + fp16 (P1.4b) ------------------------------- #
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        wl = world32 if name == "fp32" else hp.load_world(cfg, args.ckpt, dtype, device)[0]
        report["latency_clean"][name] = hp.measure_latency(wl, dtype, device)
        if name != "fp32":
            del wl
            torch.cuda.empty_cache()
    base_tick = report["latency_clean"]["fp32"]["decision_tick_p50_ms"]
    for name in ("fp32", "fp16"):
        lat = report["latency_clean"][name]
        lat["hz"] = round(1000.0 / lat["decision_tick_p50_ms"], 1)
        lat["speedup_vs_fp32"] = round(base_tick / lat["decision_tick_p50_ms"], 3)

    # --- INT8 per-module accuracy sweep -------------------------------------- #
    # Each arm: reload a clean fp32 model, quantize the target subtree, score.
    for gname in ("int8_predictor", "int8_heads", "int8_encoder", "int8_all"):
        wq, _ = hp.load_world(cfg, args.ckpt, torch.float32, device)
        groups = _target_modules(wq)
        q_bytes_fp32 = sum(_linear_conv_bytes(m) for m in groups[gname])
        n_elems = 0
        seen = set()
        for m in groups[gname]:
            # avoid double-counting overlapping subtrees (int8_all covers all)
            for sub in m.modules():
                if isinstance(sub, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    if id(sub) in seen:
                        continue
                    seen.add(id(sub))
                    with torch.no_grad():
                        n_elems += _quantize_weight_(sub.weight)
        acc = decision_accuracy(wq, cfg, show_meta, probe, device, W,
                                ref_state, ref_pred, ref_fan, ref_pick, subgoal)
        report["int8_groups"][gname] = {
            "quantized_weight_elems": n_elems,
            "quantized_weight_bytes_fp32": int(q_bytes_fp32),
            "weight_bytes_int8": int(q_bytes_fp32 // 4),
            "weight_mem_reduction_x": round(4.0, 2),
            "weight_mem_saved_MB": round(q_bytes_fp32 * 3 / 4 / 1e6, 1),
            "accuracy_vs_fp32": acc,
        }
        del wq
        torch.cuda.empty_cache()

    # --- tidy speed/efficiency-vs-accuracy summary --------------------------- #
    summary = {
        "fp32": {"decision_tick_ms": base_tick,
                 "hz": report["latency_clean"]["fp32"]["hz"],
                 "peak_vram_gb": report["latency_clean"]["fp32"]["peak_vram_gb"],
                 "selection_agreement": 1.0, "note": "reference"},
        "fp16": {"decision_tick_ms": report["latency_clean"]["fp16"]["decision_tick_p50_ms"],
                 "speedup_vs_fp32": report["latency_clean"]["fp16"]["speedup_vs_fp32"],
                 "hz": report["latency_clean"]["fp16"]["hz"],
                 "peak_vram_gb": report["latency_clean"]["fp16"]["peak_vram_gb"],
                 "note": "accuracy from half_precision_step6500.json (agreement 0.953)"},
    }
    for g, blk in report["int8_groups"].items():
        d = blk["accuracy_vs_fp32"]
        summary[g] = {
            "weight_mem_reduction_x": blk["weight_mem_reduction_x"],
            "weight_mem_saved_MB": blk["weight_mem_saved_MB"],
            "selection_agreement": d["decision"]["selection_agreement"],
            "n_flips": d["decision"]["n_flips"],
            "decoded_wp_shift_m_mean": d["decision"]["decoded_wp_shift_m_mean"],
            "encoder_rel_err_mean": d["encoder_state"]["rel_err_mean"],
            "encoder_finite": d["encoder_state"]["finite"],
        }
    report["SUMMARY_speed_efficiency_vs_accuracy"] = summary

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"SUMMARY": summary, "probe_imag_r2": probe_r2,
                      "n_showcase_windows": report["n_showcase_windows"],
                      "ckpt_step": step}, indent=2))


if __name__ == "__main__":
    main()
