#!/usr/bin/env python3
"""Per-component memory + compute budget of the DEPLOYED flagship-v1 tick.

Device-INDEPENDENT quantities only (params, FLOPs, activation bytes, weight
traffic).  NO latency is measured or emitted here on purpose: a desktop-GPU
millisecond is not an embedded millisecond, and the program has already been
burned by exactly that substitution (the retracted 14.331 ms RTX-4060 read).

What this answers that nothing in the program measured before:
  * FLOPs and DRAM weight-traffic broken down PER COMPONENT (encoder /
    predictor / step-readout / tactical+strategic planners), not just a
    401.9 GFLOP whole-tick total.
  * arithmetic intensity per component -> which side of the Orin/Thor roofline
    each component sits on.
  * activation memory at the deployed batch/window, per component.

Run:  python budget_probe.py --out artifacts/budget_report.json
Weights are random-init: params, FLOPs, activation bytes and traffic are all
weight-INDEPENDENT (architecture reads).  Verified by reproducing the registry
param count exactly (263,442,838).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.config import flagship4b_config                       # noqa: E402
from tanitad.eval.ckpt_compat import adapt_config_action_dim        # noqa: E402
from tanitad.models.fourbrain import WorldModel, run_hierarchy      # noqa: E402
from tanitad.models.metric_dynamics import (                        # noqa: E402
    HierarchicalGrounding, rollout_decode)

WINDOW = 8
K_ROLLOUT = 20          # 2 s @ 10 Hz -- the horizon the leaderboard ADE scores
BYTES = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "fp8": 1.0, "nvfp4": 0.5}


# --------------------------------------------------------------------------- #
def count_flops(fn, mods=None) -> dict:
    """FLOPs of one call.  Same convention as taniteval.efficiency._flops:
    FlopCounterMode counts conv/mm/addmm/bmm/sdpa and NOT elementwise/norm, and
    the MHA fast path is disabled during the count so attention is not silently
    dropped (understates a transformer by ~35% otherwise)."""
    from torch.utils.flop_counter import FlopCounterMode
    off = False
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        off = True
    except Exception:
        pass
    try:
        with torch.no_grad():
            fn()                                   # warm
        c = FlopCounterMode(mods=mods, display=False)
        with c, torch.no_grad():
            fn()
        counts = c.get_flop_counts()
        by_op = {str(k).split(".")[-1]: int(v)
                 for k, v in counts.get("Global", {}).items()}
        return {"total_flops": int(c.get_total_flops()),
                "gflops": round(c.get_total_flops() / 1e9, 4),
                "by_op_gflops": {k: round(v / 1e9, 4)
                                 for k, v in sorted(by_op.items(),
                                                    key=lambda kv: -kv[1])},
                "mha_fastpath_disabled_for_count": off}
    finally:
        if off:
            torch.backends.mha.set_fastpath_enabled(True)


def peak_activation_mb(fn, device) -> dict:
    """Peak allocator bytes ABOVE resident weights for one forward, batch 1.
    A graph property, not a device property (allocator granularity aside)."""
    if device.type != "cuda":
        return {"note": "cuda unavailable; activation not measured"}
    torch.cuda.synchronize(device)
    with torch.no_grad():
        fn()                                       # warm (algo choice, caches)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {"resident_mb": round(base / 1e6, 2),
            "peak_alloc_mb": round(peak / 1e6, 2),
            "activation_mb": round((peak - base) / 1e6, 3)}


def nparams(m) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/budget_report.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    cfg = flagship4b_config()
    cfg = adapt_config_action_dim(cfg, 3)          # deployed v1: speed channel
    torch.manual_seed(0)
    model = WorldModel(cfg).to(dev).eval()
    S = model.state_dim
    grounding = HierarchicalGrounding(S).to(dev).eval()
    step_readout = grounding.step["op"]

    rep: dict = {
        "what": "per-component memory + compute budget of the deployed "
                "flagship-v1 planning tick (device-independent quantities only)",
        "model": "flagship4b architecture, action_dim=3 (deployed v1 "
                 "`flagship4b-speedjerk-30k` config), RANDOM INIT "
                 "(params/FLOPs/activations are weight-independent)",
        "env": {"torch": torch.__version__,
                "device": str(dev),
                "device_name": (torch.cuda.get_device_name(0)
                                if dev.type == "cuda" else "cpu"),
                "note": "device is a MEASUREMENT HOST for graph-level "
                        "quantities only. NO latency is emitted from it."},
        "shapes": {"window": WINDOW, "k_rollout": K_ROLLOUT,
                   "state_dim": S, "batch": 1,
                   "frame": [1, cfg.encoder.in_channels, cfg.encoder.image_size,
                             cfg.encoder.image_size],
                   "n_tokens": model.encoder.n_tokens,
                   "d_model_enc": cfg.encoder.d_model,
                   "enc_depth": cfg.encoder.depth,
                   "d_model_pred": cfg.predictor.d_model,
                   "pred_depth": cfg.predictor.depth,
                   "horizons": list(cfg.predictor.horizons),
                   "action_dim": cfg.predictor.action_dim},
    }

    # ---------------- params ------------------------------------------------
    comp = {
        "encoder": model.encoder,
        "readout": model.readout,
        "predictor": model.predictor,
        "tactical_pred": model.tactical_pred,
        "tactical_policy": model.tactical_policy,
        "strategic_policy": model.strategic_policy,
        "imagination": model.imagination,
        "inv_dyn": model.inv_dyn,
    }
    p = {k: nparams(v) for k, v in comp.items() if v is not None}
    p_g = {f"grounding.{lvl}.{kind}": nparams(getattr(grounding, kind)[lvl])
           for lvl in HierarchicalGrounding.LEVELS for kind in ("invdyn", "step")}
    rep["params"] = {
        "by_component": p,
        "grounding_detail": p_g,
        "grounding_total": nparams(grounding),
        "model_total": nparams(model),
        "step_readout_op": nparams(step_readout),
        "registry_total_model": 263_442_838,
        "registry_trainable": 277_404_073,
        "reproduces_registry_total_model":
            nparams(model) == 263_442_838,
        "deployed_path_params": nparams(model.encoder) + nparams(model.readout)
                                + nparams(model.predictor)
                                + nparams(step_readout),
        "deployed_path_note":
            "the intent-free operative path that PRODUCES the scored ADE@2s: "
            "encoder + readout + operative predictor + grounding.step['op']. "
            "tactical/strategic/imagination/inv_dyn/tactical_pred are OFF this "
            "path (they are on the hierarchy path, sized separately below).",
    }

    # ---------------- inputs at deploy shapes -------------------------------
    fr1 = torch.randn(1, cfg.encoder.in_channels, cfg.encoder.image_size,
                      cfg.encoder.image_size, device=dev)
    frw = torch.randn(1, WINDOW, cfg.encoder.in_channels, cfg.encoder.image_size,
                      cfg.encoder.image_size, device=dev)
    st = torch.randn(1, WINDOW, S, device=dev)
    ac = torch.randn(1, WINDOW, cfg.predictor.action_dim, device=dev)
    fa = torch.randn(1, K_ROLLOUT, cfg.predictor.action_dim, device=dev)
    nav = torch.zeros(1, dtype=torch.long, device=dev)

    stages = {
        "encoder_1frame_tokens":  lambda: model.encoder(fr1),
        "encode_1frame_to_state": lambda: model.encode(fr1),
        "encode_window_8frames":  lambda: model.encode_window(frw),
        "predictor_1call":        lambda: model.predictor(st, ac),
        "step_readout_1call":     lambda: step_readout(st[:, -1], st[:, -1]),
        "rollout_decode_k20":     lambda: rollout_decode(
            model.predictor, st, ac, fa, step_readout, K_ROLLOUT),
        "tick_uncached_k20":      lambda: rollout_decode(
            model.predictor, model.encode_window(frw), ac, fa,
            step_readout, K_ROLLOUT),
        "tactical_pred_1call":    lambda: model.tactical_pred(st, ac),
        "hierarchy_str_plus_tac": lambda: run_hierarchy(model, st, ac, nav),
        "imagination_1call":      lambda: model.imagination(
            model.encoder(fr1), torch.zeros(1, model.encoder.n_tokens,
                                            dtype=torch.bool, device=dev)),
    }

    rep["stages"] = {}
    for name, fn in stages.items():
        entry: dict = {}
        try:
            entry["flops"] = count_flops(fn)
        except Exception as e:                                  # noqa: BLE001
            entry["flops"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        try:
            entry["memory"] = peak_activation_mb(fn, dev)
        except Exception as e:                                  # noqa: BLE001
            entry["memory"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        rep["stages"][name] = entry
        print(f"[stage] {name:26s} "
              f"gflops={entry['flops'].get('gflops', 'ERR')!s:>10} "
              f"act_mb={entry['memory'].get('activation_mb', 'ERR')}")

    # ---------------- the tick budget, composed -----------------------------
    def gf(n):
        return rep["stages"][n]["flops"].get("gflops", float("nan"))

    enc1 = gf("encode_1frame_to_state")
    enc8 = gf("encode_window_8frames")
    pred1 = gf("predictor_1call")
    sr1 = gf("step_readout_1call")
    roll = gf("rollout_decode_k20")
    hier = gf("hierarchy_str_plus_tac")

    # weight bytes STREAMED per tick.  At batch 1 the working set (>=175 MB
    # fp16) is >> Orin's 4 MB L2, so every call re-streams its weights from
    # DRAM.  Encoder is called ONCE per tick under the deployed L2 encoder
    # cache; the predictor is called K times; the step readout K times.
    pe = p["encoder"] + p["readout"]
    pp = p["predictor"]
    ps = rep["params"]["step_readout_op"]
    # the 2 horizon heads the rollout computes and DISCARDS (registry L7)
    head_params = cfg.predictor.d_model * S + S      # one Linear(d -> S)
    unused_heads = (len(cfg.predictor.horizons) - 1) * head_params

    traffic = {}
    for prec, nb in BYTES.items():
        traffic[prec] = {
            "encoder_once_mb": round(pe * nb / 1e6, 2),
            "predictor_x20_mb": round(pp * nb * K_ROLLOUT / 1e6, 2),
            "step_readout_x20_mb": round(ps * nb * K_ROLLOUT / 1e6, 2),
            "total_tick_mb": round((pe + (pp + ps) * K_ROLLOUT) * nb / 1e6, 2),
            "of_which_unused_horizon_heads_mb":
                round(unused_heads * nb * K_ROLLOUT / 1e6, 2),
        }

    rep["tick_budget"] = {
        "definition": "DEPLOYED (L4-composed) planning tick = encode ONE new "
                      "9-ch frame (7 window states cached) -> 20 sequential "
                      "predictor steps -> 20 step-readout decodes -> SE(2) "
                      "accumulate. This is the tick that produces ADE@2s.",
        "gflops": {
            "encoder_1frame": enc1,
            "predictor_x20": round(pred1 * K_ROLLOUT, 4),
            "step_readout_x20": round(sr1 * K_ROLLOUT, 4),
            "rollout_decode_k20_measured": roll,
            "deployed_tick_total": round(enc1 + roll, 4),
            "uncached_tick_total": round(enc8 + roll, 4),
            "registry_measured_uncached_tick": 401.922,
            "hierarchy_str_plus_tac_if_on": hier,
        },
        "share_of_deployed_tick_flops": {
            "encoder_pct": round(100 * enc1 / (enc1 + roll), 1),
            "rollout_pct": round(100 * roll / (enc1 + roll), 1),
        },
        "share_of_uncached_tick_flops": {
            "encoder_pct": round(100 * enc8 / (enc8 + roll), 1),
            "rollout_pct": round(100 * roll / (enc8 + roll), 1),
        },
        "weight_traffic_mb_per_tick": traffic,
        "share_of_deployed_tick_bytes_fp16": {
            "encoder_pct": round(100 * traffic["fp16"]["encoder_once_mb"]
                                 / traffic["fp16"]["total_tick_mb"], 1),
            "rollout_pct": round(100 * (traffic["fp16"]["predictor_x20_mb"]
                                        + traffic["fp16"]["step_readout_x20_mb"])
                                 / traffic["fp16"]["total_tick_mb"], 1),
        },
        "arithmetic_intensity_flop_per_byte_fp16": {
            "encoder_1frame": round(enc1 * 1e9
                                    / (traffic["fp16"]["encoder_once_mb"] * 1e6), 1),
            "predictor_1step": round(pred1 * 1e9 / (pp * 2), 2),
            "rollout_k20": round(roll * 1e9
                                 / ((traffic["fp16"]["predictor_x20_mb"]
                                     + traffic["fp16"]["step_readout_x20_mb"])
                                    * 1e6), 2),
            "deployed_tick": round((enc1 + roll) * 1e9
                                   / (traffic["fp16"]["total_tick_mb"] * 1e6), 2),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nwrote", args.out)
    print("params reproduce registry 263,442,838:",
          rep["params"]["reproduces_registry_total_model"],
          "(got", rep["params"]["model_total"], ")")
    print(json.dumps(rep["tick_budget"]["gflops"], indent=1))
    print(json.dumps(rep["tick_budget"]["share_of_deployed_tick_flops"], indent=1))
    print(json.dumps(rep["tick_budget"]["share_of_deployed_tick_bytes_fp16"], indent=1))
    print(json.dumps(rep["tick_budget"]
                     ["arithmetic_intensity_flop_per_byte_fp16"], indent=1))


if __name__ == "__main__":
    main()
