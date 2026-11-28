#!/usr/bin/env python3
"""Phase 3 (CPU-only, no GPU needed): synthesize the mixed-precision
recommendation + Orin/Thor ESTIMATED scaling from the MEASURED phase-1
(PyTorch accuracy) + phase-2 (real TensorRT latency) results already banked in
orin_int8_benchmark.json. Appends two sections; does not overwrite anything.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path("/workspace/int8_bench/orin_int8_benchmark.json")
SAFE_COS = 0.999          # mission's own stated bar ("encoder-output cosine >= ~0.999")

# ---- PUBLISHED dense tensor-core rates, all already cited+derived in this repo ----
# Source: TanitAD Research Hub/.../2026-07-20-orin-thor-deployment-and-inference-levers.md
#   SS2.1 (A40 vendor datasheet), SS2.1/SS1.3 (Orin derivation from NVIDIA's 170 sparse
#   INT8 TOPS), SS2.3 (Thor ratios / RidgeRun FP8 / NVIDIA FP4-sparse).
TOPS = {
    "A6000_or_A40_fp16_dense_tflops": 149.7,      # [P] A40 datasheet; A6000 = same GA102, 84 SM, 336 3rd-gen TC
    "A6000_or_A40_int8_dense_tops": 299.3,        # [P] A40 datasheet
    "orin_fp16_dense_tflops": 42.5,               # [E] 170 sparse INT8 TOPS (GPU only, excl. DLA) / 2 (dense) / 2 (fp16 vs int8)
    "orin_int8_dense_tops": 85.0,                 # [E] 170 sparse INT8 TOPS (GPU only, excl. DLA) / 2 (dense)
    "thor_fp16_dense_tflops": 255.0,              # [E] desk note SS2.3 "~6x Orin at matched fp16" -> 6 * 42.5
    "thor_fp8_dense_tflops": 518.0,                # [E] RidgeRun 1035 (likely sparse) / 2
    "thor_nvfp4_dense_tflops": 1035.0,             # [E] NVIDIA 2070 FP4-sparse / 2
}

# rollout DRAM-bandwidth floors already established (predictor is bandwidth-, not
# compute-bound -- TOPS-ratio scaling below is a rough compute-side upper bound;
# these are the more defensible cross-check for the ROLLOUT specifically.
BANDWIDTH_FLOOR_MS_PER_TICK = {
    # {device: {precision: ms}}  -- 20-step rollout only, from the desk note SS4.1
    "orin": {"fp16": 17.9, "int8": 9.4},
    "thor": {"fp16": 13.4, "int8_or_fp8": 7.0, "nvfp4": 3.5},
}


def log(msg):
    print(f"[bench-p3] {msg}", flush=True)


def safe_get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def mixed_precision_recommendation(rep):
    out = {"threshold_cosine": SAFE_COS, "components": {}}
    for comp, sweepkey in (("encoder", "encoder_accuracy_sweep"), ("predictor", "predictor_accuracy_sweep")):
        sweep = rep.get(sweepkey, {})
        iso = sweep.get("isolated_per_block", {})
        blanket = sweep.get("blanket", {})
        rows = []
        n_unsafe_wo, n_unsafe_wa = 0, 0
        for name, modes in iso.items():
            cos_wo = safe_get(modes, "int8_wo", "cosine")
            cos_wa = safe_get(modes, "int8_wa", "cosine")
            safe_wo = (cos_wo is not None) and (cos_wo >= SAFE_COS)
            safe_wa = (cos_wa is not None) and (cos_wa >= SAFE_COS)
            n_unsafe_wo += (0 if safe_wo else 1)
            n_unsafe_wa += (0 if safe_wa else 1)
            rows.append({
                "block": name, "isolated_int8_weight_only_cosine": cos_wo,
                "isolated_int8_weight_only_safe": safe_wo,
                "isolated_int8_weight_plus_activation_cosine": cos_wa,
                "isolated_int8_weight_plus_activation_safe": safe_wa,
                "recommendation": ("INT8 (weight-only) safe; keep activations FP16 unless W+A also safe"
                                   if safe_wo and not safe_wa else
                                   "INT8 (weight-only AND weight+activation) safe"
                                   if safe_wo and safe_wa else
                                   "MUST STAY FP16 (fails weight-only INT8 in isolation)"),
            })
        rows.sort(key=lambda r: (r["isolated_int8_weight_plus_activation_cosine"] or 0.0))
        out["components"][comp] = {
            "per_block": rows,
            "blanket_int8_weight_only": blanket.get("int8_wo"),
            "blanket_int8_weight_plus_activation": blanket.get("int8_wa"),
            "blanket_fp16": blanket.get("fp16"),
            "n_blocks_unsafe_weight_only": n_unsafe_wo,
            "n_blocks_unsafe_weight_plus_activation": n_unsafe_wa,
            "verdict": (
                "BLANKET weight-only INT8 is SAFE across the whole component "
                f"(cos={safe_get(blanket,'int8_wo','cosine')}) -- ship it."
                if safe_get(blanket, "int8_wo", "cosine", default=0) >= SAFE_COS
                else "Blanket weight-only INT8 degrades this component; use the per-block map."
            ) if blanket.get("int8_wo") else "insufficient data",
        }
    return out


def orin_thor_estimates(rep):
    trt = rep.get("trt_engines", {})
    enc_fp16 = safe_get(trt, "encoder", "fp16", "latency", "p50_ms")
    enc_int8 = safe_get(trt, "encoder", "int8", "latency", "p50_ms")
    pred_fp16 = safe_get(trt, "predictor", "fp16", "latency", "p50_ms")
    pred_int8 = safe_get(trt, "predictor", "int8", "latency", "p50_ms")

    def scale(measured_ms, tops_a6000, tops_target):
        if measured_ms is None or not tops_target:
            return None
        return round(measured_ms * (tops_a6000 / tops_target), 4)

    def component_estimates(measured_fp16, measured_int8):
        est = {
            "orin_fp16_ms": scale(measured_fp16, TOPS["A6000_or_A40_fp16_dense_tflops"], TOPS["orin_fp16_dense_tflops"]),
            "orin_int8_ms": scale(measured_int8, TOPS["A6000_or_A40_int8_dense_tops"], TOPS["orin_int8_dense_tops"]),
            "thor_fp16_ms": scale(measured_fp16, TOPS["A6000_or_A40_fp16_dense_tflops"], TOPS["thor_fp16_dense_tflops"]),
            "thor_fp8_ms": scale(measured_fp16, TOPS["A6000_or_A40_fp16_dense_tflops"], TOPS["thor_fp8_dense_tflops"]),
            "thor_nvfp4_ms": scale(measured_fp16, TOPS["A6000_or_A40_fp16_dense_tflops"], TOPS["thor_nvfp4_dense_tflops"]),
        }
        return est

    enc_est = component_estimates(enc_fp16, enc_int8)
    pred_est = component_estimates(pred_fp16, pred_int8)

    # naive (no CUDA graph) vs graph-projected full tick: encoder(1 cached call) + 20 x predictor
    GRAPH_MULT = 3.46   # MEASURED structural rollout speedup (this repo, A40/A6000-class; a STRUCTURE not a
                        # silicon-specific number -- carries as an assumption, not a hardware fact, to Jetson
    def tick(enc_ms, pred_ms):
        if enc_ms is None or pred_ms is None:
            return None
        naive = enc_ms + 20 * pred_ms
        graphed = enc_ms + (20 * pred_ms) / GRAPH_MULT
        return {"naive_ms": round(naive, 3), "graph_projected_ms": round(graphed, 3),
                "meets_10hz_naive": naive <= 100.0, "meets_10hz_graph_projected": graphed <= 100.0}

    ticks = {
        "orin_fp16": tick(enc_est["orin_fp16_ms"], pred_est["orin_fp16_ms"]),
        "orin_int8": tick(enc_est["orin_int8_ms"], pred_est["orin_int8_ms"]),
        "thor_fp16": tick(enc_est["thor_fp16_ms"], pred_est["thor_fp16_ms"]),
        "thor_fp8": tick(enc_est["thor_fp8_ms"], pred_est["thor_fp8_ms"]),
        "thor_nvfp4": tick(enc_est["thor_nvfp4_ms"], pred_est["thor_nvfp4_ms"]),
    }

    return {
        "evidence_class": "ESTIMATED -- scaled from this run's MEASURED A6000 (SM 8.6) TensorRT "
                          "latencies by PUBLISHED dense tensor-core TOPS/TFLOPS ratios. "
                          "'latency scales inversely with TOPS' is a rough COMPUTE-BOUND upper bound; "
                          "it is most defensible for the ENCODER (94% of tick FLOPs, desk note SS1.2) "
                          "and least defensible for the PREDICTOR/rollout, which the same desk note "
                          "establishes is DRAM-BANDWIDTH-bound, not compute-bound -- see "
                          "bandwidth_floor_cross_check for the more rigorous number on that component.",
        "tops_sources": TOPS,
        "measured_a6000_this_run": {"encoder_fp16_ms": enc_fp16, "encoder_int8_ms": enc_int8,
                                    "predictor_fp16_ms": pred_fp16, "predictor_int8_ms": pred_int8},
        "encoder_estimates_ms": enc_est,
        "predictor_estimates_ms": pred_est,
        "bandwidth_floor_cross_check_ms_per_20step_rollout": BANDWIDTH_FLOOR_MS_PER_TICK,
        "full_tick_estimate": ticks,
        "tick_definition": ("encoder(1 cached frame) + 20 x predictor-call; 'graph_projected' applies "
                            f"the {GRAPH_MULT}x rollout speedup MEASURED from CUDA-graph capture on "
                            "A40/A6000-class silicon in this repo -- a STRUCTURAL assumption carried to "
                            "Jetson, not a Jetson measurement (CUDA-graph-on-TRT was not built for Orin/"
                            "Thor -- hardware-blocked, no chip on hand)."),
        "budget_ms": 100.0,
    }


def main():
    rep = json.loads(OUT.read_text())
    mp = mixed_precision_recommendation(rep)
    ot = orin_thor_estimates(rep)
    rep["mixed_precision_recommendation"] = mp
    rep["orin_thor_estimates"] = ot
    OUT.write_text(json.dumps(rep, indent=2, default=str))
    log("wrote mixed_precision_recommendation + orin_thor_estimates")
    log(json.dumps(ot, indent=2)[:3000])
    for comp, data in mp["components"].items():
        log(f"{comp} verdict: {data['verdict']}")
        log(f"{comp} unsafe blocks (weight-only): {data['n_blocks_unsafe_weight_only']}, "
            f"(weight+activation): {data['n_blocks_unsafe_weight_plus_activation']}")


if __name__ == "__main__":
    main()
