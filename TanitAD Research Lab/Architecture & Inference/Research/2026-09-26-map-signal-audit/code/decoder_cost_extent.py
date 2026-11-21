"""ANALYTIC memory / FLOP estimate of the NEW-2 10 cm map branch at b16, for the /2 extent (60 m x +-16 m) and for the
census-selected extent (100 m x +-30 m). Pure arithmetic on the builder's shapes -- no tensor, no data.

Shapes: `…/2026-09-26-refcv7-map-hires/BUILD.md` §4 and `raw/cost_analytic_cpu.json` (MapHiresConfig defaults):
  lift  BEVLiftProjectFirst: 1x1 conv 512 -> Z*D (Z = 4 heights, D = d_lift 64) on the stride-8 map [52, 128], then
        grid_sample to [Z, D, X, Y] at 0.25 m, masked, summed over Z -> [64, X, Y]
  enc   3x3 stem 64 -> 64 + GN + GELU; 5 _DilatedBlocks (conv-GN-GELU-conv-GN-add-GELU) at 64 ch; 1x1 64 -> 32 + GN
  refine bilinear x2.5 -> [32, x, y] at 0.1 m; two (3x3 32 -> 32 + GN + GELU); 1x1 32 -> 8 logits
Saved-for-backward accounting (fp32): each conv saves its input; each GN its input; each GELU its input; the residual
add nothing; log_softmax its output. This is the textbook autograd accounting -- the builder's MEASURED CPU accounting
(460.0 MB / sample at 60 x 32 m) is the primary number and is scaled by grid area; this file's own count is its check.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "raw" / "decoder_cost_extent.json"
B, FP32, GB = 16, 4, 2 ** 30
Z, D, DM, DU, C = 4, 64, 64, 32, 8
S8 = (52, 128)
BUILDER_SAVED_B1_60x32 = 459969412          # raw/cost_analytic_cpu.json saved_bytes_b1 (MEASURED by the builder)
BUILDER_SAVED_B1_60x32_CKPT = 24481796      # ... saved_bytes_b1_grad_ckpt


def shapes(x_max, y_half):
    X, Y = int(round(x_max / 0.25)), int(round(2 * y_half / 0.25))
    x, y = int(round(x_max / 0.1)), int(round(2 * y_half / 0.1))
    return X, Y, x, y


def estimate(x_max, y_half):
    X, Y, x, y = shapes(x_max, y_half)
    XY, xy, s8 = X * Y, x * y, S8[0] * S8[1]
    saved = {
        "lift (proj out + sampled Z*D + out)": Z * D * s8 + Z * D * XY + D * XY,
        "encoder @0.25 m": (3 + 5 * 6 + 1) * DM * XY + DU * XY,
        "refine @0.1 m": 7 * DU * xy,
        "logits + log_softmax": 2 * C * xy,
    }
    tot = sum(saved.values())
    flops = {
        "lift 1x1 on stride-8 map": 2 * 512 * Z * D * s8,
        "encoder convs": XY * (11 * DM * DM * 9 * 2 + DM * DU * 2),
        "refine convs": xy * (2 * DU * DU * 9 * 2 + DU * C * 2),
    }
    fwd = sum(flops.values())
    return {"lift_grid_0.25m": [X, Y], "logit_grid_0.1m": [x, y],
            "saved_elems_per_sample": saved, "saved_GB_b16_this_count": round(tot * FP32 * B / GB, 2),
            "logits_GB_b16": round(C * xy * FP32 * B / GB, 3),
            "logit_grad_GB_b16": round(C * xy * FP32 * B / GB, 3),
            "largest_transient_activation_grad_GB_b16": round(max(DM * XY, DU * xy) * FP32 * B / GB, 3),
            "fwd_GFLOP_per_sample": round(fwd / 1e9, 1), "fwd_bwd_TFLOP_per_step_b16": round(3 * fwd * B / 1e12, 2)}


def main():
    base, new = estimate(60.0, 16.0), estimate(100.0, 30.0)
    area = (100.0 * 60.0) / (60.0 * 32.0)
    trunk_fwd_tflop = 16 * 10 * 7.8 * (416 * 1024) / (224 * 224) / 1e3      # 160 frame passes, resnet101
    rec = {"batch": B, "area_ratio_new_over_old": round(area, 4),
           "extent_60x32": base, "extent_100x60": new,
           "builder_measured_saved_GB_b16": {
               "60x32": round(BUILDER_SAVED_B1_60x32 * B / GB, 2),
               "100x60_scaled_by_area": round(BUILDER_SAVED_B1_60x32 * area * B / GB, 2),
               "60x32_grad_ckpt": round(BUILDER_SAVED_B1_60x32_CKPT * B / GB, 2),
               "100x60_grad_ckpt_scaled": round(BUILDER_SAVED_B1_60x32_CKPT * area * B / GB, 2)},
           "trunk_fwd_bwd_TFLOP_per_step_approx": round(3 * trunk_fwd_tflop, 1),
           "map_branch_share_of_trunk_flops": {
               "60x32": round(base["fwd_bwd_TFLOP_per_step_b16"] / (3 * trunk_fwd_tflop), 3),
               "100x60": round(new["fwd_bwd_TFLOP_per_step_b16"] / (3 * trunk_fwd_tflop), 3)},
           "reference": {"refcv6_peak_cuda_GB": 21.63, "refcv6_s_per_step": 6.63,
                         "builder_estimate_refcv7_peak_60x32_GB": "30-33"}}
    json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
