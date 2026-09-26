"""Map-signal audit, task 7: activation memory and compute of the three map-architecture options at b16.

ANALYTIC from real shapes (arithmetic only, no tensor is allocated):
  * batch 16; frame 416 x 1024; resnet101: stride-16 map (layer3) 1024 x 26 x 64 (config.json
    refcv6_perception.fmap_s16_*), stride-8 map (layer2) 512 x 52 x 128;
  * refcv6 0.5 m lift (bev_lift.py:248-270): grid_sample of the stride-16 map at Z = 4 heights on the
    120 x 64 grid -> [B, C16, Z*120, 64]; masked; permuted + reshaped (a copy) to [B, Z*C16, 120, 64];
    1x1 conv to 128 (the conv SAVES its input for backward);
  * NEW-2 map-only lift (SPEC_REFCV7 s6.2): the same operator on the stride-8 map at 0.25 m, 240 x 128;
    Z is not specified by the SPEC -> both Z = 4 (refcv6's heights) and Z = 1 (road plane only);
  * the 10 cm decoder is NOT BUILT YET: an ASSUMED decoder is priced (conv 3x3 at 240x128 with
    C_dec channels x 2 layers, x2.5 upsample to 600 x 320 with C_up channels x 2 layers, 1x1 to 8
    logits) and every such number is labelled ASSUMED;
  * fp32 activations (the branch runs outside the backbone's bf16 autocast, timm_trunk.py:719-726).
Output: raw/arch_memory_analytic.json
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "raw" / "arch_memory_analytic.json"
B = 16
FP32 = 4
GB = 2 ** 30


def gb(n_elems, bytes_per=FP32):
    return round(n_elems * bytes_per / GB, 3)


def lift(c_in, grid_xy, z, d_out=128):
    X, Y = grid_xy
    s = B * c_in * z * X * Y                     # grid_sample output == reshaped conv input (elements)
    return {"c_in": c_in, "grid": [X, Y], "heights": z, "d_out": d_out,
            "sampled_tensor_GB_fp32": gb(s),
            "saved_for_backward_GB_fp32 (1x1 conv input)": gb(s),
            "transient_forward_peak_GB_fp32 (sample + mask + reshape copy)": gb(3 * s),
            "output_GB_fp32": gb(B * d_out * X * Y),
            "proj_GFLOP_fwd": round(2 * B * X * Y * c_in * z * d_out / 1e9, 1),
            "per_height_accumulated_saved_GB_fp32 (sum_h W_h s_h, recompute per height)": gb(s / z)}


def decoder(c_dec=64, c_up=32):
    lo = B * c_dec * 240 * 128
    hi = B * c_up * 600 * 320
    logits = B * 8 * 600 * 320
    return {"ASSUMED": True, "c_dec": c_dec, "c_up": c_up,
            "saved_GB_fp32 (2 convs at 240x128 + 2 at 600x320 + norms/acts ~x2)":
                gb(2 * (2 * lo + 2 * hi)),
            "logits_GB_fp32": gb(logits), "log_softmax+grad_GB_fp32": gb(2 * logits),
            "GFLOP_fwd": round((2 * B * 240 * 128 * c_dec * c_dec * 9 * 2
                                + 2 * B * 600 * 320 * c_up * c_up * 9 * 2
                                + B * 600 * 320 * c_up * 8 * 2) / 1e9, 1)}


def main():
    trunk_frame_gflop = 7.8 * (416 * 1024) / (224 * 224)       # resnet101 ~7.8 GFLOP at 224^2
    res = {
        "batch": B, "frame": [416, 1024],
        "reference_refcv6_peak_cuda_GB": 21.63,
        "reference_refcv6_s_per_step": 6.63,
        "note_reference": "metrics.jsonl cuda_max_mem_gb max 21.63; median s/step 6.63 over 762 log gaps",
        "trunk_fwd_GFLOP_per_frame_approx": round(trunk_frame_gflop, 1),
        "trunk_fwd_TFLOP_per_step_approx (16 windows x 10 computed frames)": round(16 * 10 * trunk_frame_gflop / 1e3, 1),
        "refcv6_lift_0p5m_s16_Z4": lift(1024, (120, 64), 4),
        "new2_lift_0p25m_s8_Z4": lift(512, (240, 128), 4),
        "new2_lift_0p25m_s8_Z1_road_plane": lift(512, (240, 128), 1),
        "new2_lift_0p25m_s8_plus_s16_Z1": lift(1536, (240, 128), 1),
        "decoder_10cm_assumed": decoder(),
    }
    r6, n4, n1 = res["refcv6_lift_0p5m_s16_Z4"], res["new2_lift_0p25m_s8_Z4"], res["new2_lift_0p25m_s8_Z1_road_plane"]
    d = res["decoder_10cm_assumed"]
    dec_saved = d["saved_GB_fp32 (2 convs at 240x128 + 2 at 600x320 + norms/acts ~x2)"] + d["log_softmax+grad_GB_fp32"]
    key = "saved_for_backward_GB_fp32 (1x1 conv input)"
    res["options"] = {
        "a_keep_0p5m_aux (refcv6 lift + NEW-2 lift + decoder)": {
            "added_saved_GB_vs_refcv6 (Z4 / Z1 map lift)": [round(n4[key] + dec_saved, 2), round(n1[key] + dec_saved, 2)],
            "added_proj_GFLOP_fwd (Z4 / Z1)": [n4["proj_GFLOP_fwd"], n1["proj_GFLOP_fwd"]]},
        "b_remove_0p5m (NEW-2 lift + decoder; the 0.5 m lift stays for box3d/planner unless also removed)": {
            "added_saved_GB_vs_refcv6 (Z4 / Z1)": [round(n4[key] + dec_saved, 2), round(n1[key] + dec_saved, 2)],
            "note": "removing only the 0.5 m LOSS saves nothing: the 0.5 m lift + BEV encoder still feed box3d, "
                    "the 30x16 BEV tokens and the planner's cross-attention"},
        "c_one_0p25m_lift_feeding_both (pooled 120x64 for planner/box3d/tokens)": {
            "added_saved_GB_vs_refcv6 (Z4 / Z1)": [round(n4[key] + dec_saved - r6[key], 2),
                                                   round(n1[key] + dec_saved - r6[key], 2)],
            "note": "the stride-16 0.5 m lift is removed; the planner-facing BEV then comes from the stride-8 "
                    "map (512 ch, layer2) instead of the stride-16 map (1024 ch, layer3)"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
