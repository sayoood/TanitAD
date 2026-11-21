"""Sizing arithmetic for a v8F-mini T2 re-perception read on box 2 (RTX 4060, 8 GiB).  CPU, no I/O except stdout.
Every input is tagged: M = MEASURED (artifact named), I = INHERITED (doc named, not re-run), E = ESTIMATED (assumption stated).
Run: python -I sizing_arith.py > T2_SIZING_V8F.json
"""
import json
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")

IN = {
    "render_ms_per_tick": (20.0, "M banked_timing.json: mean render_ms 17.7-21.7 on box 2, 1920x1080, empty condition (objects adds actors: E +0..+50 %)"),
    "dinov3_ms_per_frame": (29.0, "I stack/scripts/dinov3_fp8_encode_ship.py docstring: 4060 bf16 ~2.9 s/episode of ~100 grid frames; box-2 end to end 5.01 s/clip (D3_SCOPING.md)"),
    "canon_resize_ms": (15.0, "E cylindrical_rectify to 416x1024 + bilinear to 256x640; inside refcv7's 313 ms plan, not separable"),
    "forward_ms": (15.0, "E 7.2 M-param model + HACC post-map, batch 1, GPU; T1-kin CPU smoke 14 s / 24 windows (DESIGN_T1_V8F section 7)"),
    "contention_factor": (2.5, "M banked_timing.json: same refcv7 arm plan_ms 314 (scene 1) vs 800 (scene 2)"),
    "load_s_per_arm_scene": (15.0, "E scene decode 2.9 s (M preflight.json) + DINOv3-L and model load"),
    "ticks_per_rollout": (50, "M T2-A harness: 50 ticks at 10 Hz"),
    "starts_per_scene": (10, "E T2-A used 9 (0,17..136); 10 fits the 199-frame clip"),
    "renders_per_02s_step": (2, "M harness renders every 0.1 s tick (set 1 if only policy ticks are rendered)"),
}
v = {k: x[0] for k, x in IN.items()}
step_ms = v["renders_per_02s_step"] * v["render_ms_per_tick"] + v["dinov3_ms_per_frame"] + v["canon_resize_ms"] + v["forward_ms"]
step_ms_lo = 1 * v["render_ms_per_tick"] + v["dinov3_ms_per_frame"] + 0.5 * v["canon_resize_ms"] + 0.5 * v["forward_ms"]
steps = v["ticks_per_rollout"] // 2
roll_s = steps * step_ms / 1000
arm_scene_s = v["starts_per_scene"] * roll_s + v["load_s_per_arm_scene"]
arm_scene_s_lo = v["starts_per_scene"] * steps * step_ms_lo / 1000 + v["load_s_per_arm_scene"]
arm_scene_s_hi = v["starts_per_scene"] * roll_s * v["contention_factor"] + v["load_s_per_arm_scene"]

out = {"inputs": {k: {"value": x[0], "class_and_source": x[1]} for k, x in IN.items()},
       "per_step_ms_0p2s": {"low": step_ms_lo, "central": step_ms, "note": "render x2 + DINOv3 + canon + forward"},
       "rollout_s_central": roll_s, "arm_scene_s": {"low": arm_scene_s_lo, "central": arm_scene_s, "contended_high": arm_scene_s_hi}}

# VRAM (GiB reserved)
vram = {"renderer_alone_M": 1.355, "both_resident_with_refcv7_M": 2.127, "dinov3_L_bf16_weights_E": 304e6 * 2 / 2**30, "v8f_fp32_E": 7.2e6 * 4 / 2**30,
        "dinov3_activations_canon_buffers_E": 0.4, "actor_gaussians_E": 0.3}
vram["v8f_t2_total_E"] = vram["renderer_alone_M"] + vram["dinov3_L_bf16_weights_E"] + vram["v8f_fp32_E"] + vram["dinov3_activations_canon_buffers_E"] + vram["actor_gaussians_E"]
vram["fraction_of_8GiB"] = vram["v8f_t2_total_E"] / 8.0
out["vram_gib"] = vram

# Stage 1: two on-hand scenes
arms1 = ["C0_s0", "C0_s1", "base", "P10_prior", "zero_control", "camera_blind_control", "render_replay_control"]
conds = ["objects", "empty"]
runs1 = len(arms1) * len(conds) * 2
out["stage1_two_scenes"] = {"arms": arms1, "conditions": conds, "arm_scene_runs": runs1,
                            "wall_min": {"low": runs1 * arm_scene_s_lo / 60, "central": runs1 * arm_scene_s / 60, "contended_high": runs1 * arm_scene_s_hi / 60},
                            "gates_extra_min_E": 15, "download_gb": 0.0}

# Stage 2: K M1-bearing scenes
VOL_GB, FULL_GB, SMALL_MB = 0.62, 1.90, 5.0
arms2 = 8
for K in (20, 42, 82):
    runs = K * arms2
    out[f"stage2_K{K}"] = {"scenes": K, "arms": arms2, "arm_scene_runs": runs,
                           "wall_h": {"low": runs * arm_scene_s_lo / 3600, "central": runs * arm_scene_s / 3600, "contended_high": runs * arm_scene_s_hi / 3600},
                           "download_gb_range_members": K * (VOL_GB + SMALL_MB / 1024), "download_gb_full_usdz": K * FULL_GB,
                           "screening_scenes_if_p_m1_0.25": math.ceil(K / 0.25), "screening_scenes_if_p_m1_0.5": math.ceil(K / 0.5),
                           "screening_traffic_mb_E": math.ceil(K / 0.25) * 4.0}
out["download_basis"] = ("M volume.nurec 639,393,135 and 603,238,216 B; full USDZ 1,955,916,532 and 1,835,661,788 B (ls on E:/thor_archive); "
                         "I STORED zip members allow HTTP Range pulls (STRATEGIC_FAMILY.md: 39/39 members verified); rig+tracks about 2.1+0.5..1.7 MB (M ls)")
json.dump(out, sys.stdout, indent=1)
