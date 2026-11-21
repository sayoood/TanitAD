#!/usr/bin/env python3
"""Orin / Thor tick projection with an HONEST bracket, from the measured budget.

Two failure modes this script exists to avoid:
  1. quoting a roofline FLOOR as if it were a predicted tick (it is unreachable
     by construction: 100 % of peak bandwidth);
  2. quoting a desktop-GPU millisecond as an embedded millisecond (the retracted
     14.331 ms error).

So every device tick is reported as a BRACKET:
    lower  = stage-wise roofline floor  (unreachable)
    upper  = the same tick at the DRAM-utilisation fraction the A40 actually
             achieved on the MEASURED composed tick (18.75 ms, eff_levers)
and both ends are labelled ESTIMATED.  What is decision-grade is that the
bracket's WHOLE RANGE clears (or misses) the 10 Hz budget, not the endpoints.

Run: python projection.py --budget artifacts/budget_report.json \
                          --lever artifacts/lever_report.json \
                          --out artifacts/projection_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# PUBLISHED device peaks (see lever_probe.DEVICES for sources / DERIVED chain)
DEV = {
    "a40_proxy":  {"bw": 696.0,  "fp16": 149.7, "int8": 299.3,
                   "fp8": None, "nvfp4": None, "power_w": "300 (TDP)"},
    "orin_agx64": {"bw": 204.8,  "fp16": 42.5,  "int8": 85.0,
                   "fp8": None, "nvfp4": None, "power_w": "15-60"},
    "thor_t5000": {"bw": 273.0,  "fp16": 258.75, "int8": 517.5,
                   "fp8": 517.5, "nvfp4": 1035.0, "power_w": "40-130"},
}
# MEASURED anchor: the composed L4 tick on the A40 (taniteval/results/
# eff_levers_flagship-30k.json, lever `all_levers`), fp16 weights + pruned
# horizon heads + cached 1-frame encode + whole-rollout CUDA graph.
A40_COMPOSED_TICK_MS = 18.7476
LPDDR5_PJ_PER_BIT = 7.0          # ESTIMATED, published LPDDR5 system energy
                                 # range is ~5-8 pJ/bit; midpoint used.


def stage_floor(gflops, mb, tflops, bw):
    """max(compute, bandwidth) for ONE stage, in ms."""
    tc = gflops / tflops if tflops else float("inf")
    tb = mb / bw
    return max(tc, tb), tc, tb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="artifacts/budget_report.json")
    ap.add_argument("--lever", default="artifacts/lever_report.json")
    ap.add_argument("--out", default="artifacts/projection_report.json")
    a = ap.parse_args()
    B = json.loads(Path(a.budget).read_text(encoding="utf-8"))
    L = json.loads(Path(a.lever).read_text(encoding="utf-8"))

    tb = B["tick_budget"]
    gf_enc = tb["gflops"]["encoder_1frame"]
    gf_roll = tb["gflops"]["rollout_decode_k20_measured"]
    traf = tb["weight_traffic_mb_per_tick"]

    rep = {
        "what": "Orin/Thor tick BRACKETS for the deployed flagship-v1 planning "
                "tick, from the MEASURED per-component budget. ESTIMATED "
                "throughout -- no Orin/Thor silicon in this session.",
        "measured_inputs": {
            "encoder_1frame_gflops": gf_enc,
            "rollout_k20_gflops": gf_roll,
            "weight_traffic_mb_per_tick": traf,
            "a40_composed_tick_ms_MEASURED": A40_COMPOSED_TICK_MS,
            "source": "budget_probe.py artifacts + "
                      "taniteval/results/eff_levers_flagship-30k.json",
        },
    }

    # --- 1. what DRAM utilisation did the A40 actually achieve? -------------
    a40 = DEV["a40_proxy"]
    a40_floor_enc, _, _ = stage_floor(gf_enc, traf["fp16"]["encoder_once_mb"],
                                      a40["fp16"], a40["bw"])
    a40_roll_mb = (traf["fp16"]["total_tick_mb"]
                   - traf["fp16"]["encoder_once_mb"])
    a40_floor_roll, _, _ = stage_floor(gf_roll, a40_roll_mb,
                                       a40["fp16"], a40["bw"])
    a40_floor = a40_floor_enc + a40_floor_roll
    a40_util = a40_floor / A40_COMPOSED_TICK_MS
    rep["a40_calibration"] = {
        "stagewise_fp16_floor_ms": round(a40_floor, 3),
        "measured_composed_tick_ms": A40_COMPOSED_TICK_MS,
        "achieved_fraction_of_roofline": round(a40_util, 4),
        "achieved_dram_gb_s": round(traf["fp16"]["total_tick_mb"]
                                    / A40_COMPOSED_TICK_MS, 1),
        "reading": "the A40's composed tick sits at ~%.0f%% of its own "
                   "stage-wise roofline. Consistent with the registry's "
                   "'achieved 3.7-4.3 TFLOPs => launch/serialisation-bound' "
                   "diagnosis: on a 696 GB/s datacentre part this workload "
                   "never becomes bandwidth-limited, so the A40 tick is NOT a "
                   "bandwidth measurement and its 18.75 ms does NOT scale to "
                   "Jetson by a bandwidth ratio." % (100 * a40_util),
    }

    # --- 2. the brackets ----------------------------------------------------
    out = {}
    for dname in ("orin_agx64", "thor_t5000"):
        d = DEV[dname]
        rows = {}
        for prec in ("fp16", "int8", "fp8", "nvfp4"):
            peak = d[prec]
            if peak is None:
                rows[prec] = {"supported": False,
                              "why": "no tensor-core datapath on this arch"}
                continue
            enc_mb = traf[prec]["encoder_once_mb"]
            roll_mb = traf[prec]["total_tick_mb"] - enc_mb
            f_enc, c_enc, b_enc = stage_floor(gf_enc, enc_mb, peak, d["bw"])
            f_roll, c_roll, b_roll = stage_floor(gf_roll, roll_mb, peak, d["bw"])
            floor = f_enc + f_roll
            upper = floor / a40_util          # same efficiency as the A40 tick
            rows[prec] = {
                "supported": True,
                "encoder_stage": {"floor_ms": round(f_enc, 3),
                                  "compute_ms": round(c_enc, 3),
                                  "bandwidth_ms": round(b_enc, 3),
                                  "binds_on": "compute" if c_enc > b_enc
                                              else "bandwidth"},
                "rollout_stage": {"floor_ms": round(f_roll, 3),
                                  "compute_ms": round(c_roll, 3),
                                  "bandwidth_ms": round(b_roll, 3),
                                  "binds_on": "compute" if c_roll > b_roll
                                              else "bandwidth"},
                "tick_bracket_ms": [round(floor, 2), round(upper, 2)],
                "tick_bracket_hz": [round(1000 / upper, 1), round(1000 / floor, 1)],
                "meets_10hz_across_whole_bracket": (1000 / upper) >= 10.0,
                "headroom_vs_10hz": [round(100 / upper, 2), round(100 / floor, 2)],
                "dram_energy_j_per_tick_ESTIMATED": round(
                    traf[prec]["total_tick_mb"] * 1e6 * 8 * LPDDR5_PJ_PER_BIT
                    * 1e-12, 4),
                "dram_watts_at_10hz_ESTIMATED": round(
                    traf[prec]["total_tick_mb"] * 1e6 * 8 * LPDDR5_PJ_PER_BIT
                    * 1e-12 * 10, 3),
                "module_power_envelope_w": d["power_w"],
            }
        out[dname] = rows
    rep["brackets"] = out

    # --- 3. what each lever does to the Orin fp16 bracket -------------------
    orin = DEV["orin_agx64"]
    base_enc_mb = traf["fp16"]["encoder_once_mb"]
    base_roll_mb = traf["fp16"]["total_tick_mb"] - base_enc_mb
    p_pred = 91_361_280
    p_sr = 2_107_395

    def orin_tick(gf_e, enc_mb, gf_r, roll_mb, peak=None):
        peak = peak or orin["fp16"]
        fe, _, _ = stage_floor(gf_e, enc_mb, peak, orin["bw"])
        fr, _, _ = stage_floor(gf_r, roll_mb, peak, orin["bw"])
        return fe + fr

    base = orin_tick(gf_enc, base_enc_mb, gf_roll, base_roll_mb)
    # unused-head pruning (L7): 2 of 3 heads, Linear(768->2048)
    head_p = 768 * 2048 + 2048
    lev = {}
    lev["baseline_fp16"] = {"tick_floor_ms": round(base, 3), "delta_pct": 0.0}
    # L7 drop 2 unused horizon heads
    mb_l7 = base_roll_mb - 2 * head_p * 2 * 20 / 1e6
    lev["L7_drop_2_unused_horizon_heads"] = {
        "tick_floor_ms": round(orin_tick(gf_enc, base_enc_mb,
                                         gf_roll * 0.96, mb_l7), 3)}
    # strided k=2 / k=4
    for k, calls in (("k2", 10), ("k4", 5)):
        mb = (p_pred + p_sr) * 2 * calls / 1e6
        lev[f"strided_head_{k}_{calls}calls"] = {
            "tick_floor_ms": round(orin_tick(gf_enc, base_enc_mb,
                                             gf_roll * calls / 20, mb), 3)}
    # predictor width / depth pruning (from lever_probe scans)
    for row in L["B_rollout_structure"]["width_scan"]:
        pp = row["params"] + 197_376          # + intent_proj, as deployed
        mb = (pp + p_sr) * 2 * 20 / 1e6
        lev[f"predictor_width_d{row['d_model']}"] = {
            "tick_floor_ms": round(orin_tick(
                gf_enc, base_enc_mb, row["gflops_per_step"] * 20, mb), 3),
            "params": pp}
    for row in L["B_rollout_structure"]["depth_scan"]:
        pp = row["params"] + 197_376
        mb = (pp + p_sr) * 2 * 20 / 1e6
        lev[f"predictor_depth_{row['depth']}"] = {
            "tick_floor_ms": round(orin_tick(
                gf_enc, base_enc_mb, row["gflops_per_step"] * 20, mb), 3),
            "params": pp}
    # KV cache ceiling (FLOPs only; weights unchanged)
    lev["kv_cache_perfect_FLOPs_only"] = {
        "tick_floor_ms": round(orin_tick(
            gf_enc, base_enc_mb,
            L["B_rollout_structure"]["kv_cache_ceiling"]
            ["gflops_per_step_incremental"] * 20, base_roll_mb), 3)}
    # operator fusion ceiling: removes ALL activation traffic (0.4 MB of 3913)
    lev["operator_fusion_perfect_activations_only"] = {
        "tick_floor_ms": round(orin_tick(gf_enc, base_enc_mb,
                                         gf_roll, base_roll_mb), 3),
        "note": "identical to baseline by construction: the rollout's MEASURED "
                "activation footprint is 0.404 MB against 3,739 MB of weight "
                "traffic (0.011 %). Fusion cannot move the binding term."}
    # encoder crop scenarios
    for s in L["A_encoder_scaling"]["crop_scenarios_fitted"]:
        lev["crop::" + s["scenario"][:44]] = {
            "tick_floor_ms": round(orin_tick(s["gflops_fitted"], base_enc_mb,
                                             gf_roll, base_roll_mb), 3),
            "encoder_gflops": s["gflops_fitted"]}
    for k, v in lev.items():
        if "delta_pct" not in v:
            v["delta_pct"] = round(100 * (v["tick_floor_ms"] - base) / base, 2)
    rep["orin_fp16_lever_deltas"] = {
        "note": "each lever applied ALONE to the Orin fp16 stage-wise floor. "
                "ESTIMATED. The floor is the right yardstick here because it "
                "isolates the physics (bytes and FLOPs) from the runtime; "
                "lever ORDERING on real silicon must be re-measured.",
        "baseline_tick_floor_ms": round(base, 3),
        "levers": dict(sorted(lev.items(), key=lambda kv: kv[1]["tick_floor_ms"]))}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("wrote", a.out)
    print("\nA40 achieved fraction of its own roofline: %.1f%%  (%.0f GB/s)"
          % (100 * a40_util, traf["fp16"]["total_tick_mb"]
             / A40_COMPOSED_TICK_MS))
    for dn, rows in out.items():
        print("==", dn)
        for p, r in rows.items():
            if not r.get("supported"):
                print("   %-6s UNSUPPORTED" % p)
                continue
            print("   %-6s tick %6.2f - %6.2f ms  (%5.1f - %5.1f Hz)  "
                  "10Hz-safe=%s  dram %.2f W" %
                  (p, r["tick_bracket_ms"][0], r["tick_bracket_ms"][1],
                   r["tick_bracket_hz"][0], r["tick_bracket_hz"][1],
                   r["meets_10hz_across_whole_bracket"],
                   r["dram_watts_at_10hz_ESTIMATED"]))
    print("\n-- Orin fp16 lever deltas (floor, ms) --")
    for k, v in rep["orin_fp16_lever_deltas"]["levers"].items():
        print("   %-58s %7.2f  (%+6.1f %%)"
              % (k[:58], v["tick_floor_ms"], v["delta_pct"]))


if __name__ == "__main__":
    main()
