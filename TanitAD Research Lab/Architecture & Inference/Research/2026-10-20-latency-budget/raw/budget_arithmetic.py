"""LATENCY-BUDGET arithmetic (2026-10-20). Stdlib only, CPU, no model, no data, no navtest.
Every ESTIMATE in LATENCY_BUDGET_PROPOSAL.md is computed here from the stated inputs, so the
memo's numbers can be re-derived and a changed input changes every dependent line.
Evidence classes: M = MEASURED (ours, artifact named), P = PUBLISHED (banked primary), I = INHERITED
(our doc, not re-verified), E = ESTIMATED (this script), U = UNVERIFIED assumption (a knob)."""
import json, math

# ---- inputs -----------------------------------------------------------------------------
# M: dev-box CPU, fp32, 4 threads, n=5 tokens, 1 repeat (search-thor-latency/raw/devbox5/*.json). NOT Thor.
cpu = {
    "a":  dict(p50=61.0617,  scorer=60.9858,   host=0.0759,  calls=1,  rows=64),
    "C8": dict(p50=870.2949, scorer=823.7505,  host=55.5145, calls=14, rows=896),
    "O4": dict(p50=2703.2978, scorer=2479.7571, host=214.5269, calls=38, rows=3392),
}
# REFe geometry (refe/model.py: img 960x512, patch 16, width 1024, depth 24 ViT-L, 4 cameras, 1 cls + 4 reg tokens)
H, W, P, D, L, CAMS, EXTRA = 512, 960, 16, 1024, 24, 4, 5
# I: THOR_PROFILE.md Addendum 4, bf16 4096^3 GEMM = 1.0856 ms on Thor (artifact JSON not located in the package)
gemm_ms, gemm_n = 1.0856, 4096
# M: SPEC_P6 s0.1, A40 train step bf16+TF32+compile = 0.2227 s per sample-pass (refe-final-2026-10-01/summary.json)
a40_train_s = 0.2227

out = {}
# ---- 1. dev-box CPU decomposition (M, arithmetic only) ---------------------------------------
dec = {}
for k, v in cpu.items():
    dec[k] = dict(
        ms_per_scorer_call=round(v["scorer"] / v["calls"], 2),
        ms_per_row=round(v["scorer"] / v["rows"], 3),
        host_ms_per_call=round(v["host"] / v["calls"], 2),
        host_share_pct=round(100 * v["host"] / v["p50"], 1),
        added_vs_a_ms=round(v["p50"] - cpu["a"]["p50"], 1),
    )
out["devbox_cpu_decomposition"] = dec
out["O4_over_C8_p50"] = round(cpu["O4"]["p50"] / cpu["C8"]["p50"], 2)

# ---- 2. trunk FLOPs and a Thor bf16 time band (E) ---------------------------------------------
N = (H // P) * (W // P) + EXTRA                      # tokens per camera
lin = 2 * (12 * D * D * L) * N                       # 2 * params * tokens
att = 4 * N * N * D * L                              # QK^T and AV
per_cam = lin + att
trunk_flops = per_cam * CAMS
thor_tflops = 2 * gemm_n ** 3 / (gemm_ms * 1e-3) / 1e12
out["trunk"] = dict(tokens_per_cam=N, tflop_per_cam=round(per_cam / 1e12, 3),
                    tflop_4cam=round(trunk_flops / 1e12, 2), thor_gemm_bf16_tflops=round(thor_tflops, 1))
band = {}
for u in (1.0, 0.6, 0.4):
    band[f"util_{u}"] = round(trunk_flops / (thor_tflops * 1e12 * u) * 1e3, 1)
out["trunk_ms_thor_bf16_by_utilisation"] = band
# cross-check: A40 train step -> forward-only. frozen trunk + LoRA: bwd ~ 1x fwd (dgrad only) .. 2x fwd => fwd = step/3 .. step/2
out["trunk_crosscheck_A40_fwd_ms"] = [round(a40_train_s / 3 * 1e3, 1), round(a40_train_s / 2 * 1e3, 1)]

# ---- 3. max scorer calls per stage budget (E; knobs are U) ------------------------------------
# c = Thor GPU ms per scorer call (U: launch-bound; graph-captured 1.5, eager 4.0), h = host ms per call
# (dev-box numpy 4.0 ms/call at C8 measured; ARM factor U; 0.5 if composite/aggregate are ported to GPU)
def max_calls(S, c, h):
    return int(math.floor(S / (c + h))) if S > 0 else 0
tab = {}
for name, S in (("opt1_search_10ms", 10), ("opt2_search_300ms", 300), ("opt3_search_60ms", 60)):
    row = {}
    for cn, c in (("graph1.5", 1.5), ("eager4.0", 4.0)):
        for hn, h in (("ported0.5", 0.5), ("unported4.0", 4.0)):
            row[f"{cn}|{hn}"] = max_calls(S, c, h)
    tab[name] = row
out["max_scorer_calls_per_search_stage"] = tab
out["needed_calls"] = dict(C8=14, O4=38)

# ---- 3b. sequential depth: the search latency is (dependent calls) x (c + h), not rows (U: rows are free up to a point;
# evidence for that: WoTE Tab.6, 64 -> 256 trajectories = 17.2 -> 18.7 ms on an L20, P). tts_cem.py (sha 9dea9603, read here):
# K=5 population calls are a dependent chain; the base-fan call and the n_elite final re-scorings are independent of each other
# and of the chain's results only after it ends, but the pinned code issues them one by one (A.score_replaced per elite).
depth = {"as_pinned_C8": 5 + 1 + 8, "as_pinned_O4": 5 + 1 + 32, "final_stage_batched_any_config": 5 + 1}
out["sequential_depth"] = depth
per_call = {"graph+ported": 1.5 + 0.5, "graph+unported": 1.5 + 4.0, "eager+ported": 4.0 + 0.5, "eager+unported": 4.0 + 4.0}
dd = {}
for dn, d_ in depth.items():
    dd[dn] = {pn: round(d_ * pc, 1) for pn, pc in per_call.items()}
out["search_ms_thor_by_depth_and_per_call_cost"] = dd
fits = {}
for sn, S in (("S10", 10), ("S60", 60), ("S300", 300)):
    fits[sn] = {dn: [pn for pn, v in dd[dn].items() if v <= S] for dn in depth}
out["which_per_call_regimes_fit_the_stage"] = fits

# ---- 4. full-cycle composites (E) = trunk + pick (c) + search --------------------------------------
pick = (2.0, 7.0)          # U: proposals + one scorer call, graph..eager
search = {}
for nm, calls in (("C8", 14), ("O4", 38)):
    lo = calls * (1.5 + 0.5)            # graph, host ported
    hi = calls * (4.0 + 4.0 * 3)        # eager, host unported x3 ARM factor (U)
    search[nm] = (lo, hi)
out["search_ms_thor_range"] = {k: [round(a, 1), round(b, 1)] for k, (a, b) in search.items()}
tr_lo, tr_hi = band["util_1.0"], band["util_0.4"]
out["cycle_ms_thor_range"] = {
    "pick_only": [round(tr_lo + pick[0], 1), round(tr_hi + pick[1], 1)],
    "pick+C8":  [round(tr_lo + pick[0] + search["C8"][0], 1), round(tr_hi + pick[1] + search["C8"][1], 1)],
    "pick+O4":  [round(tr_lo + pick[0] + search["O4"][0], 1), round(tr_hi + pick[1] + search["O4"][1], 1)],
}
# ---- 5. per-option share tables (design choices, sum to the cycle) --------------------------------
out["shares_ms"] = {
    "opt1_10Hz_single": dict(cycle=100, trunk=65, proposals_scorer=10, search=10, margin=15),
    "opt2_10Hz_pick__2Hz_search": dict(pick_cycle=100, pick_trunk=65, pick_proposals_scorer=10, pick_margin=25,
                                       search_cycle=500, search_stage=300, handoff_margin=200),
    "opt3_5Hz_single": dict(cycle=200, trunk=100, proposals_scorer=10, search=60, margin=30),
}
for k, v in out["shares_ms"].items():
    if k == "opt1_10Hz_single": assert v["trunk"] + v["proposals_scorer"] + v["search"] + v["margin"] == v["cycle"]
    if k == "opt3_5Hz_single": assert v["trunk"] + v["proposals_scorer"] + v["search"] + v["margin"] == v["cycle"]
    if k == "opt2_10Hz_pick__2Hz_search":
        assert v["pick_trunk"] + v["pick_proposals_scorer"] + v["pick_margin"] == v["pick_cycle"]
        assert v["search_stage"] + v["handoff_margin"] == v["search_cycle"]
print(json.dumps(out, indent=1))
