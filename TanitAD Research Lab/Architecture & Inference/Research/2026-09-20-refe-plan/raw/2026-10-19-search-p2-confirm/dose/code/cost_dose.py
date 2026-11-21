#!/usr/bin/env python3
"""CPU cost of the SEARCH-DOSE arm, ESTIMATED from Step B's MEASURED M = 64 rate.  Reads only the banked carve-B P2HEAD SEARCH file (timings, moved flags; no label, no B-REST data).
    python cost_dose.py --out ../raw/cost_dose.json
The model (every input named): time = (number of score_fn calls) x (cost of one call), where a call is one forward of the scoring decoder (model.py score_trajectories: 4 CrossBlocks, dim 256, 8 heads,
ctx 7680 x 256).  Per token and search seed the search makes K = 5 population calls of M members, ONE call on the original 64-fan (tts_cem.py:64) and n_elite = M/8 calls on the 64-fan with the pick replaced
(tts_cem.py:67-70), plus one more fan call when the output moved (stepB.py run_search); plus ONE base fan call per token.  Cost of a call of m members = FIXED (K/V projection of the 7680 ctx tokens, independent of m) +
m x PER-QUERY (FLOPs below)."""
from __future__ import annotations

import argparse
import json

import numpy as np

BANKED = "C:/Users/Admin/par18/l2stepb/run/search_P2HEAD.npz"
N_TOK, SEEDS, K = 905, 3, 5
DIM, CTX, LAYERS, HID = 256, 7680, 4, 1024


def macs(m: int) -> float:
    fixed = LAYERS * 2 * CTX * DIM * DIM                                           # K and V projections of the ctx, per call
    per_q = LAYERS * (3 * DIM * DIM + DIM * DIM + DIM * DIM + 2 * CTX * DIM + DIM * DIM + 2 * DIM * HID) + 60 * DIM + DIM * DIM    # in/out proj, Q proj, cross QK and AV, MLP, score_q_mlp
    selfattn = LAYERS * 2 * m * DIM                                                # set self-attention scores + AV, per query
    return fixed + m * (per_q + selfattn)


def calls(m: int, n_elite: int, moved: float) -> dict:
    per_seed_pop = K                                                               # calls of m members
    per_seed_fan = 1 + n_elite + moved                                             # calls of 64 members
    return {"pop_calls_per_seed": per_seed_pop, "fan_calls_per_seed": per_seed_fan, "pop_calls_per_token": SEEDS * per_seed_pop, "fan_calls_per_token": 1 + SEEDS * per_seed_fan}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(BANKED, allow_pickle=False)
    ts, tt, mv = z["t_search_s"], z["t_token_s"], z["moved"].astype(bool)
    moved = float(mv.mean())
    c64_not = float(ts[~mv].mean() / (K + 1 + 8))                                   # a not-moved seed = 14 calls of 64
    c64_mv = float(ts[mv].mean() / (K + 1 + 8 + 1))                                  # a moved seed = 15 calls
    c64 = float(tt.mean() / (1 + SEEDS * (K + 1 + 8 + moved)))                       # the whole token incl. its base call
    base = calls(64, 8, moved)
    n_c8 = base["pop_calls_per_token"] + base["fan_calls_per_token"]
    out = {"evidence": "rates MEASURED (carve-B P2HEAD search file: 400 tokens x 3 seeds, 4 threads, shared box); the call-count model is checked against them below; the M = 256 call cost is ESTIMATED (FLOP model) and UNMEASURED",
           "source": BANKED, "moved_share_carveB_P2HEAD": moved, "measured_s_per_token_3_seeds": float(tt.mean()),
           "implied_seconds_per_64_call": {"from_not_moved_seeds": c64_not, "from_moved_seeds": c64_mv, "from_whole_token": c64, "moved_minus_not_moved_seed_s (should equal one call)": float(ts[mv].mean() - ts[~mv].mean())},
           "C8_calls_per_token_of_64_equivalents": n_c8}
    rows = {}
    for name, ratio in (("lower: a 256-call costs the same as a 64-call (fixed cost dominates)", 1.0), ("central: FLOP model", macs(256) / macs(64)), ("upper: a call is linear in M", 4.0)):
        o4 = calls(256, 32, moved)
        units = o4["pop_calls_per_token"] * ratio + o4["fan_calls_per_token"]
        r = units / n_c8
        sec = float(tt.mean()) * r
        rows[name] = {"c256_over_c64": ratio, "O4_over_C8_per_token": r, "s_per_token": sec, "hours_905_tokens": sec * N_TOK / 3600.0}
    out["FLOP_model"] = {"MACs_per_call_M64": macs(64), "MACs_per_call_M256": macs(256), "fixed_share_at_M64": LAYERS * 2 * CTX * DIM * DIM / macs(64), "implied_GFLOPs_at_4_threads": 2 * macs(64) / c64 / 1e9,
                         "elite_rescoring_share_of_O4_work": calls(256, 32, moved)["fan_calls_per_token"] / (calls(256, 32, moved)["fan_calls_per_token"] + calls(256, 32, moved)["pop_calls_per_token"] * macs(256) / macs(64))}
    out["O4_search_cost"] = rows
    s_worst_p2head, s_worst_any = 5.61, 16.07                                        # Step B's worst P2HEAD shard / worst shard of any arm, s per token at M = 64 (spc raw/cost.json)
    cen = rows["central: FLOP model"]["O4_over_C8_per_token"]
    out["outlier_scaling"] = {"worst_P2HEAD_shard_rate_hours": s_worst_p2head * cen * N_TOK / 3600.0, "worst_any_arm_rate_hours (shared-box outlier)": s_worst_any * cen * N_TOK / 3600.0}
    out["other_stages_minutes"] = {"G-REACH on the O4 file (Step B budgets 1 min per arm at M=64; x ~2.9)": 3.0, "labels, 5 plans per token (M-independent; spc cost.json 16.7-27.1)": [16.7, 27.1], "analysis incl. contrast, B = 10,000": 3.0,
                                   "G-SUBID: 16 tokens x 3.95 s at M=64 plus the model load (MM_RULING_DOSE.md DQ5; the ruling says about 3)": 3.0,
                                   "G-SELFREP: 8 tokens x 11.4 s at M=256 (the ruling says about 1.5)": 1.5}
    out["total_hours_central"] = rows["central: FLOP model"]["hours_905_tokens"] + (3.0 + 3.0 + 1.5 + 3.0 + 22.0) / 60.0
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: out[k] for k in ("implied_seconds_per_64_call", "FLOP_model", "O4_search_cost", "outlier_scaling")}, indent=1))


if __name__ == "__main__":
    main()
