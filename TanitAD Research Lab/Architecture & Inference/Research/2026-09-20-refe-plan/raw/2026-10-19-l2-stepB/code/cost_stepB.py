#!/usr/bin/env python3
"""Cost of Step B, from the <= 3-token CPU smoke (MEASURED) projected to 400 tokens x 3 search seeds (ESTIMATED).  Reads TIMING fields only (t_search_s, t_token_s, the labeller's timing json); it never opens a
label or an outcome.   python cost_stepB.py --shard <shard.npz> --label-timing <json> [--stepA-label-secs 591 --stepA-tokens 400] --out cost.json"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--label-timing", required=True)
    ap.add_argument("--stepA-label-secs", type=float, default=591.0, help="MEASURED: l1run/steps4to7.log 'ZZLABEL_OK 400/400 591 s' (96 plans per token)")
    ap.add_argument("--stepA-tokens", type=int, default=400)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--arms", type=int, default=3, help="L2 head seed 0, L2 head seed 1, P2HEAD control")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(a.shard, allow_pickle=False)
    ts, tt = np.asarray(z["t_search_s"], float), np.asarray(z["t_token_s"], float)
    lt = json.load(open(a.label_timing))["timing"]
    S = ts.shape[1]
    search_tok = float(tt.mean())
    sa_tok = a.stepA_label_secs / a.stepA_tokens
    lab_smoke = float(lt["seconds_per_token_all_in"])
    out = {"evidence": "MEASURED (ours, CPU 4 threads, shared box) on the first 3 tokens of L1's Step A bundle, mechanics only; projections ESTIMATED",
           "smoke": {"n_tokens": int(ts.shape[0]), "n_seeds": S, "search_seconds_per_token_and_seed": {"mean": float(ts.mean()), "median": float(np.median(ts)), "max": float(ts.max())},
                     "search_seconds_per_token_all_seeds": search_tok, "label_seconds_per_token_smoke_cold": lab_smoke, "label_plans_per_token": lt["plans_labelled_per_token"],
                     "label_breakdown_seconds_3_tokens": {k: lt[k] for k in ("seconds_scenario_build_total", "seconds_metric_cache_parts_total", "seconds_label_plans_total")}},
           "stepA_label_reference": {"seconds_per_token_all_in_at_96_plans": sa_tok, "source": "l1run/steps4to7.log: ZZLABEL_OK 400/400 591 s"},
           "projection_per_arm_400_tokens_x_3_seeds": {"search_minutes": a.tokens * search_tok / 60.0, "label_minutes_low (StepA rate, an upper bound for 5 plans/token)": a.tokens * sa_tok / 60.0,
                                                       "label_minutes_high (smoke cold rate)": a.tokens * lab_smoke / 60.0}}
    p = out["projection_per_arm_400_tokens_x_3_seeds"]
    out["projection_total_arms"] = {"arms": a.arms, "sequential_hours_low": a.arms * (p["search_minutes"] + p["label_minutes_low (StepA rate, an upper bound for 5 plans/token)"]) / 60.0,
                                    "sequential_hours_high": a.arms * (p["search_minutes"] + p["label_minutes_high (smoke cold rate)"]) / 60.0,
                                    "note": "CPU only, no GPU, no lock; two processes in parallel roughly halve the wall clock; the gate (about 1 min per head) is extra"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["projection_per_arm_400_tokens_x_3_seeds"]), json.dumps(out["projection_total_arms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
