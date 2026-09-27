#!/usr/bin/env python3
"""ladder_table.py -- the one-frame attribution ladder (gbo_diagnose.py --parts ladder) as one markdown table.
usage: python ladder_table.py <gbo_ladder.json>"""
import json
import sys

import numpy as np


def f(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


r = json.load(open(sys.argv[1], encoding="utf-8"))
L = r["parts"]["ladder"]
print(f"frame {L['frame']} | {L['steps']} steps, log every {L['every']}")
print("| rung | what changes | p(matched) median / max @500 (eval) | max over run | p(unmatched) max | conf | "
      "AP@2m | centre p50 m | slot spread m | tgt->nearest / 2nd slot m | slot kept over 25 steps (mean, max) | "
      "memory frame-share 0 -> 500 | box loss |")
print("|" + "---|" * 14)
what = {"main": "MAIN config (baseline)", "frozen_trunk": "only box decoder + box memory train",
        "lr_2e-5": "lr 2e-5 (10x lower)", "bce_presence": "BCE-0.1 presence (not focal)",
        "no_deep_sup": "no per-layer supervision", "q100": "100 queries (not 300)",
        "refcv6_head": "refcv6 head: BCE, prior 0.05, no deep sup, 100 q, refcv6 targets"}
for name, rg in L["rungs"].items():
    if "ERROR" in rg:
        print(f"| {name} | {what.get(name, '')} | ERROR {rg['ERROR'][:80]} |")
        continue
    rows = rg["rows"]
    last = rows[-1]
    st = [x["assign_stability_25"] for x in rows if x.get("assign_stability_25") is not None]
    print(f"| {name} | {what.get(name, '')} | {f(last['p_matched_median'])} / {f(last['p_matched_max'])} | "
          f"{f(rg['max_p_matched_median'])} (median) | {f(last['p_unmatched_max'])} | {last['n_conf']} | "
          f"{f(last['ap2m'])} | {f(last['centre_p50_m'])} | {f(last['slot_centre_spread_m'], 1)} | "
          f"{f(last.get('tgt_to_nearest_slot_m'), 2)} / {f(last.get('tgt_to_2nd_slot_m'), 2)} | "
          f"{f(float(np.mean(st)) if st else None)} , {f(max(st) if st else None)} | "
          f"{f(rows[0]['mem_frame_share'])} -> {f(last['mem_frame_share'])} | {f(last.get('loss_mean'), 2)} |")
