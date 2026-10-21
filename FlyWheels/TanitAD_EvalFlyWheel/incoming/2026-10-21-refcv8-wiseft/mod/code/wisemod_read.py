#!/usr/bin/env python3
"""Apply the REGISTERED R8-WISEFT-MOD rule (7ec285d, before any score) to the pulled Thor scores.  Writes WISEMOD_READ.json + .md.
share_X(s) = (PDMS(theta_X, s) - PDMS(EMA-48,852, s)) / (PDMS(theta_0.5, s) - PDMS(EMA-48,852, s)).  X is the LOCUS iff share >= 0.75 at both
seeds AND theta_X - EMA-48,852 is admissible (separated positive at both seeds, min margin > 2 x |d_s0 - d_s1|).  Both < 0.75 -> DISTRIBUTED;
both >= 0.75 -> REDUNDANT.  Controls: EMA - EMA == 0; banked theta_0.5 (76.3342 / 76.4399) and EMA (63.3907 / 63.0232) reproduce; identical
tokens; complement sanity (reported, not a gate): PDMS(PLAN) + PDMS(PERC) vs PDMS(theta_0.5) + PDMS(EMA) -- additivity of the two stacks.
"""
import importlib.util
import json
import sys

import numpy as np

W = "C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft"
R = "C:/Users/Admin/refcv8_pf/navsim_ckpt/run"
spec = importlib.util.spec_from_file_location("b8m", "C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                              "2026-10-19-refcv8-navsim/code/navsim_bars8.py")
B = importlib.util.module_from_spec(spec)
sys.modules["b8m"] = B
spec.loader.exec_module(B)
rd = B.read_scores_csv
DIAG = f"{R}/navtrain_sel/diag/scores"
REF = {"s0": rd(f"{DIAG}/r8sel_diag_ema48852_LEGAL_s0/r8sel_diag_ema48852_LEGAL_s0.csv"),
       "s1": rd(f"{DIAG}/r8sel_diag_ema48852_LEGAL_s1/r8sel_diag_ema48852_LEGAL_s1.csv")}
SEED_ARM = {"s0": "R8_LEGAL", "s1": "R8_LEGAL_s1"}
sc = lambda lbl: rd(f"{W}/scores/{lbl}/{lbl}.csv")
W05 = {s: sc(f"r8sel_wise_l0p50_{a}") for s, a in SEED_ARM.items()}
ARMS = {g: {s: sc(f"r8sel_wmod_{g}_{a}") for s, a in SEED_ARM.items()} for g in ("PLAN", "PERC")}
toks = sorted(REF["s0"].index)
ti = B.load_inputs(R + "/inputs/navtrain_sample_inputs.json.gz").loc[toks]
cl = [str(x) for x in ti["log"]]
st = np.array([B.command_key(x) == "forward" for x in ti["command"]])
cfg = {**B.DEFAULT_CFG}
P = lambda df: round(100 * float(df.loc[toks]["PDMS"].mean()), 4)
res = {"id": "R8-WISEFT-MOD", "registered": "7ec285d (2026-10-21 15:11 Berlin, before any score)",
       "tier": "NavSim v1 open-loop PDMS, navtrain sample 2,089 tokens / 30 logs, Thor CPU scorer", "controls": {}, "arms": {}}
c = res["controls"]
c["K1_self_pair_zero"] = B.pair_record(REF["s0"], REF["s0"], toks, cl, cfg)["delta_x100"] == 0.0
c["K2_banked"] = {"theta05": [P(W05["s0"]), P(W05["s1"])], "EMA": [P(REF["s0"]), P(REF["s1"])],
                  "pass": P(W05["s0"]) == 76.3342 and P(W05["s1"]) == 76.4399 and P(REF["s0"]) == 63.3907 and P(REF["s1"]) == 63.0232}
c["K3_identical_tokens"] = all(sorted(d.index) == toks for g in ARMS for d in ARMS[g].values())
for g, arm in ARMS.items():
    d = {s: B.pair_record(arm[s], REF[s], toks, cl, cfg) for s in SEED_ARM}
    sh = {s: (P(arm[s]) - P(REF[s])) / (P(W05[s]) - P(REF[s])) for s in SEED_ARM}
    d0, d1 = d["s0"]["delta_x100"], d["s1"]["delta_x100"]
    adm = (d0 > 0 and d1 > 0 and d["s0"]["interval"].get("positive") and d["s1"]["interval"].get("positive") and min(d0, d1) > 2 * abs(d0 - d1))
    cells = {}
    for nm, m in (("straight", st), ("turn", ~st)):
        tt = [t for t, k in zip(toks, m) if k]; cc = [x for x, k in zip(cl, m) if k]
        cells[nm] = B.pair_record(arm["s0"], REF["s0"], tt, cc, cfg)["delta_x100"]
    res["arms"][g] = {"PDMS": {s: P(arm[s]) for s in SEED_ARM}, "minus_EMA48852": d, "share_of_theta05_gain": {s: round(v, 4) for s, v in sh.items()},
                      "admissible": bool(adm), "locus_condition": bool(adm and min(sh.values()) >= 0.75), "cells_minus_EMA48852_s0": cells,
                      "minus_theta05_s0": B.pair_record(arm["s0"], W05["s0"], toks, cl, cfg),
                      "subs_s0": {m: round(100 * float(arm["s0"].loc[toks][m].mean()), 4) for m in B.SUBS}}
L = [g for g in ARMS if res["arms"][g]["locus_condition"]]
hi = [g for g in ARMS if min(res["arms"][g]["share_of_theta05_gain"].values()) >= 0.75]
if len(L) == 1 and len(hi) == 1:
    v = f"LOCUS = {L[0]}"
elif len(hi) == 2:
    v = "REDUNDANT (both groups alone recover >= 0.75 of the gain)"
elif len(hi) == 0:
    v = "DISTRIBUTED (neither group alone recovers 0.75 of the gain)"
else:
    v = f"NO LOCUS ({hi[0]} reaches 0.75 but is not admissible)"
res["verdict"] = v
res["additivity_s0"] = round(P(ARMS["PLAN"]["s0"]) + P(ARMS["PERC"]["s0"]) - P(W05["s0"]) - P(REF["s0"]), 4)
c["all_pass"] = bool(c["K1_self_pair_zero"] and c["K2_banked"]["pass"] and c["K3_identical_tokens"])
if not c["all_pass"]:
    res["verdict"] = "INCOMPLETE (a control failed): " + v
json.dump(res, open(f"{W}/WISEMOD_READ.json", "w"), indent=1, default=str)
print("QQMOD " + json.dumps({"verdict": res["verdict"], "PDMS": {g: res["arms"][g]["PDMS"] for g in ARMS},
                             "share": {g: res["arms"][g]["share_of_theta05_gain"] for g in ARMS},
                             "additivity_s0": res["additivity_s0"], "controls": c["all_pass"]}) + " QQ")
