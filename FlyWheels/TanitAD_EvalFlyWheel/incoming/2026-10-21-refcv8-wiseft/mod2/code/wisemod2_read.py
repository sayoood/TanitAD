#!/usr/bin/env python3
"""Apply the REGISTERED R8-WISEFT-MOD2 rule (fc67bf6, before any MOD2 arm was built) to the pulled Thor scores.  Writes WISEMOD2_READ.json.
Derived from wisemod_read.py (R8-WISEFT-MOD): the reference gain is now theta_PLAN's, not theta_0.5's.
share_X(s) = (PDMS(X, s) - PDMS(EMA-48,852, s)) / (PDMS(theta_PLAN, s) - PDMS(EMA-48,852, s)), X in {DEC, TAC}.  X is the SUB-LOCUS iff share >= 0.75
at both seeds AND X - EMA-48,852 is admissible (separated positive at both seeds, min margin > 2 x |d_s0 - d_s1|).  Both < 0.75 -> DISTRIBUTED
within PLAN; both >= 0.75 -> REDUNDANT.  PLAN75: admissible or not, share printed.  Controls: EMA - EMA == 0; banked theta_PLAN (73.9306 / 73.8614)
and EMA (63.3907 / 63.0232) reproduce; identical tokens.  Beside: DEC + TAC additivity vs theta_PLAN + EMA (s0).
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
WP = {s: sc(f"r8sel_wmod_PLAN_{a}") for s, a in SEED_ARM.items()}
W05 = {s: sc(f"r8sel_wise_l0p50_{a}") for s, a in SEED_ARM.items()}
ARMS = {g: {s: sc(f"r8sel_wmod2_{g}_{a}") for s, a in SEED_ARM.items()} for g in ("DEC", "TAC", "PLAN75")}
toks = sorted(REF["s0"].index)
ti = B.load_inputs(R + "/inputs/navtrain_sample_inputs.json.gz").loc[toks]
cl = [str(x) for x in ti["log"]]
st = np.array([B.command_key(x) == "forward" for x in ti["command"]])
cfg = {**B.DEFAULT_CFG}
P = lambda df: round(100 * float(df.loc[toks]["PDMS"].mean()), 4)
res = {"id": "R8-WISEFT-MOD2", "registered": "fc67bf6 (2026-10-21 17:07 Berlin, before any MOD2 arm was built)",
       "tier": "NavSim v1 open-loop PDMS, navtrain sample 2,089 tokens / 30 logs, Thor CPU scorer", "controls": {}, "arms": {}}
c = res["controls"]
c["K1_self_pair_zero"] = B.pair_record(REF["s0"], REF["s0"], toks, cl, cfg)["delta_x100"] == 0.0
c["K2_banked"] = {"theta_PLAN": [P(WP["s0"]), P(WP["s1"])], "EMA": [P(REF["s0"]), P(REF["s1"])],
                  "pass": P(WP["s0"]) == 73.9306 and P(WP["s1"]) == 73.8614 and P(REF["s0"]) == 63.3907 and P(REF["s1"]) == 63.0232}
c["K3_identical_tokens"] = all(sorted(d.index) == toks for g in ARMS for d in ARMS[g].values())
for g, arm in ARMS.items():
    d = {s: B.pair_record(arm[s], REF[s], toks, cl, cfg) for s in SEED_ARM}
    sh = {s: (P(arm[s]) - P(REF[s])) / (P(WP[s]) - P(REF[s])) for s in SEED_ARM}
    d0, d1 = d["s0"]["delta_x100"], d["s1"]["delta_x100"]
    adm = (d0 > 0 and d1 > 0 and d["s0"]["interval"].get("positive") and d["s1"]["interval"].get("positive") and min(d0, d1) > 2 * abs(d0 - d1))
    cells = {}
    for nm, m in (("straight", st), ("turn", ~st)):
        tt = [t for t, k in zip(toks, m) if k]; cc = [x for x, k in zip(cl, m) if k]
        cells[nm] = B.pair_record(arm["s0"], REF["s0"], tt, cc, cfg)["delta_x100"]
    res["arms"][g] = {"PDMS": {s: P(arm[s]) for s in SEED_ARM}, "PDMS_mean": round((P(arm["s0"]) + P(arm["s1"])) / 2, 4),
                      "minus_EMA48852": d, "share_of_theta_PLAN_gain": {s: round(v, 4) for s, v in sh.items()},
                      "admissible": bool(adm), "sub_locus_condition": bool(adm and min(sh.values()) >= 0.75 and g in ("DEC", "TAC")),
                      "cells_minus_EMA48852_s0": cells, "minus_theta_PLAN_s0": B.pair_record(arm["s0"], WP["s0"], toks, cl, cfg),
                      "minus_theta05_s0": B.pair_record(arm["s0"], W05["s0"], toks, cl, cfg),
                      "subs_s0": {m: round(100 * float(arm["s0"].loc[toks][m].mean()), 4) for m in B.SUBS}}
L = [g for g in ("DEC", "TAC") if res["arms"][g]["sub_locus_condition"]]
hi = [g for g in ("DEC", "TAC") if min(res["arms"][g]["share_of_theta_PLAN_gain"].values()) >= 0.75]
if len(L) == 1 and len(hi) == 1:
    v = f"SUB-LOCUS = {L[0]}"
elif len(hi) == 2:
    v = "REDUNDANT (both sub-stacks alone recover >= 0.75 of theta_PLAN's gain)"
elif len(hi) == 0:
    v = "DISTRIBUTED within PLAN (neither sub-stack alone recovers 0.75 of theta_PLAN's gain)"
else:
    v = f"NO SUB-LOCUS ({hi[0]} reaches 0.75 but is not admissible)"
res["verdict"] = v
res["admissible_candidates_by_navtrain_PDMS"] = sorted([g for g in ARMS if res["arms"][g]["admissible"]], key=lambda g: -res["arms"][g]["PDMS_mean"])
res["additivity_s0"] = round(P(ARMS["DEC"]["s0"]) + P(ARMS["TAC"]["s0"]) - P(WP["s0"]) - P(REF["s0"]), 4)
c["all_pass"] = bool(c["K1_self_pair_zero"] and c["K2_banked"]["pass"] and c["K3_identical_tokens"])
if not c["all_pass"]:
    res["verdict"] = "INCOMPLETE (a control failed): " + v
json.dump(res, open(f"{W}/WISEMOD2_READ.json", "w"), indent=1, default=str)
print("QQMOD2 " + json.dumps({"verdict": res["verdict"], "PDMS": {g: res["arms"][g]["PDMS"] for g in ARMS},
                              "share": {g: res["arms"][g]["share_of_theta_PLAN_gain"] for g in ARMS},
                              "admissible": {g: res["arms"][g]["admissible"] for g in ARMS},
                              "candidates": res["admissible_candidates_by_navtrain_PDMS"],
                              "additivity_s0": res["additivity_s0"], "controls": c["all_pass"]}) + " QQ")
