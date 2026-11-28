#!/usr/bin/env python3
"""R8-WISEFT STAGE 2 reader -- the registered navtest rule (PREREG_R8_WISEFT.json, landed e8911b1) for theta_lambda*.

    python wiseft_read_navtest.py <TAG>      (TAG = the stage-1 lambda* tag, e.g. 0p50)

WORKS iff theta - EMA-48,852 R8_LEGAL > 0 with bars8 compact_pair 'positive' (log_name AND nuplan_drive) at BOTH inference seeds and
min(d_s0, d_s1) > 2 x |d_s0 - d_s1|; FAILED if the point is <= 0 at either seed; otherwise NOT PROVEN.
Controls (known values): K1 EMA - EMA == 0; K2 the banked EMA R8_LEGAL reproduces 57.84; K3 theta's PRIOR_ha0p equals the EMA read's
PRIOR_ha0p on every token (the prior is a pose-only construction, so a weight blend cannot move it); K4 identical 12,146 tokens.
"""
import importlib.util
import json
import sys

import numpy as np

TAG = sys.argv[1]
W = "C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft"
NR = "C:/Users/Admin/refcv8_pf/navtest_read/raw"
INP = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
SG_MAP = "sg-one-north"
spec = importlib.util.spec_from_file_location("b8s2", "C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/"
                                              "incoming/2026-10-19-refcv8-navsim/code/navsim_bars8.py")
B = importlib.util.module_from_spec(spec)
sys.modules["b8s2"] = B
spec.loader.exec_module(B)
rd = B.read_scores_csv
wt = lambda a: rd(f"{W}/stage2/scores_navtest/r8nt_w{TAG}_{a}/r8nt_w{TAG}_{a}.csv")
em = lambda a: rd(f"{NR}/full/scores_navtest/r8nt_s48852e_{a}/r8nt_s48852e_{a}.csv")
TH = {"s0": wt("R8_LEGAL"), "s1": wt("R8_LEGAL_s1")}
EM = {"s0": em("R8_LEGAL"), "s1": em("R8_LEGAL_s1")}
THP, EMP = wt("PRIOR_ha0p"), em("PRIOR_ha0p")
STOP, R7 = rd(f"{NR}/controls/STOP_navtest.csv"), rd(f"{NR}/controls/r7thor_s50400_R7_VMAXOFF.csv")
toks = sorted(EM["s0"].index)
ti = B.load_inputs(INP).loc[toks]
cl = [str(x) for x in ti["log"]]
st = np.array([B.command_key(x) == "forward" for x in ti["command"]])
sg = (ti["map_name"] == SG_MAP).to_numpy()
cfg = {**B.DEFAULT_CFG}
P = lambda df: round(100 * float(df.loc[toks]["PDMS"].mean()), 4)
k4 = []


def pr(a, b, mask=None):
    tt = toks if mask is None else [t for t, k in zip(toks, mask) if k]
    cc = cl if mask is None else [x for x, k in zip(cl, mask) if k]
    r = B.pair_record(a, b, tt, cc, cfg)
    k4.append(r["wins"] + r["ties"] + r["losses"] == r["n"])
    return r


res = {"id": "R8-WISEFT stage 2", "registered": "e8911b1", "tag": TAG,
       "tier": "NavSim v1.1 open-loop PDMS, navtest 12,146 tokens, Thor CUDA roll + Thor CPU scorer", "controls": {}}
c = res["controls"]
c["K1_self_pair_zero"] = B.pair_record(EM["s0"], EM["s0"], toks, cl, cfg)["delta_x100"] == 0.0
c["K2_EMA_LEGAL"] = {"value": P(EM["s0"]), "pass": abs(P(EM["s0"]) - 57.84) < 0.005}
dp = np.abs(THP.loc[toks]["PDMS"].to_numpy(float) - EMP.loc[toks]["PDMS"].to_numpy(float))
c["K3_prior_identical"] = {"max_abs": float(dp.max()), "pass": float(dp.max()) == 0.0}
c["K4_identical_tokens"] = all(sorted(d.index) == toks for d in (*TH.values(), THP))
d = {s: pr(TH[s], EM[s]) for s in ("s0", "s1")}
d0, d1 = d["s0"]["delta_x100"], d["s1"]["delta_x100"]
floor = abs(d0 - d1)
if d0 <= 0 or d1 <= 0:
    v = "FAILED"
elif d["s0"]["interval"].get("positive") and d["s1"]["interval"].get("positive") and min(d0, d1) > 2 * floor:
    v = "WORKS"
else:
    v = "NOT PROVEN"
res.update({"PDMS": {"theta_s0": P(TH["s0"]), "theta_s1": P(TH["s1"]), "EMA_s0": P(EM["s0"]), "EMA_s1": P(EM["s1"]),
                     "PRIOR": P(EMP), "STOP": P(STOP), "refcv7_50400": P(R7)},
            "theta_minus_EMA": d, "floor": round(floor, 4), "verdict": v,
            "beside": {"theta_minus_PRIOR_s0": pr(TH["s0"], EMP), "theta_minus_STOP_s0": pr(TH["s0"], STOP),
                       "theta_minus_refcv7_s0": pr(TH["s0"], R7),
                       "cells_theta_minus_EMA_s0": {"straight": pr(TH["s0"], EM["s0"], st)["delta_x100"],
                                                    "turn": pr(TH["s0"], EM["s0"], ~st)["delta_x100"],
                                                    "singapore": pr(TH["s0"], EM["s0"], sg)["delta_x100"],
                                                    "us": pr(TH["s0"], EM["s0"], ~sg)["delta_x100"]},
                       "subs_theta_s0": {m: round(100 * float(TH["s0"].loc[toks][m].mean()), 4) for m in B.SUBS},
                       "subs_EMA_s0": {m: round(100 * float(EM["s0"].loc[toks][m].mean()), 4) for m in B.SUBS},
                       "G_ARC_banked": 65.52}})
c["K5_counts"] = all(k4)
c["all_pass"] = bool(c["K1_self_pair_zero"] and c["K2_EMA_LEGAL"]["pass"] and c["K3_prior_identical"]["pass"]
                     and c["K4_identical_tokens"] and c["K5_counts"])
if not c["all_pass"]:
    res["verdict"] = "INCOMPLETE (a control failed): " + v
json.dump(res, open(f"{W}/stage2/WISEFT_NAVTEST_READ.json", "w"), indent=1, default=str)
print("QQWISE2 " + json.dumps({"verdict": res["verdict"], "PDMS": res["PDMS"], "d_s0": d0, "d_s1": d1,
                               "lo_s0": d["s0"]["interval"].get("lo"), "hi_s0": d["s0"]["interval"].get("hi"),
                               "controls": c["all_pass"]}) + " QQ")
