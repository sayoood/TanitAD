#!/usr/bin/env python3
"""Apply the REGISTERED R8-WISEFT-DECLAM rule (landed before any DECLAM arm was built; commit in PREREG_ACK_DECLAM.txt) to the pulled Thor
scores.  Writes WISEDECLAM_READ.json.  Derived from wisemod2_read.py (R8-WISEFT-MOD2).

Arms: the decoder-only blend at lambda 0.625 / 0.75 / 0.875 (everything outside core.decoder at EMA-48,852).  Per arm: X - EMA-48,852 paired
log-cluster bootstrap per seed; ADMISSIBLE iff positive (interval excluding 0) at both seeds with min margin > 2 x |d_s0 - d_s1| (the stage-1
rule); share_X(s) = (PDMS(X, s) - PDMS(EMA, s)) / (PDMS(theta_DEC, s) - PDMS(EMA, s)) printed (no locus rule).  Candidates = admissible arms by
mean navtrain PDMS.  Controls: EMA - EMA == 0; banked theta_DEC (76.3621 / 75.6804) and EMA (63.3907 / 63.0232) reproduce; identical tokens.
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
WD = {s: sc(f"r8sel_wmod2_DEC_{a}") for s, a in SEED_ARM.items()}
ARMS = {g: {s: sc(f"r8sel_wdecl_{g}_{a}") for s, a in SEED_ARM.items()} for g in ("DEC625", "DEC75", "DEC875")}
toks = sorted(REF["s0"].index)
ti = B.load_inputs(R + "/inputs/navtrain_sample_inputs.json.gz").loc[toks]
cl = [str(x) for x in ti["log"]]
st = np.array([B.command_key(x) == "forward" for x in ti["command"]])
cfg = {**B.DEFAULT_CFG}
P = lambda df: round(100 * float(df.loc[toks]["PDMS"].mean()), 4)
res = {"id": "R8-WISEFT-DECLAM", "registered": open(f"{W}/PREREG_ACK_DECLAM.txt", encoding="utf-8").read().strip()[:240],
       "tier": "NavSim v1 open-loop PDMS, navtrain sample 2,089 tokens / 30 logs, Thor CPU scorer", "controls": {}, "arms": {}}
c = res["controls"]
c["K1_self_pair_zero"] = B.pair_record(REF["s0"], REF["s0"], toks, cl, cfg)["delta_x100"] == 0.0
c["K2_banked"] = {"theta_DEC": [P(WD["s0"]), P(WD["s1"])], "EMA": [P(REF["s0"]), P(REF["s1"])],
                  "pass": P(WD["s0"]) == 76.3621 and P(WD["s1"]) == 75.6804 and P(REF["s0"]) == 63.3907 and P(REF["s1"]) == 63.0232}
c["K3_identical_tokens"] = all(sorted(d.index) == toks for g in ARMS for d in ARMS[g].values())
for g, arm in ARMS.items():
    d = {s: B.pair_record(arm[s], REF[s], toks, cl, cfg) for s in SEED_ARM}
    sh = {s: (P(arm[s]) - P(REF[s])) / (P(WD[s]) - P(REF[s])) for s in SEED_ARM}
    d0, d1 = d["s0"]["delta_x100"], d["s1"]["delta_x100"]
    adm = (d0 > 0 and d1 > 0 and d["s0"]["interval"].get("positive") and d["s1"]["interval"].get("positive") and min(d0, d1) > 2 * abs(d0 - d1))
    cells = {}
    for nm, m in (("straight", st), ("turn", ~st)):
        tt = [t for t, k in zip(toks, m) if k]; cc = [x for x, k in zip(cl, m) if k]
        cells[nm] = B.pair_record(arm["s0"], REF["s0"], tt, cc, cfg)["delta_x100"]
    res["arms"][g] = {"PDMS": {s: P(arm[s]) for s in SEED_ARM}, "PDMS_mean": round((P(arm["s0"]) + P(arm["s1"])) / 2, 4),
                      "minus_EMA48852": d, "share_of_theta_DEC_gain": {s: round(v, 4) for s, v in sh.items()}, "admissible": bool(adm),
                      "cells_minus_EMA48852_s0": cells, "minus_theta_DEC_s0": B.pair_record(arm["s0"], WD["s0"], toks, cl, cfg),
                      "subs_s0": {m: round(100 * float(arm["s0"].loc[toks][m].mean()), 4) for m in B.SUBS}}
res["admissible_candidates_by_navtrain_PDMS"] = sorted([g for g in ARMS if res["arms"][g]["admissible"]], key=lambda g: -res["arms"][g]["PDMS_mean"])
c["all_pass"] = bool(c["K1_self_pair_zero"] and c["K2_banked"]["pass"] and c["K3_identical_tokens"])
res["verdict"] = "READ" if c["all_pass"] else "INCOMPLETE (a control failed)"
json.dump(res, open(f"{W}/WISEDECLAM_READ.json", "w"), indent=1, default=str)
print("QQDECLAM " + json.dumps({"verdict": res["verdict"], "PDMS": {g: res["arms"][g]["PDMS"] for g in ARMS},
                                "share": {g: res["arms"][g]["share_of_theta_DEC_gain"] for g in ARMS},
                                "admissible": {g: res["arms"][g]["admissible"] for g in ARMS},
                                "candidates": res["admissible_candidates_by_navtrain_PDMS"], "controls": c["all_pass"]}) + " QQ")
