#!/usr/bin/env python3
"""read_levers -- score table + the PRE-FIXED reading rule for the 18 navtrain lever arms (PRIOR_GAP_DIAG.md §Thor queue).

Rule (fixed in PRIOR_GAP_DIAG.md before any lever arm was scored):
  * R8PATH_PRIORPROG >= LEGAL + 5 and PRIORPATH_R8PROG within 2 of PRIOR  => the loss is LONGITUDINAL (speed profile)
  * PRIORPATH_R8PROG <= PRIOR - 5 and R8PATH_PRIORPROG < LEGAL + 2        => the loss is the PATH
  * R8PATH_MINPROG must beat LEGAL and R8PATH_MAXPROG, else lever 2 (longitudinal cap) is dead
Read per inference seed (s0 vs LEGAL_s0; s1 arms vs LEGAL_s1).  POST-HOC / DESCRIPTIVE navtrain, Las Vegas only, one training seed.
Every arm's CSV is admitted only beside a counts.json with status PASS and the full 2,089-token set.  Paired log-cluster bootstrap
(the bars8 estimator, imported) against PRIOR and LEGAL.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import importlib.util

BARS8 = "C:/Users/Admin/refcv8_pf/navsim_ckpt/pkg/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/navsim_bars8.py"
RUN = "C:/Users/Admin/refcv8_pf/navsim_ckpt/run"
INPUTS = RUN + "/inputs/navtrain_sample_inputs.json.gz"
REF = {"PRIOR": RUN + "/navtrain_sel/scores/r8sel_ref_PRIOR_ha0p/r8sel_ref_PRIOR_ha0p.csv",
       "LEGAL_s0": RUN + "/navtrain_sel/diag/scores/r8sel_diag_ema48852_LEGAL_s0/r8sel_diag_ema48852_LEGAL_s0.csv",
       "LEGAL_s1": RUN + "/navtrain_sel/diag/scores/r8sel_diag_ema48852_LEGAL_s1/r8sel_diag_ema48852_LEGAL_s1.csv",
       "RT_s0": RUN + "/navtrain_sel/scores/r8sel_ema48852_s0/r8sel_ema48852_s0.csv",
       "RT_s1": RUN + "/navtrain_sel/scores/r8sel_ema48852_s1/r8sel_ema48852_s1.csv"}


def main(lever_scores_dir, out):
    spec = importlib.util.spec_from_file_location("b8l", BARS8)
    B = importlib.util.module_from_spec(spec)
    sys.modules["b8l"] = B
    spec.loader.exec_module(B)
    cfg = {**B.DEFAULT_CFG}
    ref = {k: B.read_scores_csv(p) for k, p in REF.items()}
    toks = sorted(ref["PRIOR"].index)
    assert len(toks) == 2089 and all(sorted(df.index) == toks for df in ref.values()), "reference token sets"
    ti = B.load_inputs(INPUTS).loc[toks]
    cl = [str(x) for x in ti["log"]]
    arms, refused = {}, {}
    for cj in sorted(glob.glob(os.path.join(lever_scores_dir, "r8sel_lever_*", "*.counts.json"))):
        name = os.path.basename(cj)[len("r8sel_lever_"):-len(".counts.json")]
        c = json.load(open(cj, encoding="utf-8"))
        if c.get("status") != "PASS":
            refused[name] = f"counts status {c.get('status')}"
            continue
        df = B.read_scores_csv(cj[:-len(".counts.json")] + ".csv")
        if sorted(df.index) != toks:
            refused[name] = f"token set {len(df)} != 2089"
            continue
        arms[name] = df
    res = {"label": "POST-HOC / DESCRIPTIVE navtrain lever arms (2,089 tokens, 30 Las Vegas logs, EMA-48,852, one training seed)",
           "tier": "NavSim v1 open-loop PDMS (T1-family)", "n_arms": len(arms), "refused": refused,
           "ref_PDMS_x100": {k: round(100 * float(v["PDMS"].mean()), 4) for k, v in ref.items()}, "arms": {}, "rule": {}}
    for name, df in arms.items():
        s = "s1" if name.endswith("_s1") else "s0"
        rec = {m: round(100 * float(df[m].mean()), 4) for m in B.SUBS + ("PDMS",)}
        rec["minus_PRIOR"] = B.pair_record(df, ref["PRIOR"], toks, cl, cfg)
        rec["minus_LEGAL"] = B.pair_record(df, ref[f"LEGAL_{s}"], toks, cl, cfg)
        res["arms"][name] = rec
    for s, suf in (("s0", ""), ("s1", "_s1")):
        g = lambda n: res["arms"].get(n + suf, {}).get("PDMS")
        L, P = res["ref_PDMS_x100"][f"LEGAL_{s}"], res["ref_PDMS_x100"]["PRIOR"]
        rp, pr, mn, mx = g("R8PATH_PRIORPROG"), g("PRIORPATH_R8PROG"), g("R8PATH_MINPROG"), g("R8PATH_MAXPROG")
        if None in (rp, pr):
            res["rule"][s] = {"attribution": "UNAVAILABLE (arm missing)"}
            continue
        if rp >= L + 5 and abs(pr - P) <= 2:
            att = "LONGITUDINAL"
        elif pr <= P - 5 and rp < L + 2:
            att = "PATH"
        else:
            att = "NEITHER CLAUSE (mixed / unresolved)"
        lever2 = None if None in (mn, mx) else ("ALIVE" if (mn > L and mn > mx) else "DEAD")
        res["rule"][s] = {"attribution": att, "R8PATH_PRIORPROG": rp, "PRIORPATH_R8PROG": pr, "LEGAL": L, "PRIOR": P,
                          "R8PATH_MINPROG": mn, "R8PATH_MAXPROG": mx, "lever2_longitudinal_cap": lever2}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print("QQLEVERS " + json.dumps({"n_arms": len(arms), "refused": refused, "rule": res["rule"],
                                    "PDMS": {k: v["PDMS"] for k, v in res["arms"].items()}}) + " QQ")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
