#!/usr/bin/env python3
"""Diagnostic (no training, not a decision input): what do FULL's EPO-moved SAFETY logits do to the selection?  FULL lam 1's state read (i) through its own logits and (ii) with its
NC/DAC/TTC rows swapped for the base's (banked_P.pkl view 'SH_F'), each vs FULL lam 0 (own).  (i) - (ii) = the effect of the safety-row moves alone on the pick / policy.  -> diag_full_safety_rows.json"""
import json, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_summarize as S
bk = pickle.load(open(os.path.join(HERE, "out", "banked_P.pkl"), "rb"))
out = {}
for k in (25, 50):
    own1 = {p: bk[(1.0, k)][p]["own"] for p in S.POOLS}
    swp1 = {p: bk[(1.0, k)][p]["SH_F"] for p in S.POOLS}
    own0 = {p: bk[(0.0, k)][p]["own"] for p in S.POOLS}
    swp0 = {p: bk[(0.0, k)][p]["SH_F"] for p in S.POOLS}
    out[f"K{k}"] = {"FULL1_own_vs_FULL0": S.prox(own1, own0), "FULL1_baseSafety_vs_FULL0": S.prox(swp1, own0),
                    "FULL1_own_vs_FULL1_baseSafety": S.prox(own1, swp1), "FULL0_baseSafety_vs_FULL0": S.prox(swp0, own0)}
    for nm, v in out[f"K{k}"].items():
        print(k, f"{nm:30s}", {m: round(v[m]["pooled"], 2) for m in ("P1_TTC", "P2_TTC", "P2_NC", "P2_DAC", "P1_EP", "P1_PDMS")})
json.dump(out, open(os.path.join(HERE, "diag_full_safety_rows.json"), "w"), indent=1)
