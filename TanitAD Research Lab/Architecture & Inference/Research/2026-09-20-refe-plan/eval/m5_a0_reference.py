#!/usr/bin/env python3
"""Measure 5 -- an INDEPENDENT reference for arm A0 (snapshot 015 unchanged), computed BEFORE the M5 eval exists.

WHY: m5_finetune_eval.py's A0 row comes from the scorer re-run on the NEW trunk cache. The same quantities can be
derived from artifacts banked earlier by a DIFFERENT code path -- the slow-copy probe's GPU dump
(slow_copies/gpu_dump.npz: REFePlanner's full forward, masked route, `m` rows = the 64 [seeing the 64 only], the 0.75x
copies of all 64 sources [each seeing the 64 and itself], the 0.5x copies, STOP) and the E-6 table -- with the pair
concordance re-implemented here as explicit loops (NOT m5_finetune_eval.concordance). The M5 eval's A0 must reproduce
these values; a disagreement is a defect in one of the two paths and blocks the readout.

    python eval/m5_a0_reference.py            -> eval/raw/m5_effectiveness/a0_reference.json
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
NAV = "D:/Projects/TanitAD/data/refe_navtest"
TABLE = f"{NAV}/proptable/sub200_ep015/table.npz"
RANKS = f"{NAV}/proptable/sub200_ep015/slow_copies/ranks.npz"
GDUMP = f"{NAV}/proptable/sub200_ep015/slow_copies/gpu_dump.npz"
OUT = os.path.join(HERE, "raw", "m5_effectiveness", "a0_reference.json")


def csv_pdms(p):
    out = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["token"] != "average":
                out[r["token"]] = float(r["score"]) if r["valid"] == "True" else float("nan")
    return out


def conc_loop(agg, pd, involved):
    """explicit-loop pair concordance: pairs with >= 1 involved member (None = all), harness not tied (|d| > 1e-9);
    same order 1, aggregate tie 0.5, opposite 0"""
    num = den = 0.0
    m = len(pd)
    for i in range(m):
        if not np.isfinite(pd[i]):
            continue
        for j in range(i + 1, m):
            if not np.isfinite(pd[j]):
                continue
            if involved is not None and not (involved[i] or involved[j]):
                continue
            dh = pd[i] - pd[j]
            if abs(dh) <= 1e-9:
                continue
            da = agg[i] - agg[j]
            den += 1
            num += 1.0 if da * dh > 0 else (0.5 if da == 0 else 0.0)
    return num / den if den else float("nan")


def main() -> int:
    import onpolicy_label_v4 as L4                       # the planner's own navsim_v1 aggregate (the shim)
    T = np.load(TABLE)
    R = np.load(RANKS)
    G = np.load(GDUMP, allow_pickle=False)
    tt = [str(t) for t in T["token"]]
    if [str(t) for t in G["token"]] != tt or [str(t) for t in R["token"]] != tt:
        raise SystemExit("token order differs between the table, the ranks and the GPU dump")
    n = len(tt)
    hr = [csv_pdms(f"{NAV}/score/refe_sub200_ep015_f075_r{r:02d}/refe_sub200_ep015_f075_r{r:02d}.csv") for r in range(8)]
    hs = csv_pdms(f"{NAV}/score/refe_sub200_ep015_stopzeros/refe_sub200_ep015_stopzeros.csv")
    m = G["m"].astype(np.float32)                                             # [n, 193, 6]
    top = R["top"][:, :8]
    ar = np.arange(n)[:, None]
    L73 = np.concatenate([m[:, :64], m[ar, 64 + top], m[:, 192:193]], 1)     # [n, 73, 6]
    agg73 = L4.v1_aggregate(L73.reshape(-1, 6)).reshape(n, 73)
    agg64 = L4.v1_aggregate(T["logits"].astype(np.float32).reshape(-1, 6)).reshape(n, 64)
    pd73 = np.concatenate([T["pdms"].astype(np.float64),
                           np.array([[hr[r].get(t, np.nan) for r in range(8)] + [hs.get(t, np.nan)] for t in tt])], 1)
    inv = np.array([False] * 64 + [True] * 9)
    e1 = np.array([conc_loop(agg73[i], pd73[i], inv) for i in range(n)])
    e3a = np.array([conc_loop(agg64[i], pd73[i, :64], None) for i in range(n)])
    k64 = agg64.argmax(1)
    k73 = agg73.argmax(1)
    res = {"_label": "INDEPENDENT A0 reference (slow-copy probe GPU dump + E-6 table, loop concordance); the M5 eval's "
                     "A0 must reproduce it", "n_tokens": n,
           "id_masked64_vs_s64_max_abs": float(np.abs(m[:, :64].astype(np.float64) - G["s64"].astype(np.float64)).max()),
           "s64_vs_table_logits_max_abs": float(np.abs(G["s64"].astype(np.float64) - T["logits"].astype(np.float64)).max()),
           "pick64_equals_table": int((k64 == T["pick"]).sum()),
           "E1_masked_token_mean": float(np.nanmean(e1)), "E3a_token_mean": float(np.nanmean(e3a)),
           "E3b_pdms_x100": float(100 * T["pdms"][np.arange(n), k64].mean()),
           "E2_pdms_x100": float(100 * np.nanmean(pd73[np.arange(n), k73])),
           "E2_mix": {"original": int((k73 < 64).sum()), "copy_075": int(((k73 >= 64) & (k73 < 72)).sum()),
                      "stop": int((k73 == 72).sum())},
           "per_token": {"E1_masked": [round(float(x), 6) for x in e1], "E3a": [round(float(x), 6) for x in e3a]},
           "harness_missing_extra": int(np.isnan(pd73[:, 64:]).sum())}
    # POST HOC, NOT GATING: the same A0 quantities against the REPAIRED truth (Amendment 7; m5_repaired_extras.py)
    TR = f"{NAV}/proptable/sub200_ep015_repaired/table.npz"
    rcsv = [f"{NAV}/score/refe_sub200_ep015_f075rep_r{r:02d}/refe_sub200_ep015_f075rep_r{r:02d}.csv" for r in range(8)]
    if os.path.exists(TR) and all(os.path.exists(x) for x in rcsv):
        Tr = np.load(TR)
        hrr = [csv_pdms(x) for x in rcsv]
        pr = np.concatenate([Tr["pdms"].astype(np.float64),
                             np.array([[hrr[r].get(t, np.nan) for r in range(8)] + [hs.get(t, np.nan)] for t in tt])], 1)
        e1r = np.array([conc_loop(agg73[i], pr[i], inv) for i in range(n)])
        e3ar = np.array([conc_loop(agg64[i], pr[i, :64], None) for i in range(n)])
        res["posthoc_repaired_truth"] = {
            "E1_masked_token_mean": float(np.nanmean(e1r)), "E3a_token_mean": float(np.nanmean(e3ar)),
            "E3b_pdms_x100": float(100 * pr[np.arange(n), k64].mean()),
            "E2_pdms_x100": float(100 * np.nanmean(pr[np.arange(n), k73])),
            "copies_mean_pdms_x100_unrepaired_vs_repaired": [float(100 * np.nanmean(pd73[:, 64:72])),
                                                              float(100 * np.nanmean(pr[:, 64:72]))],
            "originals_mean_pdms_x100_unrepaired_vs_repaired": [float(100 * np.nanmean(pd73[:, :64])),
                                                                 float(100 * np.nanmean(pr[:, :64]))],
            "stop_mean_pdms_x100": float(100 * np.nanmean(pr[:, 72])), "harness_missing": int(np.isnan(pr).sum())}
        # the TRUTH's own verdict on slowing: each 0.75x copy vs ITS source original (top-8), both truths
        src_u, src_r = T["pdms"][ar, top].astype(np.float64), Tr["pdms"][ar, top].astype(np.float64)
        cu, cr = pd73[:, 64:72], pr[:, 64:72]
        res["posthoc_repaired_truth"]["copy_minus_its_source_x100"] = {
            "unrepaired": {"mean": float(100 * np.nanmean(cu - src_u)), "frac_copy_better": float(np.mean(cu - src_u > 1e-9)),
                           "frac_copy_worse": float(np.mean(cu - src_u < -1e-9)),
                           "sources_mean": float(100 * src_u.mean()), "copies_mean": float(100 * np.nanmean(cu))},
            "repaired": {"mean": float(100 * np.nanmean(cr - src_r)), "frac_copy_better": float(np.mean(cr - src_r > 1e-9)),
                         "frac_copy_worse": float(np.mean(cr - src_r < -1e-9)),
                         "sources_mean": float(100 * src_r.mean()), "copies_mean": float(100 * np.nanmean(cr))},
            "n_pairs": int(cu.size)}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=0)
    print(json.dumps({k: v for k, v in res.items() if k != "per_token"}))
    print("ZZM5_A0REF_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
