#!/usr/bin/env python3
"""Zero-training decomposition of the g5 gap on the banked RL-val dumps (MEASURED, RL-val selection tier, paired log-cluster bootstrap 10,000 / seed 20260927 via the
attribution package's own functions):   B - A (the g5 read)  =  (B - base)  -  (A - base).   For R0, R0b, C0 at every dumped update.
Question it answers: is the TTC deficit against the control EPO's damage relative to the START, or the control's own gain that the base-anchored B does not share?

    python ab_vs_base.py --dl C:/Users/Admin/rls1b_dl --out ab_vs_base.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATT = os.path.join(HERE, "..", "2026-10-14-rls1b-attribution")
sys.path.insert(0, ATT)
import attrib_rls1b as AT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default="C:/Users/Admin/rls1b_dl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X = os.path.join(a.dl, "data", "x")
    kv, tgv, dv, ckv, _ = AT.load_side(os.path.join(X, "pdm_val", "pdm_val"), os.path.join(X, "terms_val", "terms_val"))
    sv = AT.comps(tgv)
    n = len(kv)
    ar = np.arange(n)
    res = {"tool": "ab_vs_base/1", "class": "MEASURED (banked RL-val dumps; selection tier; one training seed per arm; another draw of LOGS only)", "n_sets": n, "rows": {}}
    base = AT.load_scores_aligned(os.path.join(a.dl, "r0", "final", "scores_step0_base.npz"), kv, ckv)
    ib = AT.picks(base)
    for run, ups in (("c0", (250, 500, 1000)), ("r0", (250, 500, 1000)), ("r0b", (250, 500))):
        for u in ups:
            ia = AT.picks(AT.load_scores_aligned(os.path.join(a.dl, run, "final", f"scores_update{u}_A.npz"), kv, ckv))
            ibb = AT.picks(AT.load_scores_aligned(os.path.join(a.dl, run, "final", f"scores_update{u}_B.npz"), kv, ckv))
            row = {}
            for nm, (i, j) in {"B_minus_A": (ibb, ia), "B_minus_base": (ibb, ib), "A_minus_base": (ia, ib)}.items():
                row[nm] = {q: AT.boot_delta(sv[q.lower()][ar, i], sv[q.lower()][ar, j], kv) for q in ("TTC", "NC", "EP", "PDMS")}
            res["rows"][f"{run}_u{u}"] = row
    json.dump(res, open(a.out, "w"), indent=1)
    for k, row in res["rows"].items():
        f = lambda nm, q: "%+.2f [%+.2f,%+.2f]" % (row[nm][q]["mean"], row[nm][q]["ci95"][0], row[nm][q]["ci95"][1])  # noqa: E731
        print(f"{k:10s} TTC: B-A {f('B_minus_A','TTC')} | B-base {f('B_minus_base','TTC')} | A-base {f('A_minus_base','TTC')}   PDMS: B-A {f('B_minus_A','PDMS')} | B-base {f('B_minus_base','PDMS')} | A-base {f('A_minus_base','PDMS')}")
    print("ZZABBASE_OK")


if __name__ == "__main__":
    main()
