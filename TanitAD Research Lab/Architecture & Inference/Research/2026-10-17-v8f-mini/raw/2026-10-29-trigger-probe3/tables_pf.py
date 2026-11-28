#!/usr/bin/env python3
"""tables_pf.py -- the POST-FREEZE sensitivity table (raw/tables_pf.txt).  Reads raw/oofg_<pop>_pf_linear.npz; paired against the frozen G4+E / G8+E and probe 1's banked E / V+E.  Cannot move the verdict.
python -I tables_pf.py"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import g_lib as G  # noqa: E402
import numpy as np

L = G.L
LINES = []


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def main():
    pr("# tables_pf -- POST-FREEZE sensitivity arms (linear; extended C grid {1e-6..10}); NOT in PREREG_FROZEN.md; cannot move the verdict")
    for pop in G.POPS:
        p = f"{G.RAW}/oofg_{pop}_pf_linear.npz"
        if not os.path.exists(p):
            pr(f"\n## {pop}: no post-freeze arms banked")
            continue
        d = G.load_pop(pop)
        y, clip, outer = d["y"], d["clip"], d["outer"]
        cnt = L.boot_counts(clip)
        bk = G.banked(pop, "linear")
        o = dict(np.load(p))
        fr = dict(np.load(f"{G.RAW}/oofg_{pop}_linear.npz"))
        S = {"E": bk["E"], "V+E": bk["V+E"], "G4+E(frozen)": fr["G4+E"], "G8+E(frozen)": fr["G8+E"]}
        S.update({k.replace("__", ":"): v for k, v in o.items()})
        pr(f"\n## {pop.upper()} (fold-mean AUC [CI] | pooled)")
        for k, v in S.items():
            r = G.rec(v, y, clip, outer, cnt)
            pr(f"  {k:16s} {r['fold']['auc']:.3f} [{r['fold']['lo']:.3f}, {r['fold']['hi']:.3f}] | {r['pooled']['auc']:.3f}")
        pairs = [("PF:G8+E", "PF:G4+E"), ("PF:G8k4+E", "PF:G4+E"), ("PF:G8k4+E", "PF:G4k4+E"), ("PF:G8+Ex64", "PF:G4+Ex64"), ("PF:G8w64+E", "PF:G4w64+E"),
                 ("PF:G4w64+E", "V+E"), ("PF:G8w64+E", "V+E"), ("PF:G4w64+E", "E"), ("PF:G8w64+E", "E"), ("PF:G4+Ex64", "E"), ("PF:G8+Ex64", "E")]
        pr("  paired dAUC (fold-mean [CI] | pooled)")
        for a_, b_ in pairs:
            if a_ in S and b_ in S:
                r = G.paired(S[a_], S[b_], y, clip, outer, cnt)
                pr(f"  {a_:12s} - {b_:12s} {r['fold']['delta']:+.4f} [{r['fold']['lo']:+.4f}, {r['fold']['hi']:+.4f}] | {r['pooled']['delta']:+.4f}")
    open(f"{G.RAW}/tables_pf.txt", "w", encoding="utf-8").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
