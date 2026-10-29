#!/usr/bin/env python3
"""strata_g.py -- PREREG 5.1: TRAIN strata (traffic light, lead vehicle) for G4+E / G8+E on the LINEAR out-of-fold scores, with probe 1's banked E and V+E.  Same strata flags and the same 'too small' rule as strata2.py
(>= 60 windows, >= 30 per class, >= 20 clips in the SAMPLE).  Primary metric = fold-mean AUC, pooled beside.   Writes raw/results_strata_g.json.   CUDA_VISIBLE_DEVICES=-1  python -I strata_g.py"""
from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import g_lib as G  # noqa: E402
import numpy as np

L = G.L
STR = ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart")
PAIRS = (("G4+E", "E"), ("G8+E", "E"), ("G8+E", "G4+E"), ("V+E", "E"))


def scores(pop):
    bk = G.banked(pop, "linear")
    o = np.load(f"{G.RAW}/oofg_{pop}_linear.npz")
    return {"E": bk["E"], "V+E": bk["V+E"], "G4+E": o["G4+E"], "G8+E": o["G8+E"]}


def main():
    old = json.load(open(f"{G.OLDRAW}/results_strata.json"))
    out = {"note": "TRAIN only; TL_* are CLIP-level and VLM-derived (vlm-cot, disputed, untimed); lead_depart reads the future (a stratum, never a feature); GO rates (full population) are probe 1's", "pops": {}}
    for pop in G.POPS:
        d = G.load_pop(pop)
        y, clip, outer = d["y"], d["clip"], d["outer"]
        sc = scores(pop)
        P = {}
        for s in STR:
            ms = d[s].astype(bool)
            r = {"n_sample": int(ms.sum()), "n_sample_clips": int(len(np.unique(clip[ms]))) if ms.any() else 0, "n_sample_go": int(y[ms].sum()) if ms.any() else 0,
                 "go_rate_full": old["pops"][pop]["strata"][s].get("go_rate_full")}
            ok = ms.sum() >= 60 and y[ms].sum() >= 30 and (ms.sum() - y[ms].sum()) >= 30 and r["n_sample_clips"] >= 20
            if not ok:
                r["auc"] = "too small (<30 windows in a class or <20 episodes in the sample)"
            else:
                cnt = L.boot_counts(clip[ms])
                r["auc"] = {k: G.rec(v[ms], y[ms], clip[ms], outer[ms], cnt) for k, v in sc.items()}
                r["paired"] = {f"{a}_minus_{b}": G.paired(sc[a][ms], sc[b][ms], y[ms], clip[ms], outer[ms], cnt) for a, b in PAIRS}
                print(pop, s, "n", r["n_sample"], {k: round(v["fold"]["auc"], 3) for k, v in r["auc"].items()},
                      {k: f"{v['fold']['delta']:+.3f} [{v['fold']['lo']:+.3f},{v['fold']['hi']:+.3f}]" for k, v in r["paired"].items() if k.startswith("G8+E_minus_G4")}, flush=True)
            if not ok:
                print(pop, s, "n", r["n_sample"], r["auc"][:18], flush=True)
            P[s] = r
        out["pops"][pop] = P
    json.dump(out, open(f"{G.RAW}/results_strata_g.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
