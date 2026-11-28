#!/usr/bin/env python3
"""strata2.py -- PREREG_FROZEN.md 10: TRAIN strata (traffic light, lead vehicle) for the new arms, on the banked/new LINEAR out-of-fold scores.  Same strata flags, same 'too small' rule as the previous probe's strata.py.
GO rates on the FULL populations are the previous package's (raw/results_strata.json), copied by reference.  Primary metric = fold-mean AUC, pooled beside.   Writes raw/results_strata2.json.   CUDA_VISIBLE_DEVICES=-1  python -I strata2.py"""
from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import numpy as np

import common2 as C  # noqa: E402

L = C.L
STR = ("TL_ANY", "TL_RED", "TL_GREEN", "TL_YELLOW", "NO_TL", "lead_now", "no_lead", "lead_depart")


def scores(pop):
    ov, oz = np.load(f"{C.OLDRAW}/oof_{pop}_V_linear.npz"), np.load(f"{C.OLDRAW}/oof_{pop}_Z_linear.npz")
    sc = {"E": ov["E"], "V+E": ov["V+E"], "Z+E": oz["Z+E"]}
    for src, key in (("Znn", "Znn+E"), ("Znl", "Znl+E"), ("V4", "V4+E"), ("Vdiff", "Vdiff+E")):
        f = f"{C.RAW}/oof2_{src}_{pop}_linear.npz"
        if os.path.exists(f) and key in np.load(f).files:
            sc[key] = np.load(f)[key]
    f = f"{C.RAW}/oof2_nav_{pop}_linear.npz"
    if os.path.exists(f):
        for key in ("N5+E", "NL+E"):
            sc[key] = np.load(f)[key]
    return sc


def main():
    old = json.load(open(f"{C.OLDRAW}/results_strata.json"))
    out = {"note": "TRAIN only; TL_* are CLIP-level and VLM-derived (vlm-cot, disputed, untimed); LEAD_DEPART reads the future (stratum, never a feature); GO rates (full population) are the previous package's",
           "pops": {}}
    for pop in C.POPS:
        d = C.load_pop(pop)
        y, clip, outer = d["y"], d["clip"], d["outer"]
        sc = scores(pop)
        P = {"strata": {}, "go_rate_full": old["pops"][pop]["strata"]}
        for s in STR:
            ms = d[s].astype(bool)
            r = {"n_sample": int(ms.sum()), "n_sample_clips": int(len(np.unique(clip[ms]))) if ms.any() else 0, "n_sample_go": int(y[ms].sum()) if ms.any() else 0,
                 "go_rate_full": old["pops"][pop]["strata"][s].get("go_rate_full")}
            ok = ms.sum() >= 60 and y[ms].sum() >= 30 and (ms.sum() - y[ms].sum()) >= 30 and r["n_sample_clips"] >= 20
            if not ok:
                r["auc"] = "too small (<30 windows in a class or <20 episodes in the sample)"
            else:
                cnt = L.boot_counts(clip[ms])
                r["auc"] = {k: C.rec(v[ms], y[ms], clip[ms], outer[ms], cnt) for k, v in sc.items()}
                r["paired"] = {}
                for a, b in (("V+E", "E"), ("Z+E", "V+E"), ("Znn+E", "E"), ("Z+E", "Znn+E"), ("Znn+E", "V+E"), ("Znl+E", "E"), ("V4+E", "V+E"), ("Vdiff+E", "V+E"), ("N5+E", "E"), ("NL+E", "E")):
                    if a in sc and b in sc:
                        r["paired"][f"{a}_minus_{b}"] = C.paired(sc[a][ms], sc[b][ms], y[ms], clip[ms], outer[ms], cnt)
            P["strata"][s] = r
            if isinstance(r["auc"], dict):
                print(pop, s, "n", r["n_sample"], {k: round(v["fold"]["auc"], 3) for k, v in r["auc"].items()}, flush=True)
            else:
                print(pop, s, "n", r["n_sample"], r["auc"][:18], flush=True)
        out["pops"][pop] = P
    json.dump(out, open(f"{C.RAW}/results_strata2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
