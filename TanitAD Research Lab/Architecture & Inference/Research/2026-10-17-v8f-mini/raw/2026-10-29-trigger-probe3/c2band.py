#!/usr/bin/env python3
"""c2band.py -- the CONDITIONAL part of control C2 (PREREG 6): for every linear arm {G4+E, G8+E} whose seed-2 donor-label fold-mean CI misses 0.5, run donor seeds 3-6 (banked as C2:donor<seed>:<src>+E).
python -I c2band.py"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import g_lib as G  # noqa: E402
import numpy as np
import run_g  # noqa: E402

for pop in G.POPS:
    p = f"{G.RAW}/resg_{pop}_linear.json"
    if not os.path.exists(p):
        continue
    res = json.load(open(p))
    need = []
    for src in ("G8", "G4"):
        a = res["arms"].get(f"C2:donor:{src}+E")
        if a and not (a["fold"]["lo"] <= 0.5 <= a["fold"]["hi"]):
            need += [f"C2:donor{sd}:{src}+E" for sd in (3, 4, 5, 6)]
    print(pop, "band arms needed:", need, flush=True)
    if need:
        d = G.load_pop(pop)
        arms = run_g.build_arms(d, run_g.feats(pop), "linear")
        G.run_arms(pop, d, "linear", {k: v for k, v in arms.items() if k in need})
