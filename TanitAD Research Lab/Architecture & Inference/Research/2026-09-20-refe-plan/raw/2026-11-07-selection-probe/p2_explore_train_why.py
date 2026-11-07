#!/usr/bin/env python3
"""TRAIN-only: where do the ego-gated rules Bg(0.2) / Sg(1) lose PDMS? Per stratum: changed tokens, better/worse, sub-score deltas."""
import json

import numpy as np

import sel_common as sc
import sel_rules as sr

D = sc.load()
split = json.load(open(sc.OUT + "/split.json"))
idx = np.where(np.isin(D["log"], split["train_logs"]))[0]
L, pdms, sub, v0, v4 = D["logits"][idx], D["pdms"][idx], D["sub"][idx], D["v0"][idx], D["v4_all"][idx]
strat, ax0 = D["stratum"][idx], D["ax0"][idx]
s = sr.speed_ratio(v4, v0)
r = np.arange(len(idx))
G = (v0 > 5) & (np.abs(ax0) < 0.3)
base = D["pick"][idx]
for nm, pk in (("Bg0.2", sr.pick_B(L, s, 0.2)), ("Sg1", sr.pick_S(L, s, 1))):
    pk = np.where(G, pk, base)
    ch = pk != base
    for st in ("random", "zero"):
        m = ch & (strat == st)
        print(nm, st, "n changed", m.sum(), "dPDMS mean %.3f" % (pdms[r, pk] - pdms[r, base])[m].mean(),
              "n worse", ((pdms[r, pk] < pdms[r, base] - 1e-9) & m).sum(), "n better", ((pdms[r, pk] > pdms[r, base] + 1e-9) & m).sum(),
              "dsub(NC,DAC,EP,TTC,C,DDC)", np.round((sub[r, pk] - sub[r, base])[m].mean(0), 3),
              "v4/v0 base->new %.2f->%.2f" % ((v4[r, base] / v0)[m].mean(), (v4[r, pk] / v0)[m].mean()))
