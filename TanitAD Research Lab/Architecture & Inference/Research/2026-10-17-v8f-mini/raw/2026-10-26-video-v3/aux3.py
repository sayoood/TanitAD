"""aux3.py -- small derived numbers for the v3 mechanism card (numpy only, read-only on sources): mean |tau - P| per step and axis (eval-141), so that the first steps' weights can be read honestly."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_lib import *  # noqa
out = {"what": "mean |tau - P| over the 2,399 eval windows, per step (0.2 s knots) and axis (0 along, 1 cross), metres; tau = B's plan, P = the frozen P10 (hblend/score_*_perwindow.npz)"}
for seed in SEEDS:
    hb = np.load(f"{HBLEND}/score_{seed}_perwindow.npz")
    r = np.abs(hb["tau"].astype(np.float64) - hb["P"].astype(np.float64)).mean(0)      # [30, 2]
    out[seed] = {"along_by_step": r[:, 0].tolist(), "cross_by_step": r[:, 1].tolist()}
    print(seed, np.round(r[:8, 0], 4), np.round(r[:8, 1], 4), np.round(r[[14, 29]], 3))
json.dump(out, open(f"{OUT}/data/aux3.json", "w"), indent=1)
