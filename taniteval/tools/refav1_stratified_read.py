#!/usr/bin/env python3
"""refav1_stratified_read.py -- where, per manoeuvre, does the plan lose to the floor?

The banked record reports the four families POOLED over 282 windows. That
answers "how far off"; it cannot answer "on what". This stratifies the SAME
windows by the manoeuvre the ground truth actually contains, and re-runs the
SAME estimator inside each stratum — `taniteval.ci.paired_episode_cluster_bootstrap`
over the episode clusters present in that stratum.

⛔ Strata are defined from the GT trajectory, never from the model's own output:
a stratum defined by what the model did would be circular. Every stratum prints
its n_windows AND n_episode_clusters, because a stratum with few clusters is a
POWER LIMIT, not a negative (RETRACTION_LOG #17).
"""
import glob
import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "stack"), os.path.dirname(_HERE)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from taniteval import ci  # noqa: E402

DUMP = sys.argv[1] if len(sys.argv) > 1 else "full_dump"
DT = 0.2

G, EID, ARMS = [], [], {a: [] for a in ("cl", "ha0", "ha", "ol")}
LATLAB, LONLAB = [], []
for p in sorted(glob.glob(os.path.join(DUMP, "ep*.npz"))):
    d = np.load(p)
    n = d["g"].shape[0]
    G.append(d["g"])
    EID.append(np.repeat(d["eid"], n))
    for a in ARMS:
        ARMS[a].append(d[a])
    dp = np.load(os.path.join(DUMP, "decisions", os.path.basename(p)))
    LATLAB.append(dp["lat_label"])
    LONLAB.append(dp["lon_label"])
cat = np.concatenate
G, EID = cat(G), cat(EID)
ARMS = {a: cat(v) for a, v in ARMS.items()}
LATLAB, LONLAB = cat(LATLAB), cat(LONLAB)
N = G.shape[0]

# ---- per-window components (the same definitions the families use) --------
ade = {a: np.linalg.norm(ARMS[a] - G, axis=2).mean(1) for a in ARMS}
chord = lambda P: np.linalg.norm(np.diff(np.concatenate(
    [np.zeros((P.shape[0], 1, 2)), P], axis=1), axis=1), axis=2) / DT
spd = {a: chord(ARMS[a]) for a in ARMS}
gspd = chord(G)
speed_mae = {a: np.abs(spd[a] - gspd).mean(1) for a in ARMS}
cross_mae = {a: np.abs(ARMS[a][:, :, 1] - G[:, :, 1]).mean(1) for a in ARMS}

# ---- strata, defined from GT ONLY ----------------------------------------
lat_ext = np.abs(G[:, :, 1]).max(1)
dv = gspd[:, -1] - gspd[:, 0]
v0g = gspd[:, 0]
strata = {
    "ALL": np.ones(N, bool),
    "GT turning  (|lat| > 2.0 m)": lat_ext > 2.0,
    "GT curving  (0.5-2.0 m)": (lat_ext > 0.5) & (lat_ext <= 2.0),
    "GT straight (|lat| <= 0.5 m)": lat_ext <= 0.5,
    "GT braking   (dv < -1 m/s)": dv < -1.0,
    "GT accelerating (dv > +1)": dv > 1.0,
    "GT steady   (|dv| <= 1)": np.abs(dv) <= 1.0,
    "low speed  (v0 < 5 m/s)": v0g < 5.0,
    "mid speed  (5-15 m/s)": (v0g >= 5.0) & (v0g < 15.0),
    "high speed (>= 15 m/s)": v0g >= 15.0,
    "v7.2 label = a TURN": np.isin(LATLAB, [6, 7]),
    "v7.2 label = BRAKE_TO": LONLAB == 3,
    "v7.2 label = ACCELERATE": LONLAB == 7,
}

print(f"{'stratum':30s} {'n_win':>6} {'n_ep':>5} | "
      f"{'ADE cl-ha0':>22} | {'LON speed cl-ha0':>22} | {'ha0 ADE':>8} {'ol ADE':>8}")
print("-" * 118)
rows = {}
for name, m in strata.items():
    nw = int(m.sum())
    ne = int(len(np.unique(EID[m]))) if nw else 0
    if nw < 4 or ne < 3:
        print(f"{name:30s} {nw:6d} {ne:5d} | "
              f"REFUSED: under-powered (needs >= 4 windows and >= 3 episode clusters) "
              f"— a power limit, not a negative")
        continue
    r = {}
    for tag, comp in (("ade", ade), ("speed", speed_mae), ("cross", cross_mae)):
        b = ci.paired_episode_cluster_bootstrap(comp["cl"][m], comp["ha0"][m],
                                                EID[m], n_boot=2000, seed=0)
        r[tag] = b
    fmt = lambda b: (f"{b['delta']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}]"
                     + ("*" if b.get("separated") else " "))
    print(f"{name:30s} {nw:6d} {ne:5d} | {fmt(r['ade']):>22} | "
          f"{fmt(r['speed']):>22} | {ade['ha0'][m].mean():8.4f} {ade['ol'][m].mean():8.4f}")
    rows[name] = {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                      for kk, vv in v.items()} for k, v in r.items()}
    rows[name]["n_windows"], rows[name]["n_episodes"] = nw, ne
    rows[name]["ha0_ade"] = float(ade["ha0"][m].mean())
    rows[name]["ol_ade"] = float(ade["ol"][m].mean())
    rows[name]["cl_ade"] = float(ade["cl"][m].mean())
print("\n* = the paired episode-cluster bootstrap interval excludes zero.")
print("POSITIVE delta = refav1's plan is WORSE than the constant-velocity floor.")
json.dump(rows, io.open("stratified_cl_minus_ha0.json", "w", encoding="utf-8"), indent=1)
print("wrote stratified_cl_minus_ha0.json")
