"""lat_jitter.py -- MM 4a3d05, 2026-11-12, DIAGNOSTIC (no bar): is K2's lateral excess over K0 SLOW (per-episode bias) or FAST (frame-to-frame / within-window)?
Signed lateral error e_w = P[w,-1,1] - G[w,-1,1] (ego frame, final waypoint 2.0 s).  Per episode (windows ordered by ws):
  bias2  = (episode mean of e)^2, assigned to every window of the episode  -> slow part
  resid2 = (e - episode mean)^2                                               -> within-episode spread
  dlag   = |e_{w+1} - e_w| over consecutive windows                           -> frame-to-frame jitter
  wig    = |P[2,1] - 2 P[1,1] + P[0,1]| - same for G  (lateral 2nd difference inside a window) -> path wiggle
Paired episode-cluster bootstrap (taniteval.ci) B 2000 seed 0, delta = A - B.  DEV windows, T1 self-action open loop, os arm."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path[:0] = ["F:/refcv9_e3/famfix_mm/tree/stack/scripts", "F:/refcv9_e3/famfix_mm/tree/stack", "F:/refcv9_e3/famfix_mm/tree/taniteval"]
import r9_e2_families as F
from taniteval import ci
B = Path("C:/Users/Admin/r9e2_read_K2M1N20"); SRC = B / "reread_dtfix/src"
dev = F.load_dev(SRC / "DEV_HOLD_SPLIT.json")
ARMS = {"K2": SRC / "E2_K2_pf", "K0": SRC / "E2_K0_pf", "K2_mapzero": B / "abl/E2_K2abl_mapzero_pf", "K2_boxzero": B / "abl/E2_K2abl_boxzero_pf", "K2_drvzero": B / "abl/E2_K2abl_drvzero_pf"}


def rows(arm, S):
    d, _ = F.restrict_dev(F.load_dump(ARMS[arm] / f"dump_s{S}"), dev)
    by = defaultdict(list)
    for i, (c, w) in enumerate(d["keys"]):
        by[c].append((w, i))
    out = {}
    for c, lst in by.items():
        lst.sort(); idx = [i for _, i in lst]
        P, G = d["P"][idx], d["G"][idx]
        e = P[:, -1, 1] - G[:, -1, 1]
        m = e.mean()
        wig = np.abs(P[:, 2, 1] - 2 * P[:, 1, 1] + P[:, 0, 1]) - np.abs(G[:, 2, 1] - 2 * G[:, 1, 1] + G[:, 0, 1])
        out[c] = {"ws": [w for w, _ in lst], "bias2": np.full(len(e), m * m), "resid2": (e - m) ** 2, "abs_e": np.abs(e),
                  "dlag": np.abs(np.diff(e)), "wig": wig}
    return out


def pair(A, Bd, key):
    a, b, eid = [], [], []
    for c in sorted(A):
        assert A[c]["ws"] == Bd[c]["ws"], c
        a += list(A[c][key]); b += list(Bd[c][key]); eid += [c] * len(A[c][key])
    a, b = np.asarray(a), np.asarray(b)
    r = ci.paired_episode_cluster_bootstrap(a, b, eid, n_boot=2000, seed=0)
    return {"n": int(r["n_windows"]), "A": round(float(a.mean()), 6), "B": round(float(b.mean()), 6), "delta": round(float(r["delta"]), 6),
            "lo": round(float(r["lo"]), 6), "hi": round(float(r["hi"]), 6), "sep": bool(r["separated"])}


res = {"class": "DIAGNOSTIC", "tier": "T1 self-action open loop", "windows": "DEV", "estimator": "taniteval.ci paired episode-cluster bootstrap B 2000 seed 0"}
PP = [("K2_drvzero", "K0"), ("K2_drvzero", "K2"), ("K2", "K0"), ("K2_mapzero", "K0"), ("K2_boxzero", "K0"), ("K2_mapzero", "K2"), ("K2_boxzero", "K2")]
SEEDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1"]
OUTN = sys.argv[2] if len(sys.argv) > 2 else "lat_jitter.json"
for S, pairs in ((x, PP) for x in SEEDS):
    for a, b in pairs:
        A, Bd = rows(a, S), rows(b, S)
        r = {k: pair(A, Bd, k) for k in ("abs_e", "bias2", "resid2", "dlag", "wig")}
        res[f"s{S} {a} - {b}"] = r
        print(f"s{S} {a}-{b}: " + " | ".join(f"{k} {v['A']:.5f}/{v['B']:.5f} d{v['delta']:+.5f}[{v['lo']:+.5f},{v['hi']:+.5f}]{'*' if v['sep'] else ''}" for k, v in r.items()))
Path(B / "abl" / OUTN).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
