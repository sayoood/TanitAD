"""select_scenes.py -- the scene-selection rule of chapter 7 (T1-kin), FIXED BEFORE ANY T1 NUMBER PER WINDOW IS READ.

This script reads ONLY the keys  g (GT), ws, clip_index, eid  of the causal-run dump.  It never touches cl / fl / ol (the executed tracks), so no model-vs-floor
number of any window can influence the choice.  Output: SELECTION_V4.md + data/selection_v4.json.

RULE (verbatim on screen):
  1. the four v3 windows (JUNCTION #1131, CURVE #654, NO-TURN #763, FAILURE #2158 of viz/out/selection.json), if their (clip, cache step) is in the T1 grid;
  2. three more windows drawn with numpy.random.default_rng(20261026).choice(n_cand, 3, replace=False) from the candidate list = the T1-grid windows
     (dump order: clip_index ascending, then ws ascending) whose GT mean speed is >= 5 m/s, EXCLUDING the windows of (1);
     GT mean speed = mean over the 30 knots of |g[k] - g[k-1]| / 0.2 s, with g[-1] = the origin.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
SC = HERE.rsplit("/", 1)[0]
DUMP0 = "C:/Users/Admin/t1kc_run/dump_HACC_s0_m10_causal"
JOIN = "C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
V3SEL = f"{SC}/viz/out/selection.json"
SEED = 20261026
SPEED_MIN = 5.0
DT = 0.2

sys.path.insert(0, "C:/Users/Admin/t1k/stack")
sys.path.insert(0, "C:/Users/Admin/t1k/taniteval")
from tanitad.data.v8f_mini_data import load_join  # noqa: E402


def gt_mean_speed(g):
    g = np.asarray(g, np.float64)
    seg = np.diff(np.concatenate([np.zeros((g.shape[0], 1, 2)), g], axis=1), axis=1)
    return np.linalg.norm(seg, axis=-1).mean(-1) / DT


def main():
    J = load_join(JOIN)
    clips = [str(c) for c in J["clip"]]
    files = sorted(glob.glob(f"{DUMP0}/ep*.npz"))
    G, WS, CI = [], [], []
    for f in files:
        with np.load(f) as z:
            G.append(z["g"]); WS.append(z["ws"]); CI.append(z["clip_index"])
    G, WS, CI = np.concatenate(G), np.concatenate(WS), np.concatenate(CI)
    n = len(WS)
    sp = gt_mean_speed(G)
    # v3 windows
    sel = json.load(open(V3SEL))
    v3 = {}
    for name in ("JUNCTION", "CURVE", "NO-TURN", "FAILURE(worst)"):
        s = sel[name]
        ci = clips.index(s["clip"])
        hit = np.where((CI == ci) & (WS == s["t"]))[0]
        v3[name] = {"v3_window": s["window"], "clip": s["clip"], "clip_index": ci, "ws": int(s["t"]), "in_T1_grid": bool(len(hit) == 1),
                    "grid_row": int(hit[0]) if len(hit) == 1 else None}
    excl = {v["grid_row"] for v in v3.values() if v["in_T1_grid"]}
    cand = [i for i in range(n) if sp[i] >= SPEED_MIN and i not in excl]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(cand), size=3, replace=False)
    drawn = [cand[int(p)] for p in pick]
    scenes = []
    for name, v in v3.items():
        if v["in_T1_grid"]:
            scenes.append({"tag": name, "source": "v3", **v, "gt_mean_speed": float(sp[v["grid_row"]])})
    for k, row in enumerate(drawn):
        scenes.append({"tag": f"DRAW{k + 1}", "source": f"rng(default_rng({SEED})).choice", "grid_row": int(row), "clip_index": int(CI[row]), "clip": clips[int(CI[row])],
                       "ws": int(WS[row]), "gt_mean_speed": float(sp[row]), "in_T1_grid": True})
    out = {"seed": SEED, "speed_min_mps": SPEED_MIN, "n_windows": n, "n_ge5": int((sp >= SPEED_MIN).sum()), "n_candidates": len(cand), "draw_positions_in_candidates": [int(p) for p in pick],
           "v3_windows": v3, "scenes": scenes, "grid_row_equals_v3_window_id": {k: (v["grid_row"] == v["v3_window"]) for k, v in v3.items()}}
    json.dump(out, open(f"{HERE}/data/selection_v4.json", "w"), indent=1)
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    L = [f"# SELECTION_V4 -- chapter 7 (T1-kin) scenes, fixed before any per-window T1 number was read", "",
         f"script `select_scenes.py` (sha256 {sha}); reads only g / ws / clip_index of `{DUMP0}`; no executed track (cl / fl / ol) is loaded.", "",
         "RULE (stated on screen):", "1. the four v3 windows, if their (clip, cache step) is in the T1 grid;",
         f"2. three windows drawn with numpy.random.default_rng({SEED}).choice(n_cand, 3, replace=False) from the T1-grid windows (dump order: clip_index, then ws) whose GT mean speed is >= {SPEED_MIN:g} m/s, "
         "EXCLUDING the four v3 windows. GT mean speed = mean over the 30 knots of |g[k]-g[k-1]|/0.2 s, g[-1] = origin.",
         "No window is picked, swapped or dropped for how the model looks on it.", "",
         f"T1 grid: {n} windows; GT mean speed >= {SPEED_MIN:g} m/s: {out['n_ge5']}; candidates after excluding v3: {len(cand)}; draw positions {out['draw_positions_in_candidates']}.", "",
         "| tag | source | grid row | clip (first 8) | clip_index | ws (cache step) | GT mean speed (m/s) |", "|---|---|---|---|---|---|---|"]
    for s in scenes:
        L.append(f"| {s['tag']} | {s['source']} | {s['grid_row']} | {s['clip'][:8]} | {s['clip_index']} | {s['ws']} | {s['gt_mean_speed']:.2f} |")
    L.append("")
    L.append(f"v3 windows in the T1 grid: {sum(v['in_T1_grid'] for v in v3.values())} of 4. grid row == v3 window id: {out['grid_row_equals_v3_window_id']}")
    open(f"{HERE}/SELECTION_V4.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
