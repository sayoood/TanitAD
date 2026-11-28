"""select_scenes5.py -- the scene-selection rule of chapter 8 (DAgger), FIXED AND WRITTEN TO SELECTION_V5.md BEFORE any chapter-8 frame is rendered.

Unlike chapter 7's rule (GT only), a "median window" needs a per-window difference, so this script DOES read the executed tracks `cl` of the two seed-0 T1-kin runs
(DAG and DAG-0) -- and nothing else of them: the choice is made by the RULE below, never by how a window looks.  Seed 0 is primary; seed 1 is shown beside, never used to choose.

RULE (RULE_TEXT, verbatim on screen; its sha256 is printed in SELECTION_V5.md and in VIDEO_NOTES_V5.md):
  delta_w = ADE6(DAG, s0) - ADE6(DAG-0, s0) of window w (T1-kin, causal, m = 10; negative = DAG closer to the human's path), over the 2,399 windows in the dump order.
  1. For each GT stratum (JUNCTION / CURVE / NO-TURN, window_strata_eval141.npz: defined from the human's labels, not from any model): sort that stratum's windows by
     (|delta_w|, grid row) ascending and take position (n - 1) // 2 (the lower median of |delta|): the TYPICAL size of change in that stratum.
  2. WORST: the window with the largest delta_w over all 2,399 windows (DAG furthest behind DAG-0; ties -> lowest grid row).
  3. Three DRAWS: numpy.random.default_rng(20261027).choice(n_cand, 3, replace=False) over the candidates = windows (dump order) whose GT mean speed is >= 5 m/s and that
     are not already chosen by 1 or 2.  GT mean speed = mean over the 30 knots of |g[k] - g[k-1]| / 0.2 s, g[-1] = origin.
  No window is swapped or dropped for how it looks; a window where DAG is worse stays in.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts/v8f-dagger-t1"
DUMP = {"DAG": f"{ART}/dump_DAG_s0_m10_causal_heads", "DAG0": f"{ART}/dump_DAG0_s0_m10_causal_heads"}
STRATA = f"{HERE}/inputs/window_strata_eval141.npz"
SEED = 20261027
SPEED_MIN = 5.0
DT = 0.2
RULE_TEXT = (
    "delta_w = ADE6(DAG, seed 0) - ADE6(DAG-0, seed 0) of window w, T1-kin, causal, m = 10 (negative: DAG closer to the human's path), over the 2,399 windows in the dump order.\n"
    "1. For each GT stratum (JUNCTION, CURVE, NO-TURN; window_strata_eval141.npz, defined from the human's labels) sort the stratum's windows by (|delta_w|, grid row) ascending "
    "and take position (n - 1) // 2: the lower median of |delta|, the typical size of change in that stratum.\n"
    "2. WORST: the window with the largest delta_w over all 2,399 windows (ties: lowest grid row).\n"
    "3. Three DRAWS: numpy.random.default_rng(20261027).choice(n_cand, 3, replace=False) over the windows (dump order) with GT mean speed >= 5 m/s that are not already chosen by 1 or 2.\n"
    "No window is swapped or dropped for how it looks; a window where DAG is worse stays in.")
RULE_SHA256 = hashlib.sha256(RULE_TEXT.encode("utf-8")).hexdigest()
NAMES = ("JUNCTION", "CURVE", "NO-TURN")


def load(arm):
    G, CL, WS, CI, EID = [], [], [], [], []
    for f in sorted(glob.glob(f"{DUMP[arm]}/ep*.npz")):
        with np.load(f) as z:
            G.append(z["g"]); CL.append(z["cl"]); WS.append(z["ws"]); CI.append(z["clip_index"]); EID.append(z["eid"])
    return [np.concatenate(x) for x in (G, CL, WS, CI, EID)]


def ade6(p, g):
    return np.linalg.norm(np.asarray(p, np.float64) - np.asarray(g, np.float64), axis=-1).mean(-1)


def main():
    g, cl, ws, ci, eid = load("DAG")
    g0, cl0, ws0, ci0, _ = load("DAG0")
    assert np.array_equal(g, g0) and np.array_equal(ws, ws0) and np.array_equal(ci, ci0)
    z = np.load(STRATA, allow_pickle=True)
    assert np.array_equal(z["t"], ws) and np.array_equal(z["eid"], eid), "strata file order != dump order"
    stratum = z["stratum"].astype(int)
    n = len(ws)
    delta = ade6(cl, g) - ade6(cl0, g)
    seg = np.diff(np.concatenate([np.zeros((n, 1, 2)), g], axis=1), axis=1)
    sp = np.linalg.norm(seg, axis=-1).mean(-1) / DT
    chosen = []
    for k, nm in enumerate(NAMES):
        ix = np.nonzero(stratum == k)[0]
        o = sorted(ix, key=lambda i: (abs(float(delta[i])), int(i)))
        chosen.append({"tag": nm, "source": f"median abs(delta) of stratum {nm} (n = {len(ix)})", "grid_row": int(o[(len(o) - 1) // 2]), "stratum": nm})
    worst = int(np.argmax(delta))
    chosen.append({"tag": "WORST", "source": "largest delta over all windows", "grid_row": worst, "stratum": NAMES[stratum[worst]]})
    taken = {c["grid_row"] for c in chosen}
    cand = [i for i in range(n) if sp[i] >= SPEED_MIN and i not in taken]
    pick = np.random.default_rng(SEED).choice(len(cand), size=3, replace=False)
    for q, p in enumerate(pick):
        row = cand[int(p)]
        chosen.append({"tag": f"DRAW{q + 1}", "source": f"default_rng({SEED}).choice", "grid_row": int(row), "stratum": NAMES[stratum[row]]})
    clips = None
    from numpy import load as npload
    jz = npload("E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz", allow_pickle=False)
    clips = [str(c) for c in jz["clip"]]
    for c in chosen:
        r = c["grid_row"]
        c.update(clip_index=int(ci[r]), clip=clips[int(ci[r])], ws=int(ws[r]), gt_mean_speed=float(sp[r]), delta_s0_m=float(delta[r]))
    out = {"seed": SEED, "speed_min_mps": SPEED_MIN, "n_windows": n, "n_candidates_draw": len(cand), "draw_positions_in_candidates": [int(p) for p in pick],
           "rule_text": RULE_TEXT, "rule_sha256": RULE_SHA256, "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "strata_file_sha256": hashlib.sha256(open(STRATA, "rb").read()).hexdigest(), "strata_counts": {nm: int((stratum == k).sum()) for k, nm in enumerate(NAMES)},
           "scenes": chosen}
    json.dump(out, open(f"{HERE}/data/selection_v5.json", "w"), indent=1)
    L = ["# SELECTION_V5 -- chapter 8 (DAgger) scenes, fixed by a rule BEFORE any chapter-8 frame was rendered", "",
         f"script `select_scenes5.py` (sha256 {out['script_sha256'][:16]}); **rule sha256 {RULE_SHA256}** (the RULE text below, UTF-8, as the string `RULE_TEXT` in the script).",
         f"Inputs: `{DUMP['DAG']}` and `{DUMP['DAG0']}` (keys g, cl, ws, clip_index, eid; seed 0 only), `inputs/window_strata_eval141.npz` (sha256 {out['strata_file_sha256'][:16]}).", "",
         "RULE (stated on screen):", ""] + ["> " + ln for ln in RULE_TEXT.split("\n")] + [
         "", f"Strata sizes: {out['strata_counts']}; candidates for the draws: {len(cand)}; draw positions {out['draw_positions_in_candidates']}.", "",
         "| # | tag | how chosen | stratum | grid row | clip (first 8) | clip_index | ws (cache step) | GT mean speed (m/s) | delta = ADE6(DAG) - ADE6(DAG-0), s0 (m) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(chosen):
        L.append(f"| {i + 1} | {s['tag']} | {s['source']} | {s['stratum']} | {s['grid_row']} | {s['clip'][:8]} | {s['clip_index']} | {s['ws']} | {s['gt_mean_speed']:.2f} | {s['delta_s0_m']:+.4f} |")
    L += ["", "The delta column is printed here for the record; it is the quantity the rule sorts by.  The seed-1 numbers are shown in the video beside, and were not used to choose."]
    open(f"{HERE}/SELECTION_V5.md", "w", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
