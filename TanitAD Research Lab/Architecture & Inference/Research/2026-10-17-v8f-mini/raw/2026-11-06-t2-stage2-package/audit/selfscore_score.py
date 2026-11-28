"""AGENT F audit, process B (scorer side; the reader's own tree, in its OWN process). CPU only.
Reads the npz written by selfscore_run.py and scores every pair with the T2 scorer's `own_rows` (= the T1 reader's rows).
Usage: python -I selfscore_score.py IN.npz OUT.json
Output: per case, per pair, the 10 reader rows (ADE6, speed, along, accel, heading, yaw-rate, curvature, cross, L1, L2) as the window values, plus max |x|."""
from __future__ import annotations

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE + "/../code")
import numpy as np  # noqa: E402

import t2v8f_score as S  # noqa: E402

ROWS = ("ADE6", "speed MAE", "along-track MAE", "accel MAE", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)")
PAIRS = {                                             # name -> (executed side, logged side)
    "SELF  cl vs g          (T2 path)": ("cl", "g"),
    "SELF  cl vs g_t1       (T1 reader path)": ("cl", "g_t1"),
    "XCHK  g vs g_t1        (T2 g against T1 g)": ("g", "g_t1"),
    "MUT   g_shift1 vs g_t1 (a g built 0.1 s late)": ("g_shift1", "g_t1"),
    "MUT   cl_skew1 vs g    (plan rolled one tick)": ("cl_skew1", "g"),
    "MUT   cl vs g_shift1   (logged future 0.1 s late)": ("cl", "g_shift1"),
    "MUT   cl vs g_shift2   (logged future 0.2 s late)": ("cl", "g_shift2"),
    "MUT   cl vs g_yawflip  (yaw sign flipped)": ("cl", "g_yawflip"),
}


def score(z, c):
    out = {"case": str(z[f"c{c}_name"][0]), "t0": int(z[f"c{c}_t0"][0]), "pairs": {}}
    for nm, (a, b) in PAIRS.items():
        X, G = z[f"c{c}_{a}"][None], z[f"c{c}_{b}"][None]
        rows, _ = S.own_rows(X, G)
        vals = {k: (float(rows[k][0]) if np.isfinite(rows[k][0]) else None) for k in ROWS}
        fin = [abs(v) for v in vals.values() if v is not None]
        out["pairs"][nm] = {"rows": vals, "max_abs": max(fin) if fin else None, "n_nan_rows": sum(v is None for v in vals.values()),
                            "max_abs_knot_diff_m": float(np.abs(X - G).max())}
    return out


def main(inp, outp):
    S.init()
    z = np.load(inp)
    res = [score(z, c) for c in range(int(z["n_cases"][0]))]
    json.dump(res, open(outp, "w", encoding="utf-8", newline="\n"), indent=1)
    for r in res:
        print("CASE", r["case"].split("/")[-1], "t0", r["t0"])
        for nm, p in r["pairs"].items():
            v = p["rows"]
            print(f"  {nm:52s} max|knot diff| {p['max_abs_knot_diff_m']:9.3g} m | ADE6 {v['ADE6']} heading {v['heading MAE']} yaw {v['yaw-rate MAE']} cross {v['cross-track MAE']} nanrows {p['n_nan_rows']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
