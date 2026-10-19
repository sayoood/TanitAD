"""PRIVGEOM-CLEAN C-POWER plant calibration on the banked SEEN arrays ONLY (PRIVGEOM-SAFE's 41 logs; their labels were opened by that arm).  Opens no label of the 76.

The plant is `label + sigma * N(0, 1)`, one column, centred within set like the primary block, fitted by the registered L+G recipe (seed 0).  sigma* = the grid value whose Delta_F (point) on the SEEN logs is
nearest +0.030 (the effect the programme sizes SH5 for; `MM_RULING_PRIVGEOM.md` section 3).  The L arm is PRIVGEOM-SAFE's banked OOF (`run/fits/L_dac_s0.npy`), the SAME recipe and folds.

    python pgc_calibrate_plant.py --out <plant_calibration.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pgc_lib as C                                                                      # noqa: E402

P, FL = C.P, C.FL
SAFE = "C:/Users/Admin/par18/privgeom"
GRID = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
TARGET = 0.030


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A = np.load(f"{SAFE}/run/join/arrays.npz", allow_pickle=False)
    lab = np.load(f"{SAFE}/run/join/labels_sub.npy")
    feat = np.load(f"{SAFE}/run/join/feat_sub.npy", mmap_mode="r")
    col = C.ROW_COL
    z_st, y = np.ascontiguousarray(A["z6"][..., col]), np.ascontiguousarray(lab[..., col])
    an = P.Analyzer(lab, A["code"], A["log_idx"], int(A["log_idx"].max()) + 1)
    sL = an.per_log_stats(np.load(f"{SAFE}/run/fits/L_dac_s0.npy"), C.ROW)
    rows = []
    for sg in GRID:
        t0 = time.time()
        g = C.prepared(C.weak_plant(y, sg), True)
        oof = P.oof_logits("LG", feat, z_st, y, g, A["fold"], 0, A["p2_W6"][col], float(A["p2_b6"][col]))
        ci = an.diff(an.per_log_stats(oof, C.ROW), sL, "Q4")
        rows.append({"sigma": sg, "delta_F": ci, "seconds": time.time() - t0, "confirm_rule_fires": C.seed_call(ci) == "CONFIRM"})
        print(f"sigma {sg}: Delta_F {ci['point']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] ({time.time() - t0:.0f} s)", flush=True)
    star = min(rows, key=lambda r: abs(r["delta_F"]["point"] - TARGET))
    doc = {"schema": "privgeom-clean-plant-calibration-v1", "evidence": "MEASURED on PRIVGEOM-SAFE's banked SEEN arrays (41 logs, seed 0); no label of the 76 is read", "target_delta_F": TARGET, "grid": rows,
           "sigma_star": star["sigma"], "delta_F_at_sigma_star": star["delta_F"], "recipe": "centred single column label + sigma*N(0,1), seed PLANT_SEED + 1 = 20261050, L+G registered recipe, seed 0"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    print("sigma_star", star["sigma"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
