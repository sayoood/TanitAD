#!/usr/bin/env python3
"""COST of the full real run, measured on SYNTHETIC / SEMI-SYNTHETIC data of the REAL shapes (400 tokens, 24 logs, 96 perturbations each, 256-d tokens).  No real label is read.

The full run = 2 seeds x 5 folds x (H + H0)  +  5 GLOBAL-null replicates x 2 seeds x 5 folds  =  70 fold fits, 14 trained cells, 15 evaluations (P2 + 14).

    python cost_synthetic.py --world shape            (real Step A geometry + synthetic features and labels; needs only the L1 run dir)
    python cost_synthetic.py --world semi --feats F   (REAL tokens + synthetic labels: the closest stand-in for the real run)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
import synthetic  # noqa: E402


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=("shape", "semi"), default="shape")
    ap.add_argument("--feats", default="D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--recipe", default="", help="a recipe .json file, or a JSON string of Recipe fields (default: the PROVISIONAL l2lib.Recipe)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    rec = l2lib.Recipe.from_json(a.recipe) if a.recipe.endswith(".json") else (l2lib.Recipe(**json.loads(a.recipe)) if a.recipe else l2lib.Recipe())
    t0 = time.time()
    if a.world == "shape":
        D = synthetic.make_data(structure=synthetic.real_structure(), gamma=0.3, rho=0.4, seed=5)
    else:
        D = synthetic.semi_synthetic(l2lib.load_real(a.feats), beta=0.5, alpha=0.3, seed=5)
    t_build = time.time() - t0
    out = {"world": a.world, "N": D.N, "Q": D.Q, "D": D.D, "n_logs": D.n_logs, "threads": a.threads, "recipe": rec.as_dict(), "data_build_s": round(t_build, 1), "rss_mb_after_build": rss_mb()}
    t = time.time()
    p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), rec)
    ev0 = l2lib.evaluate_arm(D, p2["ep_in"], p2["ep_p"])
    out["P2_cell_incl_eval_s"] = round(time.time() - t, 1)
    t = time.time()
    h = l2lib.oof_scores(D, l2lib.ArmSpec("H", "H", seed=0), rec)
    out["H_cell_fit_5_folds_s"] = round(time.time() - t, 1)
    out["H_cell_fold_steps"] = [f["steps"] for f in h["diag"]["folds"]]
    out["H_cell_fold_pairs"] = [f["n_pairs"] for f in h["diag"]["folds"]]
    t = time.time()
    h0 = l2lib.oof_scores(D, l2lib.ArmSpec("H0", "H0", seed=0), rec)
    out["H0_cell_fit_5_folds_s"] = round(time.time() - t, 1)
    t = time.time()
    evh = l2lib.evaluate_arm(D, h["ep_in"], h["ep_p"], l2lib.ref_terms_of(ev0))
    out["evaluate_arm_s"] = round(time.time() - t, 1)
    t = time.time()
    n = l2lib.oof_scores(D, l2lib.ArmSpec("N", "Hnull", seed=0, null_rep=100, null_group="global"), rec)
    out["Hnull_cell_fit_5_folds_s"] = round(time.time() - t, 1)
    out["rss_mb_peak_approx"] = rss_mb()
    out["extrapolated_full_run_s"] = {"fits_2_H": round(2 * out["H_cell_fit_5_folds_s"], 1), "fits_2_H0": round(2 * out["H0_cell_fit_5_folds_s"], 1),
                                      "fits_10_Hnull": round(10 * out["Hnull_cell_fit_5_folds_s"], 1), "evals_15": round(15 * out["evaluate_arm_s"], 1)}
    out["extrapolated_full_run_s"]["total"] = round(sum(out["extrapolated_full_run_s"].values()), 1)
    out["extrapolated_full_run_min"] = round(out["extrapolated_full_run_s"]["total"] / 60, 1)
    out["note"] = "measured on a box at ~90 % CPU load from other jobs (the 24-thread i9-12900F); no GPU; times are upper-ish. Cells are independent: they parallelise across processes (run_l2.py --cells)."
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
