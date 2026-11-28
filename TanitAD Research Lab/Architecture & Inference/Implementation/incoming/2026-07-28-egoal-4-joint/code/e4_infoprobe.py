#!/usr/bin/env python3
"""E-GOAL-4 S3 -- ⭐⭐ IS THE GOAL INPUT NEW INFORMATION, OR A BETTER
REPRESENTATION OF INFORMATION THE SELECTOR ALREADY HAS?

This probe exists because the S1 result forces the question. The goal head the
brief names is `H_v0_ax` -- a function of EXACTLY TWO COLUMNS, `v` and `ax_fd`.
⛔ AND `S_nogoal` IS FED BOTH OF THEM. So under the FUTURE-BLIND `sel`
background, every `F_goal` column is a deterministic function of columns
`S_nogoal` already has, and any gain is INDUCTIVE BIAS, not information.

Three measurements, none of them asserted:

  P-1  `g_along ~ GBM(v, ax_fd)` R^2. Gate G-6 already proves the identity
       exactly (the per-fold refit reproduces `T_OOF|H_v0_ax` to max |Δ| = 0);
       this is the redundancy check, reported because "it is a function of two
       columns" should be measured, not argued.
  P-2  ⭐ THE PICK READ-OUT. Every arm's chosen candidate has an along-track
       endpoint. `RMS(end_along[pick] - y_true)` is the arm's EFFECTIVE
       along-track targeting accuracy -- i.e. the goal it BEHAVED as if it had.
       This is how you see an IMPLICIT goal in an arm that was never given one.
  P-3  the cross axis under each background: how much of the truth the goal's
       cross coordinate carries, and therefore what an oracle-cross arm is
       actually being credited with.

Run:  PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 python e4_infoprobe.py \
          --fan <fan600.pt> --feat <val.npz> --preds <preds.npz>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_common import (EG1, STREAM, ade, cross_background, load_all, r4)  # noqa: E402

sys.path.insert(0, str(EG1 / "code"))
from eg_fit import fit_predict  # noqa: E402
from eg_common import ci_paired, ci_single  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=str(STREAM / "raw" / "e4_infoprobe.json"))
    a = ap.parse_args(argv)
    t0 = time.time()

    D = load_all(a.fan, a.feat, a.preds)
    fan, gt, eid = D["fan"], D["gt"], D["eid"]
    W = len(gt)
    g_true = gt[:, -1, :]
    y = D["preds"]["y"].astype(np.float64)
    g_head = D["preds"]["T_OOF|H_v0_ax"].astype(np.float64)
    v, ax = D["X"][:, 0], D["X"][:, 1]
    R = {"_stream": "2026-07-28-egoal-4-joint", "_stage": "S3 information probe"}

    # ------------------------------------------------------------------ P-1 --
    p = fit_predict(np.stack([v, ax], 1), g_head, np.stack([v, ax], 1), "gbm")
    ss = 1.0 - np.sum((p - g_head) ** 2) / np.sum((g_head - g_head.mean()) ** 2)
    R["P_1_goal_is_a_function_of_two_fed_columns"] = {
        "_what": ("`g_along` regressed back onto `v` and `ax_fd` -- the two "
                  "columns `S_nogoal` is ALREADY fed"),
        "r2": r4(float(ss)),
        "rms_residual_m": r4(float(np.sqrt(np.mean((p - g_head) ** 2)))),
        "g_along_sd_m": r4(float(g_head.std())),
        "_gate": ("G-6 proves the identity exactly: the per-fold refit of "
                  "GBM(v, ax_fd) reproduces T_OOF|H_v0_ax to max |Δ| = 0"),
        "_reading": ("⛔ UNDER THE FUTURE-BLIND `sel` BACKGROUND THE GOAL INPUT "
                     "CARRIES NO INFORMATION `S_nogoal` LACKS. Any gain there "
                     "is INDUCTIVE BIAS / representation, not a new channel.")}
    print(f"[P-1] g_along ~ GBM(v, ax_fd)  R2 = {ss:.6f}", flush=True)

    # ------------------------------------------------------------------ P-2 --
    #: the PICK READ-OUT -- the along-track goal each arm BEHAVED as if it had
    end_along = fan[:, :, 3, 0]
    R["P_2_pick_readout"] = {
        "_what": ("RMS(end_along[pick] - y_true): the arm's EFFECTIVE "
                  "along-track targeting accuracy. An arm never given a goal "
                  "still has one -- this is how to see it."),
        "reference": {
            "as_trained_selector_A0": r4(float(np.sqrt(np.mean(
                (end_along[np.arange(W), D["sel"]] - y) ** 2)))),
            "goal_head_H_v0_ax_along_rms_m": r4(float(np.sqrt(np.mean(
                (g_head - y) ** 2)))),
            "best_in_fan_by_ADE": r4(float(np.sqrt(np.mean(
                (end_along[np.arange(W), np.linalg.norm(
                    fan - gt[:, None], axis=-1).mean(-1).argmin(1)] - y) ** 2)))),
        }, "arms": {}}
    for bg in ("parent_resampled", "sel"):
        f = STREAM / "raw" / f"e4_select_{bg}_realised.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            if not k.startswith("pick|"):
                continue
            nm = k.split("|", 1)[1]
            e = end_along[np.arange(W), z[k]] - y
            R["P_2_pick_readout"]["arms"].setdefault(nm, {})[bg] = {
                "effective_goal_rms_m": r4(float(np.sqrt(np.mean(e ** 2)))),
                "effective_goal_mae_m": r4(float(np.mean(np.abs(e)))),
                "effective_goal_bias_m": r4(float(e.mean()))}
        print(f"[P-2] {bg}: {len([k for k in z.files if k.startswith('pick|')])}"
              " arms read out", flush=True)

    # ------------------------------------------------------------------ P-3 --
    R["P_3_cross_axis"] = {}
    for bg in ("parent_resampled", "sel"):
        cr, _ = cross_background(bg, g_true, fan, D["sel"], 0, W)
        R["P_3_cross_axis"][bg] = {
            "corr_with_true_cross": r4(float(np.corrcoef(cr, g_true[:, 1])[0, 1])),
            "mae_m": r4(float(np.abs(cr - g_true[:, 1]).mean())),
            "future_derived_by_construction": bool(bg == "parent_resampled"),
            "_what": ("true_cross + a resampled residual -- the truth is IN it"
                      if bg == "parent_resampled" else
                      "the REF-C selector's own endpoint -- no truth in it")}
    print(f"[P-3] {json.dumps(R['P_3_cross_axis'])}", flush=True)

    R["_wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(R, indent=1))
    print(f"[probe] -> {a.out} ({R['_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
