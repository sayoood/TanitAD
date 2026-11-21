#!/usr/bin/env python3
"""E-GOAL-3 S4 -- ⭐ THE DISCRIMINATING EXPERIMENT between this stream and E-GOAL-2.

THE CONFLICT. E-GOAL-2: *"64 % of the recovery is speed history"* -- dropping
`dv_*` / `v_lag_*` cost 0.9305 -> 1.0733 m along-track RMS (13.3 %) on the DEV
corpus. E-GOAL-3, same nominal column sets, on the canonical val600 windows:
0.7449 -> 0.7628 m (2.3 %), and the ENTIRE realised lever is `v` + `ax`
(`H_ego` vs `H_v0_ax` is a NULL at T-OOF).

TWO CANDIDATE CAUSES, and they have opposite consequences for v5:
  (a) the corpus / window set differs, or
  (b) ⭐ `ax` QUALITY: E-GOAL-1's `ax` is `egomotion`'s NATIVE acceleration
      channel interpolated at t (`lead_state_gate.ego_frame` -> `at(t,"ax")`);
      mine is `(v[L] - v[L-1]) / 0.1`, a finite difference of the cache's own
      speed channel. If the native channel is the weak link, then what E-GOAL-2
      attributed to "1 s of speed history" was really "the DERIVATIVE OF SPEED,
      which the native `ax` column failed to supply" -- and the two streams
      agree on the mechanism while disagreeing on the required history length.

THE TEST, on E-GOAL-1's OWN parquet, with E-GOAL-1's OWN fitter and folds:
  `E1_nohist`        {v, ax_native, ay, curv, abs_curv, yawrate}  -> must
                     reproduce 1.0733 m (fidelity)
  `E1_nohist_axfd`   the same, with `ax` REPLACED by a within-clip finite
                     difference of `v` over one 0.1 s grid step
  `E1_v_ax`          {v, ax_native} only
  `E1_v_axfd`        {v, ax_finite_difference} only        ⭐ the E-GOAL-3 arm
  `E1_v` / `E1_ego`  the two anchors

⚠️ The dev-box corpus is keyed `14231cd29c74`, NOT parity -- this probe is about
the FEATURE, not about a headline number, and no parity-dependent quantity is
computed here.

Run:  OMP_NUM_THREADS=6 python e3_axprobe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STREAM = HERE.parent
EG1 = STREAM.parent / "2026-07-27-egoal-1-lead-vehicle"
sys.path.insert(0, str(EG1 / "code"))

from eg_common import SEED, clip_folds, ci_paired, r4, sep  # noqa: E402
from eg_fit import fit_predict  # noqa: E402

DT = 0.1
EGO = ["v", "ax", "ay", "curv", "abs_curv", "yawrate",
       "dv_0p5", "dv_1p0", "v_lag_0p5", "v_lag_1p0"]
NOHIST = ["v", "ax", "ay", "curv", "abs_curv", "yawrate"]


def main():
    t0 = time.time()
    wdf = pd.read_parquet(EG1 / "raw" / "eg_windows.parquet")
    keep = (np.isfinite(wdf["y_long"]) & np.isfinite(wdf["v"])
            & (wdf["obst_cov"] > 0))
    df = wdf.loc[keep].reset_index(drop=True)
    y = df["y_long"].to_numpy(np.float64)
    clips = df["clip"].to_numpy(str)

    # ---- the finite-difference ax, WITHIN clip, strictly backward -----------
    # rows are a 10 Hz grid inside each clip; `t_s` makes the step explicit so a
    # clip boundary or a gap can never be differenced across.
    t_s = df["t_s"].to_numpy(np.float64)
    v = df["v"].to_numpy(np.float64)
    ax_fd = np.full(len(df), np.nan)
    for c in np.unique(clips):
        m = np.flatnonzero(clips == c)
        o = m[np.argsort(t_s[m])]
        dt = np.diff(t_s[o])
        dv = np.diff(v[o])
        ok = np.abs(dt - DT) < 1e-6
        vals = np.where(ok, dv / np.where(dt == 0, np.nan, dt), np.nan)
        ax_fd[o[1:]] = vals
        ax_fd[o[0]] = 0.0                      # no earlier grid point: 0, causal
    n_bad = int(np.isnan(ax_fd).sum())
    ax_fd = np.where(np.isnan(ax_fd), 0.0, ax_fd)

    arms = {
        "E1_ego": df[EGO].to_numpy(np.float64),
        "E1_nohist": df[NOHIST].to_numpy(np.float64),
        "E1_nohist_axfd": np.column_stack(
            [df[["v"]].to_numpy(np.float64), ax_fd,
             df[["ay", "curv", "abs_curv", "yawrate"]].to_numpy(np.float64)]),
        "E1_v": df[["v"]].to_numpy(np.float64),
        "E1_v_ax": df[["v", "ax"]].to_numpy(np.float64),
        "E1_v_axfd": np.column_stack([df[["v"]].to_numpy(np.float64), ax_fd]),
    }
    pred = {k: np.full(len(y), np.nan) for k in arms}
    for f, (tr, te) in enumerate(clip_folds(clips, k=5, seed=SEED)):
        for k, X in arms.items():
            pred[k][te] = fit_predict(X[tr], y[tr], X[te], "gbm")
        print(f"  fold {f+1}/5 ({time.time()-t0:.0f}s)", flush=True)

    def rms(e):
        return float(np.sqrt(np.mean(e ** 2)))

    res = {"_stream": "2026-07-28-egoal-3-trained-head",
           "_stage": "S4 -- the ax-quality discriminating experiment",
           "_corpus": ("E-GOAL-1's dev corpus (key 14231cd29c74, NOT parity) -- "
                       "this probe is about the FEATURE, not a headline number"),
           "_n_windows": int(len(y)), "_n_clips": int(len(np.unique(clips))),
           "_ax_fd": {"what": "(v[t] - v[t-0.1]) / 0.1, WITHIN clip, backward",
                      "n_rows_without_a_valid_predecessor": n_bad,
                      "corr_with_native_ax": r4(float(
                          np.corrcoef(ax_fd, df["ax"].to_numpy(np.float64))[0, 1])),
                      "native_ax_sd": r4(float(df["ax"].std())),
                      "ax_fd_sd": r4(float(np.std(ax_fd)))},
           "along_rms_m": {k: r4(rms(pred[k] - y)) for k in arms},
           "along_mae_m": {k: r4(float(np.mean(np.abs(pred[k] - y))))
                           for k in arms},
           "EGOAL_1_published": {"E1_ego": 0.9305, "E1_nohist": 1.0733,
                                 "E0_v0": 1.5246},
           "EGOAL_3_val600_T_OOF": {"H_ego": 0.7449, "H_nohist": 0.7628,
                                    "H_v0_ax": 0.7519, "H_v0": 1.4500}}
    res["FIDELITY"] = {
        "E1_ego_reproduces_0p9305": bool(
            abs(res["along_rms_m"]["E1_ego"] - 0.9305) < 5e-3),
        "E1_nohist_reproduces_1p0733": bool(
            abs(res["along_rms_m"]["E1_nohist"] - 1.0733) < 5e-3)}

    pairs = [("E1_ego", "E1_nohist", "E-GOAL-2's history block, native ax"),
             ("E1_ego", "E1_nohist_axfd",
              "the same, with ax replaced by a finite difference"),
             ("E1_nohist_axfd", "E1_nohist", "⭐ ax_fd vs native ax"),
             ("E1_v_axfd", "E1_v_ax", "⭐ the same, minimal pair"),
             ("E1_v_ax", "E1_v", "what the NATIVE ax buys over v alone"),
             ("E1_v_axfd", "E1_v", "what a FINITE-DIFFERENCE ax buys over v"),
             ("E1_ego", "E1_v_axfd", "everything beyond v + ax_fd")]
    res["contrasts_rms"] = {}
    for x, z, why in pairs:
        ex, ez = np.abs(pred[x] - y), np.abs(pred[z] - y)
        c = ci_paired(ex, ez, clips)          # MAE-family, the powered axis
        res["contrasts_rms"][f"{x}__vs__{z}"] = {
            "_what": why,
            "delta_rms_m": r4(rms(pred[x] - y) - rms(pred[z] - y)),
            "paired_mae_delta": c, "separated": sep(c)}
        print(f"  {x:16s} vs {z:16s} dRMS={res['contrasts_rms'][f'{x}__vs__{z}']['delta_rms_m']:+.4f} "
              f"dMAE={c['delta']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] "
              f"{'SEP' if sep(c) else 'null'}", flush=True)

    res["_wall_s"] = round(time.time() - t0, 1)
    out = STREAM / "raw" / "e3_axprobe.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["along_rms_m"], indent=1), flush=True)
    print(f"-> {out} ({res['_wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
