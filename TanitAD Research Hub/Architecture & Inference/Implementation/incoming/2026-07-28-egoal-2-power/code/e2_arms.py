#!/usr/bin/env python3
"""E-GOAL-2 -- the BOTH-DIRECTIONS validation arms (PRE_REGISTRATION §5).

Three pools, all fitted with E-GOAL-1's machinery IMPORTED, not re-implemented
(`eg_fit.fit_predict`, `eg_common.clip_folds`, identical GBM hyper-parameters,
identical 5 clip-disjoint folds, identical seed):

  `E1_ego_REPRO`   -- E-GOAL-1's registered primary, RE-FIT from the parquet.
                      ⭐ THE FIDELITY CONTROL. It must reproduce the frozen pool
                      in `eg_oof_pred_gbm.npz` to ~machine precision. If it does
                      not, this stream's fitting path differs from E-GOAL-1's
                      and the two extra arms below mean nothing.
  `E1_nohist`      -- `dv_*` / `v_lag_*` DROPPED (E-GOAL-1's R3 ablation).
                      The honest reference the noise arm must land on.
  `E1_noise_hist`  -- ⛔ THE DELIBERATELY FAILING INPUT. The same four history
                      columns REPLACED by Gaussian noise matched to each
                      column's own mean and SD. The regressor still receives a
                      10-column matrix of the same shape, so any "help" it finds
                      is fitting noise. A pure-noise history MUST recover
                      nothing beyond `E1_nohist`.

⚠️ Why the noise arm and not just the ablation: dropping columns changes the
model's capacity as well as its information. Replacing them with noise holds
capacity FIXED and removes only the information -- which is the control that can
actually fail.

Run on the dev box (CPU):
    OMP_NUM_THREADS=6 python e2_arms.py
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

from eg_common import EGO_COLS, SEED, clip_folds, r4  # noqa: E402
from eg_fit import fit_predict  # noqa: E402

HIST_COLS = ["dv_0p5", "dv_1p0", "v_lag_0p5", "v_lag_1p0"]
NOHIST = [c for c in EGO_COLS if c not in HIST_COLS]


def oof(df, y, arms, k=5, seed=SEED):
    pred = {a: np.full(len(y), np.nan) for a in arms}
    clips = df["clip"].to_numpy(str)
    for f, (tr, te) in enumerate(clip_folds(clips, k=k, seed=seed)):
        for a, X in arms.items():
            pred[a][te] = fit_predict(X[tr], y[tr], X[te], "gbm")
        print(f"    fold {f+1}/{k} done", flush=True)
    return pred


def main():
    t0 = time.time()
    wdf = pd.read_parquet(EG1 / "raw" / "eg_windows.parquet")
    keep = (np.isfinite(wdf["y_long"]) & np.isfinite(wdf["v"])
            & (wdf["obst_cov"] > 0))
    df = wdf.loc[keep].reset_index(drop=True)
    y = df["y_long"].to_numpy(np.float64)
    print(f"[arms] {len(df)} windows / {df['clip'].nunique()} clips", flush=True)

    ego = df[EGO_COLS].to_numpy(np.float64)
    nohist = df[NOHIST].to_numpy(np.float64)

    # ⛔ the failing input: SAME SHAPE, SAME COLUMN POSITIONS, NO INFORMATION
    rng = np.random.default_rng(20260728)
    noisy = ego.copy()
    noise_stats = {}
    for c in HIST_COLS:
        j = EGO_COLS.index(c)
        col = ego[:, j]
        mu, sd = float(np.nanmean(col)), float(np.nanstd(col))
        noisy[:, j] = rng.normal(mu, sd, len(col))
        noise_stats[c] = {"mu": r4(mu), "sd": r4(sd)}

    arms = {"E1_ego_REPRO": ego, "E1_nohist": nohist, "E1_noise_hist": noisy}
    pred = oof(df, y, arms)

    # ---------------- the fidelity control ---------------------------------
    z = np.load(EG1 / "raw" / "eg_oof_pred_gbm.npz", allow_pickle=True)
    frozen = z["pred_E1_ego"]
    assert len(frozen) == len(y), "row count differs from the frozen pool"
    dy = float(np.abs(z["y"] - y).max())
    dp = float(np.abs(frozen - pred["E1_ego_REPRO"]).max())
    ok = dy < 1e-9 and dp < 1e-6

    def rms(v):
        return float(np.sqrt(np.mean(v ** 2)))

    res = {"_stream": "2026-07-28-egoal-2-power", "_stage": "extra arms",
           "_n_windows": int(len(df)), "_n_clips": int(df["clip"].nunique()),
           "FIDELITY_refit_reproduces_frozen_pool": {
               "max_abs_dy": dy, "max_abs_dpred": dp, "passes": bool(ok),
               "what": ("E-GOAL-1's registered E1_ego pool, re-fit from the "
                        "parquet with the imported fitter. If this does not "
                        "reproduce, the extra arms are not comparable to the "
                        "frozen pools and nothing below may be quoted.")},
           "_noise_stats": noise_stats,
           "along_rms_m": {a: r4(rms(pred[a] - y)) for a in arms},
           "along_mae_m": {a: r4(float(np.mean(np.abs(pred[a] - y))))
                           for a in arms},
           "EGOAL_1_R3_reference": {"E1_ego": 0.9305, "E1_nohist": 1.0733},
           "_wall_s": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")},
                     indent=1), flush=True)
    if not ok:
        raise RuntimeError(
            f"FIDELITY FAILED: re-fit does not reproduce the frozen E1_ego "
            f"pool (max |dpred| = {dp:.3e}). The extra arms are void.")

    out = STREAM / "raw"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "e2_extra_pools.npz", y=y,
                        **{f"pred_{a}": pred[a] for a in
                           ("E1_nohist", "E1_noise_hist")})
    (out / "e2_arms.json").write_text(json.dumps(res, indent=1))
    print(f"[arms] -> {out/'e2_extra_pools.npz'} + e2_arms.json", flush=True)


if __name__ == "__main__":
    main()
