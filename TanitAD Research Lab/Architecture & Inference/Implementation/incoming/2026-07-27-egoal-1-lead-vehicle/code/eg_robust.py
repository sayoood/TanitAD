#!/usr/bin/env python3
"""E-GOAL-1 robustness -- is the gain FEATURES or SAMPLE SIZE?

The obvious objection to this stream's headline, and it must be answered before
the headline is quoted: this fit sees ~100,000 windows over 612 clips, while the
parent's goal head saw 881 windows over 40 episodes. If ego kinematics look
better here merely because there is 100x more data, the finding is about n, not
about features, and it does not transfer to the canonical val set.

Three controls, each able to overturn the headline:

  R1 STRIDE-MATCHED  -- keep every 9th grid point (~19 windows/clip, matching the
     canonical set's ~22 windows/episode), same 612 clips. Removes the
     autocorrelated density without touching n_clips.
  R2 PARENT-REGIME   -- 40 clips x ~22 windows = ~880 windows, 5 clip-disjoint
     folds, repeated over 15 independent clip draws. This is the parent's EXACT
     data regime. If `E1_ego` still beats `E0_v0` here, the gain is features.
  R3 FEATURE-ABLATION -- drop the history block (`dv_*`, `v_lag_*`) from E1 and
     refit on the full set, isolating what the 1 s of speed history is worth.

Run:  OMP_NUM_THREADS=6 python eg_robust.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eg_common import (BAR_BREAKEVEN_M, BAR_HALF_M, EGO_COLS, HORIZON_S,  # noqa: E402
                       LEAD_COLS, SEED, XTRA_COLS, ci_paired, ci_single,
                       clip_folds, r4)
from eg_fit import fit_predict  # noqa: E402

HIST_COLS = ["dv_0p5", "dv_1p0", "v_lag_0p5", "v_lag_1p0"]
NOHIST = [c for c in EGO_COLS if c not in HIST_COLS]


def rms(e):
    return float(np.sqrt(np.mean(e ** 2)))


def run_oof(sub: pd.DataFrame, cols_by_arm: dict, kind="gbm", k=5, seed=SEED):
    y = sub["y_long"].to_numpy(np.float64)
    clips = sub["clip"].to_numpy(str)
    pred = {a: np.full(len(y), np.nan) for a in cols_by_arm}
    X = {a: sub[c].to_numpy(np.float64) for a, c in cols_by_arm.items()}
    for tr, te in clip_folds(clips, k=k, seed=seed):
        for a in cols_by_arm:
            pred[a][te] = fit_predict(X[a][tr], y[tr], X[a][te], kind)
    return y, clips, pred


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(Path(__file__).resolve().parent.parent
                                         / "raw"))
    ap.add_argument("--draws", type=int, default=15)
    a = ap.parse_args(argv)
    raw = Path(a.raw)

    df = pd.read_parquet(raw / "eg_windows.parquet")
    df = df[np.isfinite(df["y_long"]) & np.isfinite(df["v"])
            & (df["obst_cov"] > 0)].reset_index(drop=True)
    arms = {"E0_v0": ["v"], "E1_ego": EGO_COLS,
            "E1_nohist": NOHIST,
            "E1_L": EGO_COLS + LEAD_COLS,
            "E1_L_X": EGO_COLS + LEAD_COLS + XTRA_COLS}
    res = {"what": "E-GOAL-1 robustness -- features vs sample size",
           "bars": {"breakeven_m": BAR_BREAKEVEN_M, "halfprize_m": BAR_HALF_M}}

    # ------------------------------------------------------------------ R1 #
    print("[R1] stride-matched (every 9th grid point) ...", flush=True)
    s1 = df.iloc[::9].reset_index(drop=True)
    y, cl, p = run_oof(s1, arms)
    res["R1_stride_matched"] = {
        "what": "every 9th grid point; ~19 windows/clip vs the canonical ~22",
        "n_windows": int(len(s1)), "n_clips": int(s1["clip"].nunique()),
        "along_rms_m": {k: r4(rms(p[k] - y)) for k in arms},
        "speed_err_ms": {k: r4(rms(p[k] - y) / HORIZON_S) for k in arms},
        "paired_E1_ego_vs_E1_L": ci_paired((p["E1_ego"] - y) ** 2,
                                           (p["E1_L"] - y) ** 2, cl,
                                           reduce="rms"),
        "paired_E0_v0_vs_E1_ego": ci_paired((p["E0_v0"] - y) ** 2,
                                            (p["E1_ego"] - y) ** 2, cl,
                                            reduce="rms")}
    print("   ", json.dumps(res["R1_stride_matched"]["along_rms_m"]), flush=True)

    # ------------------------------------------------------------------ R2 #
    print(f"[R2] parent regime: 40 clips x ~22 windows, {a.draws} draws ...",
          flush=True)
    uniq = np.unique(df["clip"].to_numpy(str))
    rows = []
    for d in range(a.draws):
        rng = np.random.default_rng(500 + d)
        pick = set(rng.choice(uniq, 40, replace=False).tolist())
        sub = df[df["clip"].isin(pick)]
        sub = sub.iloc[::8].reset_index(drop=True)      # ~22 windows / clip
        y2, cl2, p2 = run_oof(sub, arms)
        rows.append({k: rms(p2[k] - y2) for k in arms}
                    | {"n_windows": int(len(sub))})
    R = pd.DataFrame(rows)
    res["R2_parent_regime"] = {
        "what": ("the parent's EXACT data regime: 40 clips, ~22 windows each, "
                 "5 clip-disjoint folds; repeated over independent clip draws"),
        "n_draws": a.draws,
        "median_n_windows": int(R["n_windows"].median()),
        "along_rms_m_median": {k: r4(R[k].median()) for k in arms},
        "along_rms_m_p10_p90": {k: [r4(R[k].quantile(0.10)),
                                    r4(R[k].quantile(0.90))] for k in arms},
        "frac_draws_E1_ego_beats_E0_v0": r4(float((R["E1_ego"] < R["E0_v0"]).mean())),
        "frac_draws_E1_L_beats_E1_ego": r4(float((R["E1_L"] < R["E1_ego"]).mean())),
        "frac_draws_E1_ego_clears_breakeven": r4(
            float((R["E1_ego"] < BAR_BREAKEVEN_M).mean())),
        "parent_head_reference_along_rms_m": 1.151}
    print("   ", json.dumps(res["R2_parent_regime"]["along_rms_m_median"]),
          flush=True)

    # ------------------------------------------------------------------ R3 #
    print("[R3] history ablation on the full set ...", flush=True)
    y3, cl3, p3 = run_oof(df, {"E1_ego": EGO_COLS, "E1_nohist": NOHIST})
    res["R3_history_ablation"] = {
        "what": "what the 1 s of speed/accel history is worth on its own",
        "dropped": HIST_COLS,
        "along_rms_m": {"E1_ego": r4(rms(p3["E1_ego"] - y3)),
                        "E1_nohist": r4(rms(p3["E1_nohist"] - y3))},
        "paired": ci_paired((p3["E1_nohist"] - y3) ** 2,
                            (p3["E1_ego"] - y3) ** 2, cl3, reduce="rms"),
        "paired_MAE": ci_paired(np.abs(p3["E1_nohist"] - y3),
                                np.abs(p3["E1_ego"] - y3), cl3, reduce="mean")}
    print("   ", json.dumps(res["R3_history_ablation"]["along_rms_m"]), flush=True)

    (raw / "eg_robust.json").write_text(json.dumps(res, indent=1))
    print(f"[robust] -> {raw/'eg_robust.json'}", flush=True)


if __name__ == "__main__":
    main()
