#!/usr/bin/env python3
"""E-GOAL-1 S0 -- the fidelity gate. Reproduce before quoting.

Three controls, run BEFORE any new number is quoted, each able to FAIL:

 F1  PER-ROW IDENTITY against the repo's own prior dump
     (`lead_gate_windows.parquet`, produced 2026-07-21 by
     `stack/scripts/lead_state_gate.py`). My ingest CALLS that reader, so every
     overlapping (clip, t) row must agree to 0.0 EXACTLY on `v`, `y_long`,
     `gap_m`, `closing_ms`, `ttc_s`, `lead_present`, `n_ahead_50m`. This is an
     exact per-row identity, not a mean-vs-mean comparison, so it cannot be
     passed by a wrong join. (The parent stream's Fidelity B is the same shape,
     and it is the control that actually caught a bug.)

 F2  COVERAGE RECONCILIATION -- and the number that proves the span defect is
     real. The committed `lead_gate_result.json` reports
     `lead_present_frac_windows = 0.3851` over ALL windows. My covered-only rate
     is higher. If the difference is exactly the uncovered fraction, then the
     committed rate is diluted by windows where `obstacle.offline` HAS NO DATA
     and the reader silently recorded "no lead vehicle". PREDICTED, before
     reading: `rate_all ≈ rate_covered × frac_covered`.

 F3  FAILING CONTROL for F1 -- a deliberately mis-joined row set (t shifted by
     +0.1 s) must FAIL the identity, or F1 has no power.

Run:  OMP_NUM_THREADS=6 python eg_gate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eg_common import ROOT, clip_alias, r4  # noqa: E402

PRIOR = ROOT / "lead_gate" / "lead_gate_windows.parquet"
COMMITTED = (Path(__file__).resolve().parents[5] / "Data Engineering"
             / "Implementation" / "incoming" / "2026-07-21-lead-state-gate"
             / "lead_gate_result.json")
CHECK_COLS = ["v", "ax", "ay", "curv", "yawrate", "dv_0p5", "dv_1p0",
              "v_lag_0p5", "v_lag_1p0", "abs_curv", "y_long", "y_lat",
              "lead_present", "gap_m", "closing_ms", "ttc_s", "inv_ttc",
              "lead_lat_m", "lead_is_big", "n_ahead_50m", "n_vru_near"]


def _key(df, alias_col, tshift=0.0):
    return pd.Index([f"{c}@{t + tshift:.4f}" for c, t in
                     zip(df[alias_col].to_numpy(str),
                         df["t_s"].to_numpy(np.float64))])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(Path(__file__).resolve().parent.parent
                                         / "raw"))
    a = ap.parse_args(argv)
    raw = Path(a.raw)

    mine = pd.read_parquet(raw / "eg_windows.parquet")
    res = {"what": "E-GOAL-1 S0 fidelity gate",
           "mine": {"path": str(raw / "eg_windows.parquet"),
                    "n_rows": int(len(mine)),
                    "n_clips": int(mine["clip"].nunique())}}

    # ------------------------------------------------------------ F1 / F3 #
    if PRIOR.exists():
        prior = pd.read_parquet(PRIOR)
        prior["clip"] = [clip_alias(c) for c in prior["clip_id"].astype(str)]
        km, kp = _key(mine, "clip"), _key(prior, "clip")
        mine_i = mine.set_index(km)
        prior_i = prior.set_index(kp)
        common = mine_i.index.intersection(prior_i.index)
        A, B = mine_i.loc[common], prior_i.loc[common]
        diffs = {}
        for c in CHECK_COLS:
            if c not in A.columns or c not in B.columns:
                continue
            x, y = A[c].to_numpy(np.float64), B[c].to_numpy(np.float64)
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.abs(np.where(both_nan, 0.0, x - y))
            diffs[c] = {"max_abs_diff": float(np.nanmax(d)),
                        "n_nan_mismatch": int((np.isnan(x) != np.isnan(y)).sum())}
        worst = max(v["max_abs_diff"] for v in diffs.values())
        nanmis = sum(v["n_nan_mismatch"] for v in diffs.values())
        res["F1_per_row_identity"] = {
            "what": ("every overlapping (clip, t) row must agree EXACTLY with the "
                     "repo reader's 2026-07-21 dump"),
            "prior_path": str(PRIOR), "prior_rows": int(len(prior)),
            "n_common_rows": int(len(common)),
            "max_abs_diff_over_all_columns": worst,
            "n_nan_mismatches": nanmis,
            "passes": bool(worst == 0.0 and nanmis == 0),
            "per_column": diffs}

        # F3 -- the same comparison with a deliberately wrong +0.1 s join
        kbad = _key(mine, "clip", tshift=0.1)
        mb = mine.set_index(kbad)
        cb = mb.index.intersection(prior_i.index)
        if len(cb):
            xb = mb.loc[cb, "v"].to_numpy(np.float64)
            yb = prior_i.loc[cb, "v"].to_numpy(np.float64)
            wb = float(np.nanmax(np.abs(xb - yb)))
        else:
            wb = float("nan")
        res["F3_failing_control"] = {
            "what": "a deliberately +0.1 s mis-joined row set must FAIL F1",
            "n_common_rows": int(len(cb)),
            "max_abs_diff_v": wb,
            "passes": bool(len(cb) > 0 and wb > 0.0)}
    else:
        res["F1_per_row_identity"] = {"passes": None,
                                      "why": f"{PRIOR} not present on this host"}

    # ---------------------------------------------------------------- F2 #
    cov = (mine["obst_cov"] > 0).to_numpy()
    fin = (np.isfinite(mine["y_long"]) & np.isfinite(mine["v"])).to_numpy()
    lp = mine["lead_present"].to_numpy(np.float64)
    rate_all = float(np.nanmean(lp[fin]))
    rate_cov = float(np.nanmean(lp[fin & cov]))
    frac_cov = float((fin & cov).sum() / fin.sum())
    committed = None
    if COMMITTED.exists():
        committed = json.loads(COMMITTED.read_text())["corpus"][
            "lead_present_frac_windows"]
    res["F2_coverage_reconciliation"] = {
        "what": ("the committed 0.3851 lead rate should be the COVERED rate "
                 "diluted by uncovered windows the prior reader could not "
                 "distinguish from an empty road"),
        "committed_rate_all_windows": committed,
        "mine_rate_all_windows": r4(rate_all),
        "mine_rate_obstacle_covered_only": r4(rate_cov),
        "frac_windows_obstacle_covered": r4(frac_cov),
        "predicted_rate_all_from_covered": r4(rate_cov * frac_cov),
        "abs_error_of_prediction": r4(abs(rate_cov * frac_cov - rate_all)),
        "passes": bool(abs(rate_cov * frac_cov - rate_all) < 0.005),
        "implication": ("the prior dump's lead-presence rate is DEFLATED by the "
                        "uncovered windows; the corrected rate is the covered one"),
    }

    ok = [v.get("passes") for k, v in res.items() if isinstance(v, dict)
          and "passes" in v]
    res["_all_ok"] = all(x is True for x in ok)
    (raw / "eg_gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict)
                          else {kk: vv for kk, vv in v.items()
                                if kk != "per_column"})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
