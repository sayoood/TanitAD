#!/usr/bin/env python3
"""Deliverable 1 — does the harness STILL read its known values? (any venv with pandas)

Compares this stream's re-scored E2 controls (``raw/harness_repro/score_<arm>.csv``) against
E2's BANKED official CSVs and E2's published ``raw/scores_summary.json``, cell by cell.

Committed before the comparison ran (the rule, not a post-hoc tolerance):
  * token sets must be IDENTICAL (220 = 16 stage-1 + 204 stage-2);
  * every numeric column of every token row must agree to |d| <= 1e-9 (the scorer is
    deterministic under PYTHONHASHSEED=1; E2 measured max |d| = 0.0 against E1's CV run);
  * S2-EPDMS-u (E2 SPEC §2, computed with E2's OWN ``s2_group_uniform``, imported) and the
    three ``extended_pdm_score_*`` summary rows must equal E2's published values to 1e-12.
Verdict per arm: REPRODUCED / NOT_REPRODUCED (with the offending columns named). A
NOT_REPRODUCED control STOPS the refcv6 scoring (the brief's rule: a harness that does not
read its known values voids every number scored through it).

    python code/check_harness_repro.py  ->  raw/HARNESS_REPRO.json
    python code/check_harness_repro.py --mine-dir raw/scores_warmup_s1000 --arms ECHO_ha0_ext         --out raw/controls/KH_ECHO_warmup.json      # the THIRD model-free control, same rule
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
E2 = os.path.abspath(os.path.join(PKG, "..", "..", "2026-09-19-navsim-refcv4b-bridge"))
MINE = os.path.join(PKG, "raw", "harness_repro")
ARMS = ("CV_official", "STOP_zero")
TOL_CELL = 1e-9
TOL_STAT = 1e-12


def _e2_parse():
    spec = importlib.util.spec_from_file_location(
        "e2_parse_scores", os.path.join(E2, "code", "parse_scores.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine-dir", default=MINE)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=os.path.join(PKG, "raw", "HARNESS_REPRO.json"))
    cli = ap.parse_args(argv)
    mine_dir, arms = cli.mine_dir, [x for x in cli.arms.split(",") if x]
    ps = _e2_parse()
    doc = json.load(open(os.path.join(E2, "raw", "navsim_agent_inputs.json"), encoding="utf-8"))
    stage_of = {t: r["stage"] for t, r in doc["tokens"].items()}
    mapping = doc["reactive_all_mapping"]
    pub = json.load(open(os.path.join(E2, "raw", "scores_summary.json"), encoding="utf-8"))
    out = {"_rule": {"cell_tol": TOL_CELL, "stat_tol": TOL_STAT,
                     "statistic": "S2-EPDMS-u via E2's own parse_scores.s2_group_uniform "
                                  "(imported), + the 3 official extended_pdm_score_* rows"},
           "e2_package": E2, "arms": {}}
    ok_all = True
    for arm in arms:
        pa, pb = os.path.join(mine_dir, f"score_{arm}.csv"), os.path.join(E2, "raw", f"score_{arm}.csv")
        rec = {"mine": pa, "e2": pb}
        if not (os.path.exists(pa) and os.path.exists(pb)):
            rec["verdict"] = "NOT_REPRODUCED"
            rec["reason"] = f"missing csv: mine={os.path.exists(pa)} e2={os.path.exists(pb)}"
            out["arms"][arm] = rec
            ok_all = False
            continue
        a, b = pd.read_csv(pa), pd.read_csv(pb)
        ta = a[~a.token.str.startswith("extended_pdm_score")].set_index("token").sort_index()
        tb = b[~b.token.str.startswith("extended_pdm_score")].set_index("token").sort_index()
        rec["n_tokens_mine"], rec["n_tokens_e2"] = int(len(ta)), int(len(tb))
        rec["token_sets_identical"] = bool(ta.index.equals(tb.index))
        bad_cols, max_abs = [], {}
        for col in tb.columns:
            if col not in ta.columns:
                bad_cols.append(f"{col}: absent in mine")
                continue
            if tb[col].dtype.kind in "fiub" and ta[col].dtype.kind in "fiub":
                x = ta[col].to_numpy(dtype=float)
                y = tb[col].to_numpy(dtype=float)
                both_nan = np.isnan(x) & np.isnan(y)
                d = np.where(both_nan, 0.0, np.abs(x - y))
                m = float(np.nanmax(d)) if d.size else 0.0
                if np.isnan(d).any():
                    m = float("inf")
                max_abs[col] = m
                if not (m <= TOL_CELL):
                    bad_cols.append(f"{col}: max|d|={m}")
            else:
                same = bool((ta[col].astype(str) == tb[col].astype(str)).all())
                if not same:
                    bad_cols.append(f"{col}: non-numeric cells differ")
        rec["max_abs_diff_by_column"] = max_abs
        rec["max_abs_diff_any_column"] = max(max_abs.values()) if max_abs else None
        df = ta.reset_index()
        df["stage"] = df.token.map(stage_of)
        u = ps.s2_group_uniform(df, mapping)
        rec["S2_EPDMS_u_mine"] = u.get("value")
        rec["S2_EPDMS_u_e2_published"] = pub["arms"][arm]["S2_EPDMS_u"]["value"]
        rec["S2_EPDMS_u_abs_diff"] = (abs(rec["S2_EPDMS_u_mine"] - rec["S2_EPDMS_u_e2_published"])
                                      if rec["S2_EPDMS_u_mine"] is not None else None)
        sa = a[a.token.str.startswith("extended_pdm_score")].set_index("token")["score"]
        rec["official_rows_mine"] = {k: float(v) for k, v in sa.items()}
        rec["official_rows_e2_published"] = pub["arms"][arm]["official_summary_rows"]
        row_diffs = {k: abs(rec["official_rows_mine"].get(k, float("nan")) - v)
                     for k, v in rec["official_rows_e2_published"].items()}
        rec["official_rows_abs_diff"] = row_diffs
        stat_ok = (rec["S2_EPDMS_u_abs_diff"] is not None
                   and rec["S2_EPDMS_u_abs_diff"] <= TOL_STAT
                   and all(v <= TOL_STAT for v in row_diffs.values()))
        good = rec["token_sets_identical"] and not bad_cols and stat_ok
        rec["offending"] = bad_cols
        rec["verdict"] = "REPRODUCED" if good else "NOT_REPRODUCED"
        ok_all &= good
        cnt = os.path.join(mine_dir, f"score_{arm}.counts.json")
        if os.path.exists(cnt):
            c = json.load(open(cnt, encoding="utf-8"))
            rec["count_guard"] = {k: c.get(k) for k in ("status", "log_successful", "log_failed",
                                                        "csv_valid_rows", "agent_calls",
                                                        "wall_s", "wrapper_sha256",
                                                        "seam_sha256", "seam_agent_sha256")}
            ok_all &= c.get("status") == "PASS"
        out["arms"][arm] = rec
    out["verdict"] = "HARNESS_REPRODUCED" if ok_all else "HARNESS_NOT_REPRODUCED"
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    with open(cli.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    for arm, r in out["arms"].items():
        print(arm, r["verdict"], "S2u mine", r.get("S2_EPDMS_u_mine"), "e2", r.get("S2_EPDMS_u_e2_published"),
              "max|d| cell", r.get("max_abs_diff_any_column"), "rows", r.get("official_rows_abs_diff"))
    print(out["verdict"])
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
