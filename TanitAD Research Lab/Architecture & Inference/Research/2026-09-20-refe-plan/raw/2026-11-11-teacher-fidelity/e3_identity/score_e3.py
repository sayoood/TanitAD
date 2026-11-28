#!/usr/bin/env python3
"""E3 scorer (checkpoint identity): reads each nuPlan run's aggregator parquet and runner report, per e3_identity/HANDOFF.md.

final = the row scenario == 'final_score' of the newest *weighted_average*.parquet (num_scenarios not null), score x 100.
Checks: runner_report has the expected number of rows (261 for Test14-random) and every row succeeded.
Outcome rules (TEACHER_REVIEW.md s6, E3, fixed before any number):
  A = both runs within ~1 point of Table A10's 94.50 (checkpoint consistent with the paper's);
  B = both runs >= 2 points below 94.50;
  otherwise "in between", reported as such, no new bar.
usage: score_e3.py <run_out_dir> [<run_out_dir> ...] --out <json> [--expect 261]
"""
import argparse
import glob
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--expect", type=int, default=261)
a = ap.parse_args()
PAPER = 94.50
res = {"paper_table_A10_test14_random_nr": PAPER, "runs": {}}
for d in a.runs:
    r = {"out_dir": d}
    pq = sorted(glob.glob(os.path.join(d, "aggregator_metric", "*weighted_average*.parquet")), key=os.path.getmtime)
    rr = glob.glob(os.path.join(d, "runner_report.parquet"))
    r["aggregator_files"] = [os.path.basename(p) for p in pq]
    if not pq:
        r["error"] = "no weighted_average aggregator parquet"
    else:
        df = pd.read_parquet(pq[-1])
        fin = df[(df["scenario"] == "final_score") & df["num_scenarios"].notna()]
        if len(fin) != 1:
            r["error"] = f"final_score rows: {len(fin)}"
        else:
            row = fin.iloc[0]
            r["final_score_x100"] = round(float(row["score"]) * 100, 3)
            r["num_scenarios"] = int(row["num_scenarios"])
            r["final_row_metrics"] = {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float)) else str(v)))
                                      for k, v in row.items()}
    if rr:
        rep = pd.read_parquet(rr[0])
        r["runner_rows"] = int(len(rep))
        r["runner_succeeded_all"] = bool(rep["succeeded"].all()) if "succeeded" in rep else None
        r["runner_failed"] = int((~rep["succeeded"]).sum()) if "succeeded" in rep else None
    else:
        r["runner_rows"] = None
    r["coverage_ok"] = r.get("runner_rows") == a.expect and r.get("runner_succeeded_all") is True
    res["runs"][os.path.basename(os.path.dirname(d.rstrip("/\\"))) or d] = r
scores = [v.get("final_score_x100") for v in res["runs"].values()]
if all(s is not None for s in scores) and len(scores) >= 2:
    res["spread_abs"] = round(max(scores) - min(scores), 3)
    res["minus_paper"] = [round(s - PAPER, 3) for s in scores]
    if all(abs(s - PAPER) <= 1.0 for s in scores):
        res["outcome"] = "A (both within 1 point of 94.50)"
    elif all(s <= PAPER - 2.0 for s in scores):
        res["outcome"] = "B (both >= 2 points below 94.50)"
    else:
        res["outcome"] = "in between (no new bar)"
res["all_coverage_ok"] = all(v.get("coverage_ok") for v in res["runs"].values())
with open(a.out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))
for k, v in res["runs"].items():
    print(k, v.get("final_score_x100"), v.get("num_scenarios"), v.get("runner_rows"), v.get("runner_succeeded_all"), v.get("error"))
print("ZZE3_SCORED" if res["all_coverage_ok"] and "outcome" in res else "ZZE3_INCOMPLETE")
