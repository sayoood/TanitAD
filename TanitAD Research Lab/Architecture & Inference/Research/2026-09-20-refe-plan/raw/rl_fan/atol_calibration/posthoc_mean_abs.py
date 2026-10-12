#!/usr/bin/env python3
"""POST-HOC, EXPLORATORY (written AFTER `run_calibration.py --phase analyse` had returned CHOSEN_ATOL = null; NOT pre-registered, so nothing here may decide anything).

Question the verdict leaves open: the per-set MAX delta is dominated by one bf16-quantised outlier (the floor is exactly 0.25 on every null arm), so is there a statistic of the
same delta distributions that separates the regression arms from the floor by a wider margin?  Candidate: the per-set MEAN |delta| over the set's x / y values.
For each arm: how many sets exceed a mean-abs bound of 0.0125 / 0.025 / 0.05 / 0.1, and the floor over the NULL arms.  CPU only, reads the dumps in the work dir.

    python posthoc_mean_abs.py [--work D:/Projects/TanitAD-artifacts/_rlfan_work]      ->  posthoc_mean_abs.json  (+ a table on stdout)

Replicating this on DISJOINT keys under a pre-registered rule is the next step (HANDOFF.md in this directory); until then it is a lead, not a result.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_calibration as RC  # noqa: E402

BOUNDS = [0.0125, 0.025, 0.05, 0.1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="D:/Projects/TanitAD-artifacts/_rlfan_work")
    a = ap.parse_args()
    W = a.work.replace("\\", "/")
    A = f"{W}/atol_cal"
    ref = RC.load_sets(f"{W}/sets_10075_proxy.jsonl", RC.STEP)
    out = {}
    idn = f"{W}/identity"
    srcs = [("N_full", f"{idn}/new_full/onpolicy_rlfan_full_s0of1.jsonl", "null")]
    for name, kind, var in RC.ARMS:
        f = f"{A}/runs/{name}/onpolicy_rlfan_cal_s0of1.jsonl"
        r = f"{A}/runs/{name}/relabelled_step{RC.STEP}.jsonl"
        if os.path.exists(r):
            f = r
        if os.path.exists(f):
            srcs.append((name, f, kind))
    print(f"{'arm':26s} {'kind':9s} {'sets':>5s} {'mean-abs p50':>12s} {'p90':>8s} {'max':>8s}  " + "  ".join(f">{b:<6g}" for b in BOUNDS))
    for name, path, kind in srcs:
        d = RC.arm_deltas(path, RC.STEP, ref)
        means = sorted(x[3] for x in d.values())
        row = {"kind": kind, "sets": len(means), "mean_abs_p50": RC.RF.quantile_nearest_rank(means, 0.5), "mean_abs_p90": RC.RF.quantile_nearest_rank(means, 0.9),
               "mean_abs_max": means[-1], "sets_above": {str(b): sum(1 for m in means if m > b) for b in BOUNDS}}
        out[name] = row
        print(f"{name:26s} {kind:9s} {len(means):5d} {row['mean_abs_p50']:12.5f} {row['mean_abs_p90']:8.5f} {row['mean_abs_max']:8.5f}  " + "  ".join(f"{row['sets_above'][str(b)]:<7d}" for b in BOUNDS))
    nulls = [v for n, v in out.items() if v["kind"] == "null"]
    out["_floor_over_null_arms"] = {"mean_abs_max": max(v["mean_abs_max"] for v in nulls), "note": "POST-HOC; N_full is the 1,365-key cross-GPU dump of the identity run, N0..N2 this run's 48-key nulls"}
    RC.wr_json(os.path.join(HERE, "posthoc_mean_abs.json"), out)


if __name__ == "__main__":
    main()
