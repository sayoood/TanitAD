#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The exact-assignment lever: `agent_slots.hungarian` (numpy, pure Python loop) vs
`scipy.optimize.linear_sum_assignment` (C++), on REAL cost matrices banked by the step
profiler (`--lever capture_costs` -> cost_bank.npz, float64 [N_queries, A_kept]).

For every matrix: both solvers, the SAME pair order (`hungarian`'s: target order when the
matrix is transposed), then
  * IDENTICAL = the returned (rows, cols) arrays are equal element for element;
  * the two assignment costs (sum of the chosen entries) and their difference;
  * ties: a matrix with a non-unique optimum can legally return different arrays, so the
    count of such cases is reported rather than assumed away.
Timing: best-of-3 wall per matrix for each solver, summed. CPU only; no torch GPU use.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def scipy_hungarian(cost):
    from scipy.optimize import linear_sum_assignment as lsa
    c = np.asarray(cost, dtype=np.float64)
    if c.size == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if c.shape[0] > c.shape[1]:
        r, k = lsa(c.T)
        return k.astype(np.int64), r.astype(np.int64)
    r, k = lsa(c)
    return r.astype(np.int64), k.astype(np.int64)


def best_of(fn, c, n=3):
    best = float("inf")
    out = None
    for _ in range(n):
        t = time.perf_counter()
        out = fn(c)
        best = min(best, time.perf_counter() - t)
    return best, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--bank", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.tree) / "stack"))
    from tanitad.models import agent_slots as AS
    mats = []
    for b in a.bank:
        z = np.load(b)
        mats += [z[k] for k in sorted(z.files)]
    rows = []
    for c in mats:
        t_h, (rh, ch) = best_of(AS.hungarian, c)
        t_s, (rs, cs) = best_of(scipy_hungarian, c)
        same = (np.array_equal(rh, rs) and np.array_equal(ch, cs))
        cost_h = float(c[rh, ch].sum())
        cost_s = float(c[rs, cs].sum())
        rows.append({"shape": list(c.shape), "t_hungarian_s": t_h, "t_scipy_s": t_s,
                     "identical": bool(same), "cost_h": cost_h, "cost_s": cost_s,
                     "cost_diff": cost_h - cost_s,
                     "dup_rows": int(c.shape[0] - np.unique(c, axis=0).shape[0])})
    n = len(rows)
    res = {"n_matrices": n,
           "shapes_A": {"min": min(r["shape"][1] for r in rows) if rows else None,
                        "median": statistics.median(r["shape"][1] for r in rows) if rows else None,
                        "max": max(r["shape"][1] for r in rows) if rows else None},
           "n_queries": sorted({r["shape"][0] for r in rows}),
           "identical": sum(r["identical"] for r in rows),
           "non_identical": [r for r in rows if not r["identical"]][:20],
           "max_abs_cost_diff": max((abs(r["cost_diff"]) for r in rows), default=None),
           "matrices_with_duplicate_query_rows": sum(r["dup_rows"] > 0 for r in rows),
           "t_hungarian_total_s": sum(r["t_hungarian_s"] for r in rows),
           "t_scipy_total_s": sum(r["t_scipy_s"] for r in rows),
           "t_hungarian_median_ms": 1e3 * statistics.median(r["t_hungarian_s"] for r in rows) if rows else None,
           "t_scipy_median_ms": 1e3 * statistics.median(r["t_scipy_s"] for r in rows) if rows else None,
           "host_note": "dev box i9-12900F, single thread; Thor's ARM core is slower (ratio UNVERIFIED)"}
    res["speedup"] = (res["t_hungarian_total_s"] / res["t_scipy_total_s"]
                      if res["t_scipy_total_s"] else None)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "non_identical"}, indent=1))


if __name__ == "__main__":
    main()
