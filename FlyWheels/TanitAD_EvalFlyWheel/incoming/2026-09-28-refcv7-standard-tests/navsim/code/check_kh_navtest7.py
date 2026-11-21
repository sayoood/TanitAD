#!/usr/bin/env python3
"""KH-navtest — the navtest harness reads its KNOWN value through THIS package's driver (any venv
with pandas). Compares a model-free control re-scored by ``code/score_navtest7.py --official <X>``
with W3's BANKED CSV, cell for cell.

Rule, committed before the comparison (the refcv6 suite's KH form, SPEC §7): the token sets are
IDENTICAL, every numeric cell agrees to |d| <= 1e-9, the count guard reads PASS, and PDMS x100
(mean of ``score``) is equal. A cell NaN on ONE side only is a MISMATCH (never ``nanmax``-skipped).
No token is ever printed.

    python code/check_kh_navtest7.py --mine raw/harness_navtest/r7kh_STOP/r7kh_STOP.csv \
        --w3 <W3>/raw/STOP_navtest/STOP_navtest.csv --out raw/controls/KH_navtest_STOP.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

TOL = 1e-9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True)
    ap.add_argument("--w3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = {"control": "KH-navtest: a model-free control re-scored through THIS package's driver "
                      "vs W3's banked CSV, cell for cell", "mine": a.mine, "w3": a.w3,
           "rule": "same tokens AND every numeric cell |d| <= 1e-9 AND count guard PASS"}
    if not (os.path.exists(a.mine) and os.path.exists(a.w3)):
        rep["verdict"] = "NOT_REPRODUCED"
        rep["reason"] = f"missing csv: mine={os.path.exists(a.mine)} w3={os.path.exists(a.w3)}"
    else:
        cnt = a.mine[:-4] + ".counts.json"
        cg = json.load(open(cnt, encoding="utf-8")) if os.path.exists(cnt) else {}
        rep["count_guard"] = {k: cg.get(k) for k in ("status", "log_successful", "log_failed",
                                                     "csv_valid_rows", "C1_max_abs_delta",
                                                     "wall_s")}
        x = pd.read_csv(a.mine).set_index("token").sort_index()
        y = pd.read_csv(a.w3).set_index("token").sort_index()
        x = x.drop(columns=[c for c in x.columns if c.startswith("Unnamed")])
        y = y.drop(columns=[c for c in y.columns if c.startswith("Unnamed")])
        rep["n_mine"], rep["n_w3"] = int(len(x)), int(len(y))
        rep["same_tokens"] = bool(x.index.equals(y.index))
        worst, bad = 0.0, []
        n_num = 0
        for c in y.columns:
            if c not in x.columns:
                bad.append(f"{c}: absent in mine")
                continue
            if y[c].dtype.kind in "fiub" and x[c].dtype.kind in "fiub" and rep["same_tokens"]:
                n_num += 1
                u = x[c].to_numpy(dtype=float)
                v = y[c].to_numpy(dtype=float)
                bn = np.isnan(u) & np.isnan(v)
                d = np.where(bn, 0.0, np.abs(u - v))
                m = float("inf") if np.isnan(d).any() else float(d.max() if d.size else 0.0)
                worst = max(worst, m)
                if not m <= TOL:
                    bad.append(f"{c}: max|d|={m}")
            elif rep["same_tokens"] and not (x[c].astype(str) == y[c].astype(str)).all():
                bad.append(f"{c}: non-numeric cells differ")
        rep["n_numeric_cols"] = n_num
        rep["max_abs_cell"] = worst
        rep["offending"] = bad
        rep["PDMS_x100_mine"] = round(float(x["score"].mean()) * 100, 5)
        rep["PDMS_x100_w3"] = round(float(y["score"].mean()) * 100, 5)
        ok = (rep["same_tokens"] and not bad and cg.get("status") == "PASS"
              and rep["PDMS_x100_mine"] == rep["PDMS_x100_w3"])
        rep["verdict"] = "REPRODUCED" if ok else "NOT_REPRODUCED"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({k: rep.get(k) for k in ("verdict", "n_mine", "same_tokens", "max_abs_cell",
                                              "PDMS_x100_mine", "PDMS_x100_w3")}))
    return 0 if rep["verdict"] == "REPRODUCED" else 1


if __name__ == "__main__":
    sys.exit(main())
