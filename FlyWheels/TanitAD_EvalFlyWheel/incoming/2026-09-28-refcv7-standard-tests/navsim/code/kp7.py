#!/usr/bin/env python3
"""KP -- the PRECISION FLOOR (refcv6 SPEC amendment A3, carried here): the SAME checkpoint, the SAME
per-scene seeds, R7_A1 on warmup, CPU fp32 vs CUDA bf16 -- per-scene 4 s endpoint distance and
selection identity (plans), plus the S2-EPDMS-u difference when both summaries exist. Any python.

    python code/kp7.py --a <bridge dir A> --b <bridge dir B> [--sa summary_A --sb summary_B] --out <json>

A cross-checkpoint comparison whose devices differ is read against this floor (and the seed floor):
a difference inside either is not a training effect. No token is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def rows(p: str) -> dict:
    out = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out[r["token"]] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--arm", default="R7_A1")
    ap.add_argument("--sa", default="")
    ap.add_argument("--sb", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A = rows(os.path.join(a.a, f"rows_{a.arm}.jsonl"))
    B = rows(os.path.join(a.b, f"rows_{a.arm}.jsonl"))
    ts = sorted(t for t in A if t in B and A[t]["source"] == "refcv7" and B[t]["source"] == "refcv7")
    pa = np.asarray([A[t]["poses"] for t in ts], np.float64)
    pb = np.asarray([B[t]["poses"] for t in ts], np.float64)
    end = np.hypot(pa[:, -1, 0] - pb[:, -1, 0], pa[:, -1, 1] - pb[:, -1, 1])
    same = [A[t]["diag"]["sel_idx"] == B[t]["diag"]["sel_idx"] for t in ts]
    rep = {"control": "KP precision floor (plans): same checkpoint, same per-scene seeds",
           "arm": a.arm, "a": {"dir": a.a, "device": sorted({A[t]["device"] for t in ts}),
                               "precision": sorted({A[t]["precision"] for t in ts})},
           "b": {"dir": a.b, "device": sorted({B[t]["device"] for t in ts}),
                 "precision": sorted({B[t]["precision"] for t in ts})},
           "n": len(ts), "frac_same_selection": float(np.mean(same)),
           "frac_bit_identical": float(np.mean([np.array_equal(A[t]["poses"], B[t]["poses"])
                                                for t in ts])),
           "endpoint_4s_m": {"median": float(np.median(end)), "p90": float(np.quantile(end, 0.9)),
                             "max": float(end.max())}}
    if a.sa and a.sb and os.path.exists(a.sa) and os.path.exists(a.sb):
        ua = json.load(open(a.sa, encoding="utf-8"))["arms"].get(a.arm, {}).get("S2_EPDMS_u", {})
        ub = json.load(open(a.sb, encoding="utf-8"))["arms"].get(a.arm, {}).get("S2_EPDMS_u", {})
        if ua.get("value") is not None and ub.get("value") is not None:
            rep["S2_EPDMS_u"] = {"a": ua["value"], "b": ub["value"], "a_minus_b": ua["value"] - ub["value"]}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("n", "frac_same_selection", "frac_bit_identical",
                                          "endpoint_4s_m")} | ({"S2_EPDMS_u": rep["S2_EPDMS_u"]}
                                                               if "S2_EPDMS_u" in rep else {})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
