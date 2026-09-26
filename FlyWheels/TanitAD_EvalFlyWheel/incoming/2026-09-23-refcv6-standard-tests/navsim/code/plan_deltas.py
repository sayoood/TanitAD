#!/usr/bin/env python3
"""Trajectory-level lever reads — how far each arm's PLAN moves from R6_A1's on the same scenes,
before any scorer (label-free; the scores are read separately by parse6/parse_navtest6).

    python code/plan_deltas.py --bridge raw/bridge_warmup_s1000 --out raw/controls/plan_deltas_warmup_s1000.json

Per arm vs R6_A1 (refcv6-computed rows only, identical tokens): the fraction of BIT-IDENTICAL plans,
the fraction with the same selected anchor, and the 4 s endpoint distance (median / p90 / max) —
read against the SEED FLOOR (R6_A1 vs R6_A1_s1): a lever whose plans move less than inference
noise does is not moving the plan. The max-speed arm is also split by whether the scene HAS a map
limit (where it does not, R6_A1's input is already the withheld row, so identity is trivial).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def rows(path: str) -> dict:
    out = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out[r["token"]] = r
    return out


def delta(A: dict, X: dict, toks: list) -> dict:
    ts = [t for t in toks if t in X and X[t]["source"] == "refcv6"]
    if not ts:
        return {"n": 0}
    pa = np.asarray([A[t]["poses"] for t in ts], dtype=np.float64)
    px = np.asarray([X[t]["poses"] for t in ts], dtype=np.float64)
    mx = np.abs(pa - px).reshape(len(ts), -1).max(axis=1)
    end = np.hypot(pa[:, -1, 0] - px[:, -1, 0], pa[:, -1, 1] - px[:, -1, 1])
    sel = [A[t]["diag"].get("sel_idx") == X[t]["diag"].get("sel_idx") for t in ts]
    return {"n": len(ts), "frac_bit_identical": float((mx == 0).mean()),
            "frac_same_selection": float(np.mean(sel)),
            "endpoint_4s_m": {"median": float(np.median(end)), "p90": float(np.quantile(end, 0.9)),
                              "max": float(end.max())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    arms = {os.path.basename(p)[5:-6]: rows(p) for p in glob.glob(os.path.join(a.bridge, "rows_*.jsonl"))}
    if "R6_A1" not in arms:
        sys.exit("⛔ no rows_R6_A1.jsonl")
    A = arms["R6_A1"]
    own = sorted(t for t, r in A.items() if r["source"] == "refcv6")
    out = {"_label": a.label, "bridge": os.path.abspath(a.bridge), "n_R6_A1_rows": len(own),
           "vs_R6_A1": {}}
    for k, X in sorted(arms.items()):
        if k != "R6_A1":
            out["vs_R6_A1"][k] = delta(A, X, own)
    if "R6_VMAXOFF" in arms:
        wl = [t for t in own if A[t]["vmax"]["v_max_valid"] == 1.0]
        nl = [t for t in own if A[t]["vmax"]["v_max_valid"] == 0.0]
        out["R6_VMAXOFF_split"] = {"scenes_with_map_limit": delta(A, arms["R6_VMAXOFF"], wl),
                                   "scenes_without (identical input)": delta(A, arms["R6_VMAXOFF"], nl)}
    if "R6_VMAXORACLE" in arms:
        diff = [t for t in own if t in arms["R6_VMAXORACLE"] and
                (A[t]["vmax"].get("bin_kmh"), A[t]["vmax"]["v_max_valid"]) !=
                (arms["R6_VMAXORACLE"][t]["vmax"].get("bin_kmh"), arms["R6_VMAXORACLE"][t]["vmax"]["v_max_valid"])]
        out["R6_VMAXORACLE_split"] = {"scenes_where_the_one_hot_differs": delta(A, arms["R6_VMAXORACLE"], diff)}
    sf = out["vs_R6_A1"].get("R6_A1_s1", {}).get("endpoint_4s_m", {}).get("median")
    if sf is not None:
        out["seed_floor_endpoint_median_m"] = sf
        for k, v in out["vs_R6_A1"].items():
            if k != "R6_A1_s1" and v.get("n"):
                v["endpoint_median_exceeds_seed_floor"] = v["endpoint_4s_m"]["median"] > sf
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out["vs_R6_A1"].items():
        if v.get("n"):
            print(f"{k:16s} n={v['n']:4d} identical={v['frac_bit_identical']:.3f} "
                  f"same-sel={v['frac_same_selection']:.3f} end|d| med={v['endpoint_4s_m']['median']:.3f} "
                  f"p90={v['endpoint_4s_m']['p90']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
