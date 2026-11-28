#!/usr/bin/env python
"""D4 (map-signal audit): how much of the 10 cm loss's logit gradient do the BIG classes
keep, under candidate class weightings? ANALYTIC, from the audit's own numbers.

Input: the audit's ``raw/analytic_class_signal.json`` (137 eval-kit GT files, every 10th
frame -- a PROXY for the TRAIN distribution, as the audit states). For a weighted hard CE
``L = sum_i w_{y_i} CE_i / W`` the logit gradient of a cell labelled c is ``w_c (q - e_c)
/ W``, so at any FIXED model state q (a function of the label only) the share of the
attributed gradient that class c carries is

    share_c(w) = w_c A_c / sum_k w_k A_k,

with ``A_c`` the UNWEIGHTED attribution at that state. The audit publishes ``A_c`` for
S0 (uniform init: A = the label-mass share) and S2 ("drivable-majority": right about the
big classes, never predicting a thin one -- the refcv6@35k picture). This script checks
the formula against the audit's own MF-present / MF-global shares (they must reproduce to
the rounding of the published weights), then evaluates candidate weightings.

Nothing here changes the pre-registered loss: SPEC_REFCV7 §11 (A6) routes any change to a
SPEC amendment BEFORE any NEW-2 number exists. This is the proposal's evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

NAMES = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
BIG = (0, 1, 7)
THIN = (2, 3, 4, 5, 6)


def shares(w, A) -> np.ndarray:
    x = np.asarray(w, np.float64) * np.asarray(A, np.float64)
    return x / x.sum()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    d = json.loads(a.audit_json.read_text(encoding="utf-8"))
    h = d["hard_10cm_new2"]
    A0 = np.asarray(h["none"]["S0_uniform_share"])
    A2 = np.asarray(h["none"]["S2_drivable_majority_share"])
    mf = np.asarray(h["weights"]["MF_present_clip25"])
    mfg = np.asarray(h["weights"]["MF_global_clip25"])
    # --- control: the formula reproduces the audit's published weighted shares ---
    ctrl = {}
    for key, w in (("MF_present_clip25", mf), ("MF_global_clip25", mfg)):
        for st, A in (("S0_uniform_share", A0), ("S2_drivable_majority_share", A2)):
            got = shares(w, A)
            want = np.asarray(h[key][st])
            ctrl[f"{key}|{st}"] = float(np.abs(got - want).max())
    cands = {
        "none (unweighted)": np.ones(8),
        "MF-present, clip 25 (PRE-REGISTERED)": mf,
        "MF-global, clip 25": mfg,
        "sqrt(MF-present)": np.sqrt(mf),
        "MF-present, floor 0.25": np.maximum(mf, 0.25),
        "MF-present, floor 0.5": np.maximum(mf, 0.5),
        "MF-present, floor 1.0": np.maximum(mf, 1.0),
        "0.5 * MF-present + 0.5": 0.5 * mf + 0.5,
    }
    rows = {}
    for name, w in cands.items():
        s0, s2 = shares(w, A0), shares(w, A2)
        rows[name] = {
            "weights": [round(float(v), 4) for v in w],
            "S0_share": {n: round(float(v), 4) for n, v in zip(NAMES, s0)},
            "S2_share": {n: round(float(v), 4) for n, v in zip(NAMES, s2)},
            "S0_thin_total": round(float(s0[list(THIN)].sum()), 4),
            "S2_big_each_min": round(float(s2[list(BIG)].min()), 4),
            "S2_big_total": round(float(s2[list(BIG)].sum()), 4),
            "S2_thin_min": round(float(s2[list(THIN)].min()), 4),
            "max_over_min_weight": round(float(w.max() / w.min()), 2),
        }
    rec = {"source": str(a.audit_json.name),
           "evidence": "ANALYTIC on the audit's MEASURED proxy counts (137 eval-kit GT "
                       "files); the launch weights come from TRAIN",
           "formula": "share_c(w) = w_c A_c / sum_k w_k A_k at a fixed state q",
           "control_max_abs_err_vs_audit": ctrl,
           "candidates": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    hdr = f"{'candidate':40s} {'S0 thin':>8s} {'S2 big(each min)':>17s} {'S2 big tot':>10s} {'S2 thin min':>11s} {'w max/min':>9s}"
    print(hdr)
    for n, r in rows.items():
        print(f"{n:40s} {r['S0_thin_total']:8.3f} {r['S2_big_each_min']:17.3f} "
              f"{r['S2_big_total']:10.3f} {r['S2_thin_min']:11.3f} {r['max_over_min_weight']:9.1f}")
    print("control (max |formula - audit|):", {k: f"{v:.2e}" for k, v in ctrl.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
