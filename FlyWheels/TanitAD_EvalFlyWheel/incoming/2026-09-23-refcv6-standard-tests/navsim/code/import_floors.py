#!/usr/bin/env python3
"""Stage the MODEL-FREE floors (CV, STOP, ECHO) under this package's naming, with provenance.

The floors read nothing of refcv6, so one scoring per split serves every checkpoint — PROVIDED the
harness is the same. That proviso is measured, not assumed:
* warmup: CV and STOP are THIS package's re-scores, bit-identical to E2's (``HARNESS_REPRO.json``);
  ECHO is E2's seam re-scored by this package (``raw/scores_warmup_s1000``);
* navhard: W7's banked suite run (``taniteval/results/bench/navsim_v2/navhard_two_stage/
  20260921T122714Z-…-859e25``, same devkit copy / patches / cache) — and the navhard CV re-scored
  by THIS package's driver is compared with W7's CV cell by cell (``--check-cv``), which is the
  evidence the reuse is admissible.

    python code/import_floors.py --split warmup_two_stage
    python code/import_floors.py --split navhard_two_stage --check-cv raw/harness_repro_navhard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
W7RUN = ("D:/Projects/TanitAD/taniteval/results/bench/navsim_v2/navhard_two_stage/"
         "20260921T122714Z-navsim_v2-refcv4b_b1_v72_40k-859e25")
W7_NAME = {"CV_official": "CV", "STOP_zero": "STOP", "ECHO_ha0_ext": "ECHO"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def put(src_csv, src_frame, dst_dir, tag, prov):
    os.makedirs(os.path.join(dst_dir, f"score_{tag}_wrapper"), exist_ok=True)
    d_csv = os.path.join(dst_dir, f"score_{tag}.csv")
    d_fr = os.path.join(dst_dir, f"score_{tag}_wrapper", f"{tag}_final_scores_frame.csv")
    shutil.copyfile(src_csv, d_csv)
    if src_frame and os.path.exists(src_frame):
        shutil.copyfile(src_frame, d_fr)
    prov[tag] = {"csv_src": src_csv, "csv_sha256": sha(d_csv),
                 "frame_src": src_frame if src_frame and os.path.exists(src_frame) else None}


def cell_compare(a_csv, b_csv) -> dict:
    """Every numeric cell of every row (incl. the three ``extended_pdm_score_*`` summary rows).
    ⚠️ A cell that is NaN on ONE side only is a MISMATCH (``inf``) — ``np.nanmax`` would silently
    skip it, which is the check-shares-the-defect trap (CLAUDE.md)."""
    a = pd.read_csv(a_csv).set_index("token").sort_index()
    b = pd.read_csv(b_csv).set_index("token").sort_index()
    # the unnamed first column is the writer's ROW ORDER, not a score — never compared
    a = a[[c for c in a.columns if not str(c).startswith("Unnamed")]]
    b = b[[c for c in b.columns if not str(c).startswith("Unnamed")]]
    out = {"same_tokens": bool(a.index.equals(b.index)), "max_abs": 0.0, "cols": 0,
           "cols_missing_in_a": [c for c in b.columns if c not in a.columns]}
    if not out["same_tokens"]:
        out["max_abs"] = float("inf")
        return out
    for c in b.columns:
        if c in a.columns and a[c].dtype.kind in "fiub" and b[c].dtype.kind in "fiub":
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            one_nan = np.isnan(x) ^ np.isnan(y)
            d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y))
            m = float("inf") if one_nan.any() else (float(d.max()) if d.size else 0.0)
            out["max_abs"] = max(out["max_abs"], m)
            out["cols"] += 1
    if "score" in a.columns and "extended_pdm_score_combined" in a.index:
        out["official_two_stage_EPDMS_a"] = float(a.loc["extended_pdm_score_combined", "score"])
        out["official_two_stage_EPDMS_b"] = float(b.loc["extended_pdm_score_combined", "score"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--check-cv", default="", help="dir of this package's navhard CV re-score")
    a = ap.parse_args(argv)
    dst = os.path.join(PKG, "raw", "floors", a.split)
    os.makedirs(dst, exist_ok=True)
    prov = {}
    if a.split == "warmup_two_stage":
        hr = os.path.join(PKG, "raw", "harness_repro")
        for arm in ("CV_official", "STOP_zero"):
            put(os.path.join(hr, f"score_{arm}.csv"),
                os.path.join(hr, f"score_{arm}_wrapper", f"{arm}_final_scores_frame.csv"),
                dst, arm, prov)
        es = os.path.join(PKG, "raw", "scores_warmup_s1000")
        if os.path.exists(os.path.join(es, "score_ECHO_ha0_ext.csv")):
            put(os.path.join(es, "score_ECHO_ha0_ext.csv"),
                os.path.join(es, "score_ECHO_ha0_ext_wrapper", "ECHO_ha0_ext_final_scores_frame.csv"),
                dst, "ECHO_ha0_ext", prov)
    else:
        for arm, w in W7_NAME.items():
            tag = f"{arm}__{a.split}"
            put(os.path.join(W7RUN, "raw", w, f"{w}.devkit.csv"),
                os.path.join(W7RUN, "raw", w, f"{w}_final_scores_frame.csv"), dst, tag, prov)
        if a.check_cv:
            mine = os.path.join(a.check_cv, f"score_CV_official__{a.split}.csv")
            if os.path.exists(mine):
                prov["_KH_nav_check"] = cell_compare(mine, os.path.join(
                    dst, f"score_CV_official__{a.split}.csv"))
                k = prov["_KH_nav_check"]
                # SPEC §7 KH-nav: every cell <= 1e-9 AND the official two-stage EPDMS to 1e-12
                k["official_abs_diff"] = (abs(k["official_two_stage_EPDMS_a"]
                                              - k["official_two_stage_EPDMS_b"])
                                          if "official_two_stage_EPDMS_a" in k else None)
                k["verdict"] = ("REPRODUCED" if k["same_tokens"] and k["max_abs"] <= 1e-9
                                and k["official_abs_diff"] is not None
                                and k["official_abs_diff"] <= 1e-12 else "NOT_REPRODUCED")
            else:
                prov["_KH_nav_check"] = {"verdict": "PENDING", "reason": f"{mine} absent"}
    with open(os.path.join(dst, "FLOORS_PROVENANCE.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=1)
    print(json.dumps(prov, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
