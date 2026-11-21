#!/usr/bin/env python3
"""WHERE refcv6 loses — the SPEC §5 decomposition ladder (multipliers, stage, speed band, log, and
the per-COMMAND split the bars' FAIL branch names), for every split this package scores.

⭐ NOTHING IS RE-IMPLEMENTED FOR THE TWO-STAGE SPLITS. W7's ``decompose.py`` (zero attribution with
the UNIQUE-zero control, the single-term counterfactual ceiling asserted against the devkit's own
``score`` column, per-stage sub-metrics, speed bands by v0, per-log) is IMPORTED by path and run
on a scratch run layout (``<tmp>/scores/<arm>.csv`` — copies of this package's CSVs). This file
adds only what W7 does not carry: the split by NavSim ``driving_command`` (one-hot order left,
straight, right, unknown — E2 ``tests/test_command_order.py``), arm vs each floor, with the
arm's multiplier-zero rates per command. For navtest (v1.1, single stage, plain column names)
the same per-command + speed-band table is computed here on W3's banked floor CSVs.

    python code/decompose6.py --split warmup_two_stage --scores raw/scores_warmup_s1000 \
        --floors raw/floors/warmup_two_stage --inputs <export.json> --out raw/decomp_warmup.json
    python code/decompose6.py --split navtest --arm-csv <r6 navtest csv> --inputs <navtest export .json.gz> \
        --out raw/decomp_navtest.json

⚠️ Stage-1 rows that are the declared CV STAND-IN (warmup's camera arms) are about CV, not the
model: they are reported labelled, and the per-command table reads STAGE 2 only for such arms.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
W7_DECOMPOSE = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                "2026-09-20-navhard-refcv4b/code/decompose.py")
W3_RAW = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw"
W3_FLOORS = {"STOP": f"{W3_RAW}/STOP_navtest/STOP_navtest.csv",
             "CV": f"{W3_RAW}/CV_navtest/CV_navtest.csv",
             "HUMAN": f"{W3_RAW}/HUMAN_navtest/HUMAN_navtest.csv"}
CMD = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
SUMMARY_ROWS = ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                "extended_pdm_score_combined")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_export(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["tokens"]


def command_of(rec: dict) -> str:
    dc = rec["ego_statuses"][-1]["driving_command"]
    oh = [float(x) for x in dc]
    if sorted(oh) != [0.0] * (len(oh) - 1) + [1.0]:
        return "NOT_ONE_HOT"
    return CMD[int(np.argmax(oh))]


def band(v: float) -> str:
    for lo, hi in ((0, 2), (2, 5), (5, 8), (8, 12), (12, 99)):
        if lo <= v < hi:
            return f"{lo}-{hi} m/s" if hi < 99 else ">=12 m/s"
    return "unknown"


def wtl(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> dict:
    return {"win": int((a - b > eps).sum()), "tie": int((np.abs(a - b) <= eps).sum()),
            "loss": int((b - a > eps).sum()), "n": int(len(a))}


def per_command_two_stage(arm_df, others: dict, doc: dict, stages) -> dict:
    out = {}
    for stage in stages:
        suf = "one" if stage == 1 else "two"
        idx = [t for t in arm_df.index if (arm_df.loc[t, f"ego_progress_stage_one"]
                                           == arm_df.loc[t, f"ego_progress_stage_one"]) == (stage == 1)]
        by = {}
        for t in idx:
            by.setdefault(command_of(doc[t]), []).append(t)
        blk = {}
        for c, ts in sorted(by.items()):
            a = arm_df.loc[ts, "score"].to_numpy(float)
            row = {"n": len(ts), "arm": round(float(a.mean()), 6),
                   "arm_zero_rate": {k: round(float((arm_df.loc[ts, f"{col}_stage_{suf}"] <= 1e-12).mean()), 4)
                                     for k, col in (("NC", "no_at_fault_collisions"),
                                                    ("DAC", "drivable_area_compliance"),
                                                    ("DDC", "driving_direction_compliance"),
                                                    ("TLC", "traffic_light_compliance"))},
                   "vs": {}}
            for name, odf in others.items():
                if not set(ts) <= set(odf.index):
                    row["vs"][name] = "UNAVAILABLE (tokens differ)"
                    continue
                b = odf.loc[ts, "score"].to_numpy(float)
                row["vs"][name] = {"other": round(float(b.mean()), 6),
                                   "delta": round(float(a.mean() - b.mean()), 6), "wtl": wtl(a, b)}
            blk[c] = row
        out[f"stage_{stage}"] = blk
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("warmup_two_stage", "navhard_two_stage", "navtest"))
    ap.add_argument("--scores", default="", help="dir with score_<arm><suffix>.csv (two-stage)")
    ap.add_argument("--floors", default="", help="dir with the floors' score_<floor><suffix>.csv")
    ap.add_argument("--arm", default="R6_A1")
    ap.add_argument("--vs", default="STOP_zero,CV_official,ECHO_ha0_ext")
    ap.add_argument("--csv-suffix", default="")
    ap.add_argument("--bridge", default="", help="dir with seam_<arm>.npz (stand-in detection)")
    ap.add_argument("--arm-csv", default="", help="navtest: the arm's CSV")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    doc = load_export(a.inputs)
    rep = {"_what": "SPEC §5 decomposition ladder", "_label": a.label, "split": a.split, "arm": a.arm}
    if a.split == "navtest":
        A = pd.read_csv(a.arm_csv, index_col=0)
        A = A[A.token.isin(doc)].set_index("token")
        fl = {k: pd.read_csv(p, index_col=0).set_index("token") for k, p in W3_FLOORS.items()}
        toks = list(A.index)
        v0 = {t: math.hypot(*doc[t]["ego_statuses"][-1]["ego_velocity"][:2]) for t in toks}
        cells = {"by_command": {}, "by_speed_band": {}}
        for key, fn in (("by_command", lambda t: command_of(doc[t])), ("by_speed_band", lambda t: band(v0[t]))):
            groups = {}
            for t in toks:
                groups.setdefault(fn(t), []).append(t)
            for g, ts in sorted(groups.items()):
                a_ = A.loc[ts, "score"].to_numpy(float)
                row = {"n": len(ts), "arm_PDMS_x100": round(100 * float(a_.mean()), 4),
                       "arm_NC0_pct": round(100 * float((A.loc[ts, "no_at_fault_collisions"] <= 1e-12).mean()), 2),
                       "arm_DAC0_pct": round(100 * float((A.loc[ts, "drivable_area_compliance"] <= 1e-12).mean()), 2)}
                for k, F in fl.items():
                    b_ = F.loc[ts, "score"].to_numpy(float)
                    row[f"{k}_PDMS_x100"] = round(100 * float(b_.mean()), 4)
                    if k != "HUMAN":
                        row[f"minus_{k}_x100"] = round(100 * float(a_.mean() - b_.mean()), 4)
                        row[f"wtl_vs_{k}"] = wtl(a_, b_)
                cells[key][g] = row
        # ---- the ladder W7 runs on two-stage, here for PDMS_v1: UNIQUE-zero attribution and the
        # single-term counterfactual ceiling, asserted against the devkit's own `score` FIRST.
        # PDMS_v1 = NC * DAC * (5 EP + 5 TTC + 2 C) / 12 (DDC is reported with weight 0).
        cols = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
                "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "C": "comfort"}
        V = {k: A.loc[toks, c].to_numpy(dtype=float) for k, c in cols.items()}
        actual = A.loc[toks, "score"].to_numpy(dtype=float)

        def pdms(v):
            return v["NC"] * v["DAC"] * (5 * v["EP"] + 5 * v["TTC"] + 2 * v["C"]) / 12.0
        base = pdms(V)
        chk = {"max_abs_diff_vs_devkit_score": float(np.nanmax(np.abs(base - actual))), "tol": 1e-9}
        chk["pass"] = bool(chk["max_abs_diff_vs_devkit_score"] <= chk["tol"])
        zero = actual <= 1e-12
        zm = {k: V[k] <= 1e-12 for k in ("NC", "DAC")}
        nz = zm["NC"].astype(int) + zm["DAC"].astype(int)
        lad = {"formula": "NC * DAC * (5 EP + 5 TTC + 2 C) / 12", "formula_selfcheck": chk,
               "n_zero_score": int(zero.sum()), "n": len(toks),
               "zero_attribution": {k: {"n_zeroed_with_this_term_0": int((zero & zm[k]).sum()),
                                        "n_zeroed_where_ONLY_this_term_is_0": int((zero & zm[k] & (nz == 1)).sum())}
                                    for k in ("NC", "DAC")}}
        if chk["pass"]:
            per = {}
            for k in cols:
                cf = pdms({kk: (np.ones_like(V[kk]) if kk == k else V[kk]) for kk in V})
                per[k] = {"PDMS_x100_if_this_term_were_perfect": round(100 * float(cf.mean()), 4),
                          "gain_x100": round(100 * float(cf.mean() - actual.mean()), 4),
                          "term_mean_now": round(float(np.nanmean(V[k])), 6)}
            lad["single_term_ceiling"] = dict(sorted(per.items(), key=lambda kv: -kv[1]["gain_x100"]))
            lad["_read"] = ("a CEILING under a single-term repair with everything else frozen; the "
                            "terms trade (braking buys NC/TTC, loses EP), so they do not add up")
        else:
            lad["single_term_ceiling"] = {"status": "REFUSED", "reason": "the formula does not "
                                          "reproduce the devkit score column"}
        rep.update({"arm_csv": os.path.abspath(a.arm_csv), "floors": W3_FLOORS, "n_tokens": len(toks),
                    "ladder": lad, **cells})
    else:
        w7 = _load("w7_decompose6", W7_DECOMPOSE)
        tmp = tempfile.mkdtemp(prefix="decomp6_")
        os.makedirs(os.path.join(tmp, "scores"))
        names = [a.arm] + [x for x in a.vs.split(",") if x]
        srcs = {}
        for n in names:
            d = a.scores if n.startswith("R6_") else (a.floors or a.scores)
            p = os.path.join(d, f"score_{n}{a.csv_suffix}.csv")
            if os.path.exists(p):
                shutil.copy(p, os.path.join(tmp, "scores", f"{n}.csv"))
                srcs[n] = os.path.abspath(p)
        w7_out = os.path.join(tmp, "w7.json")
        w7.main(["--run", tmp, "--arm", a.arm, "--vs", ",".join(n for n in names[1:] if n in srcs),
                 "--inputs", a.inputs, "--out", w7_out])
        rep["w7_ladder"] = json.load(open(w7_out, encoding="utf-8"))
        rep["w7_ladder"]["run"] = "scratch layout of: " + json.dumps(srcs)
        rep["sources"] = srcs
        rep["w7_decompose_py"] = W7_DECOMPOSE
        def rd(n):
            d = pd.read_csv(os.path.join(tmp, "scores", f"{n}.csv"), index_col=0)
            return d[~d.token.isin(SUMMARY_ROWS)].set_index("token")
        A = rd(a.arm)
        stand_in = False
        if a.bridge and os.path.exists(os.path.join(a.bridge, f"seam_{a.arm}.npz")):
            z = np.load(os.path.join(a.bridge, f"seam_{a.arm}.npz"), allow_pickle=False)
            stand_in = "cv_standin" in {str(x) for x in z["source"]}
        rep["stage1_is_cv_standin"] = stand_in
        others = {n: rd(n) for n in names[1:] if n in srcs}
        rep["per_command"] = per_command_two_stage(A, others, doc, (2,) if stand_in else (1, 2))
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[decompose6] {a.split} {a.arm} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
