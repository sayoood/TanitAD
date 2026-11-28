#!/usr/bin/env python3
"""extract_arrays.py -- the per-window arrays behind Read A's 19 rows of ONE arm against P10, at ONE seed (zero training, CPU only).

It reuses the instruments of the SOUP / LAUNCH-INTENT readers unchanged (DAG-ORC `Rig`: Read A's comps rows, the t1kf TAC / DK rows, the t1hd S-POS rows), and writes
the PER-WINDOW arrays those readers reduce to one row each.  It computes NO class split and NO share: that is gapdecomp.py, after PREREG_FROZEN.md is frozen.

    python extract_arrays.py --arm CTRL --seed 0 --out F:/TanitAD-scratch/gapdecomp/arrays/CTRL_s0.npz

arm: CTRL = DAG-OACC (the member of the SOUP manifest), REP, SOUP.  Windows: the 2,399 eval-141 windows of the SOUP read.
Row r of an arm X against P10 is  mean over the row's valid windows of ( a_r - b_r ):  a = X's per-window value, b = P10's (S-POS: a = S, b = CTRV of X's own run, the within-run contrast).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
ORC_DIR = f"{ART}/dagorc_rebase/read"
MANIFEST = "F:/TanitAD-scratch/soup_t1_out/runs_soup.json"
MANIFEST_SHA = "35833493575f28382aa509e2fac99819e8c64ca18d62c043a3afda855c2b6372"

LABELS = ("ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE",
          "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share")
COMPS_KEY = {"ADE6": "ade6_m", "speed MAE": "speed_mae_mps", "along-track MAE": "along_mae_m", "accel MAE": "accel_mae_mps2", "heading MAE": "heading_mae_deg", "yaw-rate MAE": "yaw_rate_mae_degps",
             "curvature MAE": "curvature_mae_1pm", "cross-track MAE": "cross_mae_m", "L1 lateral-accel MAE": "L1_lat_accel_mae_mps2", "L2 curvature MAE (GT>=5.2)": "L2_curvature_mae_fast_1pm"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("CTRL", "REP", "SOUP"))
    ap.add_argument("--seed", required=True, type=int, choices=(0, 1))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default="F:/TanitAD-scratch/gapdecomp/work/rig")
    a_ = ap.parse_args(argv)
    t0 = time.time()
    if sha256_file(MANIFEST) != MANIFEST_SHA:
        print("REFUSED: runs_soup.json is not the pinned manifest", file=sys.stderr)
        return 2
    sys.path.insert(0, ORC_DIR)
    import dag_orc_read as O  # noqa: E402
    ar = O.default_args(work_dir=f"{a_.work_dir}_{a_.arm}_s{a_.seed}", n_boot=2000)
    rig = O.Rig(ar)
    A, Tk, be = rig.A, rig.T, rig.backend
    R, T1KF = A.R, A.T
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    s = a_.seed
    ent = man["arms"][a_.arm][str(s)]["t1"]
    ref_ent = man["arms"]["CTRL"][str(s)]["t1"]
    d = A.dump(ent["dump_dir"])
    dref = A.dump(ref_ent["dump_dir"])
    same_windows = all(np.array_equal(d[k], dref[k]) for k in ("clip_index", "ws", "eid", "g", "v0"))
    same_floor = bool(np.array_equal(d["fl"], dref["fl"]))
    if not (same_windows and same_floor and len(d["eid"]) == 2399 and len(np.unique(d["eid"])) == 141):
        print(f"REFUSED: windows / floor differ from CTRL's (same_windows={same_windows}, same_floor={same_floor})", file=sys.stderr)
        return 2
    G = d["g"]
    cx = A.tacdk_context(dref)                                          # exactly ReadA's / Heavy.prepare's context: CTRL's dump
    PX, PY = T1KF.prep(a_.arm, d["cl"], cx), T1KF.prep("P10", d["fl"], cx)
    rows = {}
    cxr, cyr = R.rows_of(d["cl"], G), R.rows_of(d["fl"], G)
    for lab, k in COMPS_KEY.items():
        rows[lab] = (np.asarray(cxr[k], np.float64), np.asarray(cyr[k], np.float64))
    rows["M1 crawl share"] = (np.asarray(R.m1_window(d["cl"], G)[0], np.float64), np.asarray(R.m1_window(d["fl"], G)[0], np.float64))
    for lab in ("TAC-LAT", "TAC-LON"):
        rows[lab] = (np.asarray(PX["ind"][lab], np.float64), np.asarray(PY["ind"][lab], np.float64))
    for lab in ("DK-HW", "DK-TG"):
        rows[lab] = (np.asarray(PX["abs"][lab], np.float64), np.asarray(PY["abs"][lab], np.float64))
    LM = T1KF.LM
    va, vb = LM._pw_get(PX["lv"], ("viol", "lead_viol")), LM._pw_get(PY["lv"], ("viol", "lead_viol"))
    ra, rb = LM._pw_get(PX["lv"], ("at_risk", "lead_at_risk")), LM._pw_get(PY["lv"], ("at_risk", "lead_at_risk"))
    sel = (ra == 1.0)
    if not (np.array_equal(ra == 1.0, rb == 1.0) and np.isfinite(va[sel]).all() and np.isfinite(vb[sel]).all()):
        print("REFUSED: DK-VIOL at-risk sets differ or carry NaN inside the at-risk set", file=sys.stderr)
        return 2
    rows["DK-VIOL"] = (np.where(sel, va, np.nan), np.where(sel, vb, np.nan))
    # S-POS: the within-run contrast S - CTRV of this arm's own ticks (t1hd_read.s_rows: per-window mean over ticks >= 1, both sides NaN on the same windows)
    res, Rt, tt = Tk.run_one(be.ctx, be.a, ent["ticks"], None, f"{a_.arm}_s{s}", f"s{s}", 2000, be.a.banked_dir, full_rows=False, k7=False)
    aligned = bool(np.array_equal(tt["ci"], d["clip_index"]) and np.array_equal(tt["t0"], d["ws"]))
    if not aligned:
        print("REFUSED: ticks are not window-aligned with the dump", file=sys.stderr)
        return 2
    P = {nm: Tk.per_window(Rt["err"][nm]["pos_err_m"]) for nm in ("S", "CTRV")}
    for m, arc in enumerate((30, 50, 80)):
        sa, sb = np.asarray(P["S"][1][:, m], np.float64), np.asarray(P["CTRV"][1][:, m], np.float64)
        ok = np.isfinite(sa) & np.isfinite(sb)
        rows[f"S-POS@{arc}"] = (np.where(ok, sa, np.nan), np.where(ok, sb, np.nan))
    assert set(rows) == set(LABELS), set(LABELS) ^ set(rows)
    out = {"labels": np.array(LABELS), "eid": d["eid"], "v0": d["v0"], "g": d["g"], "clip_index": d["clip_index"], "ws": d["ws"]}
    for i, lab in enumerate(LABELS):
        out[f"a_{i}"], out[f"b_{i}"] = rows[lab]
    os.makedirs(os.path.dirname(os.path.abspath(a_.out)), exist_ok=True)
    np.savez_compressed(a_.out, **out)
    meta = {"arm": a_.arm, "seed": s, "dump_dir": ent["dump_dir"], "ticks": ent["ticks"], "manifest_sha256": MANIFEST_SHA, "out_sha256": sha256_file(a_.out), "same_windows_as_CTRL": same_windows,
            "same_floor_as_CTRL": same_floor, "ticks_window_aligned": aligned, "wall_s": round(time.time() - t0, 1)}
    with open(a_.out + ".meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print("EXTRACT OK", json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
