#!/usr/bin/env python3
"""Measure 5 -- POST-HOC, NON-GATING sensitivity input: the 9 eval extras under Amendment 7's executed-plan repair.

WHY. The registered E1/E3 ground truth is the E-6 table of snapshot 015 (eval/PREREG_MEASURE5.md §4), scored BEFORE
Amendment 7 (ADOPTED 2026-09-27: the executed plan takes heading[18] at the last native pose, refe/planner.py
repair_last_heading). Every ORIGINAL proposal carries the corrupted t = 4.0 s heading in that table, while a 0.75x copy
never reaches that pose (SPEC_NAVTEST Amendment 6 readout) -- so the unrepaired ground truth favours copies over
originals for a reason the deployed planner no longer has. The registered readout is reported AS REGISTERED; this
builds what a re-read against the DEPLOYED harness truth needs, labelled post hoc:
  * the 64 originals, repaired: ALREADY BANKED (like_for_like_016: proptable/sub200_ep015_repaired/table.npz, 64/64
    harness runs PASS, all gates pass);
  * the 8 x 0.75x copies of the top-8, repaired: built HERE (planner.repair_last_heading on the native copy, then the
    seam's own to_navsim) and scored by the UNCHANGED harness (score_navtest_refe.py);
  * STOP = zeros: the repair is the identity on it (asserted), so the banked stopzeros csv stands.
Controls: the UNREPAIRED rebuild of every copy equals slow_copies/build.npz nav_f075 bit for bit; the repaired copy
differs from it ONLY in the 4.0 s heading (x, y on all 8 poses and headings 0.5-3.5 s bit-identical).

    python eval/m5_repaired_extras.py build
    python eval/m5_repaired_extras.py score [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import eval_checkpoint as EC  # noqa: E402

NAME = "sub200_ep015"
PT = f"{EC.DATA}/proptable/{NAME}"
STOP_DUMP = f"{PT}/stop_candidate_dump.npz"
BUILD = f"{PT}/slow_copies/build.npz"
RANKS = f"{PT}/slow_copies/ranks.npz"
LANDED = f"{EC.DATA}/seams/refe_{NAME}.npz"
SEAM_DIR = f"{EC.DATA}/seams/proptable/{NAME}_m5rep"
LOGS = f"{PT}/m5rep_logs"
TOK = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")
OUT = os.path.join(HERE, "raw", "m5_effectiveness", "repaired_extras.json")


def label(r):
    return f"refe_{NAME}_f075rep_r{r:02d}"


def build(_a) -> int:
    import refe_navtest_seam as SEAM
    from planner import repair_last_heading
    from slow_copies import slow_copy
    SD = np.load(STOP_DUMP, allow_pickle=False)
    B = np.load(BUILD, allow_pickle=False)
    R = np.load(RANKS)
    S0 = np.load(LANDED)
    tt = [str(t) for t in SD["token"]]
    if [str(t) for t in B["token"]] != tt or [str(t) for t in R["token"]] != tt or [str(t) for t in S0["token"]] != tt:
        raise SystemExit("token order differs")
    n = len(tt)
    ar = np.arange(n)
    os.makedirs(SEAM_DIR, exist_ok=True)
    rec = {"controls": {}, "seams": {}}
    for r in range(8):
        src = SD["traj"][ar, R["top"][:, r]].astype(np.float32)                      # [n, 20, 3]
        nat = slow_copy(src, 0.75, dt=SEAM.SRC_DT)
        unrep = np.stack([SEAM.to_navsim(x).astype(np.float32) for x in nat])
        c_unrep = float(np.abs(unrep.astype(np.float64) - B["nav_f075"][ar, R["top"][:, r]].astype(np.float64)).max())
        rep_nat = repair_last_heading(nat)
        rep = np.stack([SEAM.to_navsim(x).astype(np.float32) for x in rep_nat])
        xy_same = bool(np.array_equal(rep[..., :2], unrep[..., :2]))
        h_same = bool(np.array_equal(rep[:, :7, 2], unrep[:, :7, 2]))
        changed = int((rep[:, 7, 2] != unrep[:, 7, 2]).sum())
        rec["controls"][f"r{r:02d}"] = {"unrepaired_rebuild_vs_build_nav_f075_max_abs": c_unrep, "xy_bit_identical": xy_same,
                                        "headings_0.5_3.5s_bit_identical": h_same, "tokens_4.0s_heading_changed": changed,
                                        "max_abs_4.0s_heading_change_rad": float(np.abs(rep[:, 7, 2] - unrep[:, 7, 2]).max())}
        if c_unrep != 0.0 or not xy_same or not h_same:
            print(f"ZZM5REP_FAIL control r{r:02d} {rec['controls'][f'r{r:02d}']}")
            return 1
        sp = os.path.join(SEAM_DIR, f"{label(r)}.npz")
        np.savez(sp, token=S0["token"], fingerprint=S0["fingerprint"], poses=rep, sampling=S0["sampling"],
                 arm=np.array(label(r).replace("refe_", "REFe_")))
        rec["seams"][label(r)] = sp
    z = np.zeros((20, 3), np.float32)
    rec["controls"]["stop_repair_is_identity"] = bool(np.array_equal(repair_last_heading(z), z))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rec["controls"]))
    print("ZZM5REP_BUILD_OK")
    return 0


def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


def score(a) -> int:
    os.makedirs(LOGS, exist_ok=True)
    rec = json.load(open(OUT, encoding="utf-8"))

    def one(r):
        lb = label(r)
        log = os.path.join(LOGS, f"{lb}.log")
        csvp = f"{EC.DATA}/score/{lb}/{lb}.csv"
        if os.path.exists(csvp) and os.path.exists(log):
            st = status_of(open(log, encoding="utf-8", errors="replace").read())
            if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == 200:
                return lb, st
        EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", os.path.join(SEAM_DIR, f"{lb}.npz"),
                "--tokens", TOK, "--out", f"{EC.DATA}/score"], HERE, dict(os.environ, PYTHONIOENCODING="utf-8"), log)
        return lb, status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None

    t0 = time.time()
    res = {}
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for lb, st in ex.map(one, range(8)):
            res[lb] = st
            print(f"  {lb}: {(st or {}).get('status')} {time.time() - t0:.0f} s", flush=True)
    bad = [lb for lb, st in res.items() if not (st and st.get("status") == "PASS" and st.get("csv_valid_rows") == 200)]
    rec["harness"] = {"runs": res, "failed": bad, "seconds": round(time.time() - t0, 1)}
    json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"ZZM5REP_SCORE_{'OK' if not bad else 'FAIL'} {8 - len(bad)}/8")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("build", "score"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    return build(a) if a.stage == "build" else score(a)


if __name__ == "__main__":
    sys.exit(main())
