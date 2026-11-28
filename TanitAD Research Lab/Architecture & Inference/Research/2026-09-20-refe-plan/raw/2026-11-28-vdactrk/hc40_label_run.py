#!/usr/bin/env python3
"""B4 skeleton (NOT RUN): harness-exact per-proposal TRACKED labels for C1's proposals on the 40 HC40 navtrain logs (PREREG_VDAC_TRK.md 4, 5, 9 B4).

Given B2's HC40 dump (C1's proposals, `props8` = the NAVSIM-grid 8-pose plans, the very poses a pick would hand the harness) it splits the dump per log into the
`<log>.dump.npz` form `sh_label.py` reads (keys `token`, `props8`), runs `sh_label.py` (navsim venv, CPU) once per log, and writes `<work>/lab/<log>.lab.npz`
(token, tr_nc/tr_dac/tr_ddc/tr_ttc/tr_ep/tr_c/tr_pdms [n,64]) = the input of `vdt_read.py calibrate --lab-glob`.

    python hc40_label_run.py --hc40-dump <vdt_hc40_dump.npz> --hc40-list <hc40_logs.txt> --work <dir> --sh-code <dir holding sh_label.py + sh_lib.py> [--workers 6] [--dry] [--det-check]

CONTRACT with B2 (`vdt_dump.py` on HC40): npz keys `token` [N] str, `props8` [N,64,8,3] float32 (= sh_dump.py's props8: `to_navsim(planner.executed(traj, j))`, the seam's
--dump-proposals conversion) and `log` [N] str (the nuPlan log of each token; else pass --token-log <json token->log>).  Proposal axis j = the dump's `s_plan` axis.
Nothing here touches a field, a head or navtest; the labels are HD-map-derived by NAVSIM's harness and are allowed on the label side (card 2).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PY_N = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"          # sh_lib.PY_N: the navsim venv (CPU labels)
NAV = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
PK_TIP = "F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
MAPS = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"        # navtrain DBs: splits/<log>.db, then splits/{trainval,test,mini}/
LAB_KEYS = ("tr_nc", "tr_dac", "tr_ddc", "tr_ttc", "tr_ep", "tr_c", "tr_pdms")


def nenv() -> dict:
    """sh_env.nenv(): the navsim-venv environment of the programme labeller (trk_chain.sh's ctrl invocation; REFE_SIM_HZ unset = the 20 Hz default, p6_pod.sh)."""
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": f"{NAV};{PK_TIP}/refe;{PK_TIP}/eval/instruments", "TANITAD_NAVSIM_ROOT": "F:/Archive/devbox-C/navsim",
              "NUPLAN_MAPS_ROOT": MAPS, "NUPLAN_DATA_ROOT": "E:/Projects/TanitAD/data/nuplan", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return e


def db_path(log: str):
    for sub in ("", "trainval", "test", "mini"):
        p = os.path.join(DB_ROOT, sub, f"{log}.db")
        if os.path.exists(p):
            return p.replace("\\", "/")
    return None


def split_by_log(dump_path: str, token_log: str | None) -> dict:
    z = np.load(dump_path, allow_pickle=False)
    tok, p8 = [str(t) for t in z["token"]], z["props8"]
    if p8.shape[1:] != (64, 8, 3):
        raise SystemExit(f"props8 shape {p8.shape} != [N,64,8,3]")
    if "log" in z.files:
        logs = [str(x) for x in z["log"]]
    elif token_log:
        m = json.load(open(token_log, encoding="utf-8"))
        logs = [m[t] for t in tok]
    else:
        raise SystemExit("the dump has no `log` key and --token-log was not given")
    by: dict = {}
    for i, (t, lg) in enumerate(zip(tok, logs)):
        by.setdefault(lg, []).append(i)
    return {lg: (np.array([tok[i] for i in ix]), p8[ix]) for lg, ix in by.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc40-dump", required=True)
    ap.add_argument("--hc40-list", required=True, help="the 40 log names (the list written by the build, sha256 recorded beside the card)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--sh-code", required=True, help="directory holding sh_label.py and sh_lib.py (PK raw/2026-11-28-safehead-data-x150/code)")
    ap.add_argument("--token-log")
    ap.add_argument("--token-log-key", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true", help="split + plan only; run no labeler")
    ap.add_argument("--det-check", action="store_true", help="label the first log twice and require byte-identical outputs (CTRL-DET-LAB)")
    a = ap.parse_args()
    want = [x.strip() for x in open(a.hc40_list, encoding="utf-8") if x.strip()]
    by = split_by_log(a.hc40_dump, a.token_log)
    extra, absent = sorted(set(by) - set(want)), sorted(set(want) - set(by))
    if extra or len(want) != 40:
        raise SystemExit(f"REFUSED: dump logs not in the HC40 list {extra[:3]}, or list has {len(want)} != 40 logs")      # the heads never see HC40's labels, and only HC40 is labelled
    W = Path(a.work)
    (W / "dump").mkdir(parents=True, exist_ok=True); (W / "lab").mkdir(exist_ok=True); (W / "logs").mkdir(exist_ok=True)
    plan = []
    for lg in want:
        if lg not in by:
            continue
        db = db_path(lg)
        plan.append({"log": lg, "n": len(by[lg][0]), "db": db})
        if db is None:
            print(f"NO DB for {lg}: cannot label", flush=True)
    json.dump({"absent_from_dump": absent, "plan": plan}, open(W / "label_plan.json", "w"), indent=1)
    print(f"HC40: {len(plan)} logs, {sum(p['n'] for p in plan)} tokens; absent from the dump: {len(absent)}", flush=True)
    if a.dry:
        return 0
    for p in plan:
        lg, db = p["log"], p["db"]
        if db is None:
            continue
        dump_p, lab_p = W / "dump" / f"{lg}.dump.npz", W / "lab" / f"{lg}.lab.npz"
        if not dump_p.exists():
            np.savez(dump_p, token=by[lg][0], props8=by[lg][1])
        if lab_p.exists():
            continue                                                                   # resumable
        t0 = time.time()
        with open(W / "logs" / f"label_{lg}.log", "w", encoding="utf-8") as fo:
            rc = subprocess.run([PY_N, str(Path(a.sh_code) / "sh_label.py"), "--dump", str(dump_p), "--db", db, "--out", str(lab_p), "--workers", str(a.workers)],
                                env=nenv(), stdout=fo, stderr=subprocess.STDOUT).returncode
        print(f"LABEL {lg} rc {rc} {time.time() - t0:.0f} s", flush=True)
    # report: labelled/asked per log (failures are in <lab>.report.json, never silently dropped), finite and in-range
    rep = {}
    for p in plan:
        lp = W / "lab" / f"{p['log']}.lab.npz"
        if not lp.exists():
            rep[p["log"]] = {"asked": p["n"], "labelled": 0}
            continue
        z = np.load(lp)
        ok = all(np.isfinite(z[k]).all() and (z[k] >= 0).all() and (z[k] <= 1).all() for k in LAB_KEYS)
        rep[p["log"]] = {"asked": p["n"], "labelled": int(len(z["token"])), "finite_in_unit_range": bool(ok)}
    json.dump(rep, open(W / "label_report.json", "w"), indent=1)
    if a.det_check and plan and plan[0]["db"]:
        lg = plan[0]["log"]
        l2 = W / "lab" / f"{lg}.det2.lab.npz"
        subprocess.run([PY_N, str(Path(a.sh_code) / "sh_label.py"), "--dump", str(W / "dump" / f"{lg}.dump.npz"), "--db", plan[0]["db"], "--out", str(l2), "--workers", "1"], env=nenv(), check=True)
        same = (W / "lab" / f"{lg}.lab.npz").read_bytes() == l2.read_bytes()
        print("CTRL-DET-LAB", "PASS" if same else "FAIL", flush=True)
        if not same:
            return 1
    n_lab = sum(v["labelled"] for v in rep.values())
    print(f"ZZHC40_LABELS {n_lab}/{sum(p['n'] for p in plan)} tokens labelled", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
