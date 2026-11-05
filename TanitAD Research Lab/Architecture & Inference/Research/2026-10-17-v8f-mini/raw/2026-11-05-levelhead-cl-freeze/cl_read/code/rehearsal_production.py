#!/usr/bin/env python3
"""rehearsal_production.py -- the PRODUCTION path of the reader (controls -> the ten extractions -> arrays -> read -> RESULTS.md) on REAL rollout files, before any CL rollout exists.

The ten CL tags are fed the finished BASE rollouts under CL names (CL_s := C0_s, CLC_s := C0C_s, FZCL_s := FZC0_s, RICL_s := RIC0_s, RICLC_s := RIC0C_s; the dump dirs, tick files and run headers are COPIED under the CL tag's
path names, the done markers are written in the chain's own format).  The reader then runs exactly as the chain runs it: run_controls (done markers, headers, KA_NEW, a pre-check file, the reused arrays' sha256, the card's
rules), rr.extract for all ten tags (the reader's own single-thread extractor, 6 workers), read_all, render_md.  Known answer: the arm IS its base, so (i) every CL array equals the base array it was copied from
(a_i / b_i, all 19 rows), (ii) the label is P-FAILS-M1-NOT-MOVED, (iii) the rehearsal's rows equal ka-null's.  Writes ONLY under F:/TanitAD-scratch/motion_c/cl_read/rehearsal/ and raw/REHEARSAL.json.
usage: python -I code/rehearsal_production.py [--workers 6]    (about 4 minutes with 6 workers; the extractor is ~95 s per tag)"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
R = "F:/TanitAD-scratch/motion_c/cl_read/rehearsal"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SRC_DIRS = {"CL": ("C0", "C:/Users/Admin/rr_scratch/out"), "CLC": ("C0C", "C:/Users/Admin/rr_scratch/out"), "FZCL": ("FZC0", "C:/Users/Admin/rr_scratch/out"),
            "RICL": ("RIC0", "F:/TanitAD-scratch/motion_c/ri/out"), "RICLC": ("RIC0C", "F:/TanitAD-scratch/motion_c/ri/out")}


def load_reader():
    spec = importlib.util.spec_from_file_location("cl_read_rehearsal", f"{PK}/code/cl_read.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    C = load_reader()
    out, raw = f"{R}/out", f"{R}/raw"
    shutil.rmtree(R, ignore_errors=True)
    os.makedirs(out)
    os.makedirs(raw)
    t0 = time.time()
    for s in C.SEEDS:
        for role, (base, d) in SRC_DIRS.items():
            tag, old = f"{role}_s{s}", f"{base}_s{s}"
            os.makedirs(f"{out}/{tag}/ticks")
            shutil.copyfile(f"{d}/{old}/ticks/ticks_{old}.npz", f"{out}/{tag}/ticks/ticks_{tag}.npz")
            shutil.copytree(f"{d}/{old}/dump_{old}", f"{out}/{tag}/dump_{tag}")
            shutil.copyfile(f"{d}/{old}/t1v8f_{old}.json", f"{out}/{tag}/t1v8f_{tag}.json")
            json.dump({"tag": tag, "flags": C.expected_flags(role), "ckpt": f"E:/rehearsal/M2b_s{s}_recft_CL/ckpt_last.pt", "ckpt_sha256": f"{s}" * 64, "tool": "rehearsal", "tool_sha256": C.TOOL_SHA, "threads": 0, "wall_s": 0},
                      open(f"{out}/done_{tag}.txt", "w", encoding="utf-8"))
    shutil.copyfile(f"{PK}/raw/KA_NEW.json", f"{raw}/KA_NEW.json")
    json.dump({"PASS": True, "w": "rehearsal", "seeds": {f"s{s}": {"ckpt_sha256": f"{s}" * 64, "e_level_abs_sum": None} for s in C.SEEDS}}, open(f"{raw}/CL_PRECHECK.json", "w", encoding="utf-8"))
    print(f"REHEARSAL staged ten pseudo-CL rollouts in {time.time() - t0:.0f} s", flush=True)
    r = subprocess.run([PY, "-I", f"{PK}/code/cl_read.py", "--mode", "production", "--out-root", out, "--work-dir", f"{R}/work", "--raw", raw, "--workers", str(a.workers), "--results-md", f"{R}/RESULTS.md"],
                       capture_output=True, text=True)
    open(f"{PK}/logs/rehearsal_read.out", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    print(f"REHEARSAL reader rc {r.returncode} after {time.time() - t0:.0f} s", flush=True)
    res = {"card": "production-path rehearsal of the CL reader (pseudo-CL rollouts = the base rollouts)", "reader_rc": r.returncode, "checks": {}}
    if r.returncode == 0:
        z = json.load(open(f"{raw}/CL_READ.json", encoding="utf-8"))
        res["reader_sha256"] = z["reader_sha256"]
        C.init()
        chk = res["checks"]
        for s in C.SEEDS:
            base, _ = C.reused_arrays(s)
            for role, (bn, _d) in SRC_DIRS.items():
                new = C.rr.read_arrays(f"{raw}/arrays/{role}_s{s}.npz")
                ref = base[bn]
                same = all(np.array_equal(np.asarray(new[k]), np.asarray(ref[k]), equal_nan=True) for k in ref if k != "labels") and set(new) == set(ref)
                chk[f"{role}_s{s}_arrays_equal_{bn}_s{s}"] = bool(same)
        ka = json.load(open(f"{PK}/raw/KA_READER_ka-null.json", encoding="utf-8"))
        for k in z["seeds"]:
            for c in ("RICL_minus_RIC0", "CL_minus_C0", "RICLC_minus_P10", "CLC_minus_P10", "FZCL_minus_FZC0"):
                chk[f"{k}_{c}_rows_equal_ka_null"] = z["seeds"][k][c] == ka["seeds"][k][c]
        chk["label_is_M1_not_moved"] = z["outcome"]["label"] == "P-FAILS-M1-NOT-MOVED"
        chk["all_controls_pass"] = all(v.get("PASS") for v in z["controls"].values())
        chk["results_md_written"] = os.path.isfile(f"{R}/RESULTS.md") and "**Outcome: P-FAILS-M1-NOT-MOVED**" in open(f"{R}/RESULTS.md", encoding="utf-8").read()
    res["PASS"] = bool(r.returncode == 0 and res["checks"] and all(res["checks"].values()))
    res["clock"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(res, open(f"{PK}/raw/REHEARSAL.json", "w", encoding="utf-8", newline="\n"), indent=1)
    for k, v in res["checks"].items():
        if not v:
            print("  RED", k)
    print("REHEARSAL PASS" if res["PASS"] else "REHEARSAL FAIL", f"({sum(res['checks'].values())}/{len(res['checks'])} checks)")
    return 0 if res["PASS"] else 2


if __name__ == "__main__":
    sys.exit(main())
