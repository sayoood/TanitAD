#!/usr/bin/env python3
"""STAGE D2 of the free-point navtest read (PLAN.md): W3's harness on each point's seam.  CPU only, the call `stage2_sft.py` makes, UNCHANGED:

    <driverl-eval python> score_e.py --label refe_fp_<POINT> --seam <seam.npz> [--tokens <subset.json>] --out <DATA>/score

* the label starts with "refe" (the harness refuses any other), the harness's OWN status must read PASS (a CSV existing is not evidence), and for the WHOLE navtest the
  filter is the restriction: `--tokens` is omitted (12,146 tokens = 230,812 chars = WinError 206), so the harness's guard C2 demands that every filter token scores.
* a free-RAM wait (the harness aborts below ~3 GB), up to 6 attempts, as stage2_sft.py.

    python fp_score.py score  --work W --points BASE,C0A,R0B,M25 [--parallel 1]
    python fp_score.py subset --work W --point C0A --n 1000       -> IC5's harness arm: a 1,000-token subset of C0A's seam scored as its own run
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"
DATA = "E:/Projects/TanitAD/data/refe_navtest"
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
FULL_TOKENS_MD5 = "3a8b3653fd70891e50a230a37eb76ae8"


def free_mb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def csv_path(label):
    return f"{DATA}/score/{label}/{label}.csv"


def score_one(work, label, seam, tokens_json=None, log=print):
    out_dir = f"{work}/score_logs"
    os.makedirs(out_dir, exist_ok=True)
    stp = f"{out_dir}/{label}.status.json"
    status = json.load(open(stp)) if os.path.exists(stp) else {}
    for att in range(1, 7):
        if status.get("status") == "PASS" and os.path.exists(csv_path(label)):
            break
        while free_mb() < 4000:
            time.sleep(60)
        lp = f"{out_dir}/{label}.attempt{att}.log"
        tok_arg = ["--tokens", tokens_json] if tokens_json else []
        subprocess.call([PY_D, "score_e.py", "--label", label, "--seam", seam, *tok_arg, "--out", f"{DATA}/score"], cwd=SP,
                        env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=open(lp, "w"), stderr=subprocess.STDOUT)
        lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
        status = next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})
        json.dump(status, open(stp, "w"), indent=1)
        log(f"{label} attempt {att}: {status.get('status')}")
        if status.get("status") != "PASS":
            time.sleep(120)
    ok = status.get("status") == "PASS" and os.path.exists(csv_path(label))
    return ok, status


def cmd_score(a):
    pts = a.points.split(",")
    res = {}

    def one(p):
        t0 = time.time()
        ok, st = score_one(a.work, f"{a.label_prefix}{p}", f"{a.work}/seam_{p}.npz", a.tokens_file)
        return p, ok, st, time.time() - t0
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for p, ok, st, secs in ex.map(one, pts):
            res[p] = {"ok": ok, "status": st, "seconds": round(secs)}
            print(f"ZZSCORE_{'OK' if ok else 'FAIL'} {p} {secs:.0f}s", flush=True)
    json.dump(res, open(f"{a.work}/score_status.json", "w"), indent=1)
    return 0 if all(v["ok"] for v in res.values()) else 1


def cmd_subset(a):
    z = np.load(f"{a.work}/seam_{a.point}.npz", allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    rng = np.random.default_rng(20261017)
    idx = sorted(rng.choice(len(toks), size=a.n, replace=False).tolist())
    sub = f"{a.work}/seam_{a.point}_sub{a.n}.npz"
    np.savez(sub, token=z["token"][idx], fingerprint=z["fingerprint"][idx], poses=z["poses"][idx], sampling=z["sampling"], arm=z["arm"])
    tj = f"{a.work}/tokens_{a.point}_sub{a.n}.json"
    sel = [toks[i] for i in idx]
    json.dump({"tokens": sel, "token_log": {t: exp[t]["log_name"] for t in sel}}, open(tj, "w"))
    ok, st = score_one(a.work, f"refe_fp_{a.point}_sub{a.n}", sub, tj)
    print(f"ZZSUBSET_{'OK' if ok else 'FAIL'} {a.point} {a.n}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "subset"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--points", default="BASE,C0A,R0B,M25")
    ap.add_argument("--point", default="C0A")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--tokens-file", default=None, help="DRY RUNS ONLY: score a --tokens subset (the real read scores the whole navtest filter)")
    ap.add_argument("--label-prefix", default="refe_fp_")
    a = ap.parse_args()
    return {"score": cmd_score, "subset": cmd_subset}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
