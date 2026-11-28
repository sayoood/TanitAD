#!/usr/bin/env python3
"""TF-N1 AMENDMENT 1 run chain (PREREG_TF_N1_A1.md): the four N1 plan seams scored through the PINNED W3 HARNESS, CPU only, at most
2 processes, resumable. Modelled on tf_n1_chain.py (its guard pattern); it does NOT touch the void run's outputs: everything goes
under <run>/a1/.

REFUSES TO START unless ALL hold (each is tested in test_tf_n1_a1_chain.py):
  1. sha256(--amendment) == --amendment-sha256
  2. the K ledger holds the key "TF-N1-A1:<that sha256>:E3" with n == --k (42), the void "TF-N1:" entry (n 41) is present, and no entry has n > --k
     (a later landing shifts K: the MM corrects the card and re-freezes)
  3. every file of N1_PINS.json under --run-dir matches its sha256 (the void run's frozen instruments, unchanged)
  4. sha256(A1_PINS.json) == --a1-pins-sha256 (quoted in the amendment), every file of it matches, and the harness tree holds no unpinned file
  5. --procs <= 2

Stages (markers <run>/a1/markers/<stage>.done | .FAIL; a FAIL stops the chain):
  score_T   re-score the REGISTERED T SEAM through score_p4_a1.py -> eval/harness/score_e.py (one process)
  cidh      C-ID-H, EARLY: that CSV against the banked refe_p4_teacher_navtest.csv, |d| <= 1e-6 on NC DAC DDC EP TTC C PDMS, ALL 12,146 tokens.
            FAIL => STOP. refe_tfn1_{tc20s,tw,m} are then NEVER scored: no N1-C / N1-W / N1-M number exists, the read is void again.
  score_CWM refe_tfn1_tc20s, refe_tfn1_tw, refe_tfn1_m through the harness (C and W in parallel, then M)
  read      tf_n1_analyze_a1.py read --harness-csv N1T= N1C= N1W= N1M= ... (K = 42), controls first
Nothing here ever runs W-TRANSP, and nothing reads a W-TRANSP row: the existing fam/*.json are only handed (pinned) to the read.
Launch: Win32_Process Create, CreateFlags 0x08000200, `cmd /c "python tf_n1_a1_chain.py ... >> chain_a1.log 2>&1 < NUL"` (launch_n1_a1.ps1 prints it).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
TOKEN_LOG = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
MAPS = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
FAM_ARMS = ("N1C", "N1T", "N1W", "CERT", "human")
N_ALL = 12146
ID_TOL = 1e-6
MET = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ddc": "driving_direction_compliance",
       "ep": "ego_progress", "ttc": "time_to_collision_within_bound", "c": "comfort", "pdms": "score"}
# (stage label, arm name in the read, seam path relative to the run dir)
SEAMS = {"T": ("refe_a1_tfn1_t", "N1T", "inputs/refe_p4_teacher_navtest.npz"),
         "C": ("refe_a1_tfn1_c", "N1C", "build/out/refe_tfn1_tc20s.npz"),
         "W": ("refe_a1_tfn1_w", "N1W", "build/out/refe_tfn1_tw.npz"),
         "M": ("refe_a1_tfn1_m", "N1M", "build/out/refe_tfn1_m.npz")}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def free_virt_mb() -> int:
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)"],
                       capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def env(run: str) -> dict:
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "REFE_SIM_HZ": "10", "NUPLAN_MAPS_ROOT": MAPS, "PYTHONIOENCODING": "utf-8",
              "PYTHONPATH": ";".join([DEVKIT.replace("/", "\\"), os.path.join(run, "w", "pk", "refe").replace("/", "\\")])})
    return e


# ------------------------------------------------------------------------------------------------------------- guards
def guards(a) -> dict:
    """raise SystemExit('REFUSING: ...') on the first failed guard; return facts for the log"""
    if a.procs > 2:
        raise SystemExit("REFUSING: at most 2 processes (PREREG_TF_N1 section 12)")
    if a.procs < 1:
        raise SystemExit("REFUSING: --procs must be 1 or 2")
    run = a.run_dir
    if not os.path.isfile(a.amendment):
        raise SystemExit(f"REFUSING: amendment {a.amendment} is missing")
    got = sha256(a.amendment)
    if got != a.amendment_sha256:
        raise SystemExit(f"REFUSING: sha256({a.amendment}) = {got} != --amendment-sha256 {a.amendment_sha256}")
    if not os.path.isfile(a.ledger):
        raise SystemExit(f"REFUSING: ledger {a.ledger} is missing")
    reads = json.load(open(a.ledger, encoding="utf-8")).get("reads", [])
    key = f"TF-N1-A1:{a.amendment_sha256}:E3"
    mine = [r for r in reads if r.get("key") == key]
    if not mine:
        raise SystemExit(f"REFUSING: the K ledger has no entry {key} (freeze, then ledger, then run)")
    if mine[0].get("n") != a.k:
        raise SystemExit(f"REFUSING: ledger entry {key} has n = {mine[0].get('n')}, the amendment says K = {a.k}")
    if not any(str(r.get("key", "")).startswith("TF-N1:") and r.get("n") == 41 for r in reads):
        raise SystemExit("REFUSING: the void TF-N1 entry (n = 41) is not in the ledger: the K count is not the one the amendment states")
    top = max((r.get("n", 0) for r in reads), default=0)
    if top > a.k:
        raise SystemExit(f"REFUSING: the ledger holds n = {top} > K = {a.k}: another read landed; correct the amendment and re-freeze")
    n1 = json.load(open(os.path.join(run, "n1", "N1_PINS.json"), encoding="utf-8"))["files"]
    bad = [k for k, v in n1.items() if not os.path.exists(os.path.join(run, k)) or sha256(os.path.join(run, k)) != v]
    if bad:
        raise SystemExit(f"REFUSING: N1 pinned files differ or are missing: {bad}")
    p_a1 = os.path.join(run, "n1", "a1", "A1_PINS.json")
    if not os.path.isfile(p_a1) or sha256(p_a1) != a.a1_pins_sha256:
        raise SystemExit(f"REFUSING: A1_PINS.json missing or its sha256 != --a1-pins-sha256 {a.a1_pins_sha256}")
    a1 = json.load(open(p_a1, encoding="utf-8"))["files"]
    for must in ("w/pk/eval/harness/HARNESS_PINS.json", "w/pk/eval/harness/score_e.py", "w/pk/eval/score_p4_a1.py",
                 "n1/a1/tf_n1_analyze_a1.py", "n1/a1/tf_n1_a1_chain.py", "build/out/refe_tfn1_tc20s.npz", "build/out/refe_tfn1_tw.npz",
                 "build/out/refe_tfn1_m.npz", "inputs/refe_p4_teacher_navtest.npz"):
        if must not in a1:
            raise SystemExit(f"REFUSING: A1_PINS.json does not pin {must}")
    bad = [k for k, v in a1.items() if not os.path.exists(os.path.join(run, k)) or sha256(os.path.join(run, k)) != v]
    if bad:
        raise SystemExit(f"REFUSING: A1 pinned files differ or are missing: {bad}")
    hroot = os.path.join(run, "w", "pk", "eval", "harness")
    on_disk = {os.path.relpath(os.path.join(d, f), run).replace(os.sep, "/") for d, _, fs in os.walk(hroot) for f in fs
               if "__pycache__" not in d}
    extra = sorted(on_disk - set(a1))
    if extra:
        raise SystemExit(f"REFUSING: unpinned files under the harness tree: {extra[:5]}")
    for x in FAM_ARMS:
        if not os.path.exists(os.path.join(run, "fam", f"fam_{x}.json")):
            raise SystemExit(f"REFUSING: fam/fam_{x}.json (the void run's W-TRANSP family file, handed to the read) is missing")
    return {"amendment": got[:12], "k": a.k, "n1_pins": len(n1), "a1_pins": len(a1)}


# ------------------------------------------------------------------------------------------------------------ helpers
class Runner:
    def __init__(self, a):
        self.a, self.run = a, a.run_dir
        self.mk = os.path.join(self.run, "a1", "markers")
        os.makedirs(self.mk, exist_ok=True)

    def p(self, *x):
        return os.path.join(self.run, "a1", *x)

    def done(self, st):
        return os.path.exists(os.path.join(self.mk, f"{st}.done"))

    def mark(self, st, ok, info):
        fp = os.path.join(self.mk, f"{st}.{'done' if ok else 'FAIL'}")
        json.dump(dict(info, stage=st, ok=ok, time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(fp, "w", encoding="utf-8"), indent=1)
        log(f"STAGE {st} {'DONE' if ok else 'FAIL'} {json.dumps(info)[:400]}")
        if not ok:
            log("ZZN1A1_CHAIN STOP")
            sys.exit(1)

    def wait_mem(self):
        while True:
            fv = free_virt_mb()
            if fv > self.a.min_free_virt_mb:
                return fv
            log(f"waiting: FreeVirtualMemory {fv} MB <= {self.a.min_free_virt_mb} MB")
            time.sleep(60)

    def parallel(self, jobs):
        rcs, live, queue = [None] * len(jobs), {}, list(enumerate(jobs))
        while queue or live:
            while queue and len(live) < self.a.procs:
                i, (argv, lp) = queue.pop(0)
                fv = self.wait_mem()
                log(f"start job {i} (free virt {fv} MB): {' '.join(argv)[:300]}")
                live[i] = subprocess.Popen(argv, stdout=open(lp, "a", encoding="utf-8"), stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, env=env(self.run), cwd=self.run)
                time.sleep(20)
            for i, pr in list(live.items()):
                if pr.poll() is not None:
                    rcs[i] = pr.returncode
                    log(f"job {i} rc {pr.returncode}")
                    del live[i]
            time.sleep(15)
        return rcs

    def csv_of(self, which):
        lab = SEAMS[which][0]
        return self.p("harness_score", "score", lab, f"{lab}.csv")

    def score_job(self, which):
        lab, _, seam = SEAMS[which]
        os.makedirs(self.p("logs"), exist_ok=True)
        argv = [PY, os.path.join(self.run, "w", "pk", "eval", "score_p4_a1.py"), "--label", lab,
                "--seam", os.path.join(self.run, seam), "--tokens", "all", "--log-dir", self.p("logs"),
                "--score-root", self.p("harness_score")]
        return argv, self.p("logs", f"{lab}_job.log")

    def score_ok(self, which):
        lab = SEAMS[which][0]
        st = os.path.join(self.p("logs"), f"{lab}_status.json")
        status = json.load(open(st, encoding="utf-8")) if os.path.exists(st) else {}
        cp = self.csv_of(which)
        n = 0
        if os.path.exists(cp):
            n = sum(1 for r in csv.DictReader(open(cp, encoding="utf-8")) if r["token"] != "average")
        return status.get("status") == "PASS" and n == N_ALL, {"status": status.get("status"), "csv_tokens": n}


def early_cidh(t_csv: str, banked_csv: str) -> dict:
    """C-ID-H on the harness-re-scored T seam against the banked per-token harness CSV (the control's own scope: N1-T only)."""
    def load(p):
        with open(p, encoding="utf-8", newline="") as fh:
            return {r["token"]: r for r in csv.DictReader(fh) if r["token"] != "average" and r.get("valid", "True") == "True"}
    A, B = load(t_csv), load(banked_csv)
    miss = len(set(B) - set(A)) + len(set(A) - set(B))
    worst = {m: 0.0 for m in MET}
    over = 0
    for t in set(A) & set(B):
        d = {m: abs(float(A[t][c]) - float(B[t][c])) for m, c in MET.items()}
        for m in MET:
            worst[m] = max(worst[m], d[m])
        over += max(d.values()) > ID_TOL
    ok = miss == 0 and over == 0 and len(A) == N_ALL
    return {"pass": ok, "tol": ID_TOL, "tokens_rescored": len(A), "tokens_banked": len(B), "token_set_mismatch": miss,
            "tokens_over_tol": over, "max_abs_diff_by_metric": worst}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--amendment", required=True)
    ap.add_argument("--amendment-sha256", required=True)
    ap.add_argument("--a1-pins-sha256", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--k", type=int, default=42)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--min-free-virt-mb", type=int, default=10240)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stop-after", default="", help="score_T | cidh | score_CWM")
    ap.add_argument("--check-only", action="store_true", help="run the guards and exit (rc 0 = they pass); nothing is started")
    a = ap.parse_args()
    facts = guards(a)
    log(f"guards PASS: {json.dumps(facts)}")
    if a.check_only:
        print("ZZN1A1 GUARDS PASS", flush=True)
        return 0
    R = Runner(a)
    run = a.run_dir
    banked = os.path.join(run, "inputs", "refe_p4_teacher_navtest.csv")
    if not R.done("score_T"):
        for att in range(1, a.retries + 1):
            R.parallel([R.score_job("T")])
            ok, info = R.score_ok("T")
            log(f"score_T attempt {att}: {info}")
            if ok:
                break
        R.mark("score_T", ok, info)
    if a.stop_after == "score_T":
        return 0
    if not R.done("cidh"):
        res = early_cidh(R.csv_of("T"), banked)
        os.makedirs(R.p("out"), exist_ok=True)
        json.dump(res, open(R.p("out", "cidh_early.json"), "w", encoding="utf-8"), indent=1)
        R.mark("cidh", res["pass"], {k: res[k] for k in ("tokens_over_tol", "token_set_mismatch", "tokens_rescored")})
    if a.stop_after == "cidh":
        return 0
    if not R.done("score_CWM"):
        pend = [w for w in ("C", "W", "M") if not R.score_ok(w)[0]]
        for att in range(1, a.retries + 1):
            if not pend:
                break
            R.parallel([R.score_job(w) for w in pend])
            pend = [w for w in pend if not R.score_ok(w)[0]]
            log(f"score_CWM attempt {att}: pending {pend}")
        R.mark("score_CWM", not pend, {"pending": pend})
    if a.stop_after == "score_CWM":
        return 0
    if not R.done("read"):
        os.makedirs(R.p("out"), exist_ok=True)
        an = [PY, os.path.join(run, "n1", "a1", "tf_n1_analyze_a1.py"), "read"]
        argv = an + ["--harness-csv"] + [f"{SEAMS[w][1]}={R.csv_of(w)}" for w in ("T", "C", "W", "M")] + [
            "--known36", os.path.join(run, "inputs", "frame_control_navtest_full.json"),
            "--build-report", os.path.join(run, "build", "out", "n1_build_report.json"),
            "--identity-csv", banked, "--human-csv", os.path.join(run, "inputs", "refe_p4_human_navtest.csv"),
            "--pins", os.path.join(run, "n1", "N1_PINS.json"), "--pin-root", run,
            "--proxy-eval", os.path.join(run, "w", "pk", "eval"), "--prereg-sha256", a.amendment_sha256,
            "--families"] + [f"{x}={os.path.join(run, 'fam', f'fam_{x}.json')}" for x in FAM_ARMS] + [
            "--out", R.p("out", "n1_a1_result.json")]
        rc = subprocess.call(argv, stdout=open(R.p("out", "read.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env(run), cwd=run)
        R.mark("read", rc == 0, {"rc": rc, "log": R.p("out", "read.log")})
    log("ZZN1A1_CHAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
