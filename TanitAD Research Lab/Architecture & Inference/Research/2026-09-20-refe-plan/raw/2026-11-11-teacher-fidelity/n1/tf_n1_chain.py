#!/usr/bin/env python3
"""TF-N1 run chain (PREREG_TF_N1.md section 10). ONE detached process; it runs at most --procs (2) children at a time,
all CPU-only Python (CUDA_VISIBLE_DEVICES=-1), never bash. Resumable: a stage whose marker exists is skipped, and every
child is itself resumable (finished tokens are skipped, errored ones retried).

Refuses to start unless (1) sha256(--prereg) == --prereg-sha256, (2) the K ledger (--ledger) holds the key
"TF-N1:<that sha256>:E3", and (3) every pinned file under --run-dir matches N1_PINS.json. Before every child it waits
until FreeVirtualMemory > --min-free-virt-mb (default 10,240 MB).

Stages (markers in <run>/markers/<stage>.done):
  cderiv    C-DERIV + C-DAM on the 1,018 navtrain held-out rows (p5s_cubic.npz), 2 shards, --check-dam   -> <run>/cderiv/
  build     the navtest plan build, 12,146 tokens, 2 shards, then merge (C-BUILD-COV, C-SEAM)          -> <run>/build/
  score     transp_score.py, arms N1T N1C N1W N1M + human, --split navtest, 2 shards                    -> <run>/score/
  cert      tf_n1_analyze.py cert (CERT virtual arm; primary-only rows for the families)                -> <run>/score/
  families  families_run.py for N1C N1T N1W CERT human on the 12,110, executed_tracked primary          -> <run>/fam/
  read      tf_n1_analyze.py read: controls first, then the registered outcomes                         -> <run>/out/
A stage that fails its own check writes <run>/markers/<stage>.FAIL and the chain STOPS (exit 1): no later stage runs.

Layout under --run-dir (made by n1_prep.sh): w/pk/{refe,eval} and w/tf2_score.py (tip export), repo/{taniteval,stack}
(tip export, for W-FAM), n1/ (these scripts + N1_PINS.json + PREREG_TF_N1.md), inputs/ (rollouts, T seam, CSVs,
known36, p5s_cubic.npz).
"""
from __future__ import annotations

import argparse
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
N_ALL, N_PRIMARY = 12146, 12110        # PREREG_TF_N1 section 2


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


class Runner:
    def __init__(self, a):
        self.a = a
        self.run = a.run_dir
        self.mk = os.path.join(self.run, "markers")
        os.makedirs(self.mk, exist_ok=True)

    def p(self, *x):
        return os.path.join(self.run, *x)

    def done(self, st):
        return os.path.exists(os.path.join(self.mk, f"{st}.done"))

    def mark(self, st, ok: bool, info: dict):
        fp = os.path.join(self.mk, f"{st}.{'done' if ok else 'FAIL'}")
        json.dump(dict(info, stage=st, ok=ok, time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(fp, "w", encoding="utf-8"), indent=1)
        log(f"STAGE {st} {'DONE' if ok else 'FAIL'} {json.dumps(info)[:400]}")
        if not ok:
            log("ZZN1_CHAIN STOP")
            sys.exit(1)

    def wait_mem(self):
        while True:
            fv = free_virt_mb()
            if fv > self.a.min_free_virt_mb:
                return fv
            log(f"waiting: FreeVirtualMemory {fv} MB <= {self.a.min_free_virt_mb} MB")
            time.sleep(60)

    def parallel(self, jobs: list) -> list:
        """jobs: [(argv, logpath)] -> rcs; at most --procs at once, memory-gated before each start."""
        rcs, live, queue = [None] * len(jobs), {}, list(enumerate(jobs))
        while queue or live:
            while queue and len(live) < self.a.procs:
                i, (argv, lp) = queue.pop(0)
                fv = self.wait_mem()
                log(f"start job {i} (free virt {fv} MB): {' '.join(argv)[:300]}")
                live[i] = subprocess.Popen(argv, stdout=open(lp, "a", encoding="utf-8"), stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, env=env(self.run), cwd=self.run)
                time.sleep(20)                                     # let the child allocate before the next memory check
            for i, pr in list(live.items()):
                if pr.poll() is not None:
                    rcs[i] = pr.returncode
                    log(f"job {i} rc {pr.returncode}")
                    del live[i]
            time.sleep(15)
        return rcs


def count_ok(patterns) -> tuple[int, int]:
    ok, err = set(), 0
    for pat in patterns:
        for fp in glob.glob(pat):
            for ln in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    err += 1
                elif "token" in r:
                    ok.add(r["token"])
    return len(ok), err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--prereg-sha256", required=True)
    ap.add_argument("--ledger", required=True, help="the K ledger json (selectors/ledger_K.json) holding the TF-N1 entry")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--min-free-virt-mb", type=int, default=10240)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--fam-n-boot", type=int, default=10000)
    ap.add_argument("--stop-after", default="", help="optional: stop after this stage")
    a = ap.parse_args()
    if a.procs > 2:
        raise SystemExit("REFUSING: at most 2 processes (PREREG_TF_N1 section 10)")
    R = Runner(a)
    run, n1, w, inp = a.run_dir, os.path.join(a.run_dir, "n1"), os.path.join(a.run_dir, "w"), os.path.join(a.run_dir, "inputs")
    # ---- guards: frozen card, K-ledger entry, pinned bytes
    got = sha256(a.prereg)
    if got != a.prereg_sha256:
        raise SystemExit(f"REFUSING: sha256({a.prereg}) = {got} != --prereg-sha256 {a.prereg_sha256}")
    led = json.load(open(a.ledger, encoding="utf-8"))
    key = f"TF-N1:{a.prereg_sha256}:E3"
    if not any(r.get("key") == key for r in led.get("reads", [])):
        raise SystemExit(f"REFUSING: the K ledger {a.ledger} has no entry {key} (freeze, then ledger, then run)")
    pins = json.load(open(os.path.join(n1, "N1_PINS.json"), encoding="utf-8"))["files"]
    bad = [k for k, v in pins.items() if not os.path.exists(os.path.join(run, k)) or sha256(os.path.join(run, k)) != v]
    if bad:
        raise SystemExit(f"REFUSING: pinned files differ or are missing: {bad}")
    log(f"guards PASS: prereg {got[:12]}, ledger key present, {len(pins)} pins")
    build = [PY, os.path.join(n1, "tf_n1_build.py")]
    tf2 = os.path.join(w, "tf2_score.py")
    p5s = os.path.join(inp, "p5s_cubic.npz")
    rollouts = sorted(glob.glob(os.path.join(inp, "rollouts", "rows_s*of4.jsonl*")))
    seam = os.path.join(inp, "refe_p4_teacher_navtest.npz")

    def shards(stage, mk_argv, out_pat, n_expect):
        for att in range(1, a.retries + 1):
            jobs = [(mk_argv(i), R.p(stage, f"{stage}_s{i}of{a.procs}.log")) for i in range(a.procs)]
            R.parallel(jobs)
            n_ok, n_err = count_ok([out_pat])
            log(f"{stage} attempt {att}: {n_ok}/{n_expect} tokens ok, {n_err} error lines")
            if n_ok == n_expect:
                return n_ok
        return n_ok

    # ---- stage cderiv (navtrain only): every row of p5s_cubic.npz (1,018 in the pinned file)
    if not R.done("cderiv"):
        import numpy as np
        N_NAVTRAIN = int(len(np.load(p5s, allow_pickle=False)["tokens"]))
        os.makedirs(R.p("cderiv"), exist_ok=True)
        n = shards("cderiv", lambda i: build + ["derive", "--w", os.path.join(w, "pk"), "--tf2-score", tf2, "--split", "navtrain",
                                                "--p5s", p5s, "--check-dam", "--out", R.p("cderiv", f"derive_s{i}of{a.procs}.jsonl"),
                                                "--shard", f"{i}/{a.procs}"],
                   R.p("cderiv", "derive_s*.jsonl"), N_NAVTRAIN)
        rc = subprocess.call(build + ["merge", "--w", os.path.join(w, "pk"), "--split", "navtrain", "--p5s", p5s,
                                      "--rows", R.p("cderiv", "derive_s*.jsonl"), "--out-dir", R.p("cderiv", "out")],
                             stdout=open(R.p("cderiv", "merge.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env(run), cwd=run)
        rp = R.p("cderiv", "out", "n1_build_report.json")
        rep = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
        ok = rc == 0 and n == N_NAVTRAIN and rep.get("verdict") == "PASS" and "C-DAM (validation only)" in rep.get("controls", {})
        R.mark("cderiv", ok, {"tokens": n, "merge_rc": rc, "controls": rep.get("controls")})
    if a.stop_after == "cderiv":
        return 0
    # ---- stage build (navtest plans; no scorer)
    if not R.done("build"):
        os.makedirs(R.p("build"), exist_ok=True)
        n = shards("build", lambda i: build + ["derive", "--w", os.path.join(w, "pk"), "--tf2-score", tf2, "--split", "navtest",
                                               "--rollouts"] + rollouts + ["--out", R.p("build", f"derive_s{i}of{a.procs}.jsonl"),
                                                                           "--shard", f"{i}/{a.procs}"],
                   R.p("build", "derive_s*.jsonl"), N_ALL)
        rc = subprocess.call(build + ["merge", "--w", os.path.join(w, "pk"), "--split", "navtest", "--rollouts"] + rollouts +
                             ["--seam", seam, "--rows", R.p("build", "derive_s*.jsonl"), "--out-dir", R.p("build", "out")],
                             stdout=open(R.p("build", "merge.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env(run), cwd=run)
        rp = R.p("build", "out", "n1_build_report.json")
        rep = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
        R.mark("build", rc == 0 and n == N_ALL and rep.get("verdict") == "PASS",
               {"tokens": n, "merge_rc": rc, "controls": {k: v.get("pass") for k, v in rep.get("controls", {}).items()},
                "files": rep.get("files"), "timing": rep.get("timing")})
    if a.stop_after == "build":
        return 0
    # ---- stage score (the ONE navtest read of this card)
    if not R.done("score"):
        os.makedirs(R.p("score"), exist_ok=True)
        plans = [f"{nm}={R.p('build', 'out', f'n1_plans_{arm}.npz')}" for nm, arm in
                 (("N1T", "T"), ("N1C", "Tc20s"), ("N1W", "Tw"), ("N1M", "M"))]
        ts = os.path.join(w, "pk", "eval", "instruments", "transp_score.py")
        n = shards("score", lambda i: [PY, ts, "--refe", os.path.join(w, "pk", "refe"), "--split", "navtest", "--token-log", TOKEN_LOG,
                                       "--plans"] + plans + ["--human-arm", "--out", R.p("score", f"rows_s{i}of{a.procs}.jsonl"),
                                                             "--shard", f"{i}/{a.procs}"],
                   R.p("score", "rows_s*.jsonl"), N_ALL)
        R.mark("score", n == N_ALL, {"tokens": n})
    if a.stop_after == "score":
        return 0
    an = [PY, os.path.join(n1, "tf_n1_analyze.py")]
    known36 = os.path.join(inp, "frame_control_navtest_full.json")
    if not R.done("cert"):
        rc = subprocess.call(an + ["cert", "--rows", R.p("score", "rows_s*.jsonl"), "--known36", known36,
                                   "--out", R.p("score", "rows_cert.jsonl"), "--out-primary", R.p("score", "rows_cert_primary.jsonl")],
                             stdout=open(R.p("score", "cert.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env(run), cwd=run)
        R.mark("cert", rc == 0, {"rc": rc})
    if not R.done("families"):
        os.makedirs(R.p("fam"), exist_ok=True)
        fr = os.path.join(w, "pk", "eval", "instruments", "families_run.py")
        jobs = [([PY, fr, "--rows", R.p("score", "rows_cert_primary.jsonl"), "--arm", arm, "--label", f"TF-N1 {arm}",
                  "--out", R.p("fam", f"fam_{arm}.json"), "--n-boot", str(a.fam_n_boot), "--tanitad-repo", R.p("repo")],
                 R.p("fam", f"fam_{arm}.log")) for arm in FAM_ARMS]
        rcs = R.parallel(jobs)
        R.mark("families", all(r == 0 for r in rcs) and all(os.path.exists(R.p("fam", f"fam_{x}.json")) for x in FAM_ARMS),
               {"rcs": rcs})
    if not R.done("read"):
        os.makedirs(R.p("out"), exist_ok=True)
        rc = subprocess.call(an + ["read", "--rows", R.p("score", "rows_cert.jsonl"), "--known36", known36,
                                   "--build-report", R.p("build", "out", "n1_build_report.json"),
                                   "--identity-csv", os.path.join(inp, "refe_p4_teacher_navtest.csv"),
                                   "--human-csv", os.path.join(inp, "refe_p4_human_navtest.csv"),
                                   "--pins", os.path.join(n1, "N1_PINS.json"), "--pin-root", run,
                                   "--proxy-eval", os.path.join(w, "pk", "eval"), "--prereg-sha256", a.prereg_sha256,
                                   "--families"] + [f"{x}={R.p('fam', f'fam_{x}.json')}" for x in FAM_ARMS] +
                             ["--out", R.p("out", "n1_result.json")],
                             stdout=open(R.p("out", "read.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, env=env(run), cwd=run)
        R.mark("read", rc == 0, {"rc": rc, "log": R.p("out", "read.log")})
    log("ZZN1_CHAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
