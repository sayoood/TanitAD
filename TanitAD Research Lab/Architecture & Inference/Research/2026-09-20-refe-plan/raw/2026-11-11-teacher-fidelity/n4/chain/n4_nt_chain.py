#!/usr/bin/env python3
"""TF-N4 navtest run chain (PREREG_TF_N4.md section 11): guards -> builder tracked scoring of the N2-R plans (C-DACMAP, C-DERIV) -> REPAIR-D / N4-X / N4-DM
(C-UNCH, C-DET) -> plan sets in TF-N2 A1's seam format -> FULL-seam harness scoring of the three arms -> read -> families. CPU only, at most 2 processes,
detached, marker-based, resumable. Modelled on tfn2_nt_chain.py (TF-N2 A1).

REFUSES TO START unless ALL hold (each is tested in test_n4_chain.py):
  1. sha256(--card) == --card-sha256
  2. the K ledger holds "TF-N4:<that sha256>:E3" with n == --k (45), the TF-N2-A1 entry (n 44) is present, and no entry has n > --k
  3. sha256(N4_PINS.json) == --pins-sha256, every pinned file under --run-dir matches, every REQUIRED file is pinned, no unpinned .py/.json/.yaml file
     exists under w/, repo/, n2/ or n4/, and the metric-cache listing digest equals the pinned one
  4. --procs <= 2; --margin / --iters / --sign-n4x equal the card (0.05 / 5 / -1); the argv the chain builds for the build stage names no --mutate
  5. `n4_nt_build.py reach`: margin, cap, sign and certificate READ BACK from the three built repair objects equal the card, and a stub-scorer probe shows
     they reach the shift and the loop (the build re-checks the same before it derives a single token)

Stages (markers <root>/markers/<stage>.done | .FAIL; a FAIL stops the chain; ZZN4_CHAIN STOP):
  derive        n4_nt_build.py derive, 2 shards (builder tracked score of the N2-R plan on all 12,146 tokens; repair where DAC = 0)
  build         n4_nt_build.py merge: C-COVB, C-DACMAP (0 mismatches on 12,146), C-DERIV (200 tokens, 1e-9), C-UNCH, C-DET, C-CERT-B; writes plans + seams
  unch          C-UNCH again on the files on disk, just before the harness reads them
  score_D_DM    the N4-D and N4-DM seams, FULL 12,146 tokens, through score_p4_a1.py -> eval/harness/score_e.py (the pinned W3 harness); PASS + full row count
  score_X       the N4-X seam, the same way
  read          n4_nt_analyze.py read (controls first; exits before any N4-D aggregate if a control fails)
  families      transp_score.py --metric-cache-root (N2R, N4D, human) -> famprep -> families_run.py --per-window (N2R, N4D) -> fam_ci.py paired -> famread
Launch (detached): Win32_Process Create, CreateFlags 0x08000200, `cmd /c "python n4_nt_chain.py ... >> chain.log 2>&1 < NUL"` (see CHAIN_N4.md).
`--dry-run` prints every command and writes nothing; `--check-only` runs the guards and exits."""
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
NAVSIM_ROOT = "F:/Archive/devbox-C/navsim"
CACHE_ROOT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
TOKEN_LOG = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
MAPS = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
N_ALL, N_PRIMARY = 12146, 12110
CARD = {"margin": 0.05, "iters": 5, "sign_n4x": -1.0}
SEAMS = {"D": "refe_tfn4_n4d", "DM": "refe_tfn4_n4dm", "X": "refe_tfn4_n4x"}
REQUIRED = ("w/pk/eval/harness/HARNESS_PINS.json", "w/pk/eval/harness/score_e.py", "w/pk/eval/score_p4_a1.py", "w/pk/eval/run_p4.py",
            "w/pk/eval/instruments/transp_score.py", "w/pk/eval/instruments/families_run.py", "w/pk/eval/proxy_eval.py", "w/p5r/p5r_labels.py",
            "w/pk/refe/navsim_dac.py", "w/pk/eval/p5_target_labels.py",
            "n2/tf_n1_analyze_a1.py", "n2/tfn2_nt_analyze.py", "n4/n4_nt_chain.py", "n4/n4_nt_analyze.py", "n4/n4_nt_build.py", "n4/n4_geom.py", "n4/n4_repair.py",
            "n4/n4_common.py", "n4/fam_ci.py",
            "inputs/refe_p4_teacher_navtest.npz", "inputs/refe_p4_teacher_navtest.csv", "inputs/refe_p4_human_navtest.csv", "inputs/frame_control_navtest_full.json",
            "inputs/n1_csv/refe_a1_tfn1_c.csv", "inputs/n2r/n2_plans_N2R.npz", "inputs/n2r/refe_tfn2_n2r.npz", "inputs/n2r/refe_tfn2_n2r.csv",
            "inputs/n2r/refe_tfn2_n2c.csv", "inputs/n2r/rolls_RJ5/rows_s0of2.jsonl", "inputs/n2r/rolls_RJ5/rows_s1of2.jsonl", TOKEN_LOG)
WATCH_EXT = (".py", ".json", ".yaml", ".yml")
WATCH_TOP = ("w", "repo", "n2", "n4")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_listing_sha(root: str) -> tuple[str, int]:
    """digest of the sorted `<log>/unknown/<token>/metric_cache.pkl:<size>` lines of the harness metric cache"""
    lines = []
    for lg in sorted(os.listdir(root)):
        d = os.path.join(root, lg, "unknown")
        if os.path.isdir(d):
            for tok in sorted(os.listdir(d)):
                fp = os.path.join(d, tok, "metric_cache.pkl")
                if os.path.isfile(fp):
                    lines.append(f"{lg}/unknown/{tok}/metric_cache.pkl:{os.path.getsize(fp)}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), len(lines)


def free_virt_mb() -> int:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1024)"],
                       capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def env(run: str) -> dict:
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "REFE_SIM_HZ": "10",
              "NUPLAN_MAPS_ROOT": MAPS, "PYTHONIOENCODING": "utf-8", "TANITAD_NAVSIM_ROOT": NAVSIM_ROOT,
              "N4_W": os.path.join(run, "w", "pk").replace("\\", "/"), "N4_P5R": os.path.join(run, "w", "p5r").replace("\\", "/"),
              "PYTHONPATH": ";".join([DEVKIT.replace("/", "\\"), os.path.join(run, "w", "pk", "refe").replace("/", "\\")])})
    return e


def build_argv(run: str, root: str, shard: int, procs: int) -> list:
    """the derive argv the chain issues; has no mutation flag, and the card numbers are not argv at all (they are read back from the built repair)"""
    return [PY, os.path.join(run, "n4", "n4_nt_build.py"), "derive", "--route", "cache", "--n2r-plans", os.path.join(run, "inputs", "n2r", "n2_plans_N2R.npz"),
            "--rolls", os.path.join(run, "inputs", "n2r", "rolls_RJ5", "rows_s*of2.jsonl"), "--metric-cache-root", CACHE_ROOT,
            "--out", os.path.join(root, "build", f"derive_s{shard}of{procs}.jsonl"), "--shard", f"{shard}/{procs}"]


def run_reach(run: str) -> tuple:
    r = subprocess.run([PY, os.path.join(run, "n4", "n4_nt_build.py"), "reach"], capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL,
                       env=env(run), cwd=run)
    return r.returncode, r.stdout + r.stderr


def guards(a) -> dict:
    """raise SystemExit('REFUSING: ...') on the first failed guard"""
    if a.procs > 2 or a.procs < 1:
        raise SystemExit("REFUSING: --procs must be 1 or 2 (the card, section 11)")
    for k, want in (("margin", CARD["margin"]), ("iters", CARD["iters"]), ("sign_n4x", CARD["sign_n4x"])):
        if getattr(a, k) != want:
            raise SystemExit(f"REFUSING: --{k.replace('_', '-')} {getattr(a, k)} != the card's {want}")
    if any("--mutate" in x or "--smoke-allow-mutation" in x for x in build_argv(a.run_dir, a.run_dir, 0, a.procs)):
        raise SystemExit("REFUSING: the build argv names a mutation flag")
    run = a.run_dir
    if not os.path.isfile(a.card):
        raise SystemExit(f"REFUSING: card {a.card} is missing")
    got = sha256(a.card)
    if got != a.card_sha256:
        raise SystemExit(f"REFUSING: sha256({a.card}) = {got} != --card-sha256 {a.card_sha256}")
    if not os.path.isfile(a.ledger):
        raise SystemExit(f"REFUSING: ledger {a.ledger} is missing")
    reads = json.load(open(a.ledger, encoding="utf-8")).get("reads", [])
    key = f"TF-N4:{a.card_sha256}:E3"
    mine = [r for r in reads if r.get("key") == key]
    if not mine:
        raise SystemExit(f"REFUSING: the K ledger has no entry {key} (freeze, then ledger, then run)")
    if mine[0].get("n") != a.k:
        raise SystemExit(f"REFUSING: ledger entry {key} has n = {mine[0].get('n')}, the card says K = {a.k}")
    if not any(str(r.get("key", "")).startswith("TF-N2-A1:") and r.get("n") == 44 for r in reads):
        raise SystemExit("REFUSING: the TF-N2-A1 entry (n = 44) is not in the ledger: the K count is not the one the card states")
    top = max((r.get("n", 0) for r in reads), default=0)
    if top > a.k:
        raise SystemExit(f"REFUSING: the ledger holds n = {top} > K = {a.k}: another read landed; correct the card and re-freeze")
    pp = os.path.join(run, "n4", "N4_PINS.json")
    if not os.path.isfile(pp) or sha256(pp) != a.pins_sha256:
        raise SystemExit(f"REFUSING: N4_PINS.json missing or its sha256 != --pins-sha256 {a.pins_sha256}")
    pj = json.load(open(pp, encoding="utf-8"))
    pins = pj["files"]
    for must in REQUIRED:
        if must not in pins:
            raise SystemExit(f"REFUSING: N4_PINS.json does not pin {must}")
    bad = [k for k, v in pins.items() if not os.path.exists(os.path.join(run, k)) or sha256(os.path.join(run, k)) != v]
    if bad:
        raise SystemExit(f"REFUSING: pinned files differ or are missing: {bad[:6]}")
    on_disk = set()
    for top_d in WATCH_TOP:
        for d, _, fs in os.walk(os.path.join(run, top_d)):
            if "__pycache__" in d:
                continue
            for f in fs:
                if f.endswith(WATCH_EXT):
                    on_disk.add(os.path.relpath(os.path.join(d, f), run).replace(os.sep, "/"))
    extra = sorted(on_disk - set(pins) - {"n4/N4_PINS.json"})
    if extra:
        raise SystemExit(f"REFUSING: unpinned files under w/, repo/, n2/ or n4/: {extra[:5]}")
    want_cache = pj.get("metric_cache_listing_sha256")
    if not a.skip_cache_listing:
        got_c, n_c = cache_listing_sha(CACHE_ROOT)
        if got_c != want_cache or n_c < N_ALL:
            raise SystemExit(f"REFUSING: metric-cache listing digest {got_c} ({n_c} files) != pinned {want_cache}")
    rc, txt = run_reach(run)
    if rc != 0:
        raise SystemExit("REFUSING: reachability (margin / cap / sign / certificate read back from the built repair) failed: " + txt[-600:])
    return {"card": got[:12], "k": a.k, "pins": len(pins), "reach": "PASS"}


def compare_rows(csv_path: str, n_expect: int) -> tuple:
    n = sum(1 for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["token"] != "average") if os.path.exists(csv_path) else 0
    return n == n_expect, n


class Runner:
    def __init__(self, a):
        self.a, self.run, self.root = a, a.run_dir, os.path.join(a.run_dir, "n4run")
        self.mk = os.path.join(self.root, "markers")
        if not a.dry_run:
            os.makedirs(self.mk, exist_ok=True)

    def p(self, *x):
        return os.path.join(self.root, *x)

    def done(self, st):
        return os.path.exists(os.path.join(self.mk, f"{st}.done"))

    def mark(self, st, ok, info):
        if self.a.dry_run:
            log(f"DRY-RUN stage {st} would be marked")
            return
        fp = os.path.join(self.mk, f"{st}.{'done' if ok else 'FAIL'}")
        json.dump(dict(info, stage=st, ok=ok, time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(fp, "w", encoding="utf-8"), indent=1)
        log(f"STAGE {st} {'DONE' if ok else 'FAIL'} {json.dumps(info)[:400]}")
        if not ok:
            log("ZZN4_CHAIN STOP")
            sys.exit(1)

    def wait_mem(self):
        while not self.a.dry_run:
            fv = free_virt_mb()
            if fv > self.a.min_free_virt_mb:
                return fv
            log(f"waiting: FreeVirtualMemory {fv} MB <= {self.a.min_free_virt_mb} MB")
            time.sleep(60)
        return 0

    def call(self, argv, logpath, cwd=None):
        if self.a.dry_run:
            log("DRY-RUN CMD " + " ".join(argv))
            return 0
        return subprocess.call(argv, stdout=open(logpath, "a", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env(self.run), cwd=cwd or self.run)

    def parallel(self, jobs, cwd=None):
        """jobs = [(argv, logpath)] with at most --procs alive; returns the return codes"""
        if self.a.dry_run:
            for argv, _lp in jobs:
                log("DRY-RUN CMD " + " ".join(argv))
            return [0] * len(jobs)
        rcs, live, queue = [None] * len(jobs), {}, list(enumerate(jobs))
        while queue or live:
            while queue and len(live) < self.a.procs:
                i, (argv, lp) = queue.pop(0)
                fv = self.wait_mem()
                log(f"start job {i} (free virt {fv} MB): {' '.join(argv)[:300]}")
                live[i] = subprocess.Popen(argv, stdout=open(lp, "a", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env(self.run), cwd=cwd or self.run)
                time.sleep(20)
            for i, pr in list(live.items()):
                if pr.poll() is not None:
                    rcs[i] = pr.returncode
                    log(f"job {i} rc {pr.returncode}")
                    del live[i]
            time.sleep(15)
        return rcs

    def csv_of(self, which):
        lab = SEAMS[which]
        return self.p("harness_score", "score", lab, f"{lab}.csv")

    def score_job(self, which):
        lab = SEAMS[which]
        argv = [PY, os.path.join(self.run, "w", "pk", "eval", "score_p4_a1.py"), "--label", lab, "--seam", self.p("build", "out", lab + ".npz"), "--tokens", "all",
                "--log-dir", self.p("logs"), "--score-root", self.p("harness_score")]
        return argv, self.p("logs", f"{lab}_job.log")

    def score_ok(self, which):
        lab = SEAMS[which]
        st = os.path.join(self.p("logs"), f"{lab}_status.json")
        status = json.load(open(st, encoding="utf-8")) if os.path.exists(st) else {}
        ok, n = compare_rows(self.csv_of(which), N_ALL)
        return status.get("status") == "PASS" and ok, {"status": status.get("status"), "csv_tokens": n, "want": N_ALL}

    def score_stage(self, which_list):
        if self.a.dry_run:
            self.parallel([self.score_job(w) for w in which_list])
            return []
        pend = [w for w in which_list if not self.score_ok(w)[0]]
        for att in range(1, self.a.retries + 1):
            if not pend:
                break
            self.parallel([self.score_job(w) for w in pend])
            pend = [w for w in pend if not self.score_ok(w)[0]]
            log(f"score {which_list} attempt {att}: pending {pend}")
        return pend


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--card-sha256", required=True)
    ap.add_argument("--pins-sha256", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--k", type=int, default=45)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--margin", type=float, default=CARD["margin"])
    ap.add_argument("--iters", type=int, default=CARD["iters"])
    ap.add_argument("--sign-n4x", type=float, default=CARD["sign_n4x"])
    ap.add_argument("--min-free-virt-mb", type=int, default=10240)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stop-after", default="", help="derive | build | unch | score_D_DM | score_X | read | families")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print every command, write nothing; guard failures are printed, not fatal")
    ap.add_argument("--skip-cache-listing", action="store_true", help="tests only")
    a = ap.parse_args()
    try:
        facts = guards(a)
        log(f"guards PASS: {json.dumps(facts)}")
    except SystemExit as e:
        if not a.dry_run:
            raise
        log(f"DRY-RUN: guard would {e}")
    if a.check_only:
        print("ZZN4 GUARDS PASS", flush=True)
        return 0
    R = Runner(a)
    run, procs = a.run_dir, a.procs
    stop = lambda st: a.stop_after == st                                        # noqa: E731
    if not a.dry_run:
        for d in ("build", "build/out", "logs", "out", "score", "fam"):
            os.makedirs(R.p(*d.split("/")), exist_ok=True)
    B = os.path.join(run, "n4", "n4_nt_build.py")
    inp = lambda *x: os.path.join(run, "inputs", *x)                            # noqa: E731
    # ---- derive: builder tracked score of the N2-R plans + the repairs
    if not R.done("derive"):
        jobs = [(build_argv(run, R.root, i, procs), R.p("build", f"derive_s{i}.log")) for i in range(procs)]
        rcs = R.parallel(jobs)
        R.mark("derive", not any(rcs), {"rcs": rcs})
    if stop("derive"):
        return 0
    common_merge = ["--n2r-plans", inp("n2r", "n2_plans_N2R.npz"), "--n2r-seam", inp("n2r", "refe_tfn2_n2r.npz")]
    if not R.done("build"):
        mg = [PY, B, "merge", "--route", "cache"] + common_merge + ["--rolls", "", "--rows", R.p("build", "derive_s*of*.jsonl"), "--seam", inp("refe_p4_teacher_navtest.npz"),
              "--harness-csv-n2r", inp("n2r", "refe_tfn2_n2r.csv"), "--out-dir", R.p("build", "out")]
        rc = R.call(mg, R.p("build", "merge.log"))
        rp = R.p("build", "out", "n4_build_report.json")
        rep = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
        R.mark("build", rc == 0 and (rep.get("verdict") == "PASS" or a.dry_run),
               {"merge_rc": rc, "controls": {k: v.get("pass") for k, v in rep.get("controls", {}).items()}, "dac0": rep.get("dac0_tokens_builder"),
                "repaired": {k: len(v) for k, v in rep.get("repaired", {}).items()}})
    if stop("build"):
        return 0
    if not R.done("unch"):
        rc = R.call([PY, B, "unch", "--out-dir", R.p("build", "out")] + common_merge + ["--out", R.p("out", "unch.json")], R.p("out", "unch.log"))
        R.mark("unch", rc == 0, {"rc": rc})
    if stop("unch"):
        return 0
    if not R.done("score_D_DM"):
        pend = R.score_stage(["D", "DM"])
        R.mark("score_D_DM", not pend, {"pending": pend, **(R.score_ok("D")[1] if not a.dry_run else {})})
    if stop("score_D_DM"):
        return 0
    if not R.done("score_X"):
        pend = R.score_stage(["X"])
        R.mark("score_X", not pend, {"pending": pend})
    if stop("score_X"):
        return 0
    if not R.done("read"):
        an = [PY, os.path.join(run, "n4", "n4_nt_analyze.py"), "read", "--harness-csv", f"N2R={inp('n2r', 'refe_tfn2_n2r.csv')}", f"N4D={R.csv_of('D')}",
              f"N4X={R.csv_of('X')}", f"N4DM={R.csv_of('DM')}", f"N2C={inp('n2r', 'refe_tfn2_n2c.csv')}", f"N1C={inp('n1_csv', 'refe_a1_tfn1_c.csv')}",
              f"T={inp('refe_p4_teacher_navtest.csv')}", f"human={inp('refe_p4_human_navtest.csv')}", "--build-report", R.p("build", "out", "n4_build_report.json"),
              "--unch", R.p("out", "unch.json"), "--pins", os.path.join(run, "n4", "N4_PINS.json"), "--pin-root", run,
              "--proxy-eval", os.path.join(run, "w", "pk", "eval"), "--prereg-sha256", a.card_sha256, "--token-log", TOKEN_LOG,
              "--known36", inp("frame_control_navtest_full.json"), "--k", str(a.k), "--a1-analyze", os.path.join(run, "n2", "tf_n1_analyze_a1.py"),
              "--out", R.p("out", "n4_result.json")]
        rc = R.call(an, R.p("out", "read.log"))
        R.mark("read", rc == 0, {"rc": rc, "log": R.p("out", "read.log")})
    if stop("read"):
        return 0
    if not R.done("families"):
        ts = os.path.join(run, "w", "pk", "eval", "instruments", "transp_score.py")
        plans = [f"N2R={inp('n2r', 'n2_plans_N2R.npz')}", f"N4D={R.p('build', 'out', 'n4_plans_N4D.npz')}"]
        jobs = [([PY, ts, "--refe", os.path.join(run, "w", "pk", "refe"), "--split", "navtest", "--token-log", TOKEN_LOG, "--metric-cache-root", CACHE_ROOT, "--plans"] + plans +
                 ["--human-arm", "--out", R.p("score", f"rows_s{i}of{procs}.jsonl"), "--shard", f"{i}/{procs}"], R.p("score", f"transp_s{i}.log")) for i in range(procs)]
        rcs = R.parallel(jobs)
        n2 = os.path.join(run, "n2", "tfn2_nt_analyze.py")
        rc = R.call([PY, n2, "famprep", "--rows", R.p("score", "rows_s*of*.jsonl"), "--out", R.p("score", "rows_primary.jsonl"), "--known36", inp("frame_control_navtest_full.json")],
                    R.p("score", "famprep.log"))
        fr = os.path.join(run, "w", "pk", "eval", "instruments", "families_run.py")
        fj = [([PY, fr, "--rows", R.p("score", "rows_primary.jsonl"), "--arm", arm, "--label", f"TF-N4 {arm}", "--out", R.p("fam", f"fam_{arm}.json"), "--per-window",
                "--n-boot", "2000", "--tanitad-repo", os.path.join(run, "repo")], R.p("fam", f"fam_{arm}.log")) for arm in ("N2R", "N4D")]
        frc = R.parallel(fj)
        rc2 = R.call([PY, os.path.join(run, "n4", "fam_ci.py"), "--a", R.p("fam", "fam_N2R.json"), "--b", R.p("fam", "fam_N4D.json"), "--out", R.p("out", "fam_ci_N4D_vs_N2R.json")],
                     R.p("out", "fam_ci.log"))
        rc3 = R.call([PY, os.path.join(run, "n4", "n4_nt_analyze.py"), "famread", "--k", str(a.k), "--fam-ci", R.p("out", "fam_ci_N4D_vs_N2R.json"), "--out", R.p("out", "n4_families.json")],
                     R.p("out", "famread.log"))
        R.mark("families", not any(rcs) and rc == 0 and not any(frc) and rc2 == 0 and rc3 == 0, {"transp_rcs": rcs, "famprep_rc": rc, "fam_rcs": frc, "fam_ci_rc": rc2, "famread_rc": rc3})
    log("ZZN4_CHAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
