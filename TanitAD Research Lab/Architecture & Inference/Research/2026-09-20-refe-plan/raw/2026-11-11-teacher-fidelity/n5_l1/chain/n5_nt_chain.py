#!/usr/bin/env python3
"""TF-N5 navtest run chain (PREREG_TF_N5_DRAFT.md sections 3 to 9): guards -> L1 reachability on the real fixtures -> builder tracked scoring of the N2-R plans on all
12,146 tokens (C-DACMAP, C-DERIV) with L1 run on the REPAIR-D residual (N5-A / N5-X on the 25, N5-R builder-only on all 116) -> plan sets in TF-N4 A1's seam format
(C-UNCH, C-DET, C-CERT-B) -> FULL-seam harness scoring of N5-A and N5-X -> read -> families. CPU only, at most 2 processes, detached, marker-based, resumable.
Adapted from n4_nt_chain.py (TF-N4 A1); the pinned A1 files it builds on (n4_geom.py, n4_repair.py, n4_nt_build.py, fam_ci.py) are imported, never copied.

REFUSES TO START unless ALL hold (each is tested in test_n5_chain.py):
  1. sha256(--card) == --card-sha256
  2. the K ledger holds "TF-N5:<that sha256>:E3" with n == --k (47), the TF-N4-A1 entry (n 46) is present, and no entry has n > --k
  3. sha256(N5_PINS.json) == --pins-sha256, every pinned file under --run-dir matches, every REQUIRED file is pinned, no unpinned .py/.json/.yaml file exists under
     w/, repo/, n2/, n4/ or n5/, the metric-cache listing digest equals the pinned one, AND the pinned `families_run.py` is the TIP version (sha256 44c62807...):
     the TF-N4 A1 families stage failed because its tree pinned an older one that writes no `tokens`/`logs` under --per-window
  4. --procs <= 2; every L1 flag (--margin --iters --cert --centre-m --fallback-margin --damp --sign --sign-n5x, no --damp-cum) equals the card; the argv the chain builds
     for the build stage names no --mutate
  5. `n5_nt_build.py reach`: margin, cap, sign, certificate, centre_m, fallback_margin and damp READ BACK from the three built repair objects equal the card, and their
     behavioural signature on synthetic corridors equals the card object's
  6. the run drive has >= --min-free-gb free (20 GB)

Stages (markers <root>/markers/<stage>.done | .FAIL; a FAIL stops the chain; ZZN5_CHAIN STOP):
  reach_real  n5_nt_build.py reach --real (the two real navtrain fixture instances: C-REACH)
  derive      n5_nt_build.py derive, 2 shards
  build       n5_nt_build.py merge: C-COVB, C-DACMAP, C-DERIV, C-LABELEGO, C-DET, C-SET, C-CERT-B, C-SEAMFN, C-UNCH, C-REACH; writes plans + seams
  unch        C-UNCH again on the files on disk, just before the harness reads them
  score_A_X   the N5-A and N5-X seams, FULL 12,146 tokens, through score_p4_a1.py -> eval/harness/score_e.py (the pinned W3 harness); PASS + full row count
  read        n5_nt_analyze.py read (controls first; exits before any N5-A aggregate if a control fails)
  families    transp_score.py (N4D, N5A, human) -> famprep -> CHECK the rows -> families_run.py --per-window (N4D, N5A) -> CHECK `tokens`/`logs` in both JSONs ->
              fam_ci.py paired (A = N4D, B = N5A) -> famread; sub-steps carry their own markers, so a failed late step re-runs only itself
Launch (detached): Win32_Process Create, CreateFlags 0x08000200, `cmd /c "python n5_nt_chain.py ... >> chain.log 2>&1 < NUL"` (see launch.cmd / CHAIN_N5.md).
`--dry-run` prints every command and writes nothing; `--check-only` runs the guards and exits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
NAVSIM_ROOT = "F:/Archive/devbox-C/navsim"
CACHE_ROOT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
TOKEN_LOG = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
MAPS = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
N_ALL, N_PRIMARY = 12146, 12110
TIP_FAMILIES = "w/pk/eval/instruments/families_run.py"
TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"
# the card's frozen L1 numbers (PREREG_TF_N5 section 3); flag name -> value
CARD = {"margin": 0.05, "iters": 5, "cert": "full", "centre_m": 0.05, "fallback_margin": 0.005, "damp": 0.5, "sign": 1.0, "sign_n5x": -1.0}
SEAMS = {"A": "refe_tfn5_n5a", "X": "refe_tfn5_n5x"}
REQUIRED = ("w/pk/eval/harness/HARNESS_PINS.json", "w/pk/eval/harness/score_e.py", "w/pk/eval/score_p4_a1.py", "w/pk/eval/run_p4.py",
            "w/pk/eval/instruments/transp_score.py", TIP_FAMILIES, "w/pk/eval/proxy_eval.py", "w/p5r/p5r_labels.py",
            "w/pk/refe/navsim_dac.py", "w/pk/eval/p5_target_labels.py",
            "n2/tf_n1_analyze_a1.py", "n2/tfn2_nt_analyze.py", "n4/n4_geom.py", "n4/n4_repair.py", "n4/n4_nt_build.py", "n4/n4_common.py", "n4/fam_ci.py",
            "n5/n4_repair_l1.py", "n5/n5_env.py", "n5/test_n4_repair_l1.py", "n5/fixture_real_instances.json", "n5/n5_nt_chain.py", "n5/n5_nt_build.py", "n5/n5_nt_analyze.py",
            "n5/n5_make_pins.py", "n5/test_n5_chain.py",
            "inputs/refe_p4_teacher_navtest.csv", "inputs/refe_p4_human_navtest.csv", "inputs/frame_control_navtest_full.json",
            "inputs/n2r/n2_plans_N2R.npz", "inputs/n2r/refe_tfn2_n2r.npz", "inputs/n2r/refe_tfn2_n2r.csv",
            "inputs/n2r/rolls_RJ5/rows_s0of2.jsonl", "inputs/n2r/rolls_RJ5/rows_s1of2.jsonl",
            "inputs/n4d/n4_plans_N4D.npz", "inputs/n4d/refe_tfn4_n4d.npz", "inputs/n4d/refe_tfn4_n4d.csv", "inputs/n4d/n4_build_report.json",
            "inputs/n4d/derive_s0of2.jsonl", "inputs/n4d/derive_s1of2.jsonl", "inputs/n5/residual25.json", TOKEN_LOG)
WATCH_EXT = (".py", ".json", ".yaml", ".yml")
WATCH_TOP = ("w", "repo", "n2", "n4", "n5")
PINS_REL = "n5/N5_PINS.json"


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


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def env(run: str) -> dict:
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "REFE_SIM_HZ": "10",
              "NUPLAN_MAPS_ROOT": MAPS, "PYTHONIOENCODING": "utf-8", "TANITAD_NAVSIM_ROOT": NAVSIM_ROOT, "PYTHONDONTWRITEBYTECODE": "1",
              "N4_RUN": run.replace("\\", "/"),
              "N4_W": os.path.join(run, "w", "pk").replace("\\", "/"), "N4_P5R": os.path.join(run, "w", "p5r").replace("\\", "/"),
              "PYTHONPATH": ";".join([DEVKIT.replace("/", "\\"), os.path.join(run, "w", "pk", "refe").replace("/", "\\")])})
    return e


def l1_flags(a) -> list:
    """the L1 flags as ARGV: every card number is passed explicitly to the build, so the read-back in the build compares what the argv said with the card"""
    return ["--margin", str(a.margin), "--iters", str(a.iters), "--cert", a.cert, "--centre-m", str(a.centre_m), "--fallback-margin", str(a.fallback_margin),
            "--damp", str(a.damp), "--sign", str(a.sign), "--sign-x", str(a.sign_n5x)] + (["--damp-cum"] if a.damp_cum else [])


def build_argv(a, root: str, shard: int, procs: int) -> list:
    run = a.run_dir
    return [PY, os.path.join(run, "n5", "n5_nt_build.py"), "derive", "--route", "cache", "--n2r-plans", os.path.join(run, "inputs", "n2r", "n2_plans_N2R.npz"),
            "--n4d-report", os.path.join(run, "inputs", "n4d", "n4_build_report.json"),
            "--rolls", os.path.join(run, "inputs", "n2r", "rolls_RJ5", "rows_s*of2.jsonl"), "--metric-cache-root", CACHE_ROOT,
            "--out", os.path.join(root, "build", f"derive_s{shard}of{procs}.jsonl"), "--shard", f"{shard}/{procs}"] + l1_flags(a)


def run_reach(a, real: bool = False, out: str = "") -> tuple:
    run = a.run_dir
    argv = [PY, os.path.join(run, "n5", "n5_nt_build.py"), "reach"] + l1_flags(a)
    if real:
        argv += ["--real", "--fixture", os.path.join(run, "n5", "fixture_real_instances.json"), "--out", out]
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, env=env(run), cwd=run)
    return r.returncode, r.stdout + r.stderr


def guards(a) -> dict:
    """raise SystemExit('REFUSING: ...') on the first failed guard"""
    if a.procs > 2 or a.procs < 1:
        raise SystemExit("REFUSING: --procs must be 1 or 2 (the card, section 5)")
    for k, want in CARD.items():
        if getattr(a, k) != want:
            raise SystemExit(f"REFUSING: --{k.replace('_', '-')} {getattr(a, k)} != the card's {want}")
    if a.damp_cum:
        raise SystemExit("REFUSING: --damp-cum is not the card's (damp 0.5 non-cumulative)")
    if any("--mutate" in x or "--smoke-allow-mutation" in x for x in build_argv(a, a.run_dir, 0, a.procs)):
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
    key = f"TF-N5:{a.card_sha256}:E3"
    mine = [r for r in reads if r.get("key") == key]
    if not mine:
        raise SystemExit(f"REFUSING: the K ledger has no entry {key} (freeze, then ledger, then run)")
    if mine[0].get("n") != a.k:
        raise SystemExit(f"REFUSING: ledger entry {key} has n = {mine[0].get('n')}, the card says K = {a.k}")
    if not any(str(r.get("key", "")).startswith("TF-N4:") and r.get("n") == 46 for r in reads):
        raise SystemExit("REFUSING: the TF-N4-A1 entry (n = 46) is not in the ledger: the K count is not the one the card states")
    top = max((r.get("n", 0) for r in reads), default=0)
    if top > a.k:
        raise SystemExit(f"REFUSING: the ledger holds n = {top} > K = {a.k}: another read landed; correct the card and re-freeze")
    pp = os.path.join(run, PINS_REL)
    if not os.path.isfile(pp) or sha256(pp) != a.pins_sha256:
        raise SystemExit(f"REFUSING: N5_PINS.json missing or its sha256 != --pins-sha256 {a.pins_sha256}")
    pj = json.load(open(pp, encoding="utf-8"))
    pins = pj["files"]
    for must in REQUIRED:
        if must not in pins:
            raise SystemExit(f"REFUSING: N5_PINS.json does not pin {must}")
    if pins[TIP_FAMILIES] != TIP_FAMILIES_SHA256:
        raise SystemExit(f"REFUSING: N5_PINS.json pins {TIP_FAMILIES} at {pins[TIP_FAMILIES][:12]}..., not the TIP version {TIP_FAMILIES_SHA256[:12]}... "
                         "(an older families_run.py writes no tokens/logs under --per-window: the TF-N4 A1 families stage failed on exactly this)")
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
    extra = sorted(on_disk - set(pins) - {PINS_REL})
    if extra:
        raise SystemExit(f"REFUSING: unpinned files under w/, repo/, n2/, n4/ or n5/: {extra[:5]}")
    want_cache = pj.get("metric_cache_listing_sha256")
    if not a.skip_cache_listing:
        got_c, n_c = cache_listing_sha(CACHE_ROOT)
        if got_c != want_cache or n_c < N_ALL:
            raise SystemExit(f"REFUSING: metric-cache listing digest {got_c} ({n_c} files) != pinned {want_cache}")
    if free_gb(run) < a.min_free_gb:
        raise SystemExit(f"REFUSING: {run} has {free_gb(run):.1f} GB free < --min-free-gb {a.min_free_gb}")
    rc, txt = run_reach(a)
    if rc != 0:
        raise SystemExit("REFUSING: reachability (L1 parameters read back from the built repair, and their behavioural signature) failed: " + txt[-600:])
    return {"card": got[:12], "k": a.k, "pins": len(pins), "reach": "PASS", "families_run": "TIP", "free_gb": round(free_gb(run), 1)}


def compare_rows(csv_path: str, n_expect: int) -> tuple:
    n = sum(1 for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["token"] != "average") if os.path.exists(csv_path) else 0
    return n == n_expect, n


def fam_json_ok(path: str) -> tuple:
    """a families_run --per-window JSON must carry top-level tokens and logs (the TF-N4 A1 failure); cheap structural check, no stage runs on a JSON without them"""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return False, f"unreadable: {e}"
    ok = isinstance(d.get("tokens"), list) and isinstance(d.get("logs"), list) and len(d["tokens"]) == len(d["logs"]) > 0
    return ok, "ok" if ok else "no top-level tokens/logs (an older families_run.py?)"


class Runner:
    def __init__(self, a):
        self.a, self.run, self.root = a, a.run_dir, os.path.join(a.run_dir, "n5run")
        self.mk = os.path.join(self.root, "markers")
        if not a.dry_run:
            os.makedirs(self.mk, exist_ok=True)

    def p(self, *x):
        return os.path.join(self.root, *x)

    def done(self, st):
        return os.path.exists(os.path.join(self.mk, f"{st}.done"))

    def mark(self, st, ok, info, fatal=True):
        if self.a.dry_run:
            log(f"DRY-RUN stage {st} would be marked")
            return
        fp = os.path.join(self.mk, f"{st}.{'done' if ok else 'FAIL'}")
        json.dump(dict(info, stage=st, ok=ok, time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(fp, "w", encoding="utf-8"), indent=1)
        log(f"STAGE {st} {'DONE' if ok else 'FAIL'} {json.dumps(info)[:400]}")
        if not ok and fatal:
            log("ZZN5_CHAIN STOP")
            sys.exit(1)

    def wait_mem(self):
        while not self.a.dry_run:
            fv = free_virt_mb()
            if fv > self.a.min_free_virt_mb:
                return fv
            log(f"waiting: FreeVirtualMemory {fv} MB <= {self.a.min_free_virt_mb} MB")
            time.sleep(60)
        return 0

    def wait_disk(self):
        while not self.a.dry_run and free_gb(self.run) < self.a.min_free_gb:
            log(f"waiting: {self.run} has {free_gb(self.run):.1f} GB free < {self.a.min_free_gb}")
            time.sleep(120)

    def call(self, argv, logpath, cwd=None):
        if self.a.dry_run:
            log("DRY-RUN CMD " + " ".join(argv))
            return 0
        self.wait_disk()
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
                self.wait_disk()
                fv = self.wait_mem()
                log(f"start job {i} (free virt {fv} MB): {' '.join(argv)[:300]}")
                live[i] = subprocess.Popen(argv, stdout=open(lp, "a", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env(self.run), cwd=cwd or self.run)
                time.sleep(self.a.stagger_s)
            for i, pr in list(live.items()):
                if pr.poll() is not None:
                    rcs[i] = pr.returncode
                    log(f"job {i} rc {pr.returncode}")
                    del live[i]
            time.sleep(self.a.poll_s)
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


def families_stage(R: Runner, a) -> None:
    """transp rows -> famprep -> families_run --per-window (checked) -> fam_ci -> famread. Each sub-step has a marker and a structural check, so the TF-N4 A1 failure
    (an old families_run.py: no tokens/logs, found only at fam_ci, after 35 minutes) is found at the first families_run JSON, and a late failure re-runs only itself."""
    run, procs = a.run_dir, a.procs
    inp = lambda *x: os.path.join(run, "inputs", *x)                            # noqa: E731
    sub = lambda n: R.done("families_" + n)                                      # noqa: E731
    PYS = os.path.join(run, "w", "pk", "eval", "instruments")
    if not sub("rows"):
        ts = os.path.join(PYS, "transp_score.py")
        plans = [f"N4D={inp('n4d', 'n4_plans_N4D.npz')}", f"N5A={R.p('build', 'out', 'n5_plans_N5A.npz')}"]
        jobs = [([PY, ts, "--refe", os.path.join(run, "w", "pk", "refe"), "--split", "navtest", "--token-log", TOKEN_LOG, "--metric-cache-root", CACHE_ROOT, "--plans"] + plans +
                 ["--human-arm", "--out", R.p("score", f"rows_s{i}of{procs}.jsonl"), "--shard", f"{i}/{procs}"], R.p("score", f"transp_s{i}.log")) for i in range(procs)]
        rcs = R.parallel(jobs)
        rc = R.call([PY, os.path.join(run, "n2", "tfn2_nt_analyze.py"), "famprep", "--rows", R.p("score", "rows_s*of*.jsonl"), "--out", R.p("score", "rows_primary.jsonl"),
                     "--known36", inp("frame_control_navtest_full.json")], R.p("score", "famprep.log"))
        n = sum(1 for _ in open(R.p("score", "rows_primary.jsonl"), encoding="utf-8")) if not a.dry_run and os.path.exists(R.p("score", "rows_primary.jsonl")) else 0
        R.mark("families_rows", not any(rcs) and rc == 0 and (a.dry_run or n == N_PRIMARY), {"transp_rcs": rcs, "famprep_rc": rc, "primary_rows": n, "want": N_PRIMARY})
    if not sub("run"):
        fr = os.path.join(PYS, "families_run.py")
        fj = [([PY, fr, "--rows", R.p("score", "rows_primary.jsonl"), "--arm", arm, "--label", f"TF-N5 {arm}", "--out", R.p("fam", f"fam_{arm}.json"), "--per-window",
                "--n-boot", "2000", "--tanitad-repo", os.path.join(run, "repo")], R.p("fam", f"fam_{arm}.log")) for arm in ("N4D", "N5A")]
        frc = R.parallel(fj)
        chk = {arm: (fam_json_ok(R.p("fam", f"fam_{arm}.json")) if not a.dry_run else (True, "dry")) for arm in ("N4D", "N5A")}
        R.mark("families_run", not any(frc) and all(v[0] for v in chk.values()), {"fam_rcs": frc, "tokens_logs_check": {k: v[1] for k, v in chk.items()}})
    if not sub("ci"):
        rc2 = R.call([PY, os.path.join(run, "n4", "fam_ci.py"), "--a", R.p("fam", "fam_N4D.json"), "--b", R.p("fam", "fam_N5A.json"), "--out", R.p("out", "fam_ci_N5A_vs_N4D.json")],
                     R.p("out", "fam_ci.log"))
        R.mark("families_ci", rc2 == 0, {"fam_ci_rc": rc2})
    rc3 = R.call([PY, os.path.join(run, "n5", "n5_nt_analyze.py"), "famread", "--k", str(a.k), "--fam-ci", R.p("out", "fam_ci_N5A_vs_N4D.json"), "--out", R.p("out", "n5_families.json")],
                 R.p("out", "famread.log"))
    R.mark("families", rc3 == 0, {"famread_rc": rc3})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--card-sha256", required=True)
    ap.add_argument("--pins-sha256", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--k", type=int, default=47)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--margin", type=float, default=CARD["margin"])
    ap.add_argument("--iters", type=int, default=CARD["iters"])
    ap.add_argument("--cert", default=CARD["cert"])
    ap.add_argument("--centre-m", type=float, default=CARD["centre_m"])
    ap.add_argument("--fallback-margin", type=float, default=CARD["fallback_margin"])
    ap.add_argument("--damp", type=float, default=CARD["damp"])
    ap.add_argument("--damp-cum", action="store_true")
    ap.add_argument("--sign", type=float, default=CARD["sign"])
    ap.add_argument("--sign-n5x", type=float, default=CARD["sign_n5x"])
    ap.add_argument("--min-free-virt-mb", type=int, default=10240)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stagger-s", type=int, default=20)
    ap.add_argument("--poll-s", type=int, default=15)
    ap.add_argument("--stop-after", default="", help="reach_real | derive | build | unch | score_A_X | read | families")
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
        print("ZZN5 GUARDS PASS", flush=True)
        return 0
    R = Runner(a)
    run, procs = a.run_dir, a.procs
    stop = lambda st: a.stop_after == st                                        # noqa: E731
    if not a.dry_run:
        for d in ("build", "build/out", "logs", "out", "score", "fam"):
            os.makedirs(R.p(*d.split("/")), exist_ok=True)
    B = os.path.join(run, "n5", "n5_nt_build.py")
    inp = lambda *x: os.path.join(run, "inputs", *x)                            # noqa: E731
    reach_json = R.p("out", "reach_real.json")
    if not R.done("reach_real"):
        argv = [PY, B, "reach"] + l1_flags(a) + ["--real", "--fixture", os.path.join(run, "n5", "fixture_real_instances.json"), "--out", reach_json]
        rc = R.call(argv, R.p("out", "reach_real.log"))
        R.mark("reach_real", rc == 0, {"rc": rc})
    if stop("reach_real"):
        return 0
    if not R.done("derive"):
        jobs = [(build_argv(a, R.root, i, procs), R.p("build", f"derive_s{i}.log")) for i in range(procs)]
        rcs = R.parallel(jobs)
        R.mark("derive", not any(rcs), {"rcs": rcs})
    if stop("derive"):
        return 0
    common = ["--n2r-plans", inp("n2r", "n2_plans_N2R.npz"), "--n4d-plans", inp("n4d", "n4_plans_N4D.npz"), "--n4d-seam", inp("n4d", "refe_tfn4_n4d.npz")]
    if not R.done("build"):
        mg = [PY, B, "merge"] + common + ["--n4d-report", inp("n4d", "n4_build_report.json"), "--rows", R.p("build", "derive_s*of*.jsonl"),
              "--harness-csv-n2r", inp("n2r", "refe_tfn2_n2r.csv"), "--residual", inp("n5", "residual25.json"), "--reach-json", reach_json,
              "--a1-base-rows", inp("n4d", "derive_s*of2.jsonl"), "--out-dir", R.p("build", "out")]
        rc = R.call(mg, R.p("build", "merge.log"))
        rp = R.p("build", "out", "n5_build_report.json")
        rep = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
        R.mark("build", rc == 0 and (rep.get("verdict") == "PASS" or a.dry_run),
               {"merge_rc": rc, "controls": {k: v.get("pass") for k, v in rep.get("controls", {}).items()}, "dac0": rep.get("dac0_tokens_builder"),
                "repaired": {k: len(v) for k, v in rep.get("repaired", {}).items()}, "l1_run": len(rep.get("l1_run", []))})
    if stop("build"):
        return 0
    if not R.done("unch"):
        rc = R.call([PY, B, "unch", "--out-dir", R.p("build", "out")] + common + ["--out", R.p("out", "unch.json")], R.p("out", "unch.log"))
        R.mark("unch", rc == 0, {"rc": rc})
    if stop("unch"):
        return 0
    if not R.done("score_A_X"):
        pend = R.score_stage(["A", "X"])
        R.mark("score_A_X", not pend, {"pending": pend, **(R.score_ok("A")[1] if not a.dry_run else {})})
    if stop("score_A_X"):
        return 0
    if not R.done("read"):
        an = [PY, os.path.join(run, "n5", "n5_nt_analyze.py"), "read", "--harness-csv", f"N2R={inp('n2r', 'refe_tfn2_n2r.csv')}", f"N4D={inp('n4d', 'refe_tfn4_n4d.csv')}",
              f"N5A={R.csv_of('A')}", f"N5X={R.csv_of('X')}", "--build-report", R.p("build", "out", "n5_build_report.json"), "--unch", R.p("out", "unch.json"),
              "--pins", os.path.join(run, PINS_REL), "--pin-root", run, "--proxy-eval", os.path.join(run, "w", "pk", "eval"), "--prereg-sha256", a.card_sha256,
              "--token-log", TOKEN_LOG, "--known36", inp("frame_control_navtest_full.json"), "--residual", inp("n5", "residual25.json"), "--k", str(a.k),
              "--a1-analyze", os.path.join(run, "n2", "tf_n1_analyze_a1.py"), "--out", R.p("out", "n5_result.json")]
        rc = R.call(an, R.p("out", "read.log"))
        R.mark("read", rc == 0, {"rc": rc, "log": R.p("out", "read.log")})
    if stop("read"):
        return 0
    if not R.done("families"):
        families_stage(R, a)
    log("ZZN5_CHAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
