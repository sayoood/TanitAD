#!/usr/bin/env python3
"""TF-N7 navtest run chain (PREREG_TF_N7.md sections 3 to 9): guards -> reachability (card tables, the three selectors, every tier on a real navtrain token) -> builder tracked
scoring of the banked N5-A plan on all 12,146 tokens (C-DACMAP6) with the candidate generator (every lever of the cascade, run twice; the live TF-N6 selection) on the residual ->
plan sets in the N5-A seam format (C-UNCH, C-SELECT, C-CERT-B7, C-ORDER, C-X0, C-N6SAME) -> FULL-seam harness scoring of N7-T3 (PRIMARY) and N7-T (SECONDARY) (N7-X is proven
bit-identical to N5-A and is not scored) -> read -> families. CPU only, at most 2 processes, detached, marker-based, resumable. Adapted from n6_nt_chain.py (TF-N6); its helpers
(sha256, cache listing, disk / memory probes, the families JSON check) come from the PINNED copies in n5/, never copied. NO human fallback anywhere.

REFUSES TO START unless ALL hold (each is tested in test_n7_chain.py):
  1. sha256(--card) == --card-sha256
  2. the K ledger holds "TF-N7:<that sha256>:E3" with n == --k (50), the TF-N6 entry (n 49) is present, and no entry has n > --k
  3. sha256(N7_PINS.json) == --pins-sha256, every pinned file under --run-dir matches, every REQUIRED file is pinned, no unpinned .py/.json/.yaml file exists under w/, repo/, n2/, n4/,
     n5/, n6/ or n7/, the metric-cache listing digest equals the pinned one, AND the pinned `families_run.py` is the TIP version (sha256 44c62807...)
  4. --procs <= 2; the build argv names no --mutate / --smoke-allow-mutation
  5. `n7_nt_build.py reach`: the card tables (LEVERS, TIERS, certificate, grids) read back, the three selectors read back and have the card's behavioural signature, REPAIR-D and L1 of
     the live TF-N6 re-run read back, no flag has a disabling default
  6. the run drive has >= --min-free-gb free (20 GB)

Stages (markers <root>/markers/<stage>.done | .FAIL; a FAIL stops the chain; ZZN7_CHAIN STOP):
  reach_real  n7_nt_build.py reach --real (every tier T0..T4 executes on a real navtrain token; a tier list without the tier no longer chooses at it)
  derive      n7_nt_build.py derive, 2 shards
  build       n7_nt_build.py merge: C-COVB, C-DACMAP6, C-DERIV, C-LABELEGO, C-SET, C-REPLAY, C-DET, C-SELECT, C-CERT-B7, C-ORDER, C-X0, C-SEAMFN, C-UNCH, C-N6SAME, C-REACH; writes plans + seams
  unch        C-UNCH again on the files on disk, just before the harness reads them
  score_T3_T  the N7-T3 and N7-T seams, FULL 12,146 tokens, through score_p4_a1.py -> eval/harness/score_e.py (the pinned W3 harness); PASS + full row count
  read        n7_nt_analyze.py read (controls first; exits before any N7 aggregate if a control fails)
  families    transp_score.py (N5A, N7T3, N7T) -> famprep -> CHECK the rows -> families_run.py --per-window (N5A, N7T3, N7T) -> CHECK `tokens`/`logs` in the JSONs ->
              fam_ci.py paired (A = N5A, B = N7T3 and B = N7T) -> famread (guard on the primary, the secondary beside it, the changed tokens' speed MAE per token)
`--dry-run` prints every command and writes nothing; `--check-only` runs the guards and exits."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "n5"))
import n5_nt_chain as C5  # noqa: E402  (the PINNED TF-N5 chain: constants and helpers)

PY, CACHE_ROOT, TOKEN_LOG, N_ALL, N_PRIMARY, DEVKIT = C5.PY, C5.CACHE_ROOT, C5.TOKEN_LOG, C5.N_ALL, C5.N_PRIMARY, C5.DEVKIT
TIP_FAMILIES, TIP_FAMILIES_SHA256 = C5.TIP_FAMILIES, C5.TIP_FAMILIES_SHA256
sha256, log, free_gb, free_virt_mb, fam_json_ok, compare_rows, cache_listing_sha, env = (C5.sha256, C5.log, C5.free_gb, C5.free_virt_mb, C5.fam_json_ok, C5.compare_rows,
                                                                                         C5.cache_listing_sha, C5.env)
SEAMS = {"T3": "refe_tfn7_n7t3", "T": "refe_tfn7_n7t"}
K_CARD, K_PREV = 50, 49
REQUIRED = ("w/pk/eval/harness/HARNESS_PINS.json", "w/pk/eval/harness/score_e.py", "w/pk/eval/score_p4_a1.py", "w/pk/eval/run_p4.py",
            "w/pk/eval/instruments/transp_score.py", TIP_FAMILIES, "w/pk/eval/proxy_eval.py", "w/p5r/p5r_labels.py",
            "w/pk/refe/navsim_dac.py", "w/pk/eval/p5_target_labels.py",
            "n2/tf_n1_analyze_a1.py", "n2/tfn2_nt_analyze.py", "n4/n4_geom.py", "n4/n4_repair.py", "n4/n4_nt_build.py", "n4/n4_common.py", "n4/fam_ci.py",
            "n5/n4_repair_l1.py", "n5/n5_env.py", "n5/test_n4_repair_l1.py", "n5/fixture_real_instances.json", "n5/n5_nt_chain.py", "n5/n5_nt_build.py", "n5/n5_nt_analyze.py",
            "n6/n6_nt_chain.py", "n6/n6_nt_build.py", "n6/n6_nt_analyze.py", "n6/n6_make_pins.py", "n6/test_n6_chain.py",
            "n7/n7_nt_chain.py", "n7/n7_nt_build.py", "n7/n7_nt_analyze.py", "n7/n7_make_pins.py", "n7/test_n7_chain.py", "n7/n7_lib.py", "n7/n7_select.py", "n7/n7_analyze.py",
            "n7/fixture_navtrain_tiers.json",
            "inputs/refe_p4_teacher_navtest.csv", "inputs/refe_p4_human_navtest.csv", "inputs/frame_control_navtest_full.json",
            "inputs/n2r/n2_plans_N2R.npz", "inputs/n2r/refe_tfn2_n2r.csv", "inputs/n2r/rolls_RJ5/rows_s0of2.jsonl", "inputs/n2r/rolls_RJ5/rows_s1of2.jsonl",
            "inputs/n4d/n4_plans_N4D.npz", "inputs/n4d/refe_tfn4_n4d.csv",
            "inputs/n5a/n5_plans_N5A.npz", "inputs/n5a/refe_tfn5_n5a.npz", "inputs/n5a/refe_tfn5_n5a.csv", "inputs/n5a/n5_build_report.json", "inputs/n5a/replay_expect_navtest.json",
            "inputs/n6t/n6_plans_N6T.npz", "inputs/n6t/refe_tfn6_n6t.csv", "inputs/n6t/n6_build_report.json",
            TOKEN_LOG)
WATCH_EXT = (".py", ".json", ".yaml", ".yml")
WATCH_TOP = ("w", "repo", "n2", "n4", "n5", "n6", "n7")
PINS_REL = "n7/N7_PINS.json"


def build_argv(a, root: str, shard: int, procs: int) -> list:
    run = a.run_dir
    return [PY, os.path.join(run, "n7", "n7_nt_build.py"), "derive", "--route", "cache", "--n5a-plans", os.path.join(run, "inputs", "n5a", "n5_plans_N5A.npz"),
            "--n2r-plans", os.path.join(run, "inputs", "n2r", "n2_plans_N2R.npz"),
            "--rolls", os.path.join(run, "inputs", "n2r", "rolls_RJ5", "rows_s*of2.jsonl"), "--metric-cache-root", CACHE_ROOT,
            "--out", os.path.join(root, "build", f"derive_s{shard}of{procs}.jsonl"), "--shard", f"{shard}/{procs}"]


def run_reach(a, real: bool = False, out: str = "") -> tuple:
    run = a.run_dir
    argv = [PY, os.path.join(run, "n7", "n7_nt_build.py"), "reach"]
    if real:
        argv += ["--real", "--fixture", os.path.join(run, "n7", "fixture_navtrain_tiers.json"), "--out", out]
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, env=env(run), cwd=run)
    return r.returncode, r.stdout + r.stderr


def guards(a) -> dict:
    """raise SystemExit('REFUSING: ...') on the first failed guard"""
    if a.procs > 2 or a.procs < 1:
        raise SystemExit("REFUSING: --procs must be 1 or 2 (the card, section 5)")
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
    key = f"TF-N7:{a.card_sha256}:E3"
    mine = [r for r in reads if r.get("key") == key]
    if not mine:
        raise SystemExit(f"REFUSING: the K ledger has no entry {key} (freeze, then ledger, then run)")
    if mine[0].get("n") != a.k or a.k != K_CARD:
        raise SystemExit(f"REFUSING: ledger entry {key} has n = {mine[0].get('n')}, --k is {a.k}, the card says K = {K_CARD}")
    if not any(str(r.get("key", "")).startswith("TF-N6:") and r.get("n") == K_PREV for r in reads):
        raise SystemExit(f"REFUSING: the TF-N6 entry (n = {K_PREV}) is not in the ledger: the K count is not the one the card states")
    top = max((r.get("n", 0) for r in reads), default=0)
    if top > a.k:
        raise SystemExit(f"REFUSING: the ledger holds n = {top} > K = {a.k}: another read landed; correct the card and re-freeze")
    pp = os.path.join(run, PINS_REL)
    if not os.path.isfile(pp) or sha256(pp) != a.pins_sha256:
        raise SystemExit(f"REFUSING: N7_PINS.json missing or its sha256 != --pins-sha256 {a.pins_sha256}")
    pj = json.load(open(pp, encoding="utf-8"))
    pins = pj["files"]
    for must in REQUIRED:
        if must not in pins:
            raise SystemExit(f"REFUSING: N7_PINS.json does not pin {must}")
    if pins[TIP_FAMILIES] != TIP_FAMILIES_SHA256:
        raise SystemExit(f"REFUSING: N7_PINS.json pins {TIP_FAMILIES} at {pins[TIP_FAMILIES][:12]}..., not the TIP version {TIP_FAMILIES_SHA256[:12]}... "
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
        raise SystemExit(f"REFUSING: unpinned files under w/, repo/, n2/, n4/, n5/, n6/ or n7/: {extra[:5]}")
    want_cache = pj.get("metric_cache_listing_sha256")
    if not a.skip_cache_listing:
        got_c, n_c = cache_listing_sha(CACHE_ROOT)
        if got_c != want_cache or n_c < N_ALL:
            raise SystemExit(f"REFUSING: metric-cache listing digest {got_c} ({n_c} files) != pinned {want_cache}")
    if free_gb(run) < a.min_free_gb:
        raise SystemExit(f"REFUSING: {run} has {free_gb(run):.1f} GB free < --min-free-gb {a.min_free_gb}")
    rc, txt = run_reach(a)
    if rc != 0:
        raise SystemExit("REFUSING: reachability (card tables, selectors and repair objects read back and behave as the card) failed: " + txt[-600:])
    return {"card": got[:12], "k": a.k, "pins": len(pins), "reach": "PASS", "families_run": "TIP", "free_gb": round(free_gb(run), 1)}


class Runner:
    def __init__(self, a):
        self.a, self.run, self.root = a, a.run_dir, os.path.join(a.run_dir, "n7run")
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
            log("ZZN7_CHAIN STOP")
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
    """transp rows -> famprep -> families_run --per-window (checked) -> fam_ci (N7T3 and N7T against N5A) -> famread. Each sub-step has a marker and a structural check."""
    run, procs = a.run_dir, a.procs
    inp = lambda *x: os.path.join(run, "inputs", *x)                            # noqa: E731
    sub = lambda n: R.done("families_" + n)                                      # noqa: E731
    PYS = os.path.join(run, "w", "pk", "eval", "instruments")
    FARMS = ("N5A", "N7T3", "N7T")
    if not sub("rows"):
        ts = os.path.join(PYS, "transp_score.py")
        plans = [f"N5A={inp('n5a', 'n5_plans_N5A.npz')}", f"N7T3={R.p('build', 'out', 'n7_plans_N7T3.npz')}", f"N7T={R.p('build', 'out', 'n7_plans_N7T.npz')}"]
        jobs = [([PY, ts, "--refe", os.path.join(run, "w", "pk", "refe"), "--split", "navtest", "--token-log", TOKEN_LOG, "--metric-cache-root", CACHE_ROOT, "--plans"] + plans +
                 ["--human-arm", "--out", R.p("score", f"rows_s{i}of{procs}.jsonl"), "--shard", f"{i}/{procs}"], R.p("score", f"transp_s{i}.log")) for i in range(procs)]
        rcs = R.parallel(jobs)
        rc = R.call([PY, os.path.join(run, "n2", "tfn2_nt_analyze.py"), "famprep", "--rows", R.p("score", "rows_s*of*.jsonl"), "--out", R.p("score", "rows_primary.jsonl"),
                     "--known36", inp("frame_control_navtest_full.json")], R.p("score", "famprep.log"))
        n = sum(1 for _ in open(R.p("score", "rows_primary.jsonl"), encoding="utf-8")) if not a.dry_run and os.path.exists(R.p("score", "rows_primary.jsonl")) else 0
        R.mark("families_rows", not any(rcs) and rc == 0 and (a.dry_run or n == N_PRIMARY), {"transp_rcs": rcs, "famprep_rc": rc, "primary_rows": n, "want": N_PRIMARY})
    if not sub("run"):
        fr = os.path.join(PYS, "families_run.py")
        fj = [([PY, fr, "--rows", R.p("score", "rows_primary.jsonl"), "--arm", arm, "--label", f"TF-N7 {arm}", "--out", R.p("fam", f"fam_{arm}.json"), "--per-window",
                "--n-boot", "2000", "--tanitad-repo", os.path.join(run, "repo")], R.p("fam", f"fam_{arm}.log")) for arm in FARMS]
        frc = R.parallel(fj)
        chk = {arm: (fam_json_ok(R.p("fam", f"fam_{arm}.json")) if not a.dry_run else (True, "dry")) for arm in FARMS}
        R.mark("families_run", not any(frc) and all(v[0] for v in chk.values()), {"fam_rcs": frc, "tokens_logs_check": {k: v[1] for k, v in chk.items()}})
    if not sub("ci"):
        rcs2 = [R.call([PY, os.path.join(run, "n4", "fam_ci.py"), "--a", R.p("fam", "fam_N5A.json"), "--b", R.p("fam", f"fam_{arm}.json"), "--out", R.p("out", f"fam_ci_{arm}_vs_N5A.json")],
                       R.p("out", f"fam_ci_{arm}.log")) for arm in ("N7T3", "N7T")]
        R.mark("families_ci", not any(rcs2), {"fam_ci_rcs": rcs2})
    rc3 = R.call([PY, os.path.join(run, "n7", "n7_nt_analyze.py"), "famread", "--k", str(a.k), "--fam-ci-t3", R.p("out", "fam_ci_N7T3_vs_N5A.json"),
                  "--fam-ci-t", R.p("out", "fam_ci_N7T_vs_N5A.json"), "--fam-a", R.p("fam", "fam_N5A.json"), "--fam-t3", R.p("fam", "fam_N7T3.json"), "--fam-t", R.p("fam", "fam_N7T.json"),
                  "--build-report", R.p("build", "out", "n7_build_report.json"), "--out", R.p("out", "n7_families.json")], R.p("out", "famread.log"))
    R.mark("families", rc3 == 0, {"famread_rc": rc3})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--card-sha256", required=True)
    ap.add_argument("--pins-sha256", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--k", type=int, default=K_CARD)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--min-free-virt-mb", type=int, default=10240)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stagger-s", type=int, default=20)
    ap.add_argument("--poll-s", type=int, default=15)
    ap.add_argument("--stop-after", default="", help="reach_real | derive | build | unch | score_T3_T | read | families")
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
        print("ZZN7 GUARDS PASS", flush=True)
        return 0
    R = Runner(a)
    run, procs = a.run_dir, a.procs
    stop = lambda st: a.stop_after == st                                        # noqa: E731
    if not a.dry_run:
        for d in ("build", "build/out", "logs", "out", "score", "fam"):
            os.makedirs(R.p(*d.split("/")), exist_ok=True)
    B = os.path.join(run, "n7", "n7_nt_build.py")
    inp = lambda *x: os.path.join(run, "inputs", *x)                            # noqa: E731
    reach_json = R.p("out", "reach_real.json")
    if not R.done("reach_real"):
        rc = R.call([PY, B, "reach", "--real", "--fixture", os.path.join(run, "n7", "fixture_navtrain_tiers.json"), "--out", reach_json], R.p("out", "reach_real.log"))
        R.mark("reach_real", rc == 0, {"rc": rc})
    if stop("reach_real"):
        return 0
    if not R.done("derive"):
        jobs = [(build_argv(a, R.root, i, procs), R.p("build", f"derive_s{i}.log")) for i in range(procs)]
        rcs = R.parallel(jobs)
        R.mark("derive", not any(rcs), {"rcs": rcs})
    if stop("derive"):
        return 0
    common = ["--n5a-plans", inp("n5a", "n5_plans_N5A.npz"), "--n5a-seam", inp("n5a", "refe_tfn5_n5a.npz")]
    if not R.done("build"):
        mg = [PY, B, "merge"] + common + ["--n2r-plans", inp("n2r", "n2_plans_N2R.npz"), "--n5a-report", inp("n5a", "n5_build_report.json"),
              "--replay-expect-json", inp("n5a", "replay_expect_navtest.json"), "--rows", R.p("build", "derive_s*of*.jsonl"),
              "--harness-csv-n5a", inp("n5a", "refe_tfn5_n5a.csv"), "--reach-json", reach_json, "--n6-report", inp("n6t", "n6_build_report.json"),
              "--n6t-plans", inp("n6t", "n6_plans_N6T.npz"), "--out-dir", R.p("build", "out")]
        rc = R.call(mg, R.p("build", "merge.log"))
        rp = R.p("build", "out", "n7_build_report.json")
        rep = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
        R.mark("build", rc == 0 and (rep.get("verdict") == "PASS" or a.dry_run),
               {"merge_rc": rc, "controls": {k: v.get("pass") for k, v in rep.get("controls", {}).items()}, "residual": len(rep.get("residual", [])),
                "applied": {k: len(v) for k, v in rep.get("repaired", {}).items()}, "counts": rep.get("counts")})
    if stop("build"):
        return 0
    if not R.done("unch"):
        rc = R.call([PY, B, "unch", "--out-dir", R.p("build", "out")] + common + ["--out", R.p("out", "unch.json")], R.p("out", "unch.log"))
        R.mark("unch", rc == 0, {"rc": rc})
    if stop("unch"):
        return 0
    if not R.done("score_T3_T"):
        pend = R.score_stage(["T3", "T"])
        R.mark("score_T3_T", not pend, {"pending": pend, **(R.score_ok("T3")[1] if not a.dry_run else {})})
    if stop("score_T3_T"):
        return 0
    if not R.done("read"):
        an = [PY, os.path.join(run, "n7", "n7_nt_analyze.py"), "read", "--harness-csv", f"N2R={inp('n2r', 'refe_tfn2_n2r.csv')}", f"N4D={inp('n4d', 'refe_tfn4_n4d.csv')}",
              f"N5A={inp('n5a', 'refe_tfn5_n5a.csv')}", f"N6T={inp('n6t', 'refe_tfn6_n6t.csv')}", f"N7T3={R.csv_of('T3')}", f"N7T={R.csv_of('T')}",
              "--build-report", R.p("build", "out", "n7_build_report.json"),
              "--unch", R.p("out", "unch.json"), "--pins", os.path.join(run, PINS_REL), "--pin-root", run, "--proxy-eval", os.path.join(run, "w", "pk", "eval"),
              "--prereg-sha256", a.card_sha256, "--token-log", TOKEN_LOG, "--known36", inp("frame_control_navtest_full.json"), "--k", str(a.k),
              "--a1-analyze", os.path.join(run, "n2", "tf_n1_analyze_a1.py"), "--out", R.p("out", "n7_result.json")]
        rc = R.call(an, R.p("out", "read.log"))
        R.mark("read", rc == 0, {"rc": rc, "log": R.p("out", "read.log")})
    if stop("read"):
        return 0
    if not R.done("families"):
        families_stage(R, a)
    log("ZZN7_CHAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
