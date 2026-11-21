#!/usr/bin/env python3
"""TF-N2 navtest run chain (PREREG_TF_N2.md): guards -> R0 identity subset -> RJ5 navtest roll -> build (P5-R / Tc20s / Tw / N1-C plans) -> harness scoring
-> read -> families. CPU only, at most 2 processes, resumable (markers + rows already written are skipped). Modelled on tf_n1_a1_chain.py.

REFUSES TO START unless ALL hold (each is tested in test_tfn2_nt_chain.py):
  1. sha256(--card) == --card-sha256
  2. the K ledger holds the key "TF-N2:<that sha256>:E3" with n == --k (43), the TF-N1-A1 entry (n 42) is present, and no entry has n > --k
  3. sha256(N2_PINS.json) == --pins-sha256, every pinned file under --run-dir matches, every REQUIRED file is pinned, and no unpinned .py/.json/.yaml file
     exists under w/, repo/ or n2/ (the harness tree included)
  4. --procs <= 2

Stages (markers <root>/markers/<stage>.done | .FAIL; a FAIL stops the chain):
  roll_R0    teacher re-roll, NO override, on the R0-identity subset (the navtest tokens with sha1(token) % 50 == 0), 2 shards
  ctl_R0     C-R0: those rows vs the BANKED rollouts, max |d| <= 1e-6 on x, y and heading. FAIL => STOP: the RJ5 roll never starts
  roll_RJ5   the same roller with positive_jerk_limit 5.0 on all 12,146 tokens, 2 shards; every ok row carries the knob read back from the built planner
  build      tfn2_nt_build.py derive (2 shards) + merge: N2R (P5-R), N2C (Tc20s), N2W (Tw) and N1C (Tc20s of the banked rollout) as W3 seams + plan files
  score_T    the REGISTERED T seam through score_p4_a1.py -> eval/harness/score_e.py (the pinned W3 harness)
  cidh       C-ID-H: that CSV vs the banked teacher CSV, |d| <= 1e-6, ALL 12,146 tokens. FAIL => STOP, no N2 arm is ever scored
  cidc       C-ID-C: the re-derived N1-C plans on a 200-token subset, harness, vs the banked N1-C CSV, |d| <= 1e-6. FAIL => STOP
  score_N2   N2R and N2C together through the harness, then N2W
  read       tfn2_nt_analyze.py read (controls first)
  families   transp_score.py (N2R, N2C, N1C, human) -> famprep -> families_run.py x4 -> famread (the longitudinal guard)
Launch: Win32_Process Create, CreateFlags 0x08000200, `cmd /c "python tfn2_nt_chain.py ... >> chain.log 2>&1 < NUL"` (launch_tfn2_nt.ps1 prints it).
`--smoke <dir>`: a navtrain smoke (profile.json in <dir>): the guards run for real, the roll / build / families stages run on the smoke tokens under
<dir>/out, and the harness stages (navtest metric cache only) are skipped. NEVER a navtest token.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
TOKEN_LOG = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
MAPS = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
N_ALL = 12146
N_IDC = 200
R0_MOD = 50
ID_TOL = 1e-6
R0_TOL = 1e-6
JERK = {"R0": 1.0, "RJ5": 5.0}
FAM_ARMS = ("N2R", "N2C", "N1C", "human")
MET = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ddc": "driving_direction_compliance",
       "ep": "ego_progress", "ttc": "time_to_collision_within_bound", "c": "comfort", "pdms": "score"}
# stage label, seam path relative to the root's build/out (or the run dir), tokens-file (None = all)
SEAMS = {"T": ("refe_tfn2_t", "run:inputs/refe_p4_teacher_navtest.npz", None),
         "R": ("refe_tfn2_n2r", "root:build/out/refe_tfn2_n2r.npz", None),
         "C": ("refe_tfn2_n2c", "root:build/out/refe_tfn2_n2c.npz", None),
         "W": ("refe_tfn2_n2w", "root:build/out/refe_tfn2_n2w.npz", None),
         "I": ("refe_tfn2_idc", "root:build/out/refe_tfn2_n1c_sub200.npz", "root:build/out/cidc_tokens.json")}
REQUIRED = ("w/pk/eval/harness/HARNESS_PINS.json", "w/pk/eval/harness/score_e.py", "w/pk/eval/score_p4_a1.py", "w/pk/eval/run_p4.py",
            "w/pk/eval/teacher_navtest_seam.py", "w/pk/eval/tfn2_nt_roll.py", "w/pk/refe/build_teacher_rollouts.py", "w/p5r/p5r_labels.py",
            "w/pk/eval/instruments/transp_score.py", "w/pk/eval/instruments/families_run.py", "w/pk/eval/proxy_eval.py",
            "n2/tfn2_nt_build.py", "n2/tfn2_nt_analyze.py", "n2/tf_n1_analyze_a1.py", "n2/tfn2_nt_chain.py",
            "inputs/refe_p4_teacher_navtest.npz", "inputs/refe_p4_teacher_navtest.csv", "inputs/refe_p4_human_navtest.csv",
            "inputs/frame_control_navtest_full.json", "inputs/n1_csv/refe_a1_tfn1_c.csv",
            "inputs/rollouts/rows_s0of4.jsonl.xz", "inputs/rollouts/rows_s1of4.jsonl.xz", "inputs/rollouts/rows_s2of4.jsonl.xz",
            "inputs/rollouts/rows_s3of4.jsonl.xz", TOKEN_LOG)
WATCH_EXT = (".py", ".json", ".yaml", ".yml")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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
              "NUPLAN_MAPS_ROOT": MAPS, "PYTHONIOENCODING": "utf-8",
              "PYTHONPATH": ";".join([DEVKIT.replace("/", "\\"), os.path.join(run, "w", "pk", "refe").replace("/", "\\")])})
    return e


def r0_subset(tokens) -> list:
    """the R0-identity subset, fixed in advance: sha1(token) % 50 == 0"""
    return sorted(t for t in tokens if int(hashlib.sha1(t.encode()).hexdigest(), 16) % R0_MOD == 0)


# ------------------------------------------------------------------------------------------------------------- guards
def guards(a) -> dict:
    """raise SystemExit('REFUSING: ...') on the first failed guard; return facts for the log"""
    if a.procs > 2:
        raise SystemExit("REFUSING: at most 2 processes (the card, section 11)")
    if a.procs < 1:
        raise SystemExit("REFUSING: --procs must be 1 or 2")
    run = a.run_dir
    if not os.path.isfile(a.card):
        raise SystemExit(f"REFUSING: card {a.card} is missing")
    got = sha256(a.card)
    if got != a.card_sha256:
        raise SystemExit(f"REFUSING: sha256({a.card}) = {got} != --card-sha256 {a.card_sha256}")
    if not os.path.isfile(a.ledger):
        raise SystemExit(f"REFUSING: ledger {a.ledger} is missing")
    reads = json.load(open(a.ledger, encoding="utf-8")).get("reads", [])
    key = f"TF-N2:{a.card_sha256}:E3"
    mine = [r for r in reads if r.get("key") == key]
    if not mine:
        raise SystemExit(f"REFUSING: the K ledger has no entry {key} (freeze, then ledger, then run)")
    if mine[0].get("n") != a.k:
        raise SystemExit(f"REFUSING: ledger entry {key} has n = {mine[0].get('n')}, the card says K = {a.k}")
    if not any(str(r.get("key", "")).startswith("TF-N1-A1:") and r.get("n") == 42 for r in reads):
        raise SystemExit("REFUSING: the TF-N1-A1 entry (n = 42) is not in the ledger: the K count is not the one the card states")
    top = max((r.get("n", 0) for r in reads), default=0)
    if top > a.k:
        raise SystemExit(f"REFUSING: the ledger holds n = {top} > K = {a.k}: another read landed; correct the card and re-freeze")
    pp = os.path.join(run, "n2", "N2_PINS.json")
    if not os.path.isfile(pp) or sha256(pp) != a.pins_sha256:
        raise SystemExit(f"REFUSING: N2_PINS.json missing or its sha256 != --pins-sha256 {a.pins_sha256}")
    pins = json.load(open(pp, encoding="utf-8"))["files"]
    for must in REQUIRED:
        if must not in pins:
            raise SystemExit(f"REFUSING: N2_PINS.json does not pin {must}")
    bad = [k for k, v in pins.items() if not os.path.exists(os.path.join(run, k)) or sha256(os.path.join(run, k)) != v]
    if bad:
        raise SystemExit(f"REFUSING: pinned files differ or are missing: {bad[:6]}")
    on_disk = set()
    for top_d in ("w", "repo", "n2"):
        for d, _, fs in os.walk(os.path.join(run, top_d)):
            if "__pycache__" in d:
                continue
            for f in fs:
                if f.endswith(WATCH_EXT):
                    on_disk.add(os.path.relpath(os.path.join(d, f), run).replace(os.sep, "/"))
    extra = sorted(on_disk - set(pins) - {"n2/N2_PINS.json"})
    if extra:
        raise SystemExit(f"REFUSING: unpinned files under w/, repo/ or n2/: {extra[:5]}")
    return {"card": got[:12], "k": a.k, "pins": len(pins)}


# ------------------------------------------------------------------------------------------------------- csv compare
def compare_csv(a_csv: str, b_csv: str, n_expect: int, subset: bool = False) -> dict:
    """C-ID-H / C-ID-C: A (re-scored) against B (banked), |d| <= 1e-6 on NC DAC DDC EP TTC C PDMS. subset=True: A is a subset of B (B's extra tokens are fine)."""
    def load(p):
        with open(p, encoding="utf-8", newline="") as fh:
            return {r["token"]: r for r in csv.DictReader(fh) if r["token"] != "average" and r.get("valid", "True") == "True"}
    A, B = load(a_csv), load(b_csv)
    miss = len(set(A) - set(B)) + (0 if subset else len(set(B) - set(A)))
    worst = {m: 0.0 for m in MET}
    over = 0
    for t in set(A) & set(B):
        d = {m: abs(float(A[t][c]) - float(B[t][c])) for m, c in MET.items()}
        for m in MET:
            worst[m] = max(worst[m], d[m])
        over += max(d.values()) > ID_TOL
    ok = miss == 0 and over == 0 and len(A) == n_expect
    return {"pass": ok, "tol": ID_TOL, "tokens_rescored": len(A), "tokens_banked": len(B), "token_set_mismatch": miss, "tokens_over_tol": over,
            "max_abs_diff_by_metric": worst}


def rows_ok(patterns) -> tuple[dict, int]:
    """{token: row} of the ok rows (traj20 present, no error), and the number of error rows"""
    ok, err = {}, 0
    for pat in patterns:
        for fp in sorted(glob.glob(pat)):
            op = gzip.open if fp.endswith(".gz") else open
            if fp.endswith(".xz"):
                import lzma
                op = lzma.open
            for ln in op(fp, "rt", encoding="utf-8"):
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r or "traj20" not in r:
                    err += 1
                else:
                    ok[r["token"]] = r
    return ok, err


def r0_identity(rolled: dict, bank: dict, tokens: list) -> dict:
    """C-R0: every subset token rolled ok with the registered knobs, and its traj20 equals the banked one to 1e-6 (x, y and heading separately)"""
    missing = [t for t in tokens if t not in rolled]
    nobank = [t for t in tokens if t not in bank]
    mx = mh = 0.0
    n_over = 0
    for t in tokens:
        if t in rolled and t in bank:
            d = abs(np.asarray(rolled[t]["traj20"], dtype=float) - np.asarray(bank[t]["traj20"], dtype=float))
            dxy, dh = float(d[:, :2].max()), float(d[:, 2].max())
            mx, mh = max(mx, dxy), max(mh, dh)
            n_over += max(dxy, dh) > R0_TOL
    knob_bad = [t for t in tokens if t in rolled and abs(rolled[t].get("knobs", {}).get("positive_jerk_limit_when_nonnegative_acc", -1) - JERK["R0"]) > 1e-9]
    ok = not missing and not nobank and n_over == 0 and not knob_bad and len(tokens) > 0
    return {"PASS": bool(ok), "n": len(tokens), "max_abs_m": mx, "max_abs_heading_rad": mh, "n_over_bar": n_over, "missing": len(missing),
            "not_in_bank": len(nobank), "wrong_knob_rows": len(knob_bad), "bar": f"max |d| <= {R0_TOL} on x, y and heading, all {len(tokens)} tokens"}


def strip_error_rows(pattern: str) -> int:
    """S.rollouts never retries a token that has ANY row (an error row included): drop error rows so a resume re-rolls them"""
    n = 0
    for fp in glob.glob(pattern):
        keep = []
        for ln in open(fp, encoding="utf-8"):
            try:
                if "error" in json.loads(ln):
                    n += 1
                    continue
            except json.JSONDecodeError:
                continue                                   # a torn last line is dropped (it is re-derived)
            keep.append(ln if ln.endswith("\n") else ln + "\n")
        with open(fp, "w", encoding="utf-8", newline="\n") as fo:
            fo.writelines(keep)
    return n


# ------------------------------------------------------------------------------------------------------------ runner
class Runner:
    def __init__(self, a, prof):
        self.a, self.run, self.prof = a, a.run_dir, prof
        self.root = prof["root"]
        self.mk = os.path.join(self.root, "markers")
        os.makedirs(self.mk, exist_ok=True)

    def p(self, *x):
        return os.path.join(self.root, *x)

    def done(self, st):
        return os.path.exists(os.path.join(self.mk, f"{st}.done"))

    def mark(self, st, ok, info):
        fp = os.path.join(self.mk, f"{st}.{'done' if ok else 'FAIL'}")
        json.dump(dict(info, stage=st, ok=ok, time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(fp, "w", encoding="utf-8"), indent=1)
        log(f"STAGE {st} {'DONE' if ok else 'FAIL'} {json.dumps(info)[:400]}")
        if not ok:
            log("ZZN2_CHAIN STOP")
            sys.exit(1)

    def wait_mem(self):
        while True:
            fv = free_virt_mb()
            if fv > self.a.min_free_virt_mb:
                return fv
            log(f"waiting: FreeVirtualMemory {fv} MB <= {self.a.min_free_virt_mb} MB")
            time.sleep(60)

    def parallel(self, jobs, cwd=None):
        """jobs = [(argv, logpath)] with at most --procs alive; returns the return codes"""
        rcs, live, queue = [None] * len(jobs), {}, list(enumerate(jobs))
        while queue or live:
            while queue and len(live) < self.a.procs:
                i, (argv, lp) = queue.pop(0)
                fv = self.wait_mem()
                log(f"start job {i} (free virt {fv} MB): {' '.join(argv)[:300]}")
                live[i] = subprocess.Popen(argv, stdout=open(lp, "a", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           env=env(self.run), cwd=cwd or self.run)
                time.sleep(20 if not self.a.smoke else 1)
            for i, pr in list(live.items()):
                if pr.poll() is not None:
                    rcs[i] = pr.returncode
                    log(f"job {i} rc {pr.returncode}")
                    del live[i]
            time.sleep(15 if not self.a.smoke else 2)
        return rcs

    # ---- harness scoring
    def seam_path(self, spec):
        kind, rel = spec.split(":", 1)
        return os.path.join(self.run if kind == "run" else self.root, rel)

    def csv_of(self, which):
        lab = SEAMS[which][0]
        return self.p("harness_score", "score", lab, f"{lab}.csv")

    def score_job(self, which):
        lab, seam, toks = SEAMS[which]
        os.makedirs(self.p("logs"), exist_ok=True)
        argv = [PY, os.path.join(self.run, "w", "pk", "eval", "score_p4_a1.py"), "--label", lab, "--seam", self.seam_path(seam),
                "--tokens", self.seam_path(toks) if toks else "all", "--log-dir", self.p("logs"), "--score-root", self.p("harness_score")]
        return argv, self.p("logs", f"{lab}_job.log")

    def score_ok(self, which):
        lab = SEAMS[which][0]
        st = os.path.join(self.p("logs"), f"{lab}_status.json")
        status = json.load(open(st, encoding="utf-8")) if os.path.exists(st) else {}
        cp = self.csv_of(which)
        n = sum(1 for r in csv.DictReader(open(cp, encoding="utf-8")) if r["token"] != "average") if os.path.exists(cp) else 0
        want = N_IDC if which == "I" else N_ALL
        return status.get("status") == "PASS" and n == want, {"status": status.get("status"), "csv_tokens": n, "want": want}

    def score_stage(self, which_list):
        pend = [w for w in which_list if not self.score_ok(w)[0]]
        for att in range(1, self.a.retries + 1):
            if not pend:
                break
            self.parallel([self.score_job(w) for w in pend])
            pend = [w for w in pend if not self.score_ok(w)[0]]
            log(f"score {which_list} attempt {att}: pending {pend}")
        return pend

    # ---- rolls
    def roll_jobs(self, arm, tokens_file):
        outd = self.p("rolls", arm)
        os.makedirs(outd, exist_ok=True)
        os.makedirs(self.p("logs"), exist_ok=True)
        jobs = []
        for i in range(self.a.procs):
            argv = [PY, os.path.join(self.run, "w", "pk", "eval", "run_p4.py"), "--log", self.p("logs", f"p4_{arm}_s{i}.log"), "--hz", "10",
                    "--threads", "1", "--script", "tfn2_nt_roll.py", "--", "--arm", arm, "--shard", f"{i}/{self.a.procs}", "--out-dir", outd]
            if tokens_file:
                argv += ["--tokens", tokens_file]
            if self.prof.get("export"):
                argv += ["--export", self.prof["export"], "--db-dir", self.prof["db_dir"]]
            jobs.append((argv, self.p("logs", f"roll_{arm}_s{i}.log")))
        return jobs, outd

    def roll_stage(self, arm, tokens_file, tokens):
        jobs, outd = self.roll_jobs(arm, tokens_file)
        stripped = strip_error_rows(os.path.join(outd, "rows_s*of*.jsonl"))
        if stripped:
            log(f"roll_{arm}: dropped {stripped} stale error rows before the (re)start")
        t0 = time.time()
        rcs = self.parallel(jobs, cwd=os.path.join(self.run, "w", "pk", "eval"))
        ok, err = rows_ok([os.path.join(outd, "rows_s*of*.jsonl")])
        miss = [t for t in tokens if t not in ok]
        badk = [t for t in tokens if t in ok and abs(ok[t].get("knobs", {}).get("positive_jerk_limit_when_nonnegative_acc", -1) - JERK[arm]) > 1e-9]
        return ok, {"rcs": rcs, "ok_rows": len(ok), "error_rows": err, "missing": len(miss), "wrong_knob_rows": len(badk), "seconds": round(time.time() - t0),
                    "ok": not any(rcs) and not miss and not badk and err == 0}


def profile(a) -> dict:
    if a.smoke:
        pf = json.load(open(os.path.join(a.smoke, "profile.json"), encoding="utf-8"))
        toks = json.load(open(pf["tokens_file"], encoding="utf-8"))["tokens"]
        return dict(pf, smoke=True, root=os.path.join(a.smoke, "out"), n_expect=len(toks), tokens=sorted(toks), r0_tokens=sorted(toks),
                    r0_file=pf["tokens_file"], all_file=pf["tokens_file"], bank_glob=pf["bank_glob"])
    exp = json.load(gzip.open(TOKEN_LOG, "rt", encoding="utf-8"))
    toks = sorted(exp["tokens"])
    root = a.run_dir
    return {"smoke": False, "root": root, "n_expect": N_ALL, "tokens": toks, "r0_tokens": r0_subset(toks), "r0_file": os.path.join(root, "inputs", "r0_subset_tokens.json"),
            "all_file": None, "bank_glob": os.path.join(a.run_dir, "inputs", "rollouts", "rows_s*of4.jsonl.xz")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--card-sha256", required=True)
    ap.add_argument("--pins-sha256", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--k", type=int, default=43)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--min-free-virt-mb", type=int, default=10240)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--stop-after", default="", help="roll_R0 | ctl_R0 | roll_RJ5 | build | score_T | cidh | cidc | score_N2 | read | families")
    ap.add_argument("--check-only", action="store_true", help="run the guards and exit (rc 0 = they pass); nothing is started")
    ap.add_argument("--smoke", default="", help="a navtrain smoke dir with profile.json (never navtest)")
    a = ap.parse_args()
    facts = guards(a)
    log(f"guards PASS: {json.dumps(facts)}")
    if a.check_only:
        print("ZZN2 GUARDS PASS", flush=True)
        return 0
    prof = profile(a)
    if not prof["smoke"] and len(prof["tokens"]) != N_ALL:
        raise SystemExit(f"REFUSING: the token export holds {len(prof['tokens'])} != {N_ALL}")
    if not prof["smoke"] and not 200 <= len(prof["r0_tokens"]) <= 300:
        raise SystemExit(f"REFUSING: the R0 subset has {len(prof['r0_tokens'])} tokens, not 200 to 300")
    R = Runner(a, prof)
    run = a.run_dir
    stop = lambda st: a.stop_after == st                                        # noqa: E731
    bank, _e = rows_ok([prof["bank_glob"]])
    log(f"banked rollouts: {len(bank)} ok rows; population {len(prof['tokens'])}; R0 subset {len(prof['r0_tokens'])}; root {R.root}")
    if not prof["smoke"] and len(bank) != N_ALL:
        raise SystemExit(f"REFUSING: {len(bank)} banked rollout rows != {N_ALL}")
    if not prof["smoke"] and not os.path.exists(prof["r0_file"]):
        json.dump({"tokens": prof["r0_tokens"], "rule": f"sha1(token) % {R0_MOD} == 0"}, open(prof["r0_file"], "w", encoding="utf-8"), indent=1)
    pop = prof["tokens"]
    # ---- R0 identity subset
    if not R.done("roll_R0"):
        ok, info = R.roll_stage("R0", prof["r0_file"], prof["r0_tokens"])
        R.mark("roll_R0", info["ok"], info)
    if stop("roll_R0"):
        return 0
    if not R.done("ctl_R0"):
        ok, _ = rows_ok([R.p("rolls", "R0", "rows_s*of*.jsonl")])
        rep = r0_identity(ok, bank, prof["r0_tokens"])
        os.makedirs(R.p("out"), exist_ok=True)
        json.dump(rep, open(R.p("out", "ctl_R0.json"), "w", encoding="utf-8"), indent=1)
        R.mark("ctl_R0", rep["PASS"], {k: rep[k] for k in ("n", "max_abs_m", "max_abs_heading_rad", "n_over_bar")})
    if stop("ctl_R0"):
        return 0
    # ---- RJ5 navtest roll
    if not R.done("roll_RJ5"):
        ok, info = R.roll_stage("RJ5", prof["all_file"], pop)
        R.mark("roll_RJ5", info["ok"], info)
    if stop("roll_RJ5"):
        return 0
    # ---- build
    if not R.done("build"):
        os.makedirs(R.p("build", "out"), exist_ok=True)
        bld = os.path.join(run, "n2", "tfn2_nt_build.py")
        common = ["--w", os.path.join(run, "w"), "--rolls", R.p("rolls", "RJ5", "rows_s*of*.jsonl"), "--bank", prof["bank_glob"]]
        if prof["smoke"]:
            common += ["--db-root", prof["db_root"]]
            if prof.get("cderiv_ref"):
                common += ["--cderiv-ref", prof["cderiv_ref"]]
        jobs = [([PY, bld, "derive"] + common + ["--out", R.p("build", f"derive_s{i}of{a.procs}.jsonl"), "--shard", f"{i}/{a.procs}"],
                 R.p("build", f"derive_s{i}.log")) for i in range(a.procs)]
        rcs = R.parallel(jobs)
        mg = [PY, bld, "merge"] + common + ["--rows", R.p("build", "derive_s*of*.jsonl"), "--out-dir", R.p("build", "out")]
        if not prof["smoke"]:
            mg += ["--seam", os.path.join(run, "inputs", "refe_p4_teacher_navtest.npz")]
        rc = subprocess.call(mg, stdout=open(R.p("build", "merge.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=env(run), cwd=run)
        rp = R.p("build", "out", "n2_build_report.json")
        rep = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
        R.mark("build", rc == 0 and not any(rcs) and rep.get("verdict") == "PASS",
               {"derive_rcs": rcs, "merge_rc": rc, "controls": {k: v.get("pass") for k, v in rep.get("controls", {}).items()}, "timing": rep.get("timing")})
    if stop("build"):
        return 0
    if not prof["smoke"]:
        banked_t = os.path.join(run, "inputs", "refe_p4_teacher_navtest.csv")
        banked_c = os.path.join(run, "inputs", "n1_csv", "refe_a1_tfn1_c.csv")
        if not R.done("score_T"):
            pend = R.score_stage(["T"])
            R.mark("score_T", not pend, {"pending": pend, **R.score_ok("T")[1]})
        if stop("score_T"):
            return 0
        if not R.done("cidh"):
            res = compare_csv(R.csv_of("T"), banked_t, N_ALL)
            json.dump(res, open(R.p("out", "cidh_early.json"), "w", encoding="utf-8"), indent=1)
            R.mark("cidh", res["pass"], {k: res[k] for k in ("tokens_over_tol", "token_set_mismatch", "tokens_rescored")})
        if stop("cidh"):
            return 0
        if not R.done("cidc"):
            pend = R.score_stage(["I"])
            res = compare_csv(R.csv_of("I"), banked_c, N_IDC, subset=True) if not pend else {"pass": False, "pending": pend}
            json.dump(res, open(R.p("out", "cidc.json"), "w", encoding="utf-8"), indent=1)
            R.mark("cidc", res["pass"], {k: res.get(k) for k in ("tokens_over_tol", "token_set_mismatch", "tokens_rescored")})
        if stop("cidc"):
            return 0
        if not R.done("score_N2"):
            pend = R.score_stage(["R", "C"])
            if not pend:
                pend = R.score_stage(["W"])
            R.mark("score_N2", not pend, {"pending": pend})
        if stop("score_N2"):
            return 0
        if not R.done("read"):
            an = [PY, os.path.join(run, "n2", "tfn2_nt_analyze.py"), "read", "--harness-csv", f"N2R={R.csv_of('R')}", f"N2C={R.csv_of('C')}",
                  f"N2W={R.csv_of('W')}", f"N1C={banked_c}", f"T={R.csv_of('T')}", "--identity-csv", banked_t,
                  "--human-csv", os.path.join(run, "inputs", "refe_p4_human_navtest.csv"),
                  "--known36", os.path.join(run, "inputs", "frame_control_navtest_full.json"),
                  "--build-report", R.p("build", "out", "n2_build_report.json"), "--ctl-r0", R.p("out", "ctl_R0.json"), "--cidc", R.p("out", "cidc.json"),
                  "--pins", os.path.join(run, "n2", "N2_PINS.json"), "--pin-root", run, "--proxy-eval", os.path.join(run, "w", "pk", "eval"),
                  "--prereg-sha256", a.card_sha256, "--token-log", TOKEN_LOG, "--out", R.p("out", "n2_result.json")]
            rc = subprocess.call(an, stdout=open(R.p("out", "read.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=env(run), cwd=run)
            R.mark("read", rc == 0, {"rc": rc, "log": R.p("out", "read.log")})
        if stop("read"):
            return 0
    # ---- families (after the verdict: the longitudinal guard is read here)
    if not R.done("families"):
        os.makedirs(R.p("score"), exist_ok=True)
        os.makedirs(R.p("fam"), exist_ok=True)
        ts = os.path.join(run, "w", "pk", "eval", "instruments", "transp_score.py")
        plans = [f"{x}={R.p('build', 'out', f'n2_plans_{x}.npz')}" for x in ("N2R", "N2C", "N1C")]
        extra = ["--split", "navtrain", "--db-root", prof["db_root"]] if prof["smoke"] else ["--split", "navtest", "--token-log", TOKEN_LOG]
        jobs = [([PY, ts, "--refe", os.path.join(run, "w", "pk", "refe")] + extra + ["--plans"] + plans +
                 ["--human-arm", "--out", R.p("score", f"rows_s{i}of{a.procs}.jsonl"), "--shard", f"{i}/{a.procs}"], R.p("score", f"transp_s{i}.log"))
                for i in range(a.procs)]
        rcs = R.parallel(jobs)
        sc, serr = {}, 0
        for fp in glob.glob(R.p("score", "rows_s*of*.jsonl")):
            for ln in open(fp, encoding="utf-8"):
                r = json.loads(ln) if ln.strip() else {}
                if "error" in r:
                    serr += 1
                elif "token" in r:
                    sc[r["token"]] = 1
        if any(rcs) or len(sc) != prof["n_expect"]:
            R.mark("families", False, {"transp_rcs": rcs, "rows": len(sc), "expect": prof["n_expect"], "error_rows": serr})
        an = [PY, os.path.join(run, "n2", "tfn2_nt_analyze.py")]
        known = os.path.join(run, "inputs", "frame_control_navtest_full.json")
        kn = ["--known36", known] if not prof["smoke"] else []
        rc = subprocess.call(an + ["famprep", "--rows", R.p("score", "rows_s*of*.jsonl"), "--out", R.p("score", "rows_primary.jsonl")] + kn,
                             stdout=open(R.p("score", "famprep.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env(run), cwd=run)
        if rc:
            R.mark("families", False, {"famprep_rc": rc})
        fr = os.path.join(run, "w", "pk", "eval", "instruments", "families_run.py")
        fj = [([PY, fr, "--rows", R.p("score", "rows_primary.jsonl"), "--arm", arm, "--label", f"TF-N2 {arm}", "--out", R.p("fam", f"fam_{arm}.json"),
                "--n-boot", "2000", "--tanitad-repo", os.path.join(run, "repo")], R.p("fam", f"fam_{arm}.log")) for arm in FAM_ARMS]
        frc = R.parallel(fj)
        miss = [x for x in FAM_ARMS if not os.path.exists(R.p("fam", f"fam_{x}.json"))]
        rc = subprocess.call(an + ["famread", "--fam"] + [f"{x}={R.p('fam', f'fam_{x}.json')}" for x in FAM_ARMS] + ["--out", R.p("out", "n2_families.json")],
                             stdout=open(R.p("out", "famread.log"), "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env(run), cwd=run)
        R.mark("families", not any(frc) and not miss and rc == 0, {"fam_rcs": frc, "missing": miss, "famread_rc": rc})
    log("ZZN2_CHAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
