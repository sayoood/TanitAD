#!/usr/bin/env python3
"""TF-N3 combo chain (detached; CPU only, <= 2 processes, 1 thread each). Navtrain held-out tokens ONLY (469 tokens / 13 logs of the TF-N2 reroll).
Derived from n2/reroll/tfn2_reroll_chain.py. ONE new arm: RJ5H5 = --pos-jerk-limit 5.0 --policy-interval-s 0.2.
Stages (markers in <run>/markers/<stage>.done|.FAIL; last = ALL.done):
  preflight     sha256 of code, free disk, token list, banked-row presence
  ctl30_R0      R0 identity: re-roll 30 tokens (first --ctl-n of each of the 2 shards) at 1 thread vs the banked R0 rows; bar max |dxy| <= 1e-6 m.
                A FAIL STOPS THE CHAIN before the RJ5H5 roll.
  roll_RJ5H5 / score_RJ5H5 / fam_RJ5H5_{C,R} / analyze (tfn3_combo_analyze.py: paired vs the banked arms) / ALL
Resumable: re-launch with the same arguments.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

NAVSIM_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
NAVSIM_DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
HOLD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n3_combo"
BANKED = "F:/tanitad_scratch/tfn3_combo/run/banked"
ARMS = {"RJ5H5": ["--pos-jerk-limit", "5.0", "--policy-interval-s", "0.2"], "R0": []}
EXPECT_N = 469


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def log(run, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(run, "chain.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def mk(run, stage, ok, why=""):
    p = os.path.join(run, "markers", f"{stage}.{'done' if ok else 'FAIL'}")
    open(p, "w", encoding="utf-8").write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {why}\n")


def done(run, stage):
    return os.path.exists(os.path.join(run, "markers", f"{stage}.done"))


def par(run, jobs, tag):
    """jobs = [(name, cmd, env_extra)] run concurrently (<= 2); returns list of rc."""
    procs = []
    for name, cmd, envx in jobs:
        envx = dict(envx)
        cwd = envx.pop("_CWD", None)
        env = dict(os.environ)
        env.update(envx)
        env.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        lf = open(os.path.join(run, "logs", f"{tag}_{name}.log"), "a", encoding="utf-8")
        lf.write(f"# {' '.join(cmd)}\n")
        lf.flush()
        procs.append((subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       cwd=cwd), lf))
    rcs = []
    for p, lf in procs:
        rcs.append(p.wait())
        lf.close()
    return rcs


def population(run):
    return [tuple(x.strip().split(":")) for x in open(os.path.join(run, "tokens.txt"), encoding="utf-8") if x.strip()]


def ok_tokens(pattern):
    ok, err = set(), 0
    for fp in glob.glob(pattern):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                err += 1
            else:
                ok.add(r["token"])
    return ok, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="F:/tanitad_scratch/tfn3_combo/run")
    ap.add_argument("--w", default="F:/tanitad_scratch/tfn3_combo/w")
    ap.add_argument("--arms", default="RJ5H5")
    ap.add_argument("--expect-n", type=int, default=EXPECT_N)
    ap.add_argument("--ctl-n", type=int, default=15)
    ap.add_argument("--banked", default=BANKED)
    a = ap.parse_args()
    run, w = a.run, a.w
    for d in ("markers", "rolls", "scores", "logs", "fam", "result"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    arms = a.arms.split(",")
    pop = population(run)
    # ---- preflight -----------------------------------------------------------------------------------------------------------
    if not done(run, "preflight"):
        try:
            assert len(pop) == a.expect_n, f"token list has {len(pop)} != {a.expect_n}"
            assert sha(os.path.join(w, "eval", "tfn3_roll.py")) == sha(os.path.join(HOLD, "tfn3_roll.py")), "eval/tfn3_roll.py differs from the hold"
            assert sha(os.path.join(w, "refe", "build_teacher_rollouts.py")) == sha(os.path.join(HOLD, "build_teacher_rollouts.py")), "refe/build_teacher_rollouts.py differs from the hold"
            for arm0 in ("R0", "RJ5", "R5HZ", "RG8"):
                assert glob.glob(os.path.join(a.banked, "scores", f"{arm0}_s*.jsonl")) and glob.glob(os.path.join(a.banked, "rolls", f"{arm0}_s*.jsonl")), f"banked {arm0} missing"
            free = shutil.disk_usage(run).free / 2 ** 30
            assert free > 5.0, f"F: free {free:.1f} GiB < 5"
            man = {"tip": "5896f046ca1288d698b67bd155a599a75e1ab707", "run": run, "w": w, "arms": {k: ARMS[k] for k in arms}, "n_tokens": len(pop),
                   "free_gib": round(free, 1),
                   "sha256": {os.path.basename(p): sha(p) for p in (os.path.join(HOLD, n) for n in (
                       "tfn3_roll.py", "tfn3_score.py", "tfn3_combo_analyze.py", "tfn3_combo_chain.py", "build_teacher_rollouts.py", "test_tfn3_combo_reach.py"))},
                   "tokens_sha256": sha(os.path.join(run, "tokens.txt"))}
            json.dump(man, open(os.path.join(run, "manifest.json"), "w", encoding="utf-8"), indent=1)
            mk(run, "preflight", True)
        except Exception as e:                                                    # noqa: BLE001
            mk(run, "preflight", False, f"{type(e).__name__}: {e}")
            return 2
        log(run, "preflight done")
    senv = {"PYTHONPATH": f"{NAVSIM_DEVKIT};{w}/refe", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    # ---- ctl30_R0: identity control, BEFORE any RJ5H5 row ---------------------------------------------------------------------------
    if not done(run, "ctl30_R0"):
        jobs = []
        for i in range(2):
            out = os.path.join(run, "rolls", f"CTL30_R0_s{i}.jsonl")
            cmd = [NAVSIM_PY, f"{w}/eval/run_p4.py", "--log", os.path.join(run, "logs", f"p4_CTL30_s{i}.log"), "--hz", "10", "--threads", "1",
                   "--script", "tfn3_roll.py", "--", "--samples", os.path.join(run, "tokens.txt"), "--shard", f"{i}/2", "--out", out, "--arm", "R0",
                   "--limit", str(a.ctl_n)]
            jobs.append((f"CTL30_s{i}", cmd, {"_CWD": f"{w}/eval"}))
        log(run, "ctl30_R0 start")
        rcs = par(run, jobs, "roll")
        new = {}
        for fp in glob.glob(os.path.join(run, "rolls", "CTL30_R0_s*.jsonl")):
            for ln in open(fp, encoding="utf-8"):
                r = json.loads(ln)
                if "error" not in r:
                    new[r["token"]] = r
        bank = {}
        for fp in glob.glob(os.path.join(a.banked, "rolls", "R0_s*.jsonl")):
            for ln in open(fp, encoding="utf-8"):
                r = json.loads(ln)
                if "error" not in r:
                    bank[r["token"]] = r
        import numpy as np
        d = [float(np.abs(np.asarray(r["traj20"])[:, :2] - np.asarray(bank[t]["traj20"])[:, :2]).max()) for t, r in new.items() if t in bank]
        rep_ = {"n": len(d), "n_expected": 2 * a.ctl_n, "max_dxy_m": max(d) if d else None, "threads": 1, "bar": "max |dxy| <= 1e-6 m vs the banked R0 rows (committed before any RJ5H5 row)",
                "PASS": bool(d) and len(d) == 2 * a.ctl_n and max(d) <= 1e-6 and not any(rcs)}
        json.dump(rep_, open(os.path.join(run, "result", "ctl30_R0.json"), "w", encoding="utf-8"), indent=1)
        mk(run, "ctl30_R0", rep_["PASS"], json.dumps(rep_))
        log(run, f"ctl30_R0 {json.dumps(rep_)}")
        if not rep_["PASS"]:
            return 4
    for arm in arms:
        # ---- roll ------------------------------------------------------------------------------------------------------------
        st = f"roll_{arm}"
        if not done(run, st):
            jobs = []
            for i in range(2):
                out = os.path.join(run, "rolls", f"{arm}_s{i}.jsonl")
                cmd = [NAVSIM_PY, f"{w}/eval/run_p4.py", "--log", os.path.join(run, "logs", f"p4_{arm}_s{i}.log"), "--hz", "10", "--threads", "1",
                       "--script", "tfn3_roll.py", "--", "--samples", os.path.join(run, "tokens.txt"), "--shard", f"{i}/2", "--out", out,
                       "--arm", arm] + ARMS[arm]
                jobs.append((f"{arm}_s{i}", cmd, {"_CWD": f"{w}/eval"}))
            log(run, f"{st} start")
            t0 = time.time()
            rcs = par(run, jobs, "roll")
            ok, err = ok_tokens(os.path.join(run, "rolls", f"{arm}_s*.jsonl"))
            miss = {t for _, t in pop} - ok
            if any(rcs) or miss:
                mk(run, st, False, f"rc {rcs}; {len(miss)} tokens without an ok row; {err} error rows in the file")
                log(run, f"{st} FAIL rc {rcs} missing {len(miss)}")
                return 1
            mk(run, st, True, f"{len(ok)} ok rows, {err} stale error rows, {time.time() - t0:.0f}s")
            log(run, f"{st} done {time.time() - t0:.0f}s")
        # ---- score -----------------------------------------------------------------------------------------------------------
        st = f"score_{arm}"
        if not done(run, st):
            jobs = []
            for i in range(2):
                cmd = [NAVSIM_PY, f"{HOLD}/tfn3_score.py", "--w", w, "--rolls", os.path.join(run, "rolls", f"{arm}_s*.jsonl"), "--arm", arm,
                       "--out", os.path.join(run, "scores", f"{arm}_s{i}.jsonl"), "--shard", f"{i}/2"]
                jobs.append((f"{arm}_s{i}", cmd, senv))
            log(run, f"{st} start")
            t0 = time.time()
            rcs = par(run, jobs, "score")
            ok, err = ok_tokens(os.path.join(run, "scores", f"{arm}_s*.jsonl"))
            miss = {t for _, t in pop} - ok
            if any(rcs) or miss:
                mk(run, st, False, f"rc {rcs}; {len(miss)} tokens without an ok score row; {err} error rows")
                log(run, f"{st} FAIL rc {rcs} missing {len(miss)}")
                return 1
            mk(run, st, True, f"{len(ok)} ok rows, {err} stale error rows, {time.time() - t0:.0f}s")
            log(run, f"{st} done {time.time() - t0:.0f}s")
    # ---- families --------------------------------------------------------------------------------------------------------------
    fjobs = []
    for arm in arms:
        for f in ("C", "R"):
            if not done(run, f"fam_{arm}_{f}"):
                cmd = [NAVSIM_PY, f"{w}/eval/instruments/families_run.py", "--rows", os.path.join(run, "scores", f"{arm}_s*.jsonl"), "--arm", f,
                       "--label", f"{arm}_{f}", "--out", os.path.join(run, "fam", f"fam_{arm}_{f}.json"), "--n-boot", "2000",
                       "--tanitad-repo", f"{w}/repo"]
                fjobs.append((f"{arm}_{f}", cmd, senv))
    for i in range(0, len(fjobs), 2):
        batch = fjobs[i:i + 2]
        rcs = par(run, batch, "fam")
        for (name, _, _), rc in zip(batch, rcs):
            mk(run, f"fam_{name}", rc == 0, f"rc {rc}")
            if rc:
                log(run, f"fam_{name} FAIL rc {rc}")
    # ---- analyze ---------------------------------------------------------------------------------------------------------------
    if not done(run, "analyze"):
        out = os.path.join(run, "result", "TFN3_COMBO_RESULT.json")
        rc = par(run, [("analyze", [NAVSIM_PY, f"{HOLD}/tfn3_combo_analyze.py", "--run", run, "--out", out, "--fam-dir", os.path.join(run, "fam"), "--banked", a.banked], {})], "analyze")[0]
        mk(run, "analyze", rc == 0, f"rc {rc}")
        if rc:
            log(run, "analyze FAIL")
            return 1
        log(run, "analyze done")
    nfam = len(glob.glob(os.path.join(run, "markers", "fam_*.FAIL")))
    mk(run, "ALL", True, f"result/TFN3_COMBO_RESULT.json; family stage failures: {nfam}")
    log(run, "ALL done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
