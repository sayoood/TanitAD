#!/usr/bin/env python3
"""TF-N2 teacher re-roll chain (detached; CPU only, at most 2 processes). Navtrain held-out tokens ONLY.

Stages, each with <run>/markers/<stage>.done or <stage>.FAIL (+ the reason inside); the last marker is ALL.done:
  preflight                       sha256 of the code in use, free disk, token list (469), manifest.json
  roll_<ARM>  (2 shards)          eval/run_p4.py --script tfn2_reroll_roll.py   (driverl-eval venv, CPU, REFE_SIM_HZ 10)
  score_<ARM> (2 shards)          tfn2_reroll_score.py                          (navsim venv, W-TRANSP route, forms T / C / R / human)
  ctl_R0                          R0 identity: re-rolled t20 vs the banked t20 (after score_R0; read by analyze as well)
  fam_<ARM>_<C|R>                 families_run.py (executed_tracked, STRATEGIC paused), 2 at a time
  analyze                         tfn2_reroll_analyze.py -> result/TFN2_REROLL_RESULT.json
  ALL
ARMS (knobs are explicit flags; R0 passes none):  R0 control | RJ5 --pos-jerk-limit 5.0 | RG8 --route-goal-min-speed 8.0 --route-goal-horizon 12.0
                                                  | R5HZ --policy-interval-s 0.2
Resumable: a re-launch with the same arguments skips every stage that has a .done marker, and rows already written are skipped.
A roll stage is FAIL unless every population token has an ok row and the knob read-back matched (an error row from KNOB-NOT-REACHED is a FAIL).
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
HOLD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n2/reroll"
ARMS = {"R0": [], "RJ5": ["--pos-jerk-limit", "5.0"], "RG8": ["--route-goal-min-speed", "8.0", "--route-goal-horizon", "12.0"],
        "R5HZ": ["--policy-interval-s", "0.2"]}
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
    ap.add_argument("--run", default="F:/tanitad_scratch/tfn2_reroll/run")
    ap.add_argument("--w", default="F:/tanitad_scratch/tfn2_reroll/w")
    ap.add_argument("--arms", default="R0,RJ5,RG8,R5HZ")
    a = ap.parse_args()
    run, w = a.run, a.w
    for d in ("markers", "rolls", "scores", "logs", "fam", "result"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    arms = a.arms.split(",")
    pop = population(run)
    # ---- preflight -----------------------------------------------------------------------------------------------------------
    if not done(run, "preflight"):
        try:
            assert len(pop) == EXPECT_N, f"token list has {len(pop)} != {EXPECT_N}"
            assert sha(os.path.join(w, "eval", "tfn2_reroll_roll.py")) == sha(os.path.join(HOLD, "tfn2_reroll_roll.py")), "eval/tfn2_reroll_roll.py differs from the hold"
            assert sha(os.path.join(w, "refe", "build_teacher_rollouts.py")) == sha(os.path.join(HOLD, "build_teacher_rollouts.py")), "refe/build_teacher_rollouts.py differs from the hold"
            free = shutil.disk_usage(run).free / 2 ** 30
            assert free > 5.0, f"F: free {free:.1f} GiB < 5"
            man = {"tip": "be6567aa (the N2 probe's export tip)", "run": run, "w": w, "arms": {k: ARMS[k] for k in arms}, "n_tokens": len(pop),
                   "free_gib": round(free, 1),
                   "sha256": {os.path.basename(p): sha(p) for p in (os.path.join(HOLD, n) for n in (
                       "tfn2_reroll_roll.py", "tfn2_reroll_score.py", "tfn2_reroll_analyze.py", "tfn2_reroll_chain.py", "build_teacher_rollouts.py"))},
                   "tokens_sha256": sha(os.path.join(run, "tokens.txt"))}
            json.dump(man, open(os.path.join(run, "manifest.json"), "w", encoding="utf-8"), indent=1)
            mk(run, "preflight", True)
        except Exception as e:                                                    # noqa: BLE001
            mk(run, "preflight", False, f"{type(e).__name__}: {e}")
            return 2
        log(run, "preflight done")
    senv = {"PYTHONPATH": f"{NAVSIM_DEVKIT};{w}/refe", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for arm in arms:
        # ---- roll ------------------------------------------------------------------------------------------------------------
        st = f"roll_{arm}"
        if not done(run, st):
            jobs = []
            for i in range(2):
                out = os.path.join(run, "rolls", f"{arm}_s{i}.jsonl")
                cmd = [NAVSIM_PY, f"{w}/eval/run_p4.py", "--log", os.path.join(run, "logs", f"p4_{arm}_s{i}.log"), "--hz", "10", "--threads", "1",
                       "--script", "tfn2_reroll_roll.py", "--", "--samples", os.path.join(run, "tokens.txt"), "--shard", f"{i}/2", "--out", out,
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
                cmd = [NAVSIM_PY, f"{HOLD}/tfn2_reroll_score.py", "--w", w, "--rolls", os.path.join(run, "rolls", f"{arm}_s*.jsonl"), "--arm", arm,
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
        if arm == "R0" and not done(run, "ctl_R0"):
            d = []
            for fp in glob.glob(os.path.join(run, "scores", "R0_s*.jsonl")):
                d += [json.loads(l)["bank_dxy_m"] for l in open(fp, encoding="utf-8") if '"error"' not in l]
            rep = {"n": len(d), "max_dxy_m": max(d), "share_le_1e-6": sum(x <= 1e-6 for x in d) / len(d), "bar": "max <= 1e-6 m (committed before the run)",
                   "PASS": max(d) <= 1e-6}
            json.dump(rep, open(os.path.join(run, "result", "ctl_R0.json"), "w", encoding="utf-8"), indent=1)
            mk(run, "ctl_R0", True, json.dumps(rep))
            log(run, f"ctl_R0 {json.dumps(rep)}")
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
        out = os.path.join(run, "result", "TFN2_REROLL_RESULT.json")
        rc = par(run, [("analyze", [NAVSIM_PY, f"{HOLD}/tfn2_reroll_analyze.py", "--run", run, "--out", out, "--fam-dir", os.path.join(run, "fam")], {})], "analyze")[0]
        mk(run, "analyze", rc == 0, f"rc {rc}")
        if rc:
            log(run, "analyze FAIL")
            return 1
        log(run, "analyze done")
    nfam = len(glob.glob(os.path.join(run, "markers", "fam_*.FAIL")))
    mk(run, "ALL", True, f"result/TFN2_REROLL_RESULT.json; family stage failures: {nfam}")
    log(run, "ALL done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
