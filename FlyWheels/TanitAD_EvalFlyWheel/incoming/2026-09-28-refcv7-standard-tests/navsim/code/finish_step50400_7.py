#!/usr/bin/env python3
"""EVENT-DRIVEN finisher for the step-50,400 NavSim milestone (TANITAD VENV). Detached; reads ARTIFACTS, never exit codes.

    python code/finish_step50400_7.py --phase runner  --wait-pid 31600   # after the runner exits
    python code/finish_step50400_7.py --phase vmaxoff --wait-pid <orchestrator pid>

PHASE runner  (waits for the runner PID to be GONE and for the governed warmup queue's ``QUEUE done`` line):
  1. every scorer of the three splits (main + derived arms, navtest's 200-token diagnostics) whose count guard
     does not read PASS goes through ``score_queue_gov7.py`` (sequential, governed, 6 tries) -- the runner gave
     up on it or died; PASS ones are never touched;
  2. ``complete_milestone7.postprocess_split`` (the runner's own post-drain steps, imported, line-for-line) for
     warmup / navtest / navhard;
  3. MILESTONE_SUMMARY.json: the runner's own record is KEPT; warmup's record is rebuilt from the seam
     manifests (``split_record``), every split's summary / decomposition path is refreshed; ``bars7.py``;
     ``step_compare7.py`` vs step 30,000 and step 5,000 on every split;
  4. a ``ZZFINISH50400DONEZZ`` line in ``<milestone>/finish.log`` and a refreshed landing list.
  It does NOT write MILESTONE_SUMMARY.json / BARS.json before every scorer has a verdict -- the EPDMS waiter
  reads their presence as "milestone DONE".
PHASE vmaxoff (waits for ``run_vmaxoff_legal7.py`` to exit): ``vmaxoff_legal_summary7.py`` + the landing list.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import complete_milestone7 as C  # noqa: E402  (imports run_navsim_refcv7 as C.R; runs nothing)

R = C.R
MS = os.path.join(PKG, "raw", "milestones", "step50400")
CKPT, MD5, STEP = "D:/refcv7_eval_kit/ckpt/ckpt_50400.pt", "b418d0fc4a92a6848c246a6a7c50207b", 50400
FLOG = os.path.join(MS, "finish.log")


def pid_alive(spec: str) -> bool:
    import psutil
    pid = int(spec.split(":")[0])
    try:
        p = psutil.Process(pid)
        if ":" in spec and abs(p.create_time() - float(spec.split(":")[1])) > 2.0:
            return False                                          # the pid was recycled
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def wait_gone(specs: list, log) -> None:
    n = 0
    while any(pid_alive(s) for s in specs):
        if n % 30 == 0:
            log(f"waiting for {specs} to exit")
        n += 1
        time.sleep(60)


def not_pass_jobs(splits: list) -> dict:
    """{split: [arm,...]} (main kind only; navtest sub200 handled by the label rule) not PASS."""
    out = {}
    for sk in splits:
        for kind, (d, arms) in C.dirs_of(MS, sk).items():
            for j, arm in zip(C.score_jobs(sk, kind, d, arms, STEP, MS), arms):
                if R.jget(j["counts"], "status") != "PASS" and R.seams_ready(d, [arm])[arm] == "OK":
                    out.setdefault((sk, kind), []).append(arm)
    return out


def phase_runner(a, log) -> int:
    wait_gone(a.wait_pid, log)
    q = os.path.join(MS, "scores_warmup", "queue.log")
    while not (os.path.exists(q) and "QUEUE done" in open(q, encoding="utf-8", errors="replace").read()):
        log("waiting for the warmup queue to finish")
        time.sleep(60)
    env = dict(os.environ, PYTHONPATH=f"{R.TREE}/stack;{R.TREE}/taniteval", TANITAD_REPO=R.TREE,
               OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8")
    splits = ["warmup", "navtest", "navhard"]
    for rnd in range(3):
        todo = not_pass_jobs(splits)
        log(f"round {rnd}: scorers not PASS: { {f'{k[0]}/{k[1]}': v for k, v in todo.items()} }")
        if not todo:
            break
        for (sk, kind), arms in todo.items():
            d = C.dirs_of(MS, sk)[kind][0]
            if kind == "sub200":
                continue          # the 200-token diagnostics already read PASS; never re-run them by this route
            cmd = [R.PY, os.path.join(HERE, "score_queue_gov7.py"), "--split", sk, "--bridge", d,
                   "--scores", os.path.join(MS, f"scores_{sk}"), "--step", str(STEP), "--arms", ",".join(arms)]
            subprocess.run(cmd, env=env, stdout=open(os.path.join(MS, "finish.queue.txt"), "a"),
                           stderr=subprocess.STDOUT)
    left = not_pass_jobs(splits)
    log(f"after scoring, not PASS: { {f'{k[0]}/{k[1]}': v for k, v in left.items()} }")
    label = "RESULT"
    recs = {sk: C.postprocess_split(sk, MS, STEP, label, env, FLOG) for sk in splits}
    msp = os.path.join(MS, "MILESTONE_SUMMARY.json")
    cfg_md5 = R.md5_file(R.CONFIG)
    prev = R.jget(msp) or {"ckpt": CKPT, "md5": MD5, "step": STEP, "label": label, "config_md5": cfg_md5,
                           "splits": {}}
    if prev.get("md5") != MD5:
        log(f"REFUSED: {msp} belongs to md5 {prev.get('md5')}")
        return 2
    prev.setdefault("splits", {})
    for sk in splits:
        cur = prev["splits"].get(sk) or C.split_record(MS, sk, {})
        cur.update(recs[sk])
        prev["splits"][sk] = cur
    prev["completed_by"] = {"script": "code/finish_step50400_7.py", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "why": ("the runner was launched with --splits navtest,navhard (its first incarnation died on "
                                    "ENOSPC before warmup was scored): warmup was scored by score_queue_gov7.py and every "
                                    "split re-parsed with the runner's own post-drain steps"),
                            "not_pass_after": {f"{k[0]}/{k[1]}": v for k, v in left.items()}}
    with open(msp, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, indent=1)
    R.run([R.PY, os.path.join(HERE, "bars7.py"), "--milestone", MS], env, os.path.join(MS, "bars.log"))
    log(f"BARS {'present' if os.path.exists(os.path.join(MS, 'BARS.json')) else 'ABSENT'}")
    for tag in ("step30000", "step5000"):
        ca = os.path.join(PKG, "raw", "milestones", tag)
        for sk in splits:
            co = os.path.join(MS, f"compare_vs_{tag}_{sk}.json")
            R.run([R.PY, os.path.join(HERE, "step_compare7.py"), "--split", sk, "--a", ca, "--b", MS, "--out", co],
                  env, os.path.join(MS, f"compare_vs_{tag}_{sk}.log"))
            log(f"COMPARE {sk} vs {tag}: {'present' if os.path.exists(co) else 'ABSENT'}")
    log(f"ZZFINISH50400DONEZZ not_pass={sorted(f'{k[0]}/{k[1]}:{v}' for k, v in left.items())}")
    subprocess.run([R.PY, os.path.join(HERE, "landing_navsim50k_legal7.py")], env=env)
    return 0


def phase_vmaxoff(a, log) -> int:
    wait_gone(a.wait_pid, log)
    env = dict(os.environ, PYTHONPATH=f"{R.TREE}/stack;{R.TREE}/taniteval", TANITAD_REPO=R.TREE,
               PYTHONIOENCODING="utf-8")
    r = subprocess.run([R.PY, os.path.join(HERE, "vmaxoff_legal_summary7.py"), "--root", MS], env=env,
                       capture_output=True, text=True)
    open(os.path.join(MS, "vmaxoff_legal", "summary.stdout.txt"), "w", encoding="utf-8").write(r.stdout + r.stderr)
    log(f"vmaxoff summary rc={r.returncode}: {r.stdout[-600:]}")
    log("ZZFINISHVMAXOFFDONEZZ")
    subprocess.run([R.PY, os.path.join(HERE, "landing_navsim50k_legal7.py")], env=env)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("runner", "vmaxoff"))
    ap.add_argument("--wait-pid", action="append", default=[], help="PID or PID:create_time")
    a = ap.parse_args(argv)

    def log(m):
        R.log(m, FLOG if a.phase == "runner" else os.path.join(MS, "vmaxoff_legal", "finish_vmaxoff.log"))
    return phase_runner(a, log) if a.phase == "runner" else phase_vmaxoff(a, log)


if __name__ == "__main__":
    sys.exit(main())
