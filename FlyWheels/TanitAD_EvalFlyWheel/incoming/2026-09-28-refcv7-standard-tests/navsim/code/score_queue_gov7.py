#!/usr/bin/env python3
"""A SEQUENTIAL, GOVERNED scoring queue for seams that already exist (TANITAD VENV).

    python code/score_queue_gov7.py --split warmup --bridge raw/milestones/step50400/bridge_warmup \
        --scores raw/milestones/step50400/scores_warmup --step 50400 \
        --arms R7_A1,R7_A1_s1,PRIOR_ha0p,R7_CEILDECL_d,R7_VMAXOFF,R7_FILTOFF,R7_BLIND,R7_NAVOFF,R7_A1NT

Runs the SAME scorer drivers the runner runs (``score_arm7.py`` for warmup / navhard,
``score_navtest7.py`` for navtest -- built by the runner's own constants), strictly ONE AT A TIME,
each of them behind ``ram_governor7`` (slot + RAM window + pause-instead-of-abort). A job whose
counts already read PASS is skipped; a job that ends on the scorer's RAM guard is retried (up to
``--tries``); any other FAIL is reported and the queue moves on -- never scored partial, never
retried blindly. Every decision is logged to ``<scores>/queue.log``.

WHY it exists (2026-10-04): the step-50,400 runner (``--splits navtest,navhard``) never scores the
WARMUP split -- its first incarnation died on ENOSPC before draining its pool, and the second was
launched without it -- so ``scores_warmup/`` was EMPTY although all nine warmup seams are OK. This
queue scores them without touching the runner, and without writing MILESTONE_SUMMARY.json /
BARS.json (those belong to the runner / ``complete_milestone7.py``: writing them early would tell
the EPDMS waiter the milestone is DONE).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_navsim_refcv7 as R  # noqa: E402  (its constants + jget/log; importing runs nothing)


def build_job(split: str, step: int, arm: str, bdir: str, sdir: str, exp_label: str | None = None,
              sub_tokens: str = "") -> dict:
    seam = os.path.abspath(os.path.join(bdir, f"seam_{arm}.npz"))
    sdir = os.path.abspath(sdir)
    if split == "navtest":
        lab = exp_label or f"r7s{step}_{arm}"
        cmd = [R.PY, os.path.join(HERE, "score_navtest7.py"), "--label", lab, "--seam", seam,
               "--out", sdir] + (["--tokens", sub_tokens] if sub_tokens else [])
        return {"name": f"navtest:{lab}", "cmd": cmd,
                "counts": os.path.join(sdir, lab, f"{lab}.counts.json"),
                "scorer_log": os.path.join(sdir, lab, f"{lab}.log"),
                "driver_log": os.path.join(sdir, f"{lab}.driver.txt"), "seam": seam}
    sp = R.SPLITS[split]["split"]
    tag = arm if sp == "warmup_two_stage" else f"{arm}__{sp}"
    cmd = [R.NPY, os.path.join(HERE, "score_arm7.py"), "--arm", arm, "--seam", seam, "--split", sp,
           "--out", sdir, "--exp-tag", f"e7s{step}"]
    return {"name": f"{split}:{arm}", "cmd": cmd,
            "counts": os.path.join(sdir, f"score_{tag}.counts.json"),
            "scorer_log": os.path.join(sdir, f"score_{tag}.log"),
            "driver_log": os.path.join(sdir, f"{arm}.driver.txt"), "seam": seam}


def seam_ok(job: dict) -> str:
    man = job["seam"].replace(".npz", ".manifest.json")
    if not (os.path.exists(job["seam"]) and os.path.exists(man)):
        return "NO_SEAM"
    return "PARTIAL_SEAM" if R.jget(man, "partial") else "OK"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("warmup", "navtest", "navhard"))
    ap.add_argument("--bridge", required=True, help="dir holding seam_<arm>.npz + manifests")
    ap.add_argument("--scores", required=True, help="output dir (created)")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tries", type=int, default=6)
    ap.add_argument("--label-prefix", default="", help="navtest only: label = <prefix><arm> "
                    "(default r7s<step>_<arm>); must start with 'r7'")
    ap.add_argument("--tokens", default="", help="navtest only: W3-format token subset")
    a = ap.parse_args(argv)
    os.makedirs(a.scores, exist_ok=True)
    qlog = os.path.join(os.path.abspath(a.scores), "queue.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{R.TREE}/stack;{R.TREE}/taniteval"
    env["TANITAD_REPO"] = R.TREE
    env["OMP_NUM_THREADS"] = "6"
    env["PYTHONIOENCODING"] = "utf-8"
    summary = {}
    for arm in [x for x in a.arms.split(",") if x]:
        lab = f"{a.label_prefix}{arm}" if a.label_prefix else None
        job = build_job(a.split, a.step, arm, a.bridge, a.scores, lab, a.tokens)
        so = seam_ok(job)
        if so != "OK":
            R.log(f"QUEUE {job['name']} skipped: seam {so}", qlog)
            summary[job["name"]] = f"SEAM_{so}"
            continue
        for k in range(1, a.tries + 1):
            if R.jget(job["counts"], "status") == "PASS" and k == 1:
                R.log(f"QUEUE {job['name']} PASS already", qlog)
                break
            R.log(f"QUEUE {job['name']} launch try={k}", qlog)
            with open(job["driver_log"], "a", encoding="utf-8") as fh:
                fh.write("CMD: " + " ".join(job["cmd"]) + "\n")
                fh.flush()
                t0 = time.time()
                rc = subprocess.run(job["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
            st = R.jget(job["counts"], "status")
            aborted = False
            if os.path.exists(job["scorer_log"]):
                txt = open(job["scorer_log"], encoding="utf-8", errors="replace").read()
                aborted = "RAM_GUARD" in txt
            gov = R.jget(job["counts"], "governor") or {}
            R.log(f"QUEUE {job['name']} ended counts={st} rc={rc} ram_guard_abort={aborted} "
                  f"wall={time.time() - t0:.0f}s pauses={gov.get('n_pauses')} "
                  f"paused_s={gov.get('paused_s_total')}", qlog)
            if st == "PASS":
                break
            if not aborted:
                R.log(f"QUEUE {job['name']} FAILED for a reason other than the RAM guard -- "
                      f"not retried; read {job['scorer_log']}", qlog)
                break
        summary[job["name"]] = R.jget(job["counts"], "status")
    R.log("QUEUE done " + json.dumps(summary), qlog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
