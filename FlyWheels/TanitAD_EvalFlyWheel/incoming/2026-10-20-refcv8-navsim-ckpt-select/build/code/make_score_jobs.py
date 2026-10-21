#!/usr/bin/env python3
"""make_score_jobs.py -- the Thor pool's ``jobs.jsonl`` for the SEVEN scorer arms of the D-R8-CKPT-RULE navtrain selection (any python).

The labels, seam names and output layout are taken FROM ``select_ckpt_r8`` (one source of truth): a label typed by hand into a job list is a
label ``select_ckpt_r8.spec_from_layout`` will not find, and the decision then reads INCOMPLETE for a reason nobody can see.

    python make_score_jobs.py --layout-root /home/nvidia/r8ns/navtrain_sel --sample <NAVTRAIN_SAMPLE_TOKENS.json> \
        --driver <.../navtrain_cache_score.py> --python python3 --out jobs.jsonl

Seven jobs: R8_RT and R8_RT_s1 at each of ``ema40000`` / ``ema48852`` (4), the PRIOR_ha0p seam of the default checkpoint, and the official STOP
and CV agents.  Refuses (exit 2) if a seam file is absent -- the launch script's ``ZZROLLSDONE`` is what makes them exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_ckpt_r8 as SEL  # noqa: E402

MUTATION = os.environ.get("R8JOBS_MUTATION", "")
MUTATIONS = ("label_typo", "prior_from_challenger", "seed_seams_swapped", "missing_seam_ignored")


def arm_table(root: str) -> list:
    """``[(label, extra argv, seam path or None)]`` in a fixed order."""
    root = root.replace("\\", "/").rstrip("/")
    out = []
    for ck in (SEL.CHALLENGER, SEL.DEFAULT):
        for s in SEL.SEEDS:
            seam = f"{root}/{ck}/bridge/seam_{SEL.SEAM_ARM[1 - s if MUTATION == 'seed_seams_swapped' else s]}.npz"
            lab = SEL.arm_label(ck, s)
            out.append((lab.replace("r8sel_", "r8sel-") if MUTATION == "label_typo" else lab, ["--seam", seam], seam))
    seam = f"{root}/{SEL.CHALLENGER if MUTATION == 'prior_from_challenger' else SEL.DEFAULT}/bridge/seam_PRIOR_ha0p.npz"
    out.append((f"{SEL.LABEL_PREFIX}_ref_PRIOR_ha0p", ["--seam", seam], seam))
    out.append((f"{SEL.LABEL_PREFIX}_ref_STOP", ["--official", "STOP"], None))
    out.append((f"{SEL.LABEL_PREFIX}_ref_CV", ["--official", "CV"], None))
    return out


def make_jobs(root: str, sample: str, driver: str, python: str = "python3", profile: str = "thor") -> list:
    root = root.replace("\\", "/").rstrip("/")
    jobs = []
    for label, extra, _ in arm_table(root):
        jobs.append({"name": label, "cwd": root, "log": f"{root}/pool_logs/{label}.log",
                     "argv": [python, driver, "score", "--profile", profile, "--sample", sample, "--out", root, "--label", label] + extra})
    return jobs


def run(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-root", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--python", default="python3")
    ap.add_argument("--profile", default="thor")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    missing = [seam for _, _, seam in arm_table(a.layout_root) if seam and not os.path.isfile(seam)]
    if missing and MUTATION != "missing_seam_ignored":
        print(f"ZZJOBS_REFUSED_missing_seams n={len(missing)}")
        for m in missing:
            print("  " + m)
        return 2
    jobs = make_jobs(a.layout_root, a.sample, a.driver, a.python, a.profile)
    with open(a.out, "w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    print(f"ZZJOBS_DONE n={len(jobs)}")
    return 0


def main(argv=None) -> int:
    if MUTATION:
        sys.exit("[make_score_jobs] R8JOBS_MUTATION is set -- a mutation-test environment never writes a job list")
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
