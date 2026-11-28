#!/usr/bin/env python3
"""Re-write the model-as-trained stamp on already-banked milestone summaries by RE-PARSING them
(never by editing JSON), and prove nothing but the stamp changed.

    python code/restamp6.py raw/milestones/step5000 raw/milestones/step30000

Per milestone dir: every split whose ``summary_<split>.json`` exists is re-parsed with the same
arguments the runner uses (+ the navtest diagnostic dir ``<ms>_navtest_diag`` when present), then
``bars6.py``. The before/after JSONs must be identical apart from ``model_as_trained`` /
``checkpoint_step`` — anything else is reported as a DIFF and the old file is restored.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PY = sys.executable
E2RAW = os.path.join(PKG, "..", "..", "2026-09-19-navsim-refcv4b-bridge", "raw")
INP = {"warmup": os.path.join(E2RAW, "navsim_agent_inputs.json"),
       "navhard": "C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json"}
SUB = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")
ORA = os.path.join(PKG, "raw", "inputs", "vmax_oracle_navtest.json")
MAP = os.path.join(PKG, "raw", "inputs", "speed_limits_navtest.json")
IGNORE = {"model_as_trained", "checkpoint_step"}


def _norm(d):
    return json.dumps({k: v for k, v in d.items() if k not in IGNORE}, sort_keys=True, default=str)


def navtest_cmd(ms, summ, sub):
    step = json.load(open(os.path.join(ms, "MILESTONE_SUMMARY.json"), encoding="utf-8"))["step"]
    sc = os.path.join(ms, "scores_navtest")
    tag = "_sub" if sub else ""
    lab = f"r6s{step}_R6_A1{tag}"
    cmd = [PY, os.path.join(HERE, "parse_navtest6.py"), "--arm-csv", os.path.join(sc, lab, f"{lab}.csv"),
           "--label", "RESULT" if step >= 5000 else "PIPELINE-VALIDATION", "--out", summ,
           "--bridge", os.path.join(ms, "bridge_navtest")]
    extras = []
    for arm in ("R6_A1_s1", "R6_VMAXOFF", "R6_VMAXORACLE"):
        la = f"r6s{step}_{arm}{tag}"
        ca = os.path.join(sc, la, f"{la}.csv")
        if os.path.exists(ca):
            extras.append(f"{arm}={ca}")
    if extras:
        cmd += ["--extra"] + extras
    if sub:
        cmd += ["--tokens", SUB]
    if os.path.exists(ORA):
        cmd += ["--census", ORA, "--map", MAP]
    return cmd


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    bad = 0
    for ms in (argv or sys.argv[1:]):
        dirs = [(ms, False)] + ([(ms.rstrip("/\\") + "_navtest_diag", True)]
                                if os.path.isdir(ms.rstrip("/\\") + "_navtest_diag") else [])
        for d, sub in dirs:
            for summ in sorted(glob.glob(os.path.join(d, "summary_*.json"))):
                sk = os.path.basename(summ)[len("summary_"):-len(".json")]
                old = json.load(open(summ, encoding="utf-8"))
                bak = summ + ".bak"
                shutil.copyfile(summ, bak)
                if sk == "navtest":
                    cmd = navtest_cmd(d, summ, sub)
                else:
                    split = {"warmup": "warmup_two_stage", "navhard": "navhard_two_stage"}[sk]
                    cmd = [PY, os.path.join(HERE, "parse6.py"), "--split", split,
                           "--scores", os.path.join(d, f"scores_{sk}"),
                           "--floors", os.path.join(PKG, "raw", "floors", split),
                           "--bridge", os.path.join(d, f"bridge_{sk}"), "--inputs", INP[sk],
                           "--label", old.get("_label", "RESULT"), "--out", summ]
                    if sk == "navhard":
                        cmd += ["--csv-suffix", "__navhard_two_stage"]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                new = json.load(open(summ, encoding="utf-8"))
                if _norm(old) == _norm(new):
                    print(f"RESTAMPED {summ}: {str(new.get('model_as_trained'))[:60]}…")
                    os.remove(bak)
                else:
                    bad += 1
                    shutil.copyfile(bak, summ)
                    print(f"DIFF {summ} — restored the old file")
            if not sub and os.path.exists(os.path.join(d, "MILESTONE_SUMMARY.json")):
                subprocess.run([PY, os.path.join(HERE, "bars6.py"), "--milestone", d],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"BARS re-evaluated {d}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
