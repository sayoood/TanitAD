#!/usr/bin/env python3
"""Launcher for eval/teacher_navtest_seam.py in the environment the teacher bank was built with
(eval_checkpoint.env_driverl(), every D:/ re-pointed to E:/ exactly as eval/stage2_sft.py does it).

    python eval/run_p4.py --log <file> [--hz 10] [--threads 2] -- <teacher_navtest_seam.py args>

`--hz` sets REFE_SIM_HZ (navtrain_scenarios reads it at import). The route-rank env var is REMOVED so rank 0 runs
THEIR unpatched route code (route_rank.apply(0) refuses otherwise). Exit code = the child's.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC  # noqa: E402

PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--hz", default="10", choices=("10", "20"))
    ap.add_argument("--threads", default="2")
    ap.add_argument("--script", default="teacher_navtest_seam.py", help="the eval/ script to run in this environment")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    env = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in EC.env_driverl().items()}
    env["REFE_SIM_HZ"] = a.hz
    env["OMP_NUM_THREADS"] = a.threads
    env["MKL_NUM_THREADS"] = a.threads
    env.pop("DRIVERL_EVAL_ROUTE_LANE_RANK", None)
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    cmd = [PY_D, os.path.join(HERE, a.script)] + rest + ["--threads", a.threads]
    with open(a.log, "w", encoding="utf-8") as f:
        f.write(f"# REFE_SIM_HZ={a.hz} OMP_NUM_THREADS={a.threads}\n# {' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    sys.exit(main())
