"""D6 P4 CPU stages -- at most ``D6_MAX_PAR`` (default 2) scoring processes, each launched only while free COMMIT >= ``D6_GATE_GB`` (default
6.0 GB; the 2026-10-04 evening ruling).  tanitad venv launcher; the workers run in the NAVSIM venv.  Never touches the GPU.

    python d6_p4_cpu.py --stage base                 # G0 exact re-score of all 5,912 tokens (2 shards) + the G-ORC GT bits (2 shards)
    python d6_p4_cpu.py --stage arms --arms G1,G1DER,GORC   # exact scoring of every CHANGED pick of those arms (2 shards each)

Every worker is resumable (skips finished tokens); the stage writes raw/p4_cpu_<stage>_done.json when all its workers ended.
The done-marker records each worker's exit code, but the ADMISSIBLE evidence of completion is the row count in the outputs (checked by
d6_p4_aggregate.py), never these exit codes.
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
from run_p1p2_cpu import free_gb  # noqa: E402  (commit-aware free memory, the D6 pattern)

RAW = os.path.join(os.path.dirname(HERE), "raw")
NPY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
ENV = dict(os.environ, PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit", OMP_NUM_THREADS="2",
           PYTHONIOENCODING="utf-8")
GATE_GB = float(os.environ.get("D6_GATE_GB", "6.0"))
MAX_PAR = int(os.environ.get("D6_MAX_PAR", "2"))
NSH = 2


def log(msg):
    with open(os.path.join(RAW, "p4_cpu_chain.log"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")


def run_pool(jobs):
    live, q, res = [], list(jobs), {}
    while q or live:
        for j in list(live):
            if j[2].poll() is not None:
                log(f"ended {j[0]} rc={j[2].returncode}")
                res[j[0]] = j[2].returncode
                live.remove(j)
        if q and len(live) < MAX_PAR:
            fg = free_gb()
            if fg >= GATE_GB:
                name, cmd = q.pop(0)
                fh = open(os.path.join(RAW, f"p4_cpu_{name}.log"), "a", encoding="utf-8")
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENV)
                live.append((name, cmd, p))
                log(f"launched {name} pid={p.pid} (free commit {fg:.1f} GB)")
        time.sleep(10)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("base", "arms"))
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    sc = os.path.join(RAW, "p4_scored")
    os.makedirs(sc, exist_ok=True)
    jobs = []
    if a.stage == "base":
        for k in range(NSH):
            jobs.append((f"G0_s{k}", [NPY, os.path.join(HERE, "d6_p4_score.py"), "--plans", os.path.join(RAW, "p4_plans_G0.npz"),
                                      "--out", os.path.join(sc, f"G0_s{k}.jsonl"), "--shard", f"{k}/{NSH}"]))
        for k in range(NSH):
            jobs.append((f"ORC_s{k}", [NPY, os.path.join(HERE, "d6_p4_gate.py"), "--mode", "orc", "--shard", f"{k}/{NSH}"]))
    else:
        for arm in [x for x in a.arms.split(",") if x]:
            pl = os.path.join(RAW, f"p4_plans_{arm}.npz")
            if not os.path.exists(pl):
                log(f"REFUSED: {pl} missing")
                return 3
            for k in range(NSH):
                jobs.append((f"{arm}_s{k}", [NPY, os.path.join(HERE, "d6_p4_score.py"), "--plans", pl,
                                             "--out", os.path.join(sc, f"{arm}_s{k}.jsonl"), "--shard", f"{k}/{NSH}"]))
    log(f"stage {a.stage} {a.arms}: {len(jobs)} jobs, max_par {MAX_PAR}, gate {GATE_GB} GB")
    res = run_pool(jobs)
    tag = a.stage if a.stage == "base" else "arms_" + a.arms.replace(",", "_")
    json.dump({"stage": a.stage, "arms": a.arms, "worker_rc": res, "t": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(RAW, f"p4_cpu_{tag}_done.json"), "w"), indent=1)
    log(f"stage {tag} DONE {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
