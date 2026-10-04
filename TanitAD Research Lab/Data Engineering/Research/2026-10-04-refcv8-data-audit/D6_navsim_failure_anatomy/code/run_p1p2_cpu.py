"""D6 P1 + P2 CPU stage -- waits for raw/p1p2_gpu_done.json (written by run_p1p2_chain.py) and then scores on the CPU.   tanitad venv launcher; the
scoring subprocesses run in the NAVSIM venv.  At most 3 scoring processes at once, each launched only while free RAM >= 3.0 GB.  Never touches the GPU.

  P1: d6_fan_score.py x3 shards over raw/p1_fan/fan_R7_A1.jsonl (Tier A+B for all 181 + STOP; Tier C for the best-ranked clean candidate; exact controls
      K1/K3 on spec_tokens_P1_K1ctrl.txt = Cpass u K1fail) -> raw/p1_scored_s{0,1,2}.jsonl ; K4 mutation (Cpass +30 m lateral) -> raw/p1_scored_K4.jsonl
  P2: per arm (R7_NAVOFF, R7_NAVFOLLOW, R7_NAVFLIP, and K7 = R7_A1 on P2_K1): hooks from rows (d6_make_arm_hooks.py) then d6_rescore.py x3 shards ->
      raw/p2_rescore_<arm>_s{0,1,2}.jsonl
Writes raw/p1p2_cpu_done.json.  Resumable (every scorer skips finished scenes).
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")
NPY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
TPY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
M = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step30000"
A1H = f"{M}/scores_navhard/score_R7_A1__navhard_two_stage_wrapper/R7_A1__navhard_two_stage_hooks.json"
ENV = dict(os.environ, PYTHONPATH="C:/Users/Admin/navsim-crun/devkit;C:/Users/Admin/navsim-crun/nuplan-devkit", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
GATE_GB = 3.0


def free_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return min(m.ullAvailPhys, m.ullAvailPageFile) / 2 ** 30      # commit, not just RAM: the dev box commit limit binds


def log(msg):
    with open(os.path.join(RAW, "cpu_chain.log"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")


def run_pool(jobs, max_par=3):
    """jobs: list of (name, cmd). Launch while RAM allows; wait for all."""
    live, q = [], list(jobs)
    while q or live:
        for j in list(live):
            if j[2].poll() is not None:
                log(f"ended {j[0]} rc={j[2].returncode}")
                live.remove(j)
        if q and len(live) < max_par and free_gb() >= GATE_GB:
            name, cmd = q.pop(0)
            fh = open(os.path.join(RAW, f"cpu_{name}.log"), "a", encoding="utf-8")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=ENV)
            live.append((name, cmd, p))
            log(f"launched {name} (free {free_gb():.1f} GB)")
        time.sleep(5)


def split_tokens(path, k):
    toks = [l.strip() for l in open(path) if l.strip()]
    return toks


def main() -> int:
    done_f = os.path.join(RAW, "p1p2_gpu_done.json")
    log("waiting for the GPU chain")
    while not os.path.exists(done_f):
        time.sleep(60)
    log("GPU chain done: " + open(done_f).read().replace("\n", " ")[:300])
    # ---- P1
    fan = os.path.join(RAW, "p1_fan", "fan_R7_A1.jsonl")
    ctrl = os.path.join(RAW, "spec_tokens_P1_K1ctrl.txt")
    cp = [l.strip() for l in open(os.path.join(RAW, "spec_tokens_Cpass.txt")) if l.strip()]
    kf = [l.strip() for l in open(os.path.join(RAW, "spec_tokens_P1_K1_fail.txt")) if l.strip()]
    open(ctrl, "w").write("\n".join(sorted(set(cp) | set(kf))) + "\n")
    jobs = []
    for k in range(3):
        jobs.append((f"p1_s{k}", [NPY, os.path.join(HERE, "d6_fan_score.py"), "--fan", fan, "--hooks", A1H, "--out", os.path.join(RAW, f"p1_scored_s{k}.jsonl"),
                                  "--shard", f"{k}/3", "--exact-ids", ctrl, "--k5", "40"]))
    jobs.append(("p1_K4", [NPY, os.path.join(HERE, "d6_fan_score.py"), "--fan", fan, "--hooks", A1H, "--out", os.path.join(RAW, "p1_scored_K4.jsonl"),
                           "--only-tokens", os.path.join(RAW, "spec_tokens_Cpass.txt"), "--lateral-shift", "30.0", "--no-tierc"]))
    run_pool(jobs)
    # ---- P2
    arms = [("R7_NAVOFF", "p2_bridge", "P2_all"), ("R7_NAVFOLLOW", "p2_bridge", "P2_all"), ("R7_NAVFLIP", "p2_bridge", "P2_all"), ("R7_A1", "p2_bridge_k7", "P2_K1")]
    jobs = []
    for arm, d, tokname in arms:
        hk = os.path.join(RAW, f"p2_hooks_{arm}.json")
        subprocess.run([TPY, os.path.join(HERE, "d6_make_arm_hooks.py"), A1H, os.path.join(RAW, d, f"rows_{arm}.jsonl"), hk], env=ENV, check=False)
        toks = [l.strip() for l in open(os.path.join(RAW, f"spec_tokens_{tokname}.txt")) if l.strip()]
        for k in range(3):
            tf = os.path.join(RAW, f"_p2_tokens_{arm}_{k}.txt")
            open(tf, "w").write("\n".join(t for i, t in enumerate(toks) if i % 3 == k) + "\n")
            jobs.append((f"p2_{arm}_s{k}", [NPY, os.path.join(HERE, "d6_rescore.py"), "--hooks", hk, "--tokens", tf, "--out", os.path.join(RAW, f"p2_rescore_{arm}_s{k}.jsonl")]))
    run_pool(jobs)
    json.dump({"done": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(RAW, "p1p2_cpu_done.json"), "w"))
    log("CPU stage DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
