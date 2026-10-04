"""D6 P1 + P2 GPU chain -- runs ONLY as the child of the battery's ``with_gpu_lock.py`` (the dev-box GPU lock is then HELD by the wrapper).

    <tanitad python> with_gpu_lock.py --job refcv8-d6-p1p2 --log <log> --rec <rec> --max-wait-s 43200 -- <tanitad python> run_p1p2_chain.py

Stages (SPEC_P1P2.md s1, s5, s7), each a ``bridge_fix/run_bridge7.py`` subprocess with EXACTLY the arguments of the banked step-30,000 navhard
bridge run (``navsim/raw/milestones/step30000/bridge_navhard.log`` line 1) plus the opt-in D6 flags:
  P1    arm R7_A1, tokens spec_tokens_P1_all (2,403), --export-fan raw/p1_fan                    -> raw/p1_bridge/
  P2    arms R7_NAVOFF,R7_NAVFOLLOW,R7_NAVFLIP, tokens spec_tokens_P2_all (2,312)                 -> raw/p2_bridge/
  K7    arm R7_A1, tokens spec_tokens_P2_K1 (300)                                                 -> raw/p2_bridge_k7/
A stage whose output rows are already complete is skipped (the bridge itself resumes row by row).  Writes raw/p1p2_gpu_done.json at the end.
Never touches the lock: the wrapper acquired it and releases it.  Refuses to run if the lock file does not name this job.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKGD6 = os.path.dirname(HERE)
RAW = os.path.join(PKGD6, "raw")
JOB = "refcv8-d6-p1p2"
LOCK = os.environ.get("REFCV7_GPU_LOCK", "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREE = os.environ.get("TANITAD_REPO", "C:/Users/Admin/ev7nav")
INC = f"{TREE}/FlyWheels/TanitAD_EvalFlyWheel/incoming"
R6IN = f"{INC}/2026-09-23-refcv6-standard-tests/navsim/raw/inputs"
BANKS = "C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923"
CKPT = "D:/refcv7_eval_kit/ckpt/ckpt_30000.pt"
MD5 = "ac4e4fab87e35b20de24d8d94910d910"
COMMON = ["--split", "navhard_two_stage",
          "--inputs", "C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json",
          "--speed", f"{R6IN}/speed_limits_navhard_two_stage.json",
          "--road-plane", f"{R6IN}/road_plane_navhard_warmup_logs.json",
          "--ckpt", CKPT, "--config", "D:/refcv7_eval_kit/ckpt/config.json", "--ckpt-md5", MD5,
          "--device", "cuda", "--precision", "auto", "--threads", "6", "--label", "D6P1P2",
          "--bank2", f"{BANKS}/navhard_s2", "--bank1", f"{BANKS}/navhard_s1"]


def lock_pid() -> int:
    """The pid the lock records, IF the lock names this job AND that pid is an ANCESTOR of this process (the with_gpu_lock wrapper); else -1.
    (A venv python.exe is a launcher that spawns the real interpreter, so `os.getppid()` is NOT the wrapper -- walk the ancestors.)"""
    import psutil
    try:
        cur = json.load(open(LOCK, encoding="utf-8"))
    except Exception:                                               # noqa: BLE001
        return -1
    if cur.get("job") != JOB:
        return -1
    anc = {p.pid for p in psutil.Process().parents()}
    return int(cur["pid"]) if int(cur.get("pid", -1)) in anc else -1


def lock_is_ours() -> bool:
    return lock_pid() > 0


def tokens_json(name: str) -> str:
    toks = [l.strip() for l in open(os.path.join(RAW, f"spec_tokens_{name}.txt")) if l.strip()]
    p = os.path.join(RAW, f"_tokens_{name}.json")
    json.dump({"tokens": toks}, open(p, "w"))
    return p, len(toks)


def rows_done(out: str, arm: str, n: int) -> bool:
    p = os.path.join(out, f"rows_{arm}.jsonl")
    if not os.path.exists(p):
        return False
    k = 0
    for ln in open(p, encoding="utf-8"):
        try:
            json.loads(ln)
            k += 1
        except Exception:                                           # noqa: BLE001
            pass
    return k >= n


def stage(name: str, arms: str, tokname: str, out: str, export: str, log) -> dict:
    tf, n = tokens_json(tokname)
    arm_list = arms.split(",")
    if os.environ.get("D6_CHAIN_DRYRUN"):
        cmd = [PY, os.path.join(HERE, "bridge_fix", "run_bridge7.py"), "--arms", arms, "--tokens-file", tf, "--out", out,
               "--gpu-lock-job", JOB, "--gpu-lock-pid", str(lock_pid())] + COMMON + (["--export-fan", export] if export else [])
        log("DRYRUN " + " ".join(cmd))
        return {"name": name, "dryrun": True, "n_tokens": n}
    if all(rows_done(out, a, n) for a in arm_list) and (not export or rows_done(out, arm_list[0], n)):
        log(f"stage {name}: rows complete -> skipped")
        return {"name": name, "skipped": True}
    cmd = [PY, os.path.join(HERE, "bridge_fix", "run_bridge7.py"), "--arms", arms, "--tokens-file", tf, "--out", out,
           "--gpu-lock-job", JOB, "--gpu-lock-pid", str(lock_pid())] + COMMON
    if export:
        cmd += ["--export-fan", export]
    env = dict(os.environ)
    env.update({"PYTHONPATH": f"{TREE}/stack;{TREE}/taniteval", "TANITAD_REPO": TREE, "OMP_NUM_THREADS": "6", "PYTHONIOENCODING": "utf-8"})
    t0 = time.time()
    with open(os.path.join(RAW, f"chain_{name}.log"), "a", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return {"name": name, "rc": rc, "wall_s": round(time.time() - t0, 1),
            "rows": {a: rows_done(out, a, n) for a in arm_list}, "n_expected": n}


def main() -> int:
    def log(m):
        with open(os.path.join(RAW, "chain.log"), "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + m + "\n")
    if not lock_is_ours():
        log("REFUSED: the GPU lock does not name this job / this wrapper")
        return 3
    log(f"START under the lock job={JOB} wrapper_pid={lock_pid()}")
    res = []
    res.append(stage("P1", "R7_A1", "P1_all", os.path.join(RAW, "p1_bridge"), os.path.join(RAW, "p1_fan"), log))
    if not lock_is_ours():
        log("lock lost after P1 -> stop")
        json.dump(res, open(os.path.join(RAW, "p1p2_gpu_done.json"), "w"), indent=1)
        return 4
    res.append(stage("P2", "R7_NAVOFF,R7_NAVFOLLOW,R7_NAVFLIP", "P2_all", os.path.join(RAW, "p2_bridge"), "", log))
    res.append(stage("K7", "R7_A1", "P2_K1", os.path.join(RAW, "p2_bridge_k7"), "", log))
    if os.environ.get("D6_CHAIN_DRYRUN"):
        log("DRYRUN DONE (no done-marker written) " + json.dumps(res))
        return 0
    json.dump(res, open(os.path.join(RAW, "p1p2_gpu_done.json"), "w"), indent=1)
    log("DONE " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
