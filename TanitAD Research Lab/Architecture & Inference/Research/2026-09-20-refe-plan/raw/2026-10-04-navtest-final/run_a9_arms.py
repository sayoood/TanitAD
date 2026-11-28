"""Amendment 9 GPU arms: refe_navtest_seam.py (UNCHANGED apart from the registered --goal-fix flag) on each arm's token
set, under the shared dev-box GPU lock, with every D: path re-pointed to E: (the drive was re-lettered). One seam per arm.
    python run_a9_arms.py <arm>[,<arm>...]      arms: pdm_route navgoal_straight navgoal_arc a8_lane
"""
import json
import os
import socket
import subprocess
import sys
import time

PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, f"{PKG}/eval")
import eval_checkpoint as EC  # noqa: E402

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
D = "E:/Projects/TanitAD/data/refe_navtest"
OUT = f"{PKG}/raw/2026-10-01-goal-trigger/a9"
CKPT = "E:/Projects/TanitAD/data/refe_runs_eval/model_final.pt"
FLAGS = {"pdm_route": ["--goal-fix", "pdm_route"], "navgoal_straight": ["--goal-fix", "navgoal_straight"],
         "navgoal_arc": ["--goal-fix", "navgoal_arc"], "a8_lane": ["--sanitize-goal"]}


def fix(v):
    return v.replace("D:/", "E:/") if isinstance(v, str) else v


def take_lock(me):
    while True:
        if not os.path.exists(MARK):
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, me.encode()); os.close(fd)
                return
            except FileExistsError:
                pass
        try:
            held = open(LOCK, encoding="utf-8").read()[:100]
        except OSError:
            held = "(marker)"
        print(f"waiting for the GPU lock {time.strftime('%T')}: {held}", flush=True)
        time.sleep(60)


def release(me):
    try:
        if open(LOCK, encoding="utf-8").read() == me:
            os.remove(LOCK); print(f"GPU lock released {time.strftime('%T')}", flush=True)
    except OSError:
        pass


def main():
    arms = sys.argv[1].split(",")
    os.makedirs(OUT, exist_ok=True)
    me = json.dumps({"job": "refe-a9-goalfix-arms", "pid": os.getpid(), "host": socket.gethostname(),
                     "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "arms": arms})
    take_lock(me)
    print(f"lock taken {time.strftime('%F %T')}", flush=True)
    env = {k: fix(v) for k, v in EC.env_driverl().items()}
    rc_all = 0
    try:
        for arm in arms:
            toks = f"{OUT}/tokens_{arm}.json"
            cmd = [EC.DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", CKPT, "--frames", f"{D}/frames",
                   "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test",
                   "--export", "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz",
                   "--tokens", toks, "--out", f"{OUT}/seam_{arm}.npz", "--arm", f"REFe_a9_{arm}",
                   "--record-inputs", f"{OUT}/inputs_{arm}.json"] + FLAGS[arm]
            t0 = time.time()
            with open(f"{OUT}/seam_{arm}.log", "w", encoding="utf-8") as f:
                rc = subprocess.call(cmd, cwd=f"{PKG}/eval", env=env, stdout=f, stderr=subprocess.STDOUT)
            tail = open(f"{OUT}/seam_{arm}.log", encoding="utf-8", errors="replace").read().strip().splitlines()[-1:]
            print(f"arm {arm}: rc {rc} {time.time() - t0:.0f} s  {tail}", flush=True)
            rc_all |= rc
    finally:
        release(me)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
