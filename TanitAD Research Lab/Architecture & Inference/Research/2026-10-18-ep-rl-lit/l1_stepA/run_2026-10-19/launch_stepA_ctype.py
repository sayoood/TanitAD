#!/usr/bin/env python3
"""Lock-respecting launcher for Step A's ONE GPU phase (`make_stepA_bundle.py`; every other phase is CPU and needs no lock).  The pattern is `hn34/work/gpu_locked.py` (Master Mind 2026-10-18),
hardened with the TRAPS entry "TWO LOCK SCHEMAS SHARE devbox_gpu.lock":

  * waits while `devbox_gpu.chain_active` exists (another chain owns the GPU between its jobs);
  * takes `devbox_gpu.lock` with O_EXCL and OUR owner marker (`l1_stepA`); an existing lock is NEVER removed, renamed or edited when it is a FOREIGN schema or another owner's
    ({"job", "pid" = an MSYS pid ...} is invisible to tasklist, so a dead-pid test would steal a live job's lock) -- the one exception is a lock carrying OUR owner marker whose pid
    is gone and which is older than 10 minutes (a launcher that died);
  * after taking it, requires NO compute-only (type C) process in the nvidia-smi process table and memory.used <= 6000 MiB (an unlocked GPU user means: give the lock back and wait).
    PATCHED COPY (PI decision 2026-10-19, "Allow compute-type check"; the landed launch_stepA.py stays frozen): the landed rule memory.used <= 1500 MiB can NEVER pass on this box, because the
    desktop's C+G graphics apps (Obsidian, VS Code, browsers, the screensaver) hold ~2.3 GB with no compute job. Reference: hn34/work/gpu_locked.py (compute_pids). An unreadable
    nvidia-smi fails CLOSED (compute_pids -> [-1]);
  * releases only the lock whose bytes are EXACTLY the body this process wrote;
  * (patched copy) also waits while psutil available RAM < 8 GB (handoff start condition 3), BEFORE taking the lock;
  * exit 75 = the GPU never became free within --max-wait and the command was NOT run.

    launch_stepA.py probe                                   # read-only: chain_active / lock holder / memory.used; takes nothing
    launch_stepA.py run --max-wait 7200 -- <cmd ...>        # the GPU phase
    launch_stepA.py plan --work-dir D --tree T              # prints the ordered commands of the whole package; runs nothing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
OWNER = "l1_stepA"
MIN_AVAIL_GB = 8.0                                                       # handoff start condition 3 (psutil available RAM); checked BEFORE the lock is taken, so no lock is held while RAM-starved
MAX_USED_MIB = 6000                                                      # backstop only (landed: 1500); the decisive test is compute_pids()
OWN_STALE_S = 600.0
SETTLE_S = 3.0
EX_BUSY = 75


def used_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, timeout=30)
        return int(r.stdout.decode("utf-8", "replace").strip().splitlines()[0])
    except Exception:
        return 10 ** 9                                                   # fail CLOSED: an unreadable GPU is a busy GPU


def avail_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return 0.0                                                       # fail CLOSED: an unreadable RAM probe is a starved box (the perf-counter path is LOCALISED here; psutil is the probe W3 uses)


def compute_pids(text=None):
    """pids of type-C (compute-only) processes in the nvidia-smi process table; C+G desktop apps are not counted.  Unreadable nvidia-smi -> [-1] (fail CLOSED)."""
    if text is None:
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=30)
            text = r.stdout.decode("utf-8", "replace")
        except Exception:
            return [-1]
    out = []
    for line in text.splitlines():
        t = line.strip().strip("|").split()
        if len(t) >= 6 and t[0].isdigit() and t[3].isdigit() and t[4] == "C":
            out.append(int(t[3]))
    return out


def pid_alive(pid) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)       # bytes: the German "no tasks" line carries 0x81
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def read_holder(lock, now_fn=time.time):
    try:
        return json.loads(open(lock, encoding="utf-8").read()), now_fn() - os.path.getmtime(lock)
    except (OSError, ValueError):
        return None, 0.0


def try_lock(lock=LOCK, mark=MARK, used_fn=used_mib, alive_fn=pid_alive, sleep_fn=time.sleep, now_fn=time.time, compute_fn=None):
    """-> (body, 'ok') or (None, reason).  Never touches a lock that is not ours."""
    if os.path.exists(mark):
        return None, "chain_active present"
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    for attempt in (1, 2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            holder, age = read_holder(lock, now_fn)
            if not isinstance(holder, dict) or holder.get("owner") != OWNER:
                who = (holder.get("job") or holder.get("owner") or "?") if isinstance(holder, dict) else "unreadable"
                return None, f"lock held by a FOREIGN holder ({who}); never removed"
            if "pid" not in holder or alive_fn(holder["pid"]):
                return None, "lock held by a live l1_stepA process"
            if attempt == 2 or age <= OWN_STALE_S:
                return None, f"own lock, holder pid gone but only {age:.0f} s old"
            os.remove(lock)                                              # OUR marker, dead pid, > 10 min: a launcher that died
    else:
        return None, "lock not acquired"
    body = json.dumps({"pid": os.getpid(), "owner": OWNER, "job": "l1-stepA-bundle", "t": now_fn()})
    os.write(fd, body.encode("utf-8"))
    os.close(fd)
    sleep_fn(SETTLE_S)
    cp = (compute_fn or compute_pids)()                                  # late-bound: a test or a mutation can replace the module's compute_pids
    if cp:
        release(body, lock)
        return None, f"compute process(es) on the GPU {cp} after taking the lock"
    u = used_fn()
    if u > MAX_USED_MIB:
        release(body, lock)
        return None, f"memory.used {u} MiB after taking the lock"
    return body, "ok"


def release(body, lock=LOCK):
    """remove the lock ONLY if its bytes are exactly the body we wrote"""
    try:
        if open(lock, encoding="utf-8").read() == body:
            os.remove(lock)
            return True
    except OSError:
        pass
    return False


def child_env(base=None) -> dict:
    """the child's environment: GPU visible again, and the marker that tells stepA.py / make_stepA_bundle.py they were started under the lock"""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k != "CUDA_VISIBLE_DEVICES"}
    env["L1_GPU_LOCKED"] = "1"
    return env


def run_locked(cmd, max_wait, lock=LOCK, mark=MARK, used_fn=used_mib, alive_fn=pid_alive, sleep_fn=time.sleep, now_fn=time.time, run_fn=None, poll_s=60.0, log=print, compute_fn=None, avail_fn=None, min_avail_gb=None):
    need = MIN_AVAIL_GB if min_avail_gb is None else min_avail_gb
    run_fn = run_fn or (lambda c: subprocess.run(c, env=child_env()).returncode)
    t0, last = now_fn(), ""
    while True:
        ram = (avail_fn or avail_gb)()
        if ram < need:
            body, why = None, f"RAM available {int(ram)} GB < {need:g} GB (lock not taken)"
        else:
            body, why = try_lock(lock, mark, used_fn, alive_fn, sleep_fn, now_fn, compute_fn)
        if body:
            break
        if why != last:
            log(f"ZZGPUWAIT {why}")
            last = why
        if now_fn() - t0 > max_wait:
            log(f"ZZGPU_NOT_RUN waited {max_wait:.0f} s: {why}")
            return EX_BUSY
        sleep_fn(poll_s)
    log("ZZGPULOCKED")
    try:
        rc = run_fn(cmd)
    finally:
        done = release(body, lock)
        log(f"ZZGPURELEASED {'yes' if done else 'NOT (the lock body changed; left untouched)'}")
    return rc


def probe(lock=LOCK, mark=MARK, used_fn=used_mib, compute_fn=None) -> dict:
    holder, age = read_holder(lock) if os.path.exists(lock) else (None, 0.0)
    cp = (compute_fn or compute_pids)()
    return {"chain_active": os.path.exists(mark), "lock_exists": os.path.exists(lock), "lock_holder": holder, "lock_age_s": round(age), "memory_used_mib": used_fn(), "max_used_mib": MAX_USED_MIB,
            "compute_pids": cp, "would_launch": bool(not os.path.exists(mark) and not os.path.exists(lock) and not cp and used_fn() <= MAX_USED_MIB)}


def plan(work_dir, tree, py_eval="C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe", py_cpu="C:/Users/Admin/venvs/tanitad/Scripts/python.exe", here=None) -> list:
    """the ordered, copy-pasteable command list (prints, runs nothing).  GPU is needed ONLY by step 3."""
    h = (here or os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")           # C:/ paths for Python on this box
    w = work_dir
    t = "C:/Users/Admin/p2heldout/plan/heldout_tokens.json"
    tab = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz"
    ck = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
    cpu = "CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4"
    return [
        f"# 0  adopt the amendment, copy PREREG + amendment unchanged into {w}, write their sha256 beside them (nothing below runs before that)",
        f"# 1  CPU  {cpu} {py_cpu} \"{h}/stepA.py\" sample --table {tab} --out {w}/sample.json",
        f"# 2  CPU  {cpu} {py_cpu} \"{h}/g0_overlap.py\" ... (G0: re-run on the adopted sample; both manifests are in the amendment)",
        f"# 3  GPU  {py_cpu} \"{h}/launch_stepA_ctype.py\" run --max-wait 7200 -- {py_eval} \"{h}/make_stepA_bundle.py\" --ckpt {ck} --plan {t} --sample {w}/sample.json --table {tab} --out-dir {w}/bundle",
        f"# 4  CPU  {cpu} {py_cpu} \"{h}/stepA.py\" perturb --bundle {w}/bundle/bundle.npz --out {w}/perts.npz",
        f"# 5  CPU  (the navsim env and wrapper of p2ho_score.py, see its docstring; one worker) label_perturbations.py --bundle {w}/bundle/bundle.npz --perts {w}/perts.npz --plan {t} --table {tab} --out {w}/labels.npz --g1 20 --tree \"{tree}\"",
        f"# 6  CPU  {cpu} {py_cpu} \"{h}/stepA.py\" score --bundle {w}/bundle/bundle.npz --perts {w}/perts.npz --ckpt {ck} --ctx {w}/bundle/ctx.npy --device cpu --out {w}/scores.npz",
        f"# 7  CPU  {cpu} {py_cpu} \"{h}/stepA.py\" analyse --bundle {w}/bundle/bundle.npz --perts {w}/perts.npz --scores {w}/scores.npz --labels {w}/labels.npz --out {w}/RESULT.json",
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("probe")
    r = sp.add_parser("run")
    r.add_argument("--max-wait", type=float, required=True)
    r.add_argument("--min-avail-gb", type=float, default=MIN_AVAIL_GB)
    r.add_argument("--lock", default=LOCK)
    r.add_argument("--mark", default=MARK)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    p = sp.add_parser("plan")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--tree", default="C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        print(json.dumps(probe(), indent=1))
        return 0
    if a.cmd == "plan":
        print("\n".join(plan(a.work_dir, a.tree)))
        return 0
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        print("no command after --", file=sys.stderr)
        return 2
    return run_locked(cmd, a.max_wait, a.lock, a.mark, min_avail_gb=a.min_avail_gb)


if __name__ == "__main__":
    sys.exit(main())
