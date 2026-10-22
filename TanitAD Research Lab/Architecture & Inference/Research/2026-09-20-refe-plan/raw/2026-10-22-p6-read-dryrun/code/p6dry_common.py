"""P6-READ-DRYRUN shared helpers: the dev-box environment the seam runs in, and the shared dev-box GPU lock.

Both are COPIES OF THE PROGRAMME'S OWN CONVENTIONS, not inventions:
  * env      = eval/eval_checkpoint.py `env_driverl()` (tip 1664143), set in-process so the script can run under `python -I`.
  * GPU lock = eval/aux_navtest_forward.py `acquire_gpu_lock` / `release_gpu_lock` (c3_run.py's protocol): O_EXCL create of
               C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock with a {"pid","owner","t"} body; refuse while
               devbox_gpu.chain_active exists; memory.used must read <= 3500 MiB after taking it; a FOREIGN lock is never removed.
    Unlike the seam wrappers this one NEVER WAITS: a held lock raises GpuBusy and the caller writes a HANDOFF.md and ends
    (token-budget rule 6a).
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time

PKX = os.environ.get("P6DRY_PKX", "C:/Users/Admin/par18/p6dry/pkx")      # eval/ and refe/ exported from mirror tip 1664143
DZ = "C:/Users/Admin/dz/DriveZero/DriveRL"
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
OWNER = "p6_read_dryrun"
GPU_MAX_USED_MIB = 3500


def setup_env() -> None:
    os.environ.update({
        "DZ_ROOT": DZ, "NUPLAN_DATA_ROOT": "D:/Projects/TanitAD/data/nuplan",
        "NUPLAN_MAPS_ROOT": "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0",
        "DRIVERL_EVAL_ROOT": DZ, "DRIVERL_EVAL_NUPLAN_ROOT": f"{DZ}/nuplan-devkit",
        "DRIVERL_EVAL_CONFIG_PATH": f"{DZ}/release/configs/driverl_teacher.yaml",
        "DRIVERL_EVAL_CHECKPOINT_PATH": f"{DZ}/release/checkpoints/checkpoint_2400.pt",
        "REFE_BACKBONE_ROOT": "D:/Projects/TanitAD/data/backbones", "REFE_SIM_HZ": "10",
        "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    # the seam does `sys.path.insert(0, <refe>)`; eval/ is reachable too so `refe_navtest_seam` imports unchanged
    for p in (f"{DZ}/src", f"{DZ}/nuplan-devkit", f"{PKX}/eval", f"{PKX}/refe"):
        if p not in sys.path:
            sys.path.insert(0, p)


class GpuBusy(RuntimeError):
    pass


def gpu_used_mib() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True)
    try:
        return int(r.stdout.decode("utf-8", "replace").strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def acquire_gpu_lock(settle_s: float = 4.0) -> str:
    if os.path.exists(MARK):
        raise GpuBusy(f"{MARK} exists: a GPU chain is active")
    body = json.dumps({"pid": os.getpid(), "owner": OWNER, "t": time.time()})
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            held = open(LOCK, encoding="utf-8").read()
        except OSError:
            held = "<unreadable>"
        raise GpuBusy(f"{LOCK} is held: {held[:300]}")
    os.write(fd, body.encode("utf-8"))
    os.close(fd)
    time.sleep(settle_s)
    used = gpu_used_mib()
    if not 0 <= used <= GPU_MAX_USED_MIB:
        release_gpu_lock(body)
        raise GpuBusy(f"memory.used {used} MiB > {GPU_MAX_USED_MIB} after taking the lock")
    return body


def release_gpu_lock(body: str) -> None:
    try:
        if open(LOCK, encoding="utf-8").read() == body:        # only the lock THIS call wrote
            os.remove(LOCK)
    except OSError:
        pass


@contextlib.contextmanager
def gpu_lock(enabled: bool = True):
    if not enabled:
        yield None
        return
    body = acquire_gpu_lock()
    try:
        yield body
    finally:
        release_gpu_lock(body)
