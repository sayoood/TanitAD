"""YAXN (P2 AXNOISE) shared constants and helpers. Read-only imports of YAX (ax_*) and YAXD (axd_*); nothing outside YAXN is written.

PREREG_AXD.md section 6 (frozen sha256 1d33a9ed...) is the design; IMPL_P2.md (own sha256) fixes the operational reading of its ambiguous words BEFORE any AXNOISE arm is evaluated.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAXD = HERE.parent / "YAXD"
sys.path.insert(0, str(YAXD))
import axd_common as A  # noqa: E402  (YAXD; inserts YAX on sys.path, exposes A.C = YAX ax_common)

C = A.C
YAX = A.YAX
RUN_BASE = HERE / "run"
MARKERS = HERE / "markers"
PY = C.PY_D
SIGMA, NOISE_SEED_OFFSET, SEEDS = 1.0, 7919, (0, 1)
RECIPE = {"name": "K1", "steps": 200, "lr": "5e-5"}                       # YAX/run_full/stages/select.json chosen_X1 == "K1" (checked by the pins stage)
SMOKE_TRAIN = ["--max-frames", "400", "--batch", "32", "--accum", "4", "--gate-frames", "64", "--val-every", "2"]
MINI_TRAIN = ["--max-frames", "400", "--batch", "16", "--accum", "2", "--gate-frames", "64", "--val-every", "5"]       # YAX g8_mini settings
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "10", "MKL_NUM_THREADS": "10", "PYTHONIOENCODING": "utf-8"}
P1_JSON = Path("C:/Users/Admin/yaxd_run/full/analysis/P1.json")           # u400's IDT gain on the same 923 tokens (MEASURED, banked by YAXD)
YAX_EVAL = YAX / "run_full"                                               # the control: X1_AXON_s0 / X1_AXON_s1 (same recipe K1, seed, permutation)
CONTROL = {s: f"X1_AXON_s{s}" for s in SEEDS}
AXON_FINAL = {s: YAX / "run_full" / "arms" / f"K1_AXON_s{s}" / "final.pt" for s in SEEDS}      # AMENDMENT_1: decoder tensors the X1 AXON evals were composed from (chosen_X1 = K1)
MEM_SIGS = ("MemoryError", "ArrayMemoryError", "Unable to allocate", "not enough memory", "paging file", "DefaultCPUAllocator", "bad_alloc", "CUDA out of memory")
CODE = ["axd_ft.py", "axn_common.py", "axn_seam.py", "axn_analyze.py", "axn_chain.py", "axn_run_all.py"]
PINNED_DOCS = ["IMPL_P2.md", "AMENDMENT_1.md"]
MIN_FREE_E_GIB = 4.0
TIER = ("TIER: NAVSIM v1.1 PDMS + families6 of an OPEN-LOOP plan against logged agents on the 923 A5 tokens (T0-family) -- NOT the action-closed-loop tier. Open loop has no ax feedback, so it "
        "shows what AXNOISE COSTS (against the same-recipe, same-seed AXON control) and how far it cuts the plan's ax GAIN; it is never proof that the stall is fixed. Closed-loop ADOPT = PREREG_AXOFF 6c, run by MM.")
VARIANCE = ("Q1 another draw of LOGS: paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927 (ax_stats.Boot). Q2 another TRAINING run: TWO seeds, both must pass; the seed floor "
            "F = |value(s0) - value(s1)| is reported (H-ESTIM-SEED-1). Q3 another INFERENCE run: deterministic planner; G-IDT (fresh-trunk identity arm == the seam's own call, bit for bit) measures it.")
TACTICAL = "TACTICAL: n/a as a test -- REFe has no tactical layer; an ax change moves no decision head (families6 decisions derive from executed geometry; not gated)."
STRATEGIC = "STRATEGIC: PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE.md item 25) -- not a pass, not omitted."


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def free_gib(path) -> float:
    import shutil
    return shutil.disk_usage(str(path)).free / 2 ** 30


def mem_mb() -> tuple[int, int]:
    """(free physical MB, free virtual MB) from Win32_OperatingSystem; (0, 0) if the probe fails."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "$o=Get-CimInstance Win32_OperatingSystem; \"$([int]($o.FreePhysicalMemory/1024)) $([int]($o.FreeVirtualMemory/1024))\""],
                       capture_output=True, text=True)
    try:
        a, b = r.stdout.split()
        return int(a), int(b)
    except ValueError:
        return 0, 0


def wait_mem(log, min_phys=3500, min_virt=5500, max_wait_s=1200, poll_s=30) -> dict:
    """Bounded wait for host-memory headroom (the box is shared: pytest ~9 GB, run_voe1b). Never blocks forever: after max_wait_s it proceeds and says so."""
    t0, n = time.time(), 0
    while True:
        p, v = mem_mb()
        if p >= min_phys and v >= min_virt:
            return {"waited_s": round(time.time() - t0), "free_phys_mb": p, "free_virt_mb": v, "timed_out": False}
        if time.time() - t0 > max_wait_s:
            log(f"memory headroom not reached in {max_wait_s} s (phys {p} MB, virt {v} MB): proceeding, the retry loop covers a MemoryError")
            return {"waited_s": round(time.time() - t0), "free_phys_mb": p, "free_virt_mb": v, "timed_out": True}
        if n % 10 == 0:
            log(f"waiting for memory headroom: phys {p} MB (need {min_phys}), virt {v} MB (need {min_virt})")
        n += 1
        time.sleep(poll_s)


def require_disk(path, log, min_gib=MIN_FREE_E_GIB, max_wait_s=3600):
    t0 = time.time()
    while free_gib(path) < min_gib:
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f"REFUSING: {path} has {free_gib(path):.1f} GiB free, below {min_gib} GiB")
        log(f"{path}: {free_gib(path):.1f} GiB free < {min_gib}; waiting")
        time.sleep(60)


def sh(argv, log_path, env=None, cwd=None, timeout=None) -> tuple[int, str]:
    e = dict(os.environ)
    e.update(env or {})
    with open(log_path, "wb") as lf:
        rc = subprocess.run([str(x) for x in argv], env=e, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=timeout).returncode
    return rc, tail_text(log_path)


def tail_text(path, n=24000) -> str:
    try:
        return Path(path).read_bytes()[-n:].decode("utf-8", "replace")
    except OSError:
        return ""


def is_mem_failure(text: str) -> bool:
    return any(s in text[-6000:] for s in MEM_SIGS)


def jdump(obj, path):
    C.jdump(obj, path)


def jload(path):
    return C.jload(path)


def sha256_file(p) -> str:
    return C.sha256_file(p)


def u400_gain() -> dict:
    """u400's IDT gain block from the banked P1 run: {down/up/sym: {point, ci95}}, same 923 tokens, same instrument, same bootstrap seed."""
    return jload(P1_JSON)["gain_IDT_u400"]


# composed checkpoints are deleted ONLY through these literal paths (no variable-built path ever reaches os.remove)
CKPT_LITERAL = {
    ("smoke", "AXN", 0): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/smoke/ckpt/AXN_s0.pt",
    ("smoke", "AXN", 1): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/smoke/ckpt/AXN_s1.pt",
    ("smoke", "AXON", 0): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/smoke/ckpt/AXON_s0.pt",
    ("smoke", "AXON", 1): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/smoke/ckpt/AXON_s1.pt",
    ("full", "AXN", 0): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/full/ckpt/AXN_s0.pt",
    ("full", "AXN", 1): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/full/ckpt/AXN_s1.pt",
    ("full", "AXON", 0): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/full/ckpt/AXON_s0.pt",
    ("full", "AXON", 1): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/full/ckpt/AXON_s1.pt",
    ("unit", "AXN", 0): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/unit/ckpt/AXN_s0.pt",
    ("unit", "AXN", 1): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/unit/ckpt/AXN_s1.pt",
    ("unit", "AXON", 0): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/unit/ckpt/AXON_s0.pt",
    ("unit", "AXON", 1): "E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/run/unit/ckpt/AXON_s1.pt",
}


def remove_ckpt(tag, kind, s) -> bool:
    p = CKPT_LITERAL[(tag, kind, s)]
    if os.path.exists(p):
        os.remove(p)
        return True
    return False
