"""Wait for the boxes-video render to finish (BOXES_DONE marker, or no render process left), then for
>= 7.5 GB free RAM on 3 samples 30 s apart, then launch the task-3 probe ONCE (CPU). Sleeps only; no
compute of its own. Gives up after --max-hours. Everything it decides is appended to raw/wait_then_probe.log.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
LOG = HERE.parent / "raw" / "wait_then_probe.log"
MARKERS = ["D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-boxes-video/**/BOXES_DONE*",
           "C:/Users/Admin/qland/work/mapvid/**/BOXES_DONE*",
           "C:/Users/Admin/qland/**/BOXES_DONE*"]


def log(s: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gb() -> float:
    return psutil.virtual_memory().available / 2 ** 30


def render_running() -> list:
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
        except Exception:
            continue
        if "render_refcv6_map_video" in cl and "--boxes" in cl:
            out.append(p.info["pid"])
    return out


def marker() -> list:
    hits = []
    for g in MARKERS:
        hits += glob.glob(g, recursive=True)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=6.0)
    a = ap.parse_args()
    t_end = time.time() + a.max_hours * 3600
    log("[wait] start: waiting for BOXES_DONE (or no boxes render process), then the RAM gate")
    last = 0.0
    go = HERE.parent / "raw" / "GO_PROBE"          # a manual release, written only on the Master Mind's word
    while time.time() < t_end:
        m, r = marker(), render_running()
        # ⛔ the MARKER only. "No render process" is NOT completion: at 22:33 the smoke had exited and the
        # full render had not started yet (it was waiting for RAM) -- the first version of this waiter
        # read that gap as "finished" and would have competed with it for RAM (stopped by PID 22:35).
        if m or go.exists():
            log(f"[wait] release: markers {[os.path.basename(x) for x in m][:3]}, GO_PROBE {go.exists()}, "
                f"render pids {r}")
            break
        if time.time() - last > 600:
            log(f"[wait] render still running (pids {r}); free {free_gb():.2f} GB")
            last = time.time()
        time.sleep(60)
    else:
        log("[wait] gave up waiting for the render")
        sys.exit(2)
    while time.time() < t_end:
        s = []
        for k in range(3):
            s.append(round(free_gb(), 2))
            if s[-1] < 7.5:
                break
            if k < 2:
                time.sleep(30)
        if len(s) == 3 and min(s) >= 7.5:
            log(f"[wait] RAM gate open: {s} GB; launching the probe")
            break
        if time.time() - last > 600:
            log(f"[wait] RAM gate closed: {s} GB; retrying every 60 s")
            last = time.time()
        time.sleep(60)
    else:
        log("[wait] gave up waiting for RAM")
        sys.exit(3)
    env = dict(os.environ)
    # v3 (23:15): the GPU when the dev-box GPU gate passes (used < 4300 MiB, no other python compute
    # process on it) -- the as-trained bf16 trunk, ~1 min instead of ~5 min of CPU exposure; CPU fp32
    # otherwise. The gate is the battery's rule (gpu_gate.py), read here from nvidia-smi directly.
    device = "cpu"
    try:
        q = lambda a: subprocess.run(a, capture_output=True, text=True, timeout=60).stdout
        used = int(q(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                   .strip().splitlines()[0])
        apps = q(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
        py = [ln for ln in apps.splitlines() if "python" in ln.lower()]
        if used < 4300 and not py:
            device = "cuda"
        log(f"[wait] GPU gate: used {used} MiB, python compute apps {len(py)} -> device {device}")
    except Exception as e:                                   # noqa: BLE001
        log(f"[wait] GPU gate unreadable ({type(e).__name__}); device cpu")
    rc = subprocess.call([sys.executable, str(HERE / "measure_class_signal_38k.py"), "--device", device,
                          "--threads", "4"], env=env)
    log(f"[wait] probe exited with {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
