"""In-process resource probe for training benchmarks on Jetson Thor (and pods).

WHY THIS EXISTS — the three probes that DO NOT work on Thor (MEASURED 2026-08-03,
this file's own reason to exist; each was verified before being ruled out):

  * ``torch.cuda.mem_get_info()`` reports **3.4 GB free of 131.9 GB** on an idle
    Thor and does NOT move while 60 GB is allocated and written. It is not a
    usable capacity signal on this platform.
  * ``free`` / ``tegrastats`` ``RAM x/y MB`` reports ~106 000/125 772 MB on an
    IDLE Thor with a total process RSS of 1.35 GB, and rose by only **596 MB**
    while 60 GB of CUDA tensors were allocated AND filled (checksum-verified,
    full-tensor sum). The host RAM counter does not track CUDA allocations.
  * ``/proc/<pid>/status`` ``VmRSS``/``VmHWM`` stayed at **0.62 GB** while
    ``torch.cuda.max_memory_allocated()`` read **24.01 GB** in the same process.
    An EXTERNAL memory probe therefore cannot see GPU memory on Thor, even
    though the memory is physically unified.

  ⇒ the ONLY probe that tracks GPU memory here is ``torch.cuda.max_memory_*``,
    and it must run **inside** the training process. Hence a probe module rather
    than an external sampler.

Usage — never edit the trainer, wrap it (see ``thor_bench_run.py``)::

    THOR_BENCH_OUT=/tmp/bench.jsonl python scripts/thor_bench_run.py \
        scripts/refc_train.py --data-root ... --out ...

The probe writes one JSON object per sample to ``$THOR_BENCH_OUT`` and a final
``*.summary.json`` at interpreter exit. It never touches the training loop, so a
benchmark cannot change the thing it measures beyond one background thread
sampling counters.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time

_INTERVAL_S = float(os.environ.get("THOR_BENCH_INTERVAL", "2.0"))
_OUT = os.environ.get("THOR_BENCH_OUT", "")
_state: dict = {"samples": 0, "peak_alloc_b": 0, "peak_reserved_b": 0,
                "peak_vmhwm_kb": 0, "t0": time.time()}


def _proc_kb(field: str) -> int:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(field):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _sample() -> dict | None:
    """One resource sample, or None while torch/CUDA is not up yet."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        alloc = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
    except Exception:
        return None
    vmhwm = _proc_kb("VmHWM")
    _state["peak_alloc_b"] = max(_state["peak_alloc_b"], alloc)
    _state["peak_reserved_b"] = max(_state["peak_reserved_b"], reserved)
    _state["peak_vmhwm_kb"] = max(_state["peak_vmhwm_kb"], vmhwm)
    _state["samples"] += 1
    return {"t": round(time.time() - _state["t0"], 2),
            "cuda_max_alloc_gb": round(alloc / 1e9, 3),
            "cuda_max_reserved_gb": round(reserved / 1e9, 3),
            "host_vmrss_gb": round(_proc_kb("VmRSS") / 1048576, 3),
            "host_vmhwm_gb": round(vmhwm / 1048576, 3)}


def _loop() -> None:
    fh = open(_OUT, "a", buffering=1) if _OUT else None
    while True:
        s = _sample()
        if s is not None and fh is not None:
            fh.write(json.dumps(s) + "\n")
        time.sleep(_INTERVAL_S)


def summary() -> dict:
    return {"peak_cuda_alloc_gb": round(_state["peak_alloc_b"] / 1e9, 3),
            "peak_cuda_reserved_gb": round(_state["peak_reserved_b"] / 1e9, 3),
            "peak_host_vmhwm_gb": round(_state["peak_vmhwm_kb"] / 1048576, 3),
            "n_samples": _state["samples"],
            "wall_s": round(time.time() - _state["t0"], 1),
            # Stated so a reader cannot mistake the host number for GPU memory:
            "note": ("peak_host_vmhwm_gb EXCLUDES CUDA memory on Thor "
                     "(unified memory is invisible to VmRSS — MEASURED)")}


def _write_summary() -> None:
    if not _OUT:
        return
    try:
        with open(_OUT + ".summary.json", "w") as fh:
            json.dump(summary(), fh, indent=2)
    except OSError:
        pass


def start() -> None:
    threading.Thread(target=_loop, daemon=True, name="thor-bench-probe").start()
    atexit.register(_write_summary)


start()
