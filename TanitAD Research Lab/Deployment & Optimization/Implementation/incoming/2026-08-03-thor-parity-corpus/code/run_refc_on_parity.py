#!/usr/bin/env python3
"""Run the REAL ``refc_train.py`` against a parity-guarded epcache on Thor and
report the ONLY admissible memory number.

⛔ MEMORY PROBES ON THOR LIE IN BOTH DIRECTIONS (CLAUDE.md traps preflight):
``torch.cuda.mem_get_info()`` reported 3.4 GB free while 60 GB was allocated AND
written; ``free``/``tegrastats`` showed 106 GB "used" on an idle box; ``VmRSS`` read
0.62 GB against 24 GB allocated. On unified memory none of those mean what they
mean on a discrete GPU. **Only in-process ``torch.cuda.max_memory_allocated()``
is quotable**, so this wrapper exists purely to read it in the SAME process that
did the allocating — a subprocess call could not.

⛔ THE GUARD IS NOT BYPASSED. This calls ``refc_train.main`` with an ordinary argv;
there is no ``--force``, no monkeypatch, and no environment variable that disables
the parity check (``parity.py`` deliberately provides none). If the guard refuses,
this script dies with the refusal — and that IS the result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for cand in (Path.home() / "TanitAD" / "stack", Path("/workspace/TanitAD/stack")):
    if (cand / "scripts" / "refc_train.py").exists():
        sys.path.insert(0, str(cand))
        sys.path.insert(0, str(cand / "scripts"))
        break

# ⚠️ torch spawns ~113 threads PER PROCESS; unset, concurrent arms sit at sm 0-6 %
# for 50 minutes and look exactly like a deadlock (MEASURED 2026-07-27).
os.environ.setdefault("OMP_NUM_THREADS", "6")

import torch  # noqa: E402
import refc_train  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, required=True,
                    help="0 = the WHOLE corpus (strict parity); N = the sorted "
                         "PREFIX of N (subset mode, self-labelling, NOT strict)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8,
                    help="⚠️ Thor saturates at ~8; the A40 instinct inverts here")
    ap.add_argument("--workers", type=int, default=2,
                    help="⚠️ each worker costs ~8.6 GB HOST RAM on Thor")
    ap.add_argument("--config", default="base")
    ap.add_argument("--mode", default="diffusion")
    a = ap.parse_args(argv)

    argv2 = ["--data-root", a.data_root, "--out", a.out,
             "--steps", str(a.steps), "--config", a.config, "--mode", a.mode,
             "--batch", str(a.batch), "--workers", str(a.workers),
             "--log-every", "1", "--save-every", "1000000"]
    if a.episodes:
        argv2 += ["--episodes", str(a.episodes)]

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    metrics = refc_train.main(argv2)
    dt = time.time() - t0

    rec = {
        "argv": argv2,
        "wall_seconds": round(dt, 1),
        "steps": a.steps, "batch": a.batch, "workers": a.workers,
        "s_per_step_incl_setup": round(dt / max(a.steps, 1), 3),
        # ⭐ the ONLY admissible memory number on this device
        "cuda_max_memory_allocated_B": (int(torch.cuda.max_memory_allocated())
                                        if torch.cuda.is_available() else None),
        "cuda_max_memory_allocated_GB": (round(torch.cuda.max_memory_allocated()
                                               / 1e9, 2)
                                         if torch.cuda.is_available() else None),
        "cuda_max_memory_reserved_GB": (round(torch.cuda.max_memory_reserved() / 1e9,
                                              2)
                                        if torch.cuda.is_available() else None),
        "n_params_trainable": metrics.get("n_params_trainable"),
        "final_log": metrics.get("final"),
        "val": metrics.get("val"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "torch": torch.__version__,
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
    }
    print("=== RUN RECORD ===")
    print(json.dumps(rec, indent=2, default=str))
    Path(a.out, "run_record.json").write_text(json.dumps(rec, indent=2, default=str),
                                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
