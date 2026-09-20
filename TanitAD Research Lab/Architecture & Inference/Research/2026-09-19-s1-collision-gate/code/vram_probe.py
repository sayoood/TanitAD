"""Measure the REAL arm's peak VRAM, in-process, so the gate can be re-calibrated on evidence.

⛔ NOT AN ARM. Seed 99 (never 0/1/2 = A8/P0/P0b), a scratch out dir outside the P panel, and
12 steps. Nothing it writes can be mistaken for a P arm.

⛔ ONLY the in-process allocator counter is admissible. `nvidia-smi`'s total includes the desktop,
which swings 3,111-3,958 MiB on this box — subtracting a baseline that moves by 847 MiB cannot
answer a question whose answer matters to ~600 MiB. `gp_cuda_max_mem_gb` is
`torch.cuda.max_memory_allocated()`, a high-water mark since process start, so the row read AFTER
the forced eval also covers the eval path's peak.

The argv is built with `p_runner.build_argv` — the same function P0 will use — so the probe cannot
drift from the arm it is measuring.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("D:/Projects/TanitAD")
TREE = Path("C:/Users/Admin/tanitad-a7-run")
BASE = Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/config.json")
OUT = Path("C:/Users/Admin/tanitad-caches/vram-probe-20260920")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

sys.path.insert(0, str(REPO / "stack" / "scripts"))
import p_runner as PR                                       # noqa: E402

STEPS, EVAL_EVERY, EVAL_BATCHES = 12, 10, 20


def main() -> int:
    base_argv = list(json.loads(BASE.read_text(encoding="utf-8"))["argv"])
    OUT.mkdir(parents=True, exist_ok=True)
    argv = PR.build_argv(base_argv, seed=99, out=str(OUT / "run"))
    # override the three knobs that make this a PROBE rather than an arm
    def setflag(a, flag, val):
        if flag in a:
            a[a.index(flag) + 1] = str(val)
        else:
            a += [flag, str(val)]
        return a
    for flag, val in (("--steps", STEPS), ("--eval-every", EVAL_EVERY),
                      ("--eval-batches", EVAL_BATCHES),
                      ("--grad-probe-modules", "core")):
        argv = setflag(argv, flag, val)
    (OUT / "probe_argv.json").write_text(json.dumps(argv, indent=1), encoding="utf-8")
    print("ZZVP-START steps=%d eval_every=%d ZZ" % (STEPS, EVAL_EVERY), flush=True)
    t0 = time.time()
    with open(OUT / "probe.log", "wb") as log:
        rc = subprocess.run([PY, "-u", "scripts/refc_v3_train.py"] + argv,
                            cwd=str(TREE / "stack"), stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ,
                                 "PYTHONPATH": os.pathsep.join(
                                     [str(TREE / "stack"), str(TREE), str(TREE / "taniteval")]),
                                 "PYTHONIOENCODING": "utf-8",
                                 "OMP_NUM_THREADS": "6"}).returncode
    wall = time.time() - t0
    print("ZZVP-RC %d wall=%ds ZZ" % (rc, int(wall)), flush=True)

    # ⛔ assert on the ARTIFACT, never the exit code
    met = OUT / "run" / "metrics.jsonl"
    if not met.exists():
        print("ZZVP-NO-METRICS — probe produced nothing; see probe.log ZZ", flush=True)
        return 3
    peak, rows, last = None, 0, None
    with open(met, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            rows += 1
            if "gp_cuda_max_mem_gb" in d:
                v = float(d["gp_cuda_max_mem_gb"])
                peak = v if peak is None else max(peak, v)
                last = d.get("step")
    res = {"rc": rc, "wall_s": round(wall, 1), "rows": rows,
           "peak_cuda_max_mem_gb": peak, "last_step_with_key": last,
           "peak_mib": None if peak is None else round(peak * 1024.0),
           "steps": STEPS, "eval_every": EVAL_EVERY, "eval_batches": EVAL_BATCHES,
           "note": ("torch.cuda.max_memory_allocated() high-water mark, in-process; covers the "
                    "forced eval because the mark is cumulative since process start")}
    (OUT / "vram_probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)
    print("ZZVP-PEAK-MIB %s ZZ" % res["peak_mib"], flush=True)
    return 0 if peak is not None else 4


if __name__ == "__main__":
    raise SystemExit(main())
