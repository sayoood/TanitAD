#!/usr/bin/env python3
"""AMENDMENT 1's CPU tail, handed over from run_m6_chain_v16.py (2026-09-28 ~00:25).

Why: v16's scorer ran the two P gating seams CONCURRENTLY (2 x ~0.9 GB harness RSS) and both hit the W3 harness's
own RAM guard at 00:06 (667 / 1,123 rows; "system available memory 1965 MB") while M5's 12 labelling workers held
~16 GB. Its second attempt then waited for >= 6 GB free, which M5's pool leaves only intermittently -- and a second
concurrent failure would have been the LAST attempt. This tail scores the same two seams ONE AT A TIME (half the
RSS), starting at >= 5.0 GB free, up to 3 invocations; then families, then analyze with v16's EXACT argv.

Nothing about the analysis changes: same seams, same harness (unchanged), same families, same analyze argv. The
M6 train-cache NVMe copy is NOT deleted here: it is kept as M6b's one authorised C: copy (coordinator 00:14, floor
>= 30 GiB with it present, emergency < 15 GiB -- fast_copy_e2.py's guard stays on it) and the M6b chain re-hashes
every file against the source manifest before any arm reads it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
ROOT = PKG / "raw" / "2026-09-27-m6-proxy"
E2 = ROOT / "epoch2"
ARMS = ROOT / "arms"
LOG = ROOT / "chain.log"
TIMES = ROOT / "chain_times.json"
D = Path("D:/Projects/TanitAD/data/refe_proxy")
TC = D / "cache_train_ep015"
TCF = Path("C:/Users/Admin/refe_proxy_fast/cache_train_ep015")
ECD = D / "cache_eval_ep015"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def mark(k):
    t = json.load(open(TIMES)) if TIMES.exists() else {}
    t[k] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(t, open(TIMES, "w"), indent=1)


def procs(pattern):
    me = f"{os.getpid()},{os.getppid()}"
    o = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Process | Where-Object { "
                        f"$_.CommandLine -match '{pattern}' -and $_.Name -match 'python' -and "
                        f"@({me}) -notcontains $_.ProcessId }} | Measure-Object).Count"],
                       capture_output=True, text=True).stdout.strip()
    return int(o) if o.isdigit() else -1


def run(name, cmd):
    t0 = time.time()
    mark(f"{name}_start")
    with open(ROOT / "logs" / f"{name}.log", "a", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(PKG), stdout=fh, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
    mark(f"{name}_end")
    log(f"{name}: EXIT {rc} after {(time.time() - t0) / 60:.1f} min")
    return rc


def main() -> int:
    log("E2 TAIL: finish_epoch2.py takes over AMENDMENT 1's CPU tail from chain v16 (the two P seams failed the "
        "harness RAM guard concurrently at 00:06; re-scored ONE AT A TIME from >= 5.0 GB free)")
    while procs("run_m6_chain_v16") != 0 or procs("proxy_eval.py score") != 0:
        time.sleep(10)
    log("E2 TAIL: chain v16 and its scorer are gone")
    out = str(E2)
    ok = False
    for k in range(3):
        rc = run(f"E2_score_P_seq{k + 1}", [PY, "eval/proxy_eval.py", "score", "--out", out, "--arm-suffix", "e2",
                                            "--only", "refe_m6_P0e2_off,refe_m6_P1e2_off", "--workers", "1",
                                            "--min-free-ram-gb", "5.0"])
        if rc == 0:
            ok = True
            break
    rc = run("E2_score_gating_check", [PY, "eval/proxy_eval.py", "score", "--out", out, "--arm-suffix", "e2",
                                       "--only", "refe_m6_W0e2_on,refe_m6_W1e2_on,refe_m6_P0e2_off,refe_m6_P1e2_off",
                                       "--workers", "1", "--min-free-ram-gb", "5.0"])
    log(f"E2 TAIL: gating seams {'ALL PASS' if (ok and rc == 0) else 'NOT all PASS -- analyze reports it'}")
    run("E2_families", [PY, "eval/proxy_eval.py", "families", "--out", out, "--arm-suffix", "e2"])
    run("E2_analyze", [PY, "eval/proxy_eval.py", "analyze", "--out", out, "--arm-suffix", "e2", "--epoch1-out",
                       str(ROOT), "--train-cache", str(TCF if (TCF / "manifest.json").exists() else TC),
                       "--eval-caches", str(ECD), "--arms-root", str(ARMS)])
    mark("EPOCH2_VERDICT_WRITTEN")
    log("EPOCH-2 (AMENDMENT 1) OUTCOME WRITTEN")
    log("E2 TAIL: the NVMe train-cache copy is KEPT as M6b's one authorised C: copy (fast_copy_e2.py's guard stays on)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
