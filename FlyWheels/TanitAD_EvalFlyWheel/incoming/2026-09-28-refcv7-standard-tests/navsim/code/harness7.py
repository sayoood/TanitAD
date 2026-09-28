#!/usr/bin/env python3
"""KH -- the harness reads its KNOWN values through THIS package's drivers (SPEC §7). TANITAD VENV.

    python code/harness7.py warmup     # wait for the warmup CV/STOP/ECHO scores, then check vs E2
    python code/harness7.py navtest    # score STOP on all 12,146 navtest tokens, check vs W3
    python code/harness7.py navhard    # score CV on all 5,912 navhard tokens, check vs W7

Replaces the bash chain of ``run_harness7.sh`` for these two steps. ⚠️ MEASURED 2026-09-28: under
MSYS, ``P="$(cd .. && pwd)"`` is a ``/d/...`` path, and every JSON read the bash chain did through
``python -c "open(r'$P/...')"`` failed and printed ABSENT (the refcv6 queue's "status=NO_COUNTS"
lines are the same defect). The navtest retry loop would therefore have re-scored a PASSED
control six times (~3 h). Python reads the artifacts directly; every verdict is read from its
JSON, never from an exit code.

⚠️ MEASURED 2026-09-28 05:50-05:58 Berlin: other sessions' jobs (~13 GB RSS) held the box at
1.8-3.4 GB available and E1's RAM guard aborted BOTH scorers twice. Each launch therefore waits for
``R7_KH_RAM_GB`` (default 9) GB free on 3 consecutive samples -- headroom for the others'
fluctuations above the guard's 3 GB floor -- and a guard abort is retried up to 12 times.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
W3RAW = os.path.abspath(os.path.join(PKG, "..", "..", "2026-09-19-navsim-v1-navtest", "raw"))
NPY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
RAM_GB = float(os.environ.get("R7_KH_RAM_GB", "9"))
TRIES = 12


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(os.path.join(PKG, "raw", "harness7.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def jget(path: str, *keys):
    try:
        d = json.load(open(path, encoding="utf-8"))
        for k in keys:
            d = d[k]
        return d
    except Exception:                                                     # noqa: BLE001
        return None


def ram_wait(min_gb: float) -> float:
    import psutil
    ok = 0
    while True:
        f = psutil.virtual_memory().available / 2**30
        ok = ok + 1 if f >= min_gb else 0
        if ok >= 3:
            return f
        time.sleep(30)


def run(cmd: list, logfile: str) -> None:
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)


def warmup() -> None:
    hr = os.path.join(PKG, "raw", "harness_repro")
    arms = ("CV_official", "STOP_zero", "ECHO_ha0_ext")
    t0 = time.time()
    while not all(jget(os.path.join(hr, f"score_{a}.counts.json"), "status") for a in arms):
        if time.time() - t0 > 3 * 3600:
            log("KH_WARMUP TIMEOUT waiting for the warmup scores")
            return
        time.sleep(30)
    st = {a: jget(os.path.join(hr, f"score_{a}.counts.json"), "status") for a in arms}
    log(f"KH_WARMUP count guards {st}")
    lg = os.path.join(PKG, "raw", "harness7_warmup.driver.txt")
    run([PY, os.path.join(HERE, "check_harness_repro7.py"), "--mine-dir", hr,
         "--arms", "CV_official,STOP_zero", "--out", os.path.join(PKG, "raw", "HARNESS_REPRO.json")], lg)
    run([PY, os.path.join(HERE, "check_harness_repro7.py"), "--mine-dir", hr,
         "--arms", "ECHO_ha0_ext", "--out", os.path.join(PKG, "raw", "controls", "KH_ECHO_warmup.json")], lg)
    run([PY, os.path.join(HERE, "import_floors7.py"), "--split", "warmup_two_stage"], lg)
    log(f"KH_WARMUP CV_STOP={jget(os.path.join(PKG, 'raw', 'HARNESS_REPRO.json'), 'verdict')} "
        f"ECHO={jget(os.path.join(PKG, 'raw', 'controls', 'KH_ECHO_warmup.json'), 'verdict')}")


def navtest() -> None:
    nt = os.path.join(PKG, "raw", "harness_navtest")
    cnt = os.path.join(nt, "r7kh_STOP", "r7kh_STOP.counts.json")
    for tr in range(1, TRIES + 1):
        if jget(cnt, "status") == "PASS":
            break
        f = ram_wait(RAM_GB)
        log(f"KH_NAVTEST score try={tr} (free {f:.1f} GB)")
        run([PY, os.path.join(HERE, "score_navtest7.py"), "--label", "r7kh_STOP", "--official",
             "STOP", "--out", nt], nt + ".driver.txt")
        lg = os.path.join(nt, "r7kh_STOP", "r7kh_STOP.log")
        ram_abort = os.path.exists(lg) and "RAM_GUARD" in open(lg, encoding="utf-8",
                                                               errors="replace").read()
        log(f"KH_NAVTEST try={tr} counts={jget(cnt, 'status')} ram_guard_abort={ram_abort}")
        if not ram_abort:
            break
    run([PY, os.path.join(HERE, "check_kh_navtest7.py"), "--mine",
         os.path.join(nt, "r7kh_STOP", "r7kh_STOP.csv"), "--w3",
         os.path.join(W3RAW, "STOP_navtest", "STOP_navtest.csv"), "--out",
         os.path.join(PKG, "raw", "controls", "KH_navtest_STOP.json")],
        os.path.join(PKG, "raw", "harness7_navtest.driver.txt"))
    log(f"KH_NAVTEST {jget(os.path.join(PKG, 'raw', 'controls', 'KH_navtest_STOP.json'), 'verdict')}")


def navhard() -> None:
    out = os.path.join(PKG, "raw", "harness_repro_navhard")
    os.makedirs(out, exist_ok=True)
    tag = "CV_official__navhard_two_stage"
    cnt = os.path.join(out, f"score_{tag}.counts.json")
    for tr in range(1, TRIES + 1):
        if jget(cnt, "status") == "PASS":
            break
        f = ram_wait(RAM_GB)
        log(f"KH_NAVHARD score try={tr} (free {f:.1f} GB)")
        run([NPY, os.path.join(HERE, "score_arm7.py"), "--arm", "CV_official", "--official-agent",
             "constant_velocity_agent", "--split", "navhard_two_stage", "--out", out,
             "--exp-tag", "e7kh"], os.path.join(out, "CV_official.driver.txt"))
        lg = os.path.join(out, f"score_{tag}.log")
        ram_abort = os.path.exists(lg) and "E1_RAM_GUARD_ABORT" in open(
            lg, encoding="utf-8", errors="replace").read()
        log(f"KH_NAVHARD try={tr} counts={jget(cnt, 'status')} ram_guard_abort={ram_abort}")
        if ram_abort:
            os.makedirs(os.path.join(out, "_ramguard_abort"), exist_ok=True)
            os.replace(lg, os.path.join(out, "_ramguard_abort", f"score_{tag}.try{tr}.log"))
            continue
        break
    run([PY, os.path.join(HERE, "import_floors7.py"), "--split", "navhard_two_stage",
         "--check-cv", out], os.path.join(PKG, "raw", "harness7_navhard.driver.txt"))
    log(f"KH_NAVHARD {jget(os.path.join(PKG, 'raw', 'floors', 'navhard_two_stage', 'FLOORS_PROVENANCE.json'), '_KH_nav_check', 'verdict')}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "warmup"
    log(f"START {what}")
    {"warmup": warmup, "navtest": navtest, "navhard": navhard}[what]()
    log(f"ZZHARNESS7_{what}_DONEZZ")
