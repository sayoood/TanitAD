#!/usr/bin/env python3
"""p6_read.py -- ONE command for the pre-registered P6 scorer-speed read (PREREG_P6_SCORER_SPEED.md). Takes a checkpoint path and an output directory and ends in the
card's verdict, computed mechanically by selc6_analyze.py.

    python p6_read.py --ckpt <P6 final ckpt> --out DIR --run-config <config.json> --train-logs <train_logs.txt> --p6-seam <seam.npz> --p6-csv <p6.csv> [--expect-ckpt-sha SHA]

Stages (idempotent; stages/<name>.json rc 0 = done; markers/p6read.DONE|FAILED), run DETACHED (launch via WMI `cmd /c ... < NUL`, see RUNBOOK.md):
  preflight   G6-PIN (prereg + code + pins) and the u400 control (p6_control.py must PASS)          CPU, ~1 min
  dump        selc6_dump.py segments (GPU, YAX lock, FP32), ESTIMATED ~0.7 h                         GPU (driverl-eval venv)
  consolidate selc6_dump.py --consolidate -> dump/dump_p6.npz
  score       selc6_score.py (W3 NAVSIM v1.1 harness, 130 calls) -> score/table_p6.npz              CPU, ESTIMATED 1.3-1.8 h
  analyze     selc6_analyze.py -> p6_speed_result.json, P6_SPEED_RESULT.md, VERDICT                  CPU, ~2 min
Never kills a process, never touches the GPU lock file. A failed stage writes markers/p6read.FAILED and stops. Re-running resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
CPU = {"CUDA_VISIBLE_DEVICES": "-1"}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-config")
    ap.add_argument("--train-logs")
    ap.add_argument("--p6-seam")
    ap.add_argument("--p6-csv")
    ap.add_argument("--expect-ckpt-sha")
    ap.add_argument("--workers", default="3")
    ap.add_argument("--work", default="F:/TanitAD_p6read_work/score", help="NTFS scratch for harness run dirs (>= 2 folders deep on F:)")
    a = ap.parse_args(argv)
    out = a.out
    for d in ("markers", "stages", "logs", "dump", "score"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    logp = os.path.join(out, "CHAIN.log")

    def say(m):
        line = f"[{now()}] {m}"
        print(line, flush=True)
        open(logp, "a", encoding="utf-8").write(line + "\n")

    def marker(n, **kw):
        kw["t"] = now()
        json.dump(kw, open(os.path.join(out, "markers", n), "w"), indent=1)

    def done(s):
        p = os.path.join(out, "stages", s + ".json")
        return os.path.exists(p) and json.load(open(p)).get("rc") == 0

    def run(stage, cmd, env_extra=None):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", **(env_extra or {}))
        say(f"{stage}: {' '.join(cmd[1:])}")
        with open(os.path.join(out, "logs", stage + ".log"), "a", encoding="utf-8") as f:
            rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        json.dump({"rc": rc, "t": now()}, open(os.path.join(out, "stages", stage + ".json"), "w"))
        say(f"{stage}: rc {rc}")
        return rc

    def fail(stage, why):
        marker("p6read.FAILED", stage=stage, why=why)
        say(f"FAILED at {stage}: {why}")
        return 1

    for m in ("p6read.DONE", "p6read.FAILED"):
        if os.path.exists(os.path.join(out, "markers", m)):
            say(f"REFUSED: markers/{m} exists (remove it deliberately to re-run)")
            return 4
    marker("p6read.STARTED")
    dump_dir, score_dir = os.path.join(out, "dump"), os.path.join(out, "score")
    if not done("preflight"):
        if run("preflight", [PY_T, "p6_control.py", "--out", os.path.join(out, "control_result.json")], CPU) != 0:
            return fail("preflight", "pins or the u400 control are red (logs/preflight.log, control_result.json)")
    if not done("dump"):
        n_err = 0
        while True:
            rc = run("dump", [PY_D, "selc6_dump.py", "--ckpt", a.ckpt, "--out", dump_dir, "--give-up-h", "10"] + (["--expect-ckpt-sha", a.expect_ckpt_sha] if a.expect_ckpt_sha else []))
            if rc == 0:
                break
            if rc == 3:
                say("dump: segment done, more remain; yielding the GPU for 180 s")
                time.sleep(180)
                continue
            if rc == 1 and n_err < 3:
                n_err += 1
                time.sleep(300)
                continue
            return fail("dump", {4: "refused (disk / checkpoint sha)", 5: "GPU busy 10 h"}.get(rc, f"rc {rc}"))
        json.dump({"rc": 0, "t": now()}, open(os.path.join(out, "stages", "dump.json"), "w"))
    if not done("consolidate") and run("consolidate", [PY_D, "selc6_dump.py", "--consolidate", "--ckpt", a.ckpt, "--out", dump_dir]) != 0:
        return fail("consolidate", "logs/consolidate.log")
    if not done("score"):
        rc = None
        for _ in (1, 2):
            rc = run("score", [PY_T, "selc6_score.py", "--dump", os.path.join(dump_dir, "dump_p6.npz"), "--out", score_dir, "--workers", a.workers, "--work", a.work], CPU)
            if rc == 0 or rc not in (1, 3):
                break
            time.sleep(600)
        if rc != 0:
            return fail("score", f"rc {rc} (logs/score.log, score/score.log)")
    cmd = [PY_T, "selc6_analyze.py", "--dump", os.path.join(dump_dir, "dump_p6.npz"), "--table", os.path.join(score_dir, "table_p6.npz"), "--out", out]
    for flag, val in (("--run-config", a.run_config), ("--train-logs", a.train_logs), ("--p6-seam", a.p6_seam), ("--p6-csv", a.p6_csv), ("--expect-ckpt-sha", a.expect_ckpt_sha)):
        if val:
            cmd += [flag, val]
    rc = run("analyze", cmd, CPU)                  # rc 2 = INSTRUMENT FAILURE (a real, reportable outcome, not a crash)
    if rc not in (0, 2):
        return fail("analyze", f"rc {rc} (logs/analyze.log)")
    r = json.load(open(os.path.join(out, "p6_speed_result.json")))
    marker("p6read.DONE", verdict=r["verdict"]["VERDICT"], red_gates=r["red_gates"])
    say(f"DONE: {r['verdict']['VERDICT']} red gates {r['red_gates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
