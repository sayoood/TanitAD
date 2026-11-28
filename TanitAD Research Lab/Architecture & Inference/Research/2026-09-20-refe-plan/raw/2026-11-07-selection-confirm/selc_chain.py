#!/usr/bin/env python3
"""The detached chain of the H-SEL-1 fresh confirmation (stream YSEL2). Run with the tanitad venv; it launches the dump and the harness in driverl-eval.

  preflight   PREREG_SEL_CONFIRM.md sha256 + CODE_PIN.json + selc_gates.py (G-RULE, G-PIN) + selc_analyze.py --replay-p2 (G-ANA)
  dump_smoke  GPU, under the shared lock: the 6 smoke C3 tokens; bit-for-bit identity vs C3 table + ego probe (+ RED) -> dump/smoke_identity.json
  score_smoke CPU: 64 harness columns + RED on the smoke tokens == C3 table rows bit for bit -> score/smoke_score_gate.json
  dump        GPU segments of <= 1,000 tokens; the lock is released and the process exits between segments (YAXN or anyone else may take the GPU)
  consolidate dump/dump_confirm.npz
  verdict     CPU: banked + rule columns per chunk, gates -> score/verdict_columns.npz
  analyze     selc_analyze.py --verdict -> CONFIRM_RESULT.md, confirm_result.json; marker markers/selc_verdict.DONE
  table       CPU: the 64 proposal columns per chunk -> score/table_confirm.npz
  table_read  selc_analyze.py --table -> CONFIRM_TABLE_APPENDIX.md; marker markers/selc_full.DONE
Any failure: markers/selc_full.FAILED (json, `why`), and nothing after it runs. Stages are idempotent (stages/<name>.json, rc 0 = done).
Never touches the GPU lock file itself; never kills anything.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
MK = os.path.join(HERE, "markers")
ST = os.path.join(HERE, "stages")
LOG = os.path.join(HERE, "CHAIN.log")
MIN_GIB = 3.0


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def say(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def marker(name, **kw):
    kw["t"] = now()
    kw["pid"] = os.getpid()
    with open(os.path.join(MK, name), "w", encoding="utf-8") as f:
        f.write(json.dumps(kw, indent=1))


def done(stage):
    p = os.path.join(ST, stage + ".json")
    return os.path.exists(p) and json.load(open(p)).get("rc") == 0


def record(stage, rc, **kw):
    kw.update(rc=rc, t=now())
    json.dump(kw, open(os.path.join(ST, stage + ".json"), "w"), indent=1)


def disk_ok(max_wait_s=2 * 3600):
    t0 = time.time()
    while True:
        free = {d: shutil.disk_usage(d + "/").free / 2 ** 30 for d in ("C:", "E:", "F:")}
        if all(v >= MIN_GIB for v in free.values()):
            return True
        if time.time() - t0 > max_wait_s:
            say(f"disk below {MIN_GIB} GiB for {max_wait_s} s: {free}")
            return False
        say(f"waiting for disk >= {MIN_GIB} GiB: {free}")
        time.sleep(300)


def run(stage, cmd, logname, env_extra=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env.update(env_extra or {})
    lp = os.path.join(HERE, "logs", logname)
    say(f"{stage}: {' '.join(cmd[1:])}")
    with open(lp, "a", encoding="utf-8") as f:
        f.write(f"\n===== {now()} {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    say(f"{stage}: rc {rc}")
    return rc


CPU = {"CUDA_VISIBLE_DEVICES": "-1"}


def main():
    os.makedirs(MK, exist_ok=True)
    os.makedirs(ST, exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    for m in ("selc_full.DONE", "selc_full.FAILED"):
        if os.path.exists(os.path.join(MK, m)):
            say(f"REFUSED: markers/{m} exists (remove it deliberately to re-run)")
            return 4
    marker("selc_full.STARTED")
    tok = os.path.join(HERE, "tokens_confirm.json")
    smk = os.path.join(HERE, "tokens_smoke.json")
    dump = os.path.join(HERE, "dump")

    def fail(stage, why):
        marker("selc_full.FAILED", stage=stage, why=why)
        say(f"FAILED at {stage}: {why}")
        return 1

    # ---- preflight
    if not done("preflight"):
        for name, cmd in (("pins", [PY_T, "-c", "import selc_score as S; S.check_pins(); print('pins OK')"]),
                          ("gates", [PY_T, "selc_gates.py"]), ("replay", [PY_T, "selc_analyze.py", "--replay-p2"])):
            rc = run("preflight." + name, cmd, "preflight.log", CPU)
            if rc != 0:
                record("preflight", rc, step=name)
                return fail("preflight", f"{name} rc {rc} (logs/preflight.log)")
        record("preflight", 0)
    # ---- GPU stages: smoke dump, then segments
    for stage, extra in (("dump_smoke", ["--smoke-only"]), ("dump", [])):
        if done(stage):
            continue
        n_err = 0
        while True:
            if not disk_ok():
                return fail(stage, "disk")
            rc = run(stage, [PY_D, "selc_dump.py", "--tokens", tok, "--smoke", smk, "--out", dump, "--segment-tokens", "1000", "--give-up-h", "10"] + extra, "dump.log")
            if rc == 0:
                record(stage, 0)
                break
            if rc == 3:
                say(f"{stage}: segment done, more remain; yielding the GPU for 180 s")
                time.sleep(180)
                continue
            if rc == 1 and n_err < 5:                       # e.g. a host-RAM MemoryError from another session's job: bounded, logged retry
                n_err += 1
                with open(os.path.join(HERE, "logs", "retries.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": now(), "stage": stage, "rc": rc, "attempt": n_err}) + "\n")
                time.sleep(300)
                continue
            record(stage, rc)
            return fail(stage, {2: "smoke identity gate RED (dump/smoke_identity.json)", 4: "refused (disk / checkpoint)", 5: "GPU busy for 10 h"}.get(rc, f"rc {rc} after {n_err} retries (dump/dump.log)"))
        if stage == "dump_smoke" and not done("score_smoke"):
            if not disk_ok():
                return fail("score_smoke", "disk")
            rc = run("score_smoke", [PY_T, "selc_score.py", "smoke", "--workers", "3"], "score.log", CPU)
            record("score_smoke", rc)
            if rc != 0:
                return fail("score_smoke", f"smoke scoring gate rc {rc} (score/smoke_score_gate.json)")
    # ---- consolidate, verdict, analyze
    steps = [("consolidate", [PY_D, "selc_dump.py", "--consolidate", "--tokens", tok, "--out", dump], "dump.log"),
             ("verdict", [PY_T, "selc_score.py", "verdict", "--workers", "3"], "score.log"),
             ("analyze", [PY_T, "selc_analyze.py", "--verdict"], "analyze.log"),
             ("table", [PY_T, "selc_score.py", "table", "--workers", "3"], "score.log"),
             ("table_read", [PY_T, "selc_analyze.py", "--table"], "analyze.log")]
    for stage, cmd, lg in steps:
        if done(stage):
            continue
        rc = None
        for att in (1, 2):
            if not disk_ok():
                return fail(stage, "disk")
            rc = run(stage, cmd, lg, CPU)
            if rc == 0 or rc not in (1, 3):                 # 1 = crash (e.g. MemoryError), 3 = some harness calls failed: one stage-level retry
                break
            time.sleep(600)
        record(stage, rc)
        if rc != 0:
            return fail(stage, f"rc {rc} (logs/{lg})")
        if stage == "analyze":
            r = json.load(open(os.path.join(HERE, "confirm_result.json")))
            marker("selc_verdict.DONE", verdict=r["verdict"]["VERDICT"], d_pdms_mix_x100=r["paired"]["d_pdms_mix_x100"],
                   d_slow_share_cruise=r["paired"]["d_slow_share_cruise"])
    r = json.load(open(os.path.join(HERE, "confirm_table_result.json")))
    marker("selc_full.DONE", verdict=json.load(open(os.path.join(HERE, "confirm_result.json")))["verdict"]["VERDICT"],
           table_gates=r["G_PASS"], h_sel_lane_1=r["H_SEL_LANE_1"]["VERDICT"])
    say("DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException:                                   # noqa: B902 -- the marker must be written whatever happens
        tb = traceback.format_exc()
        try:
            say("CHAIN CRASH\n" + tb)
            marker("selc_full.FAILED", stage="chain", why=tb[-1500:])
        except Exception:
            pass
        sys.exit(1)
