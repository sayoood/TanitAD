#!/usr/bin/env python3
"""W3 agent-free chain: the HEAVY navtest work, started ONLY after A8's done-marker exists.

Why a chain: the full metric cache (~12k scenes), the full scoring runs and the frame-bank build
are HEAVY and the brief allows them only after
``C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/summary.json`` exists, then with ONE
worker. This process launches the steps one at a time (never two heavy steps at once), each as
its own process at BELOW_NORMAL priority, and banks every result as it lands.

Steps (each is SKIPPED if its own done-condition already holds — the chain is resumable):
  S1  cache_navtest     run_v1.py cache  (full navtest, D:)          retried on a RAM-guard abort
  S2  CV / HUMAN / STOP run_v1.py score  (--patch-loader: C7 PASS)   the reproduction + floors
  S3  analysis          analyze_navtest.py  (C5, STOP decomposition, paired deltas, verdicts)
  S4  export            export_navtest_inputs.py (all 136 logs)      for the bank + the bridge
  S5  frame bank        build_navtest_frames.py per VERIFIED shard, polling the receipt

Off-switch: create ``raw/chain_w3/STOP`` — the chain exits before the next step.
Every step's rc / wall / RAM is appended to ``raw/chain_w3/chain.log`` and ``steps.json``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RAW = os.path.join(PKG, "raw")
CH = os.path.join(RAW, "chain_w3")
A8_DONE = "C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/summary.json"
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
TA_PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
V11_TREE = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
EXP = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1"
EXPORT = f"{EXP}/inputs/navtest_inputs.json.gz"
BANK = f"{EXP}/frame_bank"
RECEIPT = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
           "2026-09-19-navhard-download/raw/receipt_navtest_camera.json")
MIN_AVAIL_MB = 4000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    os.makedirs(CH, exist_ok=True)
    with open(os.path.join(CH, "chain.log"), "a", encoding="utf-8") as fh:
        fh.write(f"{now()} {msg}\n")


def avail_mb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2**20


def stopped() -> bool:
    if os.path.exists(os.path.join(CH, "STOP")):
        log("STOP file present — exiting")
        return True
    return False


def training_procs() -> list:
    """Live training-ish processes (evidence beside every heavy step: the box is shared —
    the Master Mind has an S1 pass and A7 booked, and A7 also reads D:)."""
    out = []
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            c = " ".join(p.info.get("cmdline") or [])
            if "python" in (p.info.get("name") or "").lower() and any(
                    k in c for k in ("_train.py", "p_runner.py", "train_v6", "trainer")):
                out.append({"pid": p.info["pid"], "cmd": c[:160]})
    except Exception as e:                                            # noqa: BLE001
        out.append({"probe_error": repr(e)})
    return out


def record(step: str, **kw) -> None:
    p = os.path.join(CH, "steps.json")
    d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    d[step] = {**d.get(step, {}), **kw, "utc": now()}
    json.dump(d, open(p, "w", encoding="utf-8"), indent=1)


def counts_pass(label: str) -> bool:
    p = os.path.join(RAW, label, f"{label}.counts.json")
    return os.path.exists(p) and json.load(open(p, encoding="utf-8")).get("status") == "PASS"


def run(step: str, cmd: list, env_extra: dict | None = None, tries: int = 1) -> int:
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    if env_extra:
        env.update(env_extra)
    rc = None
    for k in range(1, tries + 1):
        for _ in range(180):                          # wait up to 3 h for RAM headroom
            if avail_mb() >= MIN_AVAIL_MB:
                break
            time.sleep(60)
        if stopped():
            sys.exit(0)
        tp = training_procs()
        log(f"step={step} try={k} start avail_mb={avail_mb():.0f} training_procs={json.dumps(tp)}")
        t0 = time.time()
        out = os.path.join(CH, f"{step}_try{k}.out")
        with open(out, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=PKG).returncode
        log(f"step={step} try={k} rc={rc} wall_s={time.time() - t0:.0f} avail_mb={avail_mb():.0f}")
        record(step, rc=rc, tries=k, wall_s=round(time.time() - t0, 1), out=os.path.relpath(out, PKG),
               training_procs_at_start=tp)
        if rc == 0:
            return 0
        time.sleep(900)
    return rc


def main() -> int:
    log(f"chain start (pid {os.getpid()}); waiting for A8 done-marker {A8_DONE}")
    for _ in range(216):                               # up to 18 h, polling every 5 min
        if os.path.exists(A8_DONE):
            break
        if stopped():
            return 0
        time.sleep(300)
    if not os.path.exists(A8_DONE):
        log("A8 done-marker never appeared in 18 h — nothing started")
        return 5
    log("A8 done-marker seen — starting the heavy steps, ONE at a time")
    nv_env = {"PYTHONPATH": V11_TREE}
    # S1 --------------------------------------------------------------------------
    if not counts_pass("cache_navtest"):
        if run("S1_cache_navtest", [NV_PY, "code/run_v1.py", "cache", "--label", "cache_navtest",
                                    "--cache-name", "navtest", "--per-log",
                                    "--logs-per-process", "4"], nv_env, tries=4) != 0:
            log("S1 FAILED — stopping (no score without a complete cache)")
            return 1
    # S2 --------------------------------------------------------------------------
    for arm in ("CV", "HUMAN", "STOP"):
        lab = f"{arm}_navtest"
        if not counts_pass(lab):
            run(f"S2_{lab}", [NV_PY, "code/run_v1.py", "score", "--label", lab, "--arm", arm,
                              "--cache-name", "navtest", "--patch-loader"], nv_env, tries=2)
    # S4 (before S3: the analysis's per-log table needs the token -> log map) -------
    if not os.path.exists(os.path.join(RAW, "export_navtest.manifest.json")):
        run("S4_export", [NV_PY, "code/export_navtest_inputs.py", "--out", EXPORT,
                          "--manifest", "raw/export_navtest.manifest.json"], nv_env, tries=2)
    # S3 --------------------------------------------------------------------------
    run("S3_analysis", [NV_PY, "code/analyze_navtest.py", "--split-run", "navtest"], nv_env)
    # S3b: the TanitEval artifacts + the FOUR FAMILIES + criteria_check per arm (TanitAD venv:
    # taniteval.four_families imports torch). Binding: an eval that reports only the benchmark's
    # own score is incomplete.
    run("S3b_artifacts", [TA_PY, "code/artifacts_navtest.py", "--arms", "CV,HUMAN,STOP",
                          "--split-run", "navtest", "--criteria"],
        {"PYTHONPATH": "D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval"})
    if not os.path.exists(EXPORT):
        log("S4 export missing — frame bank not started")
        return 1
    # S5 --------------------------------------------------------------------------
    ta_env = {"PYTHONPATH": "D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval"}
    last_new = time.time()
    while True:
        if stopped():
            return 0
        rec = json.load(open(RECEIPT, encoding="utf-8"))
        ver = sorted(int(n.rsplit("_", 1)[1].split(".")[0]) for n, r in rec.get("files", {}).items()
                     if r.get("status") in ("COMPLETE_VERIFIED", "ALREADY_COMPLETE_VERIFIED"))
        todo = [i for i in ver if not os.path.exists(f"{BANK}/shard_{i:02d}.DONE.json")]
        for i in todo:
            # --threads 6: grid_sample is deterministic per output element, so the thread count
            # cannot change a pixel (KB1 pins the bytes); it is still ONE process = one worker.
            run(f"S5_bank_s{i:02d}", [TA_PY, "code/build_navtest_frames.py", "--inputs", EXPORT,
                                      "--bank", BANK, "--shards", str(i), "--threads", "6"], ta_env)
            last_new = time.time()
        n_done = sum(os.path.exists(f"{BANK}/shard_{i:02d}.DONE.json") for i in range(32))
        record("S5_bank", n_shards_done=n_done)
        if n_done == 32:
            break
        if time.time() - last_new > 4 * 3600:
            log(f"S5: no new verified shard for 4 h ({n_done}/32 done) — exiting; rerun resumes")
            return 2
        time.sleep(600)
    run("S5_bank_finalize", [TA_PY, "code/bank_finalize.py", "--bank", BANK, "--inputs", EXPORT], ta_env)
    log("CHAIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
