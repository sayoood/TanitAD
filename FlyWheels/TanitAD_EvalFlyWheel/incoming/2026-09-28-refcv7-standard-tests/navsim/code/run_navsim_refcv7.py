#!/usr/bin/env python3
"""ONE COMMAND: a refcv7 checkpoint through the whole NavSim suite (TANITAD VENV).

    python code/run_navsim_refcv7.py --ckpt D:/refcv7_eval_kit/ckpt/ckpt_5000.pt
    python code/run_navsim_refcv7.py --ckpt <p> --splits warmup --out <dir>      # one split

Adapted from the refcv6 suite's ``run_navsim_refcv6.py``. Each step is its OWN process, gated on
the ARTIFACT of the step before (never on an exit code -- CLAUDE.md "assert on the artifact"):

  0. md5 of the checkpoint; its step read locally; label ``RESULT`` from step 5,000, else
     ``VALIDATION ONLY`` (SPEC §6); the run's ``config.json`` md5 recorded (recipe stamp);
  1. per split, the GPU: the dev-box lock (``gpu_lock.py``: lock ABSENT + no other python compute
     app) is acquired for the split's BRIDGE only and released before scoring; not acquired within
     the wait -> CPU fp32 for the WHOLE split (one device per split, refcv6 amendment A3). On CUDA
     the K0 / KD controls are re-measured first (``tests/test_model_seam7.py -k "K0 or KD"``);
  2. ``run_bridge7.py`` (resumable rows) -> seams + manifests, ``--derived`` (R7_CEILDECL_d,
     PRIOR_ha0p); navtest also the 200-token diagnostic arms (SPEC §3) in their own bridge dir;
  3. the official scorer per seam, CPU, RAM-gated, retried on the RAM guard, up to ``--max-par``
     at once, launched as soon as a split's seams exist; a seam whose manifest reads ``partial`` is
     REFUSED; a CSV is used only beside a count guard reading PASS;
  4. statistics (``parse7.py`` / ``parse_navtest7.py``), the SPEC §5 ladder (``decompose7.py``),
     four families (``families7.py``), label-free plan deltas (``plan_deltas7.py``);
  5. ``MILESTONE_SUMMARY.json`` (merged per split, refused across checkpoints) and ``BARS.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gpu_lock  # noqa: E402
from pytest_control7 import control_passed  # noqa: E402

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
NPY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
TREE = os.environ.get("TANITAD_REPO", "C:/Users/Admin/ev7nav")
INC = f"{TREE}/FlyWheels/TanitAD_EvalFlyWheel/incoming"
R6IN = f"{INC}/2026-09-23-refcv6-standard-tests/navsim/raw/inputs"
BANKS = "C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923"
CONFIG = "D:/refcv7_eval_kit/ckpt/config.json"
ROAD = f"{R6IN}/road_plane_navhard_warmup_logs.json"
SUB200 = f"{INC}/2026-09-19-navsim-v1-navtest/raw/A1_sub200_tokens.json"
SPLITS = {
    "warmup": {"split": "warmup_two_stage",
               "arms": "R7_A1,R7_A1_s1,R7_FILTOFF,R7_BLIND,R7_VMAXOFF,R7_A1NT,R7_NAVOFF",
               "inputs": f"{INC}/2026-09-19-navsim-refcv4b-bridge/raw/navsim_agent_inputs.json",
               "speed": f"{R6IN}/speed_limits_warmup_two_stage.json",
               "bank2": f"{BANKS}/warmup_two_stage", "bank1": "", "gpu_wait_s": 900},
    "navtest": {"split": "navtest", "arms": "R7_A1,R7_A1_s1",
                "diag_arms": "R7_VMAXOFF,R7_FILTOFF,R7_VMAXORACLE",
                "inputs": "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz",
                "speed": os.path.join(PKG, "raw", "inputs", "speed_limits_navtest.json"),
                "speed_oracle": f"{R6IN}/vmax_oracle_navtest.json",
                "bank2": "", "bank1": "D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank",
                "logs_root": "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test",
                "gpu_wait_s": 10800},
    "navhard": {"split": "navhard_two_stage", "arms": "R7_A1,R7_A1_s1",
                "inputs": ("C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/"
                           "navsim_agent_inputs.json"),
                "speed": f"{R6IN}/speed_limits_navhard_two_stage.json",
                "bank2": f"{BANKS}/navhard_s2", "bank1": f"{BANKS}/navhard_s1", "gpu_wait_s": 10800},
}
DERIVED = ("R7_CEILDECL_d", "PRIOR_ha0p")


def log(msg: str, path: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def md5_file(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jget(path: str, *keys):
    try:
        d = json.load(open(path, encoding="utf-8"))
        for k in keys:
            d = d[k]
        return d
    except Exception:                                                     # noqa: BLE001
        return None


def free_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2**30


def ram_wait(min_gb: float, n_ok: int = 3, every_s: int = 30) -> float:
    ok = 0
    while True:
        f = free_gb()
        ok = ok + 1 if f >= min_gb else 0
        if ok >= n_ok:
            return f
        time.sleep(every_s)


def run(cmd: list, env: dict, logfile: str) -> int:
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


class ScorePool:
    """CPU scorers, up to ``max_par`` at once, each launched only after a RAM gate; a job whose
    counts already read PASS is skipped; a RAM-guard abort is retried (up to 8)."""

    def __init__(self, max_par: int, env: dict, qlog: str):
        self.max_par, self.env, self.qlog = max_par, env, qlog
        self.queue: list = []
        self.live: list = []

    def add(self, job: dict) -> None:
        if jget(job["counts"], "status") == "PASS":
            log(f"SCORE {job['name']} PASS already", self.qlog)
            return
        job["tries"] = 0
        self.queue.append(job)

    #: ⚠️ 8 GB, not 6: MEASURED 2026-09-28 05:50-05:58 Berlin, other sessions' jobs held the box at
    #: 1.8-3.4 GB free and E1's guard (3 GB floor) aborted this stream's scorers twice each.
    GATE_GB = float(os.environ.get("R7_SCORE_RAM_GB", "8"))

    def _launch(self, job: dict) -> None:
        f = free_gb()
        job["tries"] += 1
        fh = open(job["driver_log"], "a", encoding="utf-8")
        fh.write("CMD: " + " ".join(job["cmd"]) + "\n")
        fh.flush()
        job["proc"] = subprocess.Popen(job["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=self.env)
        job["fh"] = fh
        log(f"SCORE {job['name']} launched try={job['tries']} (free {f:.1f} GB)", self.qlog)

    def poll(self) -> None:
        for job in list(self.live):
            if job["proc"].poll() is None:
                continue
            job["fh"].close()
            self.live.remove(job)
            st = jget(job["counts"], "status")
            aborted = False
            lg = job.get("scorer_log")
            if lg and os.path.exists(lg):
                txt = open(lg, encoding="utf-8", errors="replace").read()
                aborted = ("RAM_GUARD" in txt) or ("E1_RAM_GUARD_ABORT" in txt)
            log(f"SCORE {job['name']} ended counts={st} ram_guard_abort={aborted}", self.qlog)
            if st != "PASS" and aborted and job["tries"] < 8:
                self.queue.append(job)
        # NON-BLOCKING: at most one launch per poll, and only while RAM is above the gate on this
        # sample AND the previous one (the runner's GPU bridges never wait on the scorers' RAM)
        f = free_gb()
        ok = f >= self.GATE_GB and getattr(self, "_last_free", 0.0) >= self.GATE_GB
        self._last_free = f
        if self.queue and len(self.live) < self.max_par and ok:
            job = self.queue.pop(0)
            self._launch(job)
            self.live.append(job)

    def drain(self) -> None:
        while self.queue or self.live:
            self.poll()
            time.sleep(20)


def gpu_for_split(sk: str, wait_s: int, qlog: str, device: str):
    """-> (device, token). ``device`` 'cpu' forces CPU; 'cuda'/'auto' try the lock."""
    if device == "cpu":
        return "cpu", None
    tok = gpu_lock.acquire(f"refcv7-navsim-{sk}", wait_s, log=lambda m: None)
    if tok:
        log(f"GPU lock ACQUIRED for {sk} ({tok})", qlog)
        return "cuda", tok
    log(f"GPU lock NOT acquired within {wait_s} s for {sk} -> CPU fp32 for the whole split", qlog)
    return "cpu", None


def bridge_cmd(sp: dict, sk: str, arms: str, ckpt: str, md5: str, dev: str, tok, label: str,
               out: str, threads: int, tokens_file: str = "", derived: bool = True,
               dedup: bool = True) -> list:
    cmd = [PY, os.path.join(HERE, "run_bridge7.py"), "--split", sp["split"], "--arms", arms,
           "--inputs", sp["inputs"], "--speed", sp["speed"], "--road-plane", ROAD,
           "--ckpt", ckpt, "--config", CONFIG, "--ckpt-md5", md5, "--device", dev,
           "--precision", "auto", "--threads", str(threads), "--label", label, "--out", out]
    if dedup:
        cmd.append("--exact-dedup")
    if derived:
        cmd.append("--derived")
    if sp["bank2"]:
        cmd += ["--bank2", sp["bank2"]]
    if sp["bank1"]:
        cmd += ["--bank1", sp["bank1"]]
    if sk == "navtest":
        cmd += ["--bank1-kind", "navtest", "--logs-root", sp["logs_root"]]
        if "R7_VMAXORACLE" in arms.split(","):
            cmd += ["--speed-oracle", sp["speed_oracle"]]
    if tokens_file:
        cmd += ["--tokens-file", tokens_file]
    if dev == "cuda":
        cmd += ["--gpu-lock-token", tok]
    return cmd


def seams_ready(bdir: str, arms: list) -> dict:
    out = {}
    for arm in arms:
        seam = os.path.join(bdir, f"seam_{arm}.npz")
        man = os.path.join(bdir, f"seam_{arm}.manifest.json")
        if not (os.path.exists(seam) and os.path.exists(man)):
            out[arm] = "NO_SEAM"
        elif jget(man, "partial"):
            out[arm] = "PARTIAL_SEAM"
        else:
            out[arm] = "OK"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--md5", default="", help="expected md5 (refused on mismatch)")
    ap.add_argument("--splits", default="warmup,navtest,navhard")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-par", type=int, default=4)
    ap.add_argument("--no-diag", action="store_true", help="skip navtest's 200-token diagnostics")
    ap.add_argument("--gpu-wait-s", type=int, default=-1,
                    help="override every split's wait for the GPU lock before CPU (default: per split)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if os.environ.get("R7_MUTATION"):
        sys.exit("⛔ R7_MUTATION is set -- a mutation-test environment never scores a checkpoint")
    if not os.path.exists(a.ckpt):
        sys.exit(f"⛔ checkpoint missing: {a.ckpt}")
    import torch
    step = int(torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True).get("step", -1))
    label = "RESULT" if step >= 5000 else "VALIDATION ONLY (SPEC §6)"
    out = a.out or os.path.join(PKG, "raw", "milestones", f"step{step}")
    os.makedirs(out, exist_ok=True)
    qlog = os.path.join(out, "runner.log")
    md5 = md5_file(a.ckpt)
    if a.md5 and md5 != a.md5:
        sys.exit(f"⛔ md5 {md5} != --md5 {a.md5}")
    cfg_md5 = md5_file(CONFIG)
    log(f"START ckpt={a.ckpt} step={step} md5={md5} label={label} config_md5={cfg_md5}", qlog)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{TREE}/stack;{TREE}/taniteval"
    env["TANITAD_REPO"] = TREE
    env["OMP_NUM_THREADS"] = str(a.threads)
    env["PYTHONIOENCODING"] = "utf-8"
    pool = ScorePool(a.max_par, env, qlog)
    summary = {"ckpt": a.ckpt, "md5": md5, "step": step, "label": label,
               "config_md5": cfg_md5, "splits": {}}
    splits = [s for s in a.splits.split(",") if s]
    post = []                                   # (split, bridge dirs, arms) parsed after drain
    for sk in splits:
        sp = SPLITS[sk]
        dev, tok = gpu_for_split(sk, sp["gpu_wait_s"] if a.gpu_wait_s < 0 else a.gpu_wait_s,
                                 qlog, a.device)
        e = dict(env)
        rec = {"device": dev, "gpu_lock_token": tok}
        try:
            if dev == "cuda":
                ce = dict(e, R7_TEST_DEVICE="cuda", R7_TEST_CKPT=a.ckpt, R7_TEST_CONFIG=CONFIG)
                clog = os.path.join(out, f"cuda_controls_{sk}.log")
                rc = run([PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
                          os.path.join(PKG, "tests", "test_model_seam7.py"), "-k", "K0 or KD"],
                         ce, clog)
                txt = open(clog, encoding="utf-8", errors="replace").read().replace("\\", "/")
                # prefix-agnostic (pytest_control7.py): the literal "PASSED tests/..." match read a PASSED
                # K0 as FAILED at step 50,400 and refused CUDA on two splits (MEASURED 2026-10-04)
                k0 = control_passed(txt, "test_K0")
                kd = control_passed(txt, "test_KD")
                rec["cuda_controls"] = {"pytest_rc": rc, "K0_pass": k0, "KD_pass": kd,
                                        "KD": jget(os.path.join(PKG, "raw", "controls",
                                                                "KD_exact_dedup_cuda.json"))}
                log(f"CUDA controls {sk}: K0={k0} KD={kd}", qlog)
                # the refcv6 suite's rule (its D15): a K0 (determinism) failure REFUSES CUDA -- the
                # seed replicate and common random numbers rest on it; a KD failure only DROPS the
                # eval-only dedup lever (the trunk's native path runs). MEASURED 2026-09-28 on the
                # step-1,500 checkpoint: CUDA bf16 exact-dedup vs native = selections 4/4 identical,
                # max |d| 1.196 mm > the registered 1 mm -> native path on CUDA.
                rec["exact_dedup"] = bool(kd)
                if not k0:
                    log(f"CUDA REFUSED for {sk}: K0 failed on CUDA -> CPU", qlog)
                    gpu_lock.release(tok)
                    dev, tok = "cpu", None
                    rec["device"] = "cpu"
                elif not kd:
                    log(f"KD failed on CUDA for {sk}: the exact dedup is DROPPED (native path)", qlog)
            if dev == "cpu":
                e["CUDA_VISIBLE_DEVICES"] = "-1"
                f = ram_wait(6.0)
                log(f"RAM gate passed for the CPU bridge of {sk} ({f:.1f} GB)", qlog)
            bdir = os.path.join(out, f"bridge_{sk}")
            log(f"BRIDGE {sk} arms={sp['arms']} device={dev}", qlog)
            dedup = True if dev == "cpu" else bool(rec.get("exact_dedup", False))
            rec["exact_dedup"] = dedup
            run(bridge_cmd(sp, sk, sp["arms"], a.ckpt, md5, dev, tok, label, bdir, a.threads,
                           dedup=dedup), e, bdir + ".log")
            dirs = {"main": (bdir, sp["arms"].split(",") + list(DERIVED))}
            if sk == "navtest" and not a.no_diag:
                ddir = os.path.join(out, "bridge_navtest_sub200")
                log(f"BRIDGE navtest sub200 arms={sp['diag_arms']} device={dev}", qlog)
                run(bridge_cmd(sp, sk, sp["diag_arms"], a.ckpt, md5, dev, tok, label, ddir,
                               a.threads, tokens_file=SUB200, derived=False, dedup=dedup),
                    e, ddir + ".log")
                dirs["sub200"] = (ddir, sp["diag_arms"].split(","))
        finally:
            if tok:
                log(f"GPU lock RELEASED={gpu_lock.release(tok)} after {sk}", qlog)
        rec["seams"] = {k: seams_ready(d, arms) for k, (d, arms) in dirs.items()}
        log(f"SEAMS {sk} {rec['seams']}", qlog)
        # ---- the scorers for this split, into the pool (they run while the next split bridges)
        sdir = os.path.join(out, f"scores_{sk}")
        os.makedirs(sdir, exist_ok=True)
        for kind, (d, arms) in dirs.items():
            for arm in arms:
                if rec["seams"][kind].get(arm) != "OK":
                    continue
                seam = os.path.join(d, f"seam_{arm}.npz")
                if sk == "navtest":
                    lab = f"r7s{step}_{arm}" + ("_sub" if kind == "sub200" else "")
                    cmd = [PY, os.path.join(HERE, "score_navtest7.py"), "--label", lab, "--seam",
                           seam, "--out", sdir] + (["--tokens", SUB200] if kind == "sub200" else [])
                    pool.add({"name": f"{sk}:{lab}", "cmd": cmd,
                              "counts": os.path.join(sdir, lab, f"{lab}.counts.json"),
                              "scorer_log": os.path.join(sdir, lab, f"{lab}.log"),
                              "driver_log": os.path.join(sdir, f"{lab}.driver.txt")})
                else:
                    tag = arm if sp["split"] == "warmup_two_stage" else f"{arm}__{sp['split']}"
                    cmd = [NPY, os.path.join(HERE, "score_arm7.py"), "--arm", arm, "--seam", seam,
                           "--split", sp["split"], "--out", sdir, "--exp-tag", f"e7s{step}"]
                    pool.add({"name": f"{sk}:{arm}", "cmd": cmd,
                              "counts": os.path.join(sdir, f"score_{tag}.counts.json"),
                              "scorer_log": os.path.join(sdir, f"score_{tag}.log"),
                              "driver_log": os.path.join(sdir, f"{arm}.driver.txt")})
        pool.poll()
        summary["splits"][sk] = rec
        post.append((sk, dirs))
    log("all bridges done; draining the scorer pool", qlog)
    pool.drain()
    for sk, dirs in post:
        sp = SPLITS[sk]
        rec = summary["splits"][sk]
        bdir = dirs["main"][0]
        sdir = os.path.join(out, f"scores_{sk}")
        summ = os.path.join(out, f"summary_{sk}.json")
        dec = os.path.join(out, f"decomposition_{sk}.json")
        run([PY, os.path.join(HERE, "plan_deltas7.py"), "--bridge", bdir, "--label", label,
             "--out", os.path.join(out, f"plan_deltas_{sk}.json")], env,
            os.path.join(out, f"plan_deltas_{sk}.log"))
        if sk == "navtest":
            def csv_of(arm, sub=False):
                lab = f"r7s{step}_{arm}" + ("_sub" if sub else "")
                c = os.path.join(sdir, lab, f"{lab}.csv")
                ok = jget(os.path.join(sdir, lab, f"{lab}.counts.json"), "status") == "PASS"
                return c if ok and os.path.exists(c) else None
            a1 = csv_of("R7_A1")
            if a1:
                extra = [f"{arm}={csv_of(arm)}" for arm in ("R7_A1_s1",) + DERIVED if csv_of(arm)]
                run([PY, os.path.join(HERE, "parse_navtest7.py"), "--arm-csv", a1, "--label", label,
                     "--out", summ, "--bridge", bdir] + (["--extra"] + extra if extra else []),
                    env, os.path.join(out, "parse_navtest.log"))
                run([PY, os.path.join(HERE, "decompose7.py"), "--split", "navtest", "--arm-csv", a1,
                     "--inputs", sp["inputs"], "--label", label, "--out", dec], env,
                    os.path.join(out, "decompose_navtest.log"))
                if "sub200" in dirs:
                    sub_extra = [f"{arm}={csv_of(arm, True)}" for arm in dirs["sub200"][1]
                                 if csv_of(arm, True)]
                    run([PY, os.path.join(HERE, "parse_navtest7.py"), "--arm-csv", a1,
                         "--tokens", SUB200, "--label", label + " -- 200-token DIAGNOSTIC",
                         "--out", os.path.join(out, "summary_navtest_sub200.json"),
                         "--bridge", dirs["sub200"][0], "--census", sp["speed_oracle"],
                         "--map", sp["speed"]] + (["--extra"] + sub_extra if sub_extra else []),
                        env, os.path.join(out, "parse_navtest_sub200.log"))
            else:
                log("navtest R7_A1 has no PASSING score -- no summary (never a partial CSV)", qlog)
        else:
            floors = os.path.join(PKG, "raw", "floors", sp["split"])
            suffix = [] if sp["split"] == "warmup_two_stage" else ["--csv-suffix", f"__{sp['split']}"]
            run([PY, os.path.join(HERE, "parse7.py"), "--split", sp["split"], "--scores", sdir,
                 "--floors", floors, "--bridge", bdir, "--inputs", sp["inputs"], "--label", label,
                 "--out", summ] + suffix, env, os.path.join(out, f"parse_{sk}.log"))
            run([PY, os.path.join(HERE, "decompose7.py"), "--split", sp["split"], "--scores", sdir,
                 "--floors", floors, "--bridge", bdir, "--inputs", sp["inputs"], "--label", label,
                 "--out", dec] + suffix, env, os.path.join(out, f"decompose_{sk}.log"))
        if sk in ("navhard", "navtest"):
            for arm in ("R7_A1", "PRIOR_ha0p"):
                sm = os.path.join(bdir, f"seam_{arm}.npz")
                if os.path.exists(sm):
                    run([PY, os.path.join(HERE, "families7.py"), "--seam", sm, "--inputs",
                         sp["inputs"], "--stage", "1", "--label", label, "--out",
                         os.path.join(out, f"families_{sk}_{arm}.json")], env,
                        os.path.join(out, f"families_{sk}.log"))
        rec["summary"] = summ if os.path.exists(summ) else None
        rec["decomposition"] = dec if os.path.exists(dec) else None
        log(f"SPLIT {sk} parsed: summary={'present' if rec['summary'] else 'ABSENT'}", qlog)
    msp = os.path.join(out, "MILESTONE_SUMMARY.json")
    if os.path.exists(msp):
        prev = json.load(open(msp, encoding="utf-8"))
        if prev.get("md5") != summary["md5"]:
            sys.exit(f"⛔ {msp} belongs to md5 {prev.get('md5')}, not {summary['md5']} -- refused")
        merged = dict(prev.get("splits", {}))
        merged.update(summary["splits"])
        summary["splits"] = merged
    with open(msp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    run([PY, os.path.join(HERE, "bars7.py"), "--milestone", out], env, os.path.join(out, "bars.log"))
    log(f"DONE bars={'present' if os.path.exists(os.path.join(out, 'BARS.json')) else 'ABSENT'}",
        qlog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
