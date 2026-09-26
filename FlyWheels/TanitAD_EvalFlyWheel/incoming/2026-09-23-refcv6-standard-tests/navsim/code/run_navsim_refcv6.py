#!/usr/bin/env python3
"""ONE COMMAND: a refcv6 checkpoint through the whole NavSim suite (any python; TANITAD venv).

    python code/run_navsim_refcv6.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_5000.pt
    python code/run_navsim_refcv6.py --fetch 5000            # scp read-only from Thor + md5, then run
    python code/run_navsim_refcv6.py --ckpt <p> --splits warmup --device cuda --gpu-wait-s 7200

What it does, each step its OWN process, each gated on the ARTIFACT of the step before (never on an
exit code — CLAUDE.md: "the admissible evidence that this gate did not run is the MISSING JSON"):

  0. (``--fetch N``) ``scp`` the milestone from Thor READ-ONLY (``ckpt_<N>.pt`` from the run dir, or
     the snapshot dir's full copy), md5 compared against ``md5sum`` run ON Thor; nothing runs there;
  1. md5 of the checkpoint recorded; ``config.json`` = the run's (``--config``, default the kit's);
  2. per split: ``run_bridge6.py`` (the arms, resumable rows) -> seams + declared-input manifests;
  3. the official scorer per arm (``score_arm6.py`` two-stage / ``score_navtest6.py`` navtest),
     each refused unless its count guards PASS;
  4. the statistics (``parse6.py`` / ``parse_navtest6.py``) against the floors;
  5. ``MILESTONE_SUMMARY.json``: every number with its label — ``PIPELINE-VALIDATION`` below
     step 5,000 (SPEC §6), ``RESULT`` from there — and the SPEC §5 bars evaluated as written.

Device: ``auto`` tries the GPU gate (memory.used < 4300 MiB, no other python in compute-apps, free
RAM >= 8 GB) for ``--gpu-wait-s`` and falls back to CPU fp32 — ONE device for all arms of a split,
recorded; the bridge refuses a mixture.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
NPY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
REPO_DEFAULT = os.environ.get("TANITAD_REPO", "C:/Users/Admin/ev6")
THOR = "tanitad-thor-wifi"
THOR_RUN = "/home/nvidia/refcv6_run/runs/refcv6-r101-s0"
THOR_SNAP = "/home/nvidia/refcv6_run/snapshots/refcv6-r101-s0"
KIT = "D:/refcv6_eval_kit/ckpt"
BANKS = "C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923"
NAVTEST_BANK = "D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank"
E2RAW = os.path.join(PKG, "..", "..", "2026-09-19-navsim-refcv4b-bridge", "raw")
SPLITS = {
    "warmup": {"split": "warmup_two_stage",
               "arms": "R6_A1,R6_A1_s1,R6_BLIND,R6_NAVOFF,R6_VMAXOFF,R6_A1NT",
               "inputs": os.path.join(E2RAW, "navsim_agent_inputs.json"),
               "speed": os.path.join(PKG, "raw", "inputs", "speed_limits_warmup_two_stage.json"),
               "bank2": f"{BANKS}/warmup_two_stage", "bank1": ""},
    "navhard": {"split": "navhard_two_stage", "arms": "R6_A1,R6_A1_s1",
                "inputs": "C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/"
                          "navsim_agent_inputs.json",
                "speed": os.path.join(PKG, "raw", "inputs", "speed_limits_navhard_two_stage.json"),
                "bank2": f"{BANKS}/navhard_s2", "bank1": f"{BANKS}/navhard_s1"},
    "navtest": {"split": "navtest", "arms": "R6_A1",
                "inputs": "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz",
                "speed": os.path.join(PKG, "raw", "inputs", "speed_limits_navtest.json"),
                "bank2": "", "bank1": NAVTEST_BANK,
                "logs_root": "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test",
                # SPEC §12: the PRIVILEGED diagnostic arm's input (code/vmax_oracle.py)
                "speed_oracle": os.path.join(PKG, "raw", "inputs", "vmax_oracle_navtest.json")},
}
ROAD = os.path.join(PKG, "raw", "inputs", "road_plane_navhard_warmup_logs.json")


def log(msg: str, path: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def md5_local(p: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch(step: int, dst_dir: str, qlog: str) -> str:
    """scp a milestone READ-ONLY from Thor and md5-verify against md5sum run on Thor."""
    cands = [f"{THOR_RUN}/ckpt_{step}.pt", f"{THOR_SNAP}/ckpt_{step}.pt",
             f"{THOR_SNAP}/ckpt_step{step}.pt"]
    for c in cands:
        r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", THOR,
                            f"md5sum {c}"], capture_output=True, text=True, timeout=600)
        m = re.match(r"^([0-9a-f]{32})\s", r.stdout.strip())
        if not m:
            continue
        want = m.group(1)
        dst = os.path.join(dst_dir, f"ckpt_{step}.pt")
        if not (os.path.exists(dst) and md5_local(dst) == want):
            log(f"FETCH scp {THOR}:{c} -> {dst}", qlog)
            subprocess.run(["scp", "-o", "BatchMode=yes", f"{THOR}:{c}", dst], timeout=7200)
        got = md5_local(dst)
        if got != want:
            raise SystemExit(f"⛔ md5 {got} != Thor's {want} for {c}")
        with open(os.path.join(dst_dir, "MD5SUMS"), "a", encoding="utf-8") as fh:
            fh.write(f"{want}  ckpt_{step}.pt  (from {THOR}:{c})\n")
        log(f"FETCH OK {dst} md5 {want}", qlog)
        return dst
    raise SystemExit(f"⛔ no ckpt for step {step} on Thor ({cands})")


def thor_md5(path: str) -> str:
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", THOR,
                        f"md5sum {path}"], capture_output=True, text=True, timeout=900)
    m = re.match(r"^([0-9a-f]{32})\s", r.stdout.strip())
    return m.group(1) if m else ""


def fetch_final(dst_dir: str, qlog: str) -> str:
    """The FINAL model (``ckpt.pt`` in the run dir), READ-ONLY, only once the run's own
    ``summary.json`` reads ``"done": true``. md5 on Thor BEFORE and AFTER the copy (they must agree
    — the file must not be being rewritten) and on the dev box (must equal both)."""
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", THOR,
                        f"cat {THOR_RUN}/summary.json"], capture_output=True, text=True, timeout=120)
    try:
        sm = json.loads(r.stdout)
    except Exception:                                                     # noqa: BLE001
        raise SystemExit(f"⛔ {THOR_RUN}/summary.json absent or unreadable — the run is not done")
    if sm.get("done") is not True:
        raise SystemExit(f"⛔ summary.json does not read done: true ({str(sm)[:200]})")
    src = f"{THOR_RUN}/ckpt.pt"
    before = thor_md5(src)
    if len(before) != 32:
        raise SystemExit(f"⛔ md5 of {src} on Thor unreadable")
    dst = os.path.join(dst_dir, f"ckpt_final_step{int(sm.get('step', -1))}.pt")
    if not (os.path.exists(dst) and md5_local(dst) == before):
        log(f"FETCH-FINAL scp {THOR}:{src} -> {dst} (summary step {sm.get('step')})", qlog)
        subprocess.run(["scp", "-o", "BatchMode=yes", f"{THOR}:{src}", dst], timeout=10800)
    after = thor_md5(src)
    got = md5_local(dst)
    if not (before == after == got):
        raise SystemExit(f"⛔ md5 before {before} / after {after} / dev box {got} disagree")
    with open(os.path.join(dst_dir, "MD5SUMS"), "a", encoding="utf-8") as fh:
        fh.write(f"{got}  {os.path.basename(dst)}  (from {THOR}:{src}; FINAL, summary.json done, "
                 f"md5 before == after == dev box)\n")
    log(f"FETCH-FINAL OK {dst} md5 {got}", qlog)
    return dst


def gate_ok() -> bool:
    sys.path.insert(0, HERE)
    import importlib.util
    spec = importlib.util.spec_from_file_location("rb6_gate", os.path.join(HERE, "run_bridge6.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return bool(m.gpu_gate()["ok"])


def run(cmd: list, env: dict, logfile: str) -> int:
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def ram_wait(min_gb: float, n_ok: int = 3, every_s: int = 30) -> float:
    """Block until free host RAM >= min_gb on n_ok consecutive samples (the box is shared)."""
    import psutil
    ok = 0
    while True:
        f = psutil.virtual_memory().available / 2**30
        ok = ok + 1 if f >= min_gb else 0
        if ok >= n_ok:
            return f
        time.sleep(every_s)


def counts_pass(path: str) -> bool:
    try:
        return json.load(open(path, encoding="utf-8")).get("status") == "PASS"
    except Exception:                                                     # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--fetch", type=int, default=0, help="milestone step to scp from Thor")
    ap.add_argument("--fetch-final", action="store_true",
                    help="the FINAL ckpt.pt, only after the run's summary.json reads done: true")
    ap.add_argument("--config", default=f"{KIT}/config.json")
    ap.add_argument("--md5", default="")
    ap.add_argument("--splits", default="warmup,navhard,navtest")
    ap.add_argument("--arms", default="", help="override: split=a,b;split=c")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--gpu-wait-s", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tokens-navtest", default="", help="W3-format subset file for navtest")
    ap.add_argument("--out", default="")
    ap.add_argument("--repo", default=REPO_DEFAULT)
    a = ap.parse_args(argv)
    ckpt = a.ckpt
    os.makedirs(os.path.join(PKG, "raw", "milestones"), exist_ok=True)
    tmp_log = os.path.join(PKG, "raw", "milestones", "runner.log")
    if a.fetch:
        ckpt = fetch(a.fetch, KIT, tmp_log)
    elif a.fetch_final:
        ckpt = fetch_final(KIT, tmp_log)
    if not ckpt or not os.path.exists(ckpt):
        sys.exit(f"⛔ checkpoint missing: {ckpt!r}")
    import torch
    step = int(torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True).get("step", -1))
    label = "RESULT" if step >= 5000 else "PIPELINE-VALIDATION"
    out = a.out or os.path.join(PKG, "raw", "milestones", f"step{step}")
    os.makedirs(out, exist_ok=True)
    qlog = os.path.join(out, "runner.log")
    md5 = md5_local(ckpt)
    if a.md5 and md5 != a.md5:
        sys.exit(f"⛔ md5 {md5} != --md5 {a.md5}")
    log(f"START ckpt={ckpt} step={step} md5={md5} label={label}", qlog)
    over = {}
    for part in [p for p in a.arms.split(";") if p]:
        k, v = part.split("=", 1)
        over[k] = v
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{a.repo}/stack;{a.repo}/taniteval"
    env["OMP_NUM_THREADS"] = str(a.threads)
    env["TANITAD_REPO"] = a.repo
    summary = {"ckpt": ckpt, "md5": md5, "step": step, "label": label, "splits": {}}
    for sk in [s for s in a.splits.split(",") if s]:
        sp = SPLITS[sk]
        arms = over.get(sk, sp["arms"])
        dev = a.device
        if dev == "auto":
            # ⚠️ MEASURED 2026-09-26: with the box at 97 % CPU (five sessions) a CPU bridge ran at
            # 4.6 s/scene — a full navtest split (12,146) ≈ 15 h and navhard (2 × 5,912) ≈ 16 h on CPU
            # vs ~1 h / ~2.5 h on the card. For those two EXPENSIVE splits a longer wait for the
            # brief's gate (default 3 h, env R6_EXPENSIVE_GPU_WAIT_S) bounds the downside at +3 h and
            # wins ~10 h whenever the card frees; the cheap splits keep --gpu-wait-s.
            wait_s = a.gpu_wait_s
            if (sk == "navhard" or (sk == "navtest" and not a.tokens_navtest)):
                wait_s = max(wait_s, int(os.environ.get("R6_EXPENSIVE_GPU_WAIT_S", "10800")))
            t0 = time.time()
            ok = gate_ok()
            if not ok:
                log(f"GPU gate closed at the start of {sk}: waiting up to {wait_s} s", qlog)
            while not ok and time.time() - t0 < wait_s:
                time.sleep(60)
                ok = gate_ok()
            dev = "cuda" if ok else "cpu"
            log(f"device for {sk}: {dev} (gate {'open' if ok else 'closed'} after "
                f"{int(time.time() - t0)} s)", qlog)
        elif dev == "cuda":
            # an EXPLICIT --device cuda still obeys the brief's gate before ANY CUDA work (the K0/KD
            # pytest included): wait up to --gpu-wait-s, else fall back to CPU and say so
            t0 = time.time()
            ok = gate_ok()
            while not ok and time.time() - t0 < a.gpu_wait_s:
                time.sleep(60)
                ok = gate_ok()
            if not ok:
                log(f"GPU gate closed for {a.gpu_wait_s} s on an explicit --device cuda -> CPU", qlog)
                dev = "cpu"
        e = dict(env)
        if dev == "cpu":
            e["CUDA_VISIBLE_DEVICES"] = "-1"
        dedup = True
        if dev == "cuda" and "cuda_controls" not in summary:
            # SPEC §2/§7: on CUDA the exact dedup (KD) and determinism (K0) are RE-MEASURED
            # before any arm; a KD miss drops the lever (the trunk's native path runs instead).
            ce = dict(e, R6_TEST_DEVICE="cuda", R6_TEST_CKPT=ckpt, R6_TEST_CONFIG=a.config)
            clog = os.path.join(out, "cuda_controls.log")
            n0 = os.path.getsize(clog) if os.path.exists(clog) else 0
            rc = run([PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
                      os.path.join(PKG, "tests", "test_model_seam6.py"), "-k", "K0 or KD"],
                     ce, clog)
            with open(clog, encoding="utf-8", errors="replace") as fh:
                fh.seek(n0)
                ctext = fh.read()
            k0_pass = "PASSED tests/test_model_seam6.py::test_K0" in ctext.replace("\\", "/")
            kd = os.path.join(PKG, "raw", "controls", "KD_exact_dedup_cuda.json")
            kdj = json.load(open(kd, encoding="utf-8")) if os.path.exists(kd) else None
            ok_kd = bool(kdj) and kdj["sel_identical"] == kdj["n"] and kdj["max_abs_traj_diff_m"] < 1e-3
            summary["cuda_controls"] = {"pytest_rc": rc, "KD": kdj, "KD_pass": ok_kd,
                                        "K0_pass": k0_pass}
            if os.path.exists(kd):
                os.replace(kd, os.path.join(out, "KD_exact_dedup_cuda.json"))
            log(f"CUDA controls rc={rc} K0_pass={k0_pass} KD_pass={ok_kd}", qlog)
        if dev == "cuda" and not summary["cuda_controls"].get("K0_pass", True):
            # same seed => same plan is what common random numbers and the seed replicate rest on
            log(f"CUDA REFUSED for {sk}: K0 (determinism) failed on CUDA -> CPU", qlog)
            dev = "cpu"
            e["CUDA_VISIBLE_DEVICES"] = "-1"
        if dev == "cuda":
            dedup = bool(summary["cuda_controls"]["KD_pass"])
        bdir = os.path.join(out, f"bridge_{sk}")
        cmd = [PY, os.path.join(HERE, "run_bridge6.py"), "--split", sp["split"], "--arms", arms,
               "--inputs", sp["inputs"], "--speed", sp["speed"], "--road-plane", ROAD,
               "--ckpt", ckpt, "--config", a.config, "--ckpt-md5", md5, "--device", dev,
               "--precision", "auto", "--threads", str(a.threads), "--out", bdir]
        if dedup:
            cmd.append("--exact-dedup")
        if sp["bank2"]:
            cmd += ["--bank2", sp["bank2"]]
        if sp["bank1"]:
            cmd += ["--bank1", sp["bank1"]]
        if "R6_VMAXORACLE" in arms.split(","):
            if not sp.get("speed_oracle"):
                sys.exit(f"⛔ R6_VMAXORACLE has no oracle input on {sk} (SPEC §12: navtest only)")
            cmd += ["--speed-oracle", sp["speed_oracle"]]
        if sk == "navtest":
            cmd += ["--bank1-kind", "navtest", "--logs-root", sp["logs_root"]]
            if a.tokens_navtest:
                cmd += ["--tokens-file", a.tokens_navtest]
        if dev == "cuda":
            cmd += ["--gpu-wait-s", str(a.gpu_wait_s)]
        if dev == "cpu":
            # the box is shared: never start into a squeeze. 6 GB, not 5: a CPU bridge takes
            # ~2.0-2.4 GB once the model is built, and E1's guard aborts every scorer on the box
            # after 120 s below 3 GB (MEASURED: 3 aborts on 2026-09-23/24)
            free = ram_wait(6.0)
            log(f"RAM gate passed ({free:.1f} GB free)", qlog)
        log(f"BRIDGE {sk} arms={arms} device={dev}", qlog)
        run(cmd, e, os.path.join(bdir + ".log"))
        refused = os.path.join(bdir, "GPU_GATE_REFUSED.json")
        if dev == "cuda" and os.path.exists(refused) and not any(
                f.startswith("rows_") for f in os.listdir(bdir)):
            # MEASURED 2026-09-24 (warmup@5000): the runner's gate passed, the bridge's own gate then
            # waited --gpu-wait-s and refused, and the split was scored on NOTHING. No row exists on
            # CUDA, so re-running on CPU keeps ONE device for the split.
            os.replace(refused, os.path.join(bdir, f"GPU_GATE_REFUSED_{int(time.time())}.json"))
            dev, dedup = "cpu", True
            e = dict(env, CUDA_VISIBLE_DEVICES="-1")
            cmd = [c for c in cmd if c != "--exact-dedup"]
            i = cmd.index("--device")
            cmd[i + 1] = "cpu"
            if "--gpu-wait-s" in cmd:
                j = cmd.index("--gpu-wait-s")
                del cmd[j:j + 2]
            cmd.append("--exact-dedup")
            free = ram_wait(6.0)
            log(f"BRIDGE {sk} FELL BACK to CPU after a refused CUDA gate (no CUDA row written; "
                f"{free:.1f} GB free)", qlog)
            run(cmd, e, os.path.join(bdir + ".log"))
        if os.path.exists(os.path.join(bdir, "rows_R6_A1.jsonl")) and "," in arms:
            # label-free lever reads (plans vs R6_A1, against the seed floor) — before any scorer
            run([PY, os.path.join(HERE, "plan_deltas.py"), "--bridge", bdir, "--label", label,
                 "--out", os.path.join(out, f"plan_deltas_{sk}.json")], env,
                os.path.join(out, f"plan_deltas_{sk}.log"))
        sdir = os.path.join(out, f"scores_{sk}")
        os.makedirs(sdir, exist_ok=True)
        recs = {}
        for arm in arms.split(","):
            seam = os.path.join(bdir, f"seam_{arm}.npz")
            man = os.path.join(bdir, f"seam_{arm}.manifest.json")
            if not (os.path.exists(seam) and os.path.exists(man)):
                recs[arm] = {"status": "NO_SEAM"}
                log(f"SCORE {sk} {arm} SKIPPED — no seam", qlog)
                continue
            if json.load(open(man, encoding="utf-8")).get("partial"):
                recs[arm] = {"status": "PARTIAL_SEAM"}
                log(f"SCORE {sk} {arm} REFUSED — partial seam", qlog)
                continue
            if sk == "navtest":
                lab = f"r6s{step}_{arm}" + ("_sub" if a.tokens_navtest else "")
                cnt = os.path.join(sdir, lab, f"{lab}.counts.json")
                for _try in range(6):               # RAM-gated, retried on the RAM guard's abort
                    if counts_pass(cnt):
                        break
                    ram_wait(4.0)
                    sc = [PY, os.path.join(HERE, "score_navtest6.py"), "--label", lab, "--seam",
                          seam, "--out", sdir] + (["--tokens", a.tokens_navtest]
                                                  if a.tokens_navtest else [])
                    run(sc, env, os.path.join(sdir, f"{arm}.driver.txt"))
                    lg = os.path.join(sdir, lab, f"{lab}.log")
                    if not (os.path.exists(lg) and "RAM_GUARD" in open(
                            lg, encoding="utf-8", errors="replace").read()):
                        break
                    log(f"SCORE {sk} {arm} RAM-guard abort, retry {_try + 1}", qlog)
                recs[arm] = {"status": "PASS" if counts_pass(cnt) else "FAIL", "counts": cnt}
            else:
                tag = arm if sp["split"] == "warmup_two_stage" else f"{arm}__{sp['split']}"
                cnt = os.path.join(sdir, f"score_{tag}.counts.json")
                if not counts_pass(cnt):
                    # the RAM-gated, retrying queue (score_queue6.sh), one arm
                    run(["bash", os.path.join(HERE, "score_queue6.sh"), sp["split"], sdir,
                         f"e6s{step}", f"{arm}={seam}"], env,
                        os.path.join(sdir, f"{arm}.queue.txt"))
                recs[arm] = {"status": "PASS" if counts_pass(cnt) else "FAIL", "counts": cnt}
            log(f"SCORE {sk} {arm} {recs[arm]['status']}", qlog)
        summ = os.path.join(out, f"summary_{sk}.json")
        dec = os.path.join(out, f"decomposition_{sk}.json")
        if sk == "navtest":
            sub = "_sub" if a.tokens_navtest else ""
            lab = f"r6s{step}_R6_A1{sub}"
            csv = os.path.join(sdir, lab, f"{lab}.csv")
            extras = []
            for arm in arms.split(","):
                la = f"r6s{step}_{arm}{sub}"
                ca = os.path.join(sdir, la, f"{la}.csv")
                if arm != "R6_A1" and os.path.exists(ca):
                    extras.append(f"{arm}={ca}")
            if os.path.exists(csv):
                run([PY, os.path.join(HERE, "parse_navtest6.py"), "--arm-csv", csv, "--label",
                     label, "--out", summ, "--bridge", bdir] + (["--extra"] + extras if extras else [])
                    + (["--tokens", a.tokens_navtest] if a.tokens_navtest else [])
                    + (["--census", sp["speed_oracle"], "--map", sp["speed"]]
                       if os.path.exists(sp.get("speed_oracle", "")) else []),
                    env, os.path.join(out, "parse_navtest.log"))
                # SPEC §5's ladder (per command, per speed band) — run whatever the bar says
                run([PY, os.path.join(HERE, "decompose6.py"), "--split", "navtest", "--arm-csv", csv,
                     "--inputs", sp["inputs"], "--label", label, "--out", dec], env,
                    os.path.join(out, "decompose_navtest.log"))
        else:
            floors = os.path.join(PKG, "raw", "floors", sp["split"])
            suffix = [] if sp["split"] == "warmup_two_stage" else ["--csv-suffix", f"__{sp['split']}"]
            run([PY, os.path.join(HERE, "parse6.py"), "--split", sp["split"], "--scores", sdir,
                 "--floors", floors, "--bridge", bdir, "--inputs", sp["inputs"], "--label", label,
                 "--out", summ] + suffix, env, os.path.join(out, f"parse_{sk}.log"))
            # SPEC §5's ladder (W7's decompose.py, imported, + the per-command split)
            run([PY, os.path.join(HERE, "decompose6.py"), "--split", sp["split"], "--scores", sdir,
                 "--floors", floors, "--bridge", bdir, "--inputs", sp["inputs"], "--label", label,
                 "--out", dec] + suffix, env, os.path.join(out, f"decompose_{sk}.log"))
        fam = None
        if sk in ("navhard", "navtest") and os.path.exists(os.path.join(bdir, "seam_R6_A1.npz")):
            # four families where a logged human future exists (SPEC §8)
            fam = os.path.join(out, f"families_{sk}.json")
            run([PY, os.path.join(HERE, "families6.py"), "--seam",
                 os.path.join(bdir, "seam_R6_A1.npz"), "--inputs", sp["inputs"], "--stage", "1",
                 "--label", label, "--out", fam], env, os.path.join(out, f"families_{sk}.log"))
        summary["splits"][sk] = {"device": dev, "exact_dedup": dedup, "arms": recs,
                                 "summary": summ if os.path.exists(summ) else None,
                                 "decomposition": dec if os.path.exists(dec) else None,
                                 "families": fam if fam and os.path.exists(fam) else None}
        log(f"SPLIT {sk} done: summary={'present' if os.path.exists(summ) else 'ABSENT'}", qlog)
    msp = os.path.join(out, "MILESTONE_SUMMARY.json")
    if os.path.exists(msp):
        # one milestone is run split by split (the waiter's phases): MERGE, never overwrite — and
        # only for the SAME checkpoint (a different md5 in the same dir is refused, not mixed)
        prev = json.load(open(msp, encoding="utf-8"))
        if prev.get("md5") != summary["md5"]:
            sys.exit(f"⛔ {msp} belongs to md5 {prev.get('md5')}, not {summary['md5']} — refused")
        merged = dict(prev.get("splits", {}))
        merged.update(summary["splits"])
        summary["splits"] = merged
        if "cuda_controls" in prev and "cuda_controls" not in summary:
            summary["cuda_controls"] = prev["cuda_controls"]
    with open(msp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    # the step-to-step read: this milestone vs the first RESULT reading (step 5,000), per split run
    # here, when both carry R6_A1 on the split (code/step_compare.py; never re-scores)
    ref = os.path.join(PKG, "raw", "milestones", "step5000")
    if step > 5000 and os.path.exists(os.path.join(ref, "MILESTONE_SUMMARY.json")):
        for sk in [s_ for s_ in a.splits.split(",") if s_]:
            if os.path.exists(os.path.join(ref, f"summary_{sk}.json")) and                     os.path.exists(os.path.join(out, f"summary_{sk}.json")) and not a.tokens_navtest:
                run([PY, os.path.join(HERE, "step_compare.py"), "--split", sk, "--a", ref, "--b", out,
                     "--out", os.path.join(out, f"compare_vs_step5000_{sk}.json")], env,
                    os.path.join(out, f"compare_vs_step5000_{sk}.log"))
    run([PY, os.path.join(HERE, "bars6.py"), "--milestone", out], env,
        os.path.join(out, "bars.log"))
    log(f"DONE bars={'present' if os.path.exists(os.path.join(out, 'BARS.json')) else 'ABSENT'}",
        qlog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
