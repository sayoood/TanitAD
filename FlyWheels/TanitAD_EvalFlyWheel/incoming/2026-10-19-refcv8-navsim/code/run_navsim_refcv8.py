#!/usr/bin/env python3
"""ONE COMMAND: a refcv8 checkpoint through the NavSim suite (TANITAD VENV).  Port of ``run_navsim_refcv7.py`` (its ScorePool / log /
md5 / RAM-gate helpers are IMPORTED, not copied).

    python code/run_navsim_refcv8.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --weights ema --splits navtest,navhard,warmup \
        --retime-table <gains.json> --retime-stage-file <retime_stage.py>
    python code/run_navsim_refcv8.py --ckpt <ckpt_48852.pt> --weights raw --splits navtest,navhard,warmup      # the RAW arm, beside

Arms per A-R8-NS1 (``GOALS_AND_CLAIMS.md``): EMA weights run R8_LEGAL (inference seed 0, the BAR arm) + R8_LEGAL_s1 (seed 1, the
inference-variance floor) on every split, + R8_PRIV (DIAGNOSTIC, never a bar) on navtest, + R8_BLIND (the deliberate regression that
must FAIL N3) on warmup, + R8_RT (derived from R8_LEGAL's rows, only with an explicit ``--retime-table``).  RAW weights run R8_LEGAL
only.  Each step is its OWN process, gated on the ARTIFACT of the step before (never an exit code, never through a pipe):

  0. md5 of the checkpoint, its step; the label is ``RESULT`` ONLY for step 48,852 with ``--weights`` agreeing with the file name
     (``*ema*`` <-> ema), else ``VALIDATION ONLY`` -- a validation number is never refcv8's NavSim score;
  1. per split, the GPU: the dev-box lock (``gpu_lock.py``) is acquired for the split's BRIDGE only; not acquired within the wait ->
     CPU fp32 for the WHOLE split.  On CUDA the K0 / KD controls (``tests/test_model_seam8.py``) are re-measured first;
  2. ``run_bridge8.py`` (resumable rows) -> seams + manifests (+ ``PRIOR_ha0p``, + ``R8_RT``);
  3. the official scorer per seam, CPU, RAM-gated: navtest -> ``score_navtest8.py`` (v1.1 PDMS, W3's harness); navhard / warmup ->
     the refcv7 suite's ``score_arm7.py`` UNCHANGED (arm-agnostic) in the NavSim v2 venv; a seam whose manifest reads ``partial`` is
     REFUSED; a CSV is used only beside a count guard reading PASS;
  4. ``families7.py`` (arm-agnostic) on R8_LEGAL and PRIOR_ha0p; ``INDEX.json`` (every seam, manifest, count guard and
     ``summary_x100_4dp``).

NOT BUILT HERE (named in HANDOFF_NAVSIM_R8.md): the statistics/bars layer for BAR-R8-N1..N4 (``parse*8`` / ``decompose8`` / ``bars8``
-- the refcv7 copies hard-code R7_ arm names and SPEC_REFCV7's bars); until it exists the INDEX carries the scorer's own per-arm
rows and the paired bootstrap is run from the CSVs by the refcv7 ``parse_navtest7.py`` conventions.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import boot8  # noqa: E402

sys.path.insert(0, boot8.R7CODE)
import gpu_lock  # noqa: E402
import run_navsim_refcv7 as RN7  # noqa: E402  (ScorePool, log, md5_file, jget, ram_wait, run, seams_ready)
from pytest_control7 import control_passed  # noqa: E402

PY = RN7.PY
NPY = RN7.NPY
TREE = boot8.TREE
INC = boot8.INC
R6IN = f"{boot8.R6PKG}/raw/inputs"
BANKS = RN7.BANKS
CONFIG = os.environ.get("R8_CONFIG", "E:/refcv8_eval_kit/ckpt/config.json")
ROAD = f"{R6IN}/road_plane_navhard_warmup_logs.json"
FINAL_STEP = 48852
SPLITS = {
    "warmup": {"split": "warmup_two_stage", "ema": "R8_LEGAL,R8_LEGAL_s1,R8_BLIND", "raw": "R8_LEGAL",
               "inputs": f"{INC}/2026-09-19-navsim-refcv4b-bridge/raw/navsim_agent_inputs.json",
               "speed": "", "bank2": f"{BANKS}/warmup_two_stage", "bank1": "", "gpu_wait_s": 900},
    "navtest": {"split": "navtest", "ema": "R8_LEGAL,R8_LEGAL_s1,R8_PRIV", "raw": "R8_LEGAL",
                "inputs": "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz",
                "speed": f"{R6IN}/speed_limits_navtest.json.gz", "bank2": "",
                "bank1": "D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank",
                "logs_root": "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test", "gpu_wait_s": 10800},
    "navhard": {"split": "navhard_two_stage", "ema": "R8_LEGAL,R8_LEGAL_s1", "raw": "R8_LEGAL",
                "inputs": ("C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/"
                           "navsim_agent_inputs.json"),
                "speed": "", "bank2": f"{BANKS}/navhard_s2", "bank1": f"{BANKS}/navhard_s1", "gpu_wait_s": 10800},
}
DERIVED = ("PRIOR_ha0p",)


def gpu_for_split(sk: str, wait_s: int, qlog: str, device: str):
    """-> (device, token).  'cpu' forces CPU; 'auto' tries the lock then falls back to CPU; 'cuda' tries the lock and REFUSES to
    fall back (job name refcv8-navsim-<split>)."""
    if device == "cpu":
        return "cpu", None
    tok = gpu_lock.acquire(f"refcv8-navsim-{sk}", wait_s, log=lambda m: None)
    if tok:
        RN7.log(f"GPU lock ACQUIRED for {sk} ({tok})", qlog)
        return "cuda", tok
    if device == "cuda":                        # an explicit --device cuda NEVER degrades to a ~76 h CPU read
        sys.exit(f"[run8] --device cuda: the GPU lock was not acquired within {wait_s} s for {sk}; refusing the CPU fallback "
                 f"(rerun later, or raise --gpu-wait-s; --device auto allows the CPU fallback)")
    RN7.log(f"GPU lock NOT acquired within {wait_s} s for {sk} -> CPU fp32 for the whole split", qlog)
    return "cpu", None


def bridge_cmd(sp: dict, sk: str, arms: str, a, md5: str, dev: str, tok, label: str, out: str, rt: bool,
               dedup: bool, tokens_file: str) -> list:
    cmd = [PY, os.path.join(HERE, "run_bridge8.py"), "--split", sp["split"], "--arms", arms, "--inputs", sp["inputs"],
           "--road-plane", ROAD, "--ckpt", a.ckpt, "--config", CONFIG, "--ckpt-md5", md5, "--device", dev,
           "--precision", "auto", "--threads", str(a.threads), "--label", label, "--out", out, "--derived"]
    if "R8_PRIV" in arms.split(","):
        cmd += ["--speed", sp["speed"]]
    if dedup:
        cmd.append("--exact-dedup")
    if sp["bank2"]:
        cmd += ["--bank2", sp["bank2"]]
    if sp["bank1"]:
        cmd += ["--bank1", sp["bank1"]]
    if sk == "navtest":
        cmd += ["--bank1-kind", "navtest", "--logs-root", sp["logs_root"]]
    if tokens_file:
        cmd += ["--tokens-file", tokens_file]
    if rt:
        cmd += ["--retime", "--retime-table", a.retime_table]
        if a.retime_stage_file:
            cmd += ["--retime-stage-file", a.retime_stage_file]
        if a.retime_table_sha256:
            cmd += ["--retime-table-sha256", a.retime_table_sha256]
    if dev == "cuda":
        cmd += ["--gpu-lock-token", tok]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
    ap.add_argument("--md5", default="", help="expected md5 (refused on mismatch)")
    ap.add_argument("--splits", default="navtest,navhard,warmup")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-par", type=int, default=4)
    ap.add_argument("--gpu-wait-s", type=int, default=-1)
    ap.add_argument("--arms", default="", help="override the arm list of EVERY split (comma separated)")
    ap.add_argument("--tokens-file", default="", help="navtest subset (W3 format); the scorer gets the same file")
    ap.add_argument("--retime-table", default="", help="gains JSON (+ .sha256 sidecar) for R8_RT; omitted -> no R8_RT; no default")
    ap.add_argument("--retime-table-sha256", default="")
    ap.add_argument("--retime-stage-file", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if os.environ.get("R8_MUTATION"):
        sys.exit("[run8] R8_MUTATION is set -- a mutation-test environment never scores a checkpoint")
    if not os.path.exists(a.ckpt):
        sys.exit(f"[run8] checkpoint missing: {a.ckpt}")
    if a.weights == "raw" and a.retime_table:
        sys.exit("[run8] R8_RT is an EMA-weights arm; drop --retime-table for --weights raw")
    if ("ema" in os.path.basename(a.ckpt).lower()) != (a.weights == "ema"):
        sys.exit(f"[run8] --weights {a.weights} disagrees with the file name {os.path.basename(a.ckpt)!r}")
    import torch
    step = int(torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True).get("step", -1))
    # RESULT only for the final step AND the full registered arm set on the full split: a subset or an arm override is never a result
    label = ("RESULT" if (step == FINAL_STEP and not a.tokens_file and not a.arms)
             else "VALIDATION ONLY (not refcv8's NavSim score)")
    out = a.out or os.path.join(PKG, "raw", "milestones", f"step{step}_{a.weights}")
    os.makedirs(out, exist_ok=True)
    qlog = os.path.join(out, "runner.log")
    md5 = RN7.md5_file(a.ckpt)
    if a.md5 and md5 != a.md5:
        sys.exit(f"[run8] md5 {md5} != --md5 {a.md5}")
    RN7.log(f"START ckpt={a.ckpt} step={step} weights={a.weights} md5={md5} label={label} config_md5={RN7.md5_file(CONFIG)}", qlog)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{TREE}/stack;{TREE}/taniteval"
    env["TANITAD_REPO"] = TREE
    env["OMP_NUM_THREADS"] = str(a.threads)          # torch spawns ~113 threads PER PROCESS; unset, concurrent arms stall
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    pool = RN7.ScorePool(a.max_par, env, qlog)
    summary = {"ckpt": a.ckpt, "md5": md5, "step": step, "weights": a.weights, "label": label, "splits": {}}
    post = []
    for sk in [s for s in a.splits.split(",") if s]:
        sp = SPLITS[sk]
        arms = a.arms or sp[a.weights]
        if sk == "navtest" and a.tokens_file:
            pass
        dev, tok = gpu_for_split(sk, sp["gpu_wait_s"] if a.gpu_wait_s < 0 else a.gpu_wait_s, qlog, a.device)
        e = dict(env)
        rec = {"device": dev, "gpu_lock_token": tok, "arms": arms}
        try:
            if dev == "cuda":
                ce = dict(e, R8_TEST_DEVICE="cuda", R8_TEST_CKPT=a.ckpt, R8_TEST_CONFIG=CONFIG)
                clog = os.path.join(out, f"cuda_controls_{sk}.log")
                rc = RN7.run([PY, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
                              os.path.join(PKG, "tests", "test_model_seam8.py"), "-k", "K0 or KD"], ce, clog)
                txt = open(clog, encoding="utf-8", errors="replace").read().replace("\\", "/")
                k0, kd = control_passed(txt, "test_K0"), control_passed(txt, "test_KD")
                rec["cuda_controls"] = {"pytest_rc": rc, "K0_pass": k0, "KD_pass": kd}
                RN7.log(f"CUDA controls {sk}: K0={k0} KD={kd}", qlog)
                rec["exact_dedup"] = bool(kd)
                if not k0:                                # the seed replicate rests on K0: a failure REFUSES CUDA
                    RN7.log(f"CUDA REFUSED for {sk}: K0 failed on CUDA -> CPU", qlog)
                    gpu_lock.release(tok)
                    dev, tok = "cpu", None
                    rec["device"] = "cpu"
            if dev == "cpu":
                e["CUDA_VISIBLE_DEVICES"] = "-1"
                RN7.log(f"RAM gate passed for the CPU bridge of {sk} ({RN7.ram_wait(6.0):.1f} GB)", qlog)
            bdir = os.path.join(out, f"bridge_{sk}")
            dedup = True if dev == "cpu" else bool(rec.get("exact_dedup", False))
            rec["exact_dedup"] = dedup
            rt = bool(a.retime_table) and "R8_LEGAL" in arms.split(",")
            tf = a.tokens_file if (sk == "navtest" and a.tokens_file) else ""
            RN7.log(f"BRIDGE {sk} arms={arms} device={dev} retime={rt}", qlog)
            RN7.run(bridge_cmd(sp, sk, arms, a, md5, dev, tok, label, bdir, rt, dedup, tf), e, bdir + ".log")
        finally:
            if tok:
                RN7.log(f"GPU lock RELEASED={gpu_lock.release(tok)} after {sk}", qlog)
        seam_arms = arms.split(",") + list(DERIVED) + (["R8_RT"] if rt else [])
        rec["seams"] = RN7.seams_ready(bdir, seam_arms)
        RN7.log(f"SEAMS {sk} {rec['seams']}", qlog)
        sdir = os.path.join(out, f"scores_{sk}")
        os.makedirs(sdir, exist_ok=True)
        for arm in seam_arms:
            if rec["seams"].get(arm) != "OK":
                continue
            seam = os.path.join(bdir, f"seam_{arm}.npz")
            if sk == "navtest":
                lab = f"r8s{step}{a.weights[0]}_{arm}" + ("_sub" if tf else "")
                cmd = [PY, os.path.join(HERE, "score_navtest8.py"), "--label", lab, "--seam", seam, "--out", sdir]
                cmd += ["--tokens", tf] if tf else []
                pool_job = {"name": f"{sk}:{lab}", "cmd": cmd, "counts": os.path.join(sdir, lab, f"{lab}.counts.json"),
                            "scorer_log": os.path.join(sdir, lab, f"{lab}.log"),
                            "driver_log": os.path.join(sdir, f"{lab}.driver.txt")}
            else:
                tag = arm if sp["split"] == "warmup_two_stage" else f"{arm}__{sp['split']}"
                cmd = [NPY, os.path.join(boot8.R7CODE, "score_arm7.py"), "--arm", arm, "--seam", seam, "--split", sp["split"],
                       "--out", sdir, "--exp-tag", f"e8s{step}{a.weights[0]}"]
                pool_job = {"name": f"{sk}:{arm}", "cmd": cmd, "counts": os.path.join(sdir, f"score_{tag}.counts.json"),
                            "scorer_log": os.path.join(sdir, f"score_{tag}.log"),
                            "driver_log": os.path.join(sdir, f"{arm}.driver.txt")}
            rec.setdefault("counts_of", {})[arm] = pool_job["counts"]
            pool.add(pool_job)
        pool.poll()
        summary["splits"][sk] = rec
        post.append((sk, bdir, sdir, seam_arms))
    RN7.log("all bridges done; draining the scorer pool", qlog)
    pool.drain()
    index = {}
    for sk, bdir, sdir, seam_arms in post:
        sp = SPLITS[sk]
        idx = {}
        for arm in seam_arms:
            man = os.path.join(bdir, f"seam_{arm}.manifest.json")
            cj = summary["splits"][sk].get("counts_of", {}).get(arm)       # the EXACT counts file of this arm (no substring match:
            idx[arm] = {"manifest": man if os.path.exists(man) else None,  # "R8_LEGAL" is a prefix of "R8_LEGAL_s1")
                        "partial": RN7.jget(man, "partial"), "KPR": RN7.jget(man, "KPR"),
                        "s_per_scene_forward_mean": RN7.jget(man, "s_per_scene_forward_mean"),
                        "counts": cj, "scorer_status": RN7.jget(cj, "status") if cj else None,
                        "summary_x100_4dp": RN7.jget(cj, "summary_x100_4dp") if cj else None}
        if sk in ("navtest", "navhard"):
            for arm in ("R8_LEGAL", "PRIOR_ha0p"):
                sm = os.path.join(bdir, f"seam_{arm}.npz")
                if os.path.exists(sm):
                    RN7.run([PY, os.path.join(boot8.R7CODE, "families7.py"), "--seam", sm, "--inputs", sp["inputs"], "--stage", "1",
                             "--label", label, "--out", os.path.join(out, f"families_{sk}_{arm}.json")], env,
                            os.path.join(out, f"families_{sk}.log"))
        index[sk] = idx
    msp = os.path.join(out, "MILESTONE_SUMMARY.json")
    if os.path.exists(msp):
        prev = json.load(open(msp, encoding="utf-8"))
        if prev.get("md5") != summary["md5"]:
            sys.exit(f"[run8] {msp} belongs to md5 {prev.get('md5')}, not {summary['md5']} -- refused")
        merged = dict(prev.get("splits", {}))
        merged.update(summary["splits"])
        summary["splits"] = merged
    with open(msp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(out, "INDEX.json"), "w", encoding="utf-8") as fh:
        json.dump({"label": label, "step": step, "weights": a.weights, "md5": md5, "index": index}, fh, indent=1)
    RN7.log(f"DONE label={label}", qlog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
