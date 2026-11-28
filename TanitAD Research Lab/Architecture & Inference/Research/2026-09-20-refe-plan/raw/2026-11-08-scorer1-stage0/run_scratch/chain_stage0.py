#!/usr/bin/env python3
"""SCORER1 Stage 0 chain (LAND_PLAN step B).  preflight (the trainer's own --preflight, writes run/PREFLIGHT_PASS.json) -> dump u400's step-0 scores on the 3,137 held-out sets (lr 0, max-updates 1)
-> stage0_gap.py -> markers/stage0.DONE or markers/stage0.FAILED (text = reason).  GPU lock = stage2_sft.take_lock/release (waits, never forces).
Restartable: each stage is skipped when its own output exists.  A stale FAILED marker is renamed FAILED.prev on restart; DONE refuses.  Usage: chain_stage0.py [--dry]"""
import json, os, subprocess, sys, time, traceback, socket, shutil
from pathlib import Path
S = Path("F:/tanitad_scratch/stage0_20261108")
sys.path.insert(0, str(S))
import common_args as C
MK = S / "markers"; LOG = S / "CHAIN.log"; LG = S / "logs"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"      # DRIVERL_EVAL_PYTHON of scorer1_pod.sh
PKG = C.PK
CK_SHA = "8b4dba4818b5"
def say(m):
    line = f"[{time.strftime('%F %T')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")
def env():
    e = dict(os.environ)
    e.update({"REFE_BACKBONE_ROOT": "E:/Projects/TanitAD/data/backbones", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    e.pop("CUDA_VISIBLE_DEVICES", None)
    return e
def free_gb(p): return shutil.disk_usage(p).free / 1e9
def main(dry):
    MK.mkdir(exist_ok=True); LG.mkdir(exist_ok=True)
    if (MK / "stage0.DONE").exists(): raise SystemExit("stage0.DONE exists; nothing to do")
    if (MK / "stage0.FAILED").exists() and not dry:
        (MK / "stage0.FAILED").replace(MK / "stage0.FAILED.prev")
    import hashlib
    h = hashlib.sha256()
    with open(C.CKPT, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    assert h.hexdigest().startswith(CK_SHA), "u400 sha mismatch " + h.hexdigest()
    assert free_gb("F:/") >= 8, "F: below 8 GB free"
    assert free_gb("C:/") >= 6, "C: below 6 GB free"
    pre_cmd = [PY, "scorer_finetune.py"] + C.argv(True)
    run_cmd = [PY, "scorer_finetune.py"] + C.argv(False)
    scores = Path(C.DUMP) / "scores_step0_base.npz"
    gap_cmd = [PY, str(Path(PKG) / "raw/2026-11-08-scorer1-impl-v3/stage0/stage0_gap.py"), "--scores", str(scores),
               "--sets", C.D + "/ho/refe_heldout/sets", "--targets", C.D + "/refe_navtrain/train_grow", "--pdm", C.D + "/s1aux/heldout_pdm/heldout_pdm", "--out", str(S / "stage0.json")]
    if dry:
        say("DRY pre: " + " ".join(pre_cmd)); say("DRY run: " + " ".join(run_cmd)); say("DRY gap: " + " ".join(gap_cmd))
        for p in (Path(PKG) / "refe/scorer_finetune.py", Path(C.D) / "ho/refe_heldout/sets", Path(C.D) / "train_bank/sets"): say(f"DRY exists {p}: {p.exists()}")
        return 0
    refe = str(Path(PKG) / "refe")
    sys.path.insert(0, "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval")
    import stage2_sft as S2                     # repo module, definitions only; take_lock/release are the tool's own
    need_pre = False  # MM 2026-11-08: eval-only dump, --dump-step0-only; see MM_RULING_STAGE0_INPUTS.md
    need_run = not scores.exists()
    if need_pre or need_run:
        me = json.dumps({"job": "scorer1-stage0", "pid": os.getpid(), "host": socket.gethostname(), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        say("waiting for the GPU lock")
        S2.take_lock(me)
        say("GPU lock taken")
        try:
            if need_pre:
                t0 = time.time()
                rc = subprocess.call(pre_cmd, cwd=refe, env=env(), stdout=open(LG / "preflight.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                say(f"preflight rc {rc} after {round(time.time()-t0)} s")
                if rc != 0 or not (Path(C.OUT) / "PREFLIGHT_PASS.json").exists():
                    raise RuntimeError(f"trainer --preflight rc {rc}, no PREFLIGHT_PASS.json; see {LG/'preflight.log'}")
            if need_run:
                t0 = time.time()
                rc = subprocess.call(run_cmd, cwd=refe, env=env(), stdout=open(LG / "dump.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                say(f"dump run rc {rc} after {round(time.time()-t0)} s (the step-0 dump is written before any optimizer step; rc after that is recorded, not judged)")
                if not scores.exists():
                    raise RuntimeError(f"dump rc {rc}: {scores} not written; see {LG/'dump.log'}")
        finally:
            S2.release(me)
    import numpy as np
    z = np.load(scores, allow_pickle=False)
    n = len(z["log"])
    if n != 3137: raise RuntimeError(f"scores file has {n} rows, expected 3137")
    if not (S / "stage0.json").exists():
        rc = subprocess.call(gap_cmd, cwd=str(S), env=env(), stdout=open(LG / "gap.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        say(f"stage0_gap rc {rc}")
        if rc != 0 or not (S / "stage0.json").exists(): raise RuntimeError(f"stage0_gap rc {rc}; see {LG/'gap.log'}")
    r = json.load(open(S / "stage0.json", encoding="utf-8"))
    if r.get("n_scored_tokens") != 3137: raise RuntimeError(f"stage0.json scored {r.get('n_scored_tokens')} tokens, expected 3137 (a join miss); see {LG/'gap.log'}")
    (MK / "stage0.DONE").write_text(f"stage0.json at {S/'stage0.json'}, {n} scored sets, {time.strftime('%F %T')}\n")
    return 0
if __name__ == "__main__":
    try:
        sys.exit(main("--dry" in sys.argv))
    except SystemExit: raise
    except BaseException:
        tb = traceback.format_exc(); say("FAILED: " + tb[-1500:])
        if "--dry" not in sys.argv: (MK / "stage0.FAILED").write_text(tb[-3000:])
        sys.exit(1)
