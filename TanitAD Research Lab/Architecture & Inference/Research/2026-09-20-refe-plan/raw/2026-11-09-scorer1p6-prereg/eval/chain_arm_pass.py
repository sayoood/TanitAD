#!/usr/bin/env python3
"""Launcher for arm_logits_pass.py, built the way chain_p6.py builds the seam call (same env, same flags), with the GPU-lock wait of the seam chain.
  smoke : CPU, CUDA_VISIBLE_DEVICES=-1, no lock, --limit N, --ref-tol (cross-device against the GPU-made reference)
  full  : takes the dev-box GPU lock through stage2_sft.take_lock (waits, never forces), runs all 12,146 tokens, releases it.
Usage: chain_arm_pass.py --out DIR --arm NAME=path [--arm ...] [--smoke N] [--dry]   (launch detached: WMI `cmd /c ... < NUL`)
Not run on the GPU by its author."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

W = Path("F:/TanitAD_p6read_work")
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DATA = Path("E:/Projects/TanitAD/data/refe_navtest")
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--base-ckpt", default=str(W / "ckpt" / "run" / "model_final.pt"))
    ap.add_argument("--smoke", type=int, default=0, help="CPU smoke on the first N tokens")
    ap.add_argument("--ref-tol", type=float, default=0.05, help="smoke only (CPU vs the GPU-made reference)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, PKG + "/eval")
    import stage2_sft as S2
    env = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in S2.EC.env_driverl().items()}
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [S2.PY_D, str(HERE / "arm_logits_pass.py"), "--base-ckpt", a.base_ckpt, "--out", a.out, "--frames", str(DATA / "frames"),
           "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test", "--export", S2.EXPORT,
           "--tokens", str(DATA / "navtest_full_tokens.json")]
    for x in a.arm:
        cmd += ["--arm", x]
    if a.smoke:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        cmd += ["--device", "cpu", "--limit", str(a.smoke), "--ref-tol", str(a.ref_tol), "--chunk", "1"]
    if a.dry:
        print(" ".join(cmd))
        return 0
    os.makedirs(a.out, exist_ok=True)
    log = open(Path(a.out) / "arm_pass.log", "a", encoding="utf-8")
    if a.smoke:
        return subprocess.call(cmd, cwd=str(W / "evalcode" / "eval"), env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    me = json.dumps({"job": "arm-logits-pass", "pid": os.getpid(), "host": socket.gethostname(), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    S2.take_lock(me)                      # waits for the shared GPU lock; the seam chain's own code
    try:
        return subprocess.call(cmd, cwd=str(W / "evalcode" / "eval"), env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    finally:
        S2.release(me)


if __name__ == "__main__":
    sys.exit(main())
