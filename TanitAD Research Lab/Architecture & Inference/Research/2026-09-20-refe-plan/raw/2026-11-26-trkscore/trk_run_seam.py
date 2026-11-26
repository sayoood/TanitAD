#!/usr/bin/env python3
"""Run ONE TRK-SCORE navtest seam exactly as eval/stage2_sft.py runs a seam (its env, its GPU lock, its seam command), wrapped in trk_select.py.

    trk_run_seam.py --ckpt <model_final.pt> --expect-sha256 <64 hex> --name trk_c1 --tokens <W3 token json> --yaw-table <navtest yaw table>
                    --seam-out <seam.npz> --props-out <seam.proposals.npz> --dump <trk.npz> [--red-all] [--keep-sim]

Before the lock: CTRL-SHA (the file's sha256 == --expect-sha256), the yaw table's sha256 prefix (stage2_sft.NAVTEST_YAW_SHA_PREFIX), and refusal if any
output exists. The lock: stage2_sft.take_lock / release (C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock), job name refe-trk-<name>.
Exit: the seam's rc (1 = the known Deviation-1 frame-control failure on the full navtest; the chain validates the report), 2 refusal, 3 sha mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("TANITAD_REFE_PKG", "F:/TanitAD_p5cread_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
NAV = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--ckpt", "--expect-sha256", "--name", "--tokens", "--yaw-table", "--seam-out", "--props-out", "--dump"):
        ap.add_argument(k, required=True)
    ap.add_argument("--red-all", action="store_true")
    ap.add_argument("--keep-sim", action="store_true")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    os.environ.setdefault("TANITAD_NAVSIM_ROOT", "F:/Archive/devbox-C/navsim")
    sys.path.insert(0, os.path.join(PKG, "eval"))
    import stage2_sft as S2                              # its env builder, lock, constants: the seam runs as every stage-2 seam runs
    for p in (a.seam_out, a.props_out, a.dump):
        if os.path.exists(p):
            print(f"REFUSING: {p} exists (move it aside)")
            return 2
    got = sha256_file(a.ckpt)
    if got != a.expect_sha256:
        print(f"CTRL-SHA FAIL: {a.ckpt} sha256 {got} != {a.expect_sha256}")
        return 3
    yaw = sha256_file(a.yaw_table)
    if not yaw.startswith(S2.NAVTEST_YAW_SHA_PREFIX):
        print(f"REFUSING: yaw table sha256 {yaw[:12]} is not {S2.NAVTEST_YAW_SHA_PREFIX}")
        return 2
    print(f"CTRL-SHA PASS {got}; yaw table {yaw[:12]}; export {S2.EXPORT}", flush=True)
    env = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in S2.EC.env_driverl().items()}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    red = ["--red-all"] if a.red_all else ["--red-mod", "10"]
    cmd = [S2.PY_D, os.path.join(HERE, "trk_select.py"), "--pkg", PKG, "--nav-root", NAV, "--dump", a.dump, *red, *(["--keep-sim"] if a.keep_sim else []),
           "--", os.path.join(PKG, "refe", "ego_yaw.py"), "--table", a.yaw_table, "--",
           "refe_navtest_seam.py", "--ckpt", a.ckpt, "--frames", str(S2.D / "frames"),
           "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test", "--export", S2.EXPORT,
           "--tokens", a.tokens, "--out", a.seam_out, "--arm", f"REFe_{a.name}", "--rule", "navsim_v1", "--dump-proposals", a.props_out]
    log = a.log or (os.path.splitext(a.dump)[0] + ".seam.log")
    me = json.dumps({"job": f"refe-trk-{a.name}", "pid": os.getpid(), "host": socket.gethostname(), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print("CMD " + subprocess.list2cmdline(cmd), flush=True)
    S2.take_lock(me)
    print(f"GPU lock taken {time.strftime('%T')}", flush=True)
    t0 = time.time()
    try:
        with open(log, "w", encoding="utf-8") as fo:
            rc = subprocess.call(cmd, cwd=os.path.join(PKG, "eval"), env=env, stdout=fo, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    finally:
        S2.release(me)
    print(f"ZZTRK_SEAM rc={rc} seconds={time.time() - t0:.0f} log={log}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
