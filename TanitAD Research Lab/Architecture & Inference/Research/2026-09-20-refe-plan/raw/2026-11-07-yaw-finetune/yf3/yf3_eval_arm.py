#!/usr/bin/env python3
"""YF3: evaluate ONE checkpoint on a token set exactly the way RESULT_SFT4 / eval/stage2_sft.py did (seam --rule navsim_v1_lane under the shared GPU lock, W3 harness via score_e.py,
families6), with the ONE difference the PREREG requires: the seam runs under YT's yaw shim launcher (yf3_launch_g7.py -> ego_yaw.main), which feeds a yaw-trained checkpoint
ego[4] = 10 x omega(token) from the NAVTEST table and leaves a checkpoint without meta `ego_yaw` (A0, arm C) untouched. stage2_sft.py itself cannot be used: it launches the seam as a
SUBPROCESS, so a launcher around stage2_sft.py would never reach the planner.

    python yf3_eval_arm.py --name A0 --ckpt <model.pt> --tokens <w3 token json> --out <dir> [--lock-owner yf3_yaw_eval] [--give-up-h 12]
Idempotent: a finished stage (seam.npz + seam.report.json, score_status PASS, perwindow.npz) is not repeated. Last line: ZZYF3EVAL <name> OK|FAILED <why>.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yf3_common as C  # noqa: E402

sys.path.insert(0, str(C.Y2))
sys.path.insert(0, str(C.EVAL))
import gpu_lock as GL  # noqa: E402  (YF2's lock: same file and protocol as the other dev-box GPU jobs; a foreign lock is never removed)
import eval_checkpoint as EC  # noqa: E402
import harness_pins as HP  # noqa: E402


def seam_env():
    return {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in EC.env_driverl().items()}


def run_seam(name, ckpt, tokens, outd, owner, give_up_h, rule="navsim_v1_lane", table=None):
    seam, rep = outd / "seam.npz", outd / "seam.report.json"
    if seam.exists() and rep.exists() and json.load(open(rep)).get("misses") == 0:
        return {"skipped": True}
    cmd = [C.PY_D, str(HERE / "yf3_launch_g7.py"), "--record", str(outd / "shim_g7.json"), "--table", str(table or C.TABLE_TEST), "--",
           str(C.EVAL / "refe_navtest_seam.py"), "--ckpt", str(ckpt), "--frames", str(C.D / "frames"), "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test",
           "--export", C.export_path(), "--tokens", str(tokens), "--out", str(seam), "--arm", f"REFe_{name}", "--rule", rule]
    body = GL.acquire_blocking(give_up_h * 3600, owner="yf3_yaw_eval", log=lambda m: print(m, flush=True))
    print(f"[{time.strftime('%H:%M:%S')}] GPU lock taken for {name}: {body}", flush=True)
    t0 = time.time()
    try:
        with open(outd / "seam.log", "wb") as lf:
            rc = subprocess.run(cmd, cwd=str(C.EVAL), env=seam_env(), stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    finally:
        GL.release_gpu_lock(GL.LOCK, body)
    out = {"rc": rc, "seconds": round(time.time() - t0, 1), "cmd": cmd}
    if rc != 0 or not rep.exists():
        raise RuntimeError(f"seam failed rc={rc}: {C.tail(outd / 'seam.log', 1500)}")
    return out


def parse_status(log_path):
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    return next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})


def free_mb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def run_score(name, tokens, outd, toks, tag="full", attempts=20):
    """W3's harness via score_e.py, retried while the RAM guard fires (verbatim stage2_sft.py protocol). Harness side effect (unavoidable, not ours): it also writes its run folder under
    <navsim root>/exp/w3_navtest_v1/runs/<label>/."""
    label = f"refe_yf3_{tag}_{name}"                       # the harness refuses labels not starting with "refe"; the tag keeps smoke and full runs from sharing a run folder
    sc = outd / "score"
    stp = outd / "score_status.json"
    status = json.load(open(stp)) if stp.exists() else {}
    for att in range(1, attempts + 1):
        if status.get("status") == "PASS":
            break
        while free_mb() < 4000:
            time.sleep(60)
        lp = outd / f"score_{att}.log"
        subprocess.call([C.PY_D, "score_e.py", "--label", label, "--seam", str(outd / "seam.npz"), "--tokens", str(tokens), "--out", str(sc)], cwd=str(C.HARNESS),
                        env=dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1"), stdout=open(lp, "w"), stderr=subprocess.STDOUT)
        status = parse_status(lp)
        C.jdump(status, stp)
        if status.get("status") != "PASS":
            time.sleep(120)
    csv_p = sc / label / f"{label}.csv"
    if status.get("status") != "PASS" or not csv_p.exists():
        raise RuntimeError(f"harness did not PASS: {status}")
    return {"status": status, "csv": str(csv_p)}


def run_families(name, outd, toks):
    import numpy as np
    z = np.load(outd / "seam.npz")
    tt = z["token"]
    ts = set(toks)
    idx = [i for i, t in enumerate(tt) if str(t) in ts]
    sub = outd / "families_seam.npz"
    np.savez(sub, token=tt[idx], fingerprint=z["fingerprint"][idx], poses=z["poses"][idx], sampling=z["sampling"], arm=z["arm"])
    out = outd / "families.json"
    rc = subprocess.call([C.PY_T, "families6.py", "--seam", str(sub), "--inputs", C.export_path(), "--stage", "1", "--label", name, "--out", str(out), "--n-boot", "2000"], cwd=str(C.EV6E),
                         env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=open(outd / "families.log", "w"), stderr=subprocess.STDOUT)
    if rc != 0 or not out.exists():
        raise RuntimeError(f"families6 failed rc={rc}: {C.tail(outd / 'families.log', 1200)}")
    return {"rc": rc}


def run_perwindow(outd, tokens):
    rc = subprocess.call([C.PY_T, str(HERE / "yf3_perwindow.py"), "--seam", str(outd / "families_seam.npz"), "--tokens", str(tokens), "--out", str(outd / "perwindow.npz")],
                         env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=open(outd / "perwindow.log", "w"), stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"perwindow failed rc={rc}: {C.tail(outd / 'perwindow.log', 1200)}")


def eval_arm(name, ckpt, tokens, outd, give_up_h=12.0, tag="full", seam_only=False):
    outd = Path(outd)
    outd.mkdir(parents=True, exist_ok=True)
    HP.require(C.HARNESS, caller=C.EVAL / "score_navtest_refe.py")                 # refuse to start unless the banked harness is the pinned one
    tk = C.jload(tokens)
    toks = sorted(tk["tokens"])
    res = {"name": name, "ckpt": str(ckpt), "tokens": str(tokens), "n_tokens": len(toks), "started": C.now(), "rule": "navsim_v1_lane",
           "tier": "NAVSIM v1.1 PDMS = ego pseudo-simulation of an open-loop plan against logged agents (NOT the action-closed-loop tier)",
           "estimator": f"paired log-cluster bootstrap, {C.BOOT_N} resamples, percentile 95 %, seed {C.BOOT_SEED}"}
    res["seam"] = run_seam(f"{tag}_{name}", ckpt, tokens, outd, "yf3_yaw_eval", give_up_h)
    if seam_only:
        return res
    res["score"] = run_score(name, tokens, outd, toks, tag)
    res["families"] = run_families(name, outd, toks)
    run_perwindow(outd, tokens)
    res["finished"] = C.now()
    C.jdump(res, outd / "arm.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--give-up-h", type=float, default=12.0)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--seam-only", action="store_true", help="G8 re-run: seam only (no scoring)")
    a = ap.parse_args()
    try:
        eval_arm(a.name, a.ckpt, a.tokens, a.out, a.give_up_h, a.tag, a.seam_only)
    except Exception:
        why = traceback.format_exc()[-1500:]
        print(f"ZZYF3EVAL {a.name} FAILED {why}".encode("ascii", "replace").decode())
        return 1
    print(f"ZZYF3EVAL {a.name} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
