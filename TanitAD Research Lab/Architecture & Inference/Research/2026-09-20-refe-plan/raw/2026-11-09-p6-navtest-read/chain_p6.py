#!/usr/bin/env python3
"""P6 read chain: (1) navtest seam of model_final.pt, (2) W3 harness score (full navtest filter), (3) p6_read.py.
Seam + scoring calls are the ones stage2_sft.py used for sft4_B_u400_navtest_full (same banked harness, no --tokens on the
full set because 12,146 tokens overflow Windows' 32,767-char command line: WinError 206, MEASURED in stage2_sft.py).
The seam runs from the PATH-ONLY scratch copy under evalcode/. GPU lock = stage2_sft.take_lock/release (the tool's own code).
Markers: W/markers/chain.DONE (text = p6_read rc) or chain.FAILED (text = reason). Usage: chain_p6.py [--dry]
"""
import json, os, subprocess, sys, time, traceback, socket
from pathlib import Path

W = Path("F:/TanitAD_p6read_work")
MK = W / "markers"
EC_DIR = W / "evalcode" / "eval"
CK = W / "ckpt" / "run" / "model_final.pt"
SHA = "968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1"
NAME = "p6final"
LABEL = f"refe_{NAME}"
PKGD = Path("E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read")
PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DATA = Path("E:/Projects/TanitAD/data/refe_navtest")
TOK = DATA / "navtest_full_tokens.json"
TOK_MD5 = "3a8b3653fd70891e50a230a37eb76ae8"
LOG = W / "CHAIN.log"


def say(m):
    line = f"[{time.strftime('%F %T')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def md5(p):
    import hashlib
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main(dry):
    MK.mkdir(parents=True, exist_ok=True)
    for m in ("chain.DONE", "chain.FAILED"):
        if (MK / m).exists() and not dry:
            raise SystemExit(f"marker {m} already exists; refusing to restart")
    sys.path.insert(0, PKG + "/eval")
    import stage2_sft as S2              # repo module, read-only import (definitions only)
    PY_D, PY_T, EXPORT = S2.PY_D, S2.PY_T, S2.EXPORT
    assert TOK.exists() and md5(TOK) == TOK_MD5, "token list md5 mismatch"
    assert len(json.load(open(TOK, encoding="utf-8"))["tokens"]) == 12146
    import hashlib
    h = hashlib.sha256()
    with open(CK, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == SHA, "checkpoint sha256 mismatch"
    hp = S2.HP.require(S2.HARNESS, caller=Path(PKG) / "eval" / "score_navtest_refe.py")
    say(f"preflight ok: tokens md5 {TOK_MD5}, ckpt sha ok, harness pins {hp['n_files']} files, export {EXPORT}")
    seam = DATA / "seams" / f"{LABEL}.npz"
    csv = DATA / "score" / LABEL / f"{LABEL}.csv"
    run_out = W / "run"
    out_dir = W / "navtest_p6final"
    out_dir.mkdir(parents=True, exist_ok=True)
    env_d = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in S2.EC.env_driverl().items()}   # same as stage2_sft.py; = the scratch copy's env_driverl
    seam_cmd = [PY_D, str(EC_DIR / "refe_navtest_seam.py"), "--ckpt", str(CK), "--frames", str(DATA / "frames"),
                "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test", "--export", EXPORT,
                "--tokens", str(TOK), "--out", str(seam), "--arm", f"REFe_{NAME}"]
    read_cmd = [PY_T, str(PKGD / "p6_read.py"), "--ckpt", str(CK), "--out", str(run_out),
                "--run-config", str(W / "ckpt" / "run" / "config.json"), "--train-logs", str(W / "train_logs.txt"),
                "--p6-seam", str(seam), "--p6-csv", str(csv), "--expect-ckpt-sha", SHA]
    if dry:
        say("DRY seam: " + " ".join(seam_cmd)); say("DRY read: " + " ".join(read_cmd))
        for p in (EC_DIR / "refe_navtest_seam.py", Path(EXPORT), DATA / "frames", S2.SP, PKGD / "p6_read.py", W / "train_logs.txt"):
            say(f"DRY exists {p}: {Path(p).exists()}")
        return 0
    # 0 GPU smoke (RUNBOOK step b, 3 tokens): takes the YAX lock through selc6_dump's own code; waits, never forces
    smoke = W / "smoke"
    if not (W / "smoke_PASSED.txt").exists():
        sc = [PY_D, "selc6_dump.py", "--ckpt", str(CK), "--out", str(smoke), "--smoke", "--expect-ckpt-sha", SHA]
        rc = subprocess.call(sc, cwd=str(PKGD), stdout=open(W / "smoke_stdout.log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        txt = open(W / "smoke_stdout.log", encoding="utf-8", errors="replace").read() + (open(smoke / "dump.log", encoding="utf-8", errors="replace").read() if (smoke / "dump.log").exists() else "")
        if rc != 0 or "consolidated:" not in txt:
            raise RuntimeError(f"SMOKE FAILED rc {rc}; tail:\n" + txt[-3000:])
        n = sum(f.stat().st_size for f in smoke.rglob("*") if f.is_file())
        import shutil
        shutil.rmtree("F:/TanitAD_p6read_work/smoke")          # literal path; smoke passed
        (W / "smoke_PASSED.txt").write_text(f"smoke passed {time.strftime('%F %T')}; deleted F:/TanitAD_p6read_work/smoke, {n} bytes\n")
        say(f"smoke passed; deleted F:/TanitAD_p6read_work/smoke ({n} bytes)")
    # 1 seam under the shared GPU lock
    if not seam.exists():
        me = json.dumps({"job": "refe-p6final-navtest", "pid": os.getpid(), "host": socket.gethostname(),
                         "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        S2.take_lock(me)
        try:
            t0 = time.time()
            rc = subprocess.call(seam_cmd, cwd=str(EC_DIR), env=env_d, stdout=open(out_dir / "seam.log", "w"), stderr=subprocess.STDOUT)
            say(f"seam rc {rc} after {round(time.time()-t0)} s")
        finally:
            S2.release(me)
        if rc != 0 or not seam.exists():
            raise RuntimeError(f"seam failed rc {rc}; see {out_dir/'seam.log'}")
    rep = json.load(open(seam.with_suffix(".report.json"), encoding="utf-8"))
    if rep.get("misses") != 0 or rep.get("rows") != 12146:
        raise RuntimeError(f"seam report: rows {rep.get('rows')} misses {rep.get('misses')}")
    # 2 score, the whole navtest filter, harness retried while its RAM guard fires (as stage2_sft)
    status = {}
    stp = out_dir / "score_status.json"
    if stp.exists():
        status = json.load(open(stp, encoding="utf-8"))
    for att in range(1, 7):
        if status.get("status") == "PASS" and csv.exists():
            break
        while S2.free_mb() < 4000:
            time.sleep(60)
        lp = out_dir / f"score_{att}.log"
        subprocess.call([PY_D, "score_e.py", "--label", LABEL, "--seam", str(seam), "--out", str(DATA / "score")],
                        cwd=S2.SP, env=dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1"),
                        stdout=open(lp, "w"), stderr=subprocess.STDOUT)
        lines = open(lp, encoding="utf-8", errors="replace").read().splitlines()
        status = next((json.loads(l) for l in reversed(lines) if l.startswith("{") and '"status"' in l), {})
        json.dump(status, open(stp, "w"), indent=1)
        say(f"score attempt {att}: {status.get('status')}")
        if status.get("status") != "PASS":
            time.sleep(120)
    if status.get("status") != "PASS" or not csv.exists():
        raise RuntimeError("harness did not PASS; see " + str(out_dir))
    # 3 the read
    rc = subprocess.call(read_cmd, cwd=str(PKGD), stdout=open(W / "run_stdout.log", "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
    say(f"p6_read rc {rc}")
    if rc not in (0, 2):
        raise RuntimeError(f"p6_read.py rc {rc}; see {W/'run_stdout.log'}")
    (MK / "chain.DONE").write_text(f"p6_read rc {rc} (0 = verdict issued, 2 = instrument failure) at {time.strftime('%F %T')}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main("--dry" in sys.argv))
    except SystemExit:
        raise
    except BaseException:
        if "--dry" in sys.argv:
            raise
        MK.mkdir(parents=True, exist_ok=True)
        tb = traceback.format_exc()
        say("FAILED\n" + tb)
        (MK / "chain.FAILED").write_text(tb)
        sys.exit(1)
