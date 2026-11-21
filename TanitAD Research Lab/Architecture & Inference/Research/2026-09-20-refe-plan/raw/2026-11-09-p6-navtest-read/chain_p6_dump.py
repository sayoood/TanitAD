#!/usr/bin/env python3
"""Second full navtest pass of P6's model_final.pt, with the 64-proposal dump. PREREG_P6: a second full eval of the same checkpoint must be bit-identical;
the runbook's --dump-proposals feeds selection quality and command adherence. Same command as chain_p6.py's seam plus that flag and a new output name.
Writes markers/dump.DONE or dump.FAILED. The poses are compared with refe_p6final.npz bit for bit; exit 1 of the seam on the declared frame control is expected."""
import json, os, socket, subprocess, sys, time, traceback
from pathlib import Path
import chain_p6 as C
W, DATA, MK = C.W, C.DATA, C.MK
LOG = W / "CHAIN_DUMP.log"
def say(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%F %T')}] {m}\n")
def main():
    for m in ("dump.DONE", "dump.FAILED"):
        if (MK / m).exists():
            raise SystemExit("marker exists: " + m)
    sys.path.insert(0, C.PKG + "/eval")
    import stage2_sft as S2
    import numpy as np
    seam = DATA / "seams" / "refe_p6final_pass2.npz"
    dump = W / "pass2" / "proposals_p6final.npz"
    dump.parent.mkdir(parents=True, exist_ok=True)
    env_d = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in S2.EC.env_driverl().items()}
    from_export = S2.EXPORT if hasattr(S2, "EXPORT") else None
    assert from_export, "stage2_sft has no EXPORT"
    PY_D = S2.PY_D if hasattr(S2, "PY_D") else "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
    cmd = [PY_D, str(C.EC_DIR / "refe_navtest_seam.py"), "--ckpt", str(C.CK), "--frames", str(DATA / "frames"),
           "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test", "--export", from_export,
           "--tokens", str(C.TOK), "--out", str(seam), "--arm", "REFe_p6final_pass2", "--dump-proposals", str(dump)]
    me = json.dumps({"job": "refe-p6final-pass2", "pid": os.getpid(), "host": socket.gethostname(), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    say("waiting for the GPU lock")
    S2.take_lock(me)
    say("GPU lock taken")
    try:
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(C.EC_DIR), env=env_d, stdout=open(W / "pass2" / "seam.log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        say(f"seam rc {rc} after {round(time.time()-t0)} s")
    finally:
        S2.release(me)
    if not seam.exists() or not dump.exists():
        raise RuntimeError(f"seam or dump missing, rc {rc}")
    a, b = np.load(DATA / "seams" / "refe_p6final.npz"), np.load(seam)
    same_tok = bool((a["token"] == b["token"]).all())
    same_pose = bool(np.array_equal(a["poses"], b["poses"]))
    n_diff = int((a["poses"] != b["poses"]).any(axis=(1, 2)).sum()) if a["poses"].shape == b["poses"].shape else -1
    say(f"second pass vs first: tokens equal {same_tok}, poses bit-identical {same_pose}, tokens differing {n_diff}")
    (MK / "dump.DONE").write_text(json.dumps(dict(tokens_equal=same_tok, poses_bit_identical=same_pose, n_tokens_differing=n_diff, seam_rc=rc)) + "\n")
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc(); say("FAILED: " + tb[-1500:])
        (MK / "dump.FAILED").write_text(tb[-3000:])
        sys.exit(1)
