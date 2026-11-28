"""rh_replay.py -- control K-REPLAY (card section 3, "inference is deterministic"): one 64-window chunk of RH_R0C10XHb20 seed 0 rolled twice into two folders; every array of every dump file and the tick file
must be bit-identical (0 differing arrays).  Pass is the control; a failure stops the chain.  Uses RH_s0 when its checkpoint is present, else UE_s0.A (stand-in for the dry run, said in the file).
Writes out/read/K_REPLAY.json."""
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rh_common as C  # noqa: E402
import c_exec as X  # noqa: E402

FORM = "R0C10XHb20"
N = 64


def one(label, kind, ck):
    out = f"{C.OUT}/replay/{label}"
    tg = "replay_m10_causal_heads"
    os.makedirs(out + "/ticks", exist_ok=True)
    argv = C.argv_record(kind, FORM, 0, ckpt=ck, out=out, tg=tg, nwin=N)
    with open(out + "/t1.log", "w", encoding="utf-8") as lf:
        rc = subprocess.run(argv, env=C.env(), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8").returncode
    if rc != 0:
        raise SystemExit(f"K-REPLAY rollout {label} rc {rc}")
    return out, tg


def main():
    kind = "RH" if os.path.isfile(C.ckpt_of("RH", 0)) else "UEA"
    ck = C.ckpt_of(kind, 0)
    a, ta = one("a", kind, ck)
    b, tb = one("b", kind, ck)
    d = X.cmp_dirs(f"{a}/dump_{ta}", f"{b}/dump_{tb}", "ep*.npz")
    t = X.cmp_dirs(f"{a}/ticks", f"{b}/ticks", "ticks_*.npz")
    ok = bool(d["files_a"] > 0 and d["files_a"] == d["files_b"] and d["same_names"] and d["differing_arrays"] == 0 and t["differing_arrays"] == 0)
    rep = {"control": "K-REPLAY", "chunk_windows": N, "checkpoint_used": f"{kind}_s0", "stand_in": kind == "UEA", "form": FORM, "dump_files": d["files_a"], "arrays_compared": d["arrays"],
           "differing_arrays": d["differing_arrays"] + t["differing_arrays"], "PASS": ok, "evidence_class": "MEASURED (ours)"}
    os.makedirs(C.OUT + "/read", exist_ok=True)
    json.dump(rep, open(C.OUT + "/read/K_REPLAY.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep))
    for x in (a, b):                      # own temp, literal path
        import shutil
        shutil.rmtree(x)
        open("F:/TanitAD-scratch/rollhist/DELETIONS.log", "a", encoding="utf-8").write(f"{__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime())} deleted {x} (K-REPLAY chunk copies; result in out/read/K_REPLAY.json)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
