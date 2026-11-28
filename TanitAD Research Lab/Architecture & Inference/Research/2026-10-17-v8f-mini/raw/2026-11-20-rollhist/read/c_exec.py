"""c_exec.py -- control c-EXEC (card PREREG_V8F_ROLLHIST section 3.1): UE_s0.A, form R0C10XHb20, rolled twice on the dev box CPU, all 141 eval files:
  A = the ADD68 study wrapper rzgfh (sha256 373ffe72f2655c35...)   B = the tier of record `v8f_t1_eval_hb.py --kappa-max 0.2 --inner v8f_t1_eval_reader_rule_rzg`.
PASS = 0 differing arrays (every npz array of the 141 dump files, and the tick files).  Provenance strings (`ckpt`, `tag`) are compared as strings and may differ only in the tag, which is the same here.
Usage: python c_exec.py [--max-windows N]   (N > 0 is a code check, never a result)   Writes C_EXEC.json (and out/cexec/...).  Resumable: a finished rollout (T1_DONE) is not repeated."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rh_common as C  # noqa: E402

FORM, KIND, K = "R0C10XHb20", "UEA", 0
BASE = C.OUT + "/cexec"          # a --max-windows smoke writes under out/smoke/cexec instead (set in main)


def roll(label, argv_fn, nwin):
    out = f"{BASE}/{label}"
    tg = "cexec_UE_s0A_R0C10XHb20_m10_causal_heads"
    if os.path.exists(out + "/T1_DONE"):
        print("skip (done)", label, flush=True)
        return out, tg, None
    os.makedirs(out + "/ticks", exist_ok=True)
    argv = argv_fn(KIND, FORM, K, out=out, tg=tg, nwin=nwin)
    open(out + "/argv.txt", "w", encoding="utf-8").write("\n".join(argv) + "\n")
    t = time.time()
    with open(out + "/t1.log", "w", encoding="utf-8") as lf:
        rc = subprocess.run(argv, env=C.env(), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8").returncode
    wall = round(time.time() - t, 1)
    n = len(glob.glob(f"{out}/dump_{tg}/ep*.npz"))
    print(label, "rc", rc, "dump files", n, "wall_s", wall, flush=True)
    if rc != 0:
        raise SystemExit(f"ROLLOUT {label} rc {rc} (see {out}/t1.log)")
    open(out + "/T1_DONE", "w").write(f"rc 0 files {n} wall_s {wall}\n")
    return out, tg, wall


def load(f):
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def eq(a, b):
    if a.dtype.kind in "OUS":
        return bool(np.array_equal(a, b))
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def cmp_dirs(da, db, pattern):
    fa, fb = sorted(glob.glob(da + "/" + pattern)), sorted(glob.glob(db + "/" + pattern))
    na, nb = [os.path.basename(x) for x in fa], [os.path.basename(x) for x in fb]
    res = {"files_a": len(fa), "files_b": len(fb), "same_names": na == nb, "arrays": 0, "differing_arrays": 0, "differing": []}
    if na != nb:
        return res
    for x, y in zip(fa, fb):
        A, B = load(x), load(y)
        if sorted(A) != sorted(B):
            res["differing"].append([os.path.basename(x), "key sets differ"])
            res["differing_arrays"] += 1
            continue
        for k in A:
            res["arrays"] += 1
            if not eq(A[k], B[k]):
                res["differing_arrays"] += 1
                if len(res["differing"]) < 20:
                    res["differing"].append([os.path.basename(x), k])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=0)
    a = ap.parse_args()
    global BASE
    if a.max_windows:
        BASE = C.OUT + "/smoke/cexec"
    for p in (C.ckpt_of(KIND, K),):
        if C.sha256_file(p) != C.CKPT_SHA_UEA[0]:
            raise SystemExit("REFUSED: UE_s0.A checkpoint sha256 differs from 2fd4b747...")
    outA, tgA, wA = roll("A_study_rzgfh", C.argv_study, a.max_windows)
    outB, tgB, wB = roll("B_record_hb_rzg", C.argv_record, a.max_windows)
    d = cmp_dirs(f"{outA}/dump_{tgA}", f"{outB}/dump_{tgB}", "ep*.npz")
    t = cmp_dirs(f"{outA}/ticks", f"{outB}/ticks", "ticks_*.npz") if True else {}
    # the tick files carry the tag in their file name; both runs used the same tag but different names for the readout logs, so only the tick npz are compared
    rl = []
    for lab, o in (("A", outA), ("B", outB)):
        f = glob.glob(o + "/ticks/readout_log_*.json")
        rl.append(json.load(open(f[0], encoding="utf-8")) if f else None)
    hbf = glob.glob(outB + "/ticks/hb_*.json")
    hbc = (json.load(open(hbf[0], encoding="utf-8")).get("bound_counters") if hbf else None) or {}
    keep = ("readout", "acc_gain", "path_exec", "replan_steps", "calls", "max_abs_speed_minus_last_step", "max_abs_acceleration_minus_gain_executed", "mean_abs_acceleration_read", "max_abs_acceleration_read")
    rls = [None if r is None else {k: r.get(k) for k in keep} for r in rl]
    ok = bool(d["files_a"] == 141 and d["files_b"] == 141 and d["same_names"] and d["differing_arrays"] == 0 and t["differing_arrays"] == 0 and t["files_a"] == 1 and t["files_b"] == 1
              and rls[0] == rls[1] and not a.max_windows and (hbc.get("frames_bound_acted") or 0) > 0)
    rep = {"control": "c-EXEC", "card": "PREREG_V8F_ROLLHIST.md sha256 781eea87, section 3.1", "form": FORM, "checkpoint": "UE_s0.A sha256 2fd4b747532f55b2...", "tree_commit": "43319227",
           "study_wrapper": f"{C.STUDY}.py sha256 " + C.sha256_file(C.CODE + f"/integ10k/{C.STUDY}.py")[:16], "record_executor": "v8f_t1_eval_hb.py sha256 " + C.sha256_file(C.TREE + "/stack/scripts/v8f_t1_eval_hb.py")[:16] + f" --kappa-max {C.KAPPA} --inner {C.WRAPPER}",
           "device": "dev-box CPU, 2 threads", "dump_files": [d["files_a"], d["files_b"]], "arrays_compared": d["arrays"], "differing_arrays_dump": d["differing_arrays"], "differing_arrays_ticks": t["differing_arrays"],
           "differing_first": d["differing"], "hb_counters_B": {k: hbc.get(k) for k in ("frames_total", "frames_bound_acted", "max_excess_rad")}, "bound_acted_in_B": bool((hbc.get("frames_bound_acted") or 0) > 0 or bool(a.max_windows)), "readout_logs_equal": rls[0] == rls[1], "readout_log_a": rls[0], "wall_s_a": wA, "wall_s_b": wB,
           "tier": "T1-kin/hb0.2", "evidence_class": "MEASURED (ours; zero training)", "smoke_windows": a.max_windows or None, "PASS": ok,
           "if_fail": "the stage-A reference (ADD67 / ADD68 under rzgfh) is rerun under the tier-of-record form (UE_s0.A and UE_s1.A, six forms = 12 rollouts)"}
    path = HERE_OUT = os.path.dirname(os.path.abspath(__file__)) + ("/C_EXEC.json" if not a.max_windows else "/C_EXEC_smoke.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("dump_files", "arrays_compared", "differing_arrays_dump", "differing_arrays_ticks", "readout_logs_equal", "PASS")}), flush=True)
    print("ZZ C-EXEC " + ("PASS" if ok else "FAIL") + " ZZ", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
