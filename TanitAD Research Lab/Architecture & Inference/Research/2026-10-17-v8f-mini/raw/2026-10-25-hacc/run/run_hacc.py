#!/usr/bin/env python3
"""run_hacc.py -- the detached driver.  Plain launches died on CTRL_C events, so the console control handler is disabled first.
  python run_hacc.py auto  [--n-boot 2000]        the real run: Phase A (train only) -> controls -> Phase B -> DK -> READ.json -> RESULT_V8F_HACC.md -> FILES.txt
  python run_hacc.py plumb [--n-boot 30]          PLUMBING TEST ONLY, into plumb/: the arm is replaced by HBLEND's banked H (w := the banked weights), so the read pipeline is
                                                  exercised end to end WITHOUT producing a HACC number.  Needs the real Phase A files (fit_<seed>.json) already in place.
Ends with exactly one line:  ZZHACC-DONE <OUTCOME>   or   ZZHACC-STOP <reason>   (plumb: ZZHACC-PLUMB-DONE / ZZHACC-STOP).
"""
import json
import os
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
except Exception:
    pass
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)


def main(argv):
    mode = argv[0] if argv else "auto"
    nb = int(argv[argv.index("--n-boot") + 1]) if "--n-boot" in argv else (30 if mode == "plumb" else 2000)
    t0 = time.time()
    import v8f_hacc as A
    if mode == "prep":                                      # real run, step 1 (foreground, seconds): verify the Phase A files, write the X1 control
        A.os.makedirs(HERE, exist_ok=True)
        for sd in A.SEEDS:
            f = json.load(open(f"{HERE}/fit_{sd}.json"))
            if not f["train_controls"]["all_pass"]:
                print(f"ZZHACC-STOP {sd}: a TRAIN-side control failed: {f['train_controls']['pass']}", flush=True)
                return 2
        x1 = A.x1_hspec_tie()
        A.H.jdump(x1, f"{HERE}/control_X1.json")
        print(f"[prep] fit files ok; X1 pass={x1['pass']} rows={x1['checked']}", flush=True)
        return 0 if x1["pass"] else 2
    if mode == "par":                                       # real run, step 2: one process per seed; the --finish process also does DK -> READ -> RESULT
        sd = argv[argv.index("--seed") + 1]
        rc = A.score_seed(sd, nb, HERE)
        if rc:
            print(f"ZZHACC-STOP {sd}: controls failed rc {rc}; no outcome", flush=True)
            return rc
        if "--finish" not in argv:
            print(f"ZZHACC-PART-DONE {sd}  ({time.time() - t0:.0f}s)", flush=True)
            return 0
        other = [x for x in A.SEEDS if x != sd][0]
        import numpy as np
        deadline = time.time() + 1500
        while True:
            try:
                j = json.load(open(f"{HERE}/score_{other}.json"))
                if "single" not in j:
                    print(f"ZZHACC-STOP {other}: its controls failed; no outcome", flush=True)
                    return 2
                np.load(f"{HERE}/score_{other}_perwindow.npz")["HACC"]
                break
            except Exception:
                if time.time() > deadline:
                    print(f"ZZHACC-STOP waited 25 min for {other}", flush=True)
                    return 2
                time.sleep(3)
        A.run_dk(nb, HERE)
        R = A.read_all(HERE)
        A._local("build_result", "hacc_build_result").main(HERE)
        print(f"ZZHACC-DONE {R['outcome']}  ({time.time() - t0:.0f}s)", flush=True)
        return 0
    if mode == "auto":
        rc = A.cmd_auto(nb, HERE)
        if rc:
            print(f"ZZHACC-STOP rc {rc}: a control failed (see the lines above); no outcome", flush=True)
            return rc
        R = json.load(open(f"{HERE}/READ.json"))
        print(f"ZZHACC-DONE {R['outcome']}  ({time.time() - t0:.0f}s)", flush=True)
        return 0
    # ---- plumbing test: the arm := HBLEND's banked H
    out = f"{HERE}/plumb"
    os.makedirs(out, exist_ok=True)
    for s in A.SEEDS:
        f = json.load(open(f"{HERE}/fit_{s}.json"))
        f["w_star"] = f["banked_w"]
        f["PLUMB_NOTE"] = "w_star REPLACED by HBLEND's banked w: this directory is a plumbing test of the read pipeline, NOT a HACC result"
        json.dump(f, open(f"{out}/fit_{s}.json", "w"), indent=1)
    x1 = A.x1_hspec_tie()
    A.H.jdump(x1, f"{out}/control_X1.json")
    for s in A.SEEDS:
        rc = A.score_seed(s, nb, out)
        if rc:
            print(f"ZZHACC-STOP plumb: controls failed rc {rc}", flush=True)
            return rc
    A.run_dk(nb, out)
    R = A.read_all(out)
    A._local("build_result", "hacc_build_result").main(out)
    print(f"ZZHACC-PLUMB-DONE {R['outcome']}  ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main(sys.argv[1:])
    except SystemExit as e:
        print(f"ZZHACC-STOP SystemExit: {e}", flush=True)
        rc = 2
    except Exception:
        traceback.print_exc()
        print("ZZHACC-STOP exception (traceback above)", flush=True)
        rc = 3
    sys.exit(rc)
