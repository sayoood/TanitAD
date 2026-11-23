"""ft0_4seed.py -- MM 2026-11-23. PREREG_V8F_KINLAB section 3, item "F_T0 at four seeds (beside, printed, never used here)":
|UE_sI.A - UE_sJ.A| at T0 over the four stage-A seeds, with the read's own T0 instrument (rh_t0.argv_t0 = v8f_mini_eval.py run --project --families,
CPU, the read tree 43319227), mean ade6 over the same 2,399 windows. Writes to its own folder; the landed read's out/t0 is not touched.
Control: the instrument's UE_s0.A / UE_s1.A means against the INHERITED FACTS E3 values 2.0376 / 1.9671 (printed, not a bar).
usage: python ft0_4seed.py run | read"""
import itertools
import json
import os
import subprocess
import sys

E = "F:/TanitAD-scratch/rollhist/read"
sys.path.insert(0, E)
import rh_common as C  # noqa: E402
import rh_t0 as T  # noqa: E402

OUT = "F:/TanitAD-scratch/kinlab/readrk/ft0_4seed"
ARMS = ["UE_s0.A", "UE_s1.A", "UE_s2.A", "UE_s3.A"]


def run():
    os.makedirs(OUT, exist_ok=True)
    rc = 0
    for arm in ARMS:
        if os.path.exists(f"{OUT}/eval_{arm}.npz"):
            print("skip (exists)", arm, flush=True)
            continue
        with open(f"{OUT}/log_{arm}.txt", "w", encoding="utf-8") as lf:
            r = subprocess.run(T.argv_t0(arm, out=OUT), env=C.env(), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8")
        ok = r.returncode == 0 and os.path.exists(f"{OUT}/eval_{arm}.npz")
        print(arm, "rc", r.returncode, "ok" if ok else "FAILED", flush=True)
        rc |= (not ok)
    return rc


def read():
    import numpy as np
    m, n = {}, {}
    for arm in ARMS:
        z = np.load(f"{OUT}/eval_{arm}.npz", allow_pickle=True)   # as rh_t0.guard loads it (eid)
        key = "ade6" if "ade6" in z.files else [k for k in z.files if "ade6" in k][0]
        v = np.asarray(z[key], dtype=np.float64)
        m[arm], n[arm] = float(np.nanmean(v)), int(np.isfinite(v).sum())
    pairs = {f"{a}-{b}": abs(m[a] - m[b]) for a, b in itertools.combinations(ARMS, 2)}
    out = {"item": "PREREG_V8F_KINLAB section 3: F_T0 at four seeds (beside, printed, never used here)", "tier": "T0",
           "instrument": "rh_t0.argv_t0 (v8f_mini_eval.py run --project --families, CPU), read tree 43319227", "evidence_class": "MEASURED (ours)",
           "mean_ade6_m": m, "n_windows": n, "abs_pairwise_m": pairs, "max_abs_pairwise_m": max(pairs.values()),
           "F_T0_card_m": C.F_T0, "control_inherited_FACTS_E3": {"UE_s0.A": 1.9671, "UE_s1.A": 2.0376},
           "note": "printed only; KINLAB's G-T0 uses F_T0 0.0705 as frozen. A later card may adopt a four-seed F_T0, before its own data."}
    json.dump(out, open(f"{OUT}/FT0_4SEED.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(run() if sys.argv[1] == "run" else (read() or 0))
