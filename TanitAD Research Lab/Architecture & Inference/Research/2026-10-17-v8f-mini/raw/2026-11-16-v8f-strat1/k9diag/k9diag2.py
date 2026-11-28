"""K9 follow-up (MM 2026-11-16). k9diag.py: at --limit logged the T1 tick-0 residual against T0 is <= 3.8e-6 m (7/48 and 10/48 windows),
at --limit causal it is up to 10.6 m. T0 evaluates in batches of 64, T1 in batches of 32. Committed before running:
  (a) T0 at --batch 32 against T0 at --batch 64 (same windows, same checkpoint): the evaluator's own float32 batching floor.
  (b) T1 --limit logged --batch 64 against T0 --batch 64.
  * floor(a) >= the logged residual's order (1e-6 m), or (b) <= floor(a)  -> K9's residual is batch-layout float32 noise; K9 is EXPLAINED
    as the declared limit mode plus that floor.
  * floor(a) == 0.0 AND (b) > 1e-6 m                                     -> a tick-0 input other than the limit differs; trace it.
Smoke grid only: a code check, never a result.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "F:/TanitAD-scratch/strat1/k9diag")
import k9diag as K  # noqa: E402

D = K.D
sys.path.insert(0, K.R)
import run_t0_strat1 as RT  # noqa: E402
import strat1_strategic as SS  # noqa: E402


def t0_b32(arm):
    out = D + "/t0_b32"
    os.makedirs(out, exist_ok=True)
    p = f"{out}/eval_STRAT1-{arm}_real.npz"
    if not os.path.exists(p):
        argv = RT.argv_for(arm, "real", out, K.GRID) + ["--batch", "32"]
        r = subprocess.run(argv, env=K.env, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=K.R)
        open(f"{out}/log_{arm}.txt", "w", encoding="utf-8").write(" ".join(argv) + "\n" + r.stdout[-3000:] + r.stderr[-3000:])
        K.log("T0 b32", arm, "rc", r.returncode)
        if r.returncode != 0:
            raise SystemExit(f"T0 b32 {arm} failed")
    return dict(np.load(p, allow_pickle=True))


def t1_b64(arm):
    a = K.base_argv()
    tag = f"K9DIAG-{arm}-logged-b64"
    od = f"{D}/t1_{arm}_logged_b64"
    os.makedirs(od + "/ticks", exist_ok=True)
    for f, v in (("--ckpt", f"{K.R}/ckpts/{arm}/ckpt_last.pt"), ("--tag", tag), ("--eval-grid", K.GRID), ("--out-dir", od),
                 ("--dump-ticks", od + "/ticks"), ("--limit", "logged")):
        K.set_flag(a, f, v)
    a = a + K.TEXT[arm] + ["--dump-nav", "--batch", "64"]
    p = f"{od}/ticks/ticks_{tag}.npz"
    if not os.path.exists(p):
        r = subprocess.run(a, env=K.env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        open(f"{od}/log.txt", "w", encoding="utf-8").write(" ".join(a) + "\n" + r.stdout[-3000:] + r.stderr[-3000:])
        K.log("T1 b64", arm, "rc", r.returncode)
        if r.returncode != 0:
            raise SystemExit(f"T1 b64 {arm} failed")
    return dict(np.load(p, allow_pickle=True))


def main():
    _, _, _, ci, t, _ = SS.load_inputs(K.GRID)
    ci_t = {(int(c), int(tt)): i for i, (c, tt) in enumerate(zip(ci, t))}
    rep = {"tool": "k9diag2.py", "rows": {}}
    for arm in K.ARMS:
        z64 = dict(np.load(f"{D}/t0/eval_STRAT1-{arm}_real.npz", allow_pickle=True))
        z32 = t0_b32(arm)
        fa = np.abs(np.asarray(z32["ckpt"], np.float64) - np.asarray(z64["ckpt"], np.float64))
        tk = t1_b64(arm)
        mxb, n, okb = SS.k9(tk, z64["ckpt"], ci_t)
        row = {"a_T0_b32_vs_b64_max_m": float(fa.max()), "a_windows_over_1e-6": int((fa.reshape(len(fa), -1).max(1) > 1e-6).sum()),
               "a_windows_nonzero": int((fa.reshape(len(fa), -1).max(1) > 0).sum()), "b_T1_logged_b64_vs_T0_max_m": mxb, "b_n": n}
        rep["rows"][arm] = row
        K.log(arm, json.dumps(row))
    rows = rep["rows"].values()
    floor = max(r["a_T0_b32_vs_b64_max_m"] for r in rows)
    resid = max(r["b_T1_logged_b64_vs_T0_max_m"] for r in rows)
    rep["verdict"] = ("EXPLAINED: limit mode + float32 batch-layout floor" if (floor >= 1e-6 or resid <= floor) else
                      "NOT EXPLAINED: T0 is batch-invariant but T1 logged still differs")
    json.dump(rep, open(D + "/k9diag2.json", "w"), indent=1)
    K.log("VERDICT", rep["verdict"])
    print("ZZ K9DIAG2 DONE ZZ", flush=True)


if __name__ == "__main__":
    import time
    for _ in range(300):
        cf = K.commit_free_gb()
        if cf >= 5.5:
            break
        time.sleep(60)
    else:
        print("ZZ K9DIAG2 FAIL ZZ commit stayed under 5.5 GB for 5 h", flush=True)
        raise SystemExit(1)
    K.log("commit free GB", round(cf, 2))
    main()
