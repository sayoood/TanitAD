"""K9b on the FULL eval grid (MM 2026-11-16). The STRAT1 read's K9 fails as written (T1 `--limit causal`, batch 32, vs T0 real, batch 64):
max 101.7 m (S1_s0), 93.8 m (S1_s1). The smoke grid showed T1 `--limit logged --batch 64` == T0 bit for bit (k9diag2.json).
Committed before running: for S1_s0 and S1_s1 on all 2,399 windows, T1 `--limit logged --batch 64` against the read's own T0 real arrays
(out/t0/eval_STRAT1-S1_s{k}_real.npz, batch 64) must read max abs diff <= 1e-6 m (bit-identity expected). PASS -> the T1 per-tick STRATEGIC
row of the read (causal dumps) is quotable with the limit-mode note. FAIL -> it stays unquoted and the failing windows are traced.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "F:/TanitAD-scratch/strat1/k9diag")
import k9diag as K  # noqa: E402

sys.path.insert(0, K.R)
import strat1_strategic as SS  # noqa: E402

GRID_FULL = K.R + "/inputs/eval_grid_windows.json"


def t1_full(arm):
    a = K.base_argv()
    tag = f"K9FULL-{arm}-logged-b64"
    od = f"{K.D}/full_{arm}"
    os.makedirs(od + "/ticks", exist_ok=True)
    for f, v in (("--ckpt", f"{K.R}/ckpts/{arm}/ckpt_last.pt"), ("--tag", tag), ("--eval-grid", GRID_FULL), ("--out-dir", od),
                 ("--dump-ticks", od + "/ticks"), ("--limit", "logged")):
        K.set_flag(a, f, v)
    a = a + K.TEXT.get(arm, ["--nav-text-npz", K.R + "/inputs/navtext_nav.npz", "--nav-text-sha256",
                             "03bd75bc3a3de17928681bedb8e46a48a891304de45d395c49cc876acf6e6160"]) + ["--dump-nav", "--batch", "64"]
    p = f"{od}/ticks/ticks_{tag}.npz"
    if not os.path.exists(p):
        import subprocess
        r = subprocess.run(a, env=K.env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        open(f"{od}/log.txt", "w", encoding="utf-8").write(" ".join(a) + "\n" + r.stdout[-3000:] + r.stderr[-3000:])
        K.log("T1 full", arm, "rc", r.returncode)
        if r.returncode != 0:
            raise SystemExit(f"T1 full {arm} failed")
    return dict(np.load(p, allow_pickle=True))


def main():
    _, _, _, ci, t, _ = SS.load_inputs(None)
    ci_t = {(int(c), int(tt)): i for i, (c, tt) in enumerate(zip(ci, t))}
    rep = {"tool": "k9full.py", "grid": "eval_grid_windows.json (full, 2,399 windows)", "rows": {}}
    for arm in ("S1_s0", "S1_s1"):
        z = dict(np.load(f"{K.R}/out/t0/eval_STRAT1-{arm}_real.npz", allow_pickle=True))
        assert np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)
        tk = t1_full(arm)
        mx, n, ok = SS.k9(tk, z["ckpt"], ci_t)
        rep["rows"][arm] = {"K9b_max_abs_diff_m": mx, "n_windows": n, "ok": ok}
        K.log(arm, "K9b", mx, n, ok)
    rep["verdict"] = "PASS" if all(r["ok"] for r in rep["rows"].values()) else "FAIL"
    json.dump(rep, open(K.D + "/k9full.json", "w"), indent=1)
    K.log("VERDICT", rep["verdict"])
    print("ZZ K9FULL DONE ZZ", flush=True)


if __name__ == "__main__":
    import time
    for _ in range(300):
        if K.commit_free_gb() >= 5.5:
            break
        time.sleep(60)
    else:
        print("ZZ K9FULL FAIL ZZ commit", flush=True)
        raise SystemExit(1)
    main()
