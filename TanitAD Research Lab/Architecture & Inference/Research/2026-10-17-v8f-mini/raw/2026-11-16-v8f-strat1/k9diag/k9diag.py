"""K9 discriminating experiment (MM 2026-11-16). K9 = T1 tick-0 `s_ckpt` against the T0 real-nav `ckpt` of the same window.
The smoke read K9 FAIL (12 of 48 windows, max 10.6 m S1_s0, 4.4 m C0_s0). Mechanism found in code: the strategic brain's run_s takes a
speed-limit token (v8f_mini.py:978, e_lim(limit, limit_ok)); the T1 dumps run `--limit causal` (bucket from past speed), T0 feeds the
logged limit. Committed before running:
  * K9(logged) <= 1e-6 at both arms and K9(causal) > 1e-6  -> the limit mode explains K9; the harness feeds T0's inputs at tick 0.
  * K9(logged) > 1e-6 at an arm                              -> another tick-0 input differs; trace further (nav, history, pose).
Smoke grid only (inputs/grid_smoke.json, 6 episodes): a code check, never a result. CPU, 2 threads. Output: k9diag.json + per-window rows.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

R = "F:/TanitAD-scratch/strat1/read"
D = "F:/TanitAD-scratch/strat1/k9diag"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
GRID = R + "/inputs/grid_smoke.json"
ARMS = ("S1_s0", "C0_s0")
TEXT = {"S1_s0": ["--nav-text-npz", R + "/inputs/navtext_nav.npz", "--nav-text-sha256",
                  "03bd75bc3a3de17928681bedb8e46a48a891304de45d395c49cc876acf6e6160"], "C0_s0": []}
sys.path.insert(0, R)
env = dict(os.environ)
env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=";".join(os.path.normpath(p) for p in (R + "/tree/stack", R + "/tree/taniteval")))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def base_argv():
    lines = open(R + "/t1/argv_STRAT1-S1_s1_m10_causal_heads.txt", encoding="utf-8").read().splitlines()
    out = []
    for ln in lines:
        if not ln.strip() or ln.startswith("STRAT1 extra"):
            break
        out.append(ln)
    return out


def set_flag(argv, flag, val):
    i = argv.index(flag)
    argv[i + 1] = val


def t1(arm, lim):
    a = base_argv()
    tag = f"K9DIAG-{arm}-{lim}"
    od = f"{D}/t1_{arm}_{lim}"
    os.makedirs(od + "/ticks", exist_ok=True)
    set_flag(a, "--ckpt", f"{R}/ckpts/{arm}/ckpt_last.pt")
    set_flag(a, "--tag", tag)
    set_flag(a, "--eval-grid", GRID)
    set_flag(a, "--out-dir", od)
    set_flag(a, "--dump-ticks", od + "/ticks")
    set_flag(a, "--limit", lim)
    a = a + TEXT[arm] + ["--dump-nav"]
    p = f"{od}/ticks/ticks_{tag}.npz"
    if not os.path.exists(p):
        r = subprocess.run(a, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        open(f"{od}/log.txt", "w", encoding="utf-8").write(" ".join(a) + "\n" + r.stdout[-4000:] + "\n" + r.stderr[-4000:])
        log("T1", arm, lim, "rc", r.returncode)
        if r.returncode != 0:
            raise SystemExit(f"T1 {arm} {lim} failed, see {od}/log.txt")
    return dict(np.load(p, allow_pickle=True))


def main():
    os.makedirs(D + "/t0", exist_ok=True)
    if not all(os.path.exists(f"{D}/t0/eval_STRAT1-{a}_real.npz") for a in ARMS):
        r = subprocess.run([PY, R + "/run_t0_strat1.py", "--arms", *ARMS, "--variants", "real", "--k4", "--grid", GRID, "--out", D + "/t0"],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=R)
        open(D + "/t0_log.txt", "w", encoding="utf-8").write(r.stdout[-6000:] + "\n" + r.stderr[-6000:])
        log("T0 rc", r.returncode)
        if r.returncode != 0:
            raise SystemExit("T0 failed, see t0_log.txt")
    import strat1_strategic as SS
    _, _, _, ci, t, _ = SS.load_inputs(GRID)
    ci_t = {(int(c), int(tt)): i for i, (c, tt) in enumerate(zip(ci, t))}
    rep = {"tool": "k9diag.py", "grid": "inputs/grid_smoke.json (smoke; a code check, never a result)", "rows": {}}
    for arm in ARMS:
        z = dict(np.load(f"{D}/t0/eval_STRAT1-{arm}_real.npz", allow_pickle=True))
        assert np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)
        for lim in ("causal", "logged"):
            tk = t1(arm, lim)
            mx, n, ok = SS.k9(tk, z["ckpt"], ci_t)
            m = np.asarray(tk["tick"]) == 0
            idx = [ci_t[(int(c), int(tt))] for c, tt in zip(np.asarray(tk["ci"])[m], np.asarray(tk["t0"])[m])]
            d = np.abs(np.asarray(tk["s_ckpt"], np.float64)[m] - np.asarray(z["ckpt"], np.float64)[idx]).reshape(int(m.sum()), -1).max(1)
            bad = [(int(c), int(tt), float(x)) for c, tt, x in zip(np.asarray(tk["ci"])[m], np.asarray(tk["t0"])[m], d) if x > 1e-6]
            rep["rows"][f"{arm}/{lim}"] = {"K9_max_abs_diff": mx, "n_windows": n, "ok": ok, "n_windows_over_1e-6": len(bad),
                                           "episodes_over": sorted({b[0] for b in bad}), "windows_over": bad[:60]}
            log(arm, lim, "K9 max", round(mx, 6), "n", n, "over", len(bad), "ok", ok)
    rows = rep["rows"]
    lg = all(rows[f"{a}/logged"]["ok"] for a in ARMS)
    cs = all(not rows[f"{a}/causal"]["ok"] for a in ARMS)
    rep["verdict"] = ("EXPLAINED: the limit mode is the tick-0 difference" if lg and cs else
                      "NOT EXPLAINED: a tick-0 input other than the limit differs" if not lg else
                      "CAUSAL PASSES ON THIS GRID: the smoke failure did not reproduce")
    json.dump(rep, open(D + "/k9diag.json", "w"), indent=1)
    log("VERDICT", rep["verdict"])
    print("ZZ K9DIAG DONE ZZ", flush=True)


def commit_free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


if __name__ == "__main__":
    # the 18:02Z STRAT1 dumps died of commit exhaustion: start only with >= 5.5 GB of commit free (checked every 60 s, up to 5 h)
    for _ in range(300):
        cf = commit_free_gb()
        if cf >= 5.5:
            break
        time.sleep(60)
    else:
        print("ZZ K9DIAG FAIL ZZ commit stayed under 5.5 GB for 5 h", flush=True)
        raise SystemExit(1)
    log("commit free GB", round(cf, 2))
    main()
