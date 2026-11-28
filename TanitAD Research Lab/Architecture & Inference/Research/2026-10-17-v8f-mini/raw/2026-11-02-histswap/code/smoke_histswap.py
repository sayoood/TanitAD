#!/usr/bin/env python3
"""smoke_histswap.py -- the 32-window real-checkpoint smoke of the HISTSWAP tool (after the freeze, before the chain; <= 2 threads, MECHANICAL comparisons only, no M1 row is formed):
run DAG-OACC s0 on 32 eval-141 windows twice with the FEAS2 flags (`--smooth sg72`, `--limit causal`, `--dump-heads`): `--histswap off` (O) and `--histswap logged` (X).  Checks:
  (a) O equals the reused Y (KAPPA-T1 SREP_DAGOACC_s0) on the same windows, informational: the reused run used 16 threads, this one 2, so the float thread-order noise is reported, not gated;
  (b) tick 0 of X equals tick 0 of O bit for bit (tau, P, plan, plan_raw, v0, pose, gt);  (c) tick 1's executed pose and v0 equal O's bit for bit (the executor ran from the executed state);
  (d) v0_in == v0 at tick 0 and differs after;  (e) the P10 floor of X equals O's;  (f) the tick-1 plan differs (the swap reached the network).
python -I code/smoke_histswap.py  ->  logs/smoke_histswap.out"""
import glob
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

S = "F:/TanitAD-scratch/histswap"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
Y_DIR = "F:/TanitAD-scratch/kappa_t1/out/SREP_DAGOACC_s0/dump_SREP_DAGOACC_s0"
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")


def run(mode):
    out = f"{S}/smoke/{mode}"
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "-I", f"{S}/tree_dag/stack/scripts/v8f_t1_eval_histswap.py", "run", "--ckpt", f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", "--tag", f"SMOKEH_{mode}",
           "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz",
           "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", f"{out}/ticks", "--dump-heads",
           "--device", "cpu", "--max-windows", "32", "--n-boot", "20", "--smooth", "sg72", "--histswap", mode]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(mode, "RUN FAILED rc", r.returncode, r.stderr[-800:])
        sys.exit(1)
    z = {k: v for k, v in np.load(f"{out}/ticks/ticks_SMOKEH_{mode}.npz", allow_pickle=True).items()}
    dd = {}
    for f in sorted(glob.glob(f"{out}/dump_SMOKEH_{mode}/ep*.npz")):
        e = np.load(f)
        for k in e.files:
            dd.setdefault(k, []).append(e[k])
    return z, {k: np.concatenate(v) for k, v in dd.items()}


zo, do = run("off")
zx, dx = run("logged")
# (a) O vs the reused Y on the same windows (informational)
ys = {}
for f in sorted(glob.glob(f"{Y_DIR}/ep*.npz")):
    e = np.load(f)
    for i in range(len(e["eid"])):
        ys[(int(e["clip_index"][i]), int(e["ws"][i]))] = (e["cl"][i], e["fl"][i])
dev_cl = max(float(np.abs(do["cl"][i] - ys[(int(do["clip_index"][i]), int(do["ws"][i]))][0]).max()) for i in range(len(do["eid"])))
dev_fl = max(float(np.abs(do["fl"][i] - ys[(int(do["clip_index"][i]), int(do["ws"][i]))][1]).max()) for i in range(len(do["eid"])))
print(f"SMOKEH (a) off vs reused Y on {len(do['eid'])} windows: max abs cl {dev_cl:.3e} m, fl {dev_fl:.3e} m (2 threads vs Y's 16; informational)")
o0, x0, o1, x1 = (zo["tick"] == 0), (zx["tick"] == 0), (zo["tick"] == 1), (zx["tick"] == 1)
assert set(zx) - set(zo) == {"v0_in", "histswap"} and not set(zo) - set(zx), set(zx) ^ set(zo)
for k in ("tau", "P", "plan", "plan_raw", "v0", "pose", "gt", "s_ckpt", "t_goal_prob"):
    assert np.array_equal(zo[k][o0], zx[k][x0], equal_nan=True), f"(b) tick 0 differs in {k}"
print("SMOKEH (b) tick 0 of X equals O bit for bit: tau P plan plan_raw v0 pose gt s_ckpt t_goal_prob")
for k in ("pose", "v0"):
    assert np.array_equal(zo[k][o1], zx[k][x1]), f"(c) tick 1 executed {k} differs"
print("SMOKEH (c) tick 1 executed pose and v0 of X equal O bit for bit")
d_in = np.abs(zx["v0_in"].astype(np.float64) - zx["v0"].astype(np.float64))
later = zx["tick"] >= 1
assert d_in[x0].max() == 0.0 and (d_in[later] > 0.05).any()
print(f"SMOKEH (d) v0_in == v0 at tick 0; later ticks: mean |v0_in - v0| {d_in[later].mean():.3f} m/s, share > 0.05 m/s {float((d_in[later] > 0.05).mean()):.3f}, mean v0 (executed) {zx['v0'][later].mean():.3f}, mean v0_in (logged) {zx['v0_in'][later].mean():.3f}")
assert np.array_equal(do["fl"], dx["fl"]), "(e) the P10 floor changed"
print("SMOKEH (e) the P10 floor of X equals O's bit for bit")
assert not np.array_equal(zo["plan_raw"][o1], zx["plan_raw"][x1])
print(f"SMOKEH (f) the swap reached the network: tick-1 plan_raw differs by max {float(np.abs(zo['plan_raw'][o1] - zx['plan_raw'][x1]).max()):.3f} m")
print("SMOKEH OK")
