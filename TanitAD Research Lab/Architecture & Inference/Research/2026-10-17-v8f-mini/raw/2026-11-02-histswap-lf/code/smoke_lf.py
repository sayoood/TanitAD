#!/usr/bin/env python3
"""smoke_lf.py -- the 32-window real-checkpoint smoke of the HISTSWAP-LF tool (after the freeze, before the chain; <= 2 threads, MECHANICAL comparisons only, no M1 row is formed):
DAG-OACC s0 on 32 eval-141 windows with the FEAS2 flags (`--smooth sg72`, `--limit causal`, `--dump-heads`) for `--histswap off`, `cv_exec` (HCV) and `cv_logged` (HLVL).  Checks:
  (a) off equals the reused Y (KAPPA-T1 SREP_DAGOACC_s0) on the same windows, informational (2 threads vs Y's 16: thread-order noise is reported, not gated);
  (b) HCV: v0_in == v0 at EVERY tick;  HLVL: v0_in == v0 at tick 0 and differs after;  (c) tick-0 plan_raw of both differs from off (the CV history reached the network);
  (d) KC3: tick 1's executed position = tick 0's pose composed with the tick-0 plan (<= 5e-3 m) for both;  (e) the P10 floor of both equals off's;  (f) swap_ticks > 0 in the header.
python -I code/smoke_lf.py  ->  logs/smoke_lf.out"""
import glob
import importlib.util
import json
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), "/")
spec = importlib.util.spec_from_file_location("lfread", f"{HERE}/histswap_lf_read.py")
LR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LR)

S = "F:/TanitAD-scratch/histswap_lf"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
Y_DIR = "F:/TanitAD-scratch/kappa_t1/out/SREP_DAGOACC_s0/dump_SREP_DAGOACC_s0"
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")


def run(mode):
    out = f"{S}/smoke/{mode}"
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "-I", f"{S}/tree_dag/stack/scripts/v8f_t1_eval_histswap_lf.py", "run", "--ckpt", f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", "--tag", f"SMOKELF_{mode}",
           "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz",
           "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", f"{out}/ticks", "--dump-heads",
           "--device", "cpu", "--max-windows", "32", "--n-boot", "20", "--smooth", "sg72", "--histswap", mode]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(mode, "RUN FAILED rc", r.returncode, r.stderr[-800:])
        sys.exit(1)
    z = {k: v for k, v in np.load(f"{out}/ticks/ticks_SMOKELF_{mode}.npz", allow_pickle=True).items()}
    dd = {}
    for f in sorted(glob.glob(f"{out}/dump_SMOKELF_{mode}/ep*.npz")):
        e = np.load(f)
        for k in e.files:
            dd.setdefault(k, []).append(e[k])
    hdr = json.load(open(f"{out}/t1v8f_SMOKELF_{mode}.json", encoding="utf-8"))["t1v8f"]
    return z, {k: np.concatenate(v) for k, v in dd.items()}, hdr


zo, do, _ = run("off")
zc, dc, hc = run("cv_exec")
zl, dl, hl = run("cv_logged")
ys = {}
for f in sorted(glob.glob(f"{Y_DIR}/ep*.npz")):
    e = np.load(f)
    for i in range(len(e["eid"])):
        ys[(int(e["clip_index"][i]), int(e["ws"][i]))] = (e["cl"][i], e["fl"][i])
dev_cl = max(float(np.abs(do["cl"][i] - ys[(int(do["clip_index"][i]), int(do["ws"][i]))][0]).max()) for i in range(len(do["eid"])))
print(f"SMOKELF (a) off vs reused Y on {len(do['eid'])} windows: max abs cl {dev_cl:.3e} m (2 threads vs Y's 16; informational)")
assert set(zc) - set(zo) == {"v0_in", "histswap"} and set(zl) - set(zo) == {"v0_in", "histswap"} and not set(zo) - set(zc), (set(zc) ^ set(zo))
assert str(zc["histswap"]) == "cv_exec" and str(zl["histswap"]) == "cv_logged"
d_c = np.abs(zc["v0_in"].astype(np.float64) - zc["v0"].astype(np.float64))
d_l = np.abs(zl["v0_in"].astype(np.float64) - zl["v0"].astype(np.float64))
l0, later = zl["tick"] == 0, zl["tick"] >= 1
assert d_c.max() == 0.0 and d_l[l0].max() == 0.0 and (d_l[later] > 0.05).any()
print(f"SMOKELF (b) HCV: v0_in == v0 at every tick (max abs {d_c.max():.1e}); HLVL: tick 0 equal, later ticks mean |v0_in - v0| {d_l[later].mean():.3f} m/s, share > 0.05 {float((d_l[later] > 0.05).mean()):.3f}, "
      f"mean v0 exec {zl['v0'][later].mean():.3f} / logged {zl['v0_in'][later].mean():.3f}")
o0 = zo["tick"] == 0
for nm, z in (("HCV", zc), ("HLVL", zl)):
    dev = float(np.abs(z["plan_raw"][z["tick"] == 0] - zo["plan_raw"][o0]).max())
    assert dev > 1e-3, f"(c) {nm}: the CV history did not reach the network"
    print(f"SMOKELF (c) {nm}: tick-0 plan_raw differs from off by max {dev:.3f} m")
for nm, z in (("HCV", zc), ("HLVL", zl)):
    ec = LR.exec_chain_dev(z)
    assert len(ec) == 32 and ec.max() <= LR.EXEC_TOL_M, (nm, ec.max())
    print(f"SMOKELF (d) {nm}: tick-1 executed position = tick-0 pose + plan knot 10 on {len(ec)} windows, max dev {ec.max():.2e} m")
assert np.array_equal(do["fl"], dc["fl"]) and np.array_equal(do["fl"], dl["fl"])
print("SMOKELF (e) the P10 floor of HCV and HLVL equals off's bit for bit")
for nm, h in (("HCV", hc), ("HLVL", hl)):
    st = h["histswap"]["bucket_stats_model_ticks"]
    assert st["swap_ticks"] > 0
    print(f"SMOKELF (f) {nm}: swap_ticks {st['swap_ticks']}, swap_bucket_differs {st['swap_bucket_differs']}, oracle={h['histswap']['oracle']!r}")
print("SMOKELF OK")
