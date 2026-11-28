#!/usr/bin/env python3
"""smoke_fz.py -- the 32-window real-checkpoint smoke of the FRAMEFREEZE tool (before the chain; 16 threads, MECHANICAL comparisons only, no M1 row is formed):
DAG-OACC s0 on 32 eval-141 windows with the production flags (`--smooth sg72`, `--limit causal`, `--dump-heads`) for A = `--scene logged`, B = `--scene hold_now`, C = `--scene hold_now --histswap cv_logged`.  Checks:
  (a) A equals the reused Y (KAPPA-T1 SREP_DAGOACC_s0) on the same windows: informational (batch composition and thread order can move the last bits; reported, not gated);
  (b) KFZ1 on the real rig: the run headers' freeze receipts of B and C (every item: max row deviation exactly 0.0, no unequal element), camera_motion false, strict.camera false, the card ref recorded; A carries none;
  (c) KFZ3 analogue: B's plan_raw differs from A's by > 1e-3 m at ticks >= 1 (and at tick 0, where the two share the executed state: the freeze alone);
  (d) C: v0_in == v0 at tick 0 and differs after (the oracle level reached the network) while the frames stay frozen; B carries no histswap key; both ticks files carry scene == hold_now;
  (e) KC3 for B and C: tick 1's executed position = tick 0's pose composed with the tick-0 plan (<= 5e-3 m);  (f) windows and the P10 floor of A, B, C are equal.
python -I code/smoke_fz.py  ->  logs/smoke_fz.out"""
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace(chr(92), "/")
spec = importlib.util.spec_from_file_location("framefreeze_read", f"{HERE}/framefreeze_read.py")
FR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FR)
FR.heavy_init()                                              # HISTSWAP-LF's reader by file path (pinned): exec_chain_dev, tickload

S = "F:/TanitAD-scratch/framefreeze"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
Y_DIR = "F:/TanitAD-scratch/kappa_t1/out/SREP_DAGOACC_s0/dump_SREP_DAGOACC_s0"
CARD_LINE = open(f"{FR.FT_PK}/FROZEN_SHA.txt", encoding="utf-8").readline().strip()
TH = 16
env = dict(os.environ, OMP_NUM_THREADS=str(TH), MKL_NUM_THREADS=str(TH), OPENBLAS_NUM_THREADS=str(TH), PYTHONDONTWRITEBYTECODE="1")


def cpu_load():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"], capture_output=True, text=True)
    return float(r.stdout.strip() or "nan")


def run(label, *flags):
    out = f"{S}/smoke/{label}"
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "-I", f"{S}/tree_dag/stack/scripts/v8f_t1_eval_framefreeze.py", "run", "--ckpt", f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", "--tag", f"SMOKEFZ_{label}",
           "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz",
           "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", f"{out}/ticks", "--dump-heads",
           "--device", "cpu", "--max-windows", "32", "--n-boot", "20", "--smooth", "sg72", *flags]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(label, "RUN FAILED rc", r.returncode, r.stderr[-800:])
        sys.exit(1)
    z = {k: v for k, v in np.load(f"{out}/ticks/ticks_SMOKEFZ_{label}.npz", allow_pickle=True).items() if k not in ("tag", "ckpt")}
    dd = {}
    for f in sorted(glob.glob(f"{out}/dump_SMOKEFZ_{label}/ep*.npz")):
        e = np.load(f)
        for k in e.files:
            dd.setdefault(k, []).append(e[k])
    hdr = json.load(open(f"{out}/t1v8f_SMOKEFZ_{label}.json", encoding="utf-8"))["t1v8f"]
    print(f"SMOKEFZ run {label}: {time.time() - t0:.0f} s, {len(z['tick'])} tick rows", flush=True)
    return z, {k: np.concatenate(v) for k, v in dd.items()}, hdr


loads = [cpu_load() for _ in range(3)]
print(f"SMOKEFZ cpu LoadPercentage before the runs (3 samples): {loads}; threads {TH}; card line: {CARD_LINE}", flush=True)
za, da, ha = run("A_logged", "--scene", "logged")
zb, db, hb = run("B_hold", "--scene", "hold_now", "--freeze-card-ref", CARD_LINE)
zc, dc, hc = run("C_hold_cv", "--scene", "hold_now", "--histswap", "cv_logged", "--freeze-card-ref", CARD_LINE)
ys = {}
for f in sorted(glob.glob(f"{Y_DIR}/ep*.npz")):
    e = np.load(f)
    for i in range(len(e["eid"])):
        ys[(int(e["clip_index"][i]), int(e["ws"][i]))] = e["cl"][i]
dev_cl = max(float(np.abs(da["cl"][i] - ys[(int(da["clip_index"][i]), int(da["ws"][i]))]).max()) for i in range(len(da["eid"])))
print(f"SMOKEFZ (a) A (--scene logged) vs reused Y on {len(da['eid'])} windows: max abs cl {dev_cl:.3e} m (informational)")
# (b) KFZ1 on the real rig
assert "frame_freeze" not in ha and "camera_motion" not in ha and "scene" not in za
for nm, h, z in (("B", hb, zb), ("C", hc, zc)):
    rc = FR.freeze_receipt(h, len(z["tick"]))
    assert rc["PASS"], (nm, rc)
    assert h["frame_freeze"]["card_ref"] == CARD_LINE and str(z["scene"]) == "hold_now"
    print(f"SMOKEFZ (b) KFZ1 {nm}: items_frozen {rc['items_frozen']} = {rc['items_per_tick_row']:.0f} x {rc['ticks_rows']} tick rows, max row dev {rc['max_row_dev']}, unequal elements {rc['n_unequal_elements']}, "
          f"camera_motion {rc['camera_motion']}, strict.camera {rc['strict_camera']}, card ref recorded")
# (c) KFZ3 analogue
k3 = FR.kfz3(zb, za)
assert k3["PASS"], k3
print(f"SMOKEFZ (c) KFZ3 B vs A: max plan_raw dev at ticks >= 1 {k3['max_dev_ticks_ge1_m']:.3f} m (median {k3['median_dev_ticks_ge1_m']:.3f}, {k3['frac_rows_ticks_ge1_over_tol']:.3f} of rows over 1e-3 m), "
      f"tick 0 (shared executed state) {k3['max_dev_tick0_m']:.3f} m ({k3['frac_rows_tick0_over_tol']:.3f} of rows over 1e-3 m)")
k3c = FR.kfz3(zc, za)
assert k3c["PASS"]
print(f"SMOKEFZ (c) C vs A: max plan_raw dev at ticks >= 1 {k3c['max_dev_ticks_ge1_m']:.3f} m, tick 0 {k3c['max_dev_tick0_m']:.3f} m")
# (d) histswap composition
assert "histswap" not in zb and str(zc["histswap"]) == "cv_logged"
d_c = np.abs(zc["v0_in"].astype(np.float64) - zc["v0"].astype(np.float64))
c0, later = zc["tick"] == 0, zc["tick"] >= 1
assert d_c[c0].max() == 0.0 and (d_c[later] > 0.05).any()
print(f"SMOKEFZ (d) C: v0_in == v0 at tick 0; later ticks mean |v0_in - v0| {d_c[later].mean():.3f} m/s, share > 0.05 {float((d_c[later] > 0.05).mean()):.3f}; B has no histswap key")
# (e) KC3
for nm, z in (("B", zb), ("C", zc)):
    ec = FR.LR.exec_chain_dev(z)
    assert len(ec) == 32 and ec.max() <= FR.LR.EXEC_TOL_M, (nm, ec.max())
    print(f"SMOKEFZ (e) KC3 {nm}: tick-1 executed position = tick-0 pose + plan knot 10 on {len(ec)} windows, max dev {ec.max():.2e} m")
# (f) windows and floors
for nm, d in (("B", db), ("C", dc)):
    assert all(np.array_equal(d[q], da[q]) for q in ("clip_index", "ws", "eid", "g", "v0")) and np.array_equal(d["fl"], da["fl"]), nm
print("SMOKEFZ (f) windows and the P10 floor of A, B, C are equal")
move = float(np.abs(db["cl"] - da["cl"]).max())
print(f"SMOKEFZ (h) the freeze moved the executed path: max abs cl dev B vs A {move:.3f} m")
print("SMOKEFZ OK")
