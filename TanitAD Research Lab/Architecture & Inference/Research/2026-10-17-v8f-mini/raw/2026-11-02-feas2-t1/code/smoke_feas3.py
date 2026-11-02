#!/usr/bin/env python3
"""smoke_feas3.py -- the 32-window real-checkpoint smoke of the FEAS3 tool (before the chain): for each new mode run the tool on DAG-OACC s0, 32 eval-141 windows, 16 threads, then compare the issued `plan`
of every tick with the READER's independent operator (feas3_t1_read.lever(plan_s, P, mode)) and report how much the operator changed.   python -I code/smoke_feas3.py  ->  logs/smoke_feas3.out"""
import importlib.util
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
S = "F:/TanitAD-scratch/feas2_t1"
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
spec = importlib.util.spec_from_file_location("feas3_read", f"{HERE}/feas3_t1_read.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
env = dict(os.environ, OMP_NUM_THREADS="16", MKL_NUM_THREADS="16", OPENBLAS_NUM_THREADS="16", PYTHONDONTWRITEBYTECODE="1")
for mode in ("kbox2", "lonfloor", "kbox2lf"):
    out = f"{S}/smoke3/{mode}"
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "-I", f"{S}/tree_dag/stack/scripts/v8f_t1_eval_feas3.py", "run", "--ckpt", f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", "--tag", f"SMOKE3_{mode}",
           "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache", "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz",
           "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal", "--dump-ticks", f"{out}/ticks", "--dump-heads",
           "--device", "cpu", "--max-windows", "32", "--n-boot", "20", "--smooth", "sg72", "--feasible", mode]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(mode, "RUN FAILED rc", r.returncode, r.stderr[-600:])
        sys.exit(1)
    z = np.load(f"{out}/ticks/ticks_SMOKE3_{mode}.npz", allow_pickle=True)
    ref = R.lever(z["plan_s"], z["P"], mode)
    d = float(np.abs(ref - z["plan"]).max())
    ch = np.abs(z["plan"] - z["plan_s"])
    print(f"SMOKE3 {mode}: ticks {len(z['tick'])}, max abs (reader operator - issued plan) = {d:.3e} m; operator moved the plan on {float((ch.max(axis=(1, 2)) > 1e-3).mean()):.3f} of ticks, max {float(ch.max()):.3f} m; "
          f"executed knots 1-10 moved on {float((ch[:, :10].max(axis=(1, 2)) > 1e-3).mean()):.3f} of ticks", flush=True)
    assert d <= 1e-4, mode
print("SMOKE3 OK")
