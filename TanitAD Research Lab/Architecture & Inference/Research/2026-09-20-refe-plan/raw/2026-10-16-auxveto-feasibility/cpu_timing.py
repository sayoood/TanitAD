#!/usr/bin/env python3
"""cpu_timing.py -- MEASURE the CPU cost of the aux head (train update, predict) with the probe's own train_unit/predict_head on synthetic tensors of the REAL shapes.
TIMING ONLY: random ctx, random labels, no metric.  GPU never touched (device="cpu", CUDA hidden).  Writes CPU_TIMING.json."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
spec_path = PK / "raw/2026-10-16-scene-tap-a2/scene_tap_probe.py"
sys.path.insert(0, str(spec_path.parent)); sys.path.insert(0, str(PK / "refe"))
spec = importlib.util.spec_from_file_location("stp_a2", spec_path)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
n = 24
g = torch.Generator().manual_seed(0)
ctx = torch.randn(n, 7680, 256, generator=g)
lab = torch.where(torch.rand(n, 160, 128, generator=g) > 0.5, 60, -60).to(torch.int8)
out = {"threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(), "shapes": {"ctx": list(ctx.shape), "batch": M.BATCH}}
runs = []
for n_upd in (2, 8):
    t = time.time()
    head, info = M.train_unit(ctx, lab, list(range(16)), list(range(16, 24)), "visual", 0, 0, n_updates=n_upd, device="cpu", draw=None)
    runs.append({"updates": n_upd, "wall_s": time.time() - t, "train_loop_wall_s": info["wall_s"]})
out["train_runs"] = runs
dt = (runs[1]["train_loop_wall_s"] - runs[0]["train_loop_wall_s"]) / (runs[1]["updates"] - runs[0]["updates"])
out["cpu_s_per_update_marginal"] = dt
t = time.time(); M.predict_head(head, ctx, list(range(16)), device="cpu"); out["cpu_predict_s_per_token"] = (time.time() - t) / 16
json.dump(out, open(HERE / "CPU_TIMING.json", "w"), indent=1)
print(json.dumps(out, indent=1))
