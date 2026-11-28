"""Cross-check my numpy checker (labeller.controls) against the stack's own torch recover_controls (feasible_decode.py at the tip, copy in ../src_tip)
on real emitted V1 labels from the real panel. Needs torch (run only with free RAM >= 3 GB). Writes ../raw/crosscheck_feasible_decode.json.
Where a path has no zero-length step, accel / lat_acc / speed must agree to float tolerance (the stack holds no heading on a stopped step; mine does, so paths
with a step shorter than 1e-9 m are excluded and counted).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import labeller as L  # noqa: E402

spec = importlib.util.spec_from_file_location("feasible_decode_tip", HERE.parent / "src_tip" / "feasible_decode.py")
FD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FD)

z = np.load(HERE.parent / "raw" / "real_panel.npz")
n_ok = n_skip = 0
max_d = {"speed": 0.0, "accel": 0.0, "lat_acc": 0.0}
for i, (kind, c, T, r) in enumerate(z["meta"]):
    if kind != 1:
        continue
    s, n = int(z["starts"][i]), int(T - r)
    g, v = z["G"][s:s + n], z["V"][s:s + n]
    ox, oy, oyaw, vown = z["owns"][i]
    res = L.recovery_label(g, v, 0, (ox, oy, oyaw), float(vown))
    if res["label_own"] is None:
        continue
    pts = np.concatenate([[[-float(vown) * L.DT, 0.0]], [[0.0, 0.0]], res["label_own"]])
    if (np.hypot(*(pts[1:] - pts[:-1]).T) < 1e-9).any():
        n_skip += 1
        continue
    sp, ac, lat, kap = L.controls(pts)
    t_sp, t_h, t_ac, t_lat = FD.recover_controls(torch.as_tensor(pts, dtype=torch.float64), dt=L.DT, min_speed=L.MIN_SPEED)
    max_d["speed"] = max(max_d["speed"], float(np.abs(sp - t_sp.numpy()).max()))
    max_d["accel"] = max(max_d["accel"], float(np.abs(ac - t_ac.numpy()).max()))
    max_d["lat_acc"] = max(max_d["lat_acc"], float(np.abs(lat - t_lat.numpy()).max()))
    n_ok += 1
out = {"n_compared": n_ok, "n_skipped_zero_length_step": n_skip, "max_abs_diff": max_d, "PASS": bool(n_ok > 0 and max(max_d.values()) < 1e-9)}
Path(HERE.parent / "raw" / "crosscheck_feasible_decode.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
