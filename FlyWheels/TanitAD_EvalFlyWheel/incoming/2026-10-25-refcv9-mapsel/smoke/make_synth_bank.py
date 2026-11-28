"""make_synth_bank.py -- a PLANTED world for the scorer dry-run (no model, no GPU): real window keys / human paths of the first K episodes; 149 candidates = the human path + a lateral offset
delta_k = linspace(-6, 6, 149) * ramp(slot); sel_score_v3 = |delta| (so the NATIVE pick is the most off-corridor candidate); GT corridor |y - y_h(x)| < 2.5 m drivable (code 1), a lane line at
+1.9 m (code 2), an edge strip [2.5, 2.8) (code 5), the rest sidewalk (7); OWN planes = the GT planes exactly (perfect map), so OWN-Tk must pick exactly what GT-Tk picks."""
import sys, json, os, shutil
import numpy as np
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_mapsel/code"); sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import mapsel_data as D, mapsel_lib as L, l2_common as C
K = int(sys.argv[1]); OUT = sys.argv[2]
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT + "/rast")
E = C.load_eval_pf(0)
idx = D.decisions_index(0)
keys = sorted(k for k in idx if k[0] < K)
hp, valid = D.human_plans(keys)
base_plan = {}
N = 149
delta = np.linspace(-6, 6, N).astype(np.float32)
ramp = (np.arange(8) + 1) / 8.0
fan = np.zeros((len(keys), N, 8, 2), np.float32)
Y = (np.arange(L.W) + 0.5) * 0.1 - 30.0
X = (np.arange(L.H) + 0.5) * 0.1
sv3 = np.zeros((len(keys), N), np.float32)
for j, (c, w) in enumerate(keys):
    h = hp[j] if valid[j] else np.stack([np.linspace(5, 60, 8), np.zeros(8)], -1).astype(np.float32)
    fan[j] = h[None] + np.stack([np.zeros((N, 8)), delta[:, None] * ramp[None]], -1)
    sv3[j] = np.abs(delta) + 1e-3 * np.arange(N) / N
    xs = np.concatenate([[0.0], h[:, 0]]); ys = np.concatenate([[0.0], h[:, 1]])
    order = np.argsort(xs); yh = np.interp(X, xs[order], ys[order])
    dy = Y[None, :] - yh[:, None]
    gt = np.full((L.H, L.W), 7, np.uint8)
    gt[np.abs(dy) < 2.5] = 1
    gt[(dy >= 1.85) & (dy < 1.95)] = 2
    gt[(dy >= 2.5) & (dy < 2.8)] = 5
    gt[:30, :] = 255 if False else gt[:30, :]
    pd = np.where(np.isin(gt, L.DRV_CODES), 255, 0).astype(np.uint8)
    pe = np.where(gt == 5, 255, 0).astype(np.uint8)
    psw = np.where(gt == 7, 255, 0).astype(np.uint8)
    np.savez_compressed(f"{OUT}/rast/e{c:03d}_w{w:04d}.npz", pdrv=pd, pedge=pe, psw=psw, code=gt.copy(), gt=gt)
n = len(keys)
sel = sv3.argmax(1)
tr = {"e_i": np.array([k[0] for k in keys], np.int64), "t": np.array([k[1] - 7 for k in keys], np.int64), "ws": np.array([k[1] for k in keys], np.int64),
      "plan8": fan[np.arange(n), sel], "sel": sel.astype(np.int64), "sel_base": sel.astype(np.int64), "v0_in": np.zeros(n),
      "gt_label": np.ones(n, bool), "acc_own": np.ones(n), "fan": fan, "sel_score": sv3, "sel_score_v3": sv3,
      "reach_keep": np.ones((n, N), np.int8), "ceil_keep": np.full((n, N), -1, np.int8), "emit_keep": np.ones((n, N), np.int8),
      "prior_path": np.zeros((n, 8, 2), np.float32)}
meta = {"final": True, "n_windows": n, "present_counts": {k: n for k in ("fan", "sel_score", "sel_score_v3", "prior_path")}, "window_W": 8, "n_stack": 3,
        "counters": {"fan_n_seen": [149], "batch_gt1": 0, "rule_seen": ["prior_corrected"], "logit_shapes": [[1, 8, 1000, 600]], "decide_mismatch_cells": 0, "decide_checked": 25}}
tr["meta_json"] = np.array(json.dumps(meta))
np.savez_compressed(OUT + "/trace.npz", **tr)
print("synthetic bank", n, "windows,", len(set(tr["e_i"].tolist())), "episodes ->", OUT)
