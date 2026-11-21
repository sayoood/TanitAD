"""Open-loop ADE for REF-C in AlpaSim (force-GT). Scores logged rig-frame predictions vs the
GT ego path, exactly as taniteval: de@Ts = mean L2 at each horizon; ade_0_2s = mean of the 4.
Compares to REF-C base's known taniteval open-loop ADE (0.4728 on real PhysicalAI val)."""
import json, sys
from collections import defaultdict
import numpy as np

HOR = [0.5, 1.0, 1.5, 2.0]
preds = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = defaultdict(list)
for p in preds:
    by[p["session"]].append(p)

de_by_h = defaultdict(list)
per_scene = {}
for sess, ps in by.items():
    ps.sort(key=lambda p: p["t"])
    ts = np.array([p["t"] for p in ps], dtype=float) / 1e6      # s
    xs = np.array([p["x"] for p in ps]); ys = np.array([p["y"] for p in ps])
    sde = [[] for _ in HOR]
    moved = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    for p in ps:
        t0 = p["t"] / 1e6; yaw = p["yaw"]; x0 = p["x"]; y0 = p["y"]
        cy, sy = np.cos(yaw), np.sin(yaw)
        for hi, h in enumerate(HOR):
            tf = t0 + h
            if tf > ts[-1]:
                continue                                        # no future GT
            gx = float(np.interp(tf, ts, xs)); gy = float(np.interp(tf, ts, ys))
            rgx = cy * (gx - x0) + sy * (gy - y0)               # rig x fwd
            rgy = -sy * (gx - x0) + cy * (gy - y0)              # rig y left
            px, py = p["pred_rig"][hi]
            de = float(((px - rgx) ** 2 + (py - rgy) ** 2) ** 0.5)
            sde[hi].append(de); de_by_h[hi].append(de)
    hm = [float(np.mean(s)) if s else float("nan") for s in sde]
    per_scene[sess[:20]] = {"n_pred": len(ps), "gt_moved_m": round(moved, 1),
                            "ade": round(float(np.nanmean(hm)), 4),
                            "de@Ts": [round(v, 4) for v in hm]}

hmeans = [float(np.mean(de_by_h[hi])) for hi in range(4)]
ade = float(np.mean(hmeans))
out = {"open_loop_ade_alpasim": round(ade, 4),
       "de@0.5s": round(hmeans[0], 4), "de@1s": round(hmeans[1], 4),
       "de@1.5s": round(hmeans[2], 4), "de@2s": round(hmeans[3], 4),
       "n_scenes": len(by), "n_scored_predictions": len(de_by_h[0]),
       "taniteval_ref_base_ade": 0.4728,
       "ratio_alpasim_over_taniteval": round(ade / 0.4728, 2),
       "per_scene": per_scene}
print(json.dumps(out, indent=2))
