"""P10-yaw vs the steer P10s on eval-141 (PI ruling D-PI-YAWRATE-1): ADE@2 s and ADE@6 s with the episode-cluster bootstrap, and the
PAIRED difference (yaw - steer) on the same 2,399 windows.  Estimator named on every row: `taniteval.ci` episode-cluster bootstrap
(B = 2,000, seed 0) -- it answers "another draw of EPISODES"; it is blind to P10 training-seed variance (floor 0.0213 m, S1).

    python p10_yaw_vs_steer.py <tree> <out.json>
"""
import json
import os
import sys

TREE, OUT = sys.argv[1], sys.argv[2]
for p in (os.path.join(TREE, "stack"), os.path.join(TREE, "taniteval")):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from taniteval import ci, floors_kin as fk  # noqa: E402

RA = os.path.join(TREE, "TanitAD Research Lab", "Architecture & Inference", "Research")
MINI = os.path.join(RA, "2026-10-17-v8f-mini", "raw")
HJ = os.path.join(RA, "2026-10-17-refav1-hj-mini", "raw")
grid = json.load(open(os.path.join(MINI, "eval_grid_windows.json")))
z = np.load(os.path.join(MINI, "eval141_kin.npz"))
off = np.concatenate([[0], np.cumsum(z["n"])])
gt, eid = [], []
for i, c in enumerate(z["clip"].tolist()):
    P = z["poses"][off[i]:off[i + 1]].astype(np.float64)
    _k, g, _ = D.window_kin(P, None, grid["windows"][c], "yaw")
    gt.append(g)
    eid.append(np.full(len(g), i))
gt, eid = np.concatenate(gt), np.concatenate(eid)
FILES = {"540": (os.path.join(HJ, "p10_540", "p10_eval_pred6.npy"), os.path.join(HJ, "p10_yaw540", "p10_eval_pred6.npy")),
         "600": (os.path.join(MINI, "p10", "p10_eval_pred6.npy"), os.path.join(MINI, "p10_yaw", "p10_eval_pred6.npy"))}
out = {"estimator": "taniteval.ci episode-cluster bootstrap, B=2000, seed 0, over the 141 eval episodes (episode variance ONLY; "
                    "blind to P10 training-seed variance, floor 0.0213 m)", "n_windows": int(len(gt)), "n_episodes": int(len(np.unique(eid)))}
for n, (fs, fy) in FILES.items():
    ps, py = np.load(fs).astype(np.float64), np.load(fy).astype(np.float64)
    assert ps.shape == py.shape == gt.shape
    row = {}
    for hz, sl in (("ade6_m", slice(None)), ("ade2_m", slice(0, 10))):
        a_s, a_y = fk.ade(ps[:, sl], gt[:, sl]), fk.ade(py[:, sl], gt[:, sl])
        r_s = ci.episode_cluster_bootstrap(a_s, eid, n_boot=2000, seed=0)
        r_y = ci.episode_cluster_bootstrap(a_y, eid, n_boot=2000, seed=0)
        r_d = ci.paired_episode_cluster_bootstrap(a_y, a_s, eid, n_boot=2000, seed=0)
        row[hz] = {"steer_P10": {"mean": float(a_s.mean()), "ci95": [float(r_s["lo"]), float(r_s["hi"])]},
                   "yaw_P10": {"mean": float(a_y.mean()), "ci95": [float(r_y["lo"]), float(r_y["hi"])]},
                   "paired_yaw_minus_steer": {"delta": float(r_d["delta"]), "ci95": [float(r_d["lo"]), float(r_d["hi"])],
                                              "separated": bool(r_d["separated"])}}
    out[f"P10-{n}"] = row
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
