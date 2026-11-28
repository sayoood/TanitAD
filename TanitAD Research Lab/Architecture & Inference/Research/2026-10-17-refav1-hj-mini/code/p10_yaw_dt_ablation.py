"""Does the yaw-rate sampling interval matter for the yaw-rate P10?  (PI ruling D-PI-YAWRATE-1, 2026-10-17.)

The compliant mini-rig data path derives the measured yaw rate from the cache's 0.2 s ``pose_grid`` (ONE grid frame back, dt = 0.2 s;
``v8f_mini_data.yaw_kin_from_grid``) because the S4 cache holds no odd frames. Arm alpha-1's ``p10_pose`` instead differences the native
0.1 s pose track (ONE frame back, dt = 0.1 s; ``Research/2026-10-20-v8f-rung-alpha/code/p10pose_np.pose_kin``). Same function
(``ego_channels_from_poses`` + ``_pose_kappa0``), two sampling intervals. This script fits BOTH with the identical recipe
(``v8f_mini_fit_p10`` / ``floors_kin.fit_p10``: MLP, inner CV, stride 8, seed 0, the first N clips by sha12) and scores both on the
eval-141 grid, so the cost of the cache-grid choice is a MEASURED number rather than an assumption.

    python p10_yaw_dt_ablation.py <tree> <out.json> [N=540]
"""
import json
import os
import sys
import time

TREE, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 540
os.environ["V8FA1_TREE"] = TREE
RA = os.path.join(TREE, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-20-v8f-rung-alpha", "code")
for p in (os.path.join(TREE, "stack"), os.path.join(TREE, "taniteval"), RA):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import p10pose_np as L  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402

PKG = os.path.join(TREE, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "raw")
t0 = time.time()
m = torch.load("C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt", weights_only=False)
byc = {c: D.full_from_manifest(p.numpy(), a.numpy()) for c, p, a in zip(m["clip_id"], m["poses"], m["actions"])}
clips = sorted(byc, key=D.sha12)[:N]
grid = json.load(open(os.path.join(PKG, "eval_grid_windows.json")))
z = np.load(os.path.join(PKG, "eval141_kin.npz"))
off = np.concatenate([[0], np.cumsum(z["n"])])


def kin_dt01(P, ts):
    """alpha-1's spelling at the 0.2 s operative frames k = 2t: ONE-frame (0.1 s) a0 and omega0 on the native track."""
    ts = np.asarray(ts, np.int64)
    kin = L.pose_kin(P[:, 3], P[:, 2], 2 * ts)
    g = P[0::2]
    return kin, fk.gt_waypoints(g, ts, np.arange(1, D.HORIZON + 1))


def kin_dt02(P, ts):
    kin, gt, _ = D.window_kin(P, None, ts, "yaw")
    return kin, gt


def run(fn):
    kf, y, base, g, grp = [], [], [], [], []
    for gi, c in enumerate(clips):
        P = byc[c][0]
        Tg = (len(P) + 1) // 2
        ts = np.arange(3, Tg - D.HORIZON, 8)
        ts = ts[2 * ts < len(P)]
        if len(ts) == 0:
            continue
        kin, gt = fn(P, ts)
        ok = np.isfinite(kin).all(1) & np.isfinite(gt).all((1, 2))
        kin, gt = kin[ok], gt[ok]
        ha0 = fk.ha0_paths(kin[:, 0], D.DT, D.HORIZON)
        b = fk.kd_x(ha0, fk.ha0_ext_paths(kin, D.DT, D.HORIZON, action_units="kappa")).astype(np.float64)
        kf.append(fk.kin_features(kin)); base.append(b); g.append(gt); grp.append(np.full(len(kin), gi)); y.append((gt - b).reshape(len(gt), -1))
    cat = np.concatenate
    kf, y, base, g, grp = cat(kf), cat(y), cat(base), cat(g), cat(grp)
    fit = fk.fit_p10(kf, y, base, g, grp, seed=0)
    ke, be, ge = [], [], []
    for i, c in enumerate(z["clip"].tolist()):
        P = z["poses"][off[i]:off[i + 1]].astype(np.float64)
        kin, gt = fn(P, grid["windows"][c])
        ha0 = fk.ha0_paths(kin[:, 0], D.DT, D.HORIZON)
        be.append(fk.kd_x(ha0, fk.ha0_ext_paths(kin, D.DT, D.HORIZON, action_units="kappa")).astype(np.float64))
        ke.append(fk.kin_features(kin)); ge.append(gt)
    ke, be, ge = cat(ke), cat(be), cat(ge)
    pred = fk.predict_p10(fit, ke, be)
    return {"n_fit_rows": int(len(kf)), "best": [int(fit[2][0]), float(fit[2][1]), int(fit[2][2])],
            "ade6_m": float(fk.ade(pred, ge).mean()), "ade2_first10knots_m": float(fk.ade(pred[:, :10], ge[:, :10]).mean()),
            "kd_x_ade6_m": float(fk.ade(be, ge).mean())}


res = {"clips": N, "stride": 8, "seed": 0, "eval": "eval-141 grid, 2,399 windows",
       "dt_0.2s_cache_grid (the compliant mini-rig path)": run(kin_dt02),
       "dt_0.1s_native_track (alpha-1 p10_pose spelling)": run(kin_dt01), "seconds": time.time() - t0}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
