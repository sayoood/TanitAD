"""WHERE did M26's 0.2176 -> 0.0427 rad/s come from? A read-only decomposition on the SAME
dev-box dumps the feasible-decode chain scored (0 GPU).

Per arm (base `os`, proj07 `os`, the `ha0` floor): the per-window yaw-rate MAE the pre-fix
cell computed (every pair) and the fixed cell (pred.pair_valid AND gt.pair_valid), and how
much of the pre-fix sum sits on windows that have a step pair with NO tangent. Then the
proj07 - base difference on the windows the projection actually CHANGED, both ways.
Geometry is four_families' own `_seq_geometry` (never re-derived); the fixed cell is read
from the fixed `_components`, and the pre-fix cell is recomputed here from the same
geometry as `(|yaw_p - yaw_g|).mean(1)` -- the historical line verbatim.

usage: PYTHONPATH=<tree>/stack;<tree>/taniteval python m26_yaw_decomposition.py <tree> <out.json>
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
tree, out_path = sys.argv[1], sys.argv[2]
BASE = "C:/Users/Admin/_wp56/dump/refcv3_40284_dump"
PROJ = "C:/Users/Admin/feasdec/run/proj07_dump"
DT = 0.5

#: the fixed module must be loaded from a tools/ dir that also holds its siblings
#: (refav1_arm imports t1_eval.py beside itself) -- pass that copy as argv[3].
FIXED = sys.argv[3]
spec = importlib.util.spec_from_file_location("ra_fixed", FIXED)
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)
from taniteval import four_families as ff  # noqa: E402

assert os.path.abspath(ff.__file__).lower().startswith(os.path.abspath(tree).lower()), ff.__file__
assert "pred.pair_valid AND gt.pair_valid" in ra.YAW_RATE_CELL, "not the fixed module"


def load(d, arms):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    G, P, eid = [], {a: [] for a in arms}, []
    for f in fs:
        with np.load(f) as z:
            G.append(z["g"][..., :2].astype(np.float64))
            for a in arms:
                P[a].append(z[a][..., :2].astype(np.float64))
            eid += [os.path.basename(f)] * z["g"].shape[0]
    return np.concatenate(G), {a: np.concatenate(v) for a, v in P.items()}, np.asarray(eid)


Gb, Pb, eb = load(BASE, ["os", "ha0"])
Gp, Pp, ep = load(PROJ, ["os"])
assert np.array_equal(Gb, Gp) and np.array_equal(eb, ep), "the two dumps are not the same windows"
arms = {"base:os": Pb["os"], "proj07:os": Pp["os"], "shared:ha0": Pb["ha0"]}
gt = torch.as_tensor(Gb).float()
Gg = ff._seq_geometry(gt, DT)
out = {"tool": "2026-09-26-yaw-rate-mask/code/probes/m26_yaw_decomposition.py",
       "dumps": {"base": BASE, "proj07": PROJ}, "dt_s": DT, "min_ds_m": float(Gg["min_ds_m"]),
       "n_windows": int(len(eb)), "n_episodes": int(len(set(eb.tolist()))),
       "gt_windows_with_an_invalid_pair": int((~Gg["pair_valid"]).any(1).sum()),
       "gt_windows_with_no_valid_pair": int((~Gg["pair_valid"]).all(1).sum()),
       "arms": {}}
per_old, per_new = {}, {}
for nm, P in arms.items():
    pt = torch.as_tensor(P).float()
    Pg = ff._seq_geometry(pt, DT)
    err = (Pg["yaw_rate"] - Gg["yaw_rate"]).abs()
    old = err.mean(1).numpy().astype(np.float64)                     # the pre-fix line
    new = np.asarray(ra._components(P, Gb, DT)["LAT_yaw_rate_mae_radps"], np.float64)
    bad = (~(Pg["pair_valid"] & Gg["pair_valid"])).any(1).numpy()     # a pair with no tangent
    per_old[nm], per_new[nm] = old, new
    out["arms"][nm] = {
        "prefix_cell_mean_radps": round(float(old.mean()), 4),
        "fixed_cell_mean_radps": round(float(np.nanmean(new)), 4),
        "fixed_cell_n_windows": int(np.isfinite(new).sum()),
        "windows_with_a_pair_lacking_a_tangent": int(bad.sum()),
        "share_of_prefix_sum_on_those_windows": round(float(old[bad].sum() / old.sum()), 4),
        "prefix_mean_on_fully_valid_windows": round(float(old[~bad].mean()), 4),
        "fixed_mean_on_fully_valid_windows": round(float(np.nanmean(new[~bad])), 4),
        "windows_prefix_gt_1radps": int((old > 1.0).sum()),
    }
# the windows the projection CHANGED
chg = np.abs(arms["proj07:os"] - arms["base:os"]).reshape(len(eb), -1).max(1) > 0
keep = np.isfinite(per_new["base:os"]) & np.isfinite(per_new["proj07:os"]) & \
    np.isfinite(per_new["shared:ha0"])
out["changed_windows"] = {
    "n_changed": int(chg.sum()), "n_identical": int((~chg).sum()),
    "prefix_delta_on_changed_mean_radps": round(float((per_old["proj07:os"] - per_old["base:os"])[chg].mean()), 4),
    "prefix_delta_share_of_total_sum": round(float((per_old["proj07:os"] - per_old["base:os"])[chg].sum()
                                                   / (per_old["proj07:os"] - per_old["base:os"]).sum()), 4),
    "fixed_n_changed_scored": int((chg & keep).sum()),
    "fixed_delta_on_changed_scored_mean_radps": round(float((per_new["proj07:os"] - per_new["base:os"])[chg & keep].mean()), 4),
    "changed_windows_with_gt_pair_lacking_tangent": int((chg & (~Gg["pair_valid"]).any(1).numpy()).sum()),
}
# the headline cross cell's own kept set (A, B and the floor all finite): relative change
b, p = per_new["base:os"][keep], per_new["proj07:os"][keep]
out["fixed_cross_kept_set"] = {
    "n_windows": int(keep.sum()), "base_mean_radps": round(float(b.mean()), 4),
    "proj07_mean_radps": round(float(p.mean()), 4),
    "delta_radps": round(float(p.mean() - b.mean()), 4),
    "relative_change_pct": round(100.0 * float((p.mean() - b.mean()) / b.mean()), 2)}
bo, po = per_old["base:os"], per_old["proj07:os"]
out["prefix_cross_full_set"] = {
    "n_windows": int(len(bo)), "base_mean_radps": round(float(bo.mean()), 4),
    "proj07_mean_radps": round(float(po.mean()), 4),
    "relative_change_pct": round(100.0 * float((po.mean() - bo.mean()) / bo.mean()), 2)}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
