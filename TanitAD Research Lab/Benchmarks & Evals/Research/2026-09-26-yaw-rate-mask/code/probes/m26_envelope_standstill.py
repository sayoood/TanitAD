"""ADJACENT to D-YAWMASK-1 (not the `_components` defect): WHERE are M26's envelope
violations? M26's safety headline is `envelope` 0.0865 -> 0.0000 (fan_safety.score_paths on
the driven `os` path, feasible-decode/raw/derived_dump_report.json). The projection changed
421 of 4,823 windows and 306 of those have a GT step pair with no tangent, so this asks, per
window: is a base violation a MOVING-path violation or a standstill/crawl one?

Reproduction control first: the tip's fan_safety must reproduce the banked base envelope
0.08646070957183838 exactly on the same dump, or nothing below is quoted.

usage: PYTHONPATH=<tree>/stack;<tree>/taniteval python m26_envelope_standstill.py <tools_dir> <out.json>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
tools_dir, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, tools_dir)
import fan_safety as FS  # noqa: E402
from taniteval import four_families as ff  # noqa: E402

BASE = "C:/Users/Admin/_wp56/dump/refcv3_40284_dump"
PROJ = "C:/Users/Admin/feasdec/run/proj07_dump"
BANKED_BASE_ENVELOPE = 0.08646070957183838
DT = 0.5


def load(d):
    os_, g, v0, eid = [], [], [], []
    for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        with np.load(f, allow_pickle=True) as z:
            os_.append(z["os"].astype(np.float32))
            g.append(z["g"][..., :2].astype(np.float32))
            v0.append(z["v0"].astype(np.float32).reshape(-1))
            eid += [os.path.basename(f)] * z["g"].shape[0]
    return np.concatenate(os_), np.concatenate(g), np.concatenate(v0), np.asarray(eid)


bo, g, v0, eid = load(BASE)
po, g2, v02, eid2 = load(PROJ)
assert np.array_equal(g, g2) and np.array_equal(v0, v02) and np.array_equal(eid, eid2)
v0_t = torch.from_numpy(v0).float()
sb = FS.score_paths(FS.with_origin(torch.from_numpy(bo).float()), v0_t, None)
sp = FS.score_paths(FS.with_origin(torch.from_numpy(po).float()), v0_t, None)
env_b = sb["envelope"].bool().numpy()
env_p = sp["envelope"].bool().numpy()
repro = float(sb["envelope"].float().mean())
out = {"tool": "2026-09-26-yaw-rate-mask/code/probes/m26_envelope_standstill.py",
       "scope": ("ADJACENT to the _components defect: a different instrument "
                 "(fan_safety.score_paths); reported because the same standstill "
                 "windows carry M26's other headline"),
       "n_windows": int(len(eid)), "n_episodes": int(len(set(eid.tolist()))),
       "reproduction": {"banked_base_envelope": BANKED_BASE_ENVELOPE,
                        "tip_fan_safety_base_envelope": repro,
                        "exact": repro == BANKED_BASE_ENVELOPE}}
Pg = ff._seq_geometry(torch.from_numpy(bo).float(), DT)
Gg = ff._seq_geometry(torch.from_numpy(g).float(), DT)
pred_step_short = (~Pg["valid"]).any(1).numpy()          # a base step below min_ds
gt_step_short = (~Gg["valid"]).any(1).numpy()            # a GT step below min_ds
gt_pair_bad = (~Gg["pair_valid"]).any(1).numpy()
slow_v0 = v0 < 0.5                                       # MIN_DS_MPS
changed = np.abs(po - bo).reshape(len(eid), -1).max(1) > 0
n = int(env_b.sum())


def frac(mask):
    return {"n": int((env_b & mask).sum()), "of_violating": n,
            "share": round(float((env_b & mask).sum()) / max(n, 1), 4)}


out["base_envelope_violations"] = {
    "n": n, "rate": round(n / len(eid), 4),
    "with_a_base_step_below_min_ds": frac(pred_step_short),
    "with_a_gt_step_below_min_ds": frac(gt_step_short),
    "with_a_gt_pair_lacking_a_tangent": frac(gt_pair_bad),
    "with_v0_below_0.5_mps": frac(slow_v0),
    "with_none_of_those (a MOVING violation)": frac(~(pred_step_short | gt_step_short | slow_v0)),
    "changed_by_the_projection": frac(changed),
}
out["proj07_envelope_violations"] = int(env_p.sum())
out["min_ds_m"] = float(Gg["min_ds_m"])
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
