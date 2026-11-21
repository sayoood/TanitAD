"""Pair v4-30k against v1 on the SAME windows.

The two dumps encode `eid` differently (v1: clean 0..39 ints; v4: a different
encoding), so an `eid`-equality test says "different windows" when the windows
are in fact identical. The DECISIVE test is the ground truth itself: if `gt` is
elementwise identical, the two dumps are the same 881 windows in the same order
and a PAIRED read is admissible. If it is not, the paired read is REFUSED --
pairing misaligned windows would be worse than not pairing at all.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, "/root/taniteval")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from taniteval import ci as CI  # noqa: E402
from taniteval import lateral as LAT  # noqa: E402
from taniteval import rollout  # noqa: E402

RES = "/workspace/_v4gate/results"
N_BOOT = 2000


def np_(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


wo = rollout.load_windows(f"{RES}/windows_flagship-v4-fromscratch-30k-oracle.pt")
wp = rollout.load_windows(f"{RES}/windows_flagship-v4-fromscratch-30k-produced.pt")
wv1 = rollout.load_windows("/root/taniteval/results/windows_flagship-30k.pt")

go, gp, gv1 = np_(wo["gt"]), np_(wp["gt"]), np_(wv1["gt"])
out = {"window_alignment": {
    "gt_oracle_vs_v1_max_abs_diff": float(np.abs(go - gv1).max()),
    "gt_oracle_vs_produced_max_abs_diff": float(np.abs(go - gp).max()),
    "shapes": {"oracle": list(go.shape), "produced": list(gp.shape),
               "v1": list(gv1.shape)},
    "eid_encoding_note": ("v1 dump encodes eid as 0..39; the v4 dumps use a "
                          "different encoding. Alignment is therefore proven "
                          "from the GROUND TRUTH, not from eid equality."),
}}
ALIGNED = bool(np.abs(go - gv1).max() < 1e-5)
out["window_alignment"]["v4_and_v1_are_the_SAME_windows"] = ALIGNED

eid = np_(wv1["eid"])                     # clean 0..39 clustering
out["window_alignment"]["n_episodes_from_v1_eid"] = int(np.unique(eid).size)


def ade(w):
    return np.linalg.norm(np_(w["pred"]) - np_(w["gt"]), axis=-1).mean(axis=1)


ao, ap_, av1 = ade(wo), ade(wp), ade(wv1)
out["point_estimates_ade_0_2s_4waypoint"] = {
    "v1": float(av1.mean()), "v4_oracle": float(ao.mean()),
    "v4_produced": float(ap_.mean()),
    "v1_registry_reference": 0.4271,
    "v1_reproduced_exactly": bool(abs(av1.mean() - 0.4271) < 5e-5)}

if ALIGNED:
    for nm, arm in (("v4oracle_vs_v1", ao), ("v4produced_vs_v1", ap_)):
        d = CI.paired_episode_cluster_bootstrap(arm, av1, eid, n_boot=N_BOOT, seed=0)
        d["_orientation"] = (f"delta = {nm.split('_vs_')[0]} - v1; "
                             f"POSITIVE = v4 is BEHIND v1 (worse ADE)")
        d["estimator"] = "paired_episode_cluster_bootstrap"
        out[f"paired_{nm}"] = d
    # PAIRED CROSS-TRACK (the safety-relevant axis), sparse 4-knot surface
    pc = LAT.paired_cross_track(np_(wv1["pred"]), np_(wo["pred"]), go, eid,
                                step=4, mode="ego", n_boot=N_BOOT, seed=0,
                                knot_dt=0.5)
    out["paired_cross_track_v1_vs_v4oracle"] = pc
else:
    out["paired_REFUSED"] = ("gt arrays differ -> the dumps are NOT the same "
                             "windows; a paired read is inadmissible.")

# oracle vs produced, correctly oriented this time (ci.py computes a - b)
d = CI.paired_episode_cluster_bootstrap(ap_, ao, np_(wo["eid"]), n_boot=N_BOOT, seed=0)
d["_orientation"] = ("delta = produced - oracle; POSITIVE = the goal ORACLE was "
                     "helping, i.e. the DEPLOYABLE path is worse by this much")
d["estimator"] = "paired_episode_cluster_bootstrap"
d["card_15k_gap_NOT_transplanted"] = 0.1738
d["measured_at_step"] = 29999
out["paired_produced_minus_oracle_AT_30K"] = d

print(json.dumps(out, indent=1, default=str)[:9000])
with open("/workspace/_v4gate/gate30k_paired.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nWROTE /workspace/_v4gate/gate30k_paired.json")
