"""flagship-v4 30k gate — paired deltas, lat/lon decomposition, card adjudication.

Everything here is MEASURED on the eval pod from the two MODE-B window dumps and
v1's window dump. Estimator is NAMED on every interval; the paired reads use
`paired_episode_cluster_bootstrap`. `overlapping_holdout_se` is never used.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from taniteval import ci as CI  # noqa: E402
from taniteval import lateral as LAT  # noqa: E402
from taniteval import rollout  # noqa: E402

RES = "/workspace/_v4gate/results"
OUT = {}
N_BOOT = 2000


def W(key):
    return rollout.load_windows(f"{RES}/windows_{key}.pt")


def np3(x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x


wo = W("flagship-v4-fromscratch-30k-oracle")
wp = W("flagship-v4-fromscratch-30k-produced")
wv1 = rollout.load_windows("/root/taniteval/results/windows_flagship-30k.pt")

OUT["window_dumps"] = {
    "oracle": {k: (list(np3(v).shape) if hasattr(v, "shape") else str(v)[:80])
               for k, v in wo.items()},
    "produced": {k: (list(np3(v).shape) if hasattr(v, "shape") else str(v)[:80])
                 for k, v in wp.items()},
    "v1_flagship-30k": {k: (list(np3(v).shape) if hasattr(v, "shape") else str(v)[:80])
                        for k, v in wv1.items()},
}


def ade_per_window(win):
    """4-waypoint ADE@2s per window — the ONLY convention MODEL_REGISTRY quotes."""
    p, g = np3(win["pred"]), np3(win["gt"])
    return np.linalg.norm(p - g, axis=-1).mean(axis=1)


ao, ap_, av1 = ade_per_window(wo), ade_per_window(wp), ade_per_window(wv1)
eo, ep_, ev1 = np3(wo["eid"]), np3(wp["eid"]), np3(wv1["eid"])

OUT["per_window_means"] = {
    "oracle_ade_0_2s": float(ao.mean()), "n": int(ao.size),
    "produced_ade_0_2s": float(ap_.mean()), "n_produced": int(ap_.size),
    "v1_ade_0_2s": float(av1.mean()), "n_v1": int(av1.size),
}

# ---------------------------------------------------------------- PAIRED --- #
# ORACLE vs PRODUCED on the SAME windows -- the goal-privilege size AT 30k.
# The card FORBIDS transplanting the 15k gap (+0.1738): the goal head trained on.
same_op = (ao.size == ap_.size) and bool((eo == ep_).all())
OUT["paired_oracle_vs_produced"] = {"same_windows": bool(same_op)}
if same_op:
    d = CI.paired_episode_cluster_bootstrap(ao, ap_, eo, n_boot=N_BOOT, seed=0)
    d["_orientation"] = ("b - a = produced - oracle; POSITIVE = the goal ORACLE "
                         "was helping, i.e. the deployable path is WORSE")
    d["delta_point"] = float(ap_.mean() - ao.mean())
    d["estimator"] = "paired_episode_cluster_bootstrap"
    d["n_windows"], d["n_episodes"] = int(ao.size), int(np.unique(eo).size)
    d["measured_at_step"] = 29999
    d["card_note_15k_gap_NOT_transplanted"] = 0.1738
    OUT["paired_oracle_vs_produced"].update(d)

# v4 ORACLE vs v1 on the SAME windows -- the reference comparison (0.4271).
same_v1 = (ao.size == av1.size) and bool((eo == ev1).all())
OUT["paired_v4oracle_vs_v1"] = {"same_windows": bool(same_v1)}
if same_v1:
    d = CI.paired_episode_cluster_bootstrap(av1, ao, eo, n_boot=N_BOOT, seed=0)
    d["_orientation"] = ("b - a = v4_oracle - v1; POSITIVE = v4 is BEHIND v1")
    d["delta_point"] = float(ao.mean() - av1.mean())
    d["estimator"] = "paired_episode_cluster_bootstrap"
    d["n_windows"], d["n_episodes"] = int(ao.size), int(np.unique(eo).size)
    OUT["paired_v4oracle_vs_v1"].update(d)

same_v1p = (ap_.size == av1.size) and bool((ep_ == ev1).all())
OUT["paired_v4produced_vs_v1"] = {"same_windows": bool(same_v1p)}
if same_v1p:
    d = CI.paired_episode_cluster_bootstrap(av1, ap_, ep_, n_boot=N_BOOT, seed=0)
    d["_orientation"] = "b - a = v4_produced - v1; POSITIVE = v4 is BEHIND v1"
    d["delta_point"] = float(ap_.mean() - av1.mean())
    d["estimator"] = "paired_episode_cluster_bootstrap"
    d["n_windows"], d["n_episodes"] = int(ap_.size), int(np.unique(ep_).size)
    OUT["paired_v4produced_vs_v1"].update(d)

# ------------------------------------------------- LAT / LON DECOMPOSITION - #
# Card required_reporting[0]: cross-track is the safety-relevant axis and an
# undecomposed L2 hides it. SPARSE 4-wp surface -> horizon_s MUST read 2.0.
OUT["lateral_longitudinal"] = {}
for name, win in (("oracle", wo), ("produced", wp), ("v1", wv1)):
    try:
        w = {"pred": np3(win["pred"]), "gt": np3(win["gt"]),
             "eid": np3(win["eid"]), "wp_steps": [5, 10, 15, 20]}
        if win.get("speed") is not None:
            w["speed"] = np3(win["speed"])
        r = LAT.from_sparse_windows(w, mode="ego", n_boot=N_BOOT, seed=0)
        OUT["lateral_longitudinal"][name] = {
            "surface": r.get("surface"), "dt_s": r.get("dt_s"),
            "horizon_K": r.get("horizon_K"), "horizon_s": r.get("horizon_s"),
            "energy_share": r.get("energy_share"),
            "dense_aggregate": r.get("dense_aggregate"),
            "growth": r.get("growth"),
            "by_horizon": r.get("by_horizon"),
            "axis_check": r.get("axis_check"),
            "verdict": LAT._verdict(r),
            "estimator": "episode_cluster_bootstrap",
        }
    except Exception as e:  # noqa: BLE001
        import traceback
        OUT["lateral_longitudinal"][name] = {
            "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}

# PAIRED cross-track: is v4's lateral error separated from v1's?
try:
    if same_v1:
        pc = LAT.paired_cross_track(np3(wv1["pred"]), np3(wo["pred"]), np3(wo["gt"]),
                                    eo, step=4, mode="ego", n_boot=N_BOOT, seed=0,
                                    knot_dt=0.5)
        OUT["paired_cross_track_v1_vs_v4oracle"] = pc
except Exception as e:  # noqa: BLE001
    OUT["paired_cross_track_v1_vs_v4oracle"] = {"error": str(e)}

print(json.dumps(OUT, indent=1, default=str)[:14000])
with open("/workspace/_v4gate/gate30k_analysis.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nWROTE /workspace/_v4gate/gate30k_analysis.json")
