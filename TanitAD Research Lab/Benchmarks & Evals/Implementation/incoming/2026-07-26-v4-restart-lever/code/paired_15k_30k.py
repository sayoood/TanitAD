"""Paired episode-cluster bootstrap on the v4 WITHIN-RUN regression 15k -> 30k.

The 30k gate report states the 15k->30k move was "Not tested for separation".
This supplies that test. Estimator: paired_episode_cluster_bootstrap
(taniteval/ci.py) -- NEVER overlapping_holdout_se. Unit of resampling = episode.

Harness check first: both dumps must reproduce their PUBLISHED point estimates
(15k 0.5839 / 30k 0.6423) before any delta is quoted.
"""
import json, sys
import numpy as np, torch

sys.path.insert(0, "/root/taniteval")
from taniteval.ci import (paired_episode_cluster_bootstrap,
                          episode_cluster_bootstrap)

P15 = "/root/v4eval/results/windows_flagship-v4-fromscratch-15k.pt"
P15D = "/root/v4eval/results_goalmode/windows_v4-15k-oracle-dense.pt"
P30 = "/workspace/_v4gate/results/windows_flagship-v4-fromscratch-30k-oracle.pt"
P30P = "/workspace/_v4gate/results/windows_flagship-v4-fromscratch-30k-produced.pt"
P15P = "/root/v4eval/results_goalmode/windows_v4-15k-goal-produced.pt"

L = lambda p: torch.load(p, map_location="cpu", weights_only=False)
d15, d15d, d30, d30p, d15p = L(P15), L(P15D), L(P30), L(P30P), L(P15P)

res = {"_estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py)",
       "_never": "overlapping_holdout_se",
       "_resampling_unit": "episode cluster",
       "_evidence_class": "MEASURED (ours)",
       "_goal_provenance": "BOTH arms goal-ORACLE (route/route_graded/vt_band from ego's own "
                           "future poses). NOT a deployed-capability surface.",
       "_sources": {"15k": P15, "15k_dense": P15D, "30k": P30,
                    "30k_produced": P30P, "15k_produced": P15P}}


def ade4(d):
    """ade_0_2s on the v1-comparable 4-waypoint surface: per-window mean L2."""
    return (d["pred"] - d["gt"]).norm(dim=-1).mean(1).numpy()


def miss2m(d):
    """miss@2m: final-waypoint (2.0 s) L2 error > 2 m."""
    return ((d["pred"] - d["gt"]).norm(dim=-1)[:, -1] > 2.0).float().numpy()


def alongcross(d, dense=True):
    """ego frame: axis0 = along (longitudinal), axis1 = cross (lateral)."""
    k = "pred_dense" if dense and "pred_dense" in d else "pred"
    g = "gt_dense" if dense and "gt_dense" in d else "gt"
    r = d[k] - d[g]
    return r[..., 0].abs().mean(1).numpy(), r[..., 1].abs().mean(1).numpy()


# ---------- window alignment ------------------------------------------------
e15, e30, e15d = [str(x) for x in d15["eid"]], [str(x) for x in d30["eid"]], [str(x) for x in d15d["eid"]]
res["window_alignment"] = {
    "n_15k": len(e15), "n_30k": len(e30), "n_15k_dense": len(e15d),
    "eid_sequence_identical_15k_vs_30k": e15 == e30,
    "eid_sequence_identical_15kdense_vs_30k": e15d == e30,
    "gt_identical_15k_vs_30k": bool(torch.allclose(d15["gt"], d30["gt"], atol=1e-6)),
    "gt_identical_15kdense_vs_30k": bool(torch.allclose(d15d["gt"], d30["gt"], atol=1e-6)),
    "cv_identical_15k_vs_30k": bool(torch.allclose(d15["cv"], d30["cv"], atol=1e-6)),
    "n_episodes": len(set(e30)),
    "_why": "a paired bootstrap is only valid on the SAME windows in the SAME order",
}

# ---------- harness check: reproduce the published point estimates ----------
a15, a30 = ade4(d15), ade4(d30)
res["harness_check"] = {
    "ade_0_2s_15k_recomputed": round(float(a15.mean()), 4),
    "ade_0_2s_15k_published": 0.5839,
    "ade_0_2s_30k_recomputed": round(float(a30.mean()), 4),
    "ade_0_2s_30k_published": 0.6423,
    "ade_0_2s_15kdense_recomputed": round(float(ade4(d15d).mean()), 4),
    "_rule": "no delta below is quotable unless both reproduce to <=0.001",
}
res["harness_check"]["PASS"] = (
    abs(res["harness_check"]["ade_0_2s_15k_recomputed"] - 0.5839) <= 0.001 and
    abs(res["harness_check"]["ade_0_2s_30k_recomputed"] - 0.6423) <= 0.001)

# ---------- THE regression interval ----------------------------------------
# oriented 30k - 15k, so POSITIVE = 30k is WORSE (a real regression)
res["paired_regression_30k_minus_15k"] = {}
R = res["paired_regression_30k_minus_15k"]
R["ade_0_2s_4wp"] = paired_episode_cluster_bootstrap(a30, a15, e30, n_boot=2000, seed=0)
R["miss_at_2m"] = paired_episode_cluster_bootstrap(miss2m(d30), miss2m(d15), e30, n_boot=2000, seed=0)

al30, cr30 = alongcross(d30)
al15, cr15 = alongcross(d15d)          # 15k dense dump (same windows, verified above)
R["along_abs_dense_LONGITUDINAL"] = paired_episode_cluster_bootstrap(al30, al15, e30, n_boot=2000, seed=0)
R["cross_abs_dense_LATERAL"] = paired_episode_cluster_bootstrap(cr30, cr15, e30, n_boot=2000, seed=0)

# per-arm singles for context
res["singles"] = {
    "ade_0_2s_15k": episode_cluster_bootstrap(a15, e15, n_boot=2000, seed=0),
    "ade_0_2s_30k": episode_cluster_bootstrap(a30, e30, n_boot=2000, seed=0),
    "along_abs_dense_15k": episode_cluster_bootstrap(al15, e15d, n_boot=2000, seed=0),
    "along_abs_dense_30k": episode_cluster_bootstrap(al30, e30, n_boot=2000, seed=0),
    "cross_abs_dense_15k": episode_cluster_bootstrap(cr15, e15d, n_boot=2000, seed=0),
    "cross_abs_dense_30k": episode_cluster_bootstrap(cr30, e30, n_boot=2000, seed=0),
}

# energy share of the DELTA: where did the extra error land?
res["regression_axis_split"] = {
    "delta_along_m": round(float(al30.mean() - al15.mean()), 4),
    "delta_cross_m": round(float(cr30.mean() - cr15.mean()), 4),
    "energy_share_30k": {
        "longitudinal": round(float((( d30["pred_dense"]-d30["gt_dense"])[...,0]**2).sum() /
                                    (( d30["pred_dense"]-d30["gt_dense"])**2).sum()), 4),
        "lateral": round(float(((d30["pred_dense"]-d30["gt_dense"])[...,1]**2).sum() /
                               ((d30["pred_dense"]-d30["gt_dense"])**2).sum()), 4)},
    "energy_share_15k": {
        "longitudinal": round(float(((d15d["pred_dense"]-d15d["gt_dense"])[...,0]**2).sum() /
                                    ((d15d["pred_dense"]-d15d["gt_dense"])**2).sum()), 4),
        "lateral": round(float(((d15d["pred_dense"]-d15d["gt_dense"])[...,1]**2).sum() /
                               ((d15d["pred_dense"]-d15d["gt_dense"])**2).sum()), 4)},
}

# ---------- goal-mode control: is the regression oracle-specific? -----------
# C6 guard: if the regression only exists on the oracle surface it is a goal-head
# artefact, not a selector fact. Test it on the PRODUCED surface too.
a15p, a30p = ade4(d15p), ade4(d30p)
e15p, e30p = [str(x) for x in d15p["eid"]], [str(x) for x in d30p["eid"]]
res["goalmode_control_produced"] = {
    "eid_aligned": e15p == e30p,
    "ade_0_2s_15k_produced": round(float(a15p.mean()), 4),
    "ade_0_2s_30k_produced": round(float(a30p.mean()), 4),
    "paired_30k_minus_15k_PRODUCED":
        paired_episode_cluster_bootstrap(a30p, a15p, e30p, n_boot=2000, seed=0),
    "paired_miss_at_2m_PRODUCED":
        paired_episode_cluster_bootstrap(miss2m(d30p), miss2m(d15p), e30p, n_boot=2000, seed=0),
}

print(json.dumps(res, indent=2))
with open("/root/v4_paired_15k_30k.json", "w") as f:
    json.dump(res, f, indent=2)
