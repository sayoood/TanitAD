#!/usr/bin/env python3
"""D-REFAV1-DK-COST P2 -- pick tau FROM THE CORPUS, not from convention, and put
an estimator on the direction check.

⛔ WHY P1 IS NOT ENOUGH. P1 measured that tau = 1.5 s fires on 21/90 lead windows
and that the human's mean acceleration there is negative. Both are true and
neither answers the question that decides the weight: **is 1.5 s ABOVE or BELOW
what this corpus's own drivers run?** A target above the human's own behaviour
makes the term fight the reference it is scored against -- it would buy headway
and pay ADE, and that trade must be pre-registered, not discovered afterwards.

Three things, none of which needs a GPU:
  1. the corpus's own time-gap distribution at t0 (gap0 / v0), per speed band,
     with the n for every band -- the four-families clause-5 shape;
  2. a tau SWEEP: what fraction of lead windows each tau makes violating, so the
     weight class is chosen against a measured curve;
  3. the DIRECTION check with an estimator: the episode-cluster bootstrap on
     mean(GT accel | violating) - mean(GT accel | adequate), which is the only
     honest form of "the human decelerates where the term fires".

⚠️ WHICH VARIANCE THIS INTERVAL ANSWERS. The episode-cluster bootstrap here
resamples EPISODES with the labels and the banked GT held fixed. It answers
"would another draw of episodes say this?" -- NOT another training run
(H-ESTIM-SEED-1) and NOT another inference run. Nothing here is a trained-arm
difference, so those two do not arise; the GT accel and the label gap are both
deterministic given the episode.

Same-breath controls: a permutation of the violating/adequate assignment WITHIN
episodes must read ~0, and the adequate block's shortfall must read exactly 0.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

WT = r"C:/Users/Admin/tanitad-wt"
sys.path.insert(0, os.path.join(WT, "stack"))
sys.path.insert(0, os.path.join(WT, "taniteval"))
sys.path.insert(0, os.path.join(WT, "taniteval", "tools"))

import refav1_arm as ra                                          # noqa: E402
from taniteval import lead_metrics as lm                         # noqa: E402
from tanitad.models.kinematic import unicycle_controls_from_path  # noqa: E402
from tanitad.refs import refav1_lon_cost as dk                   # noqa: E402

DUMP, BLOCK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DT, K = 0.2, 10
RNG = np.random.default_rng(0)
N_BOOT = 2000

files = sorted(f for f in (os.path.join(DUMP, x) for x in os.listdir(DUMP))
               if f.endswith(".npz") and os.path.basename(f).startswith("ep"))
manifest = json.load(open(os.path.join(DUMP, "manifest.json"), encoding="utf-8"))
blk, idx, meta = ra.load_lead_block_rows(BLOCK)
lead = ra.join_lead_block(files, manifest, blk, idx, k=K, dt=DT)
lead.pop("coverage")
state = np.asarray(lead["state"]).astype(str)
gap0 = np.asarray(lead["gap0_m"], dtype=np.float64)
leads = np.asarray(lead["leads"], dtype=np.float64)
lens = np.asarray(lead["lead_lens"], dtype=np.float64)

gts, v0s, eids, cls = [], [], [], []
for fi, f in enumerate(files):
    with np.load(f) as d:
        gts.append(np.asarray(d["g"], dtype=np.float64))
        cls.append(np.asarray(d["cl"], dtype=np.float64))
        v0s.append(np.asarray(d["v0"], dtype=np.float64).reshape(-1))
        eids.append(np.full(d["ws"].shape[0], fi, dtype=np.int64))
gt = np.concatenate(gts)
cl = np.concatenate(cls)
v0 = np.concatenate(v0s)
eid = np.concatenate(eids)
W = v0.size

is_lead = np.isfinite(gap0) & (state == "LEAD")
gt_a = unicycle_controls_from_path(torch.tensor(gt), dt=DT).numpy()[..., 0].mean(1)

# --------------------------------------------------------------------------- #
# 0. reconcile the two DIFFERENT "n" this family carries -- they are not the   #
#    same quantity and quoting one for the other is exactly the scope error    #
#    the programme keeps retracting.                                           #
# --------------------------------------------------------------------------- #
dk_cl = lm.distance_keeping(cl, leads, lens, v0, DT)
recon = {
    "n_LEAD_state_windows": int(is_lead.sum()),
    "_what_that_is": ("windows whose CAUSAL lead selection at t0 found an in-corridor "
                      "vehicle -- the population the COST sees, because the cost is "
                      "evaluated at t0 before any path exists"),
    "n_distance_keeping_cl": int(dk_cl["n"]),
    "_what_that_is": ("windows where the ARM's predicted path still has the lead inside "
                      "the |lat| < 2 m corridor for >= 1 horizon step -- the population "
                      "the METRIC scores, which depends on the arm"),
    "published_n_in_RESULT_refav1_21109": 87,
    "note": ("these differ by construction: an arm that steers away loses its lead and "
             "drops out of the metric while the cost still saw one at t0. Quote the "
             "cost's n for the cost and the metric's n for the metric."),
}

# --------------------------------------------------------------------------- #
# 1. the corpus's OWN time gap at t0, per speed band (clause-5 shape)          #
# --------------------------------------------------------------------------- #
tg0 = np.full(W, np.nan)
ok = is_lead & (v0 >= lm.MIN_SPEED_MPS)
tg0[ok] = gap0[ok] / v0[ok]
bands = [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0), (15.0, 1e9)]
by_band = {}
for lo, hi in bands:
    name = f"{lo:g}-{'15+' if hi > 1e8 else f'{hi:g}'}".replace("-15+", "+")
    name = f"{lo:g}+" if hi > 1e8 else f"{lo:g}-{hi:g}"
    m = is_lead & (v0 >= lo) & (v0 < hi)
    mt = m & np.isfinite(tg0)
    if int(m.sum()) == 0:
        by_band[name] = {"n_lead": 0, "status": "EMPTY",
                         "reason": "no LEAD-state window in this band"}
        continue
    by_band[name] = {
        "n_lead": int(m.sum()), "n_episodes": int(np.unique(eid[m]).size),
        "n_with_time_gap": int(mt.sum()),
        "gap0_m_median": round(float(np.median(gap0[m])), 4),
        "time_gap_s": ({"median": round(float(np.median(tg0[mt])), 4),
                        "p25": round(float(np.percentile(tg0[mt], 25)), 4),
                        "p10": round(float(np.percentile(tg0[mt], 10)), 4)}
                       if mt.any() else
                       {"status": "NOT-APPLICABLE",
                        "reason": f"every window in this band is below "
                                  f"MIN_SPEED_MPS={lm.MIN_SPEED_MPS}; a time gap at "
                                  f"a standstill is not a following distance"}),
    }

allm = is_lead & np.isfinite(tg0)
corpus_tg = {
    "n": int(allm.sum()), "n_episodes": int(np.unique(eid[allm]).size),
    "median_s": round(float(np.median(tg0[allm])), 4),
    "p25_s": round(float(np.percentile(tg0[allm], 25)), 4),
    "p10_s": round(float(np.percentile(tg0[allm], 10)), 4),
    "p50_at_speed_ge_10": (round(float(np.median(tg0[allm & (v0 >= 10.0)])), 4)
                           if (allm & (v0 >= 10.0)).any() else None),
    "n_at_speed_ge_10": int((allm & (v0 >= 10.0)).sum()),
}

# --------------------------------------------------------------------------- #
# 2. the tau sweep                                                            #
# --------------------------------------------------------------------------- #
sweep = []
for tau in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0):
    s_star = dk.DK_D0_M + tau * v0
    v = is_lead & (gap0 < s_star)
    sweep.append({
        "tau_s": tau, "n_violating": int(v.sum()),
        "frac_of_lead": round(float(v.sum() / max(is_lead.sum(), 1)), 4),
        "n_episodes": int(np.unique(eid[v]).size) if v.any() else 0,
        "mean_shortfall_m": (round(float(np.mean(s_star[v] - gap0[v])), 4)
                             if v.any() else 0.0),
        "gt_accel_mean_on_violating": (round(float(np.mean(gt_a[v])), 4)
                                       if v.any() else None),
    })

# --------------------------------------------------------------------------- #
# 3. the direction check, with an estimator                                   #
# --------------------------------------------------------------------------- #
def cluster_boot(mask_a, mask_b, vals, clusters, n_boot=N_BOOT, seed=0):
    """Episode-cluster bootstrap on mean(vals|a) - mean(vals|b). Resamples
    EPISODES with replacement; a draw missing either group is skipped and
    counted, never silently replaced."""
    rng = np.random.default_rng(seed)
    eps = np.unique(clusters)
    point = float(np.mean(vals[mask_a]) - np.mean(vals[mask_b]))
    out, skipped = [], 0
    for _ in range(n_boot):
        pick = rng.choice(eps, size=eps.size, replace=True)
        sel = np.concatenate([np.flatnonzero(clusters == e) for e in pick])
        a = mask_a[sel]
        b = mask_b[sel]
        if not a.any() or not b.any():
            skipped += 1
            continue
        out.append(float(np.mean(vals[sel][a]) - np.mean(vals[sel][b])))
    o = np.sort(np.asarray(out))
    return {
        "point": round(point, 5),
        "ci95": [round(float(np.percentile(o, 2.5)), 5),
                 round(float(np.percentile(o, 97.5)), 5)] if o.size else None,
        "n_boot_used": int(o.size), "n_boot_skipped": int(skipped),
        "excludes_zero": bool(o.size and (np.percentile(o, 2.5) > 0
                                          or np.percentile(o, 97.5) < 0)),
        "estimator": ("episode-cluster bootstrap over the panel's episodes; answers "
                      "'would another draw of EPISODES say this?' -- not another "
                      "training run and not another inference run (neither applies: "
                      "the GT accel and the label gap are deterministic per episode)"),
    }


direction = {}
for tau in (1.0, 1.2, 1.5):
    s_star = dk.DK_D0_M + tau * v0
    v = is_lead & (gap0 < s_star)
    a = is_lead & ~v
    if not (v.any() and a.any()):
        direction[f"tau_{tau}"] = {"status": "EMPTY"}
        continue
    d = cluster_boot(v, a, gt_a, eid, seed=7)
    # SAME-BREATH CONTROL: shuffle the violating/adequate label WITHIN the lead
    # set, preserving the counts. Must read ~0 -- if it does not, the estimator
    # is reading episode identity rather than the gap.
    rng = np.random.default_rng(11)
    perm = []
    for _ in range(200):
        p = rng.permutation(np.flatnonzero(is_lead))
        vv = np.zeros(W, dtype=bool)
        vv[p[: int(v.sum())]] = True
        aa = is_lead & ~vv
        perm.append(float(np.mean(gt_a[vv]) - np.mean(gt_a[aa])))
    direction[f"tau_{tau}"] = {
        "n_violating": int(v.sum()), "n_adequate": int(a.sum()),
        "delta_gt_accel_mps2": d,
        "SHUFFLE_CONTROL": {"mean": round(float(np.mean(perm)), 5),
                            "p2.5": round(float(np.percentile(perm, 2.5)), 5),
                            "p97.5": round(float(np.percentile(perm, 97.5)), 5),
                            "_what": ("violating/adequate labels permuted across the "
                                      "lead set with the counts preserved; must "
                                      "straddle 0")},
    }

res = {
    "task": "D-REFAV1-DK-COST P2 -- tau from the corpus + an estimator on the direction check",
    "evidence_class": "MEASURED (ours) -- banked refav1-21109-openloop dump + B1 eval lead block",
    "tier": ("NOT a driving number. Re-pricing and label statistics over a BANKED panel; "
             "no rollout, no model load."),
    "n_reconciliation": recon,
    "corpus_time_gap_at_t0": corpus_tg,
    "corpus_time_gap_by_speed_band": by_band,
    "tau_sweep": sweep,
    "direction_check": direction,
    "defaults_in_code": {"DK_TAU_TARGET_S": dk.DK_TAU_TARGET_S, "DK_D0_M": dk.DK_D0_M,
                         "DK_W": dk.DK_W},
    "provenance": {"dump": DUMP, "lead_block": BLOCK},
}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps({k: res[k] for k in ("n_reconciliation", "corpus_time_gap_at_t0",
                                      "corpus_time_gap_by_speed_band", "tau_sweep",
                                      "direction_check")}, indent=1))
