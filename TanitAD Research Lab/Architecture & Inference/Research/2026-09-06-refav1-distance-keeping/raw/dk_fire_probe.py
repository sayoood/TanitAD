#!/usr/bin/env python3
"""D-REFAV1-DK-COST P1 -- does the distance-keeping term FIRE, and does it point
where the human went?  ZERO-GPU: it re-prices the ALREADY-BANKED 21109 panel.

Three questions, in order of how badly a null on each would kill the term:

Q1  On how many of the panel's LEAD-bearing windows is the shipped plan actually
    IN VIOLATION of the desired gap?  A term that fires on nothing is inert, and
    saying so is the honest outcome.
Q2  On those windows, does the term RE-RANK the planner's own baseline set
    {cv/hold_v0 (a=0), decel_1.5} -- i.e. would the plan have changed?
Q3  DIRECTION CHECK, the one that is not guaranteed by construction: on the
    violating windows, what did the HUMAN do?  If the ground truth decelerates
    there, the term points the right way; if it does not, tau is mis-set for this
    corpus and the term would drive AWAY from the human.

Controls carried, because a panel without them is not readable:
  * the NON-violating windows are reported with their own n (the term must read
    exactly 0 there -- an identity, not an estimate);
  * a same-breath NON-ZERO control: the violating windows' shortfall must be > 0
    while the adequate windows' is == 0, from the same call.
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

import refav1_arm as ra                                        # noqa: E402
from tanitad.models.kinematic import unicycle_controls_from_path  # noqa: E402
from tanitad.refs import refav1_lon_cost as dk                 # noqa: E402

DUMP = sys.argv[1]
BLOCK = sys.argv[2]
OUT = sys.argv[3]
DT = 0.2
K = 10

files = sorted(f for f in (os.path.join(DUMP, x) for x in os.listdir(DUMP))
               if f.endswith(".npz") and os.path.basename(f).startswith("ep"))
manifest = json.load(open(os.path.join(DUMP, "manifest.json"), encoding="utf-8"))
blk, idx, meta = ra.load_lead_block_rows(BLOCK)
lead = ra.join_lead_block(files, manifest, blk, idx, k=K, dt=DT)
cov = lead.pop("coverage")

state = np.asarray(lead["state"]).astype(str)
gap0 = np.asarray(lead["gap0_m"], dtype=np.float64)
speeds = np.asarray(lead["speeds"], dtype=np.float64)
leads = np.asarray(lead["leads"], dtype=np.float64)
lens = np.asarray(lead["lead_lens"], dtype=np.float64)

# per-window arms + GT + the planner's OWN emitted controls
arms = {}
v0s, gts, eids, cl_ctrl = [], [], [], []
for fi, f in enumerate(files):
    with np.load(f) as d:
        for a in ("cl", "ha", "ha0", "ol"):
            arms.setdefault(a, []).append(np.asarray(d[a], dtype=np.float64))
        gts.append(np.asarray(d["g"], dtype=np.float64))
        v0s.append(np.asarray(d["v0"], dtype=np.float64).reshape(-1))
        eids.append(np.full(d["ws"].shape[0], fi, dtype=np.int64))
    dpath = os.path.join(DUMP, "decisions", os.path.basename(f))
    with np.load(dpath) as dd:
        cl_ctrl.append(np.asarray(dd["cl_controls"], dtype=np.float64))
arms = {k2: np.concatenate(v) for k2, v in arms.items()}
gt = np.concatenate(gts)
v0 = np.concatenate(v0s)
eid = np.concatenate(eids)
cl_controls = np.concatenate(cl_ctrl)
W = v0.size

# ---------------------------------------------------------------- Q1 ------- #
s_star = dk.DK_D0_M + dk.DK_TAU_TARGET_S * v0
is_lead = np.isfinite(gap0) & (state == "LEAD")
viol0 = is_lead & (gap0 < s_star)

# the ORACLE-gap arm's shortfall at t0 for the SHIPPED plan (the planner's own
# emitted controls), and for the two baselines iCEM actually compares
def cost_of(accel_seq, spec, only):
    """-> (cost [W], shortfall_m [W]) with NaN where the window has no lead."""
    c = np.full(W, np.nan)
    sh = np.full(W, np.nan)
    for i in np.flatnonzero(only):
        a = torch.tensor(accel_seq[i], dtype=torch.float64)[None]
        ctrl = torch.stack([a, torch.zeros_like(a)], dim=-1)
        c[i] = float(dk.distance_keeping_cost(ctrl, v0=float(v0[i]),
                                              gap0_m=float(gap0[i]), dt=DT,
                                              spec=spec))
        sh[i] = float(dk.gap_violation(a, v0=float(v0[i]), gap0_m=float(gap0[i]),
                                       dt=DT).max())
    return c, sh


spec1 = dk.DistanceKeepingSpec(w_dk=1.0, gap_source="oracle_label")
a_ship = cl_controls[..., 0]
a_zero = np.zeros((W, K))
a_dec = np.full((W, K), -1.5)

c_ship, sh_ship = cost_of(a_ship, spec1, is_lead)
c_zero, sh_zero = cost_of(a_zero, spec1, is_lead)
c_dec, sh_dec = cost_of(a_dec, spec1, is_lead)

# ---------------------------------------------------------------- Q2 ------- #
# the planner's baseline set is compared under the EXISTING terms; the shipped
# tie-break prefers hold_v0 (a = 0) whenever the costs tie. Adding the DK term
# flips a window iff  c_dec + existing_dec  <  c_zero + existing_zero.
# The existing terms on these two candidates are computable in closed form for
# the CONSTANT-acceleration baselines: jerk == 0 for both (constant a), kappa == 0
# for both, and w_vend was never armed.  => the DK term is the ONLY discriminator.
W_JERK, W_KAPPA = 0.02, 0.05
existing_zero = W_JERK * 0.0 + W_KAPPA * 0.0
existing_dec = W_JERK * 0.0 + W_KAPPA * 0.0
flip = np.zeros(W, dtype=bool)
for i in np.flatnonzero(is_lead):
    flip[i] = (c_dec[i] + existing_dec) < (c_zero[i] + existing_zero)

# ---------------------------------------------------------------- Q3 ------- #
# what the HUMAN did: recover the GT path's implied longitudinal acceleration
gt_ctrl = unicycle_controls_from_path(torch.tensor(gt), dt=DT).numpy()
gt_a_mean = gt_ctrl[..., 0].mean(axis=1)
# the arm's own implied accel, for the same window
ship_a_mean = a_ship.mean(axis=1)


def blk_stats(mask, name):
    n = int(mask.sum())
    if n == 0:
        return {"name": name, "n": 0, "status": "EMPTY"}
    return {
        "name": name, "n": n,
        "gap0_m": {"mean": round(float(np.mean(gap0[mask])), 4),
                   "min": round(float(np.min(gap0[mask])), 4),
                   "max": round(float(np.max(gap0[mask])), 4)},
        "v0_mps": {"mean": round(float(np.mean(v0[mask])), 4),
                   "max": round(float(np.max(v0[mask])), 4)},
        "s_star_m_mean": round(float(np.mean(s_star[mask])), 4),
        "shortfall_m": {"mean": round(float(np.nanmean(sh_zero[mask])), 4),
                        "max": round(float(np.nanmax(sh_zero[mask])), 4)},
        "gt_accel_mps2": {"mean": round(float(np.mean(gt_a_mean[mask])), 4),
                          "frac_decelerating": round(float(np.mean(
                              gt_a_mean[mask] < -0.1)), 4)},
        "shipped_accel_mps2": {"mean": round(float(np.mean(ship_a_mean[mask])), 4),
                               "frac_exactly_zero": round(float(np.mean(
                                   np.abs(a_ship[mask]).max(axis=1) == 0.0)), 4)},
        "n_episodes": int(np.unique(eid[mask]).size),
    }


adequate = is_lead & ~viol0
res = {
    "task": "D-REFAV1-DK-COST P1 -- does the distance-keeping term fire, and where does it point",
    "evidence_class": "MEASURED (ours) -- re-pricing of the banked refav1-21109-openloop dump",
    "tier": ("NOT a driving number. This re-prices a BANKED T1 panel's candidate set; "
             "no rollout was run and no model was loaded. It answers whether the term is "
             "LIVE on this corpus, not whether it drives better."),
    "gap_source": "ORACLE (b1 eval lead block label). A CEILING input, never a capability claim.",
    "estimator_note": ("counts and identities over the panel's own windows -- no bootstrap is "
                       "quoted here because nothing compared is a trained-arm difference; "
                       "the eval that WILL compare arms carries the paired episode-cluster CI."),
    "spec": spec1.record(),
    "panel": {"n_windows": W, "n_episodes": int(np.unique(eid).size),
              "window_states": {s: int((state == s).sum())
                                for s in sorted(set(state.tolist()))}},
    "Q1_fires": {
        "n_lead_windows": int(is_lead.sum()),
        "n_in_violation": int(viol0.sum()),
        "frac_of_lead_in_violation": round(float(viol0.sum() / max(is_lead.sum(), 1)), 4),
        "n_episodes_in_violation": int(np.unique(eid[viol0]).size) if viol0.any() else 0,
        "violating": blk_stats(viol0, "gap0 < d0 + tau*v0"),
        "adequate_CONTROL": blk_stats(adequate, "gap0 >= d0 + tau*v0"),
    },
    "Q1_controls": {
        "adequate_windows_cost_is_exactly_zero": bool(
            adequate.any() and np.all(c_zero[adequate] == 0.0)),
        "violating_windows_cost_is_strictly_positive": bool(
            viol0.any() and np.all(c_zero[viol0] > 0.0)),
        "unarmed_spec_is_exactly_zero": bool(np.all(
            cost_of(a_zero, dk.DistanceKeepingSpec(), is_lead)[0][is_lead] == 0.0)),
    },
    "Q2_rerank": {
        "n_flipped_to_decel": int(flip.sum()),
        "frac_of_violating": round(float(flip[viol0].sum() / max(viol0.sum(), 1)), 4),
        "note": ("On a CONSTANT-acceleration baseline both existing regularisers are "
                 "identically zero (mean(jerk^2) == 0 for any constant a; kappa == 0 "
                 "for both baselines) and w_vend was never armed -- so on this "
                 "comparison the DK term is the ONLY discriminator and any flip is "
                 "attributable to it alone."),
        "cost_zero_mean_on_violating": (round(float(np.nanmean(c_zero[viol0])), 6)
                                        if viol0.any() else None),
        "cost_decel_mean_on_violating": (round(float(np.nanmean(c_dec[viol0])), 6)
                                         if viol0.any() else None),
    },
    "Q3_direction": {
        "_what": ("the check that is NOT true by construction: does the human decelerate "
                  "on the windows the term penalises?"),
        "violating_gt_accel_mean": (round(float(np.mean(gt_a_mean[viol0])), 4)
                                    if viol0.any() else None),
        "adequate_gt_accel_mean": (round(float(np.mean(gt_a_mean[adequate])), 4)
                                   if adequate.any() else None),
        "violating_gt_frac_decel": (round(float(np.mean(gt_a_mean[viol0] < -0.1)), 4)
                                    if viol0.any() else None),
        "adequate_gt_frac_decel": (round(float(np.mean(gt_a_mean[adequate] < -0.1)), 4)
                                   if adequate.any() else None),
    },
    "weight_calibration": {
        "_what": ("w_dk that makes the DK term's mean on the VIOLATING windows equal "
                  "W_KAPPA's own scale on this planner's candidate box, so the two are "
                  "commensurable rather than guessed."),
        "mean_shortfall_sq_m2_on_violating": (
            round(float(np.nanmean(c_zero[viol0])), 6) if viol0.any() else None),
    },
    "provenance": {"dump": DUMP, "lead_block": BLOCK,
                   "lead_block_meta": {k2: meta.get(k2) for k2 in
                                       ("built", "dt_s", "k", "n_clips")
                                       if isinstance(meta, dict)},
                   "coverage_status": {k2: v for k2, v in cov.items()
                                       if not isinstance(v, (dict, list))}},
}

# calibrate w_dk against the existing weights on a REAL candidate spread
if viol0.any():
    kap = 0.05
    # W_KAPPA's typical contribution at this planner's kappa_max (0.2 1/m clip is
    # far outside what it emits; use the measured max |kappa| from the panel)
    kmax = float(np.abs(cl_controls[..., 1]).max())
    res["weight_calibration"]["measured_max_abs_kappa_in_panel"] = round(kmax, 6)
    res["weight_calibration"]["w_kappa_cost_at_that_kappa"] = round(kap * kmax ** 2, 8)
    m = float(np.nanmean(c_zero[viol0]))
    res["weight_calibration"]["w_dk_for_parity_with_w_kappa"] = (
        round(kap * kmax ** 2 / m, 8) if m > 0 else None)

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps({k2: res[k2] for k2 in
                  ("panel", "Q1_fires", "Q1_controls", "Q2_rerank", "Q3_direction")},
                 indent=1))
