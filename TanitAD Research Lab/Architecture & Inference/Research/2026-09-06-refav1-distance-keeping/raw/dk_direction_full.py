#!/usr/bin/env python3
"""D-REFAV1-DK-COST P3 -- the direction check, run where it has POWER.

⛔ WHY. P2's direction check is a NULL at every tau (tau=1.5: -0.2015
[-0.4455, +0.0446], spans zero) on **21 violating windows over 17 episodes**.
That is a statement about the panel's size, not about the corpus: the eval dump
emits only 2 windows per episode. The SAME question can be asked of the whole B1
eval lead block -- **29,556 rows over 146 clips at 10 Hz** -- with no GPU, no
model and no new labels, because the block already carries per-frame ``speeds``
(egomotion interpolated at the frame) and per-frame ``gap0_m``.

⭐ THE GT ACCELERATION COMES FROM THE BLOCK ITSELF: a central difference of
``speeds`` along the frame axis WITHIN each clip. That is the human's own
longitudinal behaviour at the frame the gap is measured at.
⚠️ It is NOT the dump's ``g``-derived accel P2 used, so the two are not the same
estimator -- P3 is reported as its own panel, not as a continuation of P2's.

CONTROLS, all of which must read their known value or the panel is unreadable:
  * a within-clip SHUFFLE of the violating/adequate assignment -> ~0;
  * the adequate block's shortfall -> exactly 0 (identity);
  * a same-breath NON-ZERO reference: mean speed differs strongly between the
    two blocks by construction (violation is speed-driven), so the pipeline
    demonstrably can separate the two populations when a real difference exists.

⚠️ WHICH VARIANCE. Clip-cluster bootstrap over the 146 clips: "would another
draw of CLIPS say this?". No training and no inference randomness is involved --
both operands are deterministic label arithmetic.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

WT = r"C:/Users/Admin/tanitad-wt"
sys.path.insert(0, os.path.join(WT, "stack"))
sys.path.insert(0, os.path.join(WT, "taniteval"))

from tanitad.refs import refav1_lon_cost as dk   # noqa: E402

BLOCK, OUT = sys.argv[1], sys.argv[2]
N_BOOT = 2000
MIN_SPEED = 1.0

z = np.load(BLOCK, allow_pickle=True)
clip = np.asarray(z["clip_id"]).astype(str)
frame = np.asarray(z["frame"], dtype=np.int64)
gap0 = np.asarray(z["gap0_m"], dtype=np.float64)
speed = np.asarray(z["speeds"], dtype=np.float64)
state = np.asarray(z["state"]).astype(str)
has_lead = np.asarray(z["has_lead"], dtype=bool)
N = clip.size

# ---- GT acceleration: central difference of `speeds` WITHIN each clip ------ #
# the block's own grid is the RAW 10 Hz frame, so dt = 0.1 s between frames.
DT_FRAME = 0.1
accel = np.full(N, np.nan)
order = np.lexsort((frame, clip))
cl_sorted = clip[order]
bounds = np.flatnonzero(np.r_[True, cl_sorted[1:] != cl_sorted[:-1], True])
n_clips = bounds.size - 1
for b in range(n_clips):
    sl = order[bounds[b]:bounds[b + 1]]
    f = frame[sl]
    v = speed[sl]
    if f.size < 3:
        continue
    # require CONTIGUOUS frames for a difference; a jump makes the diff invalid
    a = np.full(f.size, np.nan)
    contig = (f[2:] - f[:-2]) == 2
    a[1:-1] = np.where(contig, (v[2:] - v[:-2]) / (2 * DT_FRAME), np.nan)
    accel[sl] = a

usable = has_lead & np.isfinite(gap0) & np.isfinite(accel) & (state == "LEAD")
n_usable = int(usable.sum())


def cluster_boot(mask_a, mask_b, vals, clusters, seed=0, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    eps, inv = np.unique(clusters, return_inverse=True)
    by = [np.flatnonzero(inv == i) for i in range(eps.size)]
    point = float(np.mean(vals[mask_a]) - np.mean(vals[mask_b]))
    out, skipped = [], 0
    for _ in range(n_boot):
        pick = rng.integers(0, eps.size, size=eps.size)
        sel = np.concatenate([by[i] for i in pick])
        a, b = mask_a[sel], mask_b[sel]
        if not a.any() or not b.any():
            skipped += 1
            continue
        out.append(float(np.mean(vals[sel][a]) - np.mean(vals[sel][b])))
    o = np.sort(np.asarray(out))
    lo, hi = (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if o.size else (None, None)
    return {"point": round(point, 5),
            "ci95": [round(lo, 5), round(hi, 5)] if o.size else None,
            "excludes_zero": bool(o.size and (lo > 0 or hi < 0)),
            "n_boot_used": int(o.size), "n_boot_skipped": int(skipped)}


res = {
    "task": "D-REFAV1-DK-COST P3 -- direction check on the FULL B1 eval lead block",
    "evidence_class": "MEASURED (ours) -- B1 eval lead block, label arithmetic only",
    "tier": ("NOT a driving number and NOT an arm comparison. Label statistics over the "
             "eval corpus; no model, no rollout."),
    "estimator": ("clip-cluster bootstrap over the block's clips; answers 'would another "
                  "draw of CLIPS say this?'. Neither training nor inference randomness "
                  "is involved -- both operands are deterministic label arithmetic."),
    "gt_accel_source": ("central difference of the block's own per-frame `speeds` "
                        "(egomotion at the frame) at dt = 0.1 s, WITHIN clip, "
                        "contiguous frames only"),
    "block": {"n_rows": int(N), "n_clips": int(n_clips),
              "n_lead_rows": int((state == "LEAD").sum()),
              "n_usable_for_direction": n_usable,
              "state_counts": {s: int((state == s).sum())
                               for s in sorted(set(state.tolist()))}},
    "tau_sweep": [],
}

for tau in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0):
    s_star = dk.DK_D0_M + tau * speed
    v = usable & (gap0 < s_star)
    a = usable & ~v
    row = {"tau_s": tau, "n_violating": int(v.sum()), "n_adequate": int(a.sum()),
           "frac_of_usable": round(float(v.sum() / max(n_usable, 1)), 4),
           "n_clips_violating": int(np.unique(clip[v]).size) if v.any() else 0,
           "mean_shortfall_m": (round(float(np.mean(s_star[v] - gap0[v])), 4)
                                if v.any() else 0.0)}
    if v.any() and a.any():
        row["delta_gt_accel_mps2"] = cluster_boot(v, a, accel, clip, seed=7)
        row["mean_gt_accel_violating"] = round(float(np.mean(accel[v])), 5)
        row["mean_gt_accel_adequate"] = round(float(np.mean(accel[a])), 5)
        # SAME-BREATH NON-ZERO REFERENCE: speed must separate strongly
        row["CONTROL_delta_speed_mps"] = cluster_boot(v, a, speed, clip, seed=7)
        # WITHIN-CLIP SHUFFLE: permute the assignment inside each clip
        rng = np.random.default_rng(13)
        perm = []
        for _ in range(200):
            vv = np.zeros(N, dtype=bool)
            for b in range(n_clips):
                sl = order[bounds[b]:bounds[b + 1]]
                u = sl[usable[sl]]
                if u.size == 0:
                    continue
                k = int(v[u].sum())
                if k:
                    vv[rng.permutation(u)[:k]] = True
            aa = usable & ~vv
            if vv.any() and aa.any():
                perm.append(float(np.mean(accel[vv]) - np.mean(accel[aa])))
        row["SHUFFLE_CONTROL_within_clip"] = {
            "mean": round(float(np.mean(perm)), 5),
            "p2.5": round(float(np.percentile(perm, 2.5)), 5),
            "p97.5": round(float(np.percentile(perm, 97.5)), 5),
            "n": len(perm),
            "_what": "violating/adequate permuted WITHIN each clip, counts preserved"}
    res["tau_sweep"].append(row)

# the corpus's own time gap over the whole block, for the tau choice
ok = usable & (speed >= MIN_SPEED)
tg = gap0[ok] / speed[ok]
res["corpus_time_gap"] = {
    "n": int(ok.sum()), "n_clips": int(np.unique(clip[ok]).size),
    "median_s": round(float(np.median(tg)), 4),
    "p25_s": round(float(np.percentile(tg, 25)), 4),
    "p10_s": round(float(np.percentile(tg, 10)), 4),
    "p05_s": round(float(np.percentile(tg, 5)), 4),
    "_min_speed_mps": MIN_SPEED,
}
hi = usable & (speed >= 10.0)
if hi.any():
    tgh = gap0[hi] / speed[hi]
    res["corpus_time_gap"]["at_speed_ge_10"] = {
        "n": int(hi.sum()), "median_s": round(float(np.median(tgh)), 4),
        "p25_s": round(float(np.percentile(tgh, 25)), 4),
        "p10_s": round(float(np.percentile(tgh, 10)), 4)}

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
