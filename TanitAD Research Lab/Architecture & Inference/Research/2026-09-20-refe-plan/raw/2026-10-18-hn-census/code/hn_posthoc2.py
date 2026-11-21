#!/usr/bin/env python3
"""P6-HN step 0 -- POST-HOC 2 (NOT pre-registered; written after the zero picks were seen to sit ~1.6-1.9 m from the teacher).
Question: is the deviation of hard negatives / zero picks from the teacher LATERAL (cross-track) or LONGITUDINAL (along-track)?  The Lab's step-1 bank offsets the teacher path laterally by 0.15..0.60 m.
Decomposition: per pose, the proposal-minus-teacher vector projected on the teacher's local tangent (np.gradient of the teacher path) and its left normal; per proposal the MEAN absolute along and cross
components over the 20 poses, and the mean signed along component.  Population medians (point) -- descriptive, no CI.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I hn_posthoc2.py --joined ../raw/hn_joined.npz --out ../raw/posthoc2.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hn_census as H  # noqa: E402


def along_cross(P, E):
    """P [N,M,T,2], E [N,T,>=2] -> mean|along| [N,M], mean|cross| [N,M], mean signed along [N,M]  (teacher-tangent frame)."""
    e = E[..., :2].astype(np.float64)
    e0 = np.concatenate([np.zeros_like(e[:, :1]), e], axis=1)                  # prepend the origin so the first tangent exists
    tan = np.gradient(e0, axis=1)[:, 1:]                                       # [N,T,2]
    tan = tan / np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-6)
    nrm = np.stack([-tan[..., 1], tan[..., 0]], -1)
    d = P.astype(np.float64) - e[:, None]
    al = (d * tan[:, None]).sum(-1)
    cr = (d * nrm[:, None]).sum(-1)
    return np.abs(al).mean(-1), np.abs(cr).mean(-1), al.mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.joined)
    N = len(z["pdms"])
    P, E = z["proposals"], z["teacher"]
    nc, dac, pdms = z["nc"], z["dac"], z["pdms"]
    hn = H.hard_negative(nc, dac, pdms)
    ade, _ = H.ade_fde(P, E)
    al, cr, sal = along_cross(P, E)
    q = lambda v: {k: float(np.quantile(v, kq)) for k, kq in (("q25", .25), ("q50", .5), ("q75", .75))} if len(v) else None
    R = {"_what": "POST-HOC (not pre-registered). Teacher-tangent decomposition of the deviation from the teacher plan; medians, point.",
         "all_proposals_by_ADE_band": {}, "picks": {}}
    for lo, hi in H.BANDS[:-1]:
        bm = H.band_mask(ade, lo, hi)
        R["all_proposals_by_ADE_band"][f"({lo},{hi}]"] = {"HN": {"n": int((bm & hn).sum()), "mean|along|": q(al[bm & hn]), "mean|cross|": q(cr[bm & hn]), "signed_along": q(sal[bm & hn])},
                                                          "nonHN": {"n": int((bm & ~hn).sum()), "mean|along|": q(al[bm & ~hn]), "mean|cross|": q(cr[bm & ~hn]), "signed_along": q(sal[bm & ~hn])}}
    key2i = {(str(l), str(t), int(s), int(r)): i for i, (l, t, s, r) in enumerate(zip(z["keys_log"], z["keys_token"], z["keys_step"], z["keys_rank"]))}
    ar = np.arange(N)
    for arm in ("N-s0", "N-s1", "N-s2"):
        sc = np.load(f"D:/Projects/TanitAD-artifacts/s1_read_cache/arms/{arm}/dump/scores_update400_B.npz")
        idx = np.array([key2i[(str(l), str(t), int(s_), int(r))] for l, t, s_, r in zip(sc["log"], sc["token"], sc["step"], sc["rank"])])
        p = 1.0 / (1.0 + np.exp(-sc["scores"].astype(np.float64)[..., :6]))
        agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
        pick = np.empty(N, int); pick[idx] = agg.argmax(1)
        zp = pdms[ar, pick] == 0
        L_t = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, 1, 2)), E[..., :2].astype(float)], 1), axis=1), axis=-1).sum(1)
        pk = P[ar, pick].astype(float)
        L_p = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, 1, 2)), pk], 1), axis=1), axis=-1).sum(1)
        ratio = L_p / np.maximum(L_t, 0.5)
        mv = L_t >= 2.0                                                          # exclude near-stationary teachers from the ratio
        R["picks"][arm] = {"zero": {"n": int(zp.sum()), "mean|along|": q(al[ar, pick][zp]), "mean|cross|": q(cr[ar, pick][zp]), "signed_along": q(sal[ar, pick][zp]), "path_len_ratio_to_teacher": q(ratio[zp & mv]), "n_moving_teacher": int((zp & mv).sum())},
                           "nonzero": {"n": int((~zp).sum()), "mean|along|": q(al[ar, pick][~zp]), "mean|cross|": q(cr[ar, pick][~zp]), "signed_along": q(sal[ar, pick][~zp]), "path_len_ratio_to_teacher": q(ratio[~zp & mv]), "n_moving_teacher": int((~zp & mv).sum())}}

    # ---- POST-HOC 3: the same supply census with a CROSS-TRACK proximity (mean |cross| over the 20 poses, any speed)
    ulogs, log_idx = np.unique(z["keys_log"], return_inverse=True)
    B = H.Boot(log_idx, len(ulogs))
    R["cross_track_census"] = {"_what": "POST-HOC: proximity = mean |cross-track| to the teacher path (speed ignored); thresholds chosen BEFORE running: 0.3, 0.5, 1.0, 2.0 m (the Lab's offsets reach 0.6 m)", "A1": {}, "bands": {}, "zero_pick_coverage": {}}
    for c in (0.3, 0.5, 1.0, 2.0):
        npx, nhn = H.counts_at(cr, hn, c)
        R["cross_track_census"]["A1"][f"c={c}"] = {"share_sets_with_proximate": B.ratio((npx >= 1).astype(float)), "share_sets_with_proximate_HN": B.ratio((nhn >= 1).astype(float)),
                                                    "mean_n_prox_per_set": B.ratio(npx), "mean_n_HN_per_set": B.ratio(nhn), "pooled_HN_rate_among_proximate": B.ratio(nhn, npx)}
    for lo, hi in ((0.0, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, float("inf"))):
        bm = H.band_mask(cr, lo, hi)
        R["cross_track_census"]["bands"][f"({lo},{hi}]"] = {"HN_rate": B.ratio((bm & hn).sum(1), bm.sum(1)), "n_proposals": int(bm.sum()), "n_HN": int((bm & hn).sum())}
    for arm in ("N-s0", "N-s1", "N-s2"):
        sc = np.load(f"D:/Projects/TanitAD-artifacts/s1_read_cache/arms/{arm}/dump/scores_update400_B.npz")
        idx = np.array([key2i[(str(l), str(t), int(s_), int(r))] for l, t, s_, r in zip(sc["log"], sc["token"], sc["step"], sc["rank"])])
        p = 1.0 / (1.0 + np.exp(-sc["scores"].astype(np.float64)[..., :6]))
        agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
        pick = np.empty(N, int); pick[idx] = agg.argmax(1)
        zp = pdms[ar, pick] == 0
        R["cross_track_census"]["zero_pick_coverage"][arm] = {f"c={c}": B.ratio((zp & (cr[ar, pick] <= c)).astype(float), zp.astype(float)) for c in (0.3, 0.5, 1.0, 2.0)}
    json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
