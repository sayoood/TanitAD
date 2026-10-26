#!/usr/bin/env python3
"""HiCAP prior-mask measurement on committed artifacts (numpy only, torch-free).

Question: an ego-kinematics prior mask, built ONLY from inference-time quantities (current speed v0
and physical limits), removes what share of a trajectory candidate set, and does it ever remove the
true future?  Two numbers per limit setting:
  prune  = mean fraction of the 256 REF-C train anchors removed per val window
  recall = share of val windows whose ground-truth 2 s path itself passes the mask
plus the oracle-in-mask ADE (best surviving anchor) against the unmasked oracle ADE.

Data: 256 farthest-point anchors from 200 k parity-TRAIN windows (Data Engineering incoming/
2026-08-04-instrument-durability) and the committed val-40 dump taniteval/results/windows_refc-xl-30k.pt
(881 windows / 40 episodes; gt [W,4,2] at 0.5/1/1.5/2 s in the ego frame, speed = v0 in m/s).
Only 4 waypoints exist per path, so accelerations are 0.5 s finite differences: a coarse but
conservative test (the true 10 Hz path can only be smoother than its 2 Hz skeleton).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "TanitAD Research Hub/Architecture & Inference/Research")
from reff_v0_coverage import ANCHORS, WINDOWS, load_pt, episode_bootstrap  # noqa: E402

DT = 0.5


def path_features(P, v0):
    """P [...,4,2] ego-frame waypoints at 0.5..2.0 s (origin at t=0). Returns implied per-segment
    speeds, longitudinal accelerations (incl. from v0) and lateral accelerations."""
    pts = np.concatenate([np.zeros_like(P[..., :1, :]), P], axis=-2)          # [...,5,2]
    seg = np.linalg.norm(np.diff(pts, axis=-2), axis=-1) / DT                  # [...,4] mean speed / segment
    v_prev = np.concatenate([np.broadcast_to(v0[..., None], seg[..., :1].shape), seg[..., :-1]], axis=-1)
    a_lon = (seg - v_prev) / DT                                                # [...,4]
    d = np.diff(pts, axis=-2)                                                   # [...,4,2]
    hd = np.arctan2(d[..., 1], d[..., 0])
    dh = np.diff(hd, axis=-1)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    a_lat = np.abs(dh) / DT * seg[..., 1:]                                      # [...,3]  = yaw rate * speed
    return seg, a_lon, a_lat


def mask_ok(seg, a_lon, a_lat, a_acc, a_dec, a_lat_max):
    return ((a_lon.max(-1) <= a_acc) & (a_lon.min(-1) >= -a_dec) & (a_lat.max(-1) <= a_lat_max))


def main():
    anc = load_pt(ANCHORS)
    win = load_pt(WINDOWS)
    A = np.asarray(anc["anchors"], dtype=np.float64)          # [N,4,2]
    gt = np.asarray(win["gt"], dtype=np.float64)              # [W,4,2]
    v0 = np.asarray(win["speed"], dtype=np.float64).reshape(-1)
    eid = np.asarray(win["eid"]).reshape(-1)
    W, N = gt.shape[0], A.shape[0]
    # anchors are ABSOLUTE (speed-agnostic): for the mask they are used as-is; the speed-normalised
    # variant rescales each anchor to the window's v0 first (the residual-over-prior idea).
    seg_g, alon_g, alat_g = path_features(gt, v0)
    out = {"n_windows": int(W), "n_anchors": int(N), "estimator": "full-set mean; episode-cluster bootstrap CI95 B=2000",
           "gt_stats": {"a_lon_max_p99": float(np.percentile(alon_g.max(-1), 99)),
                        "a_lon_min_p01": float(np.percentile(alon_g.min(-1), 1)),
                        "a_lat_max_p99": float(np.percentile(alat_g.max(-1), 99)),
                        "a_lon_max_max": float(alon_g.max()), "a_lon_min_min": float(alon_g.min()),
                        "a_lat_max_max": float(alat_g.max())},
           "settings": []}
    s_anchor = np.linalg.norm(A[:, 0, :], axis=-1) / DT
    d_all = np.linalg.norm(gt[:, None] - A[None], axis=-1).mean(-1)            # absolute ADE [W,N]
    scale = np.where(s_anchor[None] > 0.5, np.clip(v0[:, None] / np.maximum(s_anchor[None], 1e-6), 0, 3), 1.0)
    for name, (aa, ad, al) in {"tight (acc 3, dec 6, lat 4 m/s^2)": (3.0, 6.0, 4.0),
                               "comfort-plus (acc 4, dec 8, lat 6)": (4.0, 8.0, 6.0),
                               "physical (acc 6, dec 10, lat 9)": (6.0, 10.0, 9.0)}.items():
        row = {"limits": name}
        for variant in ("absolute", "speed_normalised"):
            if variant == "absolute":
                Ac = np.broadcast_to(A[None], (W, N, 4, 2))
            else:
                Ac = A[None] * scale[:, :, None, None]
            seg, alon, alat = path_features(Ac, np.broadcast_to(v0[:, None], (W, N)))
            keep = mask_ok(seg, alon, alat, aa, ad, al)                        # [W,N]
            d = np.linalg.norm(gt[:, None] - Ac, axis=-1).mean(-1)
            d_masked = np.where(keep, d, np.inf).min(-1)
            d_full = d.min(-1)
            no_cand = np.isinf(d_masked)
            d_masked = np.where(no_cand, d_full, d_masked)                     # empty set -> fall back (counted below)
            rec_gt = mask_ok(seg_g, alon_g, alat_g, aa, ad, al)
            m_m, lo_m, hi_m = episode_bootstrap(d_masked, eid)
            m_f, lo_f, hi_f = episode_bootstrap(d_full, eid)
            row[variant] = {"prune_fraction_mean": round(float(1 - keep.mean()), 4),
                            "gt_path_passes_mask": round(float(rec_gt.mean()), 4),
                            "empty_candidate_sets": int(no_cand.sum()),
                            "oracle_ade_masked": [round(m_m, 4), round(lo_m, 4), round(hi_m, 4)],
                            "oracle_ade_unmasked": [round(m_f, 4), round(lo_f, 4), round(hi_f, 4)]}
        out["settings"].append(row)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
