#!/usr/bin/env python3
"""TIER 3b — reconcile our pooled R^2 with the IDM v2 diagnosis's `r = 0.9865`.

`IDM_DIAGNOSIS.md` §3.4 reports corr(steer, yaw_rate/v) = 0.9865 on PhysicalAI
and concludes the channel carries "zero information beyond (yaw_rate, speed)".
Our Tier-3 pooled, out-of-sample ridge gets R^2 = 0.5689 and a pooled
corr = 0.8747. Both can be right — they are different estimators — so this
script computes ALL of them on one table and says which supports which claim.

`idm2_diag_labels.py:114-121` computes the correlation PER EPISODE, masked to
`v > 2.0 m/s`, and the headline is the MEAN of the per-episode correlations. A
mean of within-episode correlations removes every between-episode difference in
scale and offset — it is the ceiling of a model that gets a free per-clip
recalibration, which no deployed model gets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import wb_tier3_redundancy as t3        # reuse the table builder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ego-root",
                    default=r"C:\Users\Admin\tanitad-data\physicalai\labels\egomotion")
    ap.add_argument("--cam-root",
                    default=r"C:\Users\Admin\tanitad-data\physicalai\r0\camera_front_wide")
    a = ap.parse_args()
    scratch, out = Path(a.scratch), Path(a.out)
    cam_root = Path(a.cam_root)
    cam_have = {p.name.split(".")[0]: p
                for p in cam_root.glob("*.timestamps.parquet")} if cam_root.exists() else {}
    rows = json.load(open(scratch / "corpus_dims.json", encoding="utf-8"))
    tab = t3.build_table(rows, Path(a.ego_root), cam_have)

    st, w, v, ep = (tab["steer"].astype(np.float64), tab["w"].astype(np.float64),
                    tab["v"].astype(np.float64), tab["ep"])
    res = {"_meta": {"evidence_class": "MEASURED",
                     "n_samples": int(len(st)),
                     "n_episodes": int(len(np.unique(ep))),
                     "reference": "IDM_DIAGNOSIS.md §3.4 r = 0.9865 (PhysicalAI)"}}

    for tag, m in (("all_speeds", np.ones_like(v, bool)),
                   ("v_gt_2mps_IDM_mask", v > 2.0)):
        k = w[m] / np.maximum(v[m], 1e-6)
        s = st[m]
        e = ep[m]
        pooled = float(np.corrcoef(s, k)[0, 1])
        per_ep = []
        for u in np.unique(e):
            sel = e == u
            if sel.sum() > 20 and s[sel].std() > 1e-9 and k[sel].std() > 1e-9:
                per_ep.append(float(np.corrcoef(s[sel], k[sel])[0, 1]))
        per_ep = np.array(per_ep)
        # oracle per-clip affine recalibration of kappa -> steer (the ceiling the
        # per-episode correlation implies); pooled R^2 of a single global affine
        ss_res_or, ss_res_gl, ss_tot = 0.0, 0.0, 0.0
        A = np.c_[k, np.ones(len(k))]
        gl = np.linalg.lstsq(A, s, rcond=None)[0]
        gl_pred = A @ gl
        mu = s.mean()
        for u in np.unique(e):
            sel = e == u
            if sel.sum() < 3:
                continue
            Ai = np.c_[k[sel], np.ones(sel.sum())]
            wi = np.linalg.lstsq(Ai, s[sel], rcond=None)[0]
            ss_res_or += float(((s[sel] - Ai @ wi) ** 2).sum())
        ss_res_gl = float(((s - gl_pred) ** 2).sum())
        ss_tot = float(((s - mu) ** 2).sum())
        res[tag] = {
            "n_samples": int(m.sum()),
            "pooled_corr_steer_vs_omega_over_v": round(pooled, 4),
            "pooled_r2_single_global_affine": round(1 - ss_res_gl / ss_tot, 4),
            "IDM_style_mean_per_episode_corr": round(float(per_ep.mean()), 4),
            "IDM_style_median_per_episode_corr": round(float(np.median(per_ep)), 4),
            "per_episode_corr_p05": round(float(np.quantile(per_ep, 0.05)), 4),
            "n_episodes_scored": int(len(per_ep)),
            "oracle_per_clip_affine_r2": round(1 - ss_res_or / ss_tot, 4),
        }

    res["reading"] = (
        "The IDM figure is a MEAN OF WITHIN-EPISODE correlations at v > 2 m/s. "
        "It is the ceiling of a predictor granted a free per-clip affine "
        "recalibration. The number a model actually faces is the pooled, "
        "out-of-sample one (wb_tier3_redundancy.json Q1). Quoting the "
        "per-episode figure as 'steer carries zero information beyond "
        "(yaw_rate, speed)' overstates it by the between-clip term.")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "tier3b_redundancy_reconciliation.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
