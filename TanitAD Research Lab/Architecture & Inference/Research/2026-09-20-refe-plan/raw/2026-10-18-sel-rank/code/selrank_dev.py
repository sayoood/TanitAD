#!/usr/bin/env python3
"""SEL-RANK: the navtrain HELD-OUT 64-proposal sub-score table (the DEV set) -- what exists, and the SAME descriptive attribution on it.

The dev table = 3,137 held-out sets on 24 navtrain logs, NONE of them a navtest log (checked below), with
  * banked PDM labels for all 64 proposals of every set: nc, dac, ep, ttc, c, ddc, pdms  (label_version 5, plans 'executed (A7 repair)', fan = model_final ckpt_step 10075)
      D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm/pdm_onpolicy_r{0,1}_hw{0..6}.jsonl
  * scorer logits [3137, 64, 7] of NINE SFT-4 sibling scorers (u400 + 400 updates; arms N-s0/1/2, H1-s0/1/2, H2-s0/1/2)
      D:/Projects/TanitAD-artifacts/s1_read_cache/arms/<arm>/dump/scores_update400_B.npz
  ==> The labels are scorer-independent and are for the u400 fan, NOT P2's fan.  The logits are NOT P2's scorer's.  P2 has NO held-out table (see RESULT.md).
Everything here is a DEV-SET look (it may inform an arm; navtest stays the confirmation set).  Estimator: log-cluster bootstrap over the 24 held-out logs, 10,000 resamples,
seed 20260927 (zero_tail.Boot) -- only 24 clusters, so the intervals are wide and answer "another draw of LOGS" only.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe selrank_dev.py --out ../raw/dev_replicate.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zero_tail import Boot, B_BOOT, SEED  # noqa: E402
from selrank_attr import agg_v1_np, sigmoid64, shapley_terms, ratio, TIE_TOL  # noqa: E402

CACHE = "D:/Projects/TanitAD-artifacts/s1_read_cache"
LABELS = CACHE + "/labels/x/heldout_pdm/pdm_onpolicy_*.jsonl"
ARMS = ("N-s0", "N-s1", "N-s2", "H1-s0", "H1-s1", "H1-s2", "H2-s0", "H2-s1", "H2-s2")
NAVTEST_TABLE = "D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz"
HEADS = ("nc", "dac", "ep", "ttc", "c", "ddc")


def load_dev(arm_list=ARMS):
    lab = {}
    for p in sorted(glob.glob(LABELS)):
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            lab[tuple(d["key"])] = d
    z0 = np.load(f"{CACHE}/arms/{arm_list[0]}/dump/scores_update400_B.npz")
    keys = [(str(a), str(b), int(c), int(d)) for a, b, c, d in zip(z0["log"], z0["token"], z0["step"], z0["rank"])]
    assert all(k in lab for k in keys), "label keys do not cover the score keys"
    sub = np.stack([np.stack([np.asarray(lab[k][h], np.float64) for h in HEADS], axis=-1) for k in keys])          # [N,64,6]
    pdms = np.stack([np.asarray(lab[k]["pdms"], np.float64) for k in keys])
    S = {}
    for arm in arm_list:
        z = np.load(f"{CACHE}/arms/{arm}/dump/scores_update400_B.npz")
        zk = [(str(a), str(b), int(c), int(d)) for a, b, c, d in zip(z["log"], z["token"], z["step"], z["rank"])]
        assert zk == keys and int(z["ckpt_step"][0]) == 10075, arm
        S[arm] = z["scores"].astype(np.float64)                                                                     # [N,64,7]
    log = np.array([k[0] for k in keys])
    return keys, log, sub, pdms, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raw", "dev_replicate.json"))
    a = ap.parse_args()
    keys, log, sub, pdms, S = load_dev()
    N, M, _ = sub.shape
    ulogs, log_idx = np.unique(log, return_inverse=True)
    B = Boot(log_idx, len(ulogs))
    ar = np.arange(N)
    nav_logs = set(np.load(NAVTEST_TABLE)["log"].tolist())
    R = {"_what": "SEL-RANK DEV set: navtrain held-out 64-proposal sub-score table (u400 fan, NOT P2's), sibling-scorer logits (NOT P2's). DEV-SET look, not a confirmation.",
         "estimator": {"name": "log-cluster bootstrap (zero_tail.Boot)", "B": B_BOOT, "seed": SEED, "n_logs": int(len(ulogs)), "n_sets": N,
                       "answers": "another draw of 24 LOGS only; scorers are single SFT-4 runs (3 seeds per recipe are READ SEPARATELY here)"},
         "gates": {"n_sets": N, "n_logs": int(len(ulogs)), "n_logs_shared_with_navtest": len(set(ulogs.tolist()) & nav_logs),
                   "pdms_equals_NC_DAC_W12_max_abs_diff": float(np.abs(agg_v1_np(sub) - pdms).max()),
                   "pdms_with_ddc_in_product_max_abs_diff": float(np.abs(agg_v1_np(sub) * sub[..., 5] - pdms).max()),
                   "ep_zero_among_violating": float((sub[..., 2][(sub[..., 0] == 0) | (sub[..., 1] == 0)] == 0).mean()),
                   "labels_oracle_pdms_x100": float(pdms.max(1).mean() * 100), "labels_random_pdms_x100": float(pdms.mean() * 100)}, "arms": {}}
    orc = pdms.max(1)
    ties = np.abs(pdms - orc[:, None]) <= TIE_TOL
    tiew = ties / ties.sum(1, keepdims=True)
    names5 = ("nc", "dac", "ep", "ttc", "c")
    for arm, sc7 in S.items():
        L = sc7[..., :6]
        P = sigmoid64(L)
        pk = agg_v1_np(P).argmax(1)                                                    # the 6-head navsim_v1 rule (what P2 deploys)
        pk_lane = (agg_v1_np(P) * sigmoid64(sc7[..., 6])).argmax(1)                    # the siblings' NATIVE rule navsim_v1_lane
        pp = pdms[ar, pk]
        gap = orc - pp
        nz = pp > 0
        rec = {"pick_navsim_v1_pdms_x100": ratio(B, pp, scale=100), "pick_navsim_v1_lane_pdms_x100": ratio(B, pdms[ar, pk_lane], scale=100),
               "oracle_pdms_x100": ratio(B, orc, scale=100), "oracle_minus_pick_x100": ratio(B, gap, scale=100),
               "pool_gap_from_nonzero_picks_x100": ratio(B, gap * nz, scale=100), "gap_from_zero_picks_x100": ratio(B, gap * ~nz, scale=100),
               "n_zero_picks": int((~nz).sum())}
        contrib, err = shapley_terms(sub, pk, tiew)
        cn = np.where((gap > TIE_TOL)[:, None], contrib, 0.0)
        rec["pool_gap_by_term_x100"] = {nm: ratio(B, cn[:, k] * nz, scale=100) for k, nm in enumerate(names5)}
        rec["pool_share_by_term"] = {nm: float((cn[:, k] * nz).sum() / (gap * nz).sum()) for k, nm in enumerate(names5)}
        # per-head within-token Spearman (all 64) and inside the scorer's own top-8
        sc_v1 = agg_v1_np(P)
        topk = rankdata(-sc_v1, axis=1, method="ordinal") <= 8
        rho = {}
        for h in ("ep", "ttc", "nc", "dac"):
            k = HEADS.index(h)
            x, y = P[..., k], sub[..., k]
            rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
            cx, cy = rx - rx.mean(1, keepdims=True), ry - ry.mean(1, keepdims=True)
            den = np.sqrt((cx ** 2).sum(1) * (cy ** 2).sum(1))
            d = den > 0
            r_all = np.where(d, (cx * cy).sum(1) / np.where(d, den, 1.0), 0.0)
            r8 = np.full(N, np.nan)
            for i in range(N):
                idx = np.where(topk[i])[0]
                if x[i, idx].std() > 0 and y[i, idx].std() > 0:
                    r8[i] = np.corrcoef(rankdata(x[i, idx]), rankdata(y[i, idx]))[0, 1]
            d8 = ~np.isnan(r8)
            rho[h] = {"all64_n_defined": int(d.sum()), "all64_mean_spearman": ratio(B, r_all * d, d.astype(float)),
                      "top8_n_defined": int(d8.sum()), "top8_mean_spearman": ratio(B, np.nan_to_num(r8) * d8, d8.astype(float))}
        rec["spearman_per_head"] = rho
        # oracle-head bounds (same definition as the navtest section D)
        base_p = P[..., :5].copy()
        viol = (sub[..., 0] == 0) | (sub[..., 1] == 0)

        def bound(replace, valid_only=False):
            pm = base_p.copy()
            for h in replace:
                k = HEADS.index(h)
                pm[..., k] = np.where(viol, base_p[..., k], sub[..., k]) if valid_only else sub[..., k]
            k_ = agg_v1_np(pm).argmax(1)
            d_ = pdms[ar, k_] - pp
            return {"gain_x100": ratio(B, d_, scale=100), "gain_from_nonzero_pick_x100": ratio(B, d_ * nz, scale=100),
                    "share_of_pool_captured": ratio(B, d_ * nz, gap * nz), "share_of_total_gap_captured": ratio(B, d_, gap)}
        rec["oracle_head_bounds"] = {"EP": bound(("ep",)), "EP_realised_only_on_nonviolating": bound(("ep",), True), "TTC": bound(("ttc",)), "EP_TTC": bound(("ep", "ttc")),
                                     "NC": bound(("nc",)), "DAC": bound(("dac",)), "NC_DAC": bound(("nc", "dac")), "ALL5": bound(names5)}
        R["arms"][arm] = rec
        print(arm, "pick", round(rec["pick_navsim_v1_pdms_x100"]["point"], 2), "orc", round(rec["oracle_pdms_x100"]["point"], 2), "pool", round(rec["pool_gap_from_nonzero_picks_x100"]["point"], 3), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
