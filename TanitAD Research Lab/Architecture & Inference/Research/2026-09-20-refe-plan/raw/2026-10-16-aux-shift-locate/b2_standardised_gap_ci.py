#!/usr/bin/env python3
"""Sized, interval-bearing version of the composition finding.  CPU, read-only, DESCRIPTIVE (a direct standardisation, not a causal model).
  (1) two-factor decomposition of the gap with BOTH orders and their average (Shapley for two factors):  F1 = failure-enrichment of the navtest sample, F2 = city mix.
  (2) residual gap after BOTH are standardised away, with an independent log-cluster bootstrap interval (the two domains have disjoint log sets; the population margins are FIXED constants, not resampled).
  (3) the cleanest feature-shift test the data allow: the ONE city with enough frames in both domains (LV), natural (non-enriched) navtest frames only.
Estimator for every interval: independent log-cluster bootstrap, B=4000, seed 20261016.  It answers 'another draw of LOGS' only (not another training run: the three H1 seeds are the training-run draws, reported side by side)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C  # noqa: E402
import shf_arrays as SA  # noqa: E402

ARMS = tuple(os.environ.get("SHF_ARMS", "H1-s0,H1-s1,H1-s2").split(","))
MAPS = ["LV", "BOS", "PIT", "SG"]
Dh, Dn = SA.domain("heldout", ARMS), SA.domain("navtest", ARMS)
b1 = json.load(open(os.path.join(C.HERE, os.environ.get("SHF_B1", "b1_decompose.json")), encoding="utf-8"))
T = b1["targets"]
share_ho_pop, share_train, strat_pop = T["share_heldout_logs_population"], T["share_train_all"], T["stratum_population"]
B = 4000


def logs_idx(D):
    ul = sorted(set(D["logs"]))
    pos = {l: i for i, l in enumerate(ul)}
    inv = np.array([pos[l] for l in D["logs"]])
    return ul, inv


def rake_from(w0, maps, strata, mm, ms, iters=60):
    w = w0.copy()
    for _ in range(iters):
        for lab, marg in ((maps, mm), (strata, ms)):
            if marg is None:
                continue
            tot = w.sum()
            for k, s in marg.items():
                m = lab == k
                sm_ = w[m].sum()
                if sm_ > 0:
                    w[m] *= s * tot / sm_
    return w


def ratio(i, u, w):
    return float((i * w).sum() / (u * w).sum())


res = {"estimator": "independent log-cluster bootstrap, B=4000, seed 20261016; population margins fixed; answers another draw of LOGS only"}
ulh, invh = logs_idx(Dh)
uln, invn = logs_idx(Dn)
ones_n = np.ones(Dn["n"])
for a in ARMS:
    rng = np.random.default_rng(20261016)
    ih, uh, _ = SA.frame_iou_parts(Dh, a)
    inn, un, _ = SA.frame_iou_parts(Dn, a)
    R = {}
    raw_h, raw_n = ratio(ih, uh, np.ones(Dh["n"])), ratio(inn, un, ones_n)
    R["heldout_raw"], R["navtest_raw"], R["gap_raw"] = raw_h, raw_n, raw_h - raw_n
    for tag, tgt in (("target=heldout_city_mix", share_ho_pop), ("target=train_city_mix", share_train)):
        w_s = rake_from(ones_n, Dn["maps"], Dn["stratum"], None, strat_pop)
        w_m = rake_from(ones_n, Dn["maps"], Dn["stratum"], tgt, None)
        w_sm = rake_from(ones_n, Dn["maps"], Dn["stratum"], tgt, strat_pop)
        a_s, a_m, a_sm = ratio(inn, un, w_s), ratio(inn, un, w_m), ratio(inn, un, w_sm)
        F1_first, F1_after = a_s - raw_n, a_sm - a_m
        F2_first, F2_after = a_m - raw_n, a_sm - a_s
        R[tag] = dict(navtest_stratum_standardised=a_s, navtest_city_standardised=a_m, navtest_both_standardised=a_sm,
                      F1_failure_enrichment=dict(first=F1_first, after_city=F1_after, shapley=0.5 * (F1_first + F1_after)),
                      F2_city_mix=dict(first=F2_first, after_stratum=F2_after, shapley=0.5 * (F2_first + F2_after)),
                      residual_gap_after_both=raw_h - a_sm, share_of_gap_explained=1 - (raw_h - a_sm) / (raw_h - raw_n))
    reps = {"gap_raw": [], "resid_heldout_mix": [], "F1_shapley": [], "F2_shapley": [], "navtest_natural_both": [], "heldout_raw": []}
    for r in range(B):
        sh = rng.integers(0, len(ulh), len(ulh))
        sn = rng.integers(0, len(uln), len(uln))
        mh = np.bincount(sh, minlength=len(ulh))[invh].astype(float)
        mn = np.bincount(sn, minlength=len(uln))[invn].astype(float)
        if mn.sum() == 0 or mh.sum() == 0:
            continue
        h_raw, n_raw = ratio(ih, uh, mh), ratio(inn, un, mn)
        ws_ = rake_from(mn, Dn["maps"], Dn["stratum"], None, strat_pop)
        wm_ = rake_from(mn, Dn["maps"], Dn["stratum"], share_ho_pop, None)
        wsm_ = rake_from(mn, Dn["maps"], Dn["stratum"], share_ho_pop, strat_pop)
        a_s_, a_m_, a_sm_ = ratio(inn, un, ws_), ratio(inn, un, wm_), ratio(inn, un, wsm_)
        reps["gap_raw"].append(h_raw - n_raw)
        reps["heldout_raw"].append(h_raw)
        reps["resid_heldout_mix"].append(h_raw - a_sm_)
        reps["F1_shapley"].append(0.5 * ((a_s_ - n_raw) + (a_sm_ - a_m_)))
        reps["F2_shapley"].append(0.5 * ((a_m_ - n_raw) + (a_sm_ - a_s_)))
        reps["navtest_natural_both"].append(a_sm_)
    R["bootstrap_target=heldout_city_mix"] = {k: dict(ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], n=len(v)) for k, v in reps.items()}
    res[a] = R
    print(a, json.dumps(R, indent=1, sort_keys=True), flush=True)

# (3) LV-only, natural (random-stratum) navtest frames vs held-out LV frames
lv = {}
for a in ARMS:
    ih, uh, _ = SA.frame_iou_parts(Dh, a)
    inn, un, _ = SA.frame_iou_parts(Dn, a)
    kh = Dh["maps"] == "LV"
    kn_all = Dn["maps"] == "LV"
    kn_nat = kn_all & (Dn["stratum"] == "random")
    est = lambda i, u, k: float(i[k].sum() / u[k].sum())
    rr = np.random.default_rng(99)
    reps = []
    for r in range(B):
        sh = rr.integers(0, len(ulh), len(ulh))
        sn = rr.integers(0, len(uln), len(uln))
        mh = np.bincount(sh, minlength=len(ulh))[invh].astype(float)
        mn = np.bincount(sn, minlength=len(uln))[invn].astype(float)
        d_ = (un * mn)[kn_nat].sum()
        if d_ == 0:
            continue
        reps.append((ih * mh)[kh].sum() / (uh * mh)[kh].sum() - (inn * mn)[kn_nat].sum() / d_)
    lv[a] = dict(heldout_LV=est(ih, uh, kh), n_heldout_LV=int(kh.sum()), logs_heldout_LV=int(len(set(Dh["logs"][kh]))),
                 navtest_LV_all_sample=est(inn, un, kn_all), n_navtest_LV=int(kn_all.sum()),
                 navtest_LV_natural_frames=est(inn, un, kn_nat), n_navtest_LV_natural=int(kn_nat.sum()), logs_navtest_LV_natural=int(len(set(Dn["logs"][kn_nat]))),
                 navtest_LV_zero_stratum=est(inn, un, kn_all & (Dn["stratum"] == "zero")),
                 gap_heldoutLV_minus_navtestLVnatural=est(ih, uh, kh) - est(inn, un, kn_nat),
                 gap_ci95=[float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))])
res["LV_only_natural_frames"] = lv
print(json.dumps(lv, indent=1, sort_keys=True))
C.jdump(res, os.path.join(C.HERE, os.environ.get("SHF_OUT", "b2_standardised_gap_ci.json")))
print("ZZB2_DONE")
