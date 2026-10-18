#!/usr/bin/env python3
"""P6-HN step 0 -- the expert-proximate hard-negative census.  Implements PREREG_HN_CENSUS.md (hash in PREREG_HN_CENSUS.md.sha256) and nothing else.

Pure functions first (unit-tested in test_hn_census.py), then `main`.  CPU only, numpy only.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I hn_census.py --joined ../raw/hn_joined.npz --human ../raw/human_future.npz \
        --out ../raw/census_results.json --per-set ../raw/per_set.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

B_BOOT = 10000
SEED = 20260927
GRID = (0.5, 1.0, 2.0, 4.0)
BANDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, float("inf")))
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
BIN = 0.01
NBINS = 6000          # 0 .. 60 m, the last bin is an overflow bin
TWIN_T = (0.3, 0.5)
KAPPA = 0.953         # PDMS per pp of zero rate   (LAB-RUN-019 s4.3; MEASURED inputs 4.15 / 4.36)
Z_NAVTEST = 4.36      # pp, navtest u400 (INHERITED)
R_PUB = {"DA-WAM": 0.208, "BeyondDrive": 0.227}
STAT_M, TURN_DEG = 2.0, 15.0


# ================================================================== pure functions
def ade_fde(props: np.ndarray, E: np.ndarray):
    """props [N,M,T,2], E [N,T,>=2] -> ADE [N,M] (mean over T of xy L2), FDE [N,M] (last pose)."""
    d = np.linalg.norm(props.astype(np.float64) - E[:, None, :, :2].astype(np.float64), axis=-1)
    return d.mean(-1), d[..., -1]


def hard_negative(nc: np.ndarray, dac: np.ndarray, pdms: np.ndarray) -> np.ndarray:
    """PREREG 3.3: hard negative == PDMS zero (== NC 0 or DAC 0, gate G3)."""
    return pdms == 0


def counts_at(dist: np.ndarray, hn: np.ndarray, d: float):
    """n_prox, n_hn per set at threshold d (inclusive).  dist/hn [N,M]."""
    prox = dist <= d
    return prox.sum(1), (prox & hn).sum(1)


def expert_class(E: np.ndarray):
    """STATIONARY / LEFT / STRAIGHT / RIGHT from the expert's own path (PREREG 3.6).  E [N,T,3] (x,y,yaw)."""
    xy = E[..., :2].astype(np.float64)
    steps = np.linalg.norm(np.diff(np.concatenate([np.zeros_like(xy[:, :1]), xy], axis=1), axis=1), axis=-1)
    L = steps.sum(1)
    yaw_deg = np.degrees(E[:, -1, 2].astype(np.float64))
    cls = np.where(L < STAT_M, "STATIONARY", np.where(yaw_deg >= TURN_DEG, "LEFT", np.where(yaw_deg <= -TURN_DEG, "RIGHT", "STRAIGHT")))
    return cls, L, yaw_deg


class Boot:
    """Log-cluster bootstrap of ratio estimators sum(num)/sum(den); one set of draws, reused by every call (zero_tail.Boot semantics)."""

    def __init__(self, log_idx: np.ndarray, n_logs: int, B: int = B_BOOT, seed: int = SEED):
        self.log_idx, self.n_logs, self.B = np.asarray(log_idx), n_logs, B
        self.draws = np.random.default_rng(seed).integers(0, n_logs, size=(B, n_logs))
        rows = np.repeat(np.arange(B), n_logs)
        self.C = np.zeros((B, n_logs))                       # how many times each log is drawn
        np.add.at(self.C, (rows, self.draws.ravel()), 1.0)

    def ratio(self, num, den=None, scale: float = 1.0) -> dict:
        num = np.asarray(num, float)
        den = np.ones_like(num) if den is None else np.asarray(den, float)
        s_n = np.bincount(self.log_idx, weights=num, minlength=self.n_logs)
        s_d = np.bincount(self.log_idx, weights=den, minlength=self.n_logs)
        d_all = s_d.sum()
        point = float(s_n.sum() / d_all * scale) if d_all > 0 else float("nan")
        bd = s_d[self.draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(bd > 0, s_n[self.draws].sum(axis=1) / bd * scale, np.nan)
        ok = ~np.isnan(v)
        if not ok.any():
            return {"point": point, "lo": float("nan"), "hi": float("nan"), "n": int(round(d_all))}
        return {"point": point, "lo": float(np.percentile(v[ok], 2.5)), "hi": float(np.percentile(v[ok], 97.5)), "n": int(round(d_all))}

    def per_log_hist(self, values: np.ndarray, log_of_value: np.ndarray) -> np.ndarray:
        v = np.asarray(values, float).ravel()
        li = np.asarray(log_of_value).ravel()
        b = np.minimum(np.floor(v / BIN), NBINS - 1).astype(np.int64)
        H = np.zeros((self.n_logs, NBINS))
        np.add.at(H, (li, b), 1.0)
        return H

    def hist_quantiles(self, H: np.ndarray, qs=QS, chunk: int = 250) -> dict:
        """Quantiles (in-bin linear interpolation) of the pooled per-log histograms H [L,NBINS], point + cluster-bootstrap CI."""
        out = {}
        pt = quantiles_from_hist(H.sum(0)[None, :], qs)[0]
        res = np.empty((self.B, len(qs)))
        for s in range(0, self.B, chunk):
            Hb = self.C[s:s + chunk] @ H
            res[s:s + chunk] = quantiles_from_hist(Hb, qs)
        for k, q in enumerate(qs):
            ok = ~np.isnan(res[:, k])
            out[f"q{int(round(q * 100)):02d}"] = {"point": float(pt[k]), "lo": float(np.percentile(res[ok, k], 2.5)) if ok.any() else float("nan"),
                                                  "hi": float(np.percentile(res[ok, k], 97.5)) if ok.any() else float("nan")}
        out["n"] = int(H.sum())
        out["overflow_frac"] = float(H[:, -1].sum() / max(H.sum(), 1))
        return out


def quantiles_from_hist(Hrows: np.ndarray, qs) -> np.ndarray:
    """Hrows [R,NBINS] counts -> [R,len(qs)] quantiles with linear interpolation inside the bin (value = (bin + frac) * BIN)."""
    cs = np.cumsum(Hrows, axis=1)
    tot = cs[:, -1]
    out = np.full((Hrows.shape[0], len(qs)), np.nan)
    for k, q in enumerate(qs):
        T = q * tot
        idx = (cs < T[:, None]).sum(1)
        idx = np.minimum(idx, Hrows.shape[1] - 1)
        r = np.arange(Hrows.shape[0])
        prev = np.where(idx > 0, cs[r, np.maximum(idx - 1, 0)], 0.0)
        cnt = Hrows[r, idx]
        frac = np.where(cnt > 0, (T - prev) / np.where(cnt > 0, cnt, 1), 0.0)
        out[:, k] = np.where(tot > 0, (idx + np.clip(frac, 0, 1)) * BIN, np.nan)
    return out


def band_mask(dist: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """(lo, hi] except the first band which is [0, hi]."""
    return (dist >= lo) & (dist <= hi) if lo == 0.0 else (dist > lo) & (dist <= hi)


def pair_dist(P: np.ndarray) -> np.ndarray:
    """Mean-over-poses pairwise L2 between the proposals of ONE set.  P [M,T,2] -> [M,M]."""
    return np.linalg.norm(P[:, None].astype(np.float64) - P[None].astype(np.float64), axis=-1).mean(-1)


def pair_stats(D: np.ndarray, dac: np.ndarray, pdms: np.ndarray, t: float):
    """Within-set twin pairs at threshold t on the MEAN pairwise L2 matrix D [M,M] (PREREG 3.5)."""
    near = D <= t
    d0, d1 = dac == 0, dac == 1
    p0, p1 = pdms == 0, pdms > 0
    pair_dac = near & d0[:, None] & d1[None, :]
    pair_pdm = near & p0[:, None] & p1[None, :]
    return {"n_pairs_dac": int(pair_dac.sum()), "n_dac0": int(d0.sum()), "n_dac0_twinned": int((pair_dac.any(1)).sum()),
            "n_pairs_pdm": int(pair_pdm.sum()), "n_pdms0": int(p0.sum()), "n_pdms0_twinned": int((pair_pdm.any(1)).sum())}


def cyclic_wrong_expert(E: np.ndarray, log_idx: np.ndarray) -> np.ndarray:
    """K2 control: replace each set's expert by the expert of another set of the SAME log (shift by half the log's length)."""
    out = np.empty_like(E)
    for lg in np.unique(log_idx):
        ix = np.where(log_idx == lg)[0]
        sh = max(len(ix) // 2, 1)
        out[ix] = E[ix[(np.arange(len(ix)) + sh) % len(ix)]]
    return out


# ================================================================== analysis
def a1_block(B, dist, hn_flag, dac0, nc0only, both, strata_mask=None):
    """Supply block at every d for one distance matrix.  dist/hn_flag [N,M]."""
    N = dist.shape[0]
    m = np.ones(N, bool) if strata_mask is None else strata_mask
    R = {}
    hn_total = hn_flag.sum(1)
    for d in GRID:
        npx, nhn = counts_at(dist, hn_flag, d)
        has_p, has_h = (npx >= 1), (nhn >= 1)
        sub = {}
        for nm, f in (("DAC_only", dac0 & ~both), ("NC_only", nc0only), ("both", both)):
            sub[nm] = B.ratio((counts_at(dist, f, d)[1] >= 1) * m, m.astype(float))
        R[f"d={d}"] = {
            "share_sets_with_proximate": B.ratio(has_p * m, m.astype(float)),
            "share_sets_with_proximate_HN": B.ratio(has_h * m, m.astype(float)),
            "share_HN_given_proximate": B.ratio(has_h * m, has_p * m),
            "mean_n_prox_per_set": B.ratio(npx * m, m.astype(float)),
            "mean_n_HN_per_set": B.ratio(nhn * m, m.astype(float)),
            "pooled_HN_rate_among_proximate": B.ratio(nhn * m, npx * m),
            "share_sets_with_proximate_HN_by_subtype": sub,
            "share_of_all_HN_that_are_proximate": B.ratio(nhn * m, hn_total * m),
            "hist_n_HN_per_set(point)": {"0": int(((nhn == 0) & m).sum()), "1": int(((nhn == 1) & m).sum()), "2": int(((nhn == 2) & m).sum()),
                                        "3-4": int((((nhn >= 3) & (nhn <= 4)) & m).sum()), "5+": int(((nhn >= 5) & m).sum())},
            "n_sets": int(m.sum())}
    return R


def band_block(B, dist, hn_flag):
    R = {}
    for lo, hi in BANDS:
        bm = band_mask(dist, lo, hi)
        R[f"({lo},{hi}]" if lo else f"[0,{hi}]"] = {"HN_rate": B.ratio((bm & hn_flag).sum(1), bm.sum(1)), "n_proposals": int(bm.sum()), "n_HN": int((bm & hn_flag).sum())}
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", required=True)
    ap.add_argument("--human", default=None)
    ap.add_argument("--scorers", default="D:/Projects/TanitAD-artifacts/s1_read_cache/arms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-set", default=None)
    a = ap.parse_args()

    z = np.load(a.joined)
    N = len(z["pdms"])
    P = z["proposals"]
    ET = z["teacher"]
    nc, dac, pdms = z["nc"], z["dac"], z["pdms"]
    hn = hard_negative(nc, dac, pdms)
    dac0 = dac == 0
    nc0only = (nc == 0) & (dac == 1)
    both = (nc == 0) & (dac == 0)
    logs = z["keys_log"]
    ulogs, log_idx = np.unique(logs, return_inverse=True)
    L = len(ulogs)
    B = Boot(log_idx, L)
    R = {"_what": "P6-HN step 0 census (PREREG_HN_CENSUS.md). Descriptive; log-cluster bootstrap over the 24 held-out logs, B=10000, seed 20260927 (another draw of LOGS only).",
         "n_sets": int(N), "n_logs": int(L), "n_proposals": int(hn.size), "grid_m": list(GRID)}

    # ---- base rates (D)
    per_set_rate = hn.mean(1)
    R["D_base_rates"] = {"HN_rate_all_proposals": B.ratio(hn.sum(1), np.full(N, 64.0)),
                         "share_sets_with_any_HN": B.ratio((hn.sum(1) >= 1).astype(float)),
                         "share_sets_with_any_DAC0": B.ratio((dac0.sum(1) >= 1).astype(float)),
                         "per_set_HN_rate_quantiles(point)": {f"q{int(q*100):02d}": float(np.quantile(per_set_rate, q)) for q in QS},
                         "n_HN": int(hn.sum()), "n_DAC0": int(dac0.sum()), "n_NC0_only": int(nc0only.sum()), "n_both": int(both.sum())}

    # ---- primary expert E_T
    ade, fde = ade_fde(P, ET)
    R["ET"] = {"A1_ADE": a1_block(B, ade, hn, dac0, nc0only, both), "A1_FDE": a1_block(B, fde, hn, dac0, nc0only, both),
               "A2e_bands_ADE": band_block(B, ade, hn), "A2e_bands_FDE": band_block(B, fde, hn)}
    near_all = ade.min(1)
    Hn_ = lambda v, li: B.per_log_hist(v, li)
    R["ET"]["A2_quantiles_ADE"] = {
        "a_all_proposals": B.hist_quantiles(Hn_(ade.ravel(), np.repeat(log_idx, 64))),
        "b_nearest_proposal_per_set": B.hist_quantiles(Hn_(near_all, log_idx)),
        "d_hard_negatives_pooled": B.hist_quantiles(Hn_(ade[hn], np.repeat(log_idx[:, None], 64, axis=1)[hn])),
        "d_non_hard_negatives_pooled": B.hist_quantiles(Hn_(ade[~hn], np.repeat(log_idx[:, None], 64, axis=1)[~hn]))}
    has_hn = hn.any(1)
    nearest_hn = np.where(hn, ade, np.inf).min(1)
    R["ET"]["A2_quantiles_ADE"]["c_nearest_HN_per_set"] = B.hist_quantiles(Hn_(nearest_hn[has_hn], log_idx[has_hn]))
    R["ET"]["A2_quantiles_ADE"]["c_share_sets_with_a_HN_anywhere"] = B.ratio(has_hn.astype(float))
    R["ET"]["A2_quantiles_FDE"] = {"a_all_proposals": B.hist_quantiles(Hn_(fde.ravel(), np.repeat(log_idx, 64))),
                                   "b_nearest_proposal_per_set": B.hist_quantiles(Hn_(fde.min(1), log_idx))}

    # ---- A3 strata
    cls, Lp, yaw_deg = expert_class(ET)
    R["G8_class_sign_check"] = {c: {"n": int((cls == c).sum()), "median_final_y": float(np.median(ET[cls == c][:, -1, 1])) if (cls == c).any() else None} for c in ("STATIONARY", "LEFT", "STRAIGHT", "RIGHT")}
    strata = {}
    for nm, mk in [(c, cls == c) for c in ("STATIONARY", "LEFT", "STRAIGHT", "RIGHT")] + [("rank0", z["keys_rank"] == 0), ("rank1", z["keys_rank"] == 1)]:
        nl = int(len(np.unique(log_idx[mk])))
        blk = a1_block(B, ade, hn, dac0, nc0only, both, strata_mask=mk)
        row = {"n_sets": int(mk.sum()), "n_logs": nl, "ci_valid": bool(mk.sum() >= 100 and nl >= 5), "HN_base_rate": B.ratio((hn.sum(1)) * mk, 64.0 * mk)}
        for d in GRID:
            e = blk[f"d={d}"]
            row[f"d={d}"] = {"share_sets_with_proximate_HN": e["share_sets_with_proximate_HN"], "share_sets_with_proximate": e["share_sets_with_proximate"],
                             "pooled_HN_rate_among_proximate": e["pooled_HN_rate_among_proximate"]}
        strata[nm] = row
    R["ET"]["A3_strata"] = strata

    # ---- A4 per-log table (point)
    plog = []
    for i, lg in enumerate(ulogs):
        m = log_idx == i
        row = {"log": str(lg), "n_sets": int(m.sum()), "HN_rate": float(hn[m].mean())}
        for d in GRID:
            npx, nhn = counts_at(ade[m], hn[m], d)
            row[f"s_hn_d{d}"] = float((nhn >= 1).mean()); row[f"s_prox_d{d}"] = float((npx >= 1).mean())
        plog.append(row)
    R["ET"]["A4_per_log"] = plog

    # ---- K1 identity, K2 wrong expert
    e0 = np.concatenate([P[:, 0], np.zeros((N, P.shape[2], 1), np.float32)], axis=-1)
    ade0, _ = ade_fde(P, e0)
    R["K1_identity"] = {"nearest_ADE_max": float(ade0.min(1).max()), "all_sets_have_proximate_at_every_d": bool(all((counts_at(ade0, hn, d)[0] >= 1).all() for d in GRID))}
    Ew = cyclic_wrong_expert(ET, log_idx)
    adew, fdew = ade_fde(P, Ew)
    R["K2_wrong_expert"] = {"A1_ADE": {f"d={d}": {"share_sets_with_proximate": B.ratio((counts_at(adew, hn, d)[0] >= 1).astype(float)),
                                                   "share_sets_with_proximate_HN": B.ratio((counts_at(adew, hn, d)[1] >= 1).astype(float))} for d in GRID},
                            "nearest_proposal_ADE_median_wrong": float(np.median(adew.min(1))), "nearest_proposal_ADE_median_true": float(np.median(near_all))}

    # ---- B twin pairs
    PB = {}
    Dall = [pair_dist(P[i]) for i in range(N)]
    for t in TWIN_T:
        rows = [pair_stats(Dall[i], dac[i], pdms[i], t) for i in range(N)]
        g = lambda k: np.array([r[k] for r in rows], float)
        PB[f"t={t}"] = {
            "share_sets_with_DAC_twin_pair": B.ratio((g("n_pairs_dac") >= 1).astype(float)),
            "mean_DAC_twin_pairs_per_set": B.ratio(g("n_pairs_dac")),
            "twinned_fraction_of_DAC0_proposals": B.ratio(g("n_dac0_twinned"), g("n_dac0")),
            "share_sets_with_DAC0_that_have_a_twin_pair": B.ratio((g("n_dac0_twinned") >= 1).astype(float), (g("n_dac0") >= 1).astype(float)),
            "share_sets_with_PDMS_twin_pair": B.ratio((g("n_pairs_pdm") >= 1).astype(float)),
            "twinned_fraction_of_PDMS0_proposals": B.ratio(g("n_pdms0_twinned"), g("n_pdms0"))}
    R["B_twin_pairs"] = PB

    # ---- A5 secondary expert E_H
    if a.human and os.path.exists(a.human):
        h = np.load(a.human)
        EH = h["human"]
        cov = ~np.isnan(EH).any(axis=(1, 2))
        R["E_H"] = {"coverage_sets": int(cov.sum()), "coverage_logs": int(len(np.unique(log_idx[cov]))), "n_sets": int(N)}
        if cov.any():
            adeh, fdeh = ade_fde(P[cov], np.nan_to_num(EH[cov]))
            adet_c, _ = ade_fde(P[cov], ET[cov])
            gap = np.linalg.norm(ET[cov][..., :2].astype(float) - EH[cov].astype(float), axis=-1).mean(-1)
            R["E_H"]["G7_median_ADE_teacher_vs_human"] = float(np.median(gap))
            R["E_H"]["G7_pass"] = bool(np.median(gap) <= 5.0)
            lc = log_idx[cov]
            Bc = Boot(lc, L)               # same draws (same seed, same L), restricted to the covered sets' logs
            R["E_H"]["A1_ADE_human"] = a1_block(Bc, adeh, hn[cov], dac0[cov], nc0only[cov], both[cov])
            R["E_H"]["A1_ADE_teacher_same_sets"] = a1_block(Bc, adet_c, hn[cov], dac0[cov], nc0only[cov], both[cov])
            R["E_H"]["A2e_bands_human"] = band_block(Bc, adeh, hn[cov])
            R["E_H"]["A2e_bands_teacher_same_sets"] = band_block(Bc, adet_c, hn[cov])
            R["E_H"]["nearest_proposal_ADE_median_human"] = float(np.median(adeh.min(1)))
            R["E_H"]["nearest_proposal_ADE_median_teacher_same_sets"] = float(np.median(adet_c.min(1)))
            R["E_H"]["teacher_vs_human_ADE_quantiles"] = {f"q{int(q*100):02d}": float(np.quantile(gap, q)) for q in QS}

    # ---- section 7 reach
    R["C_reach"] = {"inputs": {"kappa_PDMS_per_pp": KAPPA, "Z_pp_navtest_u400": Z_NAVTEST, "r_published": R_PUB,
                                "classes": {"kappa": "MEASURED inputs (INHERITED here)", "Z": "MEASURED on navtest (INHERITED, not re-measured)", "r": "PUBLISHED single runs"}},
                    "C1_supply_limited_PDMS": {}, "C2_zero_pick_coverage": {}}
    for d in GRID:
        s = R["ET"]["A1_ADE"][f"d={d}"]["share_sets_with_proximate_HN"]
        R["C_reach"]["C1_supply_limited_PDMS"][f"d={d}"] = {nm: {k: float(s[k] * r * Z_NAVTEST * KAPPA) for k in ("point", "lo", "hi")} for nm, r in R_PUB.items()}
    R["C_reach"]["C1_supply_limited_PDMS"]["unconstrained_standalone(s=1)"] = {nm: float(r * Z_NAVTEST * KAPPA) for nm, r in R_PUB.items()}
    key2i = {(str(a_), str(b_), int(c_), int(d_)): i for i, (a_, b_, c_, d_) in enumerate(zip(z["keys_log"], z["keys_token"], z["keys_step"], z["keys_rank"]))}
    for arm in ("N-s0", "N-s1", "N-s2"):
        s = np.load(f"{a.scorers}/{arm}/dump/scores_update400_B.npz")
        assert int(s["ckpt_step"][0]) == 10075
        idx = np.array([key2i[(str(a_), str(b_), int(c_), int(d_))] for a_, b_, c_, d_ in zip(s["log"], s["token"], s["step"], s["rank"])])
        assert len(set(idx.tolist())) == N
        lg = s["scores"].astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-lg[..., :6]))
        agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
        pick_s = agg.argmax(1)
        pick = np.empty(N, int); pick[idx] = pick_s                           # scorer rows -> joined order
        ar = np.arange(N)
        zp = pdms[ar, pick] == 0
        ade_pick = ade[ar, pick]
        rec = {"n_zero_picks": int(zp.sum()), "zero_pick_rate": B.ratio(zp.astype(float)), "pick_pdms_x100": B.ratio(pdms[ar, pick], scale=100)}
        for d in GRID:
            _, nhn = counts_at(ade, hn, d)
            c_pick = B.ratio((zp & (ade_pick <= d)).astype(float), zp.astype(float))
            c_set = B.ratio((zp & (nhn >= 1)).astype(float), zp.astype(float))
            rec[f"d={d}"] = {"zero_picks_that_are_expert_proximate": c_pick, "zero_pick_sets_with_proximate_HN": c_set,
                             "ceiling_PDMS_from_pick_coverage": {k: float(c_pick[k] * Z_NAVTEST * KAPPA) for k in ("point", "lo", "hi")},
                             "ceiling_PDMS_from_set_coverage": {k: float(c_set[k] * Z_NAVTEST * KAPPA) for k in ("point", "lo", "hi")}}
        R["C_reach"]["C2_zero_pick_coverage"][arm] = rec

    # ---- decision readings (PREREG s6)
    s05 = R["ET"]["A1_ADE"]["d=0.5"]["share_sets_with_proximate_HN"]
    def cls_(x): return "SUPPLY EXISTS" if x >= 0.5 else ("PARTIAL" if x >= 0.1 else "SPARSE")
    R["reading_s6"] = {"s_hn(0.5)": s05, "point_reading": cls_(s05["point"]), "lo_reading": cls_(s05["lo"]), "hi_reading": cls_(s05["hi"]),
                       "straddles_boundary": cls_(s05["lo"]) != cls_(s05["hi"]),
                       "twin_pair_share_t0.5": PB["t=0.5"]["share_sets_with_DAC_twin_pair"],
                       "twin_rule_reading": "SPARSE (<0.10)" if PB["t=0.5"]["share_sets_with_DAC_twin_pair"]["point"] < 0.10 else "PRESENT (>=0.10)"}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", a.out)

    if a.per_set:
        with open(a.per_set, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            hdr = ["log", "token", "rank", "expert_class", "n_HN_total", "nearest_ADE", "nearest_HN_ADE"] + [f"n_prox_d{d}" for d in GRID] + [f"n_HN_d{d}" for d in GRID]
            w.writerow(hdr)
            cnt = {d: counts_at(ade, hn, d) for d in GRID}
            for i in range(N):
                w.writerow([z["keys_log"][i], z["keys_token"][i], int(z["keys_rank"][i]), cls[i], int(hn[i].sum()), f"{near_all[i]:.4f}",
                            "" if not has_hn[i] else f"{nearest_hn[i]:.4f}"] + [int(cnt[d][0][i]) for d in GRID] + [int(cnt[d][1][i]) for d in GRID])
        print("wrote", a.per_set)
    return 0


if __name__ == "__main__":
    sys.exit(main())
