#!/usr/bin/env python3
"""SEL-RANK Deliverable 1: DESCRIPTIVE attribution of the oracle-minus-pick gap on P2's own 64-proposal table (full navtest, T1 NAVSIM v1.1).

CPU only. Nothing is fitted; navtest is the CONFIRMATION set (describe it, never fit on it).  Every interval is the paired LOG-CLUSTER bootstrap over the
136 navtest logs, 10,000 resamples, seed 20260927 (zero_tail.Boot, vendored byte-identical from raw/2026-10-17-tracking-execution/code/zero_tail.py).
It answers "would another draw of LOGS say this?" ONLY.  P2 is one training seed (H-ESTIM-SEED-1); its inference is deterministic.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe selrank_attr.py --out ../raw/attribution.json

Sections: G (gates), A (per-term attribution), B (scorer calibration per head), C (concentration + by driving command), D (oracle-head bounds + controls).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zero_tail import Boot, B_BOOT, SEED  # noqa: E402

TABLE = "D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz"
TABLE_SHA256_PREFIX = "33441082b7044c95"
INPUTS = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
HEADS = ("nc", "dac", "ep", "ttc", "c", "ddc")           # table order (sub[..., k], logits[..., k])
V1_W = (5.0, 5.0, 2.0)                                    # planner.py:520 (EP, TTC, C), sum 12
TIE_TOL = 1e-12


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sigmoid64(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def agg_v1_np(p):
    """planner.py:558-561 / scorer_finetune.py:85: NC*DAC*(5EP+5TTC+2C)/12 over [..., >=5] probabilities or labels."""
    return p[..., 0] * p[..., 1] * (V1_W[0] * p[..., 2] + V1_W[1] * p[..., 3] + V1_W[2] * p[..., 4]) / sum(V1_W)


def planner_pick_torch(logits32):
    """EXACTLY planner.REFePlanner.aggregate(rule navsim_v1) + _pick: float32 torch, sigmoid, argmax over M (first index on a tie)."""
    import torch
    s = torch.from_numpy(np.ascontiguousarray(logits32))
    p = s.sigmoid()
    w1 = torch.tensor(V1_W, dtype=p.dtype)
    agg = p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()
    return agg.argmax(dim=1).numpy(), agg.numpy()


# --------------------------------------------------------------------------------------------- bootstrap helpers (SAME draws as Boot)
def ratio(B, num, den=None, scale=1.0):
    num = np.asarray(num, float)
    den = np.ones_like(num) if den is None else np.asarray(den, float)
    return B.ratio(num, den, scale)


def corr_boot(B, x, y, msk=None):
    """Pooled within-token correlation: per-LOG sums of (xy, xx, yy) over already-centred values, resampled with Boot's draws."""
    n = B.n_logs
    li = B.log_idx if msk is None else B.log_idx[msk]
    sxy = np.bincount(li, weights=(x * y).sum(axis=1), minlength=n)
    sxx = np.bincount(li, weights=(x * x).sum(axis=1), minlength=n)
    syy = np.bincount(li, weights=(y * y).sum(axis=1), minlength=n)
    point = float(sxy.sum() / np.sqrt(sxx.sum() * syy.sum()))
    d = B.draws
    with np.errstate(divide="ignore", invalid="ignore"):
        v = sxy[d].sum(1) / np.sqrt(sxx[d].sum(1) * syy[d].sum(1))
    return {"point": point, "lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5)), "B": B.B}


def share_topfrac_boot(B, gap, frac, universe_mask):
    """Share of the summed gap carried by the top `frac` of tokens (by gap) INSIDE `universe_mask`; the threshold is RE-DETERMINED on every
    resample (the top decile of that resample), so the interval carries the selection of the tail as well as its mass."""
    idx = np.where(universe_mask)[0]
    li = B.log_idx[idx]
    g = gap[idx]
    order = np.argsort(li, kind="stable")
    li, g = li[order], g[order]
    bounds = np.searchsorted(li, np.arange(B.n_logs + 1))
    per_log = [g[bounds[j]:bounds[j + 1]] for j in range(B.n_logs)]

    def one(parts):
        a = np.concatenate(parts)
        k = max(1, int(np.ceil(frac * a.size)))
        tot = a.sum()
        if tot <= 0:
            return np.nan
        top = np.partition(a, a.size - k)[a.size - k:]
        return top.sum() / tot
    k0 = max(1, int(np.ceil(frac * g.size)))
    point = float(np.sort(g)[::-1][:k0].sum() / g.sum())
    vals = np.empty(B.B)
    for b in range(B.B):
        vals[b] = one([per_log[j] for j in B.draws[b]])
    return {"point": point, "lo": float(np.nanpercentile(vals, 2.5)), "hi": float(np.nanpercentile(vals, 97.5)), "B": B.B,
            "k_tokens_point": int(k0), "universe_tokens": int(g.size)}


# --------------------------------------------------------------------------------------------- A: Shapley decomposition
def shapley_terms(sub, pick, tiew):
    """Per-token, per-term decomposition of  PDMS(oracle) - PDMS(pick)  with PDMS = NC*DAC*W, W = (5EP+5TTC+2C)/12.
    EXACT (sums to the PDMS difference): Shapley value over the three multiplicative factors (NC, DAC, W) of switching each from its value at the
    pick to its value at the oracle proposal (6 orderings), then W's share is split over EP/TTC/C by W's own LINEAR weights (5, 5, 2)/12.
    Averaged over the set of proposals TIED at the oracle PDMS (tiew [N,64] weights, rows sum to 1).  Returns [N, 5] in order nc, dac, ep, ttc, c."""
    N, M, _ = sub.shape
    ar = np.arange(N)
    s0 = sub[ar, pick][:, None, :]                       # [N,1,6] the pick
    s1 = sub                                             # [N,M,6] every candidate oracle
    a0, b0 = s0[..., 0], s0[..., 1]
    a1, b1 = s1[..., 0], s1[..., 1]
    w0 = (V1_W[0] * s0[..., 2] + V1_W[1] * s0[..., 3] + V1_W[2] * s0[..., 4]) / 12.0
    w1 = (V1_W[0] * s1[..., 2] + V1_W[1] * s1[..., 3] + V1_W[2] * s1[..., 4]) / 12.0
    phi_a = (a1 - a0) * (b0 * w0 / 3 + b1 * w0 / 6 + b0 * w1 / 6 + b1 * w1 / 3)
    phi_b = (b1 - b0) * (a0 * w0 / 3 + a1 * w0 / 6 + a0 * w1 / 6 + a1 * w1 / 3)
    coef = (a0 * b0 / 3 + a1 * b0 / 6 + a0 * b1 / 6 + a1 * b1 / 3)            # phi_w = coef * (w1 - w0)
    out = np.empty((N, M, 5))
    out[..., 0], out[..., 1] = phi_a, phi_b
    for k, (col, wt) in enumerate(((2, V1_W[0]), (3, V1_W[1]), (4, V1_W[2]))):
        out[..., 2 + k] = coef * wt * (s1[..., col] - s0[..., col]) / 12.0
    tot_chk = (a1 * b1 * w1 - a0 * b0 * w0)
    err = float(np.abs(out.sum(-1) - tot_chk).max())
    contrib = (tiew[..., None] * out).sum(1)             # [N,5]
    return contrib, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raw", "attribution.json"))
    ap.add_argument("--no-torch", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    R = {"_what": "SEL-RANK Deliverable 1: descriptive attribution on P2's own full-navtest 64-proposal table. MEASURED (descriptive), T1 NAVSIM v1.1. Nothing fitted.",
         "estimator": {"name": "paired log-cluster bootstrap (zero_tail.Boot)", "B": B_BOOT, "seed": SEED,
                       "answers": "another draw of LOGS only; NOT another training run (H-ESTIM-SEED-1: P2 is ONE seed); NOT another inference run (deterministic argmax)"}}
    sha = sha256_file(TABLE)
    assert sha.startswith(TABLE_SHA256_PREFIX), sha
    R["table"] = {"path": TABLE, "sha256": sha}
    z = np.load(TABLE, allow_pickle=False)
    token, log = z["token"], z["log"]
    sub, pdms, logits, pick = z["sub"].astype(np.float64), z["pdms"].astype(np.float64), z["logits"], z["pick"].astype(np.int64)
    N, M = pdms.shape
    assert (N, M) == (12146, 64) and sub.shape == (N, M, 6) and logits.shape == (N, M, 6) and bool(z["valid"].all())
    ulogs, log_idx = np.unique(log, return_inverse=True)
    n_logs = len(ulogs)
    assert n_logs == 136
    B = Boot(log_idx, n_logs)
    ar = np.arange(N)

    # ------------------------------------------------------------------------------------------- G: gates
    G = {}
    f1 = agg_v1_np(sub)
    G["pdms_equals_NC_DAC_W12_max_abs_diff"] = float(np.abs(f1 - pdms).max())          # the NAVSIM v1 formula, checked against the table's own PDMS
    G["pdms_with_ddc_in_product_max_abs_diff"] = float(np.abs(f1 * sub[..., 5] - pdms).max())   # control: DDC in the product does NOT reproduce it
    if not a.no_torch:
        pk_t, agg_t = planner_pick_torch(logits)
        G["pick_equals_planner_rule_torch_f32"] = {"match": int((pk_t == pick).sum()), "of": N}
        if (pk_t != pick).any():
            bad = np.where(pk_t != pick)[0]
            G["pick_mismatch_examples"] = [{"token": str(token[i]), "table_pick": int(pick[i]), "recomputed": int(pk_t[i]),
                                            "score_gap": float(abs(agg_t[i, pick[i]] - agg_t[i, pk_t[i]]))} for i in bad[:10]]
        score = agg_t.astype(np.float64)
    else:
        score = agg_v1_np(sigmoid64(logits))
    G["pick_equals_float64_numpy_rule"] = int((agg_v1_np(sigmoid64(logits)).argmax(1) == pick).sum())
    G["pdms_at_pick_equals_banked_max_abs_diff"] = float(np.abs(pdms[ar, pick] - z["pdms_at_pick_banked"]).max())
    pick_pdms, orc_pdms = pdms[ar, pick], pdms.max(1)
    G["mean_pdms_x100"] = {"pick": float(pick_pdms.mean() * 100), "oracle": float(orc_pdms.mean() * 100), "random_selector": float(pdms.mean() * 100)}
    G["n_tokens"], G["n_logs"] = N, n_logs
    R["G_gates"] = G

    gap = orc_pdms - pick_pdms                                         # >= 0 by construction
    nz = pick_pdms > 0                                                 # non-zero pick
    S = nz & (gap > TIE_TOL)                                           # the decision pool: non-zero pick AND the oracle strictly beats it
    R["pool"] = {"n_tokens": N, "n_pick_nonzero": int(nz.sum()), "n_pick_zero": int((~nz).sum()), "n_oracle_beats_pick_all": int((gap > TIE_TOL).sum()),
                 "n_pool_S_nonzero_pick_and_oracle_beats": int(S.sum()),
                 "gap_x100_all": ratio(B, gap, scale=100), "gap_x100_from_nonzero_picks": ratio(B, gap * nz, scale=100),
                 "gap_x100_from_zero_picks": ratio(B, gap * ~nz, scale=100)}
    ties = np.abs(pdms - orc_pdms[:, None]) <= TIE_TOL
    tiew = ties / ties.sum(1, keepdims=True)
    R["pool"]["oracle_tied_proposals_per_token"] = {"mean": float(ties.sum(1).mean()), "median": float(np.median(ties.sum(1))), "max": int(ties.sum(1).max())}

    # ------------------------------------------------------------------------------------------- A: per-term attribution
    contrib, err = shapley_terms(sub, pick, tiew)
    A = {"method": "Shapley over the three multiplicative factors (NC, DAC, W) of the move pick -> oracle proposal (6 orderings), W split over EP/TTC/C by its own linear "
                   "weights 5/5/2 over 12; averaged over proposals tied at the oracle PDMS. EXACT: the five terms sum to oracle - pick token by token.",
         "formula_source": "NAVSIM v1 PDMS = NC*DAC*(5EP+5TTC+2C)/12, DDC at weight 0: repo-cited navsim @ 3e8291b pdm_scorer.py:38-42 (refe/planner.py:520 and :558-561; "
                           "refe/scorer_finetune.py:85-87; eval/rule_mismatch_diag.py:7). The vendored NAVSIM source is NOT on this box, so the file:line is INHERITED by citation; "
                           "the formula itself is MEASURED here: it reproduces the table's pdms on all 12,146 x 64 entries with max |diff| in G.",
         "max_abs_sum_error_vs_pdms_difference": err}
    cn = np.where((gap > TIE_TOL)[:, None], contrib, 0.0)
    assert np.abs(cn.sum(1) - gap).max() < 1e-9, np.abs(cn.sum(1) - gap).max()
    names5 = ("nc", "dac", "ep", "ttc", "c")
    A["pool_S_gap_by_term_x100_over_all_12146_tokens"] = {"TOTAL_from_nonzero_picks": ratio(B, gap * nz, scale=100)}
    for k, nm in enumerate(names5):
        A["pool_S_gap_by_term_x100_over_all_12146_tokens"][nm] = ratio(B, cn[:, k] * nz, scale=100)
    A["all_tokens_gap_by_term_x100"] = {"TOTAL": ratio(B, gap, scale=100), **{nm: ratio(B, cn[:, k], scale=100) for k, nm in enumerate(names5)}}
    tot5 = (gap * nz).sum()
    A["pool_S_share_of_gap_by_term"] = {nm: float((cn[:, k] * nz).sum() / tot5) for k, nm in enumerate(names5)}
    # who wins: per term, tie-weighted fraction of pool tokens where the oracle proposal is strictly better / worse than the pick
    s_pick = sub[ar, pick]
    A["pool_S_oracle_vs_pick_by_term"] = {}
    for k, nm in enumerate(names5):
        d = sub[..., k] - s_pick[:, k][:, None]                           # [N,M]
        better = (tiew * (d > TIE_TOL)).sum(1)
        worse = (tiew * (d < -TIE_TOL)).sum(1)
        A["pool_S_oracle_vs_pick_by_term"][nm] = {
            "oracle_better_frac_of_pool": ratio(B, better * S, S.astype(float)), "oracle_worse_frac_of_pool": ratio(B, worse * S, S.astype(float)),
            "mean_value_at_pick": ratio(B, s_pick[:, k] * S, S.astype(float)), "mean_value_at_oracle": ratio(B, (tiew * sub[..., k]).sum(1) * S, S.astype(float))}
    dom = np.argmax(cn, axis=1)                                          # the term that carries most of the token's gap
    A["pool_S_dominant_term_of_token"] = {
        nm: {"n_tokens": int(((dom == k) & S).sum()), "share_of_pool_tokens": float(((dom == k) & S).sum() / S.sum()),
             "gap_x100": ratio(B, gap * ((dom == k) & S), scale=100)} for k, nm in enumerate(names5)}
    R["A_attribution"] = A

    # ------------------------------------------------------------------------------------------- B: scorer calibration per head
    P = sigmoid64(logits)                                                # [N,M,6] the heads' probabilities
    sel_rank = rankdata(-score, axis=1, method="ordinal")                # 1 = the model's own best
    topk = sel_rank <= 8                                                 # the 8 proposals the scorer ranks highest ("contenders"); k fixed in advance
    Bc = {"heads_in_selection": ["nc", "dac", "ep", "ttc", "c"], "ddc_head_not_in_sel_score": True, "per_head": {}}
    univ = {"all_tokens": np.ones(N, bool), "nonzero_pick_tokens": nz, "pool_S": S}
    for h in ("ep", "ttc", "c", "nc", "dac"):
        k = HEADS.index(h)
        x, y = P[..., k], sub[..., k]
        rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)                # average ranks (ties)
        cx, cy = rx - rx.mean(1, keepdims=True), ry - ry.mean(1, keepdims=True)
        den = np.sqrt((cx ** 2).sum(1) * (cy ** 2).sum(1))
        defined = den > 0
        rho = np.where(defined, (cx * cy).sum(1) / np.where(defined, den, 1.0), 0.0)
        vx, vy = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
        rec = {"realised_varies_within_token_frac": float((sub[..., k].max(1) > sub[..., k].min(1)).mean()),
               "predicted_varies_within_token_frac": float((x.max(1) > x.min(1)).mean())}
        for un, msk in univ.items():
            m = defined & msk
            rec[un] = {"n_tokens_defined": int(m.sum()),
                       "mean_per_token_spearman": ratio(B, rho * m, m.astype(float)),
                       "pooled_within_token_rank_corr": corr_boot(B, cx[msk], cy[msk], msk),
                       "pooled_within_token_value_corr_pearson": corr_boot(B, vx[msk], vy[msk], msk),
                       "mean_prob_minus_mean_realised_over_64": ratio(B, (x.mean(1) - y.mean(1)) * msk, msk.astype(float))}
        # contenders only: Spearman of head vs realised inside the model's own top-8
        rho8 = np.full(N, np.nan)
        for i in range(N):
            idx = np.where(topk[i])[0]
            xi, yi = x[i, idx], y[i, idx]
            if xi.std() > 0 and yi.std() > 0:
                rho8[i] = np.corrcoef(rankdata(xi), rankdata(yi))[0, 1]
        d8 = ~np.isnan(rho8)
        for un, msk in (("all_tokens", np.ones(N, bool)), ("pool_S", S)):
            m = d8 & msk
            rec[f"top8_contenders_{un}"] = {"n_tokens_defined": int(m.sum()), "mean_per_token_spearman": ratio(B, np.nan_to_num(rho8) * m, m.astype(float))}
        Bc["per_head"][h] = rec
    # the DECISIVE pair: on pool tokens, does head h order (oracle, pick) the way the realised sub-score does?
    dec = {}
    for h in ("ep", "ttc", "c", "nc", "dac"):
        k = HEADS.index(h)
        dy = sub[..., k] - s_pick[:, k][:, None]                         # realised: oracle candidate minus pick
        dp = P[..., k] - P[ar, pick, k][:, None]                          # predicted: same
        w_better = tiew * (dy > TIE_TOL)                                  # tie-weighted: the oracle is truly better on this term
        right = (w_better * (dp > 0)).sum(1)
        wrong = (w_better * (dp < 0)).sum(1)
        flat = (w_better * (dp == 0)).sum(1)
        n_b = w_better.sum(1)
        dec[h] = {"pool_tokens_where_oracle_better_on_term": float((n_b * S).sum()),
                  "head_orders_pair_correctly_frac": ratio(B, right * S, n_b * S), "head_orders_pair_WRONG_frac": ratio(B, wrong * S, n_b * S),
                  "head_ties_pair_frac": ratio(B, flat * S, n_b * S),
                  "mean_prob_margin_for_pick_over_oracle_when_wrong": float(np.nanmean(np.where(((w_better * (dp < 0)).sum(1) > 0) & S, -(w_better * dp).sum(1) / np.where(n_b > 0, n_b, 1), np.nan)))}
    Bc["decisive_pair_pick_vs_oracle_on_pool_S"] = dec
    # ERROR BUDGET of the decisive pair.  For the SAME pick -> oracle move, decompose (a) the scorer's PREDICTED aggregate difference (oracle - pick; <= 0 because the pick is the
    # argmax) and (b) the REALISED PDMS difference (>= 0), with the SAME Shapley terms.  error_h = realised_h - predicted_h is the amount head h's prediction mis-states the
    # pick->oracle move on the aggregate's own scale; the head with the largest error_h is where the pick-vs-oracle inversion is manufactured.
    P5 = P[..., :5]
    pred_sub = np.concatenate([P5, np.ones((N, M, 1))], axis=-1)           # shapley_terms reads columns 0-4 only
    pred_contrib, perr = shapley_terms(pred_sub, pick, tiew)
    eb = {"note": "pool S tokens only; units = aggregate points x100 averaged over ALL 12,146 tokens (same denominator as the 5.88); predicted terms are Shapley terms of "
                  "NC_hat*DAC_hat*(5EP_hat+5TTC_hat+2C_hat)/12 on the move pick->oracle, realised terms are the section-A terms. "
                  "CAUTION: the predicted terms are conditioned on the pick being the scorer's ARGMAX (winner's curse: any noisy scorer over-states its own pick), so 'error' and its share "
                  "are NOT a head ranking; only frac_pool_tokens_head_inverts_pair is quoted, and the counterfactual section D is the head ranking.", "shapley_sum_err_predicted": perr, "per_head": {}}
    pool_pred_total = (pred_contrib.sum(1) * S)
    eb["predicted_aggregate_margin_pick_over_oracle_x100"] = ratio(B, -pool_pred_total, scale=100)
    eb["realised_gap_x100"] = ratio(B, gap * S, scale=100)
    for k, nm in enumerate(names5):
        eb["per_head"][nm] = {"realised_term_x100": ratio(B, cn[:, k] * S, scale=100), "predicted_term_x100": ratio(B, pred_contrib[:, k] * S, scale=100),
                              "error_realised_minus_predicted_x100": ratio(B, (cn[:, k] - pred_contrib[:, k]) * S, scale=100),
                              "share_of_total_error": ratio(B, (cn[:, k] - pred_contrib[:, k]) * S, (gap - pred_contrib.sum(1)) * S),
                              "frac_pool_tokens_head_inverts_pair": ratio(B, ((cn[:, k] > TIE_TOL) & (pred_contrib[:, k] < -TIE_TOL)) * S, S.astype(float))}
    Bc["decisive_pair_error_budget_pool_S"] = eb
    R["B_calibration"] = Bc

    # ------------------------------------------------------------------------------------------- C: concentration + by driving command
    C = {}
    Cn = {"pool_total_x100": ratio(B, gap * nz, scale=100)}
    for frac in (0.01, 0.05, 0.10, 0.20, 0.50):
        Cn[f"top_{int(frac * 100)}pct_of_nonzero_pick_tokens_share_of_pool"] = share_topfrac_boot(B, gap * nz, frac, nz)
    Cn["top_10pct_of_ALL_12146_tokens_share_of_pool"] = share_topfrac_boot(B, gap * nz, 0.10, np.ones(N, bool))
    Cn["top_10pct_gap_threshold_pdms_points"] = float(np.sort((gap * nz)[nz])[::-1][int(np.ceil(0.1 * nz.sum())) - 1])
    gs = np.sort((gap * nz)[nz])[::-1]
    cs = np.cumsum(gs) / gs.sum()
    Cn["tokens_to_cover_50pct_of_pool"] = int(np.searchsorted(cs, 0.5) + 1)
    Cn["tokens_to_cover_80pct_of_pool"] = int(np.searchsorted(cs, 0.8) + 1)
    Cn["gap_quantiles_over_nonzero_pick_tokens_pdms_points"] = {q: float(np.quantile(gap[nz], q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)}
    Cn["n_nonzero_pick_tokens_with_zero_gap"] = int((nz & (gap <= TIE_TOL)).sum())
    C["token_concentration"] = Cn
    perlog = np.bincount(log_idx, weights=gap * nz, minlength=n_logs) / N * 100
    o = np.argsort(-perlog)
    C["log_concentration"] = {"top10_logs_share_of_pool": float(perlog[o[:10]].sum() / perlog.sum()), "n_logs_to_cover_50pct": int(np.searchsorted(np.cumsum(perlog[o]) / perlog.sum(), 0.5) + 1),
                              "n_logs_with_any_pool_gap": int((perlog > 0).sum()),
                              "tokens_share_of_top10_gap_logs": float(np.bincount(log_idx, minlength=n_logs)[o[:10]].sum() / N),
                              "top10": [{"log": str(ulogs[j]), "pool_pts_x100_over_all_tokens": float(perlog[j]), "tokens": int((log_idx == j).sum())} for j in o[:10]]}
    # random-assignment null for the token-concentration: if the same gap values were spread over logs at random (log sizes fixed), what top-10-logs share?
    rng = np.random.default_rng(SEED)
    sizes = np.bincount(log_idx, minlength=n_logs)
    gz = (gap * nz)
    tops = np.empty(2000)
    for b in range(2000):
        perm = rng.permutation(N)
        pl = np.bincount(log_idx, weights=gz[perm], minlength=n_logs)
        tops[b] = np.sort(pl)[::-1][:10].sum() / pl.sum()
    C["log_concentration"]["permutation_null_top10_logs_share"] = {"median": float(np.median(tops)), "lo": float(np.percentile(tops, 2.5)), "hi": float(np.percentile(tops, 97.5)), "n_perm": 2000}
    # by driving command
    doc = json.load(gzip.open(INPUTS, "rt", encoding="utf-8"))["tokens"]
    cmd_vec = np.array([doc[str(t)]["ego_statuses"][-1]["driving_command"] for t in token], float)
    cmd = np.argmax(cmd_vec, axis=1)
    assert (cmd_vec.sum(1) == 1).all()
    names_cmd = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")
    byc = {}
    for c_i, c_nm in enumerate(names_cmd):
        m = cmd == c_i
        if not m.any():
            byc[c_nm] = {"n_tokens": 0}
            continue
        mf = m.astype(float)
        byc[c_nm] = {"n_tokens": int(m.sum()), "token_share": float(m.mean()),
                     "pick_pdms_x100": ratio(B, pick_pdms * m, mf, 100), "oracle_pdms_x100": ratio(B, orc_pdms * m, mf, 100),
                     "mean_gap_x100_per_token_nonzero_picks": ratio(B, gap * nz * m, mf, 100),
                     "pool_contribution_x100_over_all_tokens": ratio(B, gap * nz * m, scale=100),
                     "share_of_pool": ratio(B, gap * nz * m, gap * nz),
                     "zero_pick_rate": ratio(B, (~nz) * m, mf),
                     "gap_by_term_x100_per_token": {nm: ratio(B, cn[:, k] * nz * m, mf, 100) for k, nm in enumerate(names5)}}
    C["by_driving_command"] = {"source": INPUTS + " :: tokens[t].ego_statuses[-1].driving_command (one-hot LEFT, STRAIGHT, RIGHT, UNKNOWN; refe/nav_command.py:20-21)", "classes": byc}
    R["C_concentration"] = C

    # ------------------------------------------------------------------------------------------- D: oracle-head bounds
    D = {"definition": "at selection, replace the named head(s)' PREDICTED probability by its REALISED sub-score (same token, same 64 proposals), keep every other head as the "
                       "scorer predicted it, apply the planner's own rule NC*DAC*(5EP+5TTC+2C)/12 and take the argmax (first index on a tie, as torch.argmax in the planner). "
                       "Gain = mean PDMS at that pick minus mean PDMS at P2's pick. An UPPER bound for 'a perfect <head>', not an estimate of any trainable arm."}
    base_p = P[..., :5].copy()

    def sel_with(replace, perm=False):
        pm = base_p.copy()
        for h in replace:
            k = HEADS.index(h)
            if perm:                                                     # NEGATIVE CONTROL: realised values, but shuffled across proposals within each token
                r = np.random.default_rng(SEED + k)
                v = sub[..., k].copy()
                for i in range(N):
                    v[i] = v[i, r.permutation(M)]
                pm[..., k] = v
            else:
                pm[..., k] = sub[..., k]
        sc = agg_v1_np(pm)
        k_ = sc.argmax(1)
        top = sc[ar, k_]
        n_tie = int(((np.abs(sc - top[:, None]) <= 0).sum(1) > 1).sum())
        return k_, n_tie

    def gain_rec(replace, perm=False):
        k_, n_tie = sel_with(replace, perm)
        pd = pdms[ar, k_]
        d = pd - pick_pdms
        return {"replaced": list(replace), "pdms_at_selection_x100": ratio(B, pd, scale=100), "gain_vs_pick_x100": ratio(B, d, scale=100),
                "gain_from_nonzero_pick_tokens_x100": ratio(B, d * nz, scale=100), "gain_from_zero_pick_tokens_x100": ratio(B, d * ~nz, scale=100),
                "share_of_oracle_minus_pick_gap_captured": ratio(B, d, gap), "share_of_pool_5p88_captured": ratio(B, d * nz, gap * nz),
                "tokens_improved": int((d > TIE_TOL).sum()), "tokens_worsened": int((d < -TIE_TOL).sum()), "n_tokens_with_exact_score_tie_at_max": n_tie}
    ctrl = {}
    ctrl["identity_replace_no_head_gain_must_be_exactly_0"] = gain_rec(())
    kk, _ = sel_with(())
    ctrl["identity_pick_equals_table_pick"] = {"match": int((kk == pick).sum()), "of": N, "note": "float64 numpy recompute vs the table's float32 torch pick"}
    # the full oracle: all five heads realised -> must equal best-of-64
    ctrl["all_five_realised_must_equal_best_of_64"] = gain_rec(names5)
    ctrl["all_five_realised_pdms_minus_oracle_max_abs_per_token"] = float(np.abs(pdms[ar, sel_with(names5)[0]] - orc_pdms).max())
    ctrl["negative_control_EP_realised_but_shuffled_within_token"] = gain_rec(("ep",), perm=True)
    D["controls"] = ctrl
    arms = {}
    for nm, rep in (("oracle_EP", ("ep",)), ("oracle_TTC", ("ttc",)), ("oracle_C", ("c",)), ("oracle_EP_TTC", ("ep", "ttc")), ("oracle_EP_TTC_C", ("ep", "ttc", "c")),
                    ("oracle_NC", ("nc",)), ("oracle_DAC", ("dac",)), ("oracle_NC_DAC", ("nc", "dac")), ("oracle_NC_DAC_EP", ("nc", "dac", "ep")),
                    ("oracle_NC_DAC_TTC", ("nc", "dac", "ttc"))):
        arms[nm] = gain_rec(rep)
    # SENSITIVITY: realised EP is EXACTLY 0 on every proposal with NC=0 or DAC=0 (MEASURED: 100 % of 2,413,xxx such proposals) and is normalised by the best non-violating
    # proposal, so `oracle_EP` also hands the scorer a violation oracle.  This variant keeps the PREDICTED EP on every realised-violating proposal and uses the realised EP only
    # where NC*DAC != 0.  It is NOT violation-free information either (which proposals are replaced is itself informed), but it removes the zero-EP signal.
    pm = base_p.copy()
    viol = (sub[..., 0] == 0) | (sub[..., 1] == 0)
    pm[..., 2] = np.where(viol, base_p[..., 2], sub[..., 2])
    sc = agg_v1_np(pm)
    k_ = sc.argmax(1)
    d_ = pdms[ar, k_] - pick_pdms
    arms["oracle_EP_realised_only_on_nonviolating_proposals"] = {
        "replaced": ["ep (realised only where NC*DAC != 0)"], "pdms_at_selection_x100": ratio(B, pdms[ar, k_], scale=100), "gain_vs_pick_x100": ratio(B, d_, scale=100),
        "gain_from_nonzero_pick_tokens_x100": ratio(B, d_ * nz, scale=100), "gain_from_zero_pick_tokens_x100": ratio(B, d_ * ~nz, scale=100),
        "share_of_pool_5p88_captured": ratio(B, d_ * nz, gap * nz), "tokens_improved": int((d_ > TIE_TOL).sum()), "tokens_worsened": int((d_ < -TIE_TOL).sum())}
    # same sensitivity for TTC: realised TTC is 0 on 97 % of NC=0 proposals (a collision implies a TTC failure), so oracle_TTC partly carries the NC outcome
    pm = base_p.copy()
    pm[..., 3] = np.where(viol, base_p[..., 3], sub[..., 3])
    k_ = agg_v1_np(pm).argmax(1)
    d_ = pdms[ar, k_] - pick_pdms
    arms["oracle_TTC_realised_only_on_nonviolating_proposals"] = {
        "replaced": ["ttc (realised only where NC*DAC != 0)"], "pdms_at_selection_x100": ratio(B, pdms[ar, k_], scale=100), "gain_vs_pick_x100": ratio(B, d_, scale=100),
        "gain_from_nonzero_pick_tokens_x100": ratio(B, d_ * nz, scale=100), "gain_from_zero_pick_tokens_x100": ratio(B, d_ * ~nz, scale=100),
        "share_of_pool_5p88_captured": ratio(B, d_ * nz, gap * nz), "tokens_improved": int((d_ > TIE_TOL).sum()), "tokens_worsened": int((d_ < -TIE_TOL).sum())}
    D["realised_TTC_vs_violations"] = {"p_ttc0_given_nc0": float((sub[..., 3][sub[..., 0] == 0] == 0).mean()), "p_ttc0_given_dac0": float((sub[..., 3][sub[..., 1] == 0] == 0).mean()),
                                       "p_nc0_given_ttc0": float((sub[..., 0][sub[..., 3] == 0] == 0).mean()), "p_dac0_given_ttc0": float((sub[..., 1][sub[..., 3] == 0] == 0).mean()),
                                       "ttc0_among_nonviolating_proposals": float((sub[..., 3][~viol] == 0).mean())}
    D["realised_EP_zero_on_violating_proposals"] = {"n_violating_proposals": int(viol.sum()), "ep_zero_among_violating": float((sub[..., 2][viol] == 0).mean()),
                                                   "ep_zero_among_nonviolating": float((sub[..., 2][~viol] == 0).mean()),
                                                   "tokens_with_max_nonviolating_EP_eq_1": float((np.where(~viol, sub[..., 2], -1).max(1) > 0.9999).mean())}
    # PAIRED difference of the two head bounds (same tokens, same draws): is a perfect TTC head worth more than a perfect EP head?
    def sel_gain(replace, valid_only=False):
        pm = base_p.copy()
        for h in replace:
            k = HEADS.index(h)
            pm[..., k] = np.where(viol, base_p[..., k], sub[..., k]) if valid_only else sub[..., k]
        k_ = agg_v1_np(pm).argmax(1)
        return k_, pdms[ar, k_] - pick_pdms
    kT, dT = sel_gain(("ttc",)); kE, dE = sel_gain(("ep",)); kV, dV = sel_gain(("ep",), True); kET, dET = sel_gain(("ep", "ttc"))
    D["paired_TTC_minus_EP_bound_x100"] = {"total": ratio(B, dT - dE, scale=100), "pool_nonzero_picks": ratio(B, (dT - dE) * nz, scale=100),
                                           "pool_nonzero_picks_vs_EP_nonviolating_only": ratio(B, (dT - dV) * nz, scale=100)}
    # WHERE does the realised gain of each oracle-head selection come from?  Shapley terms of pick -> new pick, one-hot tie weights, non-zero-pick tokens
    D["gain_by_term_of_each_oracle_selection_pool_x100_over_all_tokens"] = {}
    for nm, kk_ in (("oracle_TTC", kT), ("oracle_EP", kE), ("oracle_EP_nonviolating_only", kV), ("oracle_EP_TTC", kET)):
        oh = np.zeros((N, M)); oh[ar, kk_] = 1.0
        cc, _ = shapley_terms(sub, pick, oh)
        D["gain_by_term_of_each_oracle_selection_pool_x100_over_all_tokens"][nm] = {t: ratio(B, cc[:, j] * nz, scale=100) for j, t in enumerate(names5)}
    # REACH vs HEAD QUALITY: move the head a fraction lam of the way from its prediction to the realised value (probability space); lam = 0 is P2, lam = 1 the oracle-head bound
    curve = {}
    for nm, rep, vo in (("TTC", ("ttc",), False), ("EP_all", ("ep",), False), ("EP_nonviolating_only", ("ep",), True), ("EP_and_TTC", ("ep", "ttc"), False)):
        curve[nm] = {}
        for lam in (0.1, 0.25, 0.5, 0.75, 1.0):
            pm = base_p.copy()
            for h in rep:
                k = HEADS.index(h)
                tgt = np.where(viol, base_p[..., k], sub[..., k]) if vo else sub[..., k]
                pm[..., k] = (1 - lam) * base_p[..., k] + lam * tgt
            k_ = agg_v1_np(pm).argmax(1)
            d_ = pdms[ar, k_] - pick_pdms
            curve[nm][str(lam)] = {"pool_gain_x100": ratio(B, d_ * nz, scale=100), "pool_share": ratio(B, d_ * nz, gap * nz), "total_gain_x100": ratio(B, d_, scale=100)}
    D["reach_vs_head_quality_blend"] = {"definition": "p_head' = (1 - lam) * P2's predicted probability + lam * the realised sub-score; PDMS at argmax of the planner's rule. lam is an explicit, "
                                                      "ASSUMED model of 'how far a retrained head moves toward the truth'; it is not an estimate of any arm.", "curves": curve}
    # PAIRWISE concordance of each head INSIDE the scorer's own top-8 contenders (pairs whose realised values differ; EP needs |dEP| > 0.05)
    conc = {}
    for h, thr in (("ttc", 0.0), ("ep", 0.05), ("nc", 0.0), ("dac", 0.0)):
        k = HEADS.index(h)
        num = np.zeros(N); den = np.zeros(N)
        for i in range(N):
            idx = np.where(topk[i])[0]
            y_, x_ = sub[i, idx, k], P[i, idx, k]
            dy = y_[:, None] - y_[None, :]
            m_ = dy > thr
            den[i] = m_.sum()
            dx = x_[:, None] - x_[None, :]
            num[i] = (m_ & (dx > 0)).sum() + 0.5 * (m_ & (dx == 0)).sum()
        conc[h] = {"pairs_total": int(den.sum()), "tokens_with_pairs": int((den > 0).sum()), "pairwise_concordance_top8": ratio(B, num, den),
                   "pairwise_concordance_top8_pool_S": ratio(B, num * S, den * S)}
    R["B_calibration"]["top8_pairwise_concordance_per_head"] = {"note": "AUC-like: P(head scores the better proposal higher) over ordered pairs INSIDE the scorer's own top-8 by sel_score whose realised values differ "
                                                                         "(EP: by more than 0.05); 0.5 = chance, ties count half.", **conc}
    D["arms"] = arms
    R["D_oracle_head_bounds"] = D

    R["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", a.out, R["runtime_s"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
