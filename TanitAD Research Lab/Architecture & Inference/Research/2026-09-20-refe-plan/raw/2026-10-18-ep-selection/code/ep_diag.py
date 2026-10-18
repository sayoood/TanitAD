#!/usr/bin/env python3
"""EP-1 step 1: DESCRIPTIVE diagnosis of the ego progress lost at P2's pick, on the FIT table only
(P2's own navtrain held-out table: 1,705 tokens / 24 logs, P2 scorer on P2's fan, true sub-scores, logits, 8-pose plans).
Navtest is NOT read here.  Nothing is fitted.  CPU only.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 python ep_diag.py --out ../raw/diag.json

Estimator: paired log-cluster bootstrap (zero_tail.Boot), 10,000 resamples, seed 20260927 -- another draw of LOGS only (24 logs here).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import rankdata, spearmanr

from ep_common import *  # noqa: F401,F403
from ep_common import FIT, FIT_SHA, NC, DAC, EP, TTC, C, W, load, gates, sigmoid, score_terms, geometry, at, level, paired, Boot, pick_ref

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

t = load(FIT, FIT_SHA[:16])
N = len(t["token"])
sub, pdms, logit = t["sub"], t["pdms"], t["logits"].astype(np.float64)
p = sigmoid(logit)
g_term, w_term = score_terms(p)
S = g_term * w_term
pk = pick_ref(t)
assert (pk == t["pick"]).all(), "pick not reproduced"
L, P = geometry(t["proposals"])
boot = Boot(t["log_idx"], t["n_logs"])
out = {"tier": "fit-set descriptive, P2 navtrain held-out table (NOT navtest)", "estimator": "paired log-cluster bootstrap, 10,000, seed 20260927; another draw of LOGS only (24 logs)",
       "gates": gates(t, "fit")}
ar = np.arange(N)

ep_t = sub[..., EP]
nc_t, dac_t, ttc_t, c_t = sub[..., NC], sub[..., DAC], sub[..., TTC], sub[..., C]
safe = (nc_t == 1) & (dac_t == 1) & (ttc_t == 1)          # truly safe on NC, DAC, TTC
gs = (nc_t == 1) & (dac_t == 1)                            # geometry-safe (EP is exactly 0 on violators)
has_safe = safe.any(1)
out["n_tokens"] = int(N)
out["n_tokens_with_a_safe_proposal"] = int(has_safe.sum())
out["EP_exactly_zero_on_NC_or_DAC_violators"] = float((ep_t[~(nc_t == 1) | ~(dac_t == 1)] == 0).mean())
out["safe_proposals_per_token_mean"] = float(safe.sum(1).mean())

# ---------------------------------------------------------------- (a) the fan
ep_pick = at(ep_t, pk)
pd_pick = at(pdms, pk)
ep_best_safe = np.where(has_safe, np.where(safe, ep_t, -1).max(1), np.nan)
ep_best_any = ep_t.max(1)
ep_best_gs = np.where(gs.any(1), np.where(gs, ep_t, -1).max(1), np.nan)
pick_safe = at(safe, pk)
m = has_safe
A = {}
A["mean_EP_at_pick_all_tokens"] = level(boot, ep_pick)
A["mean_EP_at_pick_tokens_with_safe"] = level(boot, np.where(m, ep_pick, 0), 1.0) if False else None
num = np.where(m, ep_pick, 0.0)
den = m.astype(float)
A["EP_pick_on_tokens_with_safe"] = boot.ratio(num, den)
A["EP_best_safe_on_tokens_with_safe"] = boot.ratio(np.where(m, ep_best_safe, 0.0), den)
A["gap_EPbestsafe_minus_EPpick"] = boot.ratio(np.where(m, ep_best_safe - ep_pick, 0.0), den)
ps = m & pick_safe
pu = m & ~pick_safe
A["n_pick_safe"] = int(ps.sum())
A["n_pick_unsafe_with_safe_available"] = int(pu.sum())
A["gap_when_pick_safe"] = boot.ratio(np.where(ps, ep_best_safe - ep_pick, 0.0), ps.astype(float))
A["gap_when_pick_unsafe"] = boot.ratio(np.where(pu, ep_best_safe - ep_pick, 0.0), pu.astype(float))
# the share of the total EP gap carried by each pick class (token-mean units)
tot = (np.where(m, ep_best_safe - ep_pick, 0.0)).sum()
A["share_of_gap_from_safe_picks"] = float(np.where(ps, ep_best_safe - ep_pick, 0.0).sum() / tot)
A["share_of_gap_from_unsafe_picks"] = float(np.where(pu, ep_best_safe - ep_pick, 0.0).sum() / tot)
A["frac_tokens_best_safe_EP_ge_0.95"] = float((ep_best_safe[m] >= 0.95).mean())
A["frac_tokens_best_safe_EP_eq_1"] = float((ep_best_safe[m] >= 1 - 1e-9).mean())
A["best_safe_EP_quantiles_5_25_50_75"] = [float(x) for x in np.nanpercentile(ep_best_safe[m], [5, 25, 50, 75])]
A["EP_pick_quantiles_5_25_50_75_tokens_with_safe"] = [float(x) for x in np.percentile(ep_pick[m], [5, 25, 50, 75])]
# PDMS value of the EP gap: the safest-fastest oracle (pick the safe proposal of max EP) vs pick, and vs best-of-64
bs_idx = np.argmax(np.where(safe, ep_t + 1e-6 * pdms, -1), axis=1)
pd_bs = np.where(has_safe, at(pdms, bs_idx), pd_pick)
A["PDMS_pick_x100"] = level(boot, pd_pick)
A["PDMS_oracle_x100"] = level(boot, pdms.max(1))
A["PDMS_best_safe_EP_proposal_x100"] = level(boot, pd_bs)
A["gain_best_safe_EP_proposal_minus_pick_x100"] = paired(boot, pd_bs, pd_pick)
A["gain_oracle_minus_pick_x100"] = paired(boot, pdms.max(1), pd_pick)
# pick favours short plans
Lr = (rankdata(L, axis=1, method="average") - 1) / 63.0     # 0 = shortest in token, 1 = longest
out_rank = {}
out_rank["pick_length_percentile_mean"] = float(at(Lr, pk).mean())
out_rank["pick_length_percentile_median"] = float(np.median(at(Lr, pk)))
out_rank["frac_picks_in_shortest_quintile"] = float((at(Lr, pk) <= 0.2).mean())
bi = np.argmax(pdms, axis=1)
out_rank["oracle_pdms_proposal_length_percentile_mean"] = float(at(Lr, bi).mean())
out_rank["best_safe_EP_proposal_length_percentile_mean"] = float(at(Lr, bs_idx)[has_safe].mean())
out_rank["pick_L_m_mean"] = float(at(L, pk).mean())
out_rank["best_safe_EP_proposal_L_m_mean"] = float(at(L, bs_idx)[has_safe].mean())
out_rank["pick_L_m_minus_best_safe_L_m_mean"] = float((at(L, pk) - at(L, bs_idx))[has_safe].mean())
out["a_fan"] = A
out["a_fan_rank"] = out_rank

# ---------------------------------------------------------------- (b) EP-head calibration
pEP = p[..., EP]
def bin_table(key, nb, mask, label):
    """rows: bins of `key` over proposals in `mask`; pEP vs true EP, plus the head's NC*DAC prob vs the realised NC*DAC rate."""
    edges = np.percentile(key[mask], np.linspace(0, 100, nb + 1))
    edges[-1] += 1e-9
    rows = []
    for b in range(nb):
        sel = mask & (key >= edges[b]) & (key < edges[b + 1])
        if sel.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(sel.sum()), f"{label}_mean": float(key[sel].mean()),
                     "pEP_mean": float(pEP[sel].mean()), "EP_true_mean": float(ep_t[sel].mean()),
                     "pEP_minus_EP": float((pEP[sel] - ep_t[sel]).mean()),
                     "pNCxDAC_mean": float(g_term[sel].mean()), "NCxDAC_true_rate": float((nc_t[sel] * dac_t[sel]).mean())})
    return rows
allm = np.ones_like(gs, dtype=bool)
B = {}
B["all_proposals_by_global_L_decile"] = bin_table(L, 10, allm, "L_m")
B["geometry_safe_proposals_by_global_L_decile"] = bin_table(L, 10, gs, "L_m")
B["geometry_safe_proposals_by_within_token_length_quintile"] = bin_table(Lr, 5, gs, "len_pct")
topk = np.zeros_like(gs)
order = np.argsort(-S, axis=1)[:, :8]
topk[ar[:, None], order] = True
B["top8_by_score_and_geometry_safe_by_within_token_length_tercile"] = bin_table(Lr, 3, gs & topk, "len_pct")
# rank agreement of the EP head with truth and with length, on geometry-safe proposals, within token
def rho(x, y, mask):
    out_ = []
    for i in range(N):
        mk = mask[i]
        if mk.sum() >= 4 and np.ptp(x[i][mk]) > 0 and np.ptp(y[i][mk]) > 0:
            out_.append(spearmanr(x[i][mk], y[i][mk])[0])
    return float(np.mean(out_)), len(out_)
B["within_token_spearman_pEP_vs_EPtrue_geomsafe"] = rho(pEP, ep_t, gs)
B["within_token_spearman_pEP_vs_L_geomsafe"] = rho(pEP, L, gs)
B["within_token_spearman_EPtrue_vs_L_geomsafe"] = rho(ep_t, L, gs)
# long-plan under-prediction: pEP - EP among geometry-safe proposals, shortest vs longest within-token quintile (paired by token is not defined; pooled)
q = Lr
for nm, lo, hi in [("shortest_quintile", 0, 0.2), ("middle", 0.2, 0.8), ("longest_quintile", 0.8, 1.0001)]:
    sel = gs & (q >= lo) & (q < hi)
    d = np.where(sel, pEP - ep_t, 0.0)
    cnt = sel.sum(1).astype(float)
    B[f"bias_pEP_minus_EP_geomsafe_{nm}"] = boot.ratio(d.sum(1), cnt)
# does the head compress the progress spread? slope of EP_true on pEP within token, geomsafe (isotonic-free)
sl = []
for i in range(N):
    mk = gs[i]
    if mk.sum() >= 6 and np.ptp(pEP[i][mk]) > 1e-6:
        x = pEP[i][mk] - pEP[i][mk].mean()
        y = ep_t[i][mk] - ep_t[i][mk].mean()
        sl.append((x * y).sum() / (x * x).sum())
B["within_token_slope_EPtrue_on_pEP_geomsafe_mean"] = float(np.mean(sl))
B["within_token_slope_n"] = len(sl)
out["b_ep_head"] = B


# ---------------------------------------------------------------- (b2) is the EP head an UNCONDITIONAL expectation (violators scored 0)?
# If so pEP already contains the violation risk and the rule's pNC*pDAC factor counts it a second time.
pEPc = np.minimum(1.0, pEP / np.maximum(g_term, 1e-9))     # the head's EP with its own violation probability divided out
B2 = {}
for nm, mk_ in [("geometry_safe", gs), ("all_proposals", allm)]:
    rows = []
    for b, (lo, hi) in enumerate([(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]):
        sel_ = mk_ & (Lr >= lo) & (Lr < hi)
        rows.append({"len_quintile": b, "n": int(sel_.sum()), "pEP": float(pEP[sel_].mean()), "pEP_div_pNCpDAC": float(pEPc[sel_].mean()),
                     "EP_true": float(ep_t[sel_].mean()), "EP_true_given_gs": float(ep_t[sel_ & gs].mean()),
                     "pNCpDAC": float(g_term[sel_].mean()), "NCDAC_true": float((nc_t * dac_t)[sel_].mean())})
    B2[nm] = rows
B2["gs_proposals_bias_pEP_div_g_minus_EP_given_gs_by_quintile"] = [float((pEPc - ep_t)[gs & (Lr >= lo) & (Lr < hi)].mean()) for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]]
B2["mean_pEP_all_proposals"] = float(pEP.mean()); B2["mean_EP_true_all_proposals"] = float(ep_t.mean())
B2["mean_pEP_geomsafe"] = float(pEP[gs].mean()); B2["mean_EP_true_geomsafe"] = float(ep_t[gs].mean())
B2["mean_pEP_geomviolating"] = float(pEP[~gs].mean()); B2["mean_EP_true_geomviolating"] = float(ep_t[~gs].mean())
B2["share_proposals_geomsafe"] = float(gs.mean())
out["b2_ep_head_unconditional"] = B2

# ---------------------------------------------------------------- (c) risk aversion of the selection rule
def logmean(x, y):
    x = np.maximum(x, 1e-12); y = np.maximum(y, 1e-12)
    return np.where(np.abs(x - y) < 1e-12, x, (x - y) / (np.log(x) - np.log(y)))

Lk = at(L, pk)
longer = L > (Lk[:, None] + 0.25)
cand = safe & longer
has_alt = cand.any(1)
# alternative = the truly best-PDMS member of the safe, longer set
alt = np.argmax(np.where(cand, pdms + 1e-9 * ep_t, -1), axis=1)
better = has_alt & (at(pdms, alt) > pd_pick + 1e-9)
Cc = {}
Cc["n_tokens"] = int(N)
Cc["n_with_safe_longer_alternative"] = int(has_alt.sum())
Cc["n_alternative_truly_better_PDMS"] = int(better.sum())
Cc["mean_true_PDMS_gain_if_swapped_to_best_safe_longer_x100"] = paired(boot, np.where(has_alt & better, at(pdms, alt), pd_pick), pd_pick)
Cc["mean_true_EP_gain_if_swapped"] = paired(boot, np.where(better, at(ep_t, alt), ep_pick), ep_pick, 1.0)
Cc["mean_L_gain_m_if_swapped"] = float((at(L, alt) - Lk)[better].mean()) if better.any() else None
Cc["pick_unsafe_share_among_better"] = float((~at(safe, pk))[better].mean()) if better.any() else None
# margin decomposition: ln S_pick - ln S_alt = d ln pNC + d ln pDAC + sum_t (w_t dp_t/12)/Lmean(W_pick, W_alt)
k, j = pk, alt
d_nc = np.log(at(p[..., NC], k)) - np.log(at(p[..., NC], j))
d_dac = np.log(at(p[..., DAC], k)) - np.log(at(p[..., DAC], j))
Wk, Wj = at(w_term, k), at(w_term, j)
Lm = logmean(Wk, Wj)
parts = {}
for nm, idx_, wt in [("EP", EP, 5.0), ("TTC", TTC, 5.0), ("C", C, 2.0)]:
    parts[nm] = wt * (at(p[..., idx_], k) - at(p[..., idx_], j)) / 12.0 / Lm
margin = np.log(at(S, k)) - np.log(at(S, j))
chk = np.abs(margin - (d_nc + d_dac + parts["EP"] + parts["TTC"] + parts["C"]))[better].max()
Cc["margin_decomposition_check_max_abs_error"] = float(chk)
terms = {"pNC": d_nc, "pDAC": d_dac, "EP_term": parts["EP"], "TTC_term": parts["TTC"], "C_term": parts["C"]}
sel = better
rows = {}
for nm, v in terms.items():
    rows[nm] = {"mean_contribution_to_margin_pick_over_alt": float(v[sel].mean()),
                "frac_tokens_term_favours_PICK(>0)": float((v[sel] > 1e-9).mean()),
                "frac_tokens_term_favours_ALT(<0)": float((v[sel] < -1e-9).mean())}
Cc["margin_terms_better_set"] = rows
Cc["mean_margin_better_set"] = float(margin[sel].mean())
Cc["median_margin_better_set"] = float(np.median(margin[sel]))
stack_ = np.stack([d_nc, d_dac, parts["EP"], parts["TTC"], parts["C"]], 1)
blocker = np.argmax(stack_, axis=1)
names = ["pNC", "pDAC", "EP_term", "TTC_term", "C_term"]
Cc["largest_positive_term_counts_better_set"] = {names[i]: int(((blocker == i) & sel).sum()) for i in range(5)}
# 'loses only because of its predicted safety terms': the alternative has the HIGHER predicted EP and still loses
alt_wins_ep = sel & (at(pEP, j) >= at(pEP, k))
Cc["better_set_alt_has_higher_or_equal_pEP"] = int(alt_wins_ep.sum())
Cc["better_set_alt_has_higher_pEP_frac"] = float(alt_wins_ep.sum() / max(sel.sum(), 1))
Cc["better_set_alt_has_lower_pEP_frac"] = float(1 - alt_wins_ep.sum() / max(sel.sum(), 1))
# among those, how much does the alt lose by on the SAFETY terms (pNC*pDAC and TTC) vs win on EP
Cc["mean_margin_terms_when_alt_has_higher_pEP"] = {nm: float(v[alt_wins_ep].mean()) for nm, v in terms.items()} if alt_wins_ep.any() else None
# predicted-vs-realised safety of the pick and alt: the head's calibration where it matters
Cc["pick_pNCxDAC_mean_vs_alt"] = [float(at(g_term, k)[sel].mean()), float(at(g_term, j)[sel].mean())]
Cc["pick_pTTC_mean_vs_alt"] = [float(at(p[..., TTC], k)[sel].mean()), float(at(p[..., TTC], j)[sel].mean())]
Cc["pick_pEP_mean_vs_alt"] = [float(at(pEP, k)[sel].mean()), float(at(pEP, j)[sel].mean())]
Cc["pick_EP_true_mean_vs_alt"] = [float(at(ep_t, k)[sel].mean()), float(at(ep_t, j)[sel].mean())]
Cc["pick_L_m_mean_vs_alt"] = [float(Lk[sel].mean()), float(at(L, j)[sel].mean())]
# how large is the margin relative to the head noise: fraction of better-set tokens with |margin| < 0.05 / 0.2
Cc["frac_margin_lt_0.05"] = float((margin[sel] < 0.05).mean())
Cc["frac_margin_lt_0.2"] = float((margin[sel] < 0.2).mean())
out["c_risk_aversion"] = Cc

# ---------------------------------------------------------------- (d) EP normaliser: inferred PDM-Closed progress scale per token
ratio = np.where(gs & (ep_t > 0.02) & (ep_t < 0.98), P / np.maximum(ep_t, 1e-9), np.nan)
cnt = np.sum(~np.isnan(ratio), axis=1)
ref = np.where(cnt >= 3, np.nanmedian(np.where(np.isnan(ratio), np.inf, ratio) if False else ratio, axis=1), np.nan)
spread = np.where(cnt >= 3, (np.nanpercentile(ratio, 75, axis=1) - np.nanpercentile(ratio, 25, axis=1)) / ref, np.nan)
D = {}
D["tokens_with_inferred_reference"] = int((cnt >= 3).sum())
D["inferred_reference_spread_IQR_over_median_median"] = float(np.nanmedian(spread))
D["inferred_reference_m_quantiles_5_25_50_75_95"] = [float(x) for x in np.nanpercentile(ref, [5, 25, 50, 75, 95])]
# token classes
sat = has_safe & (ep_best_safe >= 1 - 1e-9) & ~(cnt >= 3)           # no EP gradient visible: every geometry-safe proposal at EP ~ 1 or 0
cls = np.full(N, "no-EP-gradient/saturated", dtype=object)
for nm, lo, hi in [("ref<10m", 0, 10), ("ref10-20m", 10, 20), ("ref20-35m", 20, 35), ("ref>=35m", 35, 1e9)]:
    cls[(cnt >= 3) & (ref >= lo) & (ref < hi)] = nm
gap_tok = np.where(has_safe, ep_best_safe - ep_pick, 0.0)
rows = []
for nm in ["no-EP-gradient/saturated", "ref<10m", "ref10-20m", "ref20-35m", "ref>=35m"]:
    mk = cls == nm
    mk_s = mk & has_safe
    rows.append({"class": nm, "tokens": int(mk.sum()), "share_tokens": float(mk.mean()),
                 "mean_EP_gap_best_safe_minus_pick": float(gap_tok[mk_s].mean()) if mk_s.any() else None,
                 "share_of_total_EP_gap": float(gap_tok[mk].sum() / gap_tok.sum()),
                 "mean_EP_gap_in_metres": float((gap_tok * ref)[mk_s].mean()) if (nm.startswith("ref") and mk_s.any()) else None,
                 "mean_pick_L_m": float(Lk[mk].mean()) if mk.any() else None,
                 "EP_cost_of_1m_missing_progress": float(1.0 / ref[mk].mean()) if nm.startswith("ref") and mk.any() else None})
D["by_class"] = rows
# EP gap in metres: does the EP gap correspond to a modest absolute shortfall?  gap_m = EP_gap * ref
mk = (cnt >= 3) & has_safe
D["mean_EP_gap_in_metres_where_reference_inferred"] = float((gap_tok * ref)[mk].mean())
D["pick_shortfall_vs_best_safe_L_m_where_reference_inferred"] = float((Lk - at(L, bs_idx))[mk].mean())
out["d_ep_normaliser"] = D

with open(a.out, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)[:200000])
