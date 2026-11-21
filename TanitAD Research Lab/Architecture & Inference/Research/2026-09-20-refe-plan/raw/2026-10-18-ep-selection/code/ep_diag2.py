#!/usr/bin/env python3
"""EP-1 step 1b (fit table only): TTC / C head conditionality, progress-proxy rank power, score-margin distribution. No candidate rule is evaluated here."""
import json, sys
import numpy as np
from scipy.stats import rankdata, spearmanr
from ep_common import *
from ep_common import FIT, FIT_SHA, NC, DAC, EP, TTC, C, load, sigmoid, score_terms, geometry, at, pick_ref

t = load(FIT, FIT_SHA[:16]); N = len(t["token"]); sub = t["sub"]; p = sigmoid(t["logits"].astype(np.float64))
g, w = score_terms(p); S = g * w
L, P = geometry(t["proposals"])
gs = (sub[..., NC] == 1) & (sub[..., DAC] == 1)
Lr = (rankdata(L, axis=1) - 1) / 63.0
out = {}
rows = {}
for nm, idx in [("TTC", TTC), ("C", C)]:
    r = []
    for b, (lo, hi) in enumerate([(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]):
        s_ = (Lr >= lo) & (Lr < hi)
        r.append({"len_quintile": b, "p_all": float(p[..., idx][s_].mean()), "true_all": float(sub[..., idx][s_].mean()),
                  "p_gs": float(p[..., idx][s_ & gs].mean()), "true_gs": float(sub[..., idx][s_ & gs].mean()),
                  "p_viol": float(p[..., idx][s_ & ~gs].mean()), "true_viol": float(sub[..., idx][s_ & ~gs].mean())})
    rows[nm] = r
out["ttc_c_conditionality"] = rows
# is the product-form E[NC*DAC*TTC] ~ pNC*pDAC*pTTC calibrated?  compare against the realised mean of NC*DAC*TTC (and the 'conditional' reading pNC*pDAC*pTTC)
ndt = sub[..., NC] * sub[..., DAC] * sub[..., TTC]
out["E_NCxDACxTTC_realised_vs_product_of_heads"] = {"realised": float(ndt.mean()), "product_of_heads": float((g * p[..., TTC]).mean()),
                                                     "head_pTTC_alone": float(p[..., TTC].mean()), "realised_TTC": float(sub[..., TTC].mean())}
ndc = sub[..., NC] * sub[..., DAC] * sub[..., C]
out["E_NCxDACxC_realised_vs_product"] = {"realised": float(ndc.mean()), "product_of_heads": float((g * p[..., C]).mean())}
ndep = sub[..., NC] * sub[..., DAC] * sub[..., EP]
out["E_NCxDACxEP_realised_vs_pEP"] = {"realised": float(ndep.mean()), "pEP": float(p[..., EP].mean()), "EP_realised": float(sub[..., EP].mean())}
# progress proxies, within-token Spearman with true EP on geometry-safe proposals
def rho(x, y, mask):
    v = []
    for i in range(N):
        m = mask[i]
        if m.sum() >= 4 and np.ptp(x[i][m]) > 0 and np.ptp(y[i][m]) > 0:
            v.append(spearmanr(x[i][m], y[i][m])[0])
    return [float(np.mean(v)), len(v)]
out["within_token_spearman_vs_EPtrue_geomsafe"] = {"L_end_distance": rho(L, sub[..., EP], gs), "P_path_length": rho(P, sub[..., EP], gs),
                                                   "pEP": rho(p[..., EP], sub[..., EP], gs), "pEP_div_g": rho(np.minimum(1, p[..., EP] / np.maximum(g, 1e-9)), sub[..., EP], gs)}
# score margins among the contenders (planner rule): log S_max - log S_j, quantiles over tokens of the margin to the k-th ranked proposal
ls = np.log(np.maximum(S, 1e-300)); srt = -np.sort(-ls, axis=1)
out["log_score_margin_top1_minus_topk_quantiles_5_25_50_75_95"] = {f"k{k}": [float(x) for x in np.percentile(srt[:, 0] - srt[:, k - 1], [5, 25, 50, 75, 95])] for k in (2, 3, 5, 8, 16)}
# how many proposals sit within delta of the max score (the band size), mean over tokens
out["mean_band_size_by_delta"] = {str(d): float(((ls >= ls.max(1, keepdims=True) - d).sum(1)).mean()) for d in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)}
# predicted-safety index Q = pNC*pDAC*pTTC at the planner pick (quantiles) and its distribution over all proposals
Q = g * p[..., TTC]
pk = pick_ref(t)
out["Q_at_pick_quantiles_1_5_25_50_75"] = [float(x) for x in np.percentile(at(Q, pk), [1, 5, 25, 50, 75])]
out["Q_all_proposals_quantiles_5_25_50_75_95"] = [float(x) for x in np.percentile(Q, [5, 25, 50, 75, 95])]
out["frac_tokens_with_Qmax_below"] = {str(x): float((Q.max(1) < x).mean()) for x in (0.5, 0.7, 0.8, 0.9)}
json.dump(out, open("../raw/diag2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
