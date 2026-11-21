#!/usr/bin/env python3
"""u400 reference values for PREREG_P6_SCORER_SPEED (CPU only, banked data only, reads nothing of P6).

Imports the YSEL P1 loader (sel_common) UNCHANGED and recomputes, for u400 on the 1,058 C3 tokens:
  R1  stratum-weighted CRUISE slow-pick share (weights W_zero=529/12146, W_random=11617/12146, the selc prereg section 3 weights)
  R2  within-token Spearman of the predicted and TRUE TTC / NC / DAC / EP vs v4/v0 on CRUISE, TWO estimators:
        legacy = p1_diag.py spearman_rows (argsort-of-argsort ranks: ties broken by proposal INDEX)  -> must reproduce p1_diag_all.json
        tie    = average ranks (scipy.stats.rankdata)                                             -> the PRE-REGISTERED estimator
  paired log-cluster bootstrap (sel_common.log_boot_idx, B=10000, seed 20260927), 95 % percentile.
"""
import json, sys
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107")
import sel_common as sc

OUT = "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107"
W_ZERO, W_RANDOM = 529 / 12146, 11617 / 12146
D = sc.load()
N = len(D["token"]); r = np.arange(N)
v0 = np.maximum(D["v0"], 1e-6)
ratio = D["v4_all"] / v0[:, None]
P = 1.0 / (1.0 + np.exp(-D["logits"].astype(np.float64)))
sub, val = D["sub"], D["valid"]
slow = (ratio[r, D["pick"]] < 0.85).astype(float)
cru = D["cruise"]
zero = D["stratum"] == "zero"
ul, li, draws = sc.log_boot_idx(D["log"], 10000, 20260927)
nl = len(ul)

def boot_ratio_parts(vals, mask):
    m = mask & np.isfinite(vals)
    s = np.bincount(li[m], weights=vals[m], minlength=nl); c = np.bincount(li[m], minlength=nl).astype(float)
    return s, c

def wshare():
    sz, cz = boot_ratio_parts(slow, cru & zero); sr, cr = boot_ratio_parts(slow, cru & ~zero)
    pt = W_ZERO * sz.sum() / cz.sum() + W_RANDOM * sr.sum() / cr.sum()
    Z = sz[draws].sum(1) / np.maximum(cz[draws].sum(1), 1e-12); R = sr[draws].sum(1) / np.maximum(cr[draws].sum(1), 1e-12)
    ok = (cz[draws].sum(1) > 0) & (cr[draws].sum(1) > 0)
    b = (W_ZERO * Z + W_RANDOM * R)[ok]
    return dict(point=float(pt), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                n_cruise_zero=int((cru & zero).sum()), n_cruise_random=int((cru & ~zero).sum()),
                share_zero=float(sz.sum() / cz.sum()), share_random=float(sr.sum() / cr.sum()),
                share_unweighted=float(slow[cru].mean()), n_draws_used=int(ok.sum()))

def per_token_spearman(X, Y, M, method):
    out = np.full(N, np.nan)
    for i in np.where(M.any(1))[0]:
        m = M[i]
        if m.sum() < 5:
            continue
        if method == "legacy":
            rx = np.argsort(np.argsort(X[i, m])); ry = np.argsort(np.argsort(Y[i, m]))
        else:
            rx = rankdata(X[i, m]); ry = rankdata(Y[i, m])
        if rx.std() == 0 or ry.std() == 0:
            continue
        out[i] = np.corrcoef(rx, ry)[0, 1]
    return out

def boot_mean_tok(v, mask):
    return sc.boot_mean(v, mask, li, nl, draws)

cm = cru[:, None] & val
res = {"weights": {"W_zero": W_ZERO, "W_random": W_RANDOM}, "R1_weighted_cruise_slow_share": wshare()}
HEADS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3}
for method in ("legacy", "tie"):
    block = {}
    for h, j in HEADS.items():
        rp = per_token_spearman(P[..., j], ratio, cm, method)
        rt = per_token_spearman(sub[..., j], ratio, cm, method)
        ok = np.isfinite(rp) & np.isfinite(rt)          # tokens where BOTH are defined (paired)
        gap = np.where(ok, rp - rt, np.nan)
        block[h] = {"pred": boot_mean_tok(np.where(ok, rp, np.nan), cru), "true": boot_mean_tok(np.where(ok, rt, np.nan), cru),
                    "gap_pred_minus_true": boot_mean_tok(gap, cru), "n_tokens_paired": int(ok.sum()),
                    "pred_unpaired_mean": float(np.nanmean(rp[cru])), "true_unpaired_mean": float(np.nanmean(rt[cru])),
                    "n_pred": int(np.isfinite(rp[cru]).sum()), "n_true": int(np.isfinite(rt[cru]).sum())}
    res["R2_spearman_" + method] = block

# R3: lane-confound control. P6 has NO lane output (6-output scorer, rule navsim_v1, SPEC_P6 sec.1 row "scorer components"). Re-pick u400 with the lane
# factor DROPPED (same logits, rule navsim_v1) -> its slow share without the lane head; and the paired delta vs its banked (navsim_v1_lane) pick.
pick_nl = sc.pick_from_logits(D["logits"], "torch", lane=False)
slow_nl = (ratio[r, pick_nl] < 0.85).astype(float)
def wshare_of(v, mask_c):
    sz, cz = boot_ratio_parts(v, mask_c & zero); sr, cr = boot_ratio_parts(v, mask_c & ~zero)
    ok = (cz[draws].sum(1) > 0) & (cr[draws].sum(1) > 0)
    return sz, cz, sr, cr, ok
def wb(v):
    sz, cz, sr, cr, ok = wshare_of(v, cru)
    pt = W_ZERO * sz.sum() / cz.sum() + W_RANDOM * sr.sum() / cr.sum()
    b = (W_ZERO * sz[draws].sum(1) / np.maximum(cz[draws].sum(1), 1e-12) + W_RANDOM * sr[draws].sum(1) / np.maximum(cr[draws].sum(1), 1e-12))
    return pt, b, ok
p1, b1, ok1 = wb(slow); p2, b2, ok2 = wb(slow_nl)
d = (b2 - b1)[ok1 & ok2]
res["R3_u400_without_lane_factor"] = {"n_picks_changed_cruise": int((pick_nl[cru] != D["pick"][cru]).sum()), "n_picks_changed_all": int((pick_nl != D["pick"]).sum()),
    "weighted_slow_share": {"point": float(p2), "lo": float(np.percentile(b2[ok2], 2.5)), "hi": float(np.percentile(b2[ok2], 97.5))},
    "delta_vs_banked": {"point": float(p2 - p1), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))},
    "G_banked_rule_reproduced": int((sc.pick_from_logits(D["logits"], "torch", lane=True) == D["pick"]).sum())}
json.dump(res, open(OUT + "/u400_reference.json", "w"), indent=1)
b = json.load(open("E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107/p1_diag_all.json"))["Q4_within_token_spearman_vs_speed"]
res["G_REPRO_p1_diag_all"] = {h: {"banked_pred": b[h], "banked_true": b["true_" + h]} for h in HEADS}
json.dump(res, open(OUT + "/u400_reference.json", "w"), indent=1)
print(json.dumps(res, indent=1))
