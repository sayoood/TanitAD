#!/usr/bin/env python3
"""P1 -- descriptive diagnosis of REFe u400's slow CRUISE picks (T0, open loop, no rule, nothing fitted).

    python p1_diag.py --scope train|all|heldout

Population: CRUISE tokens (ego-probe definition, tokens_cruise.json) restricted to --scope logs, whose pick plans a > 15 %
slow-down (v4/v0 < 0.85; v4 = the probe's one-step 0.2 s chord at t = 4 s on the raw proposal, v0 = ego_real[:, 6]).
FAST set F = valid proposals with v4/v0 >= 0.9. Comparators:
  f_or = argmax over F of TRUE PDMS (ties -> higher aggregate)  -- an ORACLE comparator, descriptive only
  f_sc = argmax over F of the scorer OWN aggregate              -- what the scorer prefers once restricted to fast
Attribution: log agg(pick) - log agg(f) = dlog p_NC + dlog p_DAC + dlog avg(EP,TTC,C) + dlog p_lane (exact); the avg term is
split linearly into EP/TTC/C shares of (avg_pick - avg_f) (exact in sum).
"""
import argparse
import json

import numpy as np

import sel_common as sc

ap = argparse.ArgumentParser()
ap.add_argument("--scope", default="train", choices=("train", "all", "heldout"))
a = ap.parse_args()
D = sc.load()
split = json.load(open(sc.OUT + "/split.json"))
N = len(D["token"])
r = np.arange(N)
scope = {"all": np.ones(N, bool), "train": np.isin(D["log"], split["train_logs"]),
         "heldout": np.isin(D["log"], split["heldout_logs"])}[a.scope]
v0 = np.maximum(D["v0"], 1e-6)
ratio = D["v4_all"] / v0[:, None]
pick = D["pick"]
pdms, sub, val, L = D["pdms"], D["sub"], D["valid"], D["logits"].astype(np.float64)
P = 1.0 / (1.0 + np.exp(-L))
agg = sc.agg_np(L, np.float64)
rp = ratio[r, pick]
slow = rp < 0.85
pop = scope & D["cruise"] & slow
F = val & (ratio >= 0.9)
ul, li, draws = sc.log_boot_idx(D["log"], 10000, 20260927)
nl = len(ul)


def B(v, m):
    return sc.boot_mean(v, m, li, nl, draws)


res = {"scope": a.scope, "n_cruise": int((scope & D["cruise"]).sum()), "n_pop_slow_picks": int(pop.sum()),
       "n_logs_pop": int(len(np.unique(D["log"][pop])))}
res["slow_share_cruise"] = B(slow.astype(float), scope & D["cruise"])
pp = pdms[r, pick]
has_fast = F.any(1)
has_fast_ge = (F & (pdms >= pp[:, None] - 1e-12)).any(1)
has_fast_gt = (F & (pdms > pp[:, None] + 1e-12)).any(1)
band = val & (ratio >= 0.9) & (ratio <= 1.1)
has_band_ge = (band & (pdms >= pp[:, None] - 1e-12)).any(1)
res["Q1"] = {"share_with_any_fast": B(has_fast.astype(float), pop), "share_fast_pdms_ge_pick": B(has_fast_ge.astype(float), pop),
             "share_fast_pdms_gt_pick": B(has_fast_gt.astype(float), pop),
             "share_band_0.9_1.1_pdms_ge_pick": B(has_band_ge.astype(float), pop),
             "n_fast_per_token_median": float(np.median(F[pop].sum(1))) if pop.any() else None,
             "pick_pdms_mean": B(pp, pop), "pick_pdms_zero_share": B((pp == 0).astype(float), pop),
             "stratum_zero_share": B((D["stratum"] == "zero").astype(float), pop)}
NEG = -1e9
f_or = np.argmax(np.where(F, pdms + 1e-9 * agg, NEG), 1)
f_sc = np.argmax(np.where(F, agg, NEG), 1)
popf = pop & has_fast
TERMS = ("NC", "DAC", "EP", "TTC", "C", "lane")


def attrib(f):
    lp, lf = np.log(np.maximum(P[r, pick], 1e-300)), np.log(np.maximum(P[r, f], 1e-300))
    w = np.array(sc.V1_W)

    def avg(k):
        return (P[r, k, 2] * w[0] + P[r, k, 3] * w[1] + P[r, k, 4] * w[2]) / w.sum()

    ap_, af = avg(pick), avg(f)
    d_avg_log = np.log(ap_) - np.log(af)
    lin = {"EP": w[0] * (P[r, pick, 2] - P[r, f, 2]) / w.sum(), "TTC": w[1] * (P[r, pick, 3] - P[r, f, 3]) / w.sum(),
           "C": w[2] * (P[r, pick, 4] - P[r, f, 4]) / w.sum()}
    dd = ap_ - af
    safe = np.where(np.abs(dd) > 1e-12, dd, 1.0)
    terms = {"NC": lp[:, 0] - lf[:, 0], "DAC": lp[:, 1] - lf[:, 1], "lane": lp[:, 6] - lf[:, 6]}
    for k in ("EP", "TTC", "C"):
        terms[k] = np.where(np.abs(dd) > 1e-12, d_avg_log * lin[k] / safe, 0.0)
    tot = np.log(agg[r, pick]) - np.log(agg[r, f])
    return terms, tot


out_cmp = {}
for name, f in (("f_sc", f_sc), ("f_or", f_or)):
    terms, tot = attrib(f)
    o = {"dlog_agg_total": B(tot, popf), "dlog_agg_by_term": {k: B(v, popf) for k, v in terms.items()}}
    T = np.stack([terms[k] for k in TERMS], 1)
    dom = np.argmax(T, 1)
    o["dominant_term_counts"] = {k: int(((dom == i) & popf).sum()) for i, k in enumerate(TERMS)}
    o["dlogit_pick_minus_f"] = {h: B(L[r, pick, j] - L[r, f, j], popf) for j, h in enumerate(sc.HEADS)}
    o["true_pdms_pick"] = B(pp, popf)
    o["true_pdms_f"] = B(pdms[r, f], popf)
    o["true_pdms_f_minus_pick"] = B(pdms[r, f] - pp, popf)
    o["share_f_pdms_ge_pick"] = B((pdms[r, f] >= pp - 1e-12).astype(float), popf)
    o["share_f_pdms_gt_pick"] = B((pdms[r, f] > pp + 1e-12).astype(float), popf)
    o["v4_over_v0_pick"] = B(rp, popf)
    o["v4_over_v0_f"] = B(ratio[r, f], popf)
    o["pred_vs_true"] = {}
    for j, h in enumerate(sc.HEADS[:6]):
        o["pred_vs_true"][h] = {"pred_pick": B(P[r, pick, j], popf), "true_pick": B(sub[r, pick, j], popf),
                                "pred_f": B(P[r, f, j], popf), "true_f": B(sub[r, f, j], popf),
                                "pred_diff_pick_minus_f": B(P[r, pick, j] - P[r, f, j], popf),
                                "true_diff_pick_minus_f": B(sub[r, pick, j] - sub[r, f, j], popf)}
    o["pred_lane"] = {"pick": B(P[r, pick, 6], popf), "f": B(P[r, f, 6], popf)}
    out_cmp[name] = o
res["comparators"] = out_cmp
res["n_popf"] = int(popf.sum())
# Q3: the proposal nearest the logged human future (8 x 0.5 s grid, xy ADE) -- its TRUE PDMS vs the pick
Z = np.load(sc.EGO + "/full/results.npz")
zt = {str(t): i for i, t in enumerate(Z["token"])}
h8 = Z["human8"][[zt[t] for t in D["token"]]].astype(np.float64)
ade = np.linalg.norm(D["proposals"][..., :2] - h8[:, None, :, :2], axis=-1).mean(-1)
hn = np.argmin(np.where(val, ade, 1e9), 1)
res["Q3_human_nearest"] = {"ade_m": B(ade[r, hn], pop), "v4_over_v0": B(ratio[r, hn], pop), "true_pdms": B(pdms[r, hn], pop),
                           "true_pdms_minus_pick": B(pdms[r, hn] - pp, pop),
                           "true_EP_minus_pick": B(sub[r, hn, 2] - sub[r, pick, 2], pop),
                           "true_TTC_minus_pick": B(sub[r, hn, 3] - sub[r, pick, 3], pop),
                           "true_NC_minus_pick": B(sub[r, hn, 0] - sub[r, pick, 0], pop),
                           "true_DAC_minus_pick": B(sub[r, hn, 1] - sub[r, pick, 1], pop),
                           "agg_rank_of_human_nearest_median": float(np.median((agg[pop] > agg[pop, hn[pop]][:, None]).sum(1))) if pop.any() else None}
# Q4: calibration vs speed over ALL valid proposals of the scope's CRUISE tokens, binned by v4/v0
cm = (scope & D["cruise"])[:, None] & val
bins = [0, 0.5, 0.7, 0.85, 0.9, 1.0, 1.1, 1.3, 9]
cal = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = cm & (ratio >= lo) & (ratio < hi)
    row = {"bin": f"[{lo},{hi})", "n": int(m.sum()), "true_pdms": float(pdms[m].mean()) if m.any() else None,
           "agg": float(agg[m].mean()) if m.any() else None}
    for j, h in enumerate(sc.HEADS):
        row["pred_" + h] = float(P[..., j][m].mean()) if m.any() else None
        if j < 6:
            row["true_" + h] = float(sub[..., j][m].mean()) if m.any() else None
    cal.append(row)
res["Q4_calibration_by_speed_bin_cruise_all_valid_proposals"] = cal


def spearman_rows(X, Y, M):
    out = []
    for i in np.where(M.any(1))[0]:
        m = M[i]
        if m.sum() < 5:
            continue
        rx = np.argsort(np.argsort(X[i, m]))
        ry = np.argsort(np.argsort(Y[i, m]))
        if rx.std() == 0 or ry.std() == 0:
            continue
        out.append(np.corrcoef(rx, ry)[0, 1])
    return (float(np.mean(out)) if out else None), len(out)


sp = {h: spearman_rows(P[..., j], ratio, cm) for j, h in enumerate(sc.HEADS)}
sp["agg"] = spearman_rows(agg, ratio, cm)
sp["true_pdms"] = spearman_rows(pdms, ratio, cm)
for j, h in enumerate(sc.HEADS[:6]):
    sp["true_" + h] = spearman_rows(sub[..., j], ratio, cm)
res["Q4_within_token_spearman_vs_speed"] = sp
json.dump(res, open(sc.OUT + f"/p1_diag_{a.scope}.json", "w"), indent=1)
print("wrote", sc.OUT + f"/p1_diag_{a.scope}.json")
