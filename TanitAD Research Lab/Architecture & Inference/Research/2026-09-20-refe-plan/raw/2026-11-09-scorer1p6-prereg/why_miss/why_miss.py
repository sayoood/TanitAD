#!/usr/bin/env python3
"""Why does P6's scorer miss the best plan in its fan? Descriptive, CPU. E1 (C3, 1,058 navtest tokens, already read, stratum-weighted) and FIT (8,489 navtrain sets, P6 in-sample). E2 and E3 are NOT read.
No intervals: this informs an arm's design and decides nothing."""
import json, sys
import numpy as np
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read")
NC, DAC, EP, TTC, C, DDC = range(6)
def rule(x): return x[..., NC] * x[..., DAC] * (5 * x[..., EP] + 5 * x[..., TTC] + 2 * x[..., C]) / 12
def wm(x, w): return float((np.asarray(x, np.float64) * w).sum() / w.sum())
def auc_tok(p, y):
    pos, neg = p[y], p[~y]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    a = np.concatenate([pos, neg]); o = a.argsort(kind="stable"); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    # average ranks for ties
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True); s = np.zeros(len(u)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def analyse(name, L, SUB, PD, V, w, xy):
    N = len(L); ix = np.arange(N)
    P = 1 / (1 + np.exp(-L.astype(np.float64)))
    R = rule(P).copy(); R[~V] = -1e9
    pick = R.argmax(1); PP = PD.copy(); PP[~V] = -1; best = PP.argmax(1)
    r, b = PD[ix, pick], PD[ix, best]
    reg = 100 * (b - r)
    subp = SUB[ix, pick]
    zero = r == 0
    ttc0 = ~zero & (subp[:, TTC] == 0)
    half = ~zero & ~ttc0 & (subp[:, NC] == 0.5)
    soft = ~zero & ~ttc0 & ~half
    out = dict(n=N, pick=100 * wm(r, w), best=100 * wm(b, w), random=100 * wm((PD * V).sum(1) / V.sum(1), w), regret=wm(reg, w))
    out["regret_by_kind_points"] = {k: wm(reg * m, w) for k, m in (("pick_zero", zero), ("pick_ttc0", ttc0), ("pick_nc_half", half), ("pick_clean_ep_only", soft))}
    out["share_of_tokens_pct"] = {k: 100 * wm(m, w) for k, m in (("pick_zero", zero), ("pick_ttc0", ttc0), ("pick_nc_half", half), ("pick_clean", soft))}
    out["unweighted_counts"] = dict(pick_zero=int(zero.sum()), pick_ttc0=int(ttc0.sum()))
    clean = V & (SUB[..., NC] == 1) & (SUB[..., DAC] == 1) & (SUB[..., TTC] == 1)
    nz = V & (PD > 0)
    fail = zero | ttc0
    out["alternatives_when_pick_fails"] = {"n": int(fail.sum()), "median_clean_proposals_of_64": float(np.median(clean[fail].sum(1))), "pct_with_none_clean": 100 * float((clean[fail].sum(1) == 0).mean()), "pct_with_at_least_16_clean": 100 * float((clean[fail].sum(1) >= 16).mean()), "pct_with_at_least_32_clean": 100 * float((clean[fail].sum(1) >= 32).mean())}
    # rank of first clean proposal under P6's rule, on failing picks that have one
    idx = np.where(fail & (clean.sum(1) > 0))[0]
    ranks = []; d_near = []; d_med = []; pa = []; pb = []
    for i in idx:
        order = R[i].argsort()[::-1]
        ranks.append(int(np.argmax(clean[i][order])))
        d = np.sqrt(((xy[i] - xy[i, pick[i]]) ** 2).sum(-1)).mean(-1)
        d[pick[i]] = np.inf
        d_near.append(d[clean[i]].min()); d_med.append(np.median(d[np.isfinite(d)]))
        j = np.where(clean[i])[0][d[clean[i]].argmin()]
        pa.append(P[i, pick[i], [NC, DAC, TTC]]); pb.append(P[i, j, [NC, DAC, TTC]])
    ranks = np.array(ranks); d_near = np.array(d_near); d_med = np.array(d_med)
    out["first_clean_under_the_rule"] = {"median_position_after_pick": float(np.median(ranks)), "pct_within_next_1": 100 * float((ranks <= 1).mean()), "pct_within_next_3": 100 * float((ranks <= 3).mean()), "pct_within_next_7": 100 * float((ranks <= 7).mean())}
    out["nearest_clean_to_failing_pick_m_ade"] = {"median": float(np.median(d_near)), "median_distance_to_a_typical_other_proposal": float(np.median(d_med)), "pct_under_0p5m": 100 * float((d_near < 0.5).mean()), "pct_under_1m": 100 * float((d_near < 1).mean())}
    pa = np.array(pa); pb = np.array(pb)
    out["head_prob_failing_pick_vs_nearest_clean"] = {h: [float(pa[:, k].mean()), float(pb[:, k].mean())] for k, h in enumerate(("NC", "DAC", "TTC"))}
    aucs = {}
    for k, h in ((NC, "NC"), (DAC, "DAC"), (TTC, "TTC")):
        a = np.array([auc_tok(P[i, V[i], k], SUB[i, V[i], k] == 1) for i in range(N)])
        ok = np.isfinite(a)
        aucs[h] = dict(tokens_with_mixed_labels_pct=100 * wm(ok, w), all=float(np.average(a[ok], weights=w[ok])), pick_failed=float(np.nanmean(a[fail & ok])) if (fail & ok).any() else None, pick_ok=float(np.average(a[~fail & ok], weights=w[~fail & ok])))
    out["within_token_auc"] = aucs
    order = R.argsort(1)[:, ::-1]
    out["best_within_top_K_of_the_rule"] = {K: 100 * wm(PP[ix[:, None], order[:, :K]].max(1), w) for K in (1, 2, 3, 4, 8, 16, 32, 64)}
    return out
res = {}
Z = np.load("F:/tanitad_scratch/p6fit/fit_pack8.npz", allow_pickle=False)
pdm = Z["pdm"].astype(np.float64); ok = Z["has_pdm"].astype(bool)
print("FIT has_pdm", int(ok.sum()), "of", len(ok), "NaN in pdm", int(np.isnan(pdm[ok]).sum()))
res["FIT"] = analyse("FIT", Z["logits"][ok], pdm[ok][..., :6], pdm[ok][..., 6], ~np.isnan(pdm[ok][..., 6]), np.ones(int(ok.sum())), Z["traj_xy"][ok].astype(np.float64))
import selc6_analyze as SA
D0 = np.load("F:/TanitAD_p6read_work/run/dump/dump_p6.npz", allow_pickle=False); T = np.load("F:/TanitAD_p6read_work/run/score/table_p6.npz", allow_pickle=False)
dtok = [str(t) for t in D0["token"]]; assert [str(t) for t in T["token"]] == dtok
arrs = {k: D0[k] for k in ("logits", "pick", "traj", "props8", "ego", "gt_valid", "gt_pose20", "gt_t20", "human8", "frame_ctrl_m", "fingerprint")}; arrs.update(pdms=T["pdms"], sub=T["sub"], valid=T["valid"])
D, A = SA.align_to_c3(arrs, dtok)
zs = np.asarray(D["stratum"]) == "zero"; w = np.where(zs, SA.W_ZERO / zs.sum(), SA.W_RANDOM / (~zs).sum())
res["E1"] = analyse("E1", A["logits"], A["sub"].astype(np.float64), A["pdms"].astype(np.float64), A["valid"].astype(bool), w, A["props8"][..., :2].astype(np.float64))
json.dump(res, open("F:/tanitad_scratch/p6sel/why/why_miss.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=0, default=float))
