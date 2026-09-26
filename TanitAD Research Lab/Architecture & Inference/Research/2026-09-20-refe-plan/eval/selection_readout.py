#!/usr/bin/env python3
"""SPEC_NAVTEST E-6 readout over `proptable/<name>/table.npz` -- AMENDMENT 3's pre-registered numbers.

    python eval/selection_readout.py --name sub200_ep011 --tokens <subset.json> [--boot 10000]

(a) PDMS of the planner's pick / RANDOM (mean over M) / ORACLE (best of M), and the six sub-scores
    at each, paired log-cluster bootstrap
(b) within-token Spearman(planner aggregate, true PDMS) over tokens whose true PDMS varies; top-1 hit
    rate beside a random selector's expected hit rate
(c) per component: pooled AUC of the predicted probability vs true == 1 (NC, DAC, DDC, TTC, C),
    within-token AUC, the share of tokens on which the component varies; Spearman for EP
(d) the pick's path length against the mean over proposals, and whether the scorer / the truth
    prefer long paths within a token
CONFIRMATORY (one): medoid vs the pick. EXPLORATORY (labelled, never claimed without a disjoint-token
confirmation): logitsum, violation-only, aggregate without EP.
Writes proptable/<name>/readout.json; prints ZZREADOUT <name> <verdict>.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

DATA = "D:/Projects/TanitAD/data/refe_navtest"
PDM_W = (5.0, 5.0, 4.0)                      # refe/planner.py PDM_W: EP, TTC, comfort over 14
COMP = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "C": 4, "DDC": 5}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def aggregate(logits):
    """refe/planner.py `aggregate`, transcribed: NC x DAC x DDC x (5 EP + 5 TTC + 4 C) / 14."""
    p = sigmoid(logits.astype(np.float64))
    w = np.asarray(PDM_W)
    return p[..., 0] * p[..., 1] * p[..., 5] * (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum()


def rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(len(a))
    edges = np.flatnonzero(np.r_[True, sa[1:] != sa[:-1], True])
    for s, e in zip(edges[:-1], edges[1:]):
        r[s:e] = (s + 1 + e) / 2.0                  # average rank of a tie group (1-based)
    out = np.empty(len(a))
    out[order] = r
    return out


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(score, y):
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class Boot:
    """Log-cluster bootstrap of a token mean: resample LOGS with replacement, pool their tokens."""

    def __init__(self, logs, B, seed=0):
        self.uniq = sorted(set(logs))
        ix = {l: i for i, l in enumerate(self.uniq)}
        self.g = np.array([ix[l] for l in logs])
        self.draws = np.random.default_rng(seed).integers(0, len(self.uniq), size=(B, len(self.uniq)))

    def __call__(self, v, mask=None, scale=1.0):
        v = np.asarray(v, dtype=np.float64)
        m = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
        G = len(self.uniq)
        s = np.bincount(self.g[m], weights=v[m], minlength=G)
        c = np.bincount(self.g[m], minlength=G).astype(np.float64)
        S, C = s[self.draws].sum(1), c[self.draws].sum(1)
        est = S[C > 0] / C[C > 0]
        return {"mean": round(float(v[m].mean()) * scale, 4), "n": int(m.sum()),
                "ci95": [round(float(np.percentile(est, 2.5)) * scale, 4),
                         round(float(np.percentile(est, 97.5)) * scale, 4)]}

    def ratio(self, num, den):
        """sum(num) / sum(den) with BOTH sums taken over the same resampled logs (paired)."""
        num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
        G = len(self.uniq)
        sn = np.bincount(self.g, weights=num, minlength=G)[self.draws].sum(1)
        sd = np.bincount(self.g, weights=den, minlength=G)[self.draws].sum(1)
        ok = np.abs(sd) > 1e-12
        est = sn[ok] / sd[ok]
        return {"value": round(float(num.sum() / den.sum()), 4), "n": int(len(num)),
                "ci95": [round(float(np.percentile(est, 2.5)), 4), round(float(np.percentile(est, 97.5)), 4)]}


def path_len(xy):                                    # [..., 8, 2] -> [...]; from the ego origin
    seg = np.diff(np.concatenate([np.zeros_like(xy[..., :1, :]), xy], axis=-2), axis=-2)
    return np.linalg.norm(seg, axis=-1).sum(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--table-dir", default=None, help="override proptable/<name> (the synthetic self-test)")
    a = ap.parse_args()
    wd = a.table_dir or os.path.join(DATA, "proptable", a.name)
    T = np.load(os.path.join(wd, "table.npz"))
    tok = [str(t) for t in T["token"]]
    pdms, sub, L, P, pick = T["pdms"], T["sub"], T["logits"].astype(np.float64), T["proposals"], T["pick"]
    assert tuple(str(x) for x in T["head_order"]) == tuple(COMP), T["head_order"]
    N, M = pdms.shape
    assert np.isfinite(pdms).all() and T["valid"].all(), "table has invalid cells"
    sj = json.load(open(a.tokens, encoding="utf-8"))
    tl = sj.get("token_log") if isinstance(sj, dict) else None
    if not tl:
        raise SystemExit("the tokens json carries no token_log; the cluster bootstrap needs it")
    logs = [tl[t] for t in tok]
    bt = Boot(logs, a.boot)
    ar = np.arange(N)
    agg = aggregate(L)
    rep = {"name": a.name, "N": N, "M": M, "n_logs": len(set(logs)), "boot": a.boot,
           "spec": "SPEC_NAVTEST.md E-6 / AMENDMENT 3 (blob 4c5aa5e6)"}
    rep["pick_reproduced_by_transcribed_aggregate"] = float((agg.argmax(1) == pick).mean())

    # (a) -----------------------------------------------------------------------------------------
    act, rnd, orc = pdms[ar, pick], pdms.mean(1), pdms.max(1)
    rep["a_pdms"] = {"actual": bt(act, scale=100), "random": bt(rnd, scale=100), "oracle": bt(orc, scale=100),
                     "actual_minus_random": bt(act - rnd, scale=100), "oracle_minus_actual": bt(orc - act, scale=100)}
    oi = pdms.argmax(1)
    rep["a_components_at"] = {nm: {"pick": round(float(sub[ar, pick, c].mean()), 4),
                                   "random": round(float(sub[:, :, c].mean()), 4),
                                   "oracle_pick": round(float(sub[ar, oi, c].mean()), 4)}
                              for nm, c in COMP.items()}

    # (b) -----------------------------------------------------------------------------------------
    varies = pdms.max(1) - pdms.min(1) > 1e-12
    rho = np.array([spearman(agg[i], pdms[i]) if varies[i] else np.nan for i in range(N)])
    rep["b_ranking"] = {
        "tokens_true_pdms_varies": int(varies.sum()),
        "within_token_spearman_agg_vs_pdms": bt(np.nan_to_num(rho), mask=varies),
        "top1_hit": bt((pdms[ar, pick] >= orc - 1e-12).astype(float)),
        "top1_hit_random_expectation": bt((pdms >= orc[:, None] - 1e-12).mean(1)),
        "pooled_spearman_agg_vs_pdms": round(spearman(agg.ravel(), pdms.ravel()), 4)}

    # (c) -----------------------------------------------------------------------------------------
    comp = {}
    for nm in ("NC", "DAC", "DDC", "TTC", "C"):
        c = COMP[nm]
        y = sub[:, :, c] == 1.0
        p = sigmoid(L[:, :, c])
        vt = y.any(1) & (~y).any(1)                # the component takes both values within the token
        wa = np.array([auc(p[i], y[i]) if vt[i] else np.nan for i in range(N)])
        comp[nm] = {"pooled_auc": round(auc(p.ravel(), y.ravel()), 4),
                    "within_token_auc": bt(np.nan_to_num(wa), mask=vt) if vt.any() else None,
                    "share_tokens_varying": round(float(vt.mean()), 4),
                    "mean_pred_prob": round(float(p.mean()), 4), "true_rate": round(float(y.mean()), 4)}
        comp[nm]["FAILING"] = bool(comp[nm]["pooled_auc"] < 0.60 and comp[nm]["share_tokens_varying"] >= 0.20)
    c = COMP["EP"]
    p, y = sigmoid(L[:, :, c]), sub[:, :, c]
    vt = y.max(1) - y.min(1) > 1e-12
    ws = np.array([spearman(p[i], y[i]) if vt[i] else np.nan for i in range(N)])
    comp["EP"] = {"pooled_spearman": round(spearman(p.ravel(), y.ravel()), 4),
                  "within_token_spearman": bt(np.nan_to_num(ws), mask=vt) if vt.any() else None,
                  "share_tokens_varying": round(float(vt.mean()), 4),
                  "mean_pred_prob": round(float(p.mean()), 4), "true_mean": round(float(y.mean()), 4)}
    rep["c_components"] = comp

    # (d) -----------------------------------------------------------------------------------------
    ln = path_len(P[..., :2].astype(np.float64))                          # [N, M]
    pr = np.array([rankdata(ln[i])[pick[i]] / M for i in range(N)])
    rep["d_path_length"] = {
        "pick_over_mean": bt(ln[ar, pick] / np.maximum(ln.mean(1), 1e-6)),
        "pick_length_percentile": bt(pr),
        "within_token_spearman_agg_vs_length": bt(np.nan_to_num(np.array([spearman(agg[i], ln[i]) for i in range(N)]))),
        "within_token_spearman_truePDMS_vs_length": bt(np.nan_to_num(np.array(
            [spearman(pdms[i], ln[i]) if varies[i] else 0.0 for i in range(N)])), mask=varies),
        "within_token_spearman_predEP_vs_length": bt(np.nan_to_num(np.array(
            [spearman(sigmoid(L[i, :, 2]), ln[i]) for i in range(N)])))}

    # CONFIRMATORY: medoid ----------------------------------------------------------------------
    xy = P[..., :2].astype(np.float64)
    med = np.array([np.linalg.norm(xy[i][:, None] - xy[i][None], axis=-1).mean(-1).sum(1).argmin()
                    for i in range(N)])
    rep["confirmatory_medoid"] = {"pdms": bt(pdms[ar, med], scale=100),
                                  "medoid_minus_actual": bt(pdms[ar, med] - act, scale=100)}

    # EXPLORATORY -------------------------------------------------------------------------------
    pp = sigmoid(L)
    rules = {"logitsum": L.sum(-1),
             "violation_only": pp[..., 0] * pp[..., 1] * pp[..., 5],
             "aggregate_without_EP": pp[..., 0] * pp[..., 1] * pp[..., 5] * (5 * pp[..., 3] + 4 * pp[..., 4]) / 9}
    rep["EXPLORATORY_rules"] = {k: {"pdms": bt(pdms[ar, v.argmax(1)], scale=100),
                                    "minus_actual": bt(pdms[ar, v.argmax(1)] - act, scale=100)}
                                for k, v in rules.items()}
    rep["EXPLORATORY_rules"]["_label"] = ("post-hoc rules on the same 200 tokens -- exploratory, never "
                                          "claimed without a disjoint-token confirmation (SPEC E-6)")

    # verdict (AMENDMENT 3a) -----------------------------------------------------------------------
    # skill = the share of the gain over a random pick that the planner's pick realises. The original
    # clause (actual-random 95% upper < +2.0) could never fire at n=200: under a truly RANDOM selector
    # its upper bound is +5.3..+5.8 PDMS (selftest_selection_readout.py, 3 seeds) -- kept, reported, unused.
    skill = bt.ratio(act - rnd, orc - rnd)
    rep["a_pdms"]["selection_skill"] = skill
    headroom = rep["a_pdms"]["oracle_minus_actual"]["mean"]
    if skill["ci95"][1] < 0.25 and headroom > 10.0:
        outcome = "SELECTION_BOUND"
    elif skill["ci95"][0] > 0.25 or headroom <= 10.0:
        outcome = "NOT_SELECTION_BOUND"
    else:
        outcome = "UNDETERMINED"
    failing = [k for k, v in comp.items() if v.get("FAILING")]
    rep["verdict"] = {"outcome": outcome, "SELECTION_BOUND": outcome == "SELECTION_BOUND",
                      "failing_scorer_outputs": failing,
                      "original_clause_amendment3": bool(rep["a_pdms"]["actual_minus_random"]["ci95"][1] < 2.0
                                                         and headroom > 10.0),
                      "rule": "AMENDMENT 3a: skill = (actual-random)/(oracle-random), paired log-cluster bootstrap; "
                              "SELECTION-BOUND iff skill 95% upper < 0.25 AND oracle-actual > 10 PDMS; NOT iff "
                              "skill 95% lower > 0.25 OR oracle-actual <= 10; else UNDETERMINED (extend the tokens). "
                              "FAILING output iff pooled AUC < 0.60 and it varies within token on >= 20% of tokens"}
    json.dump(rep, open(os.path.join(wd, "readout.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"ZZREADOUT {a.name} {outcome} failing={failing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
