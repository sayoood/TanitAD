#!/usr/bin/env python3
"""REVIEW 7 -- per-head diagnostics of the final REFe scorer on navtest (CPU only; navsim venv python).

Follows gap_selection.py. Adds:
  * a LEAK-FREE EP oracle: NAVSIM's EP is multiplied by NC*DAC before normalisation, so GT EP = 0 on every unsafe
    proposal and "replace p_EP by GT EP" smuggles safety information into the EP slot. The leak-free variant replaces
    p_EP by GT EP ONLY where GT NC*DAC = 1 and keeps the model's own p_EP elsewhere.
  * per-head within-set discrimination (AUC, binary heads) and per-head within-set correlation with the proposal's
    path length, predicted vs GT: which head carries the short-plan bias of the selection.
  * calibration of the multiplicative heads: mean predicted vs GT rate, overall and on the pick.
EXPLORATORY; intervals as in gap_selection.py.
"""
import json

import numpy as np

import gap_selection as GS

OUT = GS.PKG + "/raw/2026-10-04-gap-review/gap_heads.json"


def auc_within(pred, lab, pos_val=1.0):
    """mean within-set AUC of pred for lab==pos_val vs lab<pos_val, over sets with both classes."""
    vals = []
    for p, l in zip(pred, lab):
        pos = p[l == pos_val]
        neg = p[l < pos_val]
        if len(pos) == 0 or len(neg) == 0:
            continue
        # rank-based AUC
        allv = np.concatenate([pos, neg])
        order = allv.argsort(kind="mergesort")
        ranks = np.empty(len(allv))
        ranks[order] = np.arange(1, len(allv) + 1)
        # average ties
        _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=ranks)
        ranks = (sums / cnt)[inv]
        r_pos = ranks[: len(pos)].sum()
        vals.append((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return {"mean_auc": float(np.mean(vals)), "mixed_sets": len(vals)}


def wcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ok = den > 1e-12
    return {"mean_r": float(np.mean((a * b).sum(1)[ok] / den[ok])), "sets": int(ok.sum())}


def main():
    z = np.load(GS.PROPS, allow_pickle=True)
    tok = z["token"]
    props = z["proposals"].astype(np.float64)
    logit = z["logits"]
    pick = z["pick"]
    pos = {t: i for i, t in enumerate(tok)}
    N = len(tok)
    G = {k: np.full((N, 64), np.nan) for k in ("nc", "dac", "ttc", "c", "pdms", "ep", "ddc")}
    logs = np.empty(N, dtype=object)
    for line in open(GS.CENSUS):
        r = json.loads(line)
        i = pos[r["token"]]
        logs[i] = r["log"]
        for k in ("nc", "dac", "ttc", "c", "pdms", "ddc"):
            G[k][i] = np.asarray(r[k][1:65], float)
        m = G["nc"][i] * G["dac"][i]
        with np.errstate(divide="ignore", invalid="ignore"):
            ep = np.where(m > 0, (12 * G["pdms"][i] / np.where(m > 0, m, 1) - 5 * G["ttc"][i] - 2 * G["c"][i]) / 5, 0.0)
        G["ep"][i] = np.clip(ep, 0, 1)
    logs = logs.astype(str)
    P = GS.sig(logit)
    pnc, pdac, pep, pttc, pc, pddc = (P[..., k] for k in range(6))
    rows = np.arange(N)
    base = G["pdms"][rows, pick]
    xy = props[..., :2]
    seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((N, 64, 1, 2)), xy], axis=2), axis=2), axis=-1)
    L = seg.sum(-1)
    safe = (G["nc"] * G["dac"]) == 1.0

    out = {"n_tokens": N}
    # leak-free EP oracle
    ep_lf = np.where(safe, G["ep"], pep)
    out["oracle_EP_leak_free"] = GS.boot_paired(G["pdms"][rows, GS.agg(pnc, pdac, ep_lf, pttc, pc).argmax(1)],
                                                base, logs)
    out["oracle_EP_leak_free+TTC"] = GS.boot_paired(
        G["pdms"][rows, GS.agg(pnc, pdac, ep_lf, G["ttc"], pc).argmax(1)], base, logs)
    # oracle multiplicative only, but with the model's EP/TTC/C (= gap_selection oracle_NC+DAC) for reference
    # per-head discrimination
    out["within_set_AUC"] = {
        "NC (GT nc==1 vs <1)": auc_within(pnc, G["nc"]),
        "DAC (GT dac==1 vs 0)": auc_within(pdac, G["dac"]),
        "TTC (GT ttc==1 vs 0)": auc_within(pttc, G["ttc"]),
        "C (GT c==1 vs 0)": auc_within(pc, G["c"]),
        "DDC (GT ddc==1 vs <1)": auc_within(pddc, G["ddc"]),
        "pnc*pdac for GT safe": auc_within(pnc * pdac, safe.astype(float)),
    }
    # correlation with path length, predicted vs GT (the short-plan bias)
    out["within_set_corr_with_path_length"] = {
        "pred_nc": wcorr(pnc, L), "GT_nc": wcorr(G["nc"], L),
        "pred_dac": wcorr(pdac, L), "GT_dac": wcorr(G["dac"], L),
        "pred_ttc": wcorr(pttc, L), "GT_ttc": wcorr(G["ttc"], L),
        "pred_c": wcorr(pc, L), "GT_c": wcorr(G["c"], L),
        "pred_ep": wcorr(pep, L), "GT_ep_on_safe_only(nan->set mean)": wcorr(
            np.where(safe, G["ep"], np.nanmean(np.where(safe, G["ep"], np.nan), 1, keepdims=True)), L),
        "pred_pnc*pdac": wcorr(pnc * pdac, L), "GT_safe": wcorr(safe.astype(float), L),
    }
    # calibration
    out["calibration_mean_pred_vs_GT"] = {
        "nc": [float(pnc.mean()), float(G["nc"].mean())],
        "dac": [float(pdac.mean()), float(G["dac"].mean())],
        "ttc": [float(pttc.mean()), float(G["ttc"].mean())],
        "c": [float(pc.mean()), float(G["c"].mean())],
        "on_pick_nc": [float(pnc[rows, pick].mean()), float(G["nc"][rows, pick].mean())],
        "on_pick_dac": [float(pdac[rows, pick].mean()), float(G["dac"][rows, pick].mean())],
        "on_pick_ttc": [float(pttc[rows, pick].mean()), float(G["ttc"][rows, pick].mean())],
    }
    # length deciles within set: GT safe share and pred safe share by within-set length rank quintile
    rk = L.argsort(1).argsort(1)            # 0 = shortest
    q = (rk * 5) // 64
    tab = {}
    for b in range(5):
        msk = q == b
        tab[f"Q{b + 1}_{'shortest' if b == 0 else ('longest' if b == 4 else '')}"] = {
            "GT_safe_share": float(safe[msk].mean()), "pred_pnc*pdac": float((pnc * pdac)[msk].mean()),
            "GT_ttc": float(G["ttc"][msk].mean()), "pred_ttc": float(pttc[msk].mean()),
            "GT_pdms": float(G["pdms"][msk].mean()), "pred_agg": float(GS.agg(pnc, pdac, pep, pttc, pc)[msk].mean()),
            "GT_ep": float(G["ep"][msk].mean()), "pred_ep": float(pep[msk].mean()),
            "share_of_picks": float((q[rows, pick] == b).mean())}
    out["by_within_set_length_quintile"] = tab
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
