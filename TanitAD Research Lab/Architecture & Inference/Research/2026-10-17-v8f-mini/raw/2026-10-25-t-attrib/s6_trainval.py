"""s6_trainval.py -- Q3: T's train constraint loss vs its eval error, in IDENTICAL units and the IDENTICAL reduction. (EXPLORATORY; MEASURED from the banked
train_log.jsonl of each arm and the banked eval T blocks.)

Definition of the train number (read at source):
  v8f_mini.py:675-683  tgt = (lt['cons'] - cons_mu) / cons_sd          -> SD units
                       hub = huber_loss(cons_std, tgt, delta=1.0)       -> per CELL, delta 1 SD
                       m   = isfinite(tgt) & lt['has']                  -> PRESENT cells of labelled windows
                       T['cons'] = (hub*m).sum() / m.sum()              -> ONE mean over all present cells of the 64-window batch (pooled over cells, not over windows)
  v8f_mini_train.py:770-773  the logged `cons` is that value for the single batch at the logged step (every log_every steps), NOT an average over the interval.
The eval-side equivalent computed here: mean over ALL present cells (pooled over windows and cells) of Huber(delta=1) of (T cons_std - GT std).
Differences that remain and CANNOT be separated on CPU with local files (listed in the report): train windows are re-seen over ~4 epochs; the logged batches
use p_gt (0.2 at the end: S's checkpoint is the GT with prob p_gt) while eval has p_gt=0 and S's own checkpoint; train clips vs held-out clips.
"""
import json

import numpy as np

from tcommon import *

LG = "C:/Users/Admin/box2_ship/logs/"
RUNS = {"M2b_s0": "v8f_mini/runs/M2b_s0", "M2b_s1": "v8f_mini/runs/M2b_s1", "grid8x20_s0": "v8f_grid/runs/M2b_s0_8x20",
        "hist10_s0": "v8f_hist/runs/M2b_s0_h10", "pgt_s0": "v8f_pgt/runs/M2b-pgt_s0", "wc_s0": "v8f_wc/runs/M2b-wc_s0", "df50_s0": "v8f_dscale/runs/M2b_s0_df50"}


def main():
    T = load()
    eid = T["ci"]
    ue = np.unique(eid)
    has = T["has"]
    G = T["gt_std"]
    P = np.isfinite(G) & has[:, None]
    print(f"{'arm':12s} {'steps':>6s} {'n_log':>5s} | {'TRAIN cons Huber, last 10% steps':>34s} | {'TRAIN last 25%':>14s} | {'EVAL Huber (pooled cells)':>26s} {'EVAL MSE':>8s} {'EVAL/TRAIN':>10s}")
    rows = {}
    for arm, rd in RUNS.items():
        recs = [json.loads(l) for l in open(LG + rd + "/train_log.jsonl", encoding="utf-8") if l.strip()]
        total = max(r["step"] for r in recs)
        last10 = np.array([r["cons"] for r in recs if r["step"] > 0.9 * total])
        last25 = np.array([r["cons"] for r in recs if r["step"] > 0.75 * total])
        pg = np.mean([r["p_gt"] for r in recs if r["step"] > 0.9 * total])
        R = resid(T, arm)
        H = np.where(P, huber(np.where(P, R, 0.0)), 0.0)
        e_h = H.sum() / P.sum()
        e_mse = np.where(P, R ** 2, 0.0).sum() / P.sum()
        # episode-cluster CI of the eval Huber
        es = np.array([H[eid == e].sum() for e in ue])
        en = np.array([P[eid == e].sum() for e in ue])
        rng = np.random.default_rng(0)
        bs = np.array([es[k].sum() / en[k].sum() for k in rng.integers(0, len(ue), (2000, len(ue)))])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        se = last10.std(ddof=1) / np.sqrt(len(last10))
        print(f"{arm:12s} {total:6d} {len(last10):5d} | {last10.mean():.4f} +- {se:.4f} (SE, n={len(last10)}) pg {pg:.2f} | {last25.mean():.4f}      | "
              f"{e_h:.4f} [{lo:.4f},{hi:.4f}] {e_mse:8.4f} {e_h / last10.mean():10.2f}")
        rows[arm] = (last10.mean(), se, e_h)
    # the same arithmetic as the DSCALE/WC results: ratio of last-10% train between arms
    print("\nTRAIN last-10% cons relative to M2b_s0:  " + ", ".join(f"{a} {rows[a][0] / rows['M2b_s0'][0] - 1:+.1%}" for a in RUNS if a != "M2b_s0"))
    print("EVAL Huber relative to M2b_s0:            " + ", ".join(f"{a} {rows[a][2] / rows['M2b_s0'][2] - 1:+.1%}" for a in RUNS if a != "M2b_s0"))
    print("(the quoted DSCALE train numbers: df50 0.1209 vs M2b_s0 0.1264, -4.3%; WC: 18-25% lower train Huber at w_cons 8x -- different bins, not re-derived here.)")

    # distribution of the train loss trajectory for M2b_s0: does it plateau (fit saturated) or still fall?
    recs = [json.loads(l) for l in open(LG + RUNS["M2b_s0"] + "/train_log.jsonl", encoding="utf-8") if l.strip()]
    st = np.array([r["step"] for r in recs])
    cs = np.array([r["cons"] for r in recs])
    print("\nM2b_s0 train cons Huber by quintile of training (mean +- SE of the logged batches):")
    for q in range(5):
        m = (st > q * st.max() / 5) & (st <= (q + 1) * st.max() / 5)
        print(f"   steps {int(q * st.max() / 5):6d}-{int((q + 1) * st.max() / 5):6d}: {cs[m].mean():.4f} +- {cs[m].std(ddof=1) / np.sqrt(m.sum()):.4f} (n={m.sum()}), p_gt {np.mean([r['p_gt'] for r, mm in zip(recs, m) if mm]):.2f}")

    # label distribution shift check: the eval GT in train SD units (train mu/sd recovered from the oracle block): mean ~0 and sd ~1 if eval ~ train
    print("\nEval GT in TRAIN standardiser units (mu/sd were estimated on the labelled train rows; eval present cells): mean / sd per cell")
    bad = []
    for c in range(26):
        m = P[:, c]
        if m.sum() < 30:
            continue
        mu_e, sd_e = G[m, c].mean(), G[m, c].std()
        if abs(mu_e) > 0.25 or abs(sd_e - 1) > 0.3:
            bad.append(f"{CONS_NAMES[c]} (mean {mu_e:+.2f}, sd {sd_e:.2f}, n={m.sum()})")
    print("   cells with |mean|>0.25 or |sd-1|>0.3:", "; ".join(bad) if bad else "none")
    allm = np.nanmean(np.where(P, G, np.nan)), np.sqrt(np.nanmean(np.where(P, G ** 2, np.nan)))
    print(f"   all present cells pooled: mean {allm[0]:+.3f}, RMS {allm[1]:.3f} (train = ~0 / ~1 by construction).")
    c0 = np.sqrt(np.where(P, G ** 2, 0).sum() / P.sum())
    print(f"   the constant (train-mean) predictor's eval RMSE = RMS(GT) = {c0:.3f} SD -> M2b_s0 present-cell RMSE 0.594 explains {1 - (0.5936 / c0) ** 2:.1%} of the eval variance")


if __name__ == "__main__":
    main()
