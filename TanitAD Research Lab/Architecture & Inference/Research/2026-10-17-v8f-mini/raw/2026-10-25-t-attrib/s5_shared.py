"""s5_shared.py -- Q2: is T's constraint error SHARED across trained models (a property of the windows / labels / missing information) or MODEL-SPECIFIC?
Plus: where did each tested lever move the error (lever x cell-kind), and two label-side probes (nav vs turn label; label jumps).
EXPLORATORY. MEASURED on eval-141 from the banked per-window T blocks of the seven M2b-family arms (M2b s0/s1, GRID 8x20, HIST 10, PGT, WC, DF50).
Units SD; present cells of windows with a v9 row only (2,365 windows, 26,056 cells). Estimator: episode-cluster bootstrap B=2000 (EPISODES only; a
single-seed arm difference is necessary, not sufficient -- H-ESTIM-SEED-1; the seed floor is read from M2b_s0 vs M2b_s1 in the same table).
"""
import numpy as np
from scipy.stats import spearmanr

from tcommon import *
from s3_tail import KIND

ARMS = M2B_FAMILY
SHORT = {"M2b_s0": "s0", "M2b_s1": "s1", "grid8x20_s0": "grid", "hist10_s0": "hist", "pgt_s0": "pgt", "wc_s0": "wc", "df50_s0": "df50"}


def main():
    T = load()
    eid = T["ci"]
    ue = np.unique(eid)
    has = T["has"]
    G = T["gt_std"]
    P = np.isfinite(G) & has[:, None]
    R = {a: np.where(P, resid(T, a), 0.0) for a in ARMS}
    n_c = P.sum()
    kinds = sorted(set(KIND.values()))
    kidx = {k: [i for i in KIND if KIND[i] == k] for k in kinds}

    print("=" * 100 + "\nQ2a. per-window total squared error: pairwise correlation across arms (Pearson / Spearman over the 2,365 windows with a v9 row)\n" + "=" * 100)
    W = {a: (R[a] ** 2).sum(1)[has] for a in ARMS}
    print("        " + " ".join(f"{SHORT[a]:>11s}" for a in ARMS))
    for a in ARMS:
        row = []
        for b in ARMS:
            pr = np.corrcoef(W[a], W[b])[0, 1]
            sr = spearmanr(W[a], W[b]).statistic
            row.append(f"{pr:5.2f}/{sr:4.2f}")
        print(f"{SHORT[a]:7s} " + " ".join(f"{x:>11s}" for x in row))
    iu = np.triu_indices(len(ARMS), 1)
    pm = np.array([[np.corrcoef(W[a], W[b])[0, 1] for b in ARMS] for a in ARMS])
    sm = np.array([[spearmanr(W[a], W[b]).statistic for b in ARMS] for a in ARMS])
    print(f"off-diagonal Pearson: min {pm[iu].min():.3f} median {np.median(pm[iu]):.3f} max {pm[iu].max():.3f};  Spearman: min {sm[iu].min():.3f} median {np.median(sm[iu]):.3f} max {sm[iu].max():.3f}")
    # per group
    for g, idx in GROUPS.items():
        Wg = {a: (R[a][:, idx] ** 2).sum(1)[has] for a in ARMS}
        c = np.array([[spearmanr(Wg[a], Wg[b]).statistic for b in ARMS] for a in ARMS])
        print(f"   window SSE {g:9s}: Spearman off-diagonal min {c[iu].min():.3f} median {np.median(c[iu]):.3f} max {c[iu].max():.3f}")

    print("\n" + "=" * 100 + "\nQ2b. signed residual correlation across arms (Pearson over present cells) and the SHARED fraction of the squared error\n" + "=" * 100)
    cells = {a: R[a][P] for a in ARMS}
    cm = np.corrcoef(np.stack([cells[a] for a in ARMS]))
    print(f"cell-residual Pearson across arm pairs: min {cm[iu].min():.3f} median {np.median(cm[iu]):.3f} max {cm[iu].max():.3f}  (n_cells {n_c})")
    print(f"   s0-s1 pair: {cm[0, 1]:.3f}")
    # shared fraction: mean cross-product of residual pairs / mean square. = share of MSE that is common to every pair of models.
    def shared(arms, mask):
        X = np.stack([R[a][mask] for a in arms])                 # [K, n]
        K = len(arms)
        msq = (X ** 2).mean()
        xm = X.mean(0)
        cross = ((xm ** 2).mean() * K ** 2 - (X ** 2).mean(1).sum()) / (K * (K - 1))   # mean over ordered pairs a!=b of E[r_a r_b]
        return cross / msq, (xm ** 2).mean() / msq
    for name, arms in (("seed pair {s0,s1}", ["M2b_s0", "M2b_s1"]), ("all 7 M2b-family", ARMS),
                       ("6 seed-0 levers+s0 {s0,grid,hist,pgt,wc,df50}", [a for a in ARMS if a != "M2b_s1"])):
        sh, ens = shared(arms, P)
        print(f"{name:52s}: SHARED fraction of MSE (mean pairwise E[r_a r_b] / mean E[r^2]) = {sh:.3f};  ensemble-mean-residual MSE / mean single MSE = {ens:.3f}")
    print("   per cell KIND (7 arms):")
    for k in kinds:
        m = np.zeros_like(P)
        m[:, kidx[k]] = P[:, kidx[k]]
        sh, ens = shared(ARMS, m)
        print(f"      {k:40s} shared {sh:.3f}   ensemble/single {ens:.3f}   single-model RMSE {np.sqrt(np.mean([(R[a][m] ** 2).mean() for a in ARMS])):.3f}")
    print("   per GROUP (7 arms):")
    for g, idx in GROUPS.items():
        m = np.zeros_like(P)
        m[:, idx] = P[:, idx]
        sh, ens = shared(ARMS, m)
        print(f"      {g:12s} shared {sh:.3f}   ensemble/single {ens:.3f}")
    # high-error windows: are the worst windows the same windows?
    base = W["M2b_s0"]
    top = set(np.argsort(-base)[:int(0.1 * len(base))].tolist())
    print("   overlap of the worst-10%-windows set (by window SSE) between M2b_s0 and each other arm (chance 0.10):")
    print("      " + ", ".join(f"{SHORT[a]} {len(top & set(np.argsort(-W[a])[:int(0.1 * len(base))].tolist())) / len(top):.2f}" for a in ARMS[1:]))
    # the floor: does the *ensemble mean* of the 7 get below the best single arm on the heavy-tail cells?
    Xm = np.stack([R[a] for a in ARMS]).mean(0)
    for k in kinds:
        m = np.zeros_like(P)
        m[:, kidx[k]] = P[:, kidx[k]]
        print(f"   ensemble-of-7 RMSE vs single (mean of 7) by kind: {k:40s} {np.sqrt((Xm[m] ** 2).mean()):.3f} vs {np.sqrt(np.mean([(R[a][m] ** 2).mean() for a in ARMS])):.3f}")

    print("\n" + "=" * 100 + "\nQ2c. what did each tested lever MOVE? MSE change vs M2b_s0 by cell kind (paired over episodes, B=2000); s1 row = the SEED floor\n" + "=" * 100)
    rng = np.random.default_rng(0)
    nb = 2000
    base_s = {k: np.array([(R["M2b_s0"][eid == e][:, kidx[k]] ** 2).sum() for e in ue]) for k in kinds}
    base_n = {k: np.array([P[eid == e][:, kidx[k]].sum() for e in ue]) for k in kinds}
    base_s["ALL"] = sum(base_s[k] for k in kinds)
    base_n["ALL"] = sum(base_n[k] for k in kinds)
    print(f"{'kind':40s} {'MSE s0':>7s} | " + " | ".join(f"{SHORT[a]:>17s}" for a in ARMS[1:]))
    boots = rng.integers(0, len(ue), (nb, len(ue)))
    for k in kinds + ["ALL"]:
        row = []
        for a in ARMS[1:]:
            ssa = np.array([(R[a][eid == e][:, kidx[k] if k != "ALL" else slice(None)] ** 2).sum() for e in ue])
            d = ssa - base_s[k]
            bs = np.array([d[b].sum() / base_n[k][b].sum() for b in boots])
            mse0 = base_s[k].sum() / base_n[k].sum()
            dm = d.sum() / base_n[k].sum()
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sep = "*" if (lo > 0 or hi < 0) else " "
            row.append(f"{dm / mse0:+6.1%}{sep}[{lo / mse0:+5.0%},{hi / mse0:+4.0%}]")
        print(f"{k:40s} {base_s[k].sum() / base_n[k].sum():7.3f} | " + " | ".join(f"{x:>17s}" for x in row))
    print("(* = episode-cluster interval excludes 0; relative % of M2b_s0's own MSE in that kind. One seed: necessary, not sufficient.)")

    print("\n" + "=" * 100 + "\nQ2d. label-side probes\n" + "=" * 100)
    # (i) nav vs turn label
    nav = T["nav_tok"]
    turn = np.isfinite(G[:, 1]) & has
    print("(i) turn-label windows (cells 1-9 present) by the nav token T receives (nav is an inference input):")
    E0, _ = sq(T, "M2b_s0")
    E1, _ = sq(T, "M2b_s1")
    for v in (0, 1, 2):
        m = turn & (nav == v)
        if m.sum() == 0:
            continue
        cols = list(range(1, 10))
        print(f"    nav{v}: n_win {m.sum():4d}; turn-cell RMSE s0 {np.sqrt(E0[m][:, cols].sum() / P[m][:, cols].sum()):.3f} s1 {np.sqrt(E1[m][:, cols].sum() / P[m][:, cols].sum()):.3f}; "
              f"share of turn-cell SSE {E0[m][:, cols].sum() / E0[turn][:, cols].sum():.3f}")
    print(f"    non-turn-label windows with nav token != 0 (route says turn, label says no turn): {int(((~turn) & has & (nav != 0)).sum())} of {int(((nav != 0) & has).sum())} nav-turn windows")
    # (ii) label jumps: error vs |gt(t+1 step) - gt(t)| (a label-stability probe)
    Gn = T["gt_std_next"]
    both = np.isfinite(G) & np.isfinite(Gn) & has[:, None]
    J = np.where(both, np.abs(Gn - G), np.nan)
    print("(ii) squared error by the label's own 0.2 s jump |GT(t+1)-GT(t)| (cells present at both steps; time-to-event cells drift by -0.2 s / sd by construction):")
    bins = [0.02, 0.1, 0.3, 1.0]
    nm = ["<0.02", "0.02-0.1", "0.1-0.3", "0.3-1", ">=1"]
    bj = np.digitize(np.where(both, J, 0), bins)
    ES = E0
    for j, n in enumerate(nm):
        m = both & (bj == j)
        print(f"    jump {n:9s}: n_cells {m.sum():6d} ({m.sum() / both.sum():5.1%}), RMSE {np.sqrt(ES[m].mean()):.3f}, share of SSE (on cells with both) {ES[m].sum() / ES[both].sum():.3f}")
    nj = both & ~np.isfinite(Gn)
    print(f"    cells whose label DISAPPEARS or APPEARS between t and t+1 (presence flips): present now, absent next: {int((P & ~np.isfinite(Gn)).sum())} cells carry {E0[P & ~np.isfinite(Gn)].sum() / E0.sum():.3f} of the SSE")


if __name__ == "__main__":
    main()
