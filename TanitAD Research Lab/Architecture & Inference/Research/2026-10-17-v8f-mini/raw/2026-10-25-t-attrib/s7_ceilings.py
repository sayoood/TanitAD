"""s7_ceilings.py -- Q4: lever ceilings, with the arithmetic. (EXPLORATORY; decides nothing; it ranks the next pre-registration.)

Two kinds of ceiling, kept apart because they have different evidence classes:
  (E) EXACT window-stratum ceiling  [MEASURED, tier T0 oracle]: sum_{w in S} (ADE6_own(w) - ADE6_GT-constraints(w)) / N_all, in metres of mean ADE6, and as a share of
      the T->GT gap. The GT-constraints eval feeds O the GT latcons+lonspeed+stoplead (decisions stay T's own); windows are disjoint, so the gain of fixing S is
      additive. It assumes T's constraints become PERFECT in S, so it is an upper bound on any lever acting on S.
  (D) DOSE-MODEL estimate for a CELL stratum  [ESTIMATED]: banked evals move ALL constraint cells a fraction alpha of the way to the GT (Amendment 2 read C). A lever
      that removes a share phi of the total squared error is mapped to the uniform-shrink equivalent alpha_eq = 1 - sqrt(1 - phi), and the measured r(alpha) is
      interpolated. This presumes O responds to the SIZE of the error, not to WHICH cell it sits in -- an assumption the data cannot test (no per-cell-kind
      oracle exists), flagged as such.
Bar: 2F = 0.2415 m (twice the M2b seed floor F = |ADE6 s0 - ADE6 s1| = 0.1207 m).
"""
import numpy as np
from scipy.optimize import nnls

from tcommon import *
from s3_tail import KIND

TWO_F = 0.2415
ALPHAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def boot(num_w, eid, ue, n_all, nb=2000):
    r = np.random.default_rng(0)
    s = np.array([num_w[eid == e].sum() for e in ue])
    out = np.array([s[k].sum() / n_all for k in r.integers(0, len(ue), (nb, len(ue)))])
    return np.percentile(out, [2.5, 97.5])


def main():
    T = load()
    eid = T["ci"]
    ue = np.unique(eid)
    has = T["has"]
    N = len(eid)
    G = T["gt_std"]
    nav = T["nav_tok"]
    kinds = sorted(set(KIND.values()))
    out = {}
    for seed in (0, 1):
        arm = f"M2b_s{seed}"
        g = np.load(S + f"gap_{arm}.npz")
        gc, gf = g["gc"], g["gf"]
        E, P = sq(T, arm)
        E[~has], P[~has] = 0.0, False
        wsse = E.sum(1)
        TOT = E.sum()
        print("\n" + "=" * 110 + f"\n{arm}: T->GT gap_full {gf.mean():.4f} m, gap_cons {gc.mean():.4f} m (mean ADE6 over {N} windows); bar 2F = {TWO_F} m = {TWO_F / gf.mean():.1%} of gap_full\n" + "=" * 110)
        print("(E) EXACT window-stratum ceilings (GT constraints in S, T's own elsewhere):")
        print(f"   {'stratum S':58s} {'n_win':>6s} {'win%':>6s} {'SSEshare':>8s} {'ceiling m':>9s} {'[CI95]':>16s} {'share of gap_cons':>17s} {'>2F?':>5s}")
        lead_tg_big = (T["lead_valid"] > 0) & ((np.abs(np.where(np.isfinite(G[:, 19]), G[:, 19], 0)) >= 3) | (np.abs(np.where(np.isfinite(G[:, 21]), G[:, 21], 0)) >= 3))
        turn = np.isfinite(G[:, 1]) & has
        o = np.argsort(-np.where(has, wsse, -1))
        top10 = np.zeros(N, bool); top10[o[:int(0.1 * has.sum())]] = True
        top25 = np.zeros(N, bool); top25[o[:int(0.25 * has.sum())]] = True
        top50 = np.zeros(N, bool); top50[o[:int(0.5 * has.sum())]] = True
        lon = T["lon_cls"]
        sp = T["v_T"]
        S_ = {
            "top-10% windows by T's constraint SSE": top10,
            "top-25% windows by T's constraint SSE": top25,
            "bottom-50% windows by T's constraint SSE": has & ~top50,
            "turn-label windows (any nav)": turn,
            "turn-label windows whose nav token = FOLLOW (route silent)": turn & (nav == 0),
            "turn-label windows whose nav token = TURN (route speaks)": turn & (nav != 0),
            "lead present": has & (T["lead_valid"] > 0),
            "lead present AND heavy-tail lead_tg label (|GT|>=3 SD)": has & lead_tg_big,
            "lon class FOLLOW": has & (lon == 3),
            "lon class DECELERATE/ACCELERATE": has & ((lon == 4) | (lon == 5)),
            "lon class STOP/HOLD/CREEP": has & ((lon == 2) | (lon == 0) | (lon == 1)),
            "lon class KEEP": has & (lon == 6),
            "ego speed 5-10 m/s": has & (sp >= 5) & (sp < 10),
            "ego speed >= 10 m/s": has & (sp >= 10),
            "ALL windows with a v9 row": has,
        }
        for nm, m in S_.items():
            lo, hi = boot(np.where(m, gc, 0.0), eid, ue, N)
            cm = gc[m].sum() / N
            print(f"   {nm:58s} {m.sum():6d} {m.sum() / has.sum():6.1%} {E[m].sum() / TOT:8.3f} {cm:9.3f} [{lo:6.3f},{hi:6.3f}] {gc[m].sum() / gc[has].sum():17.3f} {'yes' if cm > TWO_F else 'no':>5s}")
        # decile table: does T's window error predict where the gap is?
        print("   gap_cons by DECILE of window T-SSE (decile 10 = worst):")
        ranks = np.argsort(np.argsort(np.where(has, wsse, np.inf)))
        hs = np.where(has)[0]
        hs_sorted = hs[np.argsort(wsse[hs])]
        dec = np.array_split(hs_sorted, 10)
        print("      " + " ".join(f"{gc[d].mean():5.2f}" for d in dec) + f"   (mean gap_cons m per window; overall {gc[has].mean():.2f})")
        print("      SSE share by decile: " + " ".join(f"{E[d].sum() / TOT:5.3f}" for d in dec))

        # (D) dose model
        gfm = gf.mean()
        print(f"\n(D) DOSE-MODEL for cell strata  [ESTIMATED]: banked r(alpha) = share of gap_full recovered by moving all constraints alpha toward the GT:")
        A = ade_curve(seed)
        rr = np.array([(A["own"] - A[k]).mean() / gfm for k in ("own", "a25", "a50", "a75", "cons")])
        print("   alpha " + " ".join(f"{a:5.2f}" for a in ALPHAS) + "\n   r     " + " ".join(f"{r:5.3f}" for r in rr)
              + f"   -> gain m: " + " ".join(f"{r * gfm:5.3f}" for r in rr))
        # NOTE: the alpha=1 point of the constraints-only oracle is gap_cons/gap_full (0.89), not 1.0; keep as measured.
        def r_of_phi(phi):
            a = 1 - np.sqrt(max(1 - phi, 0.0))
            return float(np.interp(a, ALPHAS, rr)), a
        print(f"   {'cell stratum (all of its error removed)':58s} {'phi=SSEshare':>12s} {'alpha_eq':>8s} {'r':>6s} {'gain m':>7s} {'>2F?':>5s}")
        cellsets = {k: [i for i in KIND if KIND[i] == k] for k in kinds}
        cellsets["TIME-TO-EVENT + TIME-GAP (the T-error-dominant cells)"] = [i for i in KIND if KIND[i].startswith("TIME")]
        cellsets["lead_tg_s + lead_tg_min_s heavy-tail cells only (|GT|>=3)"] = None
        cellsets["ALL cells of the 3 groups"] = list(range(26))
        for k, idx in cellsets.items():
            if idx is None:
                m = P.copy(); m[:, :] = False
                for c in (19, 21):
                    m[:, c] = P[:, c] & (np.abs(np.where(P[:, c], G[:, c], 0)) >= 3)
                phi = E[m].sum() / TOT
            else:
                phi = E[:, idx].sum() / TOT
            r, a = r_of_phi(phi)
            print(f"   {k:58s} {phi:12.3f} {a:8.3f} {r:6.3f} {r * gfm:7.3f} {'yes' if r * gfm > TWO_F else 'no':>5s}")
        # lever-sized reductions of the whole block
        print("   uniform lever-sized cuts of T's constraint RMSE (alpha = RMSE cut):")
        for a, why in ((0.02, "seed-to-seed |s1-s0| level"), (0.054, "ensemble of 7 (measured RMSE cut, s5)"), (0.06, "DSCALE 2x data (PI-quoted -6%)"), (0.10, "an optimistic single lever"),
                       (0.25, "dose point"), (0.40, "")):
            if a < 0.25:
                # below the first measured point: power-law through the origin fitted to (0.25, r25) and (0.5, r50)
                b = np.log(rr[2] / rr[1]) / np.log(2.0)
                rv = rr[1] * (a / 0.25) ** b
                lin = rr[1] * a / 0.25
                print(f"      alpha {a:5.3f} ({why:42s}): r power-law {rv:6.3f} -> {rv * gfm:6.3f} m;  linear (generous) {lin:6.3f} -> {lin * gfm:6.3f} m")
            else:
                rv = float(np.interp(a, ALPHAS, rr))
                print(f"      alpha {a:5.3f} ({why:42s}): r interp {rv:6.3f} -> {rv * gfm:6.3f} m")
        need = float(np.interp(TWO_F / gfm, rr, ALPHAS))
        print(f"   ==> to clear 2F ({TWO_F / gfm:.3f} of gap_full) the constraint RMSE must fall by alpha = {need:.2f} across the block ({1 - (1 - need) ** 2:.0%} of the squared error)")
        # within-group ADE relevance: nec_g(w) ~ SSE by kind in group g
        print("\n   WITHIN-GROUP ADE relevance (NNLS of the per-window leave-one-out necessity nec_g(w) on that group's cell-kind SSE; ESTIMATED, R2 reported):")
        for gname, idx in GROUPS.items():
            y = g[f"nec_{gname}"][has]
            ks = sorted({KIND[i] for i in idx})
            X = np.stack([E[:, [i for i in idx if KIND[i] == k]].sum(1)[has] for k in ks], 1)
            c, _ = nnls(np.hstack([X, np.ones((len(X), 1))]), y)
            pred = np.hstack([X, np.ones((len(X), 1))]) @ c
            r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            contrib = c[:-1] * X.sum(0)
            print(f"      {gname:9s} mean nec {y.mean():+.3f} m; R2 {r2:.3f}; " + "; ".join(f"{k.split(' ')[0]} coef {cc:.3f} (share of explained {ct / max(contrib.sum(), 1e-9):.2f}, SSE-share-in-group {X[:, j].sum() / X.sum():.2f})"
                                                                                         for j, (k, cc, ct) in enumerate(zip(ks, c[:-1], contrib))))
        out[arm] = dict(rr=rr.tolist(), gap_full=float(gfm), gap_cons=float(gc.mean()))
    return out


def ade_curve(seed):
    L = "C:/Users/Admin/box2_ship/logs/"
    d = L + "v8f_tacgrp2/eval/"
    own = np.load(L + f"v8f_mini/eval/eval_M2b_s{seed}.npz")["ade6"].astype(np.float64)
    full = np.load(d + f"eval_M2b_s{seed}-tacorc.npz")["ade6"].astype(np.float64)
    r = {"own": own, "full": full, "cons": np.load(d + f"eval_M2b_s{seed}-tacorc-constraints.npz")["ade6"].astype(np.float64)}
    for a in (25, 50, 75):
        r[f"a{a}"] = np.load(d + f"eval_M2b_s{seed}-tacorc-constraints-a{a}.npz")["ade6"].astype(np.float64)
    return r


if __name__ == "__main__":
    main()
