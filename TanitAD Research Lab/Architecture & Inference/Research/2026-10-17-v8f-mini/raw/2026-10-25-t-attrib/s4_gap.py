"""s4_gap.py -- Q4 inputs: where does the T->GT ADE6 gap live, and is it the same place T's constraint error lives?
(EXPLORATORY. Tier: T0 -- the oracle evals feed O the GT code (a diagnostic, never a result); the own-code evals are T1-class open loop on eval-141.
MEASURED from the per-window ade6 of the banked evals; estimator: episode-cluster bootstrap B=2000, ratio-of-sums over episodes.)

gap_full(w)  = ADE6_own(w) - ADE6_GT-code(w)                     (all 60 code dims GT)
gap_cons(w)  = ADE6_own(w) - ADE6_GT-constraints(w)               (only latcons+lonspeed+stoplead GT; decisions are T's own)
nec_g(w)     = ADE6_{GT except g}(w) - ADE6_GT-code(w)            (leave-one-out: the ADE cost of leaving group g at T's value; TAC_A2 read B)
A STRATUM CEILING is  sum_{w in S} gap(w) / sum_all gap(w): the share of the gap a lever would close if it made T's constraints PERFECT in S and changed nothing
outside S (windows are disjoint, so the gain is additive across strata). It is a ceiling, not a forecast.
"""
import numpy as np

from tcommon import *

L = "C:/Users/Admin/box2_ship/logs/"


def boot_ratio(num_w, den_w, eid, ue, n_boot=2000, seed=0):
    r = np.random.default_rng(seed)
    nn = np.array([num_w[eid == e].sum() for e in ue])
    dd = np.array([den_w[eid == e].sum() for e in ue])
    out = np.empty(n_boot)
    n = len(ue)
    for b in range(n_boot):
        k = r.integers(0, n, n)
        out[b] = nn[k].sum() / dd[k].sum()
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def ade_set(seed):
    d = L + "v8f_tacgrp2/eval/"
    z = {}
    z["own"] = np.load(L + f"v8f_mini/eval/eval_M2b_s{seed}.npz")["ade6"]
    z["full"] = np.load(d + f"eval_M2b_s{seed}-tacorc.npz")["ade6"]
    z["cons"] = np.load(d + f"eval_M2b_s{seed}-tacorc-constraints.npz")["ade6"]
    for a in (25, 50, 75):
        z[f"cons_a{a}"] = np.load(d + f"eval_M2b_s{seed}-tacorc-constraints-a{a}.npz")["ade6"]
    for g in ("latcons", "lonspeed", "stoplead", "goal", "latdec", "londec"):
        z[f"all_{g}"] = np.load(d + f"eval_M2b_s{seed}-tacorc-all-{g}.npz")["ade6"]
    return {k: np.asarray(v, np.float64) for k, v in z.items()}


def main():
    T = load()
    eid = T["ci"]
    ue = np.unique(eid)
    has = T["has"]
    for seed in (0, 1):
        arm = f"M2b_s{seed}"
        A = ade_set(seed)
        print("\n" + "=" * 100 + f"\n{arm}: ADE6 means (m) over all 2399 windows -- own {A['own'].mean():.4f}  GT-code {A['full'].mean():.4f}  GT-constraints-only {A['cons'].mean():.4f}\n" + "=" * 100)
        print("   constraints-only dose alpha 0.25/0.50/0.75: " + " / ".join(f"{A[f'cons_a{a}'].mean():.4f}" for a in (25, 50, 75)))
        gf = A["own"] - A["full"]
        gc = A["own"] - A["cons"]
        print(f"   gap_full mean {gf.mean():.4f} m; gap_cons mean {gc.mean():.4f} m ({gc.mean() / gf.mean():.1%} of the full gap); windows with no v9 row: gap_full mean {gf[~has].mean():.4f} (oracle = T's own code there)")
        print("   leave-one-out necessity nec_g mean (m): " + ", ".join(f"{g} {np.mean(A['all_' + g] - A['full']):+.3f}" for g in ("latcons", "lonspeed", "stoplead", "goal", "latdec", "londec")))
        # dose curve share r(alpha) = (own - cons_a)/(own - full)
        print("   dose r(alpha)= (own - cons_alpha)/(own - full): " + ", ".join(f"a{a / 100:.2f} {np.mean(A['own'] - A[f'cons_a{a}']) / gf.mean():.3f}" for a in (25, 50, 75)) + f", a1.00 {gc.mean() / gf.mean():.3f}")

        # --- gap concentration
        print("\n-- concentration of gap_cons over windows / episodes --")
        o = np.argsort(-gc)
        cs = np.cumsum(gc[o]) / gc.sum()
        for q in (0.05, 0.10, 0.25, 0.50):
            k = int(round(q * len(gc)))
            print(f"   worst {q:4.0%} of windows by gap_cons: carry {cs[k - 1]:.3f} of the total gap; windows with NEGATIVE gap_cons (GT constraints made it worse): {(gc < 0).mean():.1%}")
        ee = np.array([gc[eid == e].sum() for e in ue])
        eo = np.argsort(-ee)
        ecs = np.cumsum(ee[eo]) / ee.sum()
        for q in (0.10, 0.25):
            k = int(round(q * len(ue)))
            print(f"   worst {q:4.0%} of episodes by gap_cons: carry {ecs[k - 1]:.3f} of the total gap")

        # --- strata
        nav = T["nav_tok"]
        strata = {
            "GT lat class": np.array([LAT_NAMES.get(int(x), str(x)) for x in T["lat_cls"]]),
            "GT lon class": np.array([LON_NAMES.get(int(x), str(x)) for x in T["lon_cls"]]),
            "ego speed band (m/s)": band(T["v_T"], [1, 5, 10, 15], ["<1", "1-5", "5-10", "10-15", ">=15"]),
            "GT path heading change 6 s": band(np.abs(T["dpsi6"]), [3, 10, 30], ["<3 straight", "3-10", "10-30", ">=30 turn"]),
            "lead agent": np.where(T["lead_valid"] > 0, "lead present", "no lead"),
            "nav token": np.array([f"nav{int(x)}" for x in nav]),
        }
        E, P = sq(T, arm)
        E[~has], P[~has] = 0.0, False
        TOT = E.sum()
        for sname, lab in strata.items():
            print("\n-- stratum: " + sname + f" -- ceiling = share of gap_cons ({arm}); SSEshare = share of T's constraint squared error (s2)")
            print(f"   {'stratum':14s} {'n_win':>6s} {'win%':>6s} {'gap_cons m':>10s} {'gap_full m':>10s} {'CEILING':>8s} {'[CI95]':>16s} {'SSEshare':>8s} {'gap/SSE':>7s}")
            for v in sorted(set(lab.tolist())):
                m = (lab == v) & has
                if m.sum() == 0:
                    continue
                ce = gc[m].sum() / gc[has].sum()
                lo, hi = boot_ratio(np.where(m, gc, 0.0), np.where(has, gc, 0.0), eid, ue)
                sse = E[m].sum() / TOT
                print(f"   {str(v):14s} {m.sum():6d} {m.sum() / has.sum():6.1%} {gc[m].mean():10.3f} {gf[m].mean():10.3f} {ce:8.3f} [{lo:6.3f},{hi:6.3f}] {sse:8.3f} {ce / max(sse, 1e-9):7.2f}")

        # --- which group's T error predicts the ADE gap? (per-window)
        print("\n-- per-window link: gap_cons vs T's constraint SSE by group (Spearman over windows with a v9 row; NNLS slope m per SD^2) --")
        from scipy.stats import spearmanr
        from scipy.optimize import nnls
        X = np.stack([E[:, idx].sum(1) for g, idx in GROUPS.items()], 1)[has]
        y = gc[has]
        for j, g in enumerate(GROUPS):
            rho = spearmanr(X[:, j], y).statistic
            print(f"   {g:10s} Spearman rho(window SSE_g, gap_cons) = {rho:+.3f}")
        coef, rn = nnls(np.hstack([X, np.ones((len(X), 1))]), y)
        pred = np.hstack([X, np.ones((len(X), 1))]) @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print("   NNLS gap_cons ~ b_lat*SSE_lat + b_lon*SSE_lon + b_stp*SSE_stp + c : " + ", ".join(f"{g} {c:.4f}" for g, c in zip(list(GROUPS) + ["const"], coef)) + f"  R2 {r2:.3f}")
        # cell-kind link: regress on SSE by kind
        # (kind map as in s3)
        from s3_tail import KIND
        kinds = sorted(set(KIND.values()))
        Xk = np.stack([E[:, [i for i in KIND if KIND[i] == k]].sum(1) for k in kinds], 1)[has]
        ck, _ = nnls(np.hstack([Xk, np.ones((len(Xk), 1))]), y)
        predk = np.hstack([Xk, np.ones((len(Xk), 1))]) @ ck
        r2k = 1 - ((y - predk) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print("   NNLS gap_cons ~ SSE by cell kind: " + ", ".join(f"{k.split(' ')[0]} {c:.4f}" for k, c in zip(kinds + ["const"], ck)) + f"  R2 {r2k:.3f}")
        # what share of the (fitted) gap does each kind carry: coef * sum(SSE_kind) / sum(fitted without constant)
        contrib = ck[:-1] * Xk.sum(0)
        print("   implied ADE contribution by kind (coef x sum SSE_kind, m summed over windows -> share of the kind-explained part): " +
              ", ".join(f"{k.split(' ')[0]} {c / contrib.sum():.3f}" for k, c in zip(kinds, contrib)))
        np.savez_compressed(S + f"gap_{arm}.npz", own=A["own"], full=A["full"], cons=A["cons"], gc=gc, gf=gf,
                            **{f"nec_{g}": A["all_" + g] - A["full"] for g in ("latcons", "lonspeed", "stoplead", "goal", "latdec", "londec")})


if __name__ == "__main__":
    main()
