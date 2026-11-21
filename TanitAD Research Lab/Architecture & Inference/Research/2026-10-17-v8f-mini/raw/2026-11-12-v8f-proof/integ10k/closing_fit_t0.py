# closing_fit_t0.py : T0 design read for the arm after card PREREG_V8F_PROOF_ADD50. Question: which estimator of the closing speed from the reader's own gap reads
# follows the RECENT label closing speed best? The read beside card ADD49 (closing_diff_t0.py) scored a two-point change over k steps against the label change over the
# same k steps; the rule wants the closing speed at the tick. Rows: the banked reader rows of card ADD46 (rows_LR16_s<k>.npz: clips of the training corpus held out from
# the reader; `fit` marks the clips the maps were fitted on). No evaluation window is read. Tier T0 (decodability on logged frames); never driving performance. Names nothing.
# Convention written into TIMES_NOTE.md before this ran:
#   target A (recent label closing) at row t: least-squares slope of the label gap over the steps t-5 .. t (1.0 s), label lead within 60 m on at least 5 of the 6 steps and at t.
#   estimators from the reader's gap reads (reader present = p >= 0.5), all needing the reader present at t:
#     D<k>  two-point change (g(t) - g(t-k)) / (0.2 k), reader present at t-k                                   k in 5, 10, 15, 25
#     L<w>  least-squares slope of g over the present steps of t-w .. t, at least 60 percent of them present    w in 10, 15, 25
#     X<w>  as L<w> with weights exp(-age / (w / 2)) (age in steps)                                            w in 15, 25
#   per reader seed: slope s of target A on the estimator (least squares through 0) on the FIT clips; on the CHECK clips Pearson r, MAE of s * estimator, MAE of the constant 0.
#   the estimator of the next arm = the one with the highest mean over the two reader seeds of r on the FIT clips;
#   it is called BETTER-THAN-D25 if on the CHECK clips its r is at least 0.05 above that of D25 at both reader seeds, else the next arm keeps D25.
# Usage (pod): python closing_fit_t0.py --rows-dir /workspace/v8proof/integ10k/rows --out CLOSING_FIT_T0.json
import argparse
import json

import numpy as np

DT, G_MAX, P_MIN = 0.2, 60.0, 0.5
EST = [("D", 5), ("D", 10), ("D", 15), ("D", 25), ("L", 10), ("L", 15), ("L", 25), ("X", 15), ("X", 25)]
MARGIN = 0.05


def dense(z):
    """per clip: dict t -> row index"""
    ci, t = np.asarray(z["ci"], np.int64), np.asarray(z["t"], np.int64)
    out = {}
    for c in np.unique(ci):
        m = np.where(ci == c)[0]
        T = int(t[m].max()) + 1
        ix = np.full(T, -1, np.int64)
        ix[t[m]] = m
        out[int(c)] = ix
    return out


def wslope(x, y, w):
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    den = (w * (x - mx) ** 2).sum()
    return float((w * (x - mx) * (y - my)).sum() / den) if den > 1e-12 else np.nan


def build(z):
    D = dense(z)
    p, g = np.asarray(z["p"], np.float64), np.asarray(z["g"], np.float64)
    lp, lg, fit = np.asarray(z["lab_pres"]).astype(bool), np.asarray(z["lab_gap"], np.float64), np.asarray(z["fit"]).astype(bool)
    lab_ok = lp & np.isfinite(lg) & (lg <= G_MAX)
    rd_ok = p >= P_MIN
    n = len(p)
    A = np.full(n, np.nan)
    E = {k: np.full(n, np.nan) for k in EST}
    for c, ix in D.items():
        T = len(ix)
        for t in range(T):
            r = ix[t]
            if r < 0:
                continue
            if lab_ok[r] and t >= 5:
                rr = ix[t - 5:t + 1]
                m = (rr >= 0)
                m[m] = lab_ok[rr[m]]
                if m.sum() >= 5:
                    xs = np.arange(-5, 1)[m] * DT
                    A[r] = wslope(xs, lg[rr[m]], np.ones(int(m.sum())))
            if not rd_ok[r]:
                continue
            for kind, w in EST:
                if t < w:
                    continue
                if kind == "D":
                    r0 = ix[t - w]
                    if r0 >= 0 and rd_ok[r0]:
                        E[(kind, w)][r] = (g[r] - g[r0]) / (DT * w)
                    continue
                rr = ix[t - w:t + 1]
                m = (rr >= 0)
                m[m] = rd_ok[rr[m]]
                if m.sum() >= 0.6 * (w + 1):
                    age = -np.arange(-w, 1)[m].astype(np.float64)
                    wt = np.ones(len(age)) if kind == "L" else np.exp(-age / (w / 2.0))
                    E[(kind, w)][r] = wslope(-age * DT, g[rr[m]], wt)
    return A, E, fit, rd_ok & lab_ok


def stats(a, e, s=None):
    m = np.isfinite(a) & np.isfinite(e)
    if m.sum() < 20:
        return {"n": int(m.sum())}
    a, e = a[m], e[m]
    out = {"n": int(m.sum()), "pearson_r": float(np.corrcoef(a, e)[0, 1]), "mae_zero_mps": float(np.abs(a).mean())}
    if s is not None:
        out["slope_used"] = float(s)
        out["mae_shrunk_mps"] = float(np.abs(s * e - a).mean())
        out["gain_of_shrunk_over_zero"] = float(1.0 - out["mae_shrunk_mps"] / out["mae_zero_mps"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"what": "T0 design read: estimators of the closing speed from the reader's gap reads against the recent label closing speed (slope over 1.0 s); names nothing", "tier": "T0", "margin": MARGIN,
           "per_reader_seed": {}}
    rfit = {k: [] for k in EST}
    rchk = {k: [] for k in EST}
    for seed in (0, 1):
        z = np.load("%s/rows_LR16_s%d.npz" % (a.rows_dir, seed), allow_pickle=False)
        A, E, fit, both = build(z)
        S = {}
        for k in EST:
            e = E[k]
            mf = fit & np.isfinite(A) & np.isfinite(e)
            s = float((A[mf] * e[mf]).sum() / max(1e-9, (e[mf] * e[mf]).sum())) if mf.sum() >= 20 else None
            f, c = stats(A[fit], e[fit]), stats(A[~fit], e[~fit], s)
            cov = float(np.isfinite(e[both & np.isfinite(A)]).mean()) if (both & np.isfinite(A)).any() else None
            S["%s%d" % k] = {"fit_clips": f, "check_clips": c, "share_of_rows_with_estimate": cov}
            rfit[k].append(f.get("pearson_r", np.nan))
            rchk[k].append(c.get("pearson_r", np.nan))
            print("s%d %s%-2d fit: n %5d r %.3f | check: n %5d r %.3f gain %s slope %s cover %.3f" % (
                seed, k[0], k[1], f["n"], f.get("pearson_r", np.nan), c["n"], c.get("pearson_r", np.nan),
                ("%.3f" % c["gain_of_shrunk_over_zero"]) if "gain_of_shrunk_over_zero" in c else "-", ("%.3f" % s) if s is not None else "-", cov if cov is not None else np.nan))
        res["per_reader_seed"][str(seed)] = S
    mean_fit = {k: float(np.mean(v)) for k, v in rfit.items()}
    best = max(EST, key=lambda k: mean_fit[k] if np.isfinite(mean_fit[k]) else -9.0)
    d25 = ("D", 25)
    better = bool(all(np.isfinite(rchk[best][i]) and np.isfinite(rchk[d25][i]) and rchk[best][i] >= rchk[d25][i] + MARGIN for i in (0, 1)))
    res["chosen_on_fit_clips"] = "%s%d" % best
    res["mean_r_fit"] = {"%s%d" % k: v for k, v in mean_fit.items()}
    res["r_check_chosen"] = rchk[best]
    res["r_check_D25"] = rchk[d25]
    res["BETTER_THAN_D25"] = bool(better and best != d25)
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("chosen on the fit clips: %s%d ; check r %s ; D25 check r %s" % (best[0], best[1], ["%.3f" % x for x in rchk[best]], ["%.3f" % x for x in rchk[d25]]))
    print("ZZ CLOSING-FIT-T0 %s" % ("BETTER-THAN-D25" if res["BETTER_THAN_D25"] else "KEEP-D25"))


if __name__ == "__main__":
    main()
