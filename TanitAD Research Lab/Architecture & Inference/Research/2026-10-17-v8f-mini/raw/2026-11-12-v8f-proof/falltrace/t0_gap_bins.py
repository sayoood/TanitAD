# t0_gap_bins.py -- lead tool, pod side, CPU only. Tier T0 diagnostic (a property of the head's gap output on recorded frames, not driving performance), no card, no name.
# On the banked gap files of the five 500-clip fits of the head D-H5 (cards ADD102, ADD103, ADD104) and their mean (card ADD105), against the label gap of the evaluation join:
#   A. mean absolute gap error of each fit and of the mean of five (control: the six recorded values of row 168, within 1e-3 m, on 3,841 rows; else one line and nothing more)
#   B. the same error after a causal filter over the last k frames of the clip that have an output (median and mean, k = 3, 5, 9), for fit seed 2 and for the mean of five
#   C. by bin of the label gap [0,10), [10,20), [20,40), [40,60]: rows, mean signed error (fed minus label), mean absolute error, share of rows with the fed gap more than 5 m
#      below and more than 5 m above the label gap, for fit seed 2 and for the mean of five
# Usage: python t0_gap_bins.py            Last line: ZZ T0-GAP-BINS DONE <rows>    or    ZZ T0-GAP-BINS CONTROL FAIL <why>
import json
import sys

import numpy as np

V = "/workspace/v8proof/"
JOIN = V + "hierrep/pins/join_eval141.npz"
FITS = (("fit0", V + "visrule10k/gapfiles/VGAP_s0.npz"), ("fit1", V + "visrule10k/gapfiles/VGAP_s1.npz"), ("fit2", V + "visseed10k/gapfiles/VGAPF_s2.npz"),
        ("fit3", V + "visseed10k/gapfiles/VGAPF_s3.npz"), ("fit4", V + "visseed10k/gapfiles/VGAPF_s4.npz"))
MEAN5 = V + "visens10k/gapfiles/VGAPE5.npz"
EXPECT = {"fit0": 3.960, "fit1": 3.913, "fit2": 4.112, "fit3": 4.079, "fit4": 4.056, "mean5": 3.850}
N_ROWS = 3841
TOL = 1e-3 + 1e-12
BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0001))
KS = (3, 5, 9)
OFF_M = 5.0


def load_gap(p, J):
    """-> (gap [rows of the join], has [rows]) from a gap file (clip_index, frame, gap, has_out; clip names must be the join's)"""
    z = np.load(p, allow_pickle=False)
    if [str(c) for c in z["clips"]] != [str(c) for c in J["clip"]]:
        raise SystemExit("CLIPS-DIFFER %s" % p)
    ci, fr = z["clip_index"].astype(np.int64), z["frame"].astype(np.int64)
    r = J["off"][ci] + fr
    n = len(J["lead_gap"])
    g, h = np.full(n, np.nan), np.zeros(n, bool)
    ok = z["has_out"].astype(bool) & np.isfinite(z["gap"])
    g[r[ok]] = z["gap"][ok].astype(np.float64)
    h[r[ok]] = True
    return g, h


def causal(g, h, J, k, how):
    """filter over the last k frames of the same clip that have an output (the current frame included); NaN where the current frame has no output"""
    out = np.full(len(g), np.nan)
    f = np.median if how == "median" else np.mean
    for c in range(len(J["n"])):
        o, n = int(J["off"][c]), int(J["n"][c])
        for i in range(n):
            if not h[o + i]:
                continue
            a = max(0, i - k + 1)
            w = g[o + a:o + i + 1]
            w = w[h[o + a:o + i + 1]]
            out[o + i] = f(w)
    return out


def main():
    z = np.load(JOIN, allow_pickle=False)
    J = {k: np.asarray(z[k]) for k in ("clip", "n", "off", "lead_gap", "lead_valid")}
    lab = J["lead_gap"].astype(np.float64)
    with np.errstate(invalid="ignore"):
        lok = (J["lead_valid"] == 1) & np.isfinite(lab) & (lab >= 0.0) & (lab <= 60.0)
    G, H = {}, {}
    for name, p in FITS:
        G[name], H[name] = load_gap(p, J)
    sel = lok.copy()
    for name, _p in FITS:
        sel &= H[name]
    G["mean5"], H["mean5"] = load_gap(MEAN5, J)
    err = {name: float(np.abs(G[name][sel] - lab[sel]).mean()) for name in G}
    bad = [] if int(sel.sum()) == N_ROWS else ["rows %d" % int(sel.sum())]
    if not H["mean5"][sel].all():
        bad.append("mean5 lacks an output on %d rows" % int((~H["mean5"][sel]).sum()))
    for name, want in EXPECT.items():
        if not abs(err[name] - want) <= TOL:
            bad.append("%s %.4f" % (name, err[name]))
    if bad:
        print("ZZ T0-GAP-BINS CONTROL FAIL " + "; ".join(bad))
        return 3
    rec = {"tier": "T0", "rows": int(sel.sum()), "mae_m": err, "filters": {}, "bins": {}}
    print("T0 CONTROL PASS rows %d | MAE m: %s" % (sel.sum(), " ".join("%s %.3f" % (k, v) for k, v in err.items())))
    for name in ("fit2", "mean5"):
        for how in ("median", "mean"):
            for k in KS:
                c = causal(G[name], H[name], J, k, how)
                m = float(np.abs(c[sel] - lab[sel]).mean())
                rec["filters"]["%s_%s_k%d" % (name, how, k)] = m
        print("T0 FILTER %s | plain %.3f | median k3 %.3f k5 %.3f k9 %.3f | mean k3 %.3f k5 %.3f k9 %.3f" % (
            name, err[name], *[rec["filters"]["%s_median_k%d" % (name, k)] for k in KS], *[rec["filters"]["%s_mean_k%d" % (name, k)] for k in KS]))
    for name in ("fit2", "mean5"):
        d = G[name] - lab
        for lo, hi in BINS:
            b = sel & (lab >= lo) & (lab < hi)
            n = int(b.sum())
            row = {"n": n, "signed_m": float(d[b].mean()) if n else None, "mae_m": float(np.abs(d[b]).mean()) if n else None,
                   "share_short": float((d[b] < -OFF_M).mean()) if n else None, "share_long": float((d[b] > OFF_M).mean()) if n else None}
            rec["bins"]["%s_%g_%g" % (name, lo, min(hi, 60.0))] = row
            print("T0 BIN %s label gap %2.0f-%2.0f m | rows %4d | signed %+.2f m | MAE %.2f m | fed more than 5 m short %.2f | more than 5 m long %.2f" % (
                name, lo, min(hi, 60.0), n, row["signed_m"], row["mae_m"], row["share_short"], row["share_long"]))
    # D (added after the first full run, logged in TIMES_NOTE before its run): the same by bin of the FED gap: is the head calibrated given its own output?
    rec["fed_bins"] = {}
    for name in ("fit2", "mean5"):
        d = lab - G[name]
        for lo, hi in BINS + ((60.0001, 1e9),):
            with np.errstate(invalid="ignore"):
                b = sel & (G[name] >= lo) & (G[name] < hi)
            n = int(b.sum())
            row = {"n": n, "label_minus_fed_m": float(d[b].mean()) if n else None, "mae_m": float(np.abs(d[b]).mean()) if n else None}
            rec["fed_bins"]["%s_%g_%g" % (name, lo, min(hi, 999.0))] = row
            if n:
                print("T0 FEDBIN %s fed gap %2.0f-%s m | rows %4d | label minus fed %+.2f m | MAE %.2f m" % (name, lo, ("%2.0f" % min(hi, 60.0)) if hi < 1e8 else "up", n, row["label_minus_fed_m"], row["mae_m"]))
    print("ZZJSON " + json.dumps(rec, sort_keys=True))
    print("ZZ T0-GAP-BINS DONE %d" % sel.sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
