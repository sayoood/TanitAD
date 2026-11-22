# t0_gap_bins_arm.py -- lead tool, pod side, CPU only. Tier T0 diagnostic (recorded frames; not driving performance), no card, no name.
# On the banked dumps of the pilot of card ADD99 (head H5, 500-clip fits, fit seeds 0 and 1): arm D (two fixed windows, the deliverable form) against arm B (the same two window
# sizes centred on the lead by the label tracks: a bound, not admissible at inference). By bin of the label gap [0,10), [10,20), [20,40), [40,60]: rows, mean signed error
# (fed minus label), mean absolute error, share more than 5 m short, share more than 5 m long, per arm and fit seed.
# Question fixed in advance: does centring the windows on the lead lower the short reading of far leads (40 to 60 m) that row 170 found for arm D?
# Row set: the 3,841 rows of row 168 (label lead, output of the five 500-clip fits of arm D) that also have an output of both fits of arm B. The count is printed.
# Control: on the 3,841 rows the dumps of arm D, fit seeds 0 and 1, give the recorded errors 3.960 and 3.913 m within 1e-3 m; else one line and nothing more.
# Usage: python t0_gap_bins_arm.py          Last line: ZZ T0-GAP-BINS-ARM DONE <rows>   or   ZZ T0-GAP-BINS-ARM CONTROL FAIL <why>
import json
import sys

import numpy as np

V = "/workspace/v8proof/"
JOIN = V + "hierrep/pins/join_eval141.npz"
FIVE = (V + "visrule10k/gapfiles/VGAP_s0.npz", V + "visrule10k/gapfiles/VGAP_s1.npz", V + "visseed10k/gapfiles/VGAPF_s2.npz", V + "visseed10k/gapfiles/VGAPF_s3.npz",
        V + "visseed10k/gapfiles/VGAPF_s4.npz")
DUMP = V + "cropread/run1/dump/%s_H5/reader_s%d_score.npz"
EXPECT = {("D", 0): 3.960, ("D", 1): 3.913}
N_ROWS = 3841
TOL = 1e-3 + 1e-12
BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0001))
OFF_M = 5.0


def rows_of(z, J):
    if [str(c) for c in z["clips"]] != [str(c) for c in J["clip"]]:
        raise SystemExit("CLIPS-DIFFER")
    return J["off"][z["clip_index"].astype(np.int64)] + z["frame"].astype(np.int64)


def load(p, J, has_key=None):
    z = np.load(p, allow_pickle=False)
    r = rows_of(z, J)
    n = len(J["lead_gap"])
    g, h = np.full(n, np.nan), np.zeros(n, bool)
    ok = np.isfinite(z["gap"]) if has_key is None else (z[has_key].astype(bool) & np.isfinite(z["gap"]))
    g[r[ok]] = z["gap"][ok].astype(np.float64)
    h[r[ok]] = True
    return g, h


def main():
    z = np.load(JOIN, allow_pickle=False)
    J = {k: np.asarray(z[k]) for k in ("clip", "n", "off", "lead_gap", "lead_valid")}
    lab = J["lead_gap"].astype(np.float64)
    with np.errstate(invalid="ignore"):
        sel5 = (J["lead_valid"] == 1) & np.isfinite(lab) & (lab >= 0.0) & (lab <= 60.0)
    for p in FIVE:
        _g, h = load(p, J, "has_out")
        sel5 &= h
    G, H = {}, {}
    for arm in ("D", "B"):
        for s in (0, 1):
            G[(arm, s)], H[(arm, s)] = load(DUMP % (arm, s), J)
    bad = [] if int(sel5.sum()) == N_ROWS else ["rows %d" % int(sel5.sum())]
    for k, want in EXPECT.items():
        if not H[k][sel5].all():
            bad.append("%s%d lacks an output on %d of the rows" % (k[0], k[1], int((~H[k][sel5]).sum())))
        else:
            e = float(np.abs(G[k][sel5] - lab[sel5]).mean())
            if not abs(e - want) <= TOL:
                bad.append("%s%d %.4f against %.3f" % (k[0], k[1], e, want))
    if bad:
        print("ZZ T0-GAP-BINS-ARM CONTROL FAIL " + "; ".join(bad))
        return 3
    sel = sel5 & H[("B", 0)] & H[("B", 1)]
    rec = {"tier": "T0", "rows": int(sel.sum()), "rows_five_fits": int(sel5.sum()), "mae_m": {}, "bins": {}}
    for k in G:
        rec["mae_m"]["%s%d" % k] = float(np.abs(G[k][sel] - lab[sel]).mean())
    print("T0 CONTROL PASS rows of row 168: %d | with an output of both fits of arm B: %d | MAE m: %s" % (sel5.sum(), sel.sum(), " ".join("%s %.3f" % (k, v) for k, v in sorted(rec["mae_m"].items()))))
    for k in (("D", 0), ("D", 1), ("B", 0), ("B", 1)):
        d = G[k] - lab
        for lo, hi in BINS:
            b = sel & (lab >= lo) & (lab < hi)
            n = int(b.sum())
            row = {"n": n, "signed_m": float(d[b].mean()) if n else None, "mae_m": float(np.abs(d[b]).mean()) if n else None,
                   "share_short": float((d[b] < -OFF_M).mean()) if n else None, "share_long": float((d[b] > OFF_M).mean()) if n else None}
            rec["bins"]["%s%d_%g_%g" % (k[0], k[1], lo, min(hi, 60.0))] = row
            print("T0 BIN arm %s fit seed %d label gap %2.0f-%2.0f m | rows %4d | signed %+.2f m | MAE %.2f m | more than 5 m short %.2f | more than 5 m long %.2f" % (
                k[0], k[1], lo, min(hi, 60.0), n, row["signed_m"], row["mae_m"], row["share_short"], row["share_long"]))
    print("ZZJSON " + json.dumps(rec, sort_keys=True))
    print("ZZ T0-GAP-BINS-ARM DONE %d" % sel.sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
