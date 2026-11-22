# t0_gap_bins_scale.py -- lead tool, pod side, CPU only. Tier T0 diagnostic (recorded frames; not driving performance), no card, no name.
# To be run ONLY after the names of card ADD106 are opened: it reads the gap files of that card. Written and logged before any value of the card was read.
# On the rows with a lead in the label and an output of all eight heads (the row set of the card's T0 read): for the 500-clip fit of fit seed 2 (M2), the large fit of fit seed 2 (L2),
# the mean of the three 500-clip fits (E3) and the mean of the three large fits (LE): by bin of the label gap [0,10), [10,20), [20,40), [40,60]: rows, mean signed error
# (fed minus label), mean absolute error, share more than 5 m short, share more than 5 m long. Question fixed in advance: does the larger fit lower the short reading of far leads
# (40 to 60 m: signed error and share short of L2 against M2, of LE against E3)?
# Control: the mean absolute errors of M2 and L2 on the row set equal the values the card's reader printed (passed as --expect M2=<m>,L2=<m>), within 1e-3 m; else one line and nothing more.
# Usage: python t0_gap_bins_scale.py --expect M2=4.1,L2=3.9          Last line: ZZ T0-GAP-BINS-SCALE DONE <rows>   or   ZZ T0-GAP-BINS-SCALE CONTROL FAIL <why>
import argparse
import json
import sys

import numpy as np

V = "/workspace/v8proof/"
JOIN = V + "hierrep/pins/join_eval141.npz"
HEADS = {"O0": V + "visrule10k/gapfiles/VGAP_s0.npz", "O1": V + "visrule10k/gapfiles/VGAP_s1.npz", "M2": V + "visseed10k/gapfiles/VGAPF_s2.npz", "M3": V + "visseed10k/gapfiles/VGAPF_s3.npz",
         "M4": V + "visseed10k/gapfiles/VGAPF_s4.npz", "L2": V + "cropscale/gapfiles/VGAPL_s2.npz", "L3": V + "cropscale/gapfiles/VGAPL_s3.npz", "L4": V + "cropscale/gapfiles/VGAPL_s4.npz"}
MEANS = {"E3": V + "cropscale/gapfiles/VGAPSE3.npz", "LE": V + "cropscale/gapfiles/VGAPLE.npz"}
SHOW = ("M2", "L2", "E3", "LE")
BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0001))
OFF_M = 5.0
TOL = 1e-3 + 1e-12


def load_gap(p, J):
    z = np.load(p, allow_pickle=False)
    if [str(c) for c in z["clips"]] != [str(c) for c in J["clip"]]:
        raise SystemExit("CLIPS-DIFFER %s" % p)
    r = J["off"][z["clip_index"].astype(np.int64)] + z["frame"].astype(np.int64)
    n = len(J["lead_gap"])
    g, h = np.full(n, np.nan), np.zeros(n, bool)
    ok = z["has_out"].astype(bool) & np.isfinite(z["gap"])
    g[r[ok]] = z["gap"][ok].astype(np.float64)
    h[r[ok]] = True
    return g, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", required=True)
    a = ap.parse_args()
    want = {k: float(v) for k, v in (x.split("=") for x in a.expect.split(","))}
    if sorted(want) != ["L2", "M2"]:
        raise SystemExit("--expect needs M2 and L2")
    z = np.load(JOIN, allow_pickle=False)
    J = {k: np.asarray(z[k]) for k in ("clip", "n", "off", "lead_gap", "lead_valid")}
    lab = J["lead_gap"].astype(np.float64)
    with np.errstate(invalid="ignore"):
        sel = (J["lead_valid"] == 1) & np.isfinite(lab) & (lab >= 0.0) & (lab <= 60.0)
    G = {}
    for name, p in HEADS.items():
        G[name], h = load_gap(p, J)
        sel &= h
    bad = []
    for name, p in MEANS.items():
        G[name], h = load_gap(p, J)
        if not h[sel].all():
            bad.append("%s lacks an output on %d rows" % (name, int((~h[sel]).sum())))
    err = {name: float(np.abs(G[name][sel] - lab[sel]).mean()) for name in G}
    for name, w in want.items():
        if not abs(err[name] - w) <= TOL:
            bad.append("%s %.4f against %.4f" % (name, err[name], w))
    if bad:
        print("ZZ T0-GAP-BINS-SCALE CONTROL FAIL " + "; ".join(bad))
        return 3
    rec = {"tier": "T0", "rows": int(sel.sum()), "mae_m": err, "bins": {}}
    print("T0 CONTROL PASS rows %d | MAE m: %s" % (sel.sum(), " ".join("%s %.3f" % (k, err[k]) for k in sorted(err))))
    for name in SHOW:
        d = G[name] - lab
        for lo, hi in BINS:
            b = sel & (lab >= lo) & (lab < hi)
            n = int(b.sum())
            row = {"n": n, "signed_m": float(d[b].mean()) if n else None, "mae_m": float(np.abs(d[b]).mean()) if n else None,
                   "share_short": float((d[b] < -OFF_M).mean()) if n else None, "share_long": float((d[b] > OFF_M).mean()) if n else None}
            rec["bins"]["%s_%g_%g" % (name, lo, min(hi, 60.0))] = row
            print("T0 BIN %s label gap %2.0f-%2.0f m | rows %4d | signed %+.2f m | MAE %.2f m | more than 5 m short %.2f | more than 5 m long %.2f" % (
                name, lo, min(hi, 60.0), n, row["signed_m"], row["mae_m"], row["share_short"], row["share_long"]))
    print("ZZJSON " + json.dumps(rec, sort_keys=True))
    print("ZZ T0-GAP-BINS-SCALE DONE %d" % sel.sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
