# t0_offcentre.py -- lead tool, pod side, CPU only. Tier T0 diagnostic (recorded frames; not driving performance), no card, no name.
# Where is the lead inside the fixed windows of arm D, and does the gap error of arm D grow when the lead is off the window centre?
# Inputs, all banked: the box-centre directions of the pilot of card ADD99 (enc_out/box/<clip>.npz: has_lead, ctr_az, ctr_el, from the label tracks), the dumps of arm D and of the
# bound arm B (head H5, fit seed 0), the evaluation join. Row set: the 3,841 rows of row 168 that have a box centre. The count is printed.
#   A. per bin of the label gap: rows, median of |az| and of |el - EL16| (EL16 = -2 deg, the elevation of the fixed 16-degree window), share with |az| above 2, 4 and 8 deg
#      (8 deg = the half width of that window: the centre of the lead's box is outside it)
#   B. per bin of |az| (0-1, 1-2, 2-4, 4-8, above 8 deg): rows, mean signed error and mean absolute error of arm D and of arm B (fit seed 0), share of arm D more than 5 m short
#   C. the same as B inside the far bin of the label gap (40 to 60 m)
# Control: arm D fit seed 0 gives 3.960 m on the 3,841 rows (within 1e-3 m); else one line and nothing more.
# Usage: python t0_offcentre.py          Last line: ZZ T0-OFFCENTRE DONE <rows>   or   ZZ T0-OFFCENTRE CONTROL FAIL <why>
import json
import os
import sys

import numpy as np

V = "/workspace/v8proof/"
JOIN = V + "hierrep/pins/join_eval141.npz"
FIVE = (V + "visrule10k/gapfiles/VGAP_s0.npz", V + "visrule10k/gapfiles/VGAP_s1.npz", V + "visseed10k/gapfiles/VGAPF_s2.npz", V + "visseed10k/gapfiles/VGAPF_s3.npz",
        V + "visseed10k/gapfiles/VGAPF_s4.npz")
DUMP = V + "cropread/run1/dump/%s_H5/reader_s0_score.npz"
BOX = V + "cropread/enc_out/box"
EL16 = -2.0
GBINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0001))
ABINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9))
TOL = 1e-3 + 1e-12


def load(p, J, has_key=None):
    z = np.load(p, allow_pickle=False)
    if [str(c) for c in z["clips"]] != [str(c) for c in J["clip"]]:
        raise SystemExit("CLIPS-DIFFER")
    r = J["off"][z["clip_index"].astype(np.int64)] + z["frame"].astype(np.int64)
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
    n = len(lab)
    with np.errstate(invalid="ignore"):
        sel5 = (J["lead_valid"] == 1) & np.isfinite(lab) & (lab >= 0.0) & (lab <= 60.0)
    for p in FIVE:
        _g, h = load(p, J, "has_out")
        sel5 &= h
    gD, hD = load(DUMP % "D", J)
    gB, hB = load(DUMP % "B", J)
    e = float(np.abs(gD[sel5] - lab[sel5]).mean()) if hD[sel5].all() else float("nan")
    if int(sel5.sum()) != 3841 or not abs(e - 3.960) <= TOL:
        print("ZZ T0-OFFCENTRE CONTROL FAIL rows %d arm D %.4f" % (sel5.sum(), e))
        return 3
    az, el, hb = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    miss = short = 0
    for c, clip in enumerate(J["clip"]):
        p = os.path.join(BOX, str(clip) + ".npz")
        if not os.path.isfile(p):
            miss += 1
            continue
        b = np.load(p, allow_pickle=False)
        m = min(int(J["n"][c]), len(b["has_lead"]))
        short += int(m != int(J["n"][c]))
        o = int(J["off"][c])
        ok = b["has_lead"][:m].astype(bool) & np.isfinite(b["ctr_az"][:m]) & np.isfinite(b["ctr_el"][:m])
        az[o:o + m] = np.where(ok, b["ctr_az"][:m], np.nan)
        el[o:o + m] = np.where(ok, b["ctr_el"][:m], np.nan)
        hb[o:o + m] = ok
    sel = sel5 & hb & hB
    a, de = np.abs(az), np.abs(el - EL16)
    rec = {"tier": "T0", "rows": int(sel.sum()), "rows_row168": int(sel5.sum()), "clips_without_box_file": miss, "clips_with_other_frame_count": short, "A": {}, "B": {}, "C": {}}
    print("T0 CONTROL PASS rows of row 168: %d | with a box centre and an output of arm B: %d | clips with no box file %d, with another frame count %d | az deg: min %.1f median %.2f max %.1f | el deg: min %.1f median %.2f max %.1f" % (
        sel5.sum(), sel.sum(), miss, short, np.nanmin(az[sel]), np.nanmedian(az[sel]), np.nanmax(az[sel]), np.nanmin(el[sel]), np.nanmedian(el[sel]), np.nanmax(el[sel])))
    for lo, hi in GBINS:
        b = sel & (lab >= lo) & (lab < hi)
        k = int(b.sum())
        row = {"n": k, "med_abs_az": float(np.median(a[b])), "med_abs_el_off": float(np.median(de[b])), "az_gt2": float((a[b] > 2).mean()), "az_gt4": float((a[b] > 4).mean()), "az_gt8": float((a[b] > 8).mean()),
               "el_off_gt4": float((de[b] > 4).mean())}
        rec["A"]["%g_%g" % (lo, min(hi, 60.0))] = row
        print("T0 A label gap %2.0f-%2.0f m | rows %4d | median |az| %.2f deg | median |el - EL16| %.2f deg | |az| above 2 deg %.2f, 4 deg %.2f, 8 deg %.2f | |el - EL16| above 4 deg %.2f" % (
            lo, min(hi, 60.0), k, row["med_abs_az"], row["med_abs_el_off"], row["az_gt2"], row["az_gt4"], row["az_gt8"], row["el_off_gt4"]))
    for tag, base in (("B", sel), ("C", sel & (lab >= 40.0))):
        for lo, hi in ABINS:
            b = base & (a >= lo) & (a < hi)
            k = int(b.sum())
            if not k:
                continue
            dD, dB = gD[b] - lab[b], gB[b] - lab[b]
            row = {"n": k, "D_signed": float(dD.mean()), "D_mae": float(np.abs(dD).mean()), "D_short": float((dD < -5).mean()), "B_signed": float(dB.mean()), "B_mae": float(np.abs(dB).mean())}
            rec[tag]["%g_%s" % (lo, ("%g" % hi) if hi < 1e8 else "up")] = row
            print("T0 %s%s |az| %s deg | rows %4d | arm D signed %+.2f m MAE %.2f m more than 5 m short %.2f | arm B signed %+.2f m MAE %.2f m" % (
                tag, " (label gap 40-60 m)" if tag == "C" else "", ("%g-%g" % (lo, hi)) if hi < 1e8 else ("above %g" % lo), k, row["D_signed"], row["D_mae"], row["D_short"], row["B_signed"], row["B_mae"]))
    print("ZZJSON " + json.dumps(rec, sort_keys=True))
    print("ZZ T0-OFFCENTRE DONE %d" % sel.sum())
    return 0


if __name__ == "__main__":
    sys.exit(main())
