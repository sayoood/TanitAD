#!/usr/bin/env python3
"""RL-S1b: a zero-training DOSE read of R0's EPO effect on RL-val, by LOGIT INTERPOLATION between the same-step BCE-only control and R0.

    s(alpha) = s_C0A + alpha * (s_R0B - s_C0A),   alpha in {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1}

C0-A (BCE only) and R0-B (BCE + EPO) share seed, data order and step count (update 1000), so s_R0B - s_C0A is the EPO term's total effect on the
logits.  First-order argument (ESTIMATED, not measured): at a stationary point of BCE + lam * EPO, the logit shift is ~ linear in lam for small lam
(implicit function theorem), so s(alpha) approximates an arm trained with lam_pg = alpha.  'prog' interpolates ONLY the progress outputs (EP, C,
lane; the safety outputs NC / DAC / TTC stay C0-A's) -- the ceiling of a stop-gradient on the safety heads.  Also: how many picks each scorer
changes versus the base and versus C0-A, and the TTC / EP of the changed ones.  Tier: RL-val lookup (selection tier, diagnosis only).
Estimator: paired log-cluster bootstrap (10,000, seed 20260927) -- another draw of LOGS only.

    python dose_rls1b.py --dl <download root> --out dose.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attrib_rls1b as AT  # noqa: E402

ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    X = os.path.join(a.dl, "data", "x")
    kv, tgv, dv, ckv, _ = AT.load_side(os.path.join(X, "pdm_val", "pdm_val"), os.path.join(X, "terms_val", "terms_val"))
    sv = AT.comps(tgv)
    ld = lambda arm, t, s: AT.load_scores_aligned(os.path.join(a.dl, arm, "final", f"scores_{t}_{s}.npz"), kv, ckv)  # noqa: E731
    base, c, r = ld("r0", "step0", "base"), ld("c0", "update1000", "A"), ld("r0", "update1000", "B")
    n = len(kv)
    ar = np.arange(n)
    ic = AT.picks(c)

    def read(x):
        i = AT.picks(x)
        out = {q: AT.boot_delta(sv[q.lower()][ar, i], sv[q.lower()][ar, ic], kv) for q in ("TTC", "NC", "EP")}
        out["PDMS"] = AT.boot_delta(sv["pdms"][ar, i], sv["pdms"][ar, ic], kv)
        out["picks_changed_vs_C0A"] = int((i != ic).sum())
        out["unsafe_picks"] = int((~sv["safe"][ar, i]).sum())
        return out
    res = {"tool": "dose_rls1b/1", "class": "ESTIMATED (logit interpolation; RL-val selection tier)", "C0A_unsafe_picks": int((~sv["safe"][ar, ic]).sum()),
           "full": {}, "prog": {}}
    for al in ALPHAS:
        res["full"][str(al)] = read(c + al * (r - c))
        xp = c.copy()
        xp[..., [2, 4, 6]] = c[..., [2, 4, 6]] + al * (r[..., [2, 4, 6]] - c[..., [2, 4, 6]])
        res["prog"][str(al)] = read(xp)
    ib, ir = AT.picks(base), AT.picks(r)
    chg = {}
    for nm, i in (("C0A", ic), ("R0B", ir)):
        m = i != ib
        chg[nm] = {"changed_vs_base": int(m.sum()), "TTC_fail_new_picks": int((sv["ttc_fail"][ar, i] & ~sv["ttc_fail"][ar, ib] & m).sum()),
                   "TTC_fixed_picks": int((~sv["ttc_fail"][ar, i] & sv["ttc_fail"][ar, ib] & m).sum()),
                   "PDMS_vs_base": AT.boot_delta(sv["pdms"][ar, i], sv["pdms"][ar, ib], kv),
                   "TTC_vs_base": AT.boot_delta(sv["ttc"][ar, i], sv["ttc"][ar, ib], kv)}
    res["pick_changes_vs_base"] = chg
    # the K3 / K4 checks of section 5 happen at u250 and u500 (and u750 is dumped): the same interpolation at each, against C0-A at the SAME step
    ks = {}
    for t in ("update250", "update500", "update750"):
        ct, rt = ld("c0", t, "A"), ld("r0", t, "B")
        ict = AT.picks(ct)
        row = {}
        for al in (0.25, 0.375, 0.5, 1.0):
            i = AT.picks(ct + al * (rt - ct))
            row[str(al)] = {q: round(100.0 * float((sv[q.lower()][ar, i] - sv[q.lower()][ar, ict]).mean()), 3) for q in ("TTC", "NC", "EP")}
            row[str(al)]["PDMS"] = round(100.0 * float((sv["pdms"][ar, i] - sv["pdms"][ar, ict]).mean()), 3)
            row[str(al)]["g5_ok"] = bool(row[str(al)]["TTC"] >= -0.5 and row[str(al)]["NC"] >= -0.5)
        ks[t] = row
    res["k3_checks_point_x100"] = ks
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("ZZDOSE_OK", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
