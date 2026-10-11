#!/usr/bin/env python3
"""P4 follow-on F2 readout (eval/PREREG_P4_TEACHER_NAVTEST.md §9, EXPLORATORY): tracker-compensated plans vs the same
plans uncompensated, paired on the 12,110 primary tokens, harness CSVs only (each run's status must have been PASS).

  python eval/p4_f2_analyze.py --out <json> [--families]
Registered reads: (i) teacher -- compensated PDMS >= 94.0 AND DAC >= 99.0 => the cap is mostly a plan<->tracker interface
effect; (ii) student -- paired dPDMS lower bound > 0 AND no family separates adversely => a zero-training test-time
lever escalated to the PI; upper bound < 0 => refuted; else not proven.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import proxy_eval as PE  # noqa: E402
import teacher_navtest_analyze as A  # noqa: E402

D = "E:/Projects/TanitAD/data/refe_navtest"
CSV = {"teacher": f"{D}/score/refe_p4_teacher_navtest/refe_p4_teacher_navtest.csv",
       "teacher_trackcomp": f"{D}/score/refe_p4_teacher_trackcomp/refe_p4_teacher_trackcomp.csv",
       "student": A.STUDENT_DEPLOYED_CSV,
       "student_trackcomp": f"{D}/score/refe_p4_student_trackcomp/refe_p4_student_trackcomp.csv",
       "human": f"{D}/score/refe_p4_human_navtest/refe_p4_human_navtest.csv"}
SEAM = {"student": A.STUDENT_DEPLOYED_SEAM, "student_trackcomp": f"{D}/seams/refe_p4_student_trackcomp.npz"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    exp = json.load(gzip.open(A.EXPORT, "rt", encoding="utf-8"))["tokens"]
    tl = {t: exp[t]["log_name"] for t in exp}
    S = {k: A.per_token(p) for k, p in CSV.items() if os.path.exists(p)}
    known = set(json.load(open(A.C1_KNOWN, encoding="utf-8")).get("over_bar", {}))
    res = {"what": "F2: tracker-aware plan compensation (p4_tracker_comp.py, 3 iterations against NAVSIM's own vendored "
                   "LQR + bicycle), paired with the uncompensated plans", "csv": CSV,
           "estimator": "paired log-cluster bootstrap over logs (proxy_eval.boot), 10,000, 95 %, seed 20260927"}
    for arm in ("teacher", "student"):
        c = f"{arm}_trackcomp"
        if arm not in S or c not in S:
            res[arm] = "MISSING CSV"
            continue
        P = sorted(t for t in S[arm] if t in S[c] and t not in known)
        res[arm] = {"n": len(P), "uncompensated": A.block(S[arm], P, tl), "compensated": A.block(S[c], P, tl),
                    "compensated_minus_uncompensated": A.paired(S[c], S[arm], P, tl)}
        if "human" in S:
            res[arm]["compensated_minus_human"] = A.paired(S[c], S["human"], P, tl)
    t = res.get("teacher")
    if isinstance(t, dict):
        p, d = t["compensated"]["PDMS"]["mean"], t["compensated"]["DAC"]["mean"]
        res["teacher_read"] = ("CAP IS MOSTLY THE PLAN<->TRACKER INTERFACE (compensated PDMS >= 94.0 AND DAC >= 99.0)"
                               if p >= 94.0 and d >= 99.0 else
                               f"RESIDUAL CAP after compensation: PDMS {p:.2f}, DAC {d:.2f} (bar 94.0 / 99.0)")
        # same flag as the registered verdict: a threshold inside its interval is reported, never used to flip the read
        res["teacher_read_sensitive_to_log_sampling"] = [
            f"{nm} threshold {th} inside CI {t['compensated'][nm]['ci95']}" for nm, th in (("PDMS", 94.0), ("DAC", 99.0))
            if t["compensated"][nm]["ci95"][0] <= th <= t["compensated"][nm]["ci95"][1]]
    s = res.get("student")
    if isinstance(s, dict):
        mu, (lo, hi) = s["compensated_minus_uncompensated"]["PDMS"]["mean"], s["compensated_minus_uncompensated"]["PDMS"]["ci95"]
        adv = {"missing": True}
        if a.families:
            P = set(t for t in S["student"] if t in S["student_trackcomp"] and t not in known)
            fb = A.families(SEAM["student"], P, "f2_student_uncomp", out_dir)
            fa = A.families(SEAM["student_trackcomp"], P, "f2_student_trackcomp", out_dir)
            adv = PE.fam_adverse(fb, fa) if fb and fa else {"missing": True}
        res["student_families_adverse"] = adv
        no_adv = not adv.get("missing") and not adv.get("adverse")
        if lo > 0 and no_adv:
            v = "LEVER (lower bound > 0, no adverse family) -- escalate to the PI for adoption"
        elif hi < 0:
            v = "REFUTED (upper bound < 0)"
        else:
            v = "NOT PROVEN" + ("" if no_adv else " (families adverse or missing)")
        res["student_read"] = f"{v}: dPDMS {mu:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZF2", res.get("teacher_read"), "|", res.get("student_read"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
