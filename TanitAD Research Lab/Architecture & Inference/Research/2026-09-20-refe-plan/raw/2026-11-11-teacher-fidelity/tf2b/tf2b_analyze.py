#!/usr/bin/env python3
"""TF-2b analysis (PREREG_TF2B.md): CERT vs Tc20s on the 687 fresh held-out navtrain targets (13 logs not in TF-2).
Estimator = TF-2's, verbatim: log-cluster bootstrap (resample LOGS with replacement, ratio mean), 10,000 draws, seed
20260927, 95 % percentile interval.  The verdict reads the UNROUNDED lower bound.
python tf2b_analyze.py --rows <tf2b rows jsonl ...> --ctl <control-row jsonl ...> --tf2 <TF-2 rows jsonl ...>
       --roll-ctl <rollout-control json> --out <json> [--prereg-sha <hex>]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--ctl", nargs="+", required=True)
ap.add_argument("--tf2", nargs="+", required=True)
ap.add_argument("--roll-ctl", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prereg-sha", default="")
a = ap.parse_args()
N_EXPECT, N_MIN, DAC_BAR, C4_BAR, C5_BAR = 687, 680, 2, 7, 68          # PREREG_TF2B s5-s6, frozen
rd = lambda ps: [json.loads(ln) for p in ps for ln in open(p, encoding="utf-8") if ln.strip()]
raw = rd(a.rows)
err = [r for r in raw if "error" in r]
rows = sorted([r for r in raw if "error" not in r], key=lambda r: (r["log"], r["token"]))
tf2 = {(r["log"], r["token"]): r for r in rd(a.tf2) if "error" not in r}
MET = ["nc", "dac", "ddc", "ttc", "ep", "c", "pdms"]
ARMS = ["T_tracked", "Tc20s_tracked", "Tc20s_re_tracked", "M_tracked", "Mw_tracked", "H_tracked", "T_sub20_tracked",
        "T_transparent", "M_transparent"]


def cert(r):
    """L3 (PREREG_TF2 s1, unchanged): first of [Tc20s, T, M] whose TRACKED dac == 1; else the largest tracked margin."""
    cands = ["Tc20s_tracked", "T_tracked", "M_tracked"]
    for c in cands:
        if r[c]["dac"] == 1.0:
            return c
    return max(cands, key=lambda c: r[c]["an"]["min_margin"])


def build(rs):
    ch = [cert(r) for r in rs]
    X = {arm: {m: np.array([r[arm][m] for r in rs]) for m in MET} for arm in ARMS}
    X["CERT_tracked"] = {m: np.array([r[c][m] for r, c in zip(rs, ch)]) for m in MET}
    for arm in X:
        X[arm]["zero"] = (X[arm]["pdms"] == 0).astype(float)
        X[arm]["dac0"] = (X[arm]["dac"] == 0).astype(float)
        X[arm]["nc0"] = (X[arm]["nc"] == 0).astype(float)
    return X, ch


MET2 = MET + ["zero", "dac0", "nc0"]


class Boot:
    def __init__(self, rs):
        logs = np.array([r["log"] for r in rs])
        self.ul, self.inv = np.unique(logs, return_inverse=True)
        self.L = len(self.ul)
        cnt = np.bincount(self.inv, minlength=self.L).astype(float)
        self.draws = np.random.default_rng(20260927).integers(0, self.L, size=(10000, self.L))
        self.den = cnt[self.draws].sum(1)

    def lohi(self, v):
        s = np.bincount(self.inv, weights=v, minlength=self.L)
        e = s[self.draws].sum(1) / self.den
        return float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))

    def ci(self, v):
        lo, hi = self.lohi(v)
        return [round(100 * float(v.mean()), 3), round(100 * lo, 3), round(100 * hi, 3)]


X, choice = build(rows)
n = len(rows)
bt = Boot(rows)
nz = {arm: {"pdms_zero": int(X[arm]["zero"].sum()), "dac_zero": int(X[arm]["dac0"].sum()),
            "nc_zero": int(X[arm]["nc0"].sum())} for arm in X}
res = {"prereg": "PREREG_TF2B.md sha256 " + a.prereg_sha, "n_samples": n, "n_logs": int(bt.L), "n_errors": len(err),
       "errors": err[:8], "log_sizes": {str(u): int((bt.inv == i).sum()) for i, u in enumerate(bt.ul)},
       "estimator": "log-cluster bootstrap (resample the 13 fresh held-out navtrain logs, ratio mean), 10,000 draws, seed "
                    "20260927, 95% percentile; answers 'another draw of LOGS'; labels, tracker, scorer deterministic",
       "tier": "NAVSIM v1.1 PDMS, non-reactive pseudo-simulation on navtrain HELD-OUT logs; NOT navtest",
       "arms": {arm: {m: bt.ci(X[arm][m]) for m in MET2} for arm in X}, "counts": nz,
       "cert_choice": {c: choice.count(c) for c in set(choice)}, "contrasts": {}, "controls": {}, "outcomes": {}}
CON = {"CERT_minus_Tc20s": ("CERT_tracked", "Tc20s_tracked"), "M_minus_Tc20s": ("M_tracked", "Tc20s_tracked"),
       "Mw_minus_Tc20s_REGRESSION": ("Mw_tracked", "Tc20s_tracked"), "CERT_minus_H": ("CERT_tracked", "H_tracked"),
       "CERT_minus_T_transparent": ("CERT_tracked", "T_transparent"), "Tc20s_minus_T_tracked": ("Tc20s_tracked", "T_tracked")}
for nm, (a1, a2) in CON.items():
    res["contrasts"][nm] = {m: bt.ci(X[a1][m] - X[a2][m]) for m in MET2}
al = (X["CERT_tracked"]["pdms"] > 0) & (X["Tc20s_tracked"]["pdms"] > 0)
res["ep_alive_both_CERT_vs_Tc20s"] = {"n": int(al.sum()), "CERT": round(100 * float(X["CERT_tracked"]["ep"][al].mean()), 3),
                                      "Tc20s": round(100 * float(X["Tc20s_tracked"]["ep"][al].mean()), 3)}
dC = X["CERT_tracked"]["pdms"] - X["Tc20s_tracked"]["pdms"]
lo_unrounded, hi_unrounded = bt.lohi(dC)
res["cert_delta_unrounded"] = {"point": float(dC.mean()), "lo": lo_unrounded, "hi": hi_unrounded,
                               "n_rows_cert_differs": int((dC != 0).sum()), "n_rows_cert_ne_Tc20s_choice": int(sum(c != "Tc20s_tracked" for c in choice))}
# descriptive: which logs carry the difference, and leave-one-log-out of the point estimate
byl = {}
for i, u in enumerate(bt.ul):
    m = bt.inv == i
    byl[str(u)] = {"n": int(m.sum()), "sum_dPDMS": round(float(dC[m].sum()), 3), "Tc20s_pdms_zero": int(X["Tc20s_tracked"]["zero"][m].sum()),
                   "Tc20s_dac_zero": int(X["Tc20s_tracked"]["dac0"][m].sum()), "CERT_pdms_zero": int(X["CERT_tracked"]["zero"][m].sum())}
res["by_log"] = byl
res["leave_one_log_out_dPDMS_point"] = {str(u): round(100 * float(dC[bt.inv != i].mean()), 3) for i, u in enumerate(bt.ul)}
# descriptive pooled: TF-2's 1,018 (11 logs) + these 687 (13 logs) = 1,705 rows, 24 logs
tf2rows = sorted(tf2.values(), key=lambda r: (r["log"], r["token"]))
if len(tf2rows) == 1018:
    P = rows + tf2rows
    XP, _ = build(P)
    bp = Boot(P)
    dP = XP["CERT_tracked"]["pdms"] - XP["Tc20s_tracked"]["pdms"]
    res["pooled_descriptive"] = {"n": len(P), "n_logs": int(bp.L),
                                 "CERT_pdms": bp.ci(XP["CERT_tracked"]["pdms"]), "Tc20s_pdms": bp.ci(XP["Tc20s_tracked"]["pdms"]),
                                 "CERT_minus_Tc20s_pdms": bp.ci(dP),
                                 "CERT_dac_zero": int(XP["CERT_tracked"]["dac0"].sum()), "Tc20s_dac_zero": int(XP["Tc20s_tracked"]["dac0"].sum()),
                                 "CERT_pdms_zero": int(XP["CERT_tracked"]["zero"].sum()), "Tc20s_pdms_zero": int(XP["Tc20s_tracked"]["zero"].sum())}

# ---- controls
C = res["controls"]
ro = json.load(open(a.roll_ctl, encoding="utf-8"))
C["C0_rollout_reproduces_bank"] = {"pass": bool(ro["pass"]), "n": ro["n_ok"], "median_max_dxy_m": ro["median_max_dxy_m"],
                                    "share_le_0.25": ro["share_le_0.25"], "p90": ro["p90_max_dxy_m"], "max": ro["max_max_dxy_m"],
                                    "n_error": ro["n_error"], "derivation_max_dev_m": ro["derivation_max_dev_m"]}
ctl = sorted([r for r in rd(a.ctl) if "error" not in r], key=lambda r: (r["log"], r["token"]))
md, miss = 0.0, 0
for r in ctl:
    t = tf2.get((r["log"], r["token"]))
    if t is None:
        miss += 1
        continue
    for arm in ARMS:
        md = max(md, max(abs(r[arm][m] - t[arm][m]) for m in MET))
C["C1_scoring_identical_to_TF2_on_control_rows"] = {"pass": bool(md == 0.0 and miss == 0 and len(ctl) >= 30), "n": len(ctl),
                                                     "max_abs_diff": md, "missing": miss}
c2 = max(r["info"]["c2_label_maxdev_m"] for r in rows)
c2b = max(max(abs(r["Tc20s_re_tracked"][m] - r["Tc20s_tracked"][m]) for m in MET) for r in rows)
C["C2_derivation_identity"] = {"pass": bool(c2 <= 1e-9 and c2b == 0.0), "label_maxdev_m": c2, "tracked_max_abs_diff": c2b}
un = [r for r in rows if r["info"]["c3_unshifted"]]
C["C3_unshifted_rows_identical"] = {"pass": all(r["info"]["c3_label_eq_re"] for r in un), "n_unshifted": len(un),
                                     "n_bad": sum(not r["info"]["c3_label_eq_re"] for r in un)}
C["C4_regression_Mw"] = {"pass": nz["Mw_tracked"]["dac_zero"] >= nz["Tc20s_tracked"]["dac_zero"] + C4_BAR,
                         "Mw_dac_zero": nz["Mw_tracked"]["dac_zero"], "Tc20s_dac_zero": nz["Tc20s_tracked"]["dac_zero"], "bar": C4_BAR}
nd = sum(not r["info"]["c3_label_eq_re"] for r in rows)
C["C5_liveness"] = {"pass": nd >= C5_BAR, "n_label_differs": nd, "bar": C5_BAR}
C["C6_coverage"] = {"pass": n >= N_MIN, "n": n, "expected": N_EXPECT, "bar": N_MIN, "errors": len(err)}
allpass = all(v["pass"] for v in C.values())
res["controls_all_pass"] = allpass

# ---- registered outcome (PREREG_TF2B s6)
O = res["outcomes"]
O["CERT"] = ("NO VERDICT (controls failed)" if not allpass else
             "WORKS" if (nz["CERT_tracked"]["dac_zero"] <= DAC_BAR and lo_unrounded > 0) else "NOT SHOWN")
O["CERT_conditions"] = {"cert_dac_zero<=2": nz["CERT_tracked"]["dac_zero"] <= DAC_BAR, "dPDMS_ci_lo>0 (unrounded)": lo_unrounded > 0}
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
print("ZZTF2B_CONTROLS", "PASS" if allpass else "FAIL", json.dumps({k: v["pass"] for k, v in C.items()}))
print("ZZTF2B_OUTCOME", json.dumps(O))
for arm in ("T_tracked", "Tc20s_tracked", "M_tracked", "CERT_tracked", "Mw_tracked", "H_tracked", "T_transparent"):
    print(f"{arm:16s} PDMS {res['arms'][arm]['pdms']} DAC {res['arms'][arm]['dac'][0]} NC {res['arms'][arm]['nc'][0]} "
          f"EP {res['arms'][arm]['ep'][0]} TTC {res['arms'][arm]['ttc'][0]} C {res['arms'][arm]['c'][0]} {nz[arm]}")
print("CERT-Tc20s", res["contrasts"]["CERT_minus_Tc20s"]["pdms"], "unrounded", res["cert_delta_unrounded"])
print("EP alive", res["ep_alive_both_CERT_vs_Tc20s"], "choice", res["cert_choice"])
print("pooled", res.get("pooled_descriptive"))
