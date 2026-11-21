#!/usr/bin/env python3
"""TF-2 analysis (PREREG_TF2.md): controls C1-C6, the registered outcomes (M primary, CERT secondary, P-S20 probe),
log-cluster bootstrap (TF-1's estimator verbatim: 11 held-out LOGS resampled, ratio mean, 10,000 draws, seed 20260927,
95 % percentile), and the residual-failure anatomy of TF-1's 11 Tc20s-tracked zero rows.
python tf2_analyze.py --rows <tf2_s*.jsonl ...> --tf1 <tf1 rows jsonl ...> --out <json>"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--tf1", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(ln) for p in a.rows for ln in open(p, encoding="utf-8") if ln.strip()]
err = [r for r in rows if "error" in r]
rows = sorted([r for r in rows if "error" not in r], key=lambda r: (r["log"], r["token"]))
keys = [(r["log"], r["token"]) for r in rows]
tf1 = {(r["log"], r["token"]): r for r in (json.loads(ln) for p in a.tf1 for ln in open(p, encoding="utf-8")) if "error" not in r}
MET = ["nc", "dac", "ddc", "ttc", "ep", "c", "pdms"]
ARMS = ["T_tracked", "Tc20s_tracked", "Tc20s_re_tracked", "M_tracked", "Mw_tracked", "H_tracked", "T_sub20_tracked",
        "T_transparent", "M_transparent"]


def cert(r):
    """L3 (PREREG s1): first of [Tc20s, T, M] whose TRACKED dac == 1; else the largest tracked min margin.
    Reads ONLY `dac` and the map-only margin -- never nc / ttc / ep."""
    cands = ["Tc20s_tracked", "T_tracked", "M_tracked"]
    for c in cands:
        if r[c]["dac"] == 1.0:
            return c
    return max(cands, key=lambda c: r[c]["an"]["min_margin"])


choice = [cert(r) for r in rows]
X = {arm: {m: np.array([r[arm][m] for r in rows]) for m in MET} for arm in ARMS}
X["CERT_tracked"] = {m: np.array([r[c][m] for r, c in zip(rows, choice)]) for m in MET}
for arm in X:
    X[arm]["zero"] = (X[arm]["pdms"] == 0).astype(float)
    X[arm]["dac0"] = (X[arm]["dac"] == 0).astype(float)
    X[arm]["nc0"] = (X[arm]["nc"] == 0).astype(float)
MET2 = MET + ["zero", "dac0", "nc0"]
logs = np.array([k[0] for k in keys])
ul, inv = np.unique(logs, return_inverse=True)
n, L = len(rows), len(ul)
rng = np.random.default_rng(20260927)
B = 10000
cnt = np.bincount(inv, minlength=L).astype(float)
draws = rng.integers(0, L, size=(B, L))
den = cnt[draws].sum(1)


def boot(v):
    s = np.bincount(inv, weights=v, minlength=L)
    return s[draws].sum(1) / den


def ci(v):
    e = boot(v)
    return [round(100 * float(v.mean()), 3), round(100 * float(np.percentile(e, 2.5)), 3),
            round(100 * float(np.percentile(e, 97.5)), 3)]


res = {"prereg": "PREREG_TF2.md sha256 65189b98841d3ae5fa0e0c0ba4d436ea55380373f4c63801c852e48ca9eca2bf",
       "n_samples": n, "n_logs": int(L), "n_errors": len(err), "errors": err[:5],
       "estimator": "log-cluster bootstrap (resample the 11 held-out navtrain logs with replacement, ratio mean), 10,000 "
                    "draws, seed 20260927, 95% percentile interval; answers 'another draw of LOGS'; labels, tracker and "
                    "scorer are deterministic",
       "tier": "NAVSIM v1.1 PDMS, non-reactive pseudo-simulation (LQR + kinematic bicycle) on navtrain HELD-OUT logs; NOT navtest",
       "arms": {arm: {m: ci(X[arm][m]) for m in MET2} for arm in X},
       "counts": {arm: {"pdms_zero": int(X[arm]["zero"].sum()), "dac_zero": int(X[arm]["dac0"].sum()),
                        "nc_zero": int(X[arm]["nc0"].sum())} for arm in X},
       "cert_choice": {c: choice.count(c) for c in set(choice)}, "contrasts": {}, "controls": {}, "outcomes": {}}
CON = {"M_minus_Tc20s": ("M_tracked", "Tc20s_tracked"), "CERT_minus_Tc20s": ("CERT_tracked", "Tc20s_tracked"),
       "Mw_minus_Tc20s_REGRESSION": ("Mw_tracked", "Tc20s_tracked"), "M_minus_H": ("M_tracked", "H_tracked"),
       "CERT_minus_H": ("CERT_tracked", "H_tracked"), "M_minus_T_transparent": ("M_tracked", "T_transparent"),
       "M_transparent_minus_T_transparent": ("M_transparent", "T_transparent"),
       "T_sub20_minus_T_tracked": ("T_sub20_tracked", "T_tracked"),
       "T_sub20_minus_T_transparent": ("T_sub20_tracked", "T_transparent"),
       "Tc20s_minus_T_tracked": ("Tc20s_tracked", "T_tracked")}
for nm, (a1, a2) in CON.items():
    res["contrasts"][nm] = {m: ci(X[a1][m] - X[a2][m]) for m in MET2}
alive =(X["M_tracked"]["pdms"] > 0) & (X["Tc20s_tracked"]["pdms"] > 0)
res["ep_alive_both_M_vs_Tc20s"] = {"n": int(alive.sum()), "M": round(100 * float(X["M_tracked"]["ep"][alive].mean()), 3),
                                   "Tc20s": round(100 * float(X["Tc20s_tracked"]["ep"][alive].mean()), 3)}
aliveC = (X["CERT_tracked"]["pdms"] > 0) & (X["Tc20s_tracked"]["pdms"] > 0)
res["ep_alive_both_CERT_vs_Tc20s"] = {"n": int(aliveC.sum()), "CERT": round(100 * float(X["CERT_tracked"]["ep"][aliveC].mean()), 3),
                                      "Tc20s": round(100 * float(X["Tc20s_tracked"]["ep"][aliveC].mean()), 3)}

# ---- controls
C = res["controls"]
diffs = {}
for arm in ("T_tracked", "Tc20s_tracked", "H_tracked", "T_transparent"):
    md = 0.0
    miss = 0
    for r in rows:
        t = tf1.get((r["log"], r["token"]))
        if t is None:
            miss += 1
            continue
        md = max(md, max(abs(r[arm][m] - t[arm][m]) for m in MET))
    diffs[arm] = {"max_abs_diff": md, "missing_in_tf1": miss}
C["C1_identity_vs_TF1"] = {"pass": all(v["max_abs_diff"] == 0.0 and v["missing_in_tf1"] == 0 for v in diffs.values()), **diffs}
c2 = max(r["info"]["c2_label_maxdev_m"] for r in rows)
c2b = max(max(abs(r["Tc20s_re_tracked"][m] - r["Tc20s_tracked"][m]) for m in MET) for r in rows)
C["C2_derivation_identity"] = {"pass": bool(c2 <= 1e-9 and c2b == 0.0), "label_maxdev_m": c2, "tracked_max_abs_diff": c2b}
un = [r for r in rows if r["info"]["c3_unshifted"]]
C["C3_unshifted_rows_identical"] = {"pass": all(r["info"]["c3_label_eq_re"] for r in un), "n_unshifted": len(un),
                                     "n_bad": sum(not r["info"]["c3_label_eq_re"] for r in un)}
nz = res["counts"]
C["C4_regression_Mw"] = {"pass": nz["Mw_tracked"]["dac_zero"] >= nz["Tc20s_tracked"]["dac_zero"] + 10,
                         "Mw_dac_zero": nz["Mw_tracked"]["dac_zero"], "Tc20s_dac_zero": nz["Tc20s_tracked"]["dac_zero"]}
nd = sum(not r["info"]["c3_label_eq_re"] for r in rows)
C["C5_liveness"] = {"pass": nd >= 100, "n_label_differs": nd}
C["C6_coverage"] = {"pass": n == 1018 and not err, "n": n, "errors": len(err)}
allpass = all(v["pass"] for v in C.values())
res["controls_all_pass"] = allpass

# ---- registered outcomes (PREREG s6)
O = res["outcomes"]
dM = res["contrasts"]["M_minus_Tc20s"]["pdms"]
mz, tz = nz["M_tracked"], nz["Tc20s_tracked"]
ep = res["ep_alive_both_M_vs_Tc20s"]
condA = [mz["dac_zero"] <= 4, dM[1] > 0, mz["nc_zero"] <= tz["nc_zero"] + 1, ep["M"] >= ep["Tc20s"] - 0.5]
condC = dM[0] < 0 or mz["nc_zero"] >= tz["nc_zero"] + 2 or ep["M"] < ep["Tc20s"] - 0.5
O["PRIMARY_M"] = ("NO VERDICT (controls failed)" if not allpass else
                  "A: L2b WORKS" if all(condA) else "C: REFUTED IN THIS FORM" if condC else "B: PARTIAL")
O["PRIMARY_M_conditions"] = {"dac_zero<=4": condA[0], "dPDMS_ci_lo>0": condA[1], "nc_zero<=Tc20s+1": condA[2],
                             "ep_alive_loss<=0.5": condA[3]}
dC = res["contrasts"]["CERT_minus_Tc20s"]["pdms"]
O["SECONDARY_CERT"] = ("NO VERDICT (controls failed)" if not allpass else
                       "WORKS" if (nz["CERT_tracked"]["dac_zero"] <= 4 and dC[1] > 0) else "NOT SHOWN")
s20, tt, ttr = res["arms"]["T_sub20_tracked"], res["arms"]["T_tracked"], res["arms"]["T_transparent"]
O["PROBE_S20"] = ("S20-CONSISTENT" if (s20["dac"][0] >= 99.0 and s20["pdms"][0] >= ttr["pdms"][0] - 1.0) else
                  "S20-NO" if s20["dac"][0] <= tt["dac"][0] + 2.0 else "S20-PARTIAL")

# ---- anatomy of TF-1's residual rows (Tc20s-tracked PDMS 0) and the transparent DAC-zero rows
CORNER = ["FL", "RL", "RR", "FR"]
an = []
for r in rows:
    if r["Tc20s_tracked"]["pdms"] == 0 or r["T_transparent"]["dac"] == 0:
        tc, tt_, h = r["Tc20s_tracked"], r["T_transparent"], r["H_tracked"]
        kind = "NC" if tc["nc"] == 0 else ("DAC" if tc["dac"] == 0 else "other")
        owner = ("policy" if tt_["dac"] == 0 else "tracker") if kind == "DAC" else (
                 "tracker" if tt_["nc"] == 1 else "policy")
        an.append({"log": r["log"], "token": r["token"], "kind": kind, "owner": owner,
                   "v0": round(r["info"]["v0"], 2), "turn_deg": round(r["info"]["turn_deg"], 1),
                   "path_m": round(r["info"]["path_m"], 1),
                   "Tc20s_tracked": {"margin_m": round(tc["an"]["min_margin"], 3), "first_viol_t": tc["an"]["first_viol_t"],
                                     "worst_t": tc["an"]["worst_t"], "corner": CORNER[tc["an"]["worst_corner"]],
                                     "in_intersection": tc["an"]["worst_in_intersection"], "coll_t": tc["an"]["coll_t"]},
                   "T_transparent": {"dac": tt_["dac"], "margin_m": round(tt_["an"]["min_margin"], 3),
                                     "first_viol_t": tt_["an"]["first_viol_t"], "corner": CORNER[tt_["an"]["worst_corner"]],
                                     "in_intersection": tt_["an"]["worst_in_intersection"]},
                   "T_tracked": {"nc": r["T_tracked"]["nc"], "dac": r["T_tracked"]["dac"],
                                 "margin_m": round(r["T_tracked"]["an"]["min_margin"], 3)},
                   "H_tracked_margin_m": round(h["an"]["min_margin"], 3),
                   "M_tracked": {"nc": r["M_tracked"]["nc"], "dac": r["M_tracked"]["dac"], "pdms": round(r["M_tracked"]["pdms"], 3),
                                 "margin_m": round(r["M_tracked"]["an"]["min_margin"], 3), "shift_max_m": round(r["info"]["shift_max_m"], 2)},
                   "M_transparent_dac": r["M_transparent"]["dac"],
                   "CERT": cert(r), "T_sub20": {"nc": r["T_sub20_tracked"]["nc"], "dac": r["T_sub20_tracked"]["dac"]}})
res["anatomy"] = an
# M's new zero rows (rows alive under Tc20s that M kills): what broke
res["M_new_zero_rows"] = [{"log": r["log"], "token": r["token"], "nc": r["M_tracked"]["nc"], "dac": r["M_tracked"]["dac"],
                           "coll_t": r["M_tracked"]["an"]["coll_t"], "shift_max_m": round(r["info"]["shift_max_m"], 2),
                           "v0": round(r["info"]["v0"], 2)}
                          for r in rows if r["M_tracked"]["pdms"] == 0 and r["Tc20s_tracked"]["pdms"] > 0]
res["shift_stats"] = {"n_shifted": int(sum(not r["info"]["c3_unshifted"] for r in rows)),
                      "shift_max_m_median_of_shifted": float(np.median([r["info"]["shift_max_m"] for r in rows if not r["info"]["c3_unshifted"]] or [0]))}
# EP anatomy (descriptive): where the teacher's progress sits against the human, on rows alive in every compared arm
EPA = ["T_transparent", "Tc20s_tracked", "M_tracked", "T_sub20_tracked", "H_tracked"]
al = np.all([X[k]["pdms"] > 0 for k in EPA], axis=0)
turn = np.array([r["info"]["turn_deg"] for r in rows])
v0 = np.array([r["info"]["v0"] for r in rows])
bins = {"all": al, "turn<15": al & (turn < 15), "turn15-45": al & (turn >= 15) & (turn < 45), "turn>=45": al & (turn >= 45),
        "v0<3": al & (v0 < 3), "v0>=3": al & (v0 >= 3)}
res["ep_anatomy_alive_all"] = {b: {"n": int(m.sum()), **{k: round(100 * float(X[k]["ep"][m].mean()), 2) for k in EPA}}
                               for b, m in bins.items() if m.sum()}
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
print("ZZTF2_CONTROLS", "PASS" if allpass else "FAIL", json.dumps({k: v["pass"] for k, v in C.items()}))
print("ZZTF2_OUTCOMES", json.dumps(O))
for arm in ("T_tracked", "Tc20s_tracked", "M_tracked", "CERT_tracked", "Mw_tracked", "T_sub20_tracked", "H_tracked",
            "T_transparent", "M_transparent"):
    print(f"{arm:18s} PDMS {res['arms'][arm]['pdms']} DAC {res['arms'][arm]['dac']} NC {res['arms'][arm]['nc'][0]} "
          f"EP {res['arms'][arm]['ep'][0]} TTC {res['arms'][arm]['ttc'][0]} C {res['arms'][arm]['c'][0]} {nz[arm]}")
for nm in ("M_minus_Tc20s", "CERT_minus_Tc20s", "Mw_minus_Tc20s_REGRESSION", "M_minus_H", "CERT_minus_H",
           "T_sub20_minus_T_tracked", "T_sub20_minus_T_transparent"):
    print(f"{nm:30s} PDMS {res['contrasts'][nm]['pdms']} DAC {res['contrasts'][nm]['dac']}")
print("EP alive", res["ep_alive_both_M_vs_Tc20s"], res["ep_alive_both_CERT_vs_Tc20s"], "CERT choice", res["cert_choice"])
print("shift", res["shift_stats"], "M new zero rows", len(res["M_new_zero_rows"]))
