"""read_closer10k.py -- card PREREG_V8F_PROOF_ADD49 (CLOSER10K): the read (numpy only). Controls h1 to h8 first; the names and the differences only if every control passed.
Rules as frozen in the card: R1 = U(RQ50h) - U(RQ50) (primary), R2 = C(RQ50h) - C(RQ50), R3 = F(RQ50h) - F(RQ50)."""
import argparse
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

SEEDS = (0, 1)
CKPT_SHA_PREFIX = {0: "2fd4b747532f55b2", 1: "f13df8ce026eed31"}                       # card ADD46: UE_s0, UE_s1
NEW_RUNS = (("I10RQ50h", 0), ("I10RQ50h", 1), ("I10RQ50R", 0))
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"
R_BAR, TOL, G_MAX = 0.3, 1e-4, 60.0


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def npz(pull, arm, k, suffix=""):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d%s.npz" % (arm, k, suffix))
    return np.load(p) if os.path.exists(p) else None


def controls(a, hw, q3):
    C = {}
    p = os.path.join(a.pull, "small_closer", "CMP_RQ50R.txt")
    txt = io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
    m = re.search(r"CMP files=(\d+) differing=(\d+)", txt)
    C["h1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    h2, h3, h4 = {}, {}, {}
    for k in SEEDS:
        Lc, Lh, Rq, Rh = npz(a.pull, "I10RLc", k), npz(a.pull, "I10RQ50h", k), npz(a.pull, "I10RQ50", k, "_rd"), npz(a.pull, "I10RQ50h", k, "_rd")
        if any(v is None for v in (Lc, Lh, Rq, Rh)) or "c_read" not in Rh.files:
            h2[k] = h3[k] = h4[k] = {"PASS": None, "why": "a log or the closing read is missing"}
            continue
        same_ticks = all(np.array_equal(Lc[c], Lh[c]) for c in ("ci", "t0", "tick"))
        lab = np.asarray(Lc["present"]).astype(bool) & np.isfinite(np.asarray(Lc["g_src"], np.float64)) & (np.asarray(Lc["g_src"], np.float64) <= G_MAX)
        rp = np.asarray(Lh["present"]).astype(bool)
        cr, cl = np.asarray(Rh["c_read"], np.float64), np.asarray(Lc["c_src"], np.float64)
        m2 = lab & rp & np.isfinite(cr) & np.isfinite(cl)
        r = float(np.corrcoef(cr[m2], cl[m2])[0, 1]) if m2.sum() > 10 else None
        h2[k] = {"n_ticks": int(m2.sum()), "pearson_r": r, "mae_mps": float(np.abs(cr[m2] - cl[m2]).mean()) if m2.any() else None, "mae_of_zero_mps": float(np.abs(cl[m2]).mean()) if m2.any() else None,
                 "mean_read_mps": float(cr[m2].mean()) if m2.any() else None, "mean_label_mps": float(cl[m2].mean()) if m2.any() else None, "PASS": bool(same_ticks and r is not None and r >= R_BAR)}
        cs, vl, vL = (np.asarray(Lh[c], np.float64) for c in ("c_src", "v_l", "v_L"))
        d_feed = float(np.max(np.abs(cs - cr))) if len(cs) == len(cr) else None
        fin = np.isfinite(cs) & np.isfinite(vl) & np.isfinite(vL)
        ident = float(np.max(np.abs(vL[fin] - np.maximum(0.0, vl[fin] + cs[fin])))) if fin.any() else None
        h3[k] = {"max_abs_fed_minus_read_mps": d_feed, "max_abs_identity_mps": ident, "n_finite": int(fin.sum()), "PASS": bool(d_feed is not None and d_feed <= TOL and ident is not None and ident <= TOL)}
        same_read = all(np.array_equal(Rq[c], Rh[c], equal_nan=True) for c in ("p", "g_read", "g_used", "ok"))
        h4[k] = {"same_read_RQ50_RQ50h": bool(same_read), "same_ticks": bool(same_ticks), "PASS": bool(same_read and same_ticks)}
    for name, d in (("h2_closing_read_T0", h2), ("h3_feed", h3), ("h4_same_read", h4)):
        C[name] = {"per_seed": d, "PASS": (None if any(v["PASS"] is None for v in d.values()) else bool(all(v["PASS"] for v in d.values())))}
    nfiles = {}
    for arm, k in NEW_RUNS:
        d = os.path.join(a.pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k))
        nfiles["%s_s%d" % (arm, k)] = len([f for f in os.listdir(d) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(d) else None
    cnt = (hw or {}).get("counts", {})
    C["h5_counts"] = {"dump_files": nfiles, "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or hashlib.sha256(open(os.path.join(base, p), "rb").read()).hexdigest() != h]
    C["h6_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "PASS": bool(len(ru["files"]) == 10 and not bad)}
    C["h7_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["h7_reader_controls"]["PASS"] = bool(C["h7_reader_controls"]["headway"] and C["h7_reader_controls"]["family"])
    sh = {}
    for arm, k in NEW_RUNS:
        p = os.path.join(a.pull, "%s_s%d" % (arm, k), "ticks", "run_%s_s%d.txt" % (arm, k))
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
        sh["%s_s%d" % (arm, k)] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
    C["h8_model"] = {"per_run": sh, "PASS": bool(all(sh.values()))}
    return C


def cell(hw, A, B, c):
    v = hw["pairs"]["%s_minus_%s" % (A, B)]["cells"][c]
    return {"per_seed": [round(float(x), 4) for x in v["d_s"]], "pooled": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--reused", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hw, q3 = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    C = controls(a, hw, q3)
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": "PREREG_V8F_PROOF_ADD49", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "names": None, "beside": None}
    print(TIER_LINE)
    print("== controls (h1 to h8)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:520]))
    if ok:
        R1, R2, R3 = (cell(hw, "I10RQ50h", "I10RQ50", c) for c in ("U", "C", "F"))
        if all(x < 0 for x in R1["per_seed"]) and R1["interval"][1] < 0:
            name = "CLOSER10K-READER-CLOSING-CUTS-UNSAFE"
        elif R1["interval"][0] > 0:
            name = "CLOSER10K-READER-CLOSING-HARMS"
        else:
            name = "CLOSER10K-READER-CLOSING-NO-U-CHANGE"
        sec = []
        if R2["interval"][1] < 0:
            sec.append("-CUTS-CONTACT")
        if R3["interval"][1] < 0:
            sec.append("-LESS-FALLING-BEHIND")
        if R3["interval"][0] > 0:
            sec.append("-MORE-FALLING-BEHIND")
        res["names"] = {"name": name, "secondary": sec, "R1_U_RQ50h_minus_RQ50": R1, "R2_C_RQ50h_minus_RQ50": R2, "R3_F_RQ50h_minus_RQ50": R3}
        lv = hw["levels"]
        curve = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in ("I10R0", "I10RLb", "I10RLc", "I10RQ50", "I10RQ50c", "I10RQ50h") if arm in lv}
        bes = {"levels_per_seed": curve}
        for B in ("I10RQ50c", "I10R0", "I10RLc"):
            bes["RQ50h_minus_%s" % B[3:]] = {c: cell(hw, "I10RQ50h", B, c) for c in ("U", "C", "F")}
        bes["families_RQ50h_minus_RQ50"] = {f: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]],
                                                "verdict": v["verdict"]} for f, v in q3["pairs"]["I10RQ50h_minus_I10RQ50"]["families"].items()}
        res["beside"] = bes
        print("== names")
        print("   %s ; secondary %s" % (name, sec if sec else "none"))
        for k in ("R1_U_RQ50h_minus_RQ50", "R2_C_RQ50h_minus_RQ50", "R3_F_RQ50h_minus_RQ50"):
            print("   %s %s" % (k, json.dumps(res["names"][k])))
        print("== levels (per model seed)")
        for arm, v in curve.items():
            print("   %-9s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        for k in ("RQ50h_minus_RQ50c", "RQ50h_minus_R0", "RQ50h_minus_RLc"):
            print("   beside %s %s" % (k, json.dumps(bes[k])[:430]))
        print("== families I10RQ50h_minus_I10RQ50")
        for f, v in bes["families_RQ50h_minus_RQ50"].items():
            print("   %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== a control did not pass: no name, the differences stay withheld")
    print("STRATEGIC: as note N1 of card ADD46 (T0 first tick: the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
