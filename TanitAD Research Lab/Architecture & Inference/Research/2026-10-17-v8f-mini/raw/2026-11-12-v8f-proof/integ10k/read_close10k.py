"""read_close10k.py -- card PREREG_V8F_PROOF_ADD48 (CLOSE10K): the read (numpy only). Controls c1 to c7 first; the names and the differences only if every control passed.
Inputs: the headway and family results on the pairs of the card (read_close10k.sh), the pulled run folders, the small files of the queue, the hashes of the reused dumps.
Rules as frozen in the card: Q1 = U(RLc) - U(RLb), Q2 = F(RLc) - F(RLb), Q3 = U(RQ50c) - U(RQ50), Q4 = F(RQ50c) - F(RQ50)."""
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
NEW_RUNS = (("I10RLc", 0), ("I10RLc", 1), ("I10RQ50c", 0), ("I10RQ50c", 1), ("I10RLbR", 0))
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"
STD_BAR, ID_TOL = 0.1, 1e-4


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def npz(pull, arm, k, suffix=""):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d%s.npz" % (arm, k, suffix))
    return np.load(p) if os.path.exists(p) else None


def closing_stats(L):
    pres = np.asarray(L["present"]).astype(bool)
    c, vl, vL = (np.asarray(L[k], np.float64) for k in ("c_src", "v_l", "v_L"))
    fin = np.isfinite(c) & np.isfinite(vl) & np.isfinite(vL)
    ident = float(np.max(np.abs(vL[fin] - np.maximum(0.0, vl[fin] + c[fin])))) if fin.any() else None
    return {"n_ticks": int(len(c)), "n_present": int(pres.sum()), "finite_on_present": bool(np.isfinite(c[pres]).all()) if pres.any() else None,
            "std_on_present_mps": float(np.std(c[pres])) if pres.any() else None, "max_abs_all": float(np.nanmax(np.abs(c))) if len(c) else None,
            "max_abs_identity_mps": ident, "n_finite": int(fin.sum())}


def controls(a, hw, q3):
    C = {}
    p = os.path.join(a.pull, "small_close", "CMP_RLBR.txt")
    txt = io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
    m = re.search(r"CMP files=(\d+) differing=(\d+)", txt)
    C["c1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    c2, c3 = {}, {}
    for k in SEEDS:
        Lb, Lc, Lq, Lqc = npz(a.pull, "I10RLb", k), npz(a.pull, "I10RLc", k), npz(a.pull, "I10RQ50", k), npz(a.pull, "I10RQ50c", k)
        Rq, Rqc = npz(a.pull, "I10RQ50", k, "_rd"), npz(a.pull, "I10RQ50c", k, "_rd")
        if any(v is None for v in (Lb, Lc, Lq, Lqc, Rq, Rqc)):
            c2[k] = c3[k] = {"PASS": None, "why": "rule log missing"}
            continue
        sb, sc, sqc = closing_stats(Lb), closing_stats(Lc), closing_stats(Lqc)
        ok_b = bool(sb["max_abs_all"] == 0.0)
        ok_c = [bool(s["finite_on_present"] is True and s["std_on_present_mps"] is not None and s["std_on_present_mps"] > STD_BAR
                     and s["max_abs_identity_mps"] is not None and s["max_abs_identity_mps"] <= ID_TOL) for s in (sc, sqc)]
        c2[k] = {"RLb": sb, "RLc": sc, "RQ50c": sqc, "RLb_zero": ok_b, "RLc_ok": ok_c[0], "RQ50c_ok": ok_c[1], "PASS": bool(ok_b and all(ok_c))}
        same_read = all(np.array_equal(Rq[c], Rqc[c], equal_nan=True) for c in ("p", "g_read", "g_used", "ok"))
        same_ticks = all(np.array_equal(Lb[c], X[c]) for X in (Lc, Lq, Lqc) for c in ("ci", "t0", "tick"))
        c3[k] = {"same_read_RQ50_RQ50c": bool(same_read), "same_ticks_all_runs": bool(same_ticks), "PASS": bool(same_read and same_ticks)}
    for name, d in (("c2_closing_feed", c2), ("c3_same_read", c3)):
        C[name] = {"per_seed": d, "PASS": (None if any(v["PASS"] is None for v in d.values()) else bool(all(v["PASS"] for v in d.values())))}
    nfiles = {}
    for arm, k in NEW_RUNS:
        d = os.path.join(a.pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k))
        nfiles["%s_s%d" % (arm, k)] = len([f for f in os.listdir(d) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(d) else None
    cnt = (hw or {}).get("counts", {})
    C["c4_counts"] = {"dump_files": nfiles, "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or hashlib.sha256(open(os.path.join(base, p), "rb").read()).hexdigest() != h]
    C["c5_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "PASS": bool(len(ru["files"]) == 6 and not bad)}
    C["c6_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["c6_reader_controls"]["PASS"] = bool(C["c6_reader_controls"]["headway"] and C["c6_reader_controls"]["family"])
    sh = {}
    for arm, k in NEW_RUNS:
        p = os.path.join(a.pull, "%s_s%d" % (arm, k), "ticks", "run_%s_s%d.txt" % (arm, k))
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
        sh["%s_s%d" % (arm, k)] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
    C["c7_model"] = {"per_run": sh, "PASS": bool(all(sh.values()))}
    return C


def cell(hw, A, B, c):
    v = hw["pairs"]["%s_minus_%s" % (A, B)]["cells"][c]
    return {"per_seed": [round(float(x), 4) for x in v["d_s"]], "pooled": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)]}


def below0(P):
    return bool(all(x < 0 for x in P["per_seed"]) and P["interval"][1] < 0)


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
    res = {"card": "PREREG_V8F_PROOF_ADD48", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "names": None, "beside": None}
    print(TIER_LINE)
    print("== controls (c1 to c7)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:400]))
    if ok:
        Q1, Q2 = cell(hw, "I10RLc", "I10RLb", "U"), cell(hw, "I10RLc", "I10RLb", "F")
        Q3, Q4 = cell(hw, "I10RQ50c", "I10RQ50", "U"), cell(hw, "I10RQ50c", "I10RQ50", "F")
        if below0(Q1):
            name = "CLOSE10K-CLOSING-CUTS-UNSAFE" + ("-AT-A-PRICE" if Q2["interval"][0] > 0 else "-AT-NO-PRICE")
        else:
            name = "CLOSE10K-CLOSING-NO-CUT"
        name2 = "CLOSE10K-READER-GAINS" if below0(Q3) else "CLOSE10K-READER-NO-GAIN"
        res["names"] = {"name": name, "name_diagnostic": name2, "Q1_U_RLc_minus_RLb": Q1, "Q2_F_RLc_minus_RLb": Q2, "Q3_U_RQ50c_minus_RQ50": Q3, "Q4_F_RQ50c_minus_RQ50": Q4}
        lv = hw["levels"]
        curve = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in ("I10R0", "I10RLb", "I10RLc", "I10RQ50", "I10RQ50c") if arm in lv}
        bes = {"levels_per_seed": curve, "U_RLc_minus_R0": cell(hw, "I10RLc", "I10R0", "U"), "U_RQ50c_minus_R0": cell(hw, "I10RQ50c", "I10R0", "U"),
               "C_RLc_minus_RLb": cell(hw, "I10RLc", "I10RLb", "C"), "C_RQ50c_minus_RQ50": cell(hw, "I10RQ50c", "I10RQ50", "C")}
        fam = {}
        for A, B in (("I10RLc", "I10RLb"), ("I10RQ50c", "I10RQ50")):
            fam["%s_minus_%s" % (A, B)] = {f: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]],
                                               "verdict": v["verdict"]} for f, v in q3["pairs"]["%s_minus_%s" % (A, B)]["families"].items()}
        bes["families"] = fam
        res["beside"] = bes
        print("== names")
        print("   %s ; %s" % (name, name2))
        for k in ("Q1_U_RLc_minus_RLb", "Q2_F_RLc_minus_RLb", "Q3_U_RQ50c_minus_RQ50", "Q4_F_RQ50c_minus_RQ50"):
            print("   %s %s" % (k, json.dumps(res["names"][k])))
        print("== levels (per model seed)")
        for arm, v in curve.items():
            print("   %-9s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        for k in ("U_RLc_minus_R0", "U_RQ50c_minus_R0", "C_RLc_minus_RLb", "C_RQ50c_minus_RQ50"):
            print("   beside %s %s" % (k, json.dumps(bes[k])))
        for pr, cells in fam.items():
            print("== families %s" % pr)
            for f, v in cells.items():
                print("   %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== a control did not pass: no name, the differences stay withheld")
    print("STRATEGIC: as note N1 of card ADD46 (T0 first tick: the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
