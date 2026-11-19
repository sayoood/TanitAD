"""read_price10k.py -- card PREREG_V8F_PROOF_ADD47 (PRICE10K): the read (numpy only). Controls m1 to m7 first; the names and the differences only if every control passed.
Inputs: the headway and family results on the pairs of the card (read_price10k.sh), the pulled run folders, the small files of the queue, the hashes of the reused dumps.
Rules as frozen in the card: deciding level 0.50 (run I10RQ50). P1 = U(RQ50) - U(R0), P2 = U(RQ50) - U(RLb), P3 = F(RQ50) - F(RLb), P4 = F(RQ50) - F(RR)."""
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
NEW, REUSED = ("I10RQ50", "I10RQ35", "I10RRaw"), ("I10R0", "I10RLb", "I10RR")
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"
TOL = 1e-9


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def rd(pull, arm, k):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d_rd.npz" % (arm, k))
    return np.load(p) if os.path.exists(p) else None


def idx(pull, arm, k):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d.npz" % (arm, k))
    return np.load(p) if os.path.exists(p) else None


def controls(a, hw, q3):
    C = {}
    mc = jl(os.path.join(a.pull, "small_price", "PRICE_MAPS_CHECK.json"))
    C["m1_maps"] = {"PASS": bool(mc and mc.get("PASS") is True), "coverage": {k: v.get("coverage") for k, v in (mc or {}).get("arms", {}).items()}, "identity_file": (mc or {}).get("identity_file_is_identity")}
    m2, m3 = {}, {}
    for k in SEEDS:
        L = {arm: rd(a.pull, arm, k) for arm in ("I10RR", "I10RQ35", "I10RQ50", "I10RRaw")}
        X = {arm: idx(a.pull, arm, k) for arm in ("I10RR", "I10RQ35", "I10RQ50", "I10RRaw")}
        if any(v is None for v in L.values()) or any(v is None for v in X.values()):
            m2[k] = m3[k] = {"PASS": None, "why": "rule log missing"}
            continue
        raw = L["I10RRaw"]
        fin = np.isfinite(raw["g_read"])
        d = float(np.max(np.abs(raw["g_used"][fin] - raw["g_read"][fin].astype(np.float64)))) if fin.any() else 0.0
        m2[k] = {"n_ticks": int(len(fin)), "n_finite": int(fin.sum()), "max_abs_used_minus_read": d, "PASS": bool(fin.any() and d == 0.0)}
        same_ticks = all(np.array_equal(X["I10RR"][c], X[arm][c]) for arm in ("I10RQ35", "I10RQ50", "I10RRaw") for c in ("ci", "t0", "tick"))
        same_read = all(np.array_equal(L["I10RR"][c], L[arm][c], equal_nan=True) for arm in ("I10RQ35", "I10RQ50", "I10RRaw") for c in ("p", "g_read", "ok"))
        u = [np.asarray(L[arm]["g_used"], np.float64)[fin] for arm in ("I10RR", "I10RQ35", "I10RQ50", "I10RRaw")]
        ordered = bool(np.all(u[0] <= u[1] + TOL) and np.all(u[1] <= u[2] + TOL) and np.all(u[2] <= u[3] + TOL))
        m3[k] = {"same_ticks": bool(same_ticks), "same_read_and_presence": bool(same_read), "fed_gap_ordered": ordered, "mean_fed_gap_m": [round(float(np.mean(x)), 3) for x in u],
                 "PASS": bool(same_ticks and same_read and ordered)}
    C["m2_identity_in_the_run"] = {"per_seed": m2, "PASS": (None if any(v["PASS"] is None for v in m2.values()) else bool(all(v["PASS"] for v in m2.values())))}
    C["m3_same_read_ordered_feed"] = {"per_seed": m3, "PASS": (None if any(v["PASS"] is None for v in m3.values()) else bool(all(v["PASS"] for v in m3.values())))}
    nfiles = {}
    for arm in NEW:
        for k in SEEDS:
            d = os.path.join(a.pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k))
            nfiles["%s_s%d" % (arm, k)] = len([f for f in os.listdir(d) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(d) else None
    cnt = (hw or {}).get("counts", {})
    C["m4_counts"] = {"dump_files": nfiles, "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or hashlib.sha256(open(os.path.join(base, p), "rb").read()).hexdigest() != h]
    C["m5_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "PASS": bool(len(ru["files"]) == 6 and not bad)}
    C["m6_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["m6_reader_controls"]["PASS"] = bool(C["m6_reader_controls"]["headway"] and C["m6_reader_controls"]["family"])
    sh = {}
    for arm in NEW:
        for k in SEEDS:
            p = os.path.join(a.pull, "%s_s%d" % (arm, k), "ticks", "run_%s_s%d.txt" % (arm, k))
            mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
            sh["%s_s%d" % (arm, k)] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
    C["m7_model"] = {"per_run": sh, "PASS": bool(all(sh.values()))}
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
    res = {"card": "PREREG_V8F_PROOF_ADD47", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "names": None, "beside": None}
    print(TIER_LINE)
    print("== controls (m1 to m7)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:330]))
    if ok:
        P1, P2 = cell(hw, "I10RQ50", "I10R0", "U"), cell(hw, "I10RQ50", "I10RLb", "U")
        P3, P4 = cell(hw, "I10RQ50", "I10RLb", "F"), cell(hw, "I10RQ50", "I10RR", "F")
        keeps = bool(all(x < 0 for x in P1["per_seed"]) and P1["interval"][1] < 0)
        if keeps:
            name = "PRICE10K-MEDIAN-KEEPS-CUT" + ("-PRICE-REMAINS" if P3["interval"][0] > 0 else "-AT-LABEL-PRICE")
        else:
            name = "PRICE10K-MEDIAN-LOSES-CUT"
        res["names"] = {"name": name, "P1_U_RQ50_minus_R0": P1, "P2_U_RQ50_minus_RLb": P2, "P2_reaches_label_rule_on_U": bool(P2["interval"][0] <= 0 <= P2["interval"][1] or P2["interval"][1] < 0),
                        "P3_F_RQ50_minus_RLb": P3, "P4_F_RQ50_minus_RR": P4, "P4_below_0": bool(P4["interval"][1] < 0)}
        lv = hw["levels"]
        curve = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in ("I10R0", "I10RLb", "I10RR", "I10RQ35", "I10RQ50", "I10RRaw") if arm in lv}
        bes = {"curve_levels_per_seed": curve}
        for arm in ("I10RQ35", "I10RRaw", "I10RR"):
            bes[arm] = {"U_minus_R0": cell(hw, arm, "I10R0", "U"), "U_minus_RLb": cell(hw, arm, "I10RLb", "U"), "F_minus_RLb": cell(hw, arm, "I10RLb", "F")}
        fam = {}
        for B in ("I10R0", "I10RLb", "I10RR"):
            fam["I10RQ50_minus_%s" % B] = {f: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]], "verdict": v["verdict"]}
                                           for f, v in q3["pairs"]["I10RQ50_minus_%s" % B]["families"].items()}
        bes["families"] = fam
        res["beside"] = bes
        print("== name")
        print("   %s" % name)
        for k in ("P1_U_RQ50_minus_R0", "P2_U_RQ50_minus_RLb", "P3_F_RQ50_minus_RLb", "P4_F_RQ50_minus_RR"):
            print("   %s %s" % (k, json.dumps(res["names"][k])))
        print("   P2 reaches the label-fed rule on U: %s ; P4 below 0: %s" % (res["names"]["P2_reaches_label_rule_on_U"], res["names"]["P4_below_0"]))
        print("== curve (per model seed)")
        for arm, v in curve.items():
            print("   %-8s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        for arm in ("I10RQ35", "I10RRaw"):
            print("   beside %s: %s" % (arm, json.dumps(bes[arm])[:420]))
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
