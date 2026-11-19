"""read_brake10k.py -- card PREREG_V8F_PROOF_ADD57 (BRAKE10K): the read (numpy only). Controls b1 to b10 first; the names and the differences only if every control passed.
Forms: LABEL = label-fed rule with the label closing speed (RLc), READER = reader-fed rule at map level 0.50 with closing 0 (RQ50). Braking limit of the rule 6.0 (banked),
8.0 (both forms), 10.0 (label-fed, dose). Replan interval 10 steps (2.0 s), the programme's cycle.
  per form   BR = U(limit 8.0) - U(limit 6.0, banked), contact C and falling behind F beside, the four families beside
  dose       label-fed 10.0 against 6.0 and against 8.0 (names nothing)"""
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
WRAPPER_SHA_PREFIX = "31da3c553eef1b8f"                                                        # the wrapper of the runs (freeze line of the card in TIMES_NOTE.md)
FORMS = (("LABEL", "I10RLc"), ("READER", "I10RQ50"))
BMAX = {"I10RLcB6": 6.0, "I10RLcB8": 8.0, "I10RQ50B8": 8.0, "I10RLcB10": 10.0}
NEW_RUNS = [("I10RLcB6", 0)] + [(arm, k) for arm in ("I10RLcB8", "I10RQ50B8", "I10RLcB10") for k in SEEDS]
LADDER = {"LABEL": ("I10RLc", "I10RLcB8", "I10RLcB10"), "READER": ("I10RQ50", "I10RQ50B8")}
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"
TOL = 1e-9


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if p and os.path.exists(p) else ""


def npz(pull, arm, k, suffix=""):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d%s.npz" % (arm, k, suffix))
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def controls(a, hw, q3):
    C, R = {}, {}
    p = os.path.join(a.pull, "small_brake", "CMP_RLcB6.txt")
    m = re.search(r"CMP files=(\d+) differing=(\d+)", io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else "")
    C["b1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    b2, b3, b5, sat = {}, {}, {}, {}
    for _, base in FORMS:
        for k in SEEDS:
            Bk = npz(a.pull, base, k)
            key = "%s_s%d" % (base, k)
            b2[key] = {"B_MAX": None if Bk is None else float(Bk["B_MAX"]), "kwargs": None if Bk is None else str(Bk["mode_kwargs"])}
            b2[key]["PASS"] = bool(Bk is not None and abs(b2[key]["B_MAX"] - 6.0) <= TOL and "b_max" not in b2[key]["kwargs"] and abs(float(Bk["TAU"]) - 1.2) <= TOL)
    for arm, k in NEW_RUNS:
        key = "%s_s%d" % (arm, k)
        L = npz(a.pull, arm, k)
        base = "I10RQ50" if arm.startswith("I10RQ50") else "I10RLc"
        Bk = npz(a.pull, base, k)
        if L is None or Bk is None:
            b2[key] = b3[key] = {"PASS": None, "why": "a rule log is missing"}
            continue
        kw = json.loads(str(L["mode_kwargs"]))
        b2[key] = {"B_MAX": float(L["B_MAX"]), "kwargs_b_max": kw.get("b_max"), "TAU": float(L["TAU"]), "replan_steps": int(L["replan_steps"])}
        b2[key]["PASS"] = bool(abs(b2[key]["B_MAX"] - BMAX[arm]) <= TOL and kw.get("b_max") == BMAX[arm] and abs(b2[key]["TAU"] - 1.2) <= TOL and b2[key]["replan_steps"] == 10)
        m0 = np.asarray(Bk["tick"]) == 0
        same_ticks = all(np.array_equal(Bk[c], L[c]) for c in ("ci", "t0", "tick"))
        worst = 0.0
        for c in ("g_src", "c_src", "v0", "v_L"):
            x, y = np.asarray(Bk[c], np.float64)[m0], np.asarray(L[c], np.float64)[m0]
            if len(x) != len(y) or not np.array_equal(np.isfinite(x), np.isfinite(y)):
                worst = float("inf")
            else:
                f = np.isfinite(x)
                worst = max(worst, float(np.max(np.abs(x[f] - y[f]))) if f.any() else 0.0)
        b3[key] = {"max_abs_first_tick_inputs": worst, "same_ticks": bool(same_ticks), "n_first_ticks": int(m0.sum()), "PASS": bool(same_ticks and worst <= 1e-6)}
        s6, sn = np.asarray(Bk["sat_exec"], np.int64), np.asarray(L["sat_exec"], np.int64)
        sat[key] = {"first_tick_sum_6.0": int(s6[m0].sum()), "first_tick_sum_run": int(sn[m0].sum()), "first_tick_rows_6.0": int((s6[m0] > 0).sum()), "first_tick_rows_run": int((sn[m0] > 0).sum()),
                    "first_tick_rows_where_run_is_above_6.0": int((sn[m0] > s6[m0]).sum()), "all_ticks_rows_6.0": int((s6 > 0).sum()), "all_ticks_rows_run": int((sn > 0).sum())}
        if arm != "I10RLcB6":                                                             # the parameter is live and acts in the known direction on identical first-tick inputs
            b5[key] = {"first_tick_saturated_steps_6.0": sat[key]["first_tick_sum_6.0"], "run": sat[key]["first_tick_sum_run"],
                       "PASS": bool(sat[key]["first_tick_sum_6.0"] > 0 and sat[key]["first_tick_sum_run"] < sat[key]["first_tick_sum_6.0"])}
        else:
            b5[key] = {"first_tick_saturated_steps_6.0": sat[key]["first_tick_sum_6.0"], "run": sat[key]["first_tick_sum_run"], "PASS": bool(np.array_equal(s6, sn))}
    b4 = {}
    for k in SEEDS:                                                                       # the reader reads logged frames only: equal at every tick whatever the braking limit
        Rq, Rn = npz(a.pull, "I10RQ50", k, "_rd"), npz(a.pull, "I10RQ50B8", k, "_rd")
        if Rq is None or Rn is None:
            b4[k] = {"PASS": None, "why": "a side log is missing"}
            continue
        same = all(np.array_equal(Rq[c], Rn[c], equal_nan=True) for c in ("p", "g_read", "g_used", "ok"))
        b4[k] = {"same_reader_reads": bool(same), "PASS": bool(same)}
    for name, d in (("b2_braking_limit_applied", b2), ("b3_first_tick_inputs_equal", b3), ("b4_same_read", b4), ("b5_limit_is_live", b5)):
        C[name] = {"per_run": d, "PASS": (None if (not d or any(v["PASS"] is None for v in d.values())) else bool(all(v["PASS"] for v in d.values())))}
    nfiles, sh = {}, {}
    for arm, k in NEW_RUNS:
        key = "%s_s%d" % (arm, k)
        d = os.path.join(a.pull, key, "dump_%s_m10_causal_heads" % key)
        nfiles[key] = len([f for f in os.listdir(d) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(d) else None
        p = os.path.join(a.pull, key, "ticks", "run_%s.txt" % key)
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
        sh[key] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
    cnt = (hw or {}).get("counts", {})
    C["b6_counts"] = {"n_runs": len(nfiles), "not_141": [n for n, v in nfiles.items() if v != 141], "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and len(nfiles) == 7 and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    root = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(root, p)) or sha(os.path.join(root, p)) != h]
    C["b7_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "PASS": bool(len(ru["files"]) == 10 and not bad)}
    C["b8_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["b8_reader_controls"]["PASS"] = bool(C["b8_reader_controls"]["headway"] and C["b8_reader_controls"]["family"])
    C["b9_model"] = {"n_runs": len(sh), "not_equal": [n for n, v in sh.items() if not v], "PASS": bool(len(sh) == 7 and all(sh.values()))}
    w1 = sha(a.wrapper)
    C["b10_wrapper"] = {"wrapper_sha256": w1[:16], "card": WRAPPER_SHA_PREFIX, "PASS": bool(w1 and w1.startswith(WRAPPER_SHA_PREFIX))}
    R["r1_saturated_steps"] = sat
    return C, R


def cell(hw, A, B, c):
    v = hw["pairs"]["%s_minus_%s" % (A, B)]["cells"][c]
    return {"per_seed": [round(float(x), 4) for x in v["d_s"]], "pooled": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)]}


def ucf(hw, A, B):
    return {c: cell(hw, A, B, c) for c in ("U", "C", "F")}


def fams(q3, A, B):
    return {f: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]], "verdict": v["verdict"]}
            for f, v in q3["pairs"]["%s_minus_%s" % (A, B)]["families"].items()}


def line(v):
    return "  ".join("%s %+.4f [%+.4f, %+.4f] %s" % (c, v[c]["pooled"], v[c]["interval"][0], v[c]["interval"][1], v[c]["per_seed"]) for c in ("U", "C", "F"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--pull", required=True)
    ap.add_argument("--reused", required=True)
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hw, q3 = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    C, R = controls(a, hw, q3)
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": "PREREG_V8F_PROOF_ADD57", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "reported_no_bar": R, "names": None}
    print(TIER_LINE)
    print("== controls (b1 to b10)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:700]))
    print("== reported without a bar")
    for k, v in R.items():
        print("   %s %s" % (k, json.dumps(v)[:1500]))
    if ok:
        lv = hw["levels"]
        levels = {fk: {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in arms} for fk, arms in LADDER.items()}
        names, br = {}, {}
        for fk, base in FORMS:
            br[fk] = {"headway": ucf(hw, base + "B8", base), "families": fams(q3, base + "B8", base)}
            d = br[fk]["headway"]["U"]
            if all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0:
                names[fk] = "BRAKE10K-%s-HARDER-BRAKING-CUTS-UNSAFE" % fk
            elif all(x > 0 for x in d["per_seed"]) and d["interval"][0] > 0:
                names[fk] = "BRAKE10K-%s-HARDER-BRAKING-RAISES-UNSAFE" % fk
            else:
                names[fk] = "BRAKE10K-%s-NO-U-CHANGE" % fk
        dose = {"B10_minus_6.0": {"headway": ucf(hw, "I10RLcB10", "I10RLc"), "families": fams(q3, "I10RLcB10", "I10RLc")}, "B10_minus_B8": {"headway": ucf(hw, "I10RLcB10", "I10RLcB8")}}
        bes = {"against_no_rule": {arm: ucf(hw, arm, "I10R0") for arm in ("I10RLcB8", "I10RQ50B8")}, "reader_minus_label_at_8.0": ucf(hw, "I10RQ50B8", "I10RLcB8")}
        res.update(names=names, levels=levels, BR=br, dose=dose, beside=bes)
        print("== names")
        for fk, _ in FORMS:
            print("   %s" % names[fk])
        print("== levels per braking limit (per model seed; 562 scored windows, 642 lead windows)")
        for fk, arms in levels.items():
            for arm, v in arms.items():
                print("   %-7s %-10s U %s  C %s  F %s" % (fk, arm, v["U"], v["C"], v["F"]))
        print("== BR: limit 8.0 minus limit 6.0, per form (pooled [interval], per seed)")
        for fk, _ in FORMS:
            print("   %-7s %s" % (fk, line(br[fk]["headway"])))
        print("== dose (label-fed, names nothing)")
        for nm, v in dose.items():
            print("   %-14s %s" % (nm, line(v["headway"])))
        print("== beside")
        for arm, v in bes["against_no_rule"].items():
            print("   %s against no rule: %s" % (arm, line(v)))
        print("   reader-fed minus label-fed, both at 8.0: %s" % line(bes["reader_minus_label_at_8.0"]))
        print("== families (limit 8.0 minus 6.0; label-fed 10.0 minus 6.0)")
        for nm, fm in (("LABEL 8.0", br["LABEL"]["families"]), ("READER 8.0", br["READER"]["families"]), ("LABEL 10.0", dose["B10_minus_6.0"]["families"])):
            for f, v in fm.items():
                print("   %-11s %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (nm, f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== a control did not pass: no name, the differences stay withheld")
    print("STRATEGIC: as note N1 of card ADD46 (T0 first tick: the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
