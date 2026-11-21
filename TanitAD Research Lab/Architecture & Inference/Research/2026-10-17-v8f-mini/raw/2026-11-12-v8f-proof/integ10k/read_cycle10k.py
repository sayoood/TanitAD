"""read_cycle10k.py -- card PREREG_V8F_PROOF_ADD52 (CYCLE10K): the read (numpy only). Controls y1 to y8 first; the names and the differences only if every control passed.
Forms: NORULE (R0), LABEL (label-fed rule with the label closing speed, RLc), READER (reader-fed rule at map level 0.50 with closing 0, RQ50). M5 = replan interval 5 steps
(1.0 s) for model and rule; the banked runs use 10 steps (2.0 s). Rules as frozen in the card:
  per form   CY = U(form, M5) - U(form, banked)            the cycle
  per rule   RU = U(rule form, M5) - U(R0, M5)             the rule at the 1.0 s cycle"""
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
FORMS = (("NORULE", "I10R0"), ("LABEL", "I10RLc"), ("READER", "I10RQ50"))
NEW_RUNS = [("I10RLcM10", 0)] + [(b + "M5", k) for k in SEEDS for _, b in FORMS]
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def npz(pull, arm, k, suffix=""):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d%s.npz" % (arm, k, suffix))
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def replan_of_result(pull, arm, k):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "t1v8f_%s_s%d_m10_causal_heads.json" % (arm, k))
    m = re.search(r'"replan_steps": (\d+)', io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
    return int(m.group(1)) if m else None


def keyed(L, cols, sel=None):
    """rows of a rule log (or its side log, read with the keys of the rule log) by (clip, window start, cache step)."""
    ci, t0, j = (np.asarray(L[c], np.int64) for c in ("ci", "t0", "j"))
    idx = np.arange(len(ci)) if sel is None else np.flatnonzero(sel)
    return {(int(ci[i]), int(t0[i]), int(j[i])): i for i in idx}, cols


def controls(a, hw, q3):
    C = {}
    p = os.path.join(a.pull, "small_cycle", "CMP_RLcM10.txt")
    txt = io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
    m = re.search(r"CMP files=(\d+) differing=(\d+)", txt)
    C["y1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    y2, y3, y4 = {}, {}, {}
    for k in SEEDS:
        y2["NORULE_s%d" % k] = {"banked": replan_of_result(a.pull, "I10R0", k), "M5": replan_of_result(a.pull, "I10R0M5", k)}
        y2["NORULE_s%d" % k]["PASS"] = bool(y2["NORULE_s%d" % k]["banked"] == 10 and y2["NORULE_s%d" % k]["M5"] == 5)
        for fk, base in FORMS[1:]:
            key = "%s_s%d" % (fk, k)
            B, N = npz(a.pull, base, k), npz(a.pull, base + "M5", k)
            if B is None or N is None:
                y2[key] = y3[key] = {"PASS": None, "why": "a rule log is missing"}
                continue
            tk = np.asarray(N["tick"], np.int64)
            y2[key] = {"banked": int(B["replan_steps"]), "M5": int(N["replan_steps"]), "rows_banked": int(len(B["ci"])), "rows_M5": int(len(tk)), "ticks_M5": sorted(set(tk.tolist())),
                       "result_file_M5": replan_of_result(a.pull, base + "M5", k)}
            y2[key]["PASS"] = bool(y2[key]["banked"] == 10 and y2[key]["M5"] == 5 and y2[key]["rows_M5"] == 2 * y2[key]["rows_banked"] and y2[key]["ticks_M5"] == [0, 1, 2, 3, 4, 5]
                                   and y2[key]["result_file_M5"] == 5)
            # the first tick of a window does not depend on the interval: equal source gap, closing speed, ego speed and lead speed
            kb, _ = keyed(B, None, np.asarray(B["tick"]) == 0)
            kn, _ = keyed(N, None, tk == 0)
            worst = 0.0 if set(kb) == set(kn) else float("inf")
            if np.isfinite(worst):
                ib = np.array([kb[q] for q in sorted(kb)])
                im = np.array([kn[q] for q in sorted(kb)])
                for c in ("g_src", "c_src", "v0", "v_L"):
                    x, y = np.asarray(B[c], np.float64)[ib], np.asarray(N[c], np.float64)[im]
                    if not np.array_equal(np.isfinite(x), np.isfinite(y)):
                        worst = float("inf")
                    else:
                        f = np.isfinite(x)
                        worst = max(worst, float(np.max(np.abs(x[f] - y[f]))) if f.any() else 0.0)
            y3[key] = {"n_first_ticks": len(kb), "max_abs_first_tick_inputs": worst, "PASS": bool(len(kb) == 2399 and worst <= 1e-6)}
        # the reader reads logged frames only: at the cache steps both intervals share, its read is equal
        B, N, Sb, Sn = npz(a.pull, "I10RQ50", k), npz(a.pull, "I10RQ50M5", k), npz(a.pull, "I10RQ50", k, "_rd"), npz(a.pull, "I10RQ50M5", k, "_rd")
        if any(v is None for v in (B, N, Sb, Sn)):
            y4[k] = {"PASS": None, "why": "a log is missing"}
        else:
            kb, _ = keyed(B, None)
            kn, _ = keyed(N, None)
            shared = [q for q in kb if q in kn]
            ib, im = np.array([kb[q] for q in shared]), np.array([kn[q] for q in shared])
            same = all(np.array_equal(np.asarray(Sb[c])[ib], np.asarray(Sn[c])[im], equal_nan=True) for c in ("p", "g_read", "g_used", "ok"))
            y4[k] = {"n_shared_ticks": len(shared), "n_banked_ticks": len(kb), "same_reader_reads": bool(same), "PASS": bool(same and len(shared) == len(kb))}
    for name, d in (("y2_interval_applied", y2), ("y3_first_tick_inputs_equal", y3), ("y4_same_read_at_shared_ticks", y4)):
        C[name] = {"per_run": d, "PASS": (None if any(v["PASS"] is None for v in d.values()) else bool(all(v["PASS"] for v in d.values())))}
    nfiles = {}
    for arm, k in NEW_RUNS:
        d = os.path.join(a.pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k))
        nfiles["%s_s%d" % (arm, k)] = len([f for f in os.listdir(d) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(d) else None
    cnt = (hw or {}).get("counts", {})
    C["y5_counts"] = {"dump_files": nfiles, "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and len(nfiles) == 7 and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or hashlib.sha256(open(os.path.join(base, p), "rb").read()).hexdigest() != h]
    C["y6_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "PASS": bool(len(ru["files"]) == 10 and not bad)}
    C["y7_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["y7_reader_controls"]["PASS"] = bool(C["y7_reader_controls"]["headway"] and C["y7_reader_controls"]["family"])
    sh = {}
    for arm, k in NEW_RUNS:
        p = os.path.join(a.pull, "%s_s%d" % (arm, k), "ticks", "run_%s_s%d.txt" % (arm, k))
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
        sh["%s_s%d" % (arm, k)] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
    C["y8_model"] = {"per_run": sh, "PASS": bool(len(sh) == 7 and all(sh.values()))}
    return C


def cell(hw, A, B, c):
    v = hw["pairs"]["%s_minus_%s" % (A, B)]["cells"][c]
    return {"per_seed": [round(float(x), 4) for x in v["d_s"]], "pooled": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)]}


def fams(q3, A, B):
    return {f: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]], "verdict": v["verdict"]}
            for f, v in q3["pairs"]["%s_minus_%s" % (A, B)]["families"].items()}


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
    res = {"card": "PREREG_V8F_PROOF_ADD52", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "names": None, "levels": None, "beside": None}
    print(TIER_LINE)
    print("== controls (y1 to y8)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:640]))
    if ok:
        lv = hw["levels"]
        levels = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for _, b in FORMS for arm in (b, b + "M5")}
        names, cy, ru = {}, {}, {}
        for fk, base in FORMS:
            cy[fk] = {c: cell(hw, base + "M5", base, c) for c in ("U", "C", "F")}
            d = cy[fk]["U"]
            if all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0:
                names["cycle_" + fk] = "CYCLE10K-%s-FASTER-CYCLE-CUTS-UNSAFE" % fk
            elif d["interval"][0] > 0:
                names["cycle_" + fk] = "CYCLE10K-%s-FASTER-CYCLE-RAISES-UNSAFE" % fk
            else:
                names["cycle_" + fk] = "CYCLE10K-%s-NO-U-CHANGE" % fk
        for fk, base in FORMS[1:]:
            ru[fk] = {"M5": {c: cell(hw, base + "M5", "I10R0M5", c) for c in ("U", "C", "F")}, "banked": {c: cell(hw, base, "I10R0", c) for c in ("U", "C", "F")}}
            d = ru[fk]["M5"]["U"]
            names["rule_" + fk] = "CYCLE10K-%s-RULE-CUTS-UNSAFE-AT-1S" % fk if (all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0) else "CYCLE10K-%s-RULE-NOT-SEPARATED-AT-1S" % fk
        bes = {"families_cycle": {fk: fams(q3, base + "M5", base) for fk, base in FORMS}, "families_rule_at_1s": {fk: fams(q3, base + "M5", "I10R0M5") for fk, base in FORMS[1:]}}
        res["names"], res["levels"], res["cycle"], res["rule"], res["beside"] = names, levels, cy, ru, bes
        print("== names")
        for k, v in names.items():
            print("   %-14s %s" % (k, v))
        print("== levels (per model seed)")
        for arm, v in levels.items():
            print("   %-10s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        print("== the cycle: 1.0 s minus 2.0 s, per form (pooled [interval], per seed)")
        for fk, v in cy.items():
            print("   %-7s %s" % (fk, "  ".join("%s %+.4f [%+.4f, %+.4f] %s" % (c, v[c]["pooled"], v[c]["interval"][0], v[c]["interval"][1], v[c]["per_seed"]) for c in ("U", "C", "F"))))
        print("== the rule against no rule, at 1.0 s and at 2.0 s")
        for fk, v in ru.items():
            for w in ("M5", "banked"):
                print("   %-7s %-6s %s" % (fk, w, "  ".join("%s %+.4f [%+.4f, %+.4f] %s" % (c, v[w][c]["pooled"], v[w][c]["interval"][0], v[w][c]["interval"][1], v[w][c]["per_seed"]) for c in ("U", "C", "F"))))
        print("== families")
        for grp, d in bes.items():
            for fk, fm in d.items():
                for f, v in fm.items():
                    print("   %-20s %-7s %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (grp, fk, f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== a control did not pass: no name, the differences stay withheld")
    print("STRATEGIC: as note N1 of card ADD46 (T0 first tick: the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
