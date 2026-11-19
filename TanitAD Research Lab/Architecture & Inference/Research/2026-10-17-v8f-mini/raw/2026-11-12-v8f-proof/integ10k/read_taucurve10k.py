"""read_taucurve10k.py -- card PREREG_V8F_PROOF_ADD51 (TAUCURVE10K): the read (numpy only). Controls t1 to t8 first; the names and the differences only if every control
passed. Forms: L = label-fed rule with the label closing speed (RLc), R = reader-fed rule at map level 0.50 with closing 0 (RQ50). Time gaps 0.8, 1.0, 1.2 (banked), 1.5, 2.0 s.
Rules as frozen in the card, per form: LONG = U(tau 2.0) - U(tau 1.2), its price on F; SHORT = U(tau 0.8) - U(tau 1.2), its gain on F."""
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
TAUS = (("08", 0.8), ("10", 1.0), ("15", 1.5), ("20", 2.0))
FORMS = (("L", "I10RLc"), ("R", "I10RQ50"))
NEW_RUNS = [("I10RLcT12", 0)] + [("%sT%s" % (b, tn), k) for tn, _ in TAUS for k in SEEDS for _, b in FORMS]
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"
TOL = 1e-9


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def npz(pull, arm, k, suffix=""):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d%s.npz" % (arm, k, suffix))
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def controls(a, hw, q3):
    C = {}
    p = os.path.join(a.pull, "small_taucurve", "CMP_RLcT12.txt")
    txt = io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
    m = re.search(r"CMP files=(\d+) differing=(\d+)", txt)
    C["t1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    t2, t3, t4 = {}, {}, {}
    for fk, base in FORMS:
        for k in SEEDS:
            B = npz(a.pull, base, k)
            key = "%s_s%d" % (fk, k)
            runs = {tn: npz(a.pull, "%sT%s" % (base, tn), k) for tn, _ in TAUS}
            if B is None or any(v is None for v in runs.values()):
                t2[key] = t3[key] = {"PASS": None, "why": "a rule log is missing"}
                continue
            got = {tn: float(runs[tn]["TAU"]) for tn, _ in TAUS}
            t2[key] = {"banked": float(B["TAU"]), "runs": got, "PASS": bool(abs(float(B["TAU"]) - 1.2) <= TOL and all(abs(got[tn] - tv) <= TOL for tn, tv in TAUS)
                                                                         and all(json.loads(str(runs[tn]["mode_kwargs"])).get("tau") == tv for tn, tv in TAUS))}
            # the inputs of the first tick do not depend on the time gap: equal source gap, closing speed, ego speed and lead speed at tick 0
            m0 = np.asarray(B["tick"]) == 0
            worst, same_ticks = 0.0, True
            for tn, _ in TAUS:
                Rn = runs[tn]
                same_ticks = same_ticks and all(np.array_equal(B[c], Rn[c]) for c in ("ci", "t0", "tick"))
                for c in ("g_src", "c_src", "v0", "v_L"):
                    x, y = np.asarray(B[c], np.float64)[m0], np.asarray(Rn[c], np.float64)[m0]
                    if len(x) != len(y) or not np.array_equal(np.isfinite(x), np.isfinite(y)):
                        worst = float("inf")
                    else:
                        f = np.isfinite(x)
                        worst = max(worst, float(np.max(np.abs(x[f] - y[f]))) if f.any() else 0.0)
            ch = {tn: int(np.asarray(runs[tn]["changed"])[m0].astype(bool).sum()) for tn, _ in TAUS}
            t3[key] = {"max_abs_first_tick_inputs": worst, "same_ticks": bool(same_ticks), "n_first_ticks": int(m0.sum()), "changed_at_first_tick": dict(ch, **{"12": int(np.asarray(B["changed"])[m0].astype(bool).sum())}),
                       "PASS": bool(same_ticks and worst <= 1e-6)}
    for k in SEEDS:                                                                       # the reader reads logged frames only: equal at every tick for every time gap
        Rq = npz(a.pull, "I10RQ50", k, "_rd")
        sides = {tn: npz(a.pull, "I10RQ50T%s" % tn, k, "_rd") for tn, _ in TAUS}
        if Rq is None or any(v is None for v in sides.values()):
            t4[k] = {"PASS": None, "why": "a side log is missing"}
            continue
        same = all(np.array_equal(Rq[c], sides[tn][c], equal_nan=True) for tn, _ in TAUS for c in ("p", "g_read", "g_used", "ok"))
        t4[k] = {"same_reader_reads": bool(same), "PASS": bool(same)}
    for name, d in (("t2_time_gap_applied", t2), ("t3_first_tick_inputs_equal", t3), ("t4_same_read", t4)):
        C[name] = {"per_run": d, "PASS": (None if any(v["PASS"] is None for v in d.values()) else bool(all(v["PASS"] for v in d.values())))}
    nfiles = {}
    for arm, k in NEW_RUNS:
        d = os.path.join(a.pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k))
        nfiles["%s_s%d" % (arm, k)] = len([f for f in os.listdir(d) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(d) else None
    cnt = (hw or {}).get("counts", {})
    C["t5_counts"] = {"n_runs": len(nfiles), "not_141": [n for n, v in nfiles.items() if v != 141], "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and len(nfiles) == 17 and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or hashlib.sha256(open(os.path.join(base, p), "rb").read()).hexdigest() != h]
    C["t6_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "PASS": bool(len(ru["files"]) == 10 and not bad)}
    C["t7_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["t7_reader_controls"]["PASS"] = bool(C["t7_reader_controls"]["headway"] and C["t7_reader_controls"]["family"])
    sh = {}
    for arm, k in NEW_RUNS:
        p = os.path.join(a.pull, "%s_s%d" % (arm, k), "ticks", "run_%s_s%d.txt" % (arm, k))
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
        sh["%s_s%d" % (arm, k)] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
    C["t8_model"] = {"n_runs": len(sh), "not_equal": [n for n, v in sh.items() if not v], "PASS": bool(len(sh) == 17 and all(sh.values()))}
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
    res = {"card": "PREREG_V8F_PROOF_ADD51", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "names": None, "curve": None, "beside": None}
    print(TIER_LINE)
    print("== controls (t1 to t8)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:640]))
    if ok:
        lv = hw["levels"]
        curve, names, diffs = {}, {}, {}
        for fk, base in FORMS:
            word = "LABEL" if fk == "L" else "READER"
            arms = [("0.8", base + "T08"), ("1.0", base + "T10"), ("1.2", base), ("1.5", base + "T15"), ("2.0", base + "T20")]
            curve[fk] = {tv: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for tv, arm in arms}
            diffs[fk] = {tv: {c: cell(hw, arm, base, c) for c in ("U", "C", "F")} for tv, arm in arms if arm != base}
            LU, LF = diffs[fk]["2.0"]["U"], diffs[fk]["2.0"]["F"]
            if all(x < 0 for x in LU["per_seed"]) and LU["interval"][1] < 0:
                n1 = "TAUCURVE10K-%s-LONGER-GAP-CUTS-UNSAFE" % word + ("-AT-A-PRICE" if LF["interval"][0] > 0 else "-AT-NO-PRICE")
            else:
                n1 = "TAUCURVE10K-%s-NO-CUT-BEYOND-1.2" % word
            SU, SF = diffs[fk]["0.8"]["U"], diffs[fk]["0.8"]["F"]
            if all(x > 0 for x in SU["per_seed"]) and SU["interval"][0] > 0:
                n2 = "TAUCURVE10K-%s-SHORTER-GAP-COSTS-SAFETY" % word
            else:
                n2 = "TAUCURVE10K-%s-SHORTER-GAP-NO-SAFETY-COST" % word
            if SF["interval"][1] < 0:
                n2 += "-LESS-FALLING-BEHIND"
            names[fk] = {"long": n1, "short": n2}
        bes = {"reader_at_longer_gap_against_label_rule_at_1.2": {tv: {c: cell(hw, "I10RQ50T%s" % tn, "I10RLc", c) for c in ("U", "C", "F")} for tn, tv in (("15", "1.5"), ("20", "2.0"))},
               "against_no_rule": {arm: {c: cell(hw, arm, "I10R0", c) for c in ("U", "C", "F")} for arm in ("I10RLcT20", "I10RQ50T20")},
               "families": {"%s_tau_%s_minus_1.2" % (fk, tv): fams(q3, base + "T" + tn, base) for fk, base in FORMS for tn, tv in (("08", "0.8"), ("20", "2.0"))}}
        res["names"], res["curve"], res["differences_against_1.2"], res["beside"] = names, curve, diffs, bes
        print("== names")
        for fk, _ in FORMS:
            print("   form %s: %s ; %s" % (fk, names[fk]["long"], names[fk]["short"]))
        print("== levels per time gap (per model seed)")
        for fk, _ in FORMS:
            for tv, v in curve[fk].items():
                print("   %s tau %s  U %s  C %s  F %s" % (fk, tv, v["U"], v["C"], v["F"]))
        print("== differences against the time gap 1.2 s (pooled [interval], per seed)")
        for fk, _ in FORMS:
            for tv, v in diffs[fk].items():
                print("   %s tau %s  %s" % (fk, tv, "  ".join("%s %+.4f [%+.4f, %+.4f] %s" % (c, v[c]["pooled"], v[c]["interval"][0], v[c]["interval"][1], v[c]["per_seed"]) for c in ("U", "C", "F"))))
        for tv, v in bes["reader_at_longer_gap_against_label_rule_at_1.2"].items():
            print("   beside: reader at %s against the label rule with closing at 1.2: %s" % (tv, json.dumps(v)[:400]))
        for arm, v in bes["against_no_rule"].items():
            print("   beside: %s against no rule: %s" % (arm, json.dumps(v)[:400]))
        print("== families (time gap against 1.2 s)")
        for nm, fm in bes["families"].items():
            for f, v in fm.items():
                print("   %-22s %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (nm, f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== a control did not pass: no name, the differences stay withheld")
    print("STRATEGIC: as note N1 of card ADD46 (T0 first tick: the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
