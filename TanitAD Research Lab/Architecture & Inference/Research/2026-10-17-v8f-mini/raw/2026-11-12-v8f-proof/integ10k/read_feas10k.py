"""read_feas10k.py -- card PREREG_V8F_PROOF_ADD53 (FEAS10K) with its Amendment 1: the read (numpy only). Controls f1 to f10 first; the names and the differences only if
every control passed. Forms: NORULE (R0), LABEL (label-fed rule with the label closing speed, RLc), READER (reader-fed rule at map level 0.50 with closing 0, RQ50).
A5 = replan interval 5 steps (1.0 s) with the limits 3.0 / 8.0 m/s2 on the executed plan; M5 = 5 steps without a limit (card ADD52); the banked runs use 10 steps, no limit.
Rules as frozen in the card:
  per form   SP = speed error (A5) - speed error (M5)          the limit at the 1.0 s cycle        RS = share of windows with last-step speed above logged + 5 m/s
  per form   GC = U(A5) - U(banked 2.0 s)                      against the programme's cycle
  no rule    PR = R0A10 - R0                                   the limit at the programme's cycle
  per rule   RU = U(rule form, A5) - U(R0A5)                   the rule at 1.0 s under the limit"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

SEEDS = (0, 1)
DT = 0.2
CKPT_SHA_PREFIX = {0: "2fd4b747532f55b2", 1: "f13df8ce026eed31"}                       # card ADD46: UE_s0, UE_s1
WRAPPER_SHA_PREFIX = "bf19862738459c84"                                               # the wrapper the unit test ran with (freeze line of the card in TIMES_NOTE.md)
FORMS = (("NORULE", "I10R0"), ("LABEL", "I10RLc"), ("READER", "I10RQ50"))
LIMITS = {"I10R0AID": (1e9, 1e9, 10), "I10R0A5": (3.0, 8.0, 5), "I10RLcA5": (3.0, 8.0, 5), "I10RQ50A5": (3.0, 8.0, 5), "I10R0A10": (3.0, 8.0, 10), "I10R0A5L": (1.5, 8.0, 5)}
NEW_RUNS = [("I10R0AID", 0)] + [(arm, k) for arm in ("I10R0A5", "I10RLcA5", "I10RQ50A5", "I10R0A10", "I10R0A5L") for k in SEEDS]
NORULE_A = ("I10R0A5", "I10R0A10", "I10R0A5L")                                         # f5 has its bar on these (Amendment 1); the rule forms are reported (r2)
RS_BAR = {"NORULE": (0.1768, 0.1726), "LABEL": (0.1534, 0.1442), "READER": (0.1284, 0.1242)}      # the card: twice the banked 2.0 s value, per model seed
RS_KNOWN = {"I10R0": (0.0884, 0.0863), "I10RLc": (0.0767, 0.0721), "I10RQ50": (0.0642, 0.0621), "I10R0M5": (0.9075, 0.9825), "I10RLcM5": (0.7078, 0.7874), "I10RQ50M5": (0.6294, 0.6886)}
SPEED = "LONGITUDINAL:speed_mae_mps"
READER_FLOOR = 0.0196
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if p and os.path.exists(p) else ""


def alog(pull, arm, k):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "alim_log_%s_s%d.npz" % (arm, k))
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def dump(pull, arm, k, keys=("g", "cl", "fl", "v0")):
    files = sorted(glob.glob(os.path.join(pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k), "ep*.npz")))
    if len(files) != 141:
        return None
    out = {n: [] for n in keys}
    for f in files:
        z = np.load(f)
        for n in keys:
            out[n].append(np.asarray(z[n], np.float64))
    return {n: np.concatenate(v) for n, v in out.items()}


def replan_of_result(pull, arm, k):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "t1v8f_%s_s%d_m10_causal_heads.json" % (arm, k))
    m = re.search(r'"replan_steps": (\d+)', io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
    return int(m.group(1)) if m else None


def speeds(tr):
    p = np.concatenate([np.zeros((len(tr), 1, 2)), tr], 1)
    return np.linalg.norm(np.diff(p, axis=1), axis=-1) / DT


def rs_of(d, key="cl"):
    """share of windows whose last-step speed lies more than 5 m/s above the logged one (finite rollouts)."""
    s, sg = speeds(d[key]), speeds(d["g"])
    fin = np.isfinite(d[key]).all(axis=(1, 2))
    return round(float(((s[:, -1] - sg[:, -1])[fin] > 5.0).mean()), 4)


def ticks_of(nb, T):
    """rows of a limit log by tick: the log holds, chunk after chunk, T calls of nb rows each. -> list over ticks of row-index arrays in dump order, or None."""
    nb = np.asarray(nb, np.int64)
    if len(nb) % T:
        return None
    start = np.concatenate([[0], np.cumsum(nb)])
    rows = [[] for _ in range(T)]
    for c in range(len(nb)):
        if nb[c] != nb[c - c % T]:
            return None
        rows[c % T].append(np.arange(start[c], start[c + 1]))
    return [np.concatenate(r) for r in rows]


def block_share(cl, m, a_max, b_lim=8.0, tol=0.25):
    """inside each executed block (never across a tick): share of steps whose chord acceleration lies above a_max + tol or below -(b_lim + tol)."""
    s = speeds(cl)
    acc = np.diff(s, axis=1) / DT                                  # acc[:, i] = from step i+1 to step i+2 (1-based steps)
    inside = np.array([(i + 1) % m != 0 for i in range(acc.shape[1])])
    x = acc[:, inside]
    f = np.isfinite(x)
    return float((((x > a_max + tol) | (x < -(b_lim + tol))) & f).sum() / max(int(f.sum()), 1))


def controls(a, hw, q3):
    C, R = {}, {}
    p = os.path.join(a.pull, "small_feas", "CMP_R0AID.txt")
    m = re.search(r"CMP files=(\d+) differing=(\d+)", io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else "")
    C["f1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    f2, f3, f4, f5, r1, r2, nfiles, shs, rs = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for arm, k in NEW_RUNS:
        key = "%s_s%d" % (arm, k)
        a_max, b_lim, m_steps = LIMITS[arm]
        T = 30 // m_steps
        L, D = alog(a.pull, arm, k), dump(a.pull, arm, k)
        nfiles[key] = None if D is None else 141
        pr = os.path.join(a.pull, key, "ticks", "run_%s.txt" % key)
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(pr, encoding="utf-8", errors="replace").read()) if os.path.exists(pr) else None
        shs[key] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
        if L is None or D is None:
            f2[key] = f3[key] = f4[key] = {"PASS": None, "why": "limit log or dump missing"}
            continue
        f2[key] = {"a_max": float(L["a_max"]), "b_lim": float(L["b_lim"]), "log_steps": int(L["replan_steps"]), "result_file_steps": replan_of_result(a.pull, arm, k)}
        f2[key]["PASS"] = bool(f2[key]["a_max"] == a_max and f2[key]["b_lim"] == b_lim and f2[key]["log_steps"] == m_steps and f2[key]["result_file_steps"] == m_steps)
        worst, rows_ok, first_ok, first_max = 0.0, True, True, 0.0
        for tag in ("m", "f"):
            v0, ve = np.asarray(L[tag + "_v0"], np.float64), np.asarray(L[tag + "_v_exec"], np.float64)
            ok = np.isfinite(v0)
            prev = np.concatenate([v0[:, None], ve[:, :-1]], 1)
            up = (ve - (prev + a_max * DT))[ok]
            dn = (np.maximum(0.0, prev - b_lim * DT) - ve)[ok]
            worst = max(worst, float(up.max()) if up.size else 0.0, float(dn.max()) if dn.size else 0.0)
            tk = ticks_of(L[tag + "_nb"], T)
            rows_ok = rows_ok and tk is not None and all(len(r) == 2399 for r in tk) and ve.shape == (2399 * T, m_steps)
            if tk is not None and len(tk[0]) == len(D["v0"]):
                d0 = float(np.nanmax(np.abs(v0[tk[0]] - D["v0"])))
                first_max = max(first_max, d0)
                first_ok = first_ok and d0 <= 1e-4
            else:
                first_ok = False
            if tag == "m" and tk is not None and arm != "I10R0AID":
                dd = np.concatenate([np.abs(v0[tk[n + 1]] - ve[tk[n], -1]) for n in range(T - 1)])
                dd = dd[np.isfinite(dd)]
                r1[key] = {"median": round(float(np.median(dd)), 4), "p99": round(float(np.quantile(dd, 0.99)), 4), "n": int(dd.size)}
        f3[key] = {"largest_excess_mps": worst, "PASS": bool(worst <= 1e-6)}
        f4[key] = {"rows_per_tick_2399": bool(rows_ok), "first_tick_v0_max_abs_diff": first_max, "PASS": bool(rows_ok and first_ok)}
        if arm != "I10R0AID":
            sh_ = round(block_share(D["cl"], m_steps, a_max), 5)
            if arm in NORULE_A:
                f5[key] = {"share": sh_, "PASS": bool(sh_ <= 0.01)}
            else:
                r2[key] = {"share": sh_}
            rs[key] = {"model": rs_of(D, "cl"), "prior": rs_of(D, "fl")}
    known = {}
    for arm, exp in RS_KNOWN.items():
        got = []
        for k in SEEDS:
            D = dump(a.pull, arm, k)
            got.append(None if D is None else rs_of(D, "cl"))
            if D is not None:
                rs["%s_s%d" % (arm, k)] = {"model": got[-1], "prior": rs_of(D, "fl")}
        known[arm] = {"got": got, "card": list(exp), "PASS": bool(all(g is not None and abs(g - e) <= 5e-5 for g, e in zip(got, exp)))}
    for name, d in (("f2_parameters_applied", f2), ("f3_limit_holds_in_the_log", f3), ("f4_rows_are_windows", f4), ("f5_executed_track_obeys_limit", f5)):
        C[name] = {"per_run": d, "PASS": (None if (not d or any(v["PASS"] is None for v in d.values())) else bool(all(v["PASS"] for v in d.values())))}
    C["f5_executed_track_obeys_limit"]["n_runs"] = len(f5)
    C["f5_executed_track_obeys_limit"]["PASS"] = bool(C["f5_executed_track_obeys_limit"]["PASS"] and len(f5) == 6)
    cnt = (hw or {}).get("counts", {})
    C["f6_counts"] = {"dump_files": nfiles, "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and len(nfiles) == 11 and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or sha(os.path.join(base, p)) != h]
    C["f7_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "rs_known_values": known, "PASS": bool(len(ru["files"]) == 12 and not bad and all(v["PASS"] for v in known.values()))}
    C["f8_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["f8_reader_controls"]["PASS"] = bool(C["f8_reader_controls"]["headway"] and C["f8_reader_controls"]["family"])
    C["f9_model"] = {"per_run": shs, "PASS": bool(len(shs) == 11 and all(shs.values()))}
    t_txt = io.open(a.test_out, encoding="utf-8", errors="replace").read() if a.test_out and os.path.exists(a.test_out) else ""
    w1, w2 = sha(a.wrapper), sha(a.test_wrapper)
    C["f10_unit_test"] = {"wrapper_sha256": w1[:16], "test_wrapper_sha256": w2[:16], "test_line": (re.findall(r"ZZ TEST-ACCEL-LIMIT.*", t_txt) or [None])[-1],
                          "PASS": bool(w1 and w1 == w2 and w1.startswith(WRAPPER_SHA_PREFIX) and "ZZ TEST-ACCEL-LIMIT PASS 18/18" in t_txt)}
    R["r1_tick_speed_against_last_executed"] = r1
    R["r2_rule_forms_block_share"] = r2
    return C, R, rs


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
    ap.add_argument("--test-wrapper", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hw, q3 = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    C, R, rs = controls(a, hw, q3)
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": "PREREG_V8F_PROOF_ADD53", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "reported_no_bar": R, "names": None}
    print(TIER_LINE)
    print("== controls (f1 to f10)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:700]))
    print("== reported, no bar")
    for k, v in R.items():
        print("   %s %s" % (k, json.dumps(v)[:900]))
    if ok:
        lv = hw["levels"]
        arms = [b + s for _, b in FORMS for s in ("", "M5", "A5")] + ["I10R0A10", "I10R0A5L"]
        levels = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in arms}
        names, sp, gc, ru = {}, {}, {}, {}
        for fk, base in FORMS:
            fm = fams(q3, base + "A5", base + "M5")
            d = fm[SPEED]
            rs_a = [rs["%sA5_s%d" % (base, k)]["model"] for k in SEEDS]
            sp[fk] = {"speed_error": d, "RS_A5": rs_a, "RS_bar": list(RS_BAR[fk]), "RS_M5": [rs["%sM5_s%d" % (base, k)]["model"] for k in SEEDS], "RS_2s": [rs["%s_s%d" % (base, k)]["model"] for k in SEEDS],
                      "headway": ucf(hw, base + "A5", base + "M5"), "families": fm}
            if all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0:
                names["limit_" + fk] = "FEAS10K-%s-LIMIT-%s" % (fk, "STABILISES-1S" if all(r <= b for r, b in zip(rs_a, RS_BAR[fk])) else "SLOWS-CLIMB")
            else:
                names["limit_" + fk] = "FEAS10K-%s-LIMIT-NO-EFFECT" % fk
            gc[fk] = {"headway": ucf(hw, base + "A5", base), "families": fams(q3, base + "A5", base)}
            u = gc[fk]["headway"]["U"]
            names["u_" + fk] = "FEAS10K-%s-%s" % (fk, "U-ABOVE-2S" if u["interval"][0] > 0 else ("U-BELOW-2S" if u["interval"][1] < 0 else "U-NOT-SEPARATED-FROM-2S"))
            if fk == "READER":
                gc[fk]["inside_reader_floor"] = bool(abs(u["pooled"]) <= READER_FLOOR)
        pr = {"headway": ucf(hw, "I10R0A10", "I10R0"), "families": fams(q3, "I10R0A10", "I10R0")}
        worse = any(v["interval"][0] > 0 for v in pr["headway"].values()) or any(v["verdict"] == "WORSE" for v in pr["families"].values())
        better = any(v["interval"][1] < 0 for v in pr["headway"].values()) or any(v["verdict"] == "BETTER" for v in pr["families"].values())
        names["cycle_2s"] = "FEAS10K-LIMIT-%s-AT-2S" % ("COSTS" if worse else ("HELPS" if better else "NEUTRAL"))
        for fk, base in FORMS[1:]:
            ru[fk] = {"A5": ucf(hw, base + "A5", "I10R0A5"), "families": fams(q3, base + "A5", "I10R0A5")}
            d = ru[fk]["A5"]["U"]
            names["rule_" + fk] = "FEAS10K-%s-RULE-%s-AT-1S-LIMITED" % (fk, "CUTS-UNSAFE" if (all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0) else "NOT-SEPARATED")
        dose = {"A5L_minus_A5": {"headway": ucf(hw, "I10R0A5L", "I10R0A5"), "families": fams(q3, "I10R0A5L", "I10R0A5")},
                "A5L_minus_M5": {"headway": ucf(hw, "I10R0A5L", "I10R0M5"), "families": fams(q3, "I10R0A5L", "I10R0M5")}}
        res.update(names=names, levels=levels, RS=rs, SP=sp, GC=gc, PR=pr, RU=ru, dose=dose)
        print("== names")
        for k, v in names.items():
            print("   %-14s %s" % (k, v))
        print("== levels (per model seed)")
        for arm, v in levels.items():
            print("   %-11s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        print("== RS: share of windows with last-step speed more than 5 m/s above logged (model rollout / prior alone), per model seed")
        for key in sorted(rs):
            print("   %-14s model %.4f  prior %.4f" % (key, rs[key]["model"], rs[key]["prior"]))
        print("== SP: the limit at 1.0 s (A5 minus M5), per form")
        for fk, v in sp.items():
            d = v["speed_error"]
            print("   %-7s speed error %+.4f [%+.4f, %+.4f] %s %s | RS A5 %s bar %s (M5 %s, 2.0 s %s) | %s" % (fk, d["D"], d["interval"][0], d["interval"][1], d["per_seed"], d["verdict"], v["RS_A5"], v["RS_bar"], v["RS_M5"], v["RS_2s"], line(v["headway"])))
        print("== GC: A5 against the banked 2.0 s run of the same form")
        for fk, v in gc.items():
            print("   %-7s %s%s" % (fk, line(v["headway"]), "  (inside the reader floor)" if v.get("inside_reader_floor") else ""))
        print("== PR: the limit at the programme's cycle (R0A10 minus R0)")
        print("   %s" % line(pr["headway"]))
        print("== RU: the rule against no rule at 1.0 s under the limit")
        for fk, v in ru.items():
            print("   %-7s %s" % (fk, line(v["A5"])))
        print("== dose (acceleration limit 1.5 m/s2, no rule; names nothing)")
        for kk, v in dose.items():
            d = v["families"][SPEED]
            print("   %-13s speed error %+.4f [%+.4f, %+.4f] %s | %s" % (kk, d["D"], d["interval"][0], d["interval"][1], d["per_seed"], line(v["headway"])))
        print("== families")
        groups = [("SP_" + fk, v["families"]) for fk, v in sp.items()] + [("GC_" + fk, v["families"]) for fk, v in gc.items()] + [("PR", pr["families"])] + [("RU_" + fk, v["families"]) for fk, v in ru.items()] + \
                 [("dose_" + kk, v["families"]) for kk, v in dose.items()]
        for grp, fm in groups:
            for f, v in fm.items():
                print("   %-18s %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (grp, f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== a control did not pass: no name, the differences stay withheld")
    print("STRATEGIC: as note N1 of card ADD46 (T0 first tick: the banked read of the two models without the rule; along the rollout NOT COMPUTED, n = 0, no instrument)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
