"""read_gaprate10k.py -- card PREREG_V8F_PROOF_ADD50 (GAPRATE10K): the read (numpy only). Controls d1 to d10 first; the names and the differences of the arm RQ50d only
if every control passed. The crossed-reader block X (RQ50x against RQ50) has its own controls (d1, d7 to d10) and names nothing.
Rules as frozen in the card: S1 = U(RQ50d) - U(RQ50) (primary), S2 = C(RQ50d) - C(RQ50), S3 = F(RQ50d) - F(RQ50); reader floor = the larger absolute value over the two
model seeds of U(RQ50x) - U(RQ50)."""
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
NEW_RUNS = (("I10RQ50d", 0), ("I10RQ50d", 1), ("I10RQ50Rd", 0), ("I10RQ50x", 0), ("I10RQ50x", 1))
TIER_LINE = "TIER: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance"
R_BAR, COV_BAR, TOL, G_MAX, K_DIFF, P_MIN = 0.3, 0.3, 1e-4, 60.0, 25, 0.5


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def npz(pull, arm, k, suffix=""):
    p = os.path.join(pull, "%s_s%d" % (arm, k), "ticks", "rule_log_%s_s%d%s.npz" % (arm, k, suffix))
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def controls(a, hw, q3):
    C = {}
    p = os.path.join(a.pull, "small_gaprate", "CMP_RQ50Rd.txt")
    txt = io.open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
    m = re.search(r"CMP files=(\d+) differing=(\d+)", txt)
    C["d1_rerun_equals_banked"] = {"line": m.group(0) if m else None, "PASS": bool(m and m.group(1) == "141" and m.group(2) == "0")}
    t0 = jl(a.t0) or {}
    d2, d3, d4, d5, d6 = {}, {}, {}, {}, {}
    for k in SEEDS:
        Lc, Ld, Rq, Rd = npz(a.pull, "I10RLc", k), npz(a.pull, "I10RQ50d", k), npz(a.pull, "I10RQ50", k, "_rd"), npz(a.pull, "I10RQ50d", k, "_rd")
        if any(v is None for v in (Lc, Ld, Rq, Rd)) or any(c not in Rd.files for c in ("c_fed", "have", "id0", "g_prev", "p_prev", "ok_prev")):
            d2[k] = d3[k] = d4[k] = d6[k] = {"PASS": None, "why": "a log or a column of the form is missing"}
        else:
            same_ticks = all(np.array_equal(Lc[c], Ld[c]) for c in ("ci", "t0", "tick"))
            lab = np.asarray(Lc["present"]).astype(bool) & np.isfinite(np.asarray(Lc["g_src"], np.float64)) & (np.asarray(Lc["g_src"], np.float64) <= G_MAX)
            have = np.asarray(Rd["have"]).astype(bool)
            rp = (np.asarray(Rd["p"], np.float64) >= P_MIN) & np.asarray(Rd["ok"]).astype(bool)
            cf, cl = np.asarray(Rd["c_fed"], np.float64), np.asarray(Lc["c_src"], np.float64)
            m2 = lab & have & np.isfinite(cf) & np.isfinite(cl)
            r = float(np.corrcoef(cf[m2], cl[m2])[0, 1]) if m2.sum() > 10 else None
            cov = float(have.sum() / max(1, rp.sum()))
            d2[k] = {"n_ticks": int(m2.sum()), "pearson_r": r, "mae_mps": float(np.abs(cf[m2] - cl[m2]).mean()) if m2.any() else None, "mae_of_zero_mps": float(np.abs(cl[m2]).mean()) if m2.any() else None,
                     "mean_fed_mps": float(cf[m2].mean()) if m2.any() else None, "mean_label_mps": float(cl[m2].mean()) if m2.any() else None, "n_reader_present": int(rp.sum()), "n_with_closing_read": int(have.sum()),
                     "share_with_closing_read": cov, "PASS": bool(same_ticks and r is not None and r >= R_BAR and cov >= COV_BAR)}
            cs, vl, vL = (np.asarray(Ld[c], np.float64) for c in ("c_src", "v_l", "v_L"))
            d_feed = float(np.max(np.abs(cs - cf))) if len(cs) == len(cf) else None
            fin = np.isfinite(cs) & np.isfinite(vl) & np.isfinite(vL)
            ident = float(np.max(np.abs(vL[fin] - np.maximum(0.0, vl[fin] + cs[fin])))) if fin.any() else None
            zero_else = bool(np.all(cf[~have] == 0.0)) if (~have).any() else True
            cnt = json.loads(str(Ld["mix_counts"])) if "mix_counts" in Ld.files else {}
            slope_run = float(Ld["slope"]) if "slope" in Ld.files else None
            try:
                slope_t0 = float(t0["per_reader_seed"][str(k)]["per_k"][str(K_DIFF)]["check_clips"]["slope_used"])
            except (KeyError, TypeError):
                slope_t0 = None
            kd = int(Ld["k_diff"]) if "k_diff" in Ld.files else None
            d3[k] = {"max_abs_fed_minus_read_mps": d_feed, "max_abs_identity_mps": ident, "zero_where_no_read": zero_else, "closing_from_labels": cnt.get("closing_from_other"), "slope_run": slope_run, "slope_t0_file": slope_t0,
                     "k_diff": kd, "PASS": bool(d_feed is not None and d_feed <= TOL and ident is not None and ident <= TOL and zero_else and cnt.get("closing_from_other") == 0
                                                and slope_run is not None and slope_t0 is not None and abs(slope_run - slope_t0) <= 1e-12 and kd == K_DIFF)}
            id0 = np.asarray(Rd["id0"], np.float64)
            mx = float(np.max(id0)) if len(id0) else None
            d4[k] = {"max_abs_gather_at_zero": mx, "PASS": bool(mx is not None and np.isfinite(mx) and mx <= TOL)}
            same_read = all(np.array_equal(Rq[c], Rd[c], equal_nan=True) for c in ("p", "g_read", "g_used", "ok"))
            d6[k] = {"same_read_RQ50_RQ50d": bool(same_read), "same_ticks": bool(same_ticks), "PASS": bool(same_read and same_ticks)}
        pc = jl(os.path.join(a.pull, "small_gaprate", "PREV_CHECK_s%d.json" % k))
        d5[k] = {"PASS": None, "why": "no check file"} if pc is None else {"n_prev": pc["i4_recomputed"]["n_prev"], "max_abs_g_prev_m": pc["i4_recomputed"]["max_abs_g_prev_m"], "max_abs_p_prev": pc["i4_recomputed"]["max_abs_p_prev"],
                                                                         "flags_equal": bool(pc["i4_recomputed"]["flag_now_equal"] and pc["i4_recomputed"]["flag_prev_equal"]), "formula": pc["i5_formula"]["PASS"], "PASS": bool(pc.get("PASS") is True)}
    for name, d in (("d2_closing_read_T0", d2), ("d3_feed", d3), ("d4_gather_at_zero", d4), ("d5_earlier_read_recomputed", d5), ("d6_same_read", d6)):
        C[name] = {"per_seed": d, "PASS": (None if any(v["PASS"] is None for v in d.values()) else bool(all(v["PASS"] for v in d.values())))}
    nfiles = {}
    for arm, k in NEW_RUNS:
        d = os.path.join(a.pull, "%s_s%d" % (arm, k), "dump_%s_s%d_m10_causal_heads" % (arm, k))
        nfiles["%s_s%d" % (arm, k)] = len([f for f in os.listdir(d) if re.match(r"^ep.*npz$", f)]) if os.path.isdir(d) else None
    cnt = (hw or {}).get("counts", {})
    C["d7_counts"] = {"dump_files": nfiles, "n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"),
                      "PASS": bool(hw and all(v == 141 for v in nfiles.values()) and cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("PASS") is True)}
    ru = jl(a.reused) or {"files": {}}
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.reused)))
    bad = [p for p, h in ru["files"].items() if not os.path.exists(os.path.join(base, p)) or hashlib.sha256(open(os.path.join(base, p), "rb").read()).hexdigest() != h]
    C["d8_reused_dumps"] = {"n_files": len(ru["files"]), "differing": bad, "PASS": bool(len(ru["files"]) == 10 and not bad)}
    C["d9_reader_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS"))}
    C["d9_reader_controls"]["PASS"] = bool(C["d9_reader_controls"]["headway"] and C["d9_reader_controls"]["family"])
    sh = {}
    for arm, k in NEW_RUNS:
        p = os.path.join(a.pull, "%s_s%d" % (arm, k), "ticks", "run_%s_s%d.txt" % (arm, k))
        mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", io.open(p, encoding="utf-8", errors="replace").read()) if os.path.exists(p) else None
        sh["%s_s%d" % (arm, k)] = bool(mm and mm.group(1).startswith(CKPT_SHA_PREFIX[k]))
    cross = {}
    for k in SEEDS:                                                       # RQ50x at model seed k must carry the reader and the map of RQ50 at model seed 1-k
        X, O, S = npz(a.pull, "I10RQ50x", k), npz(a.pull, "I10RQ50", 1 - k), npz(a.pull, "I10RQ50", k)
        if any(v is None for v in (X, O, S)):
            cross[k] = None
            continue
        cross[k] = bool(str(X["reader_sha256"]) == str(O["reader_sha256"]) and str(X["qmap_sha256"]) == str(O["qmap_sha256"]) and str(X["reader_sha256"]) != str(S["reader_sha256"])
                        and all(np.array_equal(X[c], S[c]) for c in ("ci", "t0", "tick")))
    C["d10_model_and_reader"] = {"per_run": sh, "crossed_reader_is_the_other_seed": cross, "PASS": bool(all(sh.values()) and all(v is True for v in cross.values()))}
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
    ap.add_argument("--t0", required=True, help="CLOSING_DIFF_T0.json (the slopes fitted on the training-corpus clips)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hw, q3 = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json")), jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    C = controls(a, hw, q3)
    ok = all(v.get("PASS") is True for v in C.values())
    ok_x = all(C[k].get("PASS") is True for k in ("d1_rerun_equals_banked", "d7_counts", "d8_reused_dumps", "d9_reader_controls", "d10_model_and_reader"))
    res = {"card": "PREREG_V8F_PROOF_ADD50", "tier": TIER_LINE, "controls": C, "controls_all_pass": bool(ok), "controls_of_block_X_pass": bool(ok_x), "block_X": None, "names": None, "beside": None}
    print(TIER_LINE)
    print("== controls (d1 to d10)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else ("FAIL" if v.get("PASS") is False else "NOT COMPUTABLE"), json.dumps(v)[:640]))
    floor = None
    if ok_x:
        X = {c: cell(hw, "I10RQ50x", "I10RQ50", c) for c in ("U", "C", "F")}
        floor = max(abs(x) for x in X["U"]["per_seed"])
        lv = hw["levels"]
        res["block_X"] = {"what": "level 0.50 with the reader and the map of the other seed against the own reader; the spread a second reader gives; names nothing", "RQ50x_minus_RQ50": X, "reader_floor_U": round(floor, 4),
                          "levels_RQ50x_per_seed": {c: [round(float(x), 4) for x in lv["I10RQ50x"]["cells"][c]["per_seed"]] for c in ("U", "C", "F")}, "families_RQ50x_minus_RQ50": fams(q3, "I10RQ50x", "I10RQ50")}
        print("== block X (crossed reader seeds at level 0.50; names nothing)")
        for c in ("U", "C", "F"):
            print("   %s RQ50x - RQ50 %s" % (c, json.dumps(X[c])))
        print("   reader floor on U (larger absolute per-seed value): %.4f" % floor)
        for f, v in res["block_X"]["families_RQ50x_minus_RQ50"].items():
            print("   X %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== block X: one of its controls (d1, d7 to d10) did not pass: not read")
    if ok:
        S1, S2, S3 = (cell(hw, "I10RQ50d", "I10RQ50", c) for c in ("U", "C", "F"))
        if all(x < 0 for x in S1["per_seed"]) and S1["interval"][1] < 0:
            name = "GAPRATE10K-CUTS-UNSAFE"
        elif S1["interval"][0] > 0:
            name = "GAPRATE10K-HARMS"
        else:
            name = "GAPRATE10K-NO-U-CHANGE"
        if name != "GAPRATE10K-NO-U-CHANGE":
            name += "-ABOVE-READER-FLOOR" if abs(S1["pooled"]) > floor else "-WITHIN-READER-FLOOR"
        sec = []
        if S2["interval"][1] < 0:
            sec.append("-CUTS-CONTACT")
        if S3["interval"][1] < 0:
            sec.append("-LESS-FALLING-BEHIND")
        if S3["interval"][0] > 0:
            sec.append("-MORE-FALLING-BEHIND")
        res["names"] = {"name": name, "secondary": sec, "S1_U_RQ50d_minus_RQ50": S1, "S2_C_RQ50d_minus_RQ50": S2, "S3_F_RQ50d_minus_RQ50": S3, "reader_floor_U": round(floor, 4)}
        lv = hw["levels"]
        curve = {arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in ("I10R0", "I10RLb", "I10RLc", "I10RQ50", "I10RQ50c", "I10RQ50d", "I10RQ50x") if arm in lv}
        bes = {"levels_per_seed": curve}
        for B in ("I10RQ50c", "I10R0", "I10RLc"):
            bes["RQ50d_minus_%s" % B[3:]] = {c: cell(hw, "I10RQ50d", B, c) for c in ("U", "C", "F")}
        bes["families_RQ50d_minus_RQ50"] = fams(q3, "I10RQ50d", "I10RQ50")
        res["beside"] = bes
        print("== names")
        print("   %s ; secondary %s" % (name, sec if sec else "none"))
        for k in ("S1_U_RQ50d_minus_RQ50", "S2_C_RQ50d_minus_RQ50", "S3_F_RQ50d_minus_RQ50"):
            print("   %s %s" % (k, json.dumps(res["names"][k])))
        print("== levels (per model seed)")
        for arm, v in curve.items():
            print("   %-9s U %s  C %s  F %s" % (arm, v["U"], v["C"], v["F"]))
        for k in ("RQ50d_minus_RQ50c", "RQ50d_minus_R0", "RQ50d_minus_RLc"):
            print("   beside %s %s" % (k, json.dumps(bes[k])[:430]))
        print("== families I10RQ50d_minus_I10RQ50")
        for f, v in bes["families_RQ50d_minus_RQ50"].items():
            print("   %-34s D %+8.4f [%+.4f, %+.4f] %s %s" % (f, v["D"], v["interval"][0], v["interval"][1], v["per_seed"], v["verdict"]))
    else:
        print("== a control did not pass: no name, the differences of the arm RQ50d stay withheld")
    print("STRATEGIC: as note N1 of card ADD46 (T0 first tick: the banked read of the two models without the rule; along the rule-driven rollout NOT COMPUTED, n = 0, no instrument)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
