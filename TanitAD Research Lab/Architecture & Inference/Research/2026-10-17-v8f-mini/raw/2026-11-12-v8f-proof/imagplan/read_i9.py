#!/usr/bin/env python
"""read_i9.py -- source cand-imag (the full imagined rollout under the candidate's own actions) of card PREREG_V8F_PROOF_ADD40 on one arm at two bases.
Written by the lead from read_k8.py BEFORE any run of this read existed (note N6 section 4: the source is evaluable on an arm that meets control p5b at both bases).
The names are those of section 4 of the card as written; nothing is added.

Stage 1, the controls (printed first; all must pass):
  p1 first part, p1b                                 from the compare lines of the pod sequences (deploy_k_*i3.out)
  p3 (subset form of note N5, first 600 windows)     from the compare lines of pod_cmp_subset_i.sh
  p4b, p5b (bar 0.5, the arm responds), p7           from PLUMB3cv_<arm>_s<k>.json (plumb_read3.py with the log of the cand-cv run)
  p5r                                                from REPLAY_<arm>_s<k>.json (replay_read.py)
  p8                                                 the controls of the headway and of the family read
  p2   U(cand-label) - U(off) below 0 at both bases, interval excluding 0
  p6   position error over 6 s of (imagined gap shifted by -20 m) - (cand-imag) above 0, interval excluding 0
Stage 2, the names (printed only if every control passed): IMAG-PLAN-CUTS-UNSAFE / -NO-CUT, -BEATS- / -EQUALS- / -WORSE-THAN-EXTRAPOLATION, -DOMINATES, -REACHES-ORACLE,
-NO-COST / -COSTS-<family>, with the differences beside them.
"""
import argparse
import io
import json
import os
import re


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def cmp_line(texts, a, b):
    """last compare line 'cmp <a> against <b> (...)' in the given outputs -> (found, differing)"""
    pat = re.compile(r"cmp %s against %s \(.*?\): CMP files=(\d+) differing=(\d+)" % (re.escape(a), re.escape(b)))
    hit = None
    for t in texts:
        for m in pat.finditer(t):
            hit = (int(m.group(1)), int(m.group(2)))
    return hit


def cell(c):
    return {"per_base": [round(x, 4) for x in c["d_s"]], "pooled": round(c["D"], 4), "interval": [round(c["A_lo"], 4), round(c["A_hi"], 4)], "excludes_0": bool(c["A_excludes_0"]),
            "outside_spread": c.get("outside_spread")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)                      # RLA3
    ap.add_argument("--sfx", default="x")
    ap.add_argument("--dir", required=True)                      # imagplan/
    ap.add_argument("--read-dir", required=True)                 # imagplan/read_i9/<arm>
    ap.add_argument("--seq-outs", nargs="+", required=True)      # deploy_k_*i3.out and cmp_subset_i_*.out of both bases
    ap.add_argument("--bases", default="0,1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, S = a.arm, a.sfx
    texts = [io.open(p, encoding="utf-8", errors="replace").read() for p in a.seq_outs if os.path.exists(p)]
    res = {"arm": A, "source": "cand-imag", "card": "PREREG_V8F_PROOF_ADD40 section 4 (names as written); note N6 section 4", "controls": {}, "names": None}
    C = res["controls"]
    # ---- compares, per base (a compare line carries no base: each sequence output is one base, so the outputs are scanned per file)
    for nm, x, y in (("p1_first_part", "IPid1" + S, "IPo" + S), ("p1b", "IPid3" + S, "IPo" + S), ("p3_subset_600", "IPnm", "IPim")):
        hits = [cmp_line([t], x, y) for t in texts]
        hits = [h for h in hits if h is not None]
        C[nm] = {"compares": hits, "PASS": bool(len(hits) >= 2 and all(h[1] == 0 and h[0] > 0 for h in hits))}
    for k in [int(x) for x in a.bases.split(",")]:
        p3 = jl(os.path.join(a.dir, "read_plumb", "PLUMB3cv_%s_s%d.json" % (A, k)))
        rp = jl(os.path.join(a.dir, "read_replay", "REPLAY_%s_s%d.json" % (A, k)))
        p4b, p5b, p7 = (p3 or {}).get("p4b", {}), (p3 or {}).get("p5b", {}), (p3 or {}).get("p7", {})
        C["p4b_s%d" % k] = {"PASS": bool(p4b.get("PASS")), "corr_yaw": p4b.get("corr_yaw_all_steps"), "corr_acc_2_10": p4b.get("corr_acc_steps_2_10"),
                            "median_abs_diff_acc": p4b.get("median_abs_diff_acc")}
        C["p5b_s%d" % k] = {"PASS": bool(p5b.get("PASS") is True and p5b.get("responds") == "yes"), "slope_T2.0": p5b.get("slope_T2.0"), "ci_episodes": p5b.get("ci_episodes"),
                            "bar": "slope at least 0.5 (the arm responds)"}
        C["p7_s%d" % k] = {"windows": p7.get("windows"), "counts": p7.get("PASS_counts"), "active_equal_all_ticks": p7.get("active_equal_all_ticks"),
                           "PASS": bool(p7.get("PASS_counts") is True and p7.get("active_equal_all_ticks") is True)}
        if rp:
            m = rp["matched_tool_vs_reader"]
            d = abs(m["slope_tool_T2.0"] - m["slope_reader_T2.0"])
            C["p5r_s%d" % k] = {"tool": round(m["slope_tool_T2.0"], 3), "reader": round(m["slope_reader_T2.0"], 3), "n_matched": m["n_matched"], "PASS": bool(d <= 0.10)}
        else:
            C["p5r_s%d" % k] = {"PASS": False, "why": "no replay read for this base"}
    hw = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json"))
    q3 = jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    C["p8"] = {"headway_controls": bool(hw and hw["controls"].get("ALL_PASS")), "family_controls": bool(q3 and q3["controls"].get("ALL_PASS")),
               "PASS": bool(hw and q3 and hw["controls"].get("ALL_PASS") and q3["controls"].get("ALL_PASS"))}
    n = lambda t: "%s%s_%s" % (t, S, A)                               # noqa: E731
    pk = lambda x, y: "%s_minus_%s" % (n(x), n(y))                    # noqa: E731

    def U(x, y):
        return hw["pairs"][pk(x, y)]["cells"]["U"] if hw and pk(x, y) in hw["pairs"] else None

    def E6(x, y):
        return q3["pairs"][pk(x, y)]["beside"]["ade6_m|all"] if q3 and pk(x, y) in q3["pairs"] else None
    u = U("IPl", "IPo")
    C["p2"] = dict(cell(u), PASS=bool(all(d < 0 for d in u["d_s"]) and u["A_excludes_0"])) if u else {"PASS": False, "why": "pair missing"}
    e = E6("IPs", "IPi")
    C["p6"] = dict(cell(e), PASS=bool(e["D"] > 0 and e["A_excludes_0"])) if e else {"PASS": False, "why": "pair missing"}
    allp = all(v.get("PASS") for v in C.values())
    res["controls_all_pass"] = bool(allp)
    print("== controls of cand-imag on", A)
    for k, v in C.items():
        print("  ", k, json.dumps(v)[:330])
    print("controls_all_pass", allp)
    if allp:
        N = {}
        uo, uc, ul = U("IPi", "IPo"), U("IPi", "IPc"), U("IPi", "IPl")
        N["cut"] = "IMAG-PLAN-CUTS-UNSAFE" if (all(d < 0 for d in uo["d_s"]) and uo["A_excludes_0"]) else "IMAG-PLAN-NO-CUT"
        if all(d < 0 for d in uc["d_s"]) and uc["A_excludes_0"] and uc["D"] <= -0.010:
            N["extrapolation"] = "IMAG-PLAN-BEATS-EXTRAPOLATION"
        elif all(d > 0 for d in uc["d_s"]) and uc["A_excludes_0"]:
            N["extrapolation"] = "IMAG-PLAN-WORSE-THAN-EXTRAPOLATION"
        else:
            N["extrapolation"] = "IMAG-PLAN-EQUALS-EXTRAPOLATION"
        ec = E6("IPi", "IPc")
        N["dominates"] = bool(N["extrapolation"].endswith("BEATS-EXTRAPOLATION") and ec["A_lo"] <= 0)
        N["reaches_oracle"] = bool(all(d <= 0.010 for d in ul["d_s"]) and ul["A_lo"] <= 0)
        q = q3["pairs"][pk("IPi", "IPo")]["Q3"]
        N["price"] = "IMAG-PLAN-" + str(q.get("name"))
        N["cost_families"] = q.get("cost_families")
        opt = lambda f, x, y: cell(f(x, y)) if f(x, y) else None      # noqa: E731
        B = {"U_imag_minus_off": cell(uo), "U_imag_minus_cv": cell(uc), "U_imag_minus_label": cell(ul), "U_cv_minus_off": opt(U, "IPc", "IPo"),
             "U_level_minus_imag": opt(U, "IPk", "IPi"), "U_level_minus_off": opt(U, "IPk", "IPo"), "U_level_minus_cv": opt(U, "IPk", "IPc"),
             "ade6_imag_minus_off_m": cell(E6("IPi", "IPo")), "ade6_imag_minus_cv_m": cell(ec), "ade6_imag_minus_label_m": opt(E6, "IPi", "IPl"),
             "U_levels": {m: (hw["levels"].get(n(m), {}).get("cells", {}).get("U", {}).get("per_seed")) for m in ("IPo", "IPc", "IPi", "IPk", "IPl", "IPs")}}
        res["names"], res["beside"] = N, B
        print("== names")
        print("  ", json.dumps(N))
        for k, v in B.items():
            print("  ", k, json.dumps(v)[:330])
    else:
        print("== names: NOT GIVEN (a control did not pass or is not evaluable yet); the outcome differences are not printed")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
