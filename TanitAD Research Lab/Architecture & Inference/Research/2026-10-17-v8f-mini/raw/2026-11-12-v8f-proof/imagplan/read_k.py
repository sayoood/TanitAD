#!/usr/bin/env python
"""read_k.py -- source cand-imagkin of card PREREG_V8F_PROOF_ADD40 (notes N4 section 3 and N5 section 3) on one arm at both bases. Written by the lead.

Stage 1, the controls (printed first; all must pass):
  p1 first part, p1b, p3 (subset form of note N5)   from the compare lines of the pod sequences (deploy_k_*.out)
  k1, k2, k3, p7                                     from PLUMB4_<arm>_s<k>.json (plumb_read4.py)
  p5r                                                from REPLAY_<arm>_s<k>.json (replay_read.py)
  p8                                                 the controls of the headway and of the family read
  p2   U(cand-label) - U(off) below 0 at both bases, interval excluding 0
  p6   position error over 6 s of (level shifted by -20 m) - (cand-imagkin) above 0, interval excluding 0
Stage 2, the names (printed only if every control passed): IMAG-LEVEL-CUTS-UNSAFE / -NO-CUT, -BEATS- / -EQUALS- / -WORSE-THAN-EXTRAPOLATION, -DOMINATES, -REACHES-ORACLE,
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
    ap.add_argument("--arm", required=True)                      # RLFT or RLAm
    ap.add_argument("--sfx", default="x")
    ap.add_argument("--dir", required=True)                      # imagplan/
    ap.add_argument("--read-dir", required=True)                 # imagplan/read_k/<arm>
    ap.add_argument("--seq-outs", nargs="+", required=True)      # deploy_k_*.out of both bases
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, S = a.arm, a.sfx
    texts = [io.open(p, encoding="utf-8", errors="replace").read() for p in a.seq_outs if os.path.exists(p)]
    res = {"arm": A, "controls": {}, "names": None}
    C = res["controls"]
    # ---- compares, per base (a compare line carries no base: each sequence output is one base, so the outputs are scanned per file)
    for nm, x, y in (("p1_first_part", "IPkid1" + S, "IPo" + S), ("p1b", "IPkid3" + S, "IPo" + S), ("p3_subset_600", "IPknm", "IPkm")):
        hits = [cmp_line([t], x, y) for t in texts]
        hits = [h for h in hits if h is not None]
        C[nm] = {"compares": hits, "PASS": bool(len(hits) >= 2 and all(h[1] == 0 and h[0] > 0 for h in hits))}
    for k in (0, 1):
        p4 = jl(os.path.join(a.dir, "read_plumb", "PLUMB4_%s_s%d.json" % (A, k)))
        rp = jl(os.path.join(a.dir, "read_replay", "REPLAY_%s_s%d.json" % (A, k)))
        C["k1_s%d" % k] = {"PASS": bool(p4 and p4.get("k1", {}).get("PASS")), "max_abs_m": (p4 or {}).get("k1", {}).get("max_abs_m")}
        C["k2_s%d" % k] = {"PASS": bool(p4 and p4.get("k2", {}).get("PASS")), "max_abs_m": (p4 or {}).get("k2", {}).get("max_abs_m")}
        C["k3_s%d" % k] = dict((p4 or {}).get("k3", {}), PASS=bool(p4 and p4.get("k3", {}).get("PASS")))
        C["p7_s%d" % k] = {"windows": (p4 or {}).get("p7_windows"), "active_same_set": (p4 or {}).get("p7_active_same_set_all_ticks"),
                           "PASS": bool(p4 and p4.get("p7_windows") == 2399 and p4.get("p7_active_same_set_all_ticks"))}
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
    e = E6("IPks", "IPk")
    C["p6"] = dict(cell(e), PASS=bool(e["D"] > 0 and e["A_excludes_0"])) if e else {"PASS": False, "why": "pair missing"}
    allp = all(v.get("PASS") for v in C.values())
    res["controls_all_pass"] = bool(allp)
    print("== controls of cand-imagkin on", A)
    for k, v in C.items():
        print("  ", k, json.dumps(v)[:330])
    print("controls_all_pass", allp)
    if allp:
        N = {}
        uo, uc, ul = U("IPk", "IPo"), U("IPk", "IPc"), U("IPk", "IPl")
        N["cut"] = "IMAG-LEVEL-CUTS-UNSAFE" if (all(d < 0 for d in uo["d_s"]) and uo["A_excludes_0"]) else "IMAG-LEVEL-NO-CUT"
        if all(d < 0 for d in uc["d_s"]) and uc["A_excludes_0"] and uc["D"] <= -0.010:
            N["extrapolation"] = "IMAG-LEVEL-BEATS-EXTRAPOLATION"
        elif all(d > 0 for d in uc["d_s"]) and uc["A_excludes_0"]:
            N["extrapolation"] = "IMAG-LEVEL-WORSE-THAN-EXTRAPOLATION"
        else:
            N["extrapolation"] = "IMAG-LEVEL-EQUALS-EXTRAPOLATION"
        ec = E6("IPk", "IPc")
        N["dominates"] = bool(N["extrapolation"].endswith("BEATS-EXTRAPOLATION") and ec["A_lo"] <= 0)
        N["reaches_oracle"] = bool(all(d <= 0.010 for d in ul["d_s"]) and ul["A_lo"] <= 0)
        q = q3["pairs"][pk("IPk", "IPo")]["Q3"]
        N["price"] = "IMAG-LEVEL-" + str(q.get("name"))
        N["cost_families"] = q.get("cost_families")
        B = {"U_k_minus_off": cell(uo), "U_k_minus_cv": cell(uc), "U_k_minus_label": cell(ul), "U_cv_minus_off": cell(U("IPc", "IPo")) if U("IPc", "IPo") else None,
             "U_k_minus_imag": cell(U("IPk", "IPi")) if U("IPk", "IPi") else None, "U_imag_minus_off": cell(U("IPi", "IPo")) if U("IPi", "IPo") else None,
             "ade6_k_minus_off_m": cell(E6("IPk", "IPo")), "ade6_k_minus_cv_m": cell(ec), "ade6_k_minus_label_m": cell(E6("IPk", "IPl")) if E6("IPk", "IPl") else None,
             "U_levels": {m: (hw["levels"].get(n(m), {}).get("cells", {}).get("U", {}).get("per_seed")) for m in ("IPo", "IPc", "IPk", "IPi", "IPl", "IPks")}}
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
