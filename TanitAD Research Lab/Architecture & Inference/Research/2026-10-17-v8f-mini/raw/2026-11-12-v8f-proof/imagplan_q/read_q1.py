#!/usr/bin/env python
"""read_q1.py -- source cand-cvq (note PREREG_V8F_PROOF_ADD40_N7) on one arm type at two checkpoints; the form of read_i9.py. Written BEFORE any run of this read existed.
Names are those of section 4 of the note as written; nothing is added.

Stage 1, the controls q1 to q6 (printed first; all must pass for a pair, else nothing is named and no outcome difference is printed):
  q1  compare lines of the pod sequences: cand-cvq with the constant 1e9 equals cand-cv bit for bit (every dump file)
  q2  position error over 6 s of (cand-cvq with the constant 0) - (cand-cv) above 0 at the pooled level, interval excluding 0
  q3  the head used is the banked one: the sha256 in the run's imag log equals the sha256 of the fit record, the record says check coverage in the band, the share is read from the record
  q4  U(cand-label) - U(no planner) below 0 at both checkpoints, interval excluding 0
  q5  counts 2,399 windows, 642 lead windows of 56 episodes; the active ticks of cand-cv and cand-cvq (ci, t0, j, active) are the same set at every checkpoint
  q6  the controls h1 to h6 of the headway read and q1 to q5 of the family read
Stage 2, the names (only if every control passed): QPLAN-CUTS-MORE / -CUTS-LESS / -EQUAL, QPLAN-REACHES-ORACLE, QPLAN-NO-COST / QPLAN-COSTS-<family> (against cand-cv).
Combine step (--combine FIRST.json SECOND.json): a name of the first pair is stated with the suffix -CONFIRMED only if the second pair reads the same name."""
import argparse
import io
import json
import os
import re

import numpy as np


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.exists(p) else None


def cmp_line(texts, a, b):
    pat = re.compile(r"cmp %s against %s \(.*?\): CMP files=(\d+) differing=(\d+)" % (re.escape(a), re.escape(b)))
    hit = None
    for t in texts:
        for m in pat.finditer(t):
            hit = (int(m.group(1)), int(m.group(2)))
    return hit


def cell(c):
    return {"per_base": [round(x, 4) for x in c["d_s"]], "pooled": round(c["D"], 4), "interval": [round(c["A_lo"], 4), round(c["A_hi"], 4)], "excludes_0": bool(c["A_excludes_0"]),
            "outside_spread": c.get("outside_spread")}


def log_of(pull, name):
    p = os.path.join(pull, name, "imag_log_%s.npz" % name)
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


def combine(a, b, out):
    """-CONFIRMED only if the second pair reads the same name (per name). Both pairs are always reported."""
    res = {"first": a, "second": b, "confirmed": {}, "note": "a name of the first pair is stated with -CONFIRMED only if the second pair reads the same name; else both are reported and that name is not confirmed"}
    if not (a.get("names") and b.get("names")):
        res["confirmed"] = None
        res["why"] = "a pair has no names (a control did not pass or is not evaluable)"
    else:
        for k in ("cut", "reaches_oracle", "price"):
            same = a["names"][k] == b["names"][k]
            res["confirmed"][k] = {"first": a["names"][k], "second": b["names"][k], "state": "CONFIRMED" if same else "NOT-CONFIRMED (the pairs differ)"}
    print(json.dumps(res["confirmed"], indent=1)[:1500])
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combine", nargs=2)
    ap.add_argument("--arm")                                   # LS or LSD
    ap.add_argument("--sfx", default="q")
    ap.add_argument("--bases", default="0,1")
    ap.add_argument("--pull", default="")                      # imagplan_q/pull (imag logs)
    ap.add_argument("--read-dir", default="")                  # imagplan_q/read_q1/<arm>
    ap.add_argument("--seq-outs", nargs="*", default=[])       # deploy_q_*.out of both bases
    ap.add_argument("--records", default="")                   # folder with <arm>_s<k>.qhead.json (the banked fit records)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.combine:
        return combine(jl(a.combine[0]), jl(a.combine[1]), a.out)
    A, S = a.arm, a.sfx
    bases = [int(x) for x in a.bases.split(",")]
    texts = [io.open(p, encoding="utf-8", errors="replace").read() for p in a.seq_outs if os.path.exists(p)]
    res = {"arm": A, "source": "cand-cvq", "note": "PREREG_V8F_PROOF_ADD40_N7 section 4 (names as written)", "controls": {}, "names": None}
    C = res["controls"]
    n = lambda t: "%s%s_%s" % (t, S, A)                               # noqa: E731  (IPcq + q -> IPcqq_LS: the run folders are <name>_s<k>)
    pk = lambda x, y: "%s_minus_%s" % (n(x), n(y))                    # noqa: E731
    hw = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json"))
    q3 = jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))

    def U(x, y):
        return hw["pairs"][pk(x, y)]["cells"]["U"] if hw and pk(x, y) in hw["pairs"] else None

    def E6(x, y):
        return q3["pairs"][pk(x, y)]["beside"]["ade6_m|all"] if q3 and pk(x, y) in q3["pairs"] else None
    # q1
    hits = [cmp_line([t], "IPcqid" + S, "IPc" + S) for t in texts]
    hits = [h for h in hits if h is not None]
    C["q1_identity_1e9"] = {"compares": hits, "PASS": bool(len(hits) >= len(bases) and all(h[1] == 0 and h[0] > 0 for h in hits))}
    # q2
    e = E6("IPcq0", "IPcq")
    C["q2_constant_0_rises"] = dict(cell(e), PASS=bool(e["D"] > 0 and e["A_excludes_0"])) if e else {"PASS": False, "why": "pair missing"}
    # q3 and q5 (active ticks) from the logs
    q3c, q5a = {}, {}
    for k in bases:
        lg, lc = log_of(a.pull, "%s_s%d" % (n("IPcq"), k)), log_of(a.pull, "%s_s%d" % (n("IPc"), k))
        rec = jl(os.path.join(a.records, "%s_s%d.qhead.json" % (A, k)))
        if lg is None or rec is None:
            q3c["s%d" % k] = {"PASS": False, "why": "no imag log or no fit record"}
            q5a["s%d" % k] = {"PASS": False, "why": "no imag log of cand-cvq"}
            continue
        sha = str(lg["qhead_sha256"].item()) if "qhead_sha256" in lg.files else ""
        shas = list((rec.get("head_file_sha256") or {}).values())
        share = (rec.get("coverage_check") or {}).get("share_q_above_label")
        q3c["s%d" % k] = {"sha_in_run": sha[:16], "sha_in_record": [x[:16] for x in shas], "check_share": share, "mode": str(lg["mode"].item()), "const_in_run": float(lg["qhead_const"]),
                          "PASS": bool(sha in shas and rec.get("h5_check_coverage_in_0.15_0.25") is True and share is not None and 0.15 <= share <= 0.25 and str(lg["mode"].item()) == "cand-cvq"
                                       and bool(np.isnan(float(lg["qhead_const"]))))}
        if lc is None:
            q5a["s%d" % k] = {"PASS": False, "why": "no imag log of cand-cv"}
        else:
            same = all(lg[x].shape == lc[x].shape and bool((lg[x] == lc[x]).all()) for x in ("ci", "t0", "j", "active"))
            q5a["s%d" % k] = {"n_active_cv": int(lc["active"].sum()), "n_active_cvq": int(lg["active"].sum()), "PASS": bool(same)}
    C["q3_head_is_the_banked_one"] = {"per_base": q3c, "PASS": bool(q3c and all(v["PASS"] for v in q3c.values()))}
    # q4
    u = U("IPl", "IPo")
    C["q4_label_cuts_unsafe"] = dict(cell(u), PASS=bool(all(d < 0 for d in u["d_s"]) and u["A_excludes_0"])) if u else {"PASS": False, "why": "pair missing"}
    # q5
    cnt = (hw or {}).get("counts", {})
    C["q5_counts_and_active_set"] = {"n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"), "n_episodes_with_lead": cnt.get("n_episodes_with_lead"), "active_sets": q5a,
                                     "PASS": bool(cnt.get("n_windows_total") == 2399 and cnt.get("n_lead_windows") == 642 and cnt.get("n_episodes_with_lead") == 56 and q5a and all(v["PASS"] for v in q5a.values()))}
    # q6
    C["q6_headway_and_family_controls"] = {"headway": bool(hw and hw["controls"].get("ALL_PASS")), "family": bool(q3 and q3["controls"].get("ALL_PASS")),
                                           "PASS": bool(hw and q3 and hw["controls"].get("ALL_PASS") and q3["controls"].get("ALL_PASS"))}
    allp = all(v.get("PASS") for v in C.values())
    res["controls_all_pass"] = bool(allp)
    print("== controls of cand-cvq on", A)
    for k, v in C.items():
        print("  ", k, "PASS" if v.get("PASS") else "FAIL", json.dumps(v)[:300])
    print("controls_all_pass", allp)
    if allp:
        N = {}
        uc, ul, uo = U("IPcq", "IPc"), U("IPcq", "IPl"), U("IPcq", "IPo")
        if all(d < 0 for d in uc["d_s"]) and uc["A_excludes_0"]:
            N["cut"] = "QPLAN-CUTS-MORE"
        elif all(d > 0 for d in uc["d_s"]) and uc["A_excludes_0"]:
            N["cut"] = "QPLAN-CUTS-LESS"
        else:
            N["cut"] = "QPLAN-EQUAL"
        N["reaches_oracle"] = "QPLAN-REACHES-ORACLE" if (all(d <= 0.010 for d in ul["d_s"]) and ul["A_lo"] <= 0) else "no"
        q = q3["pairs"][pk("IPcq", "IPc")]["Q3"]
        N["price"] = "QPLAN-" + str(q.get("name"))
        N["cost_families"] = q.get("cost_families")
        opt = lambda f, x, y: cell(f(x, y)) if f(x, y) else None      # noqa: E731
        B = {"U_cvq_minus_cv": cell(uc), "U_cvq_minus_label": cell(ul), "U_cvq_minus_off": opt(U, "IPcq", "IPo") if uo else None, "ade6_cvq_minus_cv_m": opt(E6, "IPcq", "IPc"),
             "ade6_cvq_minus_off_m": opt(E6, "IPcq", "IPo"), "ade6_cvq_minus_label_m": opt(E6, "IPcq", "IPl")}
        try:
            cells = hw["pairs"][pk("IPcq", "IPl")]["cells"]
            B["contact_cvq_minus_label"] = {k: cell(v) for k, v in cells.items() if k.startswith("C") and isinstance(v, dict) and "d_s" in v}
        except Exception as ex:                                       # noqa: BLE001
            B["contact_cvq_minus_label"] = "not available: %s" % ex
        res["names"], res["beside"] = N, B
        print("== names")
        print("  ", json.dumps(N))
        for k, v in B.items():
            print("  ", k, json.dumps(v)[:330])
        print("   STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument)")
    else:
        print("== names: NOT GIVEN (a control did not pass or is not evaluable yet); the outcome differences are not printed")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
