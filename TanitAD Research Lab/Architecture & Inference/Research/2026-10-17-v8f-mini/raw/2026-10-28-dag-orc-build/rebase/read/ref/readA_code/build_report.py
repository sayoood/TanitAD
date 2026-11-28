"""Assemble Read A (outcome per PREREG section 5, applied verbatim) and the markdown tables of READ_DAG_REREAD.md from the raw JSONs of this package."""
import json
import sys

sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagreread/code")
PK = "E:/Projects/TanitAD-artifacts/dagreread"


def J(p):
    return json.load(open(f"{PK}/raw/{p}", encoding="utf-8"))


TWO_F = 0.2415
SPOS_KEY = "S-POS vs CTRV (ticks>=1, within this run)"
dagread = json.load(open("E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json", encoding="utf-8"))
means = J("tacdk_means.json")

# (family, label, source, key, direction); 'lower' = error-like (negative delta = better); 'higher' = agreement (positive delta = better)
SPEC = [("PRIMARY", "ADE6", "c", "ADE6", "lower"),
        ("LONGITUDINAL", "speed MAE", "c", "speed MAE", "lower"), ("LONGITUDINAL", "along-track MAE", "c", "along-track MAE", "lower"),
        ("LONGITUDINAL", "accel MAE", "c", "accel MAE", "lower"),
        ("LONGITUDINAL", "DK-VIOL", "t", "DK-VIOL", "lower"), ("LONGITUDINAL", "DK-HW", "t", "DK-HW", "lower"), ("LONGITUDINAL", "DK-TG", "t", "DK-TG", "lower"),
        ("LATERAL", "heading MAE", "c", "heading MAE", "lower"), ("LATERAL", "yaw-rate MAE", "c", "yaw-rate MAE", "lower"),
        ("LATERAL", "curvature MAE", "c", "curvature MAE", "lower"),
        ("LATERAL", "cross-track MAE", "c", "cross-track MAE", "lower"), ("LATERAL", "L1 lateral-accel MAE", "c", "L1 lateral-accel MAE", "lower"),
        ("LATERAL", "L2 curvature MAE (GT>=5.2)", "c", "L2 curvature MAE (GT>=5.2)", "lower"),
        ("TACTICAL", "TAC-LAT", "t", "TAC-LAT", "higher"), ("TACTICAL", "TAC-LON", "t", "TAC-LON", "higher"),
        ("STRATEGIC", "S-POS@30", "s", "S-POS@30", "lower"), ("STRATEGIC", "S-POS@50", "s", "S-POS@50", "lower"), ("STRATEGIC", "S-POS@80", "s", "S-POS@80", "lower")]


def judge(r, direction):
    sep = bool(r.get("separated"))
    d = r["delta"]
    better = sep and ((d < 0) if direction == "lower" else (d > 0))
    worse = sep and ((d > 0) if direction == "lower" else (d < 0))
    return bool(better), bool(worse)


def tacdk_mean(arm, s, key, ref="P10"):
    m = means[f"s{s}"]
    if key in ("TAC-LAT", "TAC-LON"):
        return m[arm][key], m[ref][key]
    if key == "DK-VIOL":
        return m[arm]["DK-VIOL_share_of_at_risk"], m[ref]["DK-VIOL_share_of_at_risk"]
    pm = m[f"pairmask_{arm}_vs_{ref}"][key]
    return pm["mean_a_pairmask"], pm["mean_b_pairmask"]


def collect(arm_key, comps_json, t1kf_json, t1kf_cfg, t1kf_arm, spos_arm):
    out = {}
    for s in (0, 1):
        rows = {}
        cj = comps_json["seeds"][f"s{s}"]["rows_arm_minus_P10"]
        tj = t1kf_json["configs"][t1kf_cfg]["seeds"][f"s{s}"][f"rows_{t1kf_arm}_minus_P10"] if t1kf_json else None
        for fam, lab, src, key, dr in SPEC:
            if src == "c":
                r = cj[key]
                v = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n": f"{r['n_windows']}/{r['n_episodes']}",
                     "mean_arm": r["mean_a"], "mean_ref": r["mean_b"]}
            elif src == "t":
                r = tj[key]
                ma, mb = tacdk_mean(arm_key, s, key) if (t1kf_cfg == "decisive") else (None, None)
                v = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n": f"{r['n_windows']}/{r.get('n_episodes', '-')}",
                     "mean_arm": ma, "mean_ref": mb}
            else:
                if spos_arm is None:
                    v = None
                else:
                    r = dagread["beside"]["per_seed"][f"s{s}"]["per_run_t1heads"][spos_arm][SPOS_KEY][key]
                    v = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"],
                         "n": "ticks>=1, unpaired vs CTRV (DAG_READ.json)", "mean_arm": None, "mean_ref": None}
            if v is None:
                rows[lab] = {"family": fam, "status": "NOT COMPUTED"}
                continue
            b, w = judge(v, dr)
            v.update({"family": fam, "direction": dr, "separated_better": b, "separated_worse": w})
            rows[lab] = v
        m1 = cj["M1 crawl share"]
        mb_, mw_ = judge(m1, "lower")
        rows["M1 crawl share"] = {"family": "crawl", "delta": m1["delta"], "lo": m1["lo"], "hi": m1["hi"], "separated": m1["separated"],
                                  "n": f"{m1['n_windows']}/{m1['n_episodes']}", "mean_arm": m1["mean_a"], "mean_ref": m1["mean_b"],
                                  "separated_better": mb_, "separated_worse": mw_, "direction": "lower",
                                  "pooled_M1_arm": m1["pooled_M1_a"], "pooled_M1_P10": m1["pooled_M1_b"], "M2": cj["_M2"]["M2"]}
        out[s] = rows
    return out


def verdict(rows):
    res = {}
    for s in (0, 1):
        R = rows[s]
        a = R["ADE6"]
        d1 = bool(a["separated_better"] and a["delta"] <= -TWO_F)
        worse = [lab for lab, r in R.items() if r.get("separated_worse") and r.get("family") != "crawl"]
        notc = [lab for lab, r in R.items() if r.get("status") == "NOT COMPUTED"]
        d2 = bool(not worse and not notc)
        fams = {fam: [lab for lab, r in R.items() if r.get("family") == fam and r.get("separated_better")]
                for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}
        d3 = all(len(v) >= 1 for v in fams.values())
        res[s] = {"D1": d1, "D2": d2, "D3": d3, "separated_worse_rows": worse, "not_computed": notc, "separated_better_rows_by_family": fams,
                  "M1_separated_worse": bool(R["M1 crawl share"]["separated_worse"]), "L2_separated_worse": bool(R["L2 curvature MAE (GT>=5.2)"]["separated_worse"])}
    return res


def outcome(res):
    d1 = all(res[s]["D1"] for s in (0, 1))
    d2 = all(res[s]["D2"] for s in (0, 1))
    d3 = all(res[s]["D3"] for s in (0, 1))
    m1w = any(res[s]["M1_separated_worse"] for s in (0, 1))
    crawl_fail = any(res[s]["L2_separated_worse"] or res[s]["M1_separated_worse"] for s in (0, 1))
    if not d1:
        o = "NO MARGIN"
    elif d2 and d3 and not m1w:
        o = "DOMINANT"
    elif d2 and d3 and m1w:
        o = "DOMINANT, CRAWL OPEN"
    else:
        o = "MARGIN, NOT DOMINANT"
    return {"outcome": o, "D1_both": d1, "D2_both": d2, "D3_both": d3, "crawl_rows_fail": crawl_fail, "M1_separated_worse_at_ge1_seed": m1w}


A_dec = collect("DAG", J("readA_DAG_causal.json"), J("t1kf_DAG/T1KF_READ_A1_DAG.json"), "decisive", "DAG", "DAG")
V = verdict(A_dec)
O = outcome(V)
A_0 = collect("DAG0", J("readA_DAG0_causal.json"), J("t1kf_DAG0/T1KF_READ_A1_DAG0.json"), "decisive", "DAG0", "DAG0")
V0 = verdict(A_0)
A_log = collect("DAG", J("readA_DAG_logged.json"), J("t1kf_DAG/T1KF_READ_A1_DAG.json"), "logged", "DAG", "DAG")
A_str = collect("DAG", J("readA_DAGSTRICT_frozen_causal.json"), J("t1kf_DAG/T1KF_READ_A1_DAG.json"), "strict", "DAG", None)
Vs = verdict(A_str)
hacc = {f"s{s}": J("readA_DAG_causal.json")["seeds"][f"s{s}"]["beside_arm_minus_HACC"] for s in (0, 1)}
out = {"READ_A": {"outcome": O, "per_seed_verdict": {f"s{s}": V[s] for s in (0, 1)}, "rows": {f"s{s}": A_dec[s] for s in (0, 1)}},
       "beside_DAG0_minus_P10": {"verdict": {f"s{s}": V0[s] for s in (0, 1)}, "outcome_if_applied": outcome(V0), "rows": {f"s{s}": A_0[s] for s in (0, 1)}},
       "beside_STRICT_DAG_minus_P10": {"verdict": {f"s{s}": Vs[s] for s in (0, 1)}, "rows": {f"s{s}": A_str[s] for s in (0, 1)},
                                       "note": "S-POS rows NOT COMPUTED for STRICT (the DAGGER reader / T1-HEADS reducer was not run on the STRICT ticks)"},
       "beside_logged_limit_DAG_minus_P10": {f"s{s}": A_log[s] for s in (0, 1)},
       "beside_DAG_minus_HACC": hacc}
json.dump(out, open(f"{PK}/raw/READ_A.json", "w"), indent=1)


def table(rows, s):
    L = ["| family | row | arm - P10 [lo, hi] | sep | verdict | arm mean | P10 mean | n (win/ep) |", "|---|---|---|---|---|---|---|---|"]
    for lab, r in rows[s].items():
        if r.get("status") == "NOT COMPUTED":
            L.append(f"| {r['family']} | {lab} | NOT COMPUTED | | | | | |")
            continue
        vd = "BETTER" if r["separated_better"] else ("**WORSE**" if r["separated_worse"] else "ns")
        ma = "-" if r["mean_arm"] is None else f"{r['mean_arm']:.4f}"
        mb = "-" if r["mean_ref"] is None else f"{r['mean_ref']:.4f}"
        L.append(f"| {r['family']} | {lab} | {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {'yes' if r['separated'] else 'no'} | {vd} | {ma} | {mb} | {r['n']} |")
    return "\n".join(L)


md = [f"#### Read A, seed {s}: DAG - P10, T1-kin decisive (m=10, causal bucket)\n\n" + table(A_dec, s) + "\n" for s in (0, 1)]
open(f"{PK}/raw/READ_A_tables.md", "w", encoding="utf-8", newline="\n").write("\n".join(md))
md2 = []
for nm, A in (("DAG-0 - P10 (beside)", A_0), ("DAG - P10 STRICT, scene frozen + limit causal (beside)", A_str), ("DAG - P10 limit logged (beside)", A_log)):
    for s in (0, 1):
        md2.append(f"#### {nm}, seed {s}\n\n" + table(A, s) + "\n")
open(f"{PK}/raw/READ_A_beside_tables.md", "w", encoding="utf-8", newline="\n").write("\n".join(md2))
print(json.dumps(O))
for s in (0, 1):
    print(f"s{s}", {k: V[s][k] for k in ("D1", "D2", "D3", "separated_worse_rows", "M1_separated_worse", "L2_separated_worse")})
    print("   better by family:", V[s]["separated_better_rows_by_family"])
print("DAG0 beside outcome if applied:", outcome(V0), {s: V0[s]["separated_worse_rows"] for s in (0, 1)})
print("STRICT beside worse rows:", {s: Vs[s]["separated_worse_rows"] for s in (0, 1)}, "not computed", Vs[0]["not_computed"])
print("DAG - HACC:", {s: (hacc[s]["same_windows_and_gt"], hacc[s]["ADE6"]["delta"], hacc[s]["ADE6"]["lo"], hacc[s]["ADE6"]["hi"], hacc[s]["L2"]["delta"], hacc[s]["L2"]["lo"], hacc[s]["L2"]["hi"]) for s in hacc})
