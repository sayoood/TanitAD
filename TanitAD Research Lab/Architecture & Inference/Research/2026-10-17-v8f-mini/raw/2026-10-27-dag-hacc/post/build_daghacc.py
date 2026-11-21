"""Assemble PREREG_V8F_DAG_HACC's read from the raw JSONs of this package (no hand-typed number): Read A rows (D1 / D2 / D3, crawl rows) for DAG-HACC - P10, KEEP (DAG-HACC - DAG, ADE6),
the controls, and the outcome per section 4 (REFUSED / BLOCKED first, then KEEP, then DOMINANT, LONGITUDINAL FIXED, NOT FIXED takes the rest), applied verbatim.
Writes raw/READ_DAG_HACC.json and READ_DAG_HACC.md."""
import json
import os
import sys

PK = "E:/Projects/TanitAD-artifacts/daghacc"
DR = "E:/Projects/TanitAD-artifacts/dagreread"
J = lambda p: json.load(open(p, encoding="utf-8"))  # noqa: E731
TWO_F, F, KEEP_BAR = 0.2415, 0.1207, 0.0604
SPOS_KEY = "S-POS vs CTRV (ticks>=1, within this run)"
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
# the comps-based readers' label (readA.ROWS_A) for each SPEC 'c' key
CKEY = {"ADE6": "ADE6", "speed MAE": "speed MAE", "along-track MAE": "along-track MAE", "accel MAE": "accel MAE", "heading MAE": "heading MAE", "yaw-rate MAE": "yaw-rate MAE",
        "curvature MAE": "curvature MAE", "cross-track MAE": "cross-track MAE", "L1 lateral-accel MAE": "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)": "L2 curvature MAE (GT>=5.2)"}

rA = J(f"{PK}/raw/readA_DAGHACC.json")
rT = J(f"{PK}/raw/t1kf_DAGHACC/T1KF_READ_A1_DAGHACC.json")
rS = J(f"{PK}/raw/spos_daghacc.json")
rK = J(f"{PK}/raw/tacdk_daghacc.json")
CT = J(f"{PK}/raw/CONTROLS.json")
dagA = J(f"{DR}/raw/READ_A.json")["READ_A"]["rows"]                      # plain DAG - P10 (the banked reread rows), for the side-by-side column
dagread = J("E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json")


def judge(r, direction):
    sep = bool(r.get("separated"))
    d = r["delta"]
    return bool(sep and ((d < 0) if direction == "lower" else (d > 0))), bool(sep and ((d > 0) if direction == "lower" else (d < 0)))


def mean_pair(key, s, cfg_rows=None):
    return None, None


def collect(cfg, tcfg, spos_label, s, with_means=False):
    rows = {}
    cj = rA["cfgs"][cfg][f"s{s}"]["rows_arm_minus_P10"]
    tj = rT["configs"][tcfg]["seeds"][f"s{s}"]["rows_DAGHACC_minus_P10"] if rT["configs"].get(tcfg) else None
    sp = rS["runs"].get(spos_label)
    for fam, lab, src, key, dr in SPEC:
        if src == "c":
            r = cj[CKEY[key]]
            v = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n": f"{r['n_windows']}/{r['n_episodes']}", "mean_arm": r["mean_a"], "mean_ref": r["mean_b"]}
        elif src == "t":
            if tj is None or tj[key].get("status") != "OK":
                rows[lab] = {"family": fam, "status": "NOT COMPUTED", "reason": (tj[key].get("reason") or tj[key].get("status")) if tj else "t1kf config absent"}
                continue
            r = tj[key]
            mk = {"TAC-LAT": "TAC-LAT", "TAC-LON": "TAC-LON", "DK-VIOL": "DK-VIOL_share_of_at_risk", "DK-HW": "DK-HW_mean_abs_err_m_own_finite", "DK-TG": "DK-TG_mean_abs_err_s_own_finite"}[key]
            ma = rK["seeds"][f"s{s}"]["means"]["DAGHACC"][mk] if with_means else None
            mb = rK["seeds"][f"s{s}"]["means"]["P10"][mk] if with_means else None
            v = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n": f"{r.get('n_windows')}/{r.get('n_episodes', '-')}", "mean_arm": ma, "mean_ref": mb}
        else:
            if sp is None or "S-POS" not in sp:
                rows[lab] = {"family": fam, "status": "NOT COMPUTED", "reason": (sp or {}).get("reason", "no S-POS block for this run")}
                continue
            r = sp["S-POS"][key]
            v = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n": "ticks>=1, unpaired vs CTRV", "mean_arm": None, "mean_ref": None}
        b, w = judge(v, dr)
        v.update({"family": fam, "direction": dr, "separated_better": b, "separated_worse": w})
        rows[lab] = v
    m1 = cj["M1 crawl share"]
    mb_, mw_ = judge(m1, "lower")
    rows["M1 crawl share"] = {"family": "crawl", "delta": m1["delta"], "lo": m1["lo"], "hi": m1["hi"], "separated": m1["separated"], "n": f"{m1['n_windows']}/{m1['n_episodes']}", "mean_arm": m1["mean_a"],
                              "mean_ref": m1["mean_b"], "separated_better": mb_, "separated_worse": mw_, "direction": "lower", "pooled_M1_arm": m1["pooled_M1_a"], "pooled_M1_P10": m1["pooled_M1_b"],
                              "M2": rA["cfgs"][cfg][f"s{s}"]["rows_arm_minus_P10"]["_M2"]["M2"]}
    return rows


def verdict(rows):
    R = rows
    a = R["ADE6"]
    d1 = bool(a["separated_better"] and a["delta"] <= -TWO_F)
    worse = [lab for lab, r in R.items() if r.get("separated_worse") and r.get("family") != "crawl"]
    notc = [lab for lab, r in R.items() if r.get("status") == "NOT COMPUTED"]
    d2 = bool(not worse and not notc)
    fams = {fam: [lab for lab, r in R.items() if r.get("family") == fam and r.get("separated_better")] for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}
    d3 = all(len(v) >= 1 for v in fams.values())
    lon = [lab for lab in worse if R[lab]["family"] == "LONGITUDINAL"]
    return {"D1": d1, "D1_multiple_of_F": (-a["delta"] / F), "D2": d2, "D3": d3, "separated_worse_rows": worse, "not_computed": notc, "separated_better_rows_by_family": fams,
            "LONGITUDINAL_separated_worse": lon, "M1_separated_worse": bool(R["M1 crawl share"]["separated_worse"]), "L2_separated_worse": bool(R["L2 curvature MAE (GT>=5.2)"]["separated_worse"])}


def outcome(V, keep, controls_ok):
    if not controls_ok:
        return {"outcome": "REFUSED", "why": "a control failed", "KEEP_both": None}
    keep_both = all(keep[s]["holds"] for s in (0, 1))
    d1 = all(V[s]["D1"] for s in (0, 1))
    d2 = all(V[s]["D2"] for s in (0, 1))
    d3 = all(V[s]["D3"] for s in (0, 1))
    m1w = any(V[s]["M1_separated_worse"] for s in (0, 1))
    lon_worse = any(V[s]["LONGITUDINAL_separated_worse"] for s in (0, 1))
    other_worse = any([r for r in V[s]["separated_worse_rows"] if r not in V[s]["LONGITUDINAL_separated_worse"]] or V[s]["M1_separated_worse"] for s in (0, 1))
    if not keep_both:
        o, note = "COSTS POSITION", "KEEP fails at >= 1 seed"
    elif d1 and d2 and d3 and not m1w:
        o, note = "DOMINANT", ""
    elif d1 and d2 and d3 and m1w:
        o, note = "DOMINANT, CRAWL OPEN", ""
    elif not lon_worse and other_worse:
        o, note = "LONGITUDINAL FIXED", ""
    elif lon_worse:
        o, note = "NOT FIXED", ""
    else:
        o, note = "NOT FIXED", "RESIDUAL CASE: no row separated-worse anywhere, yet D1 or D3 fails; placed by the table's 'NOT FIXED takes the rest' rule"
    return {"outcome": o, "note": note, "KEEP_both": keep_both, "D1_both": d1, "D2_both": d2, "D3_both": d3, "M1_separated_worse_at_ge1_seed": m1w, "LONGITUDINAL_row_separated_worse_at_ge1_seed": lon_worse}


NEXT = {"DOMINANT": "the dominance claim at T1-kin over the kinematic floor, on all four families, at both seeds, is registered for DAG-HACC. Signal the PI. Evidence videos are re-rendered on DAG-HACC. FLAT-DAG stays the attribution read",
        "DOMINANT, CRAWL OPEN": "as DOMINANT, and DAgger iteration 2 for the crawl (box 2, PI go), evaluated under the same post-map",
        "LONGITUDINAL FIXED": "the post-map is the longitudinal lever and DAG-HACC is the base's deployed form. The remaining rows go to DAgger iteration 2, read under the post-map",
        "NOT FIXED": "the post-map is not enough on DAG's own motion. Next: OACC's acceleration-matching term added to DAG's fine-tune recipe (box 2, PI go), under its own PREREG",
        "COSTS POSITION": "not adopted; DAG stays plain. Next: as NOT FIXED",
        "REFUSED": "fix and re-run", "BLOCKED": "the missing input is named; it needs a box-2 eval of DAG on W_fit (PI go)"}

A_dec = {s: collect("causal", "decisive", f"DAGHACC_s{s}_decisive", s, with_means=True) for s in (0, 1)}
V = {s: verdict(A_dec[s]) for s in (0, 1)}
keep = {}
for s in (0, 1):
    r = rA["cfgs"]["causal"][f"s{s}"]["rows_arm_minus_DAG"]["ADE6"]
    keep[s] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "mean_arm": r["mean_a"], "mean_DAG": r["mean_b"], "n": f"{r['n_windows']}/{r['n_episodes']}",
               "bar_hi": KEEP_BAR, "holds": bool(r["hi"] <= KEEP_BAR), "holds_vs_exact_F_over_2_0.06035": bool(r["hi"] <= F / 2)}
ctl_ok = bool(CT["SUMMARY"]["all_registered_controls_pass"])
O = outcome(V, keep, ctl_ok)
O["committed_next_verbatim"] = NEXT[O["outcome"]]
A_str = {s: collect("frozen_causal", "strict", f"DAGHACCSTRICT_s{s}_beside", s) for s in (0, 1)}
A_log = {s: collect("logged", "logged", "none", s) for s in (0, 1)}
Vs, Vl = {s: verdict(A_str[s]) for s in (0, 1)}, {s: verdict(A_log[s]) for s in (0, 1)}
beside_dag = {f"s{s}": {lab: {k: v[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_a", "mean_b") if k in v}
                       for lab, v in rA["cfgs"]["causal"][f"s{s}"]["rows_arm_minus_DAG"].items() if not lab.startswith("_")} for s in (0, 1)}
beside_dag_t = {f"s{s}": {k: {x: v.get(x) for x in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "status")} for k, v in rK["seeds"][f"s{s}"]["rows_DAGHACC_minus_DAG"].items()} for s in (0, 1)}
out = {"prereg": "Project Steering/PREREG_V8F_DAG_HACC.md @ 2a416ca7", "claim_row": "H-V8F-DAG-HACC-1", "controls_all_registered_pass": ctl_ok, "outcome": O,
       "per_seed_verdict": {f"s{s}": V[s] for s in (0, 1)}, "KEEP": {f"s{s}": keep[s] for s in (0, 1)}, "rows_DAGHACC_minus_P10_decisive": {f"s{s}": A_dec[s] for s in (0, 1)},
       "beside": {"STRICT_verdict": {f"s{s}": Vs[s] for s in (0, 1)}, "STRICT_rows": {f"s{s}": A_str[s] for s in (0, 1)}, "logged_limit_verdict": {f"s{s}": Vl[s] for s in (0, 1)},
                  "DAGHACC_minus_DAG_comps_rows": beside_dag, "DAGHACC_minus_DAG_tacdk_rows": beside_dag_t}}
json.dump(out, open(f"{PK}/raw/READ_DAG_HACC.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def fmt(r):
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def table(rows, s, dagrows=None):
    L = ["| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |", "|---|---|---|---|---|---|---|---|---|"]
    for lab, r in rows.items():
        if r.get("status") == "NOT COMPUTED":
            L.append(f"| {r['family']} | {lab} | NOT COMPUTED ({r.get('reason')}) | | | | | | |")
            continue
        vd = "BETTER" if r["separated_better"] else ("**WORSE**" if r["separated_worse"] else "ns")
        ma = "-" if r["mean_arm"] is None else f"{r['mean_arm']:.4f}"
        mb = "-" if r["mean_ref"] is None else f"{r['mean_ref']:.4f}"
        dg = "-"
        if dagrows is not None and lab in dagrows[f"s{s}"]:
            q = dagrows[f"s{s}"][lab]
            dg = f"{q['delta']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}]" + (" WORSE" if q.get("separated_worse") else (" better" if q.get("separated_better") else ""))
        L.append(f"| {r['family']} | {lab} | {fmt(r)} | {'yes' if r['separated'] else 'no'} | {vd} | {ma} | {mb} | {r['n']} | {dg} |")
    return "\n".join(L)


md = []
for s in (0, 1):
    md.append(f"#### Read A, seed {s}: DAG-HACC - P10, T1-kin decisive (m = 10, causal bucket, dead-reckoned nav, logged scene)\n\n" + table(A_dec[s], s, dagA) + "\n")
open(f"{PK}/raw/READ_A_tables_DAGHACC.md", "w", encoding="utf-8", newline="\n").write("\n".join(md))
md2 = []
for nm, A in (("STRICT (scene frozen + limit causal), BESIDE", A_str), ("limit logged, BESIDE", A_log)):
    for s in (0, 1):
        md2.append(f"#### DAG-HACC - P10, {nm}, seed {s}\n\n" + table(A[s], s, None) + "\n")
open(f"{PK}/raw/READ_A_beside_tables_DAGHACC.md", "w", encoding="utf-8", newline="\n").write("\n".join(md2))
print(json.dumps(O, indent=1))
for s in (0, 1):
    print(f"s{s}", {k: V[s][k] for k in ("D1", "D2", "D3", "separated_worse_rows", "LONGITUDINAL_separated_worse", "M1_separated_worse", "L2_separated_worse", "not_computed")}, "KEEP", keep[s]["delta"], keep[s]["hi"], keep[s]["holds"])
print("ZZBUILD-DONE")
