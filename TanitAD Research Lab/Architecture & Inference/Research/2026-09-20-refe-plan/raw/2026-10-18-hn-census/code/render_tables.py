#!/usr/bin/env python3
"""Render the markdown tables of RESULT.md from raw/census_results.json + raw/posthoc.json (so no number is transcribed by hand).
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I render_tables.py ../raw > ../raw/tables.md
"""
import json
import sys

root = sys.argv[1]
R = json.load(open(f"{root}/census_results.json", encoding="utf-8"))
P = json.load(open(f"{root}/posthoc.json", encoding="utf-8"))
P2 = json.load(open(f"{root}/posthoc2.json", encoding="utf-8"))
G = R["grid_m"]


def ci(x, p=3, pct=False):
    m = 100.0 if pct else 1.0
    return f"{x['point']*m:.{p}f} [{x['lo']*m:.{p}f}, {x['hi']*m:.{p}f}]"


def row(*c):
    return "| " + " | ".join(str(x) for x in c) + " |"


out = []
D = R["D_base_rates"]
out += ["### T0 base rates (all 64 x 3,137 proposals)", "",
        row("quantity", "value"), row("---", "---"),
        row("hard-negative (PDMS = 0) rate, all proposals", ci(D["HN_rate_all_proposals"], 2, True) + " %"),
        row("sets with >= 1 hard negative anywhere in the 64", ci(D["share_sets_with_any_HN"], 1, True) + " %"),
        row("sets with >= 1 DAC-0 proposal", ci(D["share_sets_with_any_DAC0"], 1, True) + " %"),
        row("counts: HN / DAC-0 / NC-0 only / both", f"{D['n_HN']:,} / {D['n_DAC0']:,} / {D['n_NC0_only']:,} / {D['n_both']:,}"),
        row("per-set HN rate q05/q25/q50/q75/q95 (point)", " / ".join(f"{v:.3f}" for v in D["per_set_HN_rate_quantiles(point)"].values())), ""]

for key, title in (("A1_ADE", "T1 SUPPLY, proximity = ADE to the teacher plan (PRIMARY)"), ("A1_FDE", "T1b SUPPLY, proximity = FDE (sensitivity)")):
    out += [f"### {title}", "", row("d (m)", "sets with a proximate proposal %", "**sets with >= 1 proximate HN %**", "mean proximate / set", "mean proximate HN / set", "HN rate among proximate %", "share of ALL HN that are proximate %"), row(*["---"] * 7)]
    for d in G:
        e = R["ET"][key][f"d={d}"]
        out.append(row(d, ci(e["share_sets_with_proximate"], 1, True), "**" + ci(e["share_sets_with_proximate_HN"], 1, True) + "**", ci(e["mean_n_prox_per_set"], 1), ci(e["mean_n_HN_per_set"], 2), ci(e["pooled_HN_rate_among_proximate"], 1, True),
                       ci(e["share_of_all_HN_that_are_proximate"], 1, True)))
    out.append("")

out += ["### T1c sets with >= 1 proximate HN by sub-type (ADE), % [95 % CI over logs]", "", row("d (m)", "DAC-only", "NC-only", "both"), row("---", "---", "---", "---")]
for d in G:
    s = R["ET"]["A1_ADE"][f"d={d}"]["share_sets_with_proximate_HN_by_subtype"]
    out.append(row(d, ci(s["DAC_only"], 1, True), ci(s["NC_only"], 1, True), ci(s["both"], 1, True)))
out += ["", "### T1d histogram of proximate-HN count per set (ADE; point; 3,137 sets)", "", row("d (m)", "0", "1", "2", "3-4", "5+"), row(*["---"] * 6)]
for d in G:
    h = R["ET"]["A1_ADE"][f"d={d}"]["hist_n_HN_per_set(point)"]
    out.append(row(d, h["0"], h["1"], h["2"], h["3-4"], h["5+"]))

out += ["", "### T2 hard-negative rate by distance band, % [95 % CI over logs]", "", row("band (m)", "ADE: HN rate", "ADE: n proposals", "ADE: n HN", "FDE: HN rate", "FDE: n proposals"), row(*["---"] * 6)]
for (k, v), (k2, v2) in zip(R["ET"]["A2e_bands_ADE"].items(), R["ET"]["A2e_bands_FDE"].items()):
    out.append(row(k, ci(v["HN_rate"], 1, True), f"{v['n_proposals']:,}", f"{v['n_HN']:,}", ci(v2["HN_rate"], 1, True), f"{v2['n_proposals']:,}"))

out += ["", "### T3 distance quantiles (ADE, m) [95 % CI over logs; exact 0.01 m histograms]", "", row("population", "n", "q05", "q25", "q50", "q75", "q95"), row(*["---"] * 7)]
Q = R["ET"]["A2_quantiles_ADE"]
for nm, lab in (("a_all_proposals", "all proposals"), ("b_nearest_proposal_per_set", "nearest proposal per set"), ("c_nearest_HN_per_set", "nearest HN per set (sets with an HN anywhere)"),
                ("d_hard_negatives_pooled", "hard negatives, pooled"), ("d_non_hard_negatives_pooled", "non-hard-negatives, pooled")):
    v = Q[nm]
    out.append(row(lab, f"{v['n']:,}", *[f"{v[q]['point']:.3f} [{v[q]['lo']:.3f}, {v[q]['hi']:.3f}]" for q in ("q05", "q25", "q50", "q75", "q95")]))
out += ["", f"Sets with a hard negative anywhere in the 64: {ci(Q['c_share_sets_with_a_HN_anywhere'], 1, True)} %.  FDE: all proposals median "
        f"{R['ET']['A2_quantiles_FDE']['a_all_proposals']['q50']['point']:.2f} m, nearest-proposal median {R['ET']['A2_quantiles_FDE']['b_nearest_proposal_per_set']['q50']['point']:.3f} m.", ""]

out += ["### T4 strata (expert-manoeuvre class from the teacher plan; teacher rank): sets with >= 1 proximate HN, ADE", "", row("stratum", "sets", "logs", "CI valid", "HN rate %", "d=0.5 %", "d=1 %", "d=2 %"), row(*["---"] * 8)]
for nm, v in R["ET"]["A3_strata"].items():
    out.append(row(nm, v["n_sets"], v["n_logs"], "yes" if v["ci_valid"] else "point only", ci(v["HN_base_rate"], 1, True), ci(v["d=0.5"]["share_sets_with_proximate_HN"], 1, True),
                   ci(v["d=1.0"]["share_sets_with_proximate_HN"], 1, True), ci(v["d=2.0"]["share_sets_with_proximate_HN"], 1, True)))

out += ["", "### T5 per-log (point; sorted by s_hn(0.5))", "", row("log", "sets", "HN rate", "s_hn 0.5", "s_hn 1", "s_hn 2", "s_hn 4"), row(*["---"] * 7)]
for r in sorted(R["ET"]["A4_per_log"], key=lambda r: -r["s_hn_d0.5"]):
    out.append(row(r["log"], r["n_sets"], f"{r['HN_rate']:.3f}", f"{r['s_hn_d0.5']:.2f}", f"{r['s_hn_d1.0']:.2f}", f"{r['s_hn_d2.0']:.2f}", f"{r['s_hn_d4.0']:.2f}"))

out += ["", "### T6 Lab step-0 twin pairs inside a set (mean pairwise L2 <= t), % or count [95 % CI over logs]", "", row("t (m)", "sets with >= 1 DAC0/DAC1 twin pair %", "mean such pairs / set", "DAC-0 proposals that have a DAC-1 twin %", "sets WITH a DAC-0 that have a twin pair %", "sets with >= 1 PDMS0/PDMS>0 pair %"), row(*["---"] * 6)]
for t, v in R["B_twin_pairs"].items():
    out.append(row(t, ci(v["share_sets_with_DAC_twin_pair"], 1, True), ci(v["mean_DAC_twin_pairs_per_set"], 1), ci(v["twinned_fraction_of_DAC0_proposals"], 1, True),
                   ci(v["share_sets_with_DAC0_that_have_a_twin_pair"], 1, True), ci(v["share_sets_with_PDMS_twin_pair"], 1, True)))

EH = R["E_H"]
out += ["", f"### T7 SECONDARY expert = logged human future ({EH['coverage_sets']:,} of {EH['n_sets']:,} sets, {EH['coverage_logs']} of 24 logs); gate G7 median ADE(teacher, human) = {EH['G7_median_ADE_teacher_vs_human']:.3f} m (<= 5: {'PASS' if EH['G7_pass'] else 'FAIL'})", "",
        row("d (m)", "human: sets with >= 1 proximate HN %", "teacher, SAME sets: same quantity %", "human: HN rate among proximate %", "teacher same sets %"), row(*["---"] * 5)]
for d in G:
    h, t = EH["A1_ADE_human"][f"d={d}"], EH["A1_ADE_teacher_same_sets"][f"d={d}"]
    out.append(row(d, ci(h["share_sets_with_proximate_HN"], 1, True), ci(t["share_sets_with_proximate_HN"], 1, True), ci(h["pooled_HN_rate_among_proximate"], 1, True), ci(t["pooled_HN_rate_among_proximate"], 1, True)))
out += ["", f"Nearest-proposal ADE median: human {EH['nearest_proposal_ADE_median_human']:.3f} m, teacher (same sets) {EH['nearest_proposal_ADE_median_teacher_same_sets']:.3f} m.  teacher-vs-human ADE quantiles q05/25/50/75/95: "
        + " / ".join(f"{v:.2f}" for v in EH["teacher_vs_human_ADE_quantiles"].values()) + " m.", ""]

K2 = R["K2_wrong_expert"]
out += ["### K2 control: wrong expert (teacher of another set of the same log)", "", row("d (m)", "true expert: sets with proximate HN %", "wrong expert %"), row("---", "---", "---")]
for d in G:
    out.append(row(d, ci(R["ET"]["A1_ADE"][f"d={d}"]["share_sets_with_proximate_HN"], 1, True), ci(K2["A1_ADE"][f"d={d}"]["share_sets_with_proximate_HN"], 1, True)))
out += ["", f"Nearest-proposal ADE median: true {K2['nearest_proposal_ADE_median_true']:.3f} m, wrong {K2['nearest_proposal_ADE_median_wrong']:.3f} m.  K1 identity: nearest ADE max {R['K1_identity']['nearest_ADE_max']}, proximate at every d: {R['K1_identity']['all_sets_have_proximate_at_every_d']}.", ""]

C = R["C_reach"]
out += ["### T8 reach C1 (ESTIMATED): supply-limited = s_hn(d) * r * Z * kappa, PDMS points, [CI endpoints of s_hn only]", "", row("d (m)", "r = 0.208 (DA-WAM)", "r = 0.227 (BeyondDrive)"), row("---", "---", "---")]
for d in G:
    v = C["C1_supply_limited_PDMS"][f"d={d}"]
    out.append(row(d, ci(v["DA-WAM"], 2), ci(v["BeyondDrive"], 2)))
u = C["C1_supply_limited_PDMS"]["unconstrained_standalone(s=1)"]
out += ["", f"Unconstrained standalone (s = 1, the Lab's number): {u['DA-WAM']:.2f} / {u['BeyondDrive']:.2f} PDMS.", ""]
out += ["### T9 reach C2 (ESTIMATED ceilings): where the sibling scorers' zero picks sit, each scorer read separately", "",
        row("scorer", "zero picks (n, rate %)", "d (m)", "zero picks within d of the teacher %", "ceiling = coverage*Z*kappa (PDMS)", "zero-pick sets that contain a proximate HN %"), row(*["---"] * 6)]
for arm, v in C["C2_zero_pick_coverage"].items():
    for d in G:
        e = v[f"d={d}"]
        out.append(row(arm if d == G[0] else "", f"{v['n_zero_picks']}, {ci(v['zero_pick_rate'], 2, True)}" if d == G[0] else "", d, ci(e["zero_picks_that_are_expert_proximate"], 1, True),
                       ci(e["ceiling_PDMS_from_pick_coverage"], 2), ci(e["zero_pick_sets_with_proximate_HN"], 1, True)))
out += ["", "### T10 POST-HOC (not pre-registered): distance of the siblings' picks to the teacher plan (point; m)", "", row("scorer", "zero picks n", "ADE zero picks q25/q50/q75", "ADE non-zero picks q25/q50/q75", "FDE zero picks q50", "FDE non-zero picks q50", "zero picks: DAC-0 / NC-0 only"), row(*["---"] * 7)]
for arm, v in P["Q3_pick_distance_to_teacher(point)"].items():
    f3 = lambda q: " / ".join(f"{q[k]:.2f}" for k in ("q25", "q50", "q75"))
    out.append(row(arm, v["n_zero_picks"], f3(v["ADE_zero_picks"]), f3(v["ADE_nonzero_picks"]), f"{v['FDE_zero_picks']['q50']:.2f}", f"{v['FDE_nonzero_picks']['q50']:.2f}", f"{v['zero_pick_is_DAC0']} / {v['zero_pick_is_NC0_only']}"))
q1 = P["Q1_calculator_vs_PDM_label_on_proposals"]
out += ["", "POST-HOC Q1/Q2 (point unless a CI is shown): teacher flagged by its own collision calculator in "
        f"{P['Q1_teacher_unsafe_by_its_own_calculators']['collision']} of 3,137 sets; its DriveRL `dac.violation` flags {P['Q1_teacher_unsafe_by_its_own_calculators']['dac_violation']} sets, and the same field flags only {q1['dac.violation>0 vs dac==0']['both']} of "
        f"{q1['dac.violation>0 vs dac==0']['both']+q1['dac.violation>0 vs dac==0']['label_only']} PDM DAC-0 proposals (UNUSABLE as a teacher-DAC check); collision-calculator vs PDM NC==0 on proposals: both {q1['collision_info>0 vs nc==0']['both']:,}, "
        f"calculator only {q1['collision_info>0 vs nc==0']['calc_only']:,}, label only {q1['collision_info>0 vs nc==0']['label_only']:,}.  Share of proximate-HN sets (d=0.5) in which the teacher is collision-flagged: "
        f"{ci(P['Q1_d=0.5']['share_of_proximate_HN_sets_where_teacher_is_unsafe'], 2, True)} %.  Concentration at d=0.5: {P['Q2_concentration_d0.5']['sets_with_proximate_HN']} sets with a proximate HN, half of them in {P['Q2_concentration_d0.5']['k50_logs']} of 24 logs, "
        f"top log {P['Q2_concentration_d0.5']['top3'][0]['log']} = {P['Q2_concentration_d0.5']['top3'][0]['share']*100:.1f} %.", ""]

KZ = 0.953 * 4.36
X = P2["cross_track_census"]
out += ["", "### T11 POST-HOC (not pre-registered): the supply census with CROSS-TRACK proximity (mean abs cross-track to the teacher path over the 20 poses; speed ignored); thresholds fixed before the run", "",
        row("c (m)", "sets with a proximate proposal %", "**sets with >= 1 proximate HN %**", "mean proximate / set", "mean proximate HN / set", "HN rate among proximate %", "C1 reach r=0.208 (PDMS)", "C1 reach r=0.227 (PDMS)"), row(*["---"] * 8)]
for c, v in X["A1"].items():
    sh = v["share_sets_with_proximate_HN"]
    out.append(row(c.split("=")[1], ci(v["share_sets_with_proximate"], 1, True), "**" + ci(sh, 1, True) + "**", ci(v["mean_n_prox_per_set"], 1), ci(v["mean_n_HN_per_set"], 2), ci(v["pooled_HN_rate_among_proximate"], 1, True),
                   f"{sh['point']*0.208*KZ:.2f} [{sh['lo']*0.208*KZ:.2f}, {sh['hi']*0.208*KZ:.2f}]", f"{sh['point']*0.227*KZ:.2f} [{sh['lo']*0.227*KZ:.2f}, {sh['hi']*0.227*KZ:.2f}]"))
out += ["", row("cross-track band (m)", "HN rate %", "n proposals", "n HN"), row("---", "---", "---", "---")]
for k, v in X["bands"].items():
    out.append(row(k, ci(v["HN_rate"], 1, True), f"{v['n_proposals']:,}", f"{v['n_HN']:,}"))
out += ["", row("scorer", "zero picks within c = 0.3 %", "c = 0.5 %", "c = 1.0 %", "ceiling at c = 0.5 (PDMS)"), row(*["---"] * 5)]
for arm, v in X["zero_pick_coverage"].items():
    cv = v["c=0.5"]
    out.append(row(arm, ci(v["c=0.3"], 1, True), ci(v["c=0.5"], 1, True), ci(v["c=1.0"], 1, True), f"{cv['point']*KZ:.2f} [{cv['lo']*KZ:.2f}, {cv['hi']*KZ:.2f}]"))

q3 = lambda d: "n/a" if d is None else " / ".join(f"{d[k]:.2f}" for k in ("q25", "q50", "q75"))
out += ["", "### T12 POST-HOC: how hard negatives and zero picks deviate from the teacher path (teacher-tangent frame, m; medians q25/q50/q75 over proposals; point)", "",
        row("population", "n", "mean abs along-track", "mean abs cross-track", "mean signed along-track (negative = behind the teacher)"), row(*["---"] * 5)]
for b, v in P2["all_proposals_by_ADE_band"].items():
    for k in ("HN", "nonHN"):
        x = v[k]
        out.append(row(f"ADE band {b}: {k}", f"{x['n']:,}", q3(x["mean|along|"]), q3(x["mean|cross|"]), q3(x["signed_along"])))
for arm, v in P2["picks"].items():
    for k in ("zero", "nonzero"):
        x = v[k]
        out.append(row(f"{arm} picks: {k}", x["n"], q3(x["mean|along|"]), q3(x["mean|cross|"]), q3(x["signed_along"]) + f"; path-length ratio to teacher {q3(x['path_len_ratio_to_teacher'])}"))

print("\n".join(out))
