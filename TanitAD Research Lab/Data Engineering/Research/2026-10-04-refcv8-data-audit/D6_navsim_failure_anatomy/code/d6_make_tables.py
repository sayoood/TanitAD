"""D6 -- render raw/d6_tables.md from the part-1..5 JSONs (so RESULT.md tables are generated, never hand-copied)."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "raw")


def J(n):
    return json.load(open(os.path.join(RAW, n), encoding="utf-8"))


def pc(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f} %"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def main():
    p1, p2, p3, p4, p5 = J("d6_part1.json"), J("d6_part2.json"), J("d6_part3_navtest.json"), J("d6_part4_intervals.json"), J("d6_part5_sensitivity.json")
    L = []
    w = L.append
    # ---- T1 official counterfactuals
    cf = p1["official_epdms_counterfactuals_A1_30k"]
    w("### T1 Official two-stage EPDMS: what each zero term costs (A1 step 30,000; counterfactual, ESTIMATED upper bound)\n")
    w("| change applied to A1's per-scene sub-scores | official EPDMS | delta vs A1 0.2269 | A1 vs STOP 0.2985 after |")
    w("|---|---|---|---|")
    base, stop = cf["A1_official"], cf["STOP_official"]
    for k, lab in (("match_STOP_on_DAC_where_A1_zero_STOP_passes", "DAC := STOP's DAC where A1 DAC=0 and STOP DAC=1"),
                   ("match_STOP_on_NC_where_A1_zero_STOP_passes", "NC := STOP's NC where A1 NC=0 and STOP NC=1"),
                   ("match_STOP_on_DDC_where_A1_zero_STOP_passes", "DDC := STOP's DDC likewise"),
                   ("match_STOP_on_TLC_where_A1_zero_STOP_passes", "TLC := STOP's TLC likewise"),
                   ("fix_DAC_to_1_all_scenes", "DAC := 1 on every scene (ceiling)"), ("fix_NC_to_1_all_scenes", "NC := 1 on every scene (ceiling)"),
                   ("fix_DAC_to_1_stage1_rows_only", "DAC := 1 on the 450 stage-1 rows only"), ("fix_DAC_to_1_stage2_rows_only", "DAC := 1 on the 5,462 stage-2 rows only"),
                   ("fix_NC_to_1_stage1_rows_only", "NC := 1 on stage-1 rows only"), ("fix_NC_to_1_stage2_rows_only", "NC := 1 on stage-2 rows only"),
                   ("fix_DAC_NC_DDC_TLC_to_1", "all four multipliers := 1 (ceiling)")):
        d = cf[k]
        w(f"| {lab} | {base + d:.4f} | {d:+.4f} | {base + d - stop:+.4f} |")
    w("")
    # ---- T2 cross-arm
    w("### T2 Who else fails where refcv7 (A1 step 30k) fails -- zero terms, per stage\n")
    w("| stage | term | A1 zero n (rate) | STOP rate | PRIOR rate | A1 zero & STOP also zero | A1 zero & STOP passes | A1 zero & PRIOR also zero | A1 zero & seed-1 also zero | A1 zero & 5k also zero | seed Jaccard |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for st in ("stage1", "stage2"):
        for t in ("DAC", "NC", "DDC", "TLC"):
            r = p1["cross_arm_patterns"][st][t]
            w(f"| {st[-1]} | {t} | {r['n_A1_zero']} ({pc(r['rate_A1'])}) | {pc(r['rate_STOP'])} | {pc(r['rate_PRIOR'])} | {r['of_A1_zero__STOP_also_zero']} | {r['of_A1_zero__STOP_passes']} | {r['of_A1_zero__PRIOR_also_zero']} | {r['of_A1_zero__A1s1_also_zero']} | {r['of_A1_zero__5k_also_zero']} | {r['seed_jaccard']:.2f} |")
    w("")
    # ---- T3 DAC scene-level flag + time
    a = p2["DAC_anatomy"]
    w(f"### T3 DAC-zero scenes (A1 30k): scene-level flag from the exact re-score (rescored {a['rescored_DACzero']['n']} of {a['rescored_DACzero']['of']})\n")
    w("| scene-level flag | stage 1 n (share) | stage 2 n (share) | both n (share) [95 % log-cluster CI] |")
    w("|---|---|---|---|")
    for k in ("INITIAL-STATE", "REF-ALSO-FAILS", "PLAN-INDUCED"):
        s1, s2, b = (a["scene_level_flag"][x].get(k, {"n": 0, "share": 0}) for x in ("stage1", "stage2", "both"))
        c = p4["DAC_scene_flag_shares_both"].get(k, {}).get("ci95_logs")
        w(f"| {k} | {s1['n']} ({pc(s1['share'])}) | {s2['n']} ({pc(s2['share'])}) | {b['n']} ({pc(b['share'])}) {ci(c) if c else ''} |")
    w("")
    w("First non-drivable instant of the plan (all rescored DAC-zero scenes):\n")
    w("| time bin | stage 1 | stage 2 | both |")
    w("|---|---|---|---|")
    ft = a["first_nondrivable_time_s"]
    for k in ("t=0", "(0,1]", "(1,2]", "(2,3]", "(3,4]"):
        w(f"| {k} s | {ft['stage1'].get(k, {'n': 0})['n']} | {ft['stage2'].get(k, {'n': 0})['n']} | {ft['both'].get(k, {'n': 0})['n']} ({pc(ft['both'].get(k, {'share': 0})['share'])}) |")
    w(f"\nmedian first non-drivable instant (both): {ft['median_s_both']:.1f} s\n")
    fp = J("d6_failure_patterns.json")["DAC_feasibility_cross"]
    w("First, for the same rescored DAC-zero scenes: does a clean path exist (PDM-closed reference passes DAC), does doing nothing pass (STOP), does the kinematic prior pass (PRIOR)?\n")
    w("| reference | STOP | PRIOR | stage 1 n | stage 2 n | both n (share) |")
    w("|---|---|---|---|---|---|")
    keys = sorted(set(fp["both"]["cells"]) | set(fp["stage1"]["cells"]) | set(fp["stage2"]["cells"]), key=lambda k: -fp["both"]["cells"].get(k, 0))
    for k in keys:
        r, st, pr = k.split("|")
        b = fp["both"]["cells"].get(k, 0)
        w(f"| {r} | {st} | {pr} | {fp['stage1']['cells'].get(k, 0)} | {fp['stage2']['cells'].get(k, 0)} | {b} ({pc(b / fp['both']['n'])}) |")
    w(f"\nseed-1 plan DAC-clean on {fp['both']['seed1_clean']} of {fp['both']['n']} ({pc(fp['both']['seed1_clean'] / fp['both']['n'])}); step-5,000 plan DAC-clean on {fp['both']['5k_clean']} ({pc(fp['both']['5k_clean'] / fp['both']['n'])}); reference clean AND none of STOP / PRIOR / seed-1 clean: {fp['both']['ref_clean_and_none_of_STOP_PRIOR_seed1_clean']} ({pc(fp['both']['ref_clean_and_none_of_STOP_PRIOR_seed1_clean'] / fp['both']['n'])}).\n")
    # ---- T4 geometry classes
    w("### T4 DAC-zero scenes by plan geometry class (route-referenced; literals in section 2)\n")
    w("| class | DAC-zero n (both) | share [95 % log CI] | PLAN-INDUCED subset n (share) | all scenes in class n | P(DAC=0 given class) A1 | STOP | PRIOR |")
    w("|---|---|---|---|---|---|---|---|")
    enr = a["enrichment_P_DACzero_given_class"]["12"]
    pi = a["PLAN_INDUCED_geometry_class"]["both"]
    for c in ("LATERAL-DRIFT", "ROUTE-FOLLOWING", "ON-ROUTE", "OVER-STEER", "SPEED", "NO-RECOVERY", "WRONG-SIDE", "STOP-LIKE"):
        b = a["geometry_class_DACzero"]["both"].get(c, {"n": 0, "share": 0})
        cc = p4["DAC_geometry_class_shares_both"].get(c, {}).get("ci95_logs")
        e = enr.get(c, {"n": 0, "A1_DAC0_rate": 0, "STOP_DAC0_rate": 0, "PRIOR_DAC0_rate": 0})
        pp = pi.get(c, {"n": 0, "share": 0})
        w(f"| {c} | {b['n']} | {pc(b['share'])} {ci(cc) if cc else ''} | {pp['n']} ({pc(pp['share'])}) | {e['n']} | {pc(e['A1_DAC0_rate'])} | {pc(e['STOP_DAC0_rate'])} | {pc(e['PRIOR_DAC0_rate'])} |")
    w("")
    # ---- T5 5k / prior
    w("### T5 Class mix of the DAC-zero scenes: step 5,000 vs 30,000 vs PRIOR vs seed-1 (geometry classes only; same literals)\n")
    cm = p2["class_mix_across_arms"]
    w("| class | A1 5k n (share) | A1 30k n (share) | A1 30k seed-1 n | PRIOR n (share) |")
    w("|---|---|---|---|---|")
    for c in ("LATERAL-DRIFT", "ROUTE-FOLLOWING", "ON-ROUTE", "OVER-STEER", "SPEED", "NO-RECOVERY", "WRONG-SIDE", "STOP-LIKE"):
        def g(k):
            return cm[k]["12"]["classes"].get(c, {"n": 0, "share": 0})
        w(f"| {c} | {g('A1_5k')['n']} ({pc(g('A1_5k')['share'])}) | {g('A1_30k')['n']} ({pc(g('A1_30k')['share'])}) | {g('A1_30k_seed1')['n']} | {g('PRIOR')['n']} ({pc(g('PRIOR')['share'])}) |")
    w(f"\nDAC-zero totals: 5k {cm['A1_5k']['12']['n_DACzero']}, 30k {cm['A1_30k']['12']['n_DACzero']}, seed-1 {cm['A1_30k_seed1']['12']['n_DACzero']}, PRIOR {cm['PRIOR']['12']['n_DACzero']}.\n")
    # ---- T6 command split
    w("### T6 Command x route geometry in the horizon (all 5,912 scenes) and DAC-zero rates\n")
    w("| command | n | route in horizon: junction turn | curve turn | straight | A1 DAC0 | STOP DAC0 | PRIOR DAC0 |")
    w("|---|---|---|---|---|---|---|---|")
    for c, v in p2["command_vs_route_geometry"]["12"].items():
        r = v["route_in_horizon"]
        w(f"| {c} | {v['n']} | {r.get('turn-junction', 0)} | {r.get('turn-curve', 0)} | {r.get('straight-in-horizon', 0)} | {pc(v['A1_DAC0_rate'])} | {pc(v['STOP_DAC0_rate'])} | {pc(v['PRIOR_DAC0_rate'])} |")
    w("\nPlan complied with the command (LEFT: plan yaw at 4 s >= +10 deg; RIGHT: <= -10 deg; STRAIGHT: |yaw| < 20 deg) and DAC-zero rate, by command x route-in-horizon:\n")
    w("| command | route in horizon | n | plan complied | A1 DAC0 | STOP DAC0 | A1 DAC0 if complied (n) | A1 DAC0 if not complied (n) |")
    w("|---|---|---|---|---|---|---|---|")
    for k, v in p2["command_compliance"].items():
        cmd, rt = k.split(" | ")
        w(f"| {cmd} | {rt} | {v['n']} | {pc(v['plan_complied'])} | {pc(v['A1_DAC0_rate'])} | {pc(v['STOP_DAC0_rate'])} | {pc(v['A1_DAC0_rate_complied'])} ({v['n_complied']}) | {pc(v['A1_DAC0_rate_not_complied'])} ({v['n_not_complied']}) |")
    w("\nDAC-zero ROUTE-FOLLOWING / WRONG-SIDE / OVER-STEER / LATERAL-DRIFT scenes by command and by route kind (n per cell):\n")
    w("| class | DAC-zero n | LEFT | RIGHT | STRAIGHT | junction turn | curve turn | straight route | all scenes of class: LEFT/RIGHT/STRAIGHT | P(DAC0) in class: LEFT / RIGHT / STRAIGHT | all scenes of class: junction/curve/straight | P(DAC0) in class: junction / curve |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c, v in p2["class_x_command_x_routekind"].items():
        d, ab, rk, rr = v["DACzero_by_command"], v["all_by_command"], v["DACzero_by_route_kind"], v["DACzero_rate_by_command"]
        ak = v["all_by_route_kind"]
        rj = rk["junction"] / ak["junction"] if ak["junction"] else None
        rc = rk["curve"] / ak["curve"] if ak["curve"] else None
        w(f"| {c} | {v['n_DACzero']} | {d.get('LEFT', 0)} | {d.get('RIGHT', 0)} | {d.get('STRAIGHT', 0)} | {rk['junction']} | {rk['curve']} | {rk['straight']} | {ab.get('LEFT', 0)}/{ab.get('RIGHT', 0)}/{ab.get('STRAIGHT', 0)} | {pc(rr.get('LEFT'))} / {pc(rr.get('RIGHT'))} / {pc(rr.get('STRAIGHT'))} | {ak['junction']}/{ak['curve']}/{ak['straight']} | {pc(rj)} / {pc(rc)} |")
    w("")
    # ---- T7 NC
    nc = p2["NC_anatomy"]
    w(f"### T7 NC-zero scenes (A1 30k): first at-fault collision from the exact re-score (rescored {nc['rescored']['n']} of {nc['rescored']['of']})\n")
    w("| class | stage 1 n | stage 2 n | both n (share) [95 % log CI] | median time s | ego speed m/s | object speed m/s | rel. longitudinal m | also DAC-zero | STOP also NC0 | PRIOR also NC0 | seed-1 also NC0 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c, b in sorted(nc["class"]["both"].items(), key=lambda kv: -kv[1]["n"]):
        s1, s2 = nc["class"]["stage1"].get(c, {"n": 0}), nc["class"]["stage2"].get(c, {"n": 0})
        cc = p4.get("NC_class_shares_both", {}).get(c, {}).get("ci95_logs")
        sx = nc["STOP_also_NC_zero"][c]
        w(f"| {c} | {s1['n']} | {s2['n']} | {b['n']} ({pc(b['share'])}) {ci(cc) if cc else ''} | {nc['collision_time_s_median_by_class'][c]:.1f} | {nc['ego_speed_at_collision_median_by_class'][c]:.1f} | {nc['obj_speed_at_collision_median_by_class'][c]:.1f} | {nc['rel_lon_median_by_class'][c]:.1f} | {nc['also_DAC_zero'][c]['DAC0']} | {sx['STOP_NC0']} | {sx['PRIOR_NC0']} | {sx['A1s1_NC0']} |")
    w(f"\nplan 4-s speed minus PDM-closed reference 4-s speed (finite-difference), median: NC-zero {nc['plan_speed4_minus_ref_speed4_m_s']['NC_zero_median']:+.2f} m/s vs all scenes {nc['plan_speed4_minus_ref_speed4_m_s']['all_scenes_median']:+.2f} m/s; share with plan >= 3 m/s faster than the reference: NC-zero {pc(nc['plan_faster_than_ref_by_3ms_share']['NC_zero'])} vs all {pc(nc['plan_faster_than_ref_by_3ms_share']['all_scenes'])}.\n")
    w("NC-zero rate by ego speed at t0 (all 5,912 scenes):\n")
    w("| v0 band m/s | n | A1 NC0 | STOP NC0 | PRIOR NC0 |")
    w("|---|---|---|---|---|")
    for b, v in nc["by_v0_band"].items():
        w(f"| {b} | {v['n_all']} | {pc(v['A1_NC0_rate'])} | {pc(v['STOP_NC0_rate'])} | {pc(v['PRIOR_NC0_rate'])} |")
    w("")
    # ---- T8 vmax
    w("### T8 Max-speed input known vs unknown (D4 finding 1)\n")
    w("NavSim feeds `v_max_valid = 0` (all-zero row, 'unknown') on these shares; zero-rates are per-stratum, n per cell:\n")
    w("| split | stage | stratum | n (share of scenes) | A1 DAC0 | STOP DAC0 | PRIOR DAC0 | A1 NC0 | STOP NC0 | PRIOR NC0 | share of A1's DAC0 / NC0 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in ("1", "2", "12"):
        for k in ("unknown", "known"):
            v = p2["vmax_split_navhard"][s][k]
            w(f"| navhard | {'both' if s == '12' else s} | {k} | {v['n']} ({pc(v['share_of_scenes'])}) | {pc(v['A1_DAC0'])} | {pc(v['STOP_DAC0'])} | {pc(v['PRIOR_DAC0'])} | {pc(v['A1_NC0'])} | {pc(v['STOP_NC0'])} | {pc(v['PRIOR_NC0'])} | {pc(v['share_of_A1_DAC0'])} / {pc(v['share_of_A1_NC0'])} |")
    for k in ("unknown", "known"):
        v = p3["vmax_split_navtest"][k]
        w(f"| navtest | 1 | {k} | {v['n']} ({pc(v['share_of_scenes'])}) | {pc(v['A1_DAC0'])} | {pc(v['STOP_DAC0'])} | {pc(v['PRIOR_DAC0'])} | {pc(v['A1_NC0'])} | {pc(v['STOP_NC0'])} | {pc(v['PRIOR_NC0'])} | {pc(v['share_of_A1_DAC0'])} / {pc(v['share_of_A1_NC0'])} |")
    w("\nStandardised (over stage x v0-band x command cells) unknown-minus-known zero-rate difference, percentage points; PRIOR is an input-free kinematic control, STOP a plan-free one:\n")
    w("| split | term | A1 | PRIOR (control) | STOP (control) |")
    w("|---|---|---|---|---|")
    sd = p2["vmax_standardised_diff_unknown_minus_known"]
    sdn = p3["vmax_standardised_diff_unknown_minus_known_navtest"]
    for t in ("DAC", "NC"):
        w(f"| navhard | {t} | {100 * sd[f'A1_{t}0']:+.2f} | {100 * sd[f'PRIOR_{t}0']:+.2f} | {100 * sd[f'STOP_{t}0']:+.2f} |")
        w(f"| navtest | {t} | {100 * sdn[f'A1_{t}0']:+.2f} | {100 * sdn[f'PRIOR_{t}0']:+.2f} | {100 * sdn[f'STOP_{t}0']:+.2f} |")
    w("\nDifference-in-differences [(A1 - reference arm)_unknown - (A1 - reference arm)_known], log-cluster 95 % CI:\n")
    w("| split | term | reference arm | DiD (pp) | 95 % CI (pp) | n unknown / known |")
    w("|---|---|---|---|---|---|")
    for refn, key in (("STOP (plan-free)", "difference_in_differences_A1_vs_STOP"), ("PRIOR (input-free kinematic plan)", "difference_in_differences_A1_vs_PRIOR")):
        for nm, blk in (("navhard both", {t: p4["vmax_navhard"][f"both_{t}"][key] for t in ("DAC", "NC")}),
                        ("navhard stage 2", {t: p4["vmax_navhard"][f"stage2_{t}"][key] for t in ("DAC", "NC")}),
                        ("navtest", {t: p4["vmax_navtest"][t][key] for t in ("DAC", "NC")})):
            for t, r in blk.items():
                w(f"| {nm} | {t} | {refn} | {100 * r['did']:+.2f} | [{100 * r['ci95_logs'][0]:+.2f}, {100 * r['ci95_logs'][1]:+.2f}] | {r['n_unknown']} / {r['n_known']} |")
    w("")
    # ---- T9 navtest
    nt = p3["DAC_anatomy"]
    w("### T9 NAVTEST (single-stage PDMS v1, 12,146 tokens): DAC-zero classes against the HUMAN future\n")
    w("| class | A1 30k DAC0 n (share) | A1 5k n (share) | PRIOR n (share) | P(DAC0 given class) A1 30k | all scenes in class |")
    w("|---|---|---|---|---|---|")
    for c in ("ON-HUMAN", "ROUTE-FOLLOWING", "OVER-STEER", "LATERAL-DRIFT", "SPEED", "WRONG-SIDE", "STOP-LIKE"):
        def g(k):
            return nt[k]["class_mix"].get(c, {"n": 0, "share": 0})
        r = nt["A1_30k"]["rate_by_class"].get(c, {"n": 0, "DAC0_rate": 0})
        w(f"| {c} | {g('A1_30k')['n']} ({pc(g('A1_30k')['share'])}) | {g('A1_5k')['n']} ({pc(g('A1_5k')['share'])}) | {g('PRIOR_30k')['n']} ({pc(g('PRIOR_30k')['share'])}) | {pc(r['DAC0_rate'])} | {r['n']} |")
    w(f"\nnavtest DAC-zero totals: 30k {nt['A1_30k']['n_DACzero']} ({pc(nt['A1_30k']['rate'])}), 5k {nt['A1_5k']['n_DACzero']} ({pc(nt['A1_5k']['rate'])}), PRIOR {nt['PRIOR_30k']['n_DACzero']} ({pc(nt['PRIOR_30k']['rate'])}).")
    cx = p3["DAC_cross_arm"]
    w(f"Among A1 30k's {cx['n']} DAC-zero tokens: STOP also zero {cx['STOP_also_zero']}, STOP passes {cx['STOP_passes']}, PRIOR also zero {cx['PRIOR_also_zero']}, seed-1 also zero {cx['A1s1_also_zero']}, 5k also zero {cx['A1_5k_also_zero']}, STOP and PRIOR both pass {cx['STOP_and_PRIOR_pass']}; seed Jaccard {cx['seed_jaccard']:.2f}.\n")
    w("NAVTEST NC-zero (A1 30k) by plan class vs the human, and plan-vs-human distance:\n")
    w("| class | NC0 n (share) | P(NC0 given class) A1 | STOP | all scenes in class |")
    w("|---|---|---|---|---|")
    for c, v in sorted(p3["NC_class_mix"]["A1_30k"].items(), key=lambda kv: -kv[1]["n"]):
        r = p3["NC_class_mix"]["rate_by_class"][c]
        w(f"| {c} | {v['n']} ({pc(v['share'])}) | {pc(r['NC0_rate'])} | {pc(r['STOP_NC0_rate'])} | {r['n']} |")
    nd = p3["NC_plan_vs_human_distance"]
    w(f"\nplan 4-s distance / human 4-s distance, median: NC-zero {nd['NC_zero_median_ratio']:.2f} vs all {nd['all_median_ratio']:.2f}; share with plan >= 4 m further than the human: NC-zero {pc(nd['NC_zero_plan_faster_than_human_by_4m_share'])} vs all {pc(nd['all_plan_faster_than_human_by_4m_share'])}.")
    nx = p3["NC_cross_arm"]
    w(f"NC-zero {nx['n']} ({pc(nx['rate'])}): STOP also {nx['STOP_also_zero']}, PRIOR also {nx['PRIOR_also_zero']}, seed-1 also {nx['A1s1_also_zero']}, 5k also {nx['A1_5k_also_zero']}, also DAC-zero {nx['also_DAC_zero']}; seed Jaccard {nx['seed_jaccard']:.2f}.\n")
    w("navtest command split:\n")
    w("| command | n | GT turn in 4 s (L/R/S) | A1 DAC0 | STOP DAC0 | PRIOR DAC0 | A1 NC0 | STOP NC0 | plan complied |")
    w("|---|---|---|---|---|---|---|---|---|")
    for c, v in p3["command_split"].items():
        gt = v["gt_turn_in_4s"]
        w(f"| {c} | {v['n']} | {gt.get('L', 0)}/{gt.get('R', 0)}/{gt.get('S', 0)} | {pc(v['A1_DAC0'])} | {pc(v['STOP_DAC0'])} | {pc(v['PRIOR_DAC0'])} | {pc(v['A1_NC0'])} | {pc(v['STOP_NC0'])} | {pc(v['plan_complied'])} |")
    w("")
    # ---- T10 sensitivity
    w("### T10 Threshold sensitivity of the DAC-zero class shares (navhard, both stages; every literal x0.5 / x1.5)\n")
    w("| class | share at the literals | min over scenarios | max |")
    w("|---|---|---|---|")
    for c, v in sorted(p5["range_by_class"].items(), key=lambda kv: -kv[1]["base"]):
        w(f"| {c} | {pc(v['base'])} | {pc(v['min'])} | {pc(v['max'])} |")
    w("")
    open(os.path.join(RAW, "d6_tables.md"), "w", encoding="utf-8").write("\n".join(L))
    print("tables:", len(L), "lines")


if __name__ == "__main__":
    main()
