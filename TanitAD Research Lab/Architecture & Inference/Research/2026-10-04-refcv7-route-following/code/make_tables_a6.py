"""Markdown tables for RESULT_A6.md, read straight from raw/a6_dense_score.json (no number is typed by hand).
Usage: python make_tables_a6.py <raw dir>   -> prints markdown (redirect to raw/TABLES_a6_generated.md)"""
import json
import sys
from pathlib import Path


def ci(c, nd=3):
    if not c or c[0] is None:
        return ""
    return f" [{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


def d(L, arm, cls, m="ade", nd=3):
    e = L[arm][cls][m]
    if e.get("delta_vs_V0") is None:
        return "–"
    return f"{e['delta_vs_V0']:+.{nd}f}{ci(e['delta_ci95'], nd)}"


def mean(L, arm, cls, m, nd=3):
    e = L[arm][cls][m]
    return "–" if e["mean"] is None else f"{e['mean']:.{nd}f}"


def bar_rows(L0, L1, bars, title):
    out = [f"\n**{title}**\n",
           "| arm | picks changed | turn dADE [CI] | turn dir-correct d [CI] | straight dADE [CI] | all dADE [CI] | seed 1: turn dADE [CI] | seed 1: turn dir-correct d [CI] | criteria 1/2/3/4 | bar |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    nm = {"T3a": "**T3a (REPORTED)**", "T3": "T3 (foil: all entries)", "T2a": "T2a (hard filter, announced)", "T3a-c": "T3a-c CONTROL (derangement)"}
    for k in ("T3a", "T3", "T2a", "T3a-c"):
        b = bars[k]
        cr = "/".join("Y" if b[x] else "N" for x in ("1_turn_ade_and_dircorrect", "2_straight_no_regression", "3_all_ade_not_worse", "4_replicates_seed1"))
        out.append(f"| {nm[k]} | {L0[k]['pick_changed_frac']:.3f} | {d(L0, k, 'turn')} | {d(L0, k, 'turn', 'dir_correct')} | {d(L0, k, 'straight')} | {d(L0, k, 'all')} | "
                   f"{d(L1, k, 'turn')} | {d(L1, k, 'turn', 'dir_correct')} | {cr} | **{'CLEARS' if b['CLEARS'] else 'FAILED'}** |")
    for k in ("B1t", "ORACLE"):
        out.append(f"| {k} (label-side bound) | {L0[k]['pick_changed_frac']:.3f} | {d(L0, k, 'turn')} | {d(L0, k, 'turn', 'dir_correct')} | {d(L0, k, 'straight')} | {d(L0, k, 'all')} | "
                   f"{d(L1, k, 'turn')} | {d(L1, k, 'turn', 'dir_correct')} | – | bound |")
    return out


def main():
    R = json.loads((Path(sys.argv[1]) / "a6_dense_score.json").read_text(encoding="utf-8"))
    L0, L1 = R["dense_seed0"], R["dense_seed1"]
    out = []
    c = R["controls"]
    out.append("## controls\n")
    out.append("| control | measured | verdict |\n|---|---|---|")
    for k, v in c.items():
        if k.startswith("C_overlap"):
            for nm in ("seed0_vs_eval_s0g", "seed1_vs_eval_s1"):
                x = v[nm]
                out.append(f"| overlap identity {nm} ({x['n_overlap_windows']} windows) | fan max abs diff {x['fan_max_abs_diff']:.1e}, s_e9 {x['s_e9_max_abs_diff']:.1e}, sel_idx mismatches {x['sel_idx_mismatch']} | {'PASS' if x['PASS'] else 'FAIL'} |")
        elif k.startswith("K1"):
            out.append(f"| K1 clock vs replay bank | max abs diff {v['max_abs_diff_s']:.2e} s over {v['n_windows']} windows | {'PASS' if v['PASS'] else 'FAIL'} |")
        elif k.startswith("K2_nav"):
            out.append(f"| K2 nav_tl == clip token at t_rel ~ 0 (dense windows) | {v['n_equal']}/{v['n_windows']} | {'PASS' if v['PASS'] else 'FAIL'} |")
        elif k.startswith("K2_sup"):
            out.append(f"| K2 supplementary (one t per eval clip) | {v['n_clips_with_t_in_band_and_time_s_le_6'] - v['n_unequal']}/{v['n_clips_with_t_in_band_and_time_s_le_6']} | {'PASS' if v['PASS'] else 'FAIL'} |")
        elif k.startswith("K0_join"):
            out.append(f"| {k} | {v['n_equal']}/{v['n']} | {'PASS' if v['PASS'] else 'FAIL'} |")
        elif k.startswith("C1_"):
            out.append(f"| {k} | traj-vs-fan {v['traj_vs_fan_max_abs']}, E9 argmax mismatches {v['n_e9_argmax_mismatch']}, core {v['n_core_argmax_mismatch']}, forwards/window {v['max_model_forwards_per_window']} | {'PASS' if v['PASS'] else 'FAIL'} |")
        elif k.startswith("C_selection_GT"):
            out.append(f"| selection-time GT == captured GT | max abs diff {v['max_abs_diff_m']:.1e} m, validity mismatches {v['valid_mismatch']} (n={v['n']}) | {'PASS' if v['PASS'] else 'FAIL'} |")
        elif k.startswith("C_turn"):
            out.append(f"| GT-turn windows captured == selected | {v['captured_GT_turn']} == {v['selected_turn']} | {'PASS' if v['PASS'] else 'FAIL'} |")
        elif k.startswith("C_windows"):
            out.append(f"| seed-0 / seed-1 / window-list windows identical | n={v['n']} | {'PASS' if v['PASS'] else 'FAIL'} |")
    cap = R["capture"]
    wl = cap["window_list"]
    out.append("\n## capture\n")
    out.append(f"* {cap['n_windows']} windows over {cap['n_episodes']} episodes: **{wl['n_turn_windows']} GT-turn windows (every one of the {wl['n_windows_total_eval']} eval windows with |terminal heading| >= 30 deg)** "
               f"+ {wl['n_rest_sampled']} seeded (seed 0) random others. Totals over all eval windows {wl['class_counts_total']}; selected {wl['class_counts_selected']}.")
    for s in ("seed0", "seed1"):
        cp = cap[f"compute_{s}"]
        out.append(f"* {s}: GPU forward {cp['gpu_forward_s']} s, loop {cp['loop_s']} s, wall {cp['wall_s']} s, peak {cp['cuda_max_memory_allocated_gib']} GiB ({cp['device']})")
    out.append("\n## window counts (dense set)\n")
    out.append("| class | n | nav_ann active | nav_tl active | clip token active | nav_ann differs from nav_tl | nav_ann == GT dir | nav_ann opposite of GT |\n|---|---|---|---|---|---|---|---|")
    for k, v in R["window_counts_dense"].items():
        out.append(f"| {k} | {v['n']} | {v['nav_ann_active']} | {v['nav_tl_active']} | {v['clip_token_active']} | {v['nav_ann_differs_from_nav_tl']} | {v['nav_ann_eq_GT_dir']} | {v['nav_ann_opposite_of_GT']} |")
    out += bar_rows(L0, L1, R["bar_dense"], "THE REGISTERED SCORING: bar on the dense capture (paired episode-cluster bootstrap vs V0, B = 2000; seed 1 = sampler replicate)")
    f = R["B1t_fraction"]
    out.append(f"\nB1t fraction: dADE_turn(T3a) {f['T3a_dADE_turn']:+.4f} / B1t (recomputed on this capture) {f['B1t_dADE_turn_recomputed']:+.4f} = **{f['fraction_recomputed']:.3f}** "
               f"(per-draw ratio 95 % {f['per_draw_ratio_ci95']}); against A5's registered -0.566: {f['fraction_of_A5_registered_B1t_-0.566']:.3f}; T3 {f['T3_fraction_recomputed']:.3f}, T2a {f['T2a_fraction_recomputed']:.3f}.")
    out.append(f"\n**Reading rule: {R['reading_rule']['branch']}** -- {R['reading_rule']['text']}")
    cols = [("ade", "ADE"), ("fde", "FDE"), ("along_abs_6s", "LON |along| 6 s"), ("along_signed_6s", "LON along 6 s signed"), ("speed_mae_0_2s", "LON speed MAE 0-2 s"),
            ("cross_abs_6s", "LAT |cross| 6 s"), ("heading_mae_0_2s_deg", "LAT heading MAE 0-2 s deg"), ("curv_mae_0_2s", "LAT curv MAE 0-2 s"),
            ("term_heading_err_deg", "LAT term. heading err deg"), ("dir_correct", "TAC dir correct"), ("nav_complies", "STR nav compliance (vs the CLIP token)")]
    out.append("\n## four families (SPEC 2.1 format), dense set, sampler seed 0\n")
    for cls, lab in (("turn", "GT-turn windows"), ("straight", "GT-straight windows"), ("all", "all classified windows (turn-enriched)")):
        n = L0["V0"][cls]["n"]
        out.append(f"\n**{lab} (n={n})**\n")
        out.append("| arm | " + " | ".join(nm for _, nm in cols) + " |\n|---|" + "---|" * len(cols))
        for k in ("V0", "T3a", "T3", "T2a", "T3a-c", "B1t", "ORACLE"):
            out.append(f"| {k} | " + " | ".join(mean(L0, k, cls, m) for m, _ in cols) + " |")
    out.append("\n**TACTICAL (`four_families.tactical_from_trajectory`, 0-2 s, all dense windows)**\n")
    out.append("| arm | lateral accuracy / kappa | longitudinal accuracy / kappa |\n|---|---|---|")
    for k, v in R["tactical_from_trajectory_dense_seed0"].items():
        out.append(f"| {k} | {v['lateral_decision']['accuracy']:.4f} / {v['lateral_decision']['kappa']:.4f} | {v['longitudinal_decision']['accuracy']:.4f} / {v['longitudinal_decision']['kappa']:.4f} |")
    out.append("\n*LONGITUDINAL distance keeping UNAVAILABLE (no lead tracks); STRATEGIC decision UNAVAILABLE (`--no-strategic`).*")
    sub = R["subset_A5_windows"]
    out += bar_rows(sub["seed0"], sub["seed1"], sub["bar"], f"A5's EVAL-grid windows as a SUBSET of the dense capture (n={sub['n_windows']}; classes {sub['class_counts']})")
    ph = R["POSTHOC"]
    out.append("\n## POST-HOC (descriptive; NOT part of the bar or the reading rule)\n")
    out.append(f"Natural-frequency weights for the non-turn classes: {ph['natural_frequency_weights_(non-turn_classes)']} (turn windows weight 1: all of them are in the capture).\n")
    out.append("| arm | all-window dADE re-weighted to natural frequency, seed 0 | seed 1 | both draws pooled: turn | straight | all (turn-enriched) |\n|---|---|---|---|---|---|")
    for k in ("T3a", "T3", "T2a", "T3a-c"):
        a0, a1 = ph["all_window_dADE_natural_frequency_reweighted_seed0"][k], ph["all_window_dADE_natural_frequency_reweighted_seed1"][k]
        p = ph["pooled_two_draws_dADE"][k]
        out.append(f"| {k} | {a0['mean']:+.4f}{ci(a0['ci95'], 4)} | {a1['mean']:+.4f}{ci(a1['ci95'], 4)} | " + " | ".join(f"{p[cn]['mean']:+.3f}{ci(p[cn]['ci95'])}" for cn in ("turn", "straight", "all")) + " |")
    out.append("\n| arm | GT-turn windows | picks changed | improved | worse | episodes with a change | sum dADE over changed | sum of 5 largest gains |\n|---|---|---|---|---|---|---|---|")
    for k in ("T3a", "T3"):
        v = ph[f"{k}_turn_windows_seed0"]
        out.append(f"| {k} | {v['n_turn_windows']} | {v['n_pick_changed']} | {v['n_changed_improved']} | {v['n_changed_worse']} | {v['n_episodes_with_a_changed_turn_window']} | {v['sum_dADE_over_changed']} | {v['sum_of_5_largest_gains']} |")
    out.append("\n**True announced series vs donor series, paired ADE (T3a - T3a-c; negative = the true series is better)**\n")
    out.append("| draw | turn | straight | all |\n|---|---|---|---|")
    for tag, v in ph["paired_T3a_minus_T3a-c_ADE_(negative = the true series is better)"].items():
        out.append(f"| {tag} | " + " | ".join(f"{v[cn]['mean']:+.3f}{ci(v[cn]['ci95'])}" for cn in ("turn", "straight", "all")) + " |")
    out.append("\n**Across episodes (seed 0, GT-turn windows)**\n")
    out.append("| arm | turn episodes | episodes dADE < 0 | > 0 | unchanged | leave-one-episode-out turn dADE (min, max) | largest single-episode share of the gain |\n|---|---|---|---|---|---|---|")
    for k in ("T3a", "T2a"):
        v = ph[f"{k}_per_episode_turn_dADE_seed0"]
        out.append(f"| {k} | {v['n_turn_episodes']} | {v['n_episodes_dADE_lt_0']} | {v['n_episodes_dADE_gt_0']} | {v['n_episodes_unchanged']} | {v['leave_one_episode_out_turn_dADE_min_max']} | {v['largest_single_episode_share_of_total_gain']} |")
    out.append(f"\nT3a vs T3 picks differ on {ph['T3a_vs_T3_picks_differ']['windows']} windows ({ph['T3a_vs_T3_picks_differ']['GT_turn_windows']} GT-turn).")
    print("\n".join(out))


if __name__ == "__main__":
    main()
