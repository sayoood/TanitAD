"""Markdown tables for RESULT_A5.md, read straight from raw/a5_nav_tl.json (no number is typed by hand).
Usage: python make_tables_a5.py <raw dir>   -> prints markdown (redirect to raw/TABLES_a5_generated.md)"""
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


def main():
    R = json.loads((Path(sys.argv[1]) / "a5_nav_tl.json").read_text(encoding="utf-8"))
    L0, L1 = R["eval_s0g"], R["eval_s1"]
    out = []
    c = R["controls"]
    out.append("## controls\n")
    k1 = c["K1_clock_vs_replay_bank_t_label_s"]
    k2 = c["K2_nav_tl_eq_clip_token_at_t_rel_0_captured_windows"]
    k2d = c["K2_supplementary_dense_one_t_per_eval_clip"]
    out.append("| control | expected | measured | verdict |\n|---|---|---|---|")
    out.append(f"| **K1** computed `t_now_raw` vs the replay bank's `t_label_s` | max abs diff <= 1e-3 s on every window | "
               f"**{k1['max_abs_diff_s']:.2e} s** over {k1['n_windows']} windows / {k1['n_clips']} clips (the bank rounds to 4 dp, so a correct "
               f"clock reads <= 5e-5); W re-derived from the bank = {k1['W_derived_from_bank']} | {'PASS' if k1['PASS'] else 'FAIL'} |")
    out.append(f"| **K2** nav_tl == clip token's side at t_rel in [-0.05, 0.05], `args.time_s` <= 6.0 (captured windows) | 100 % | "
               f"**{k2['n_windows_total_captured'] - k2['n_unequal_total_captured']}/{k2['n_windows_total_captured']}** "
               f"(EVAL grid n={k2['eval_s0g_grid']['n_windows']}, TRAIN grid n={k2['train_s0_grid']['n_windows']}, reel n={k2['reel_s0_every_window']['n_windows']}) | "
               f"{'PASS' if k2['PASS'] else 'FAIL'} (the EVAL grid itself hits the band 0 times: VACUOUS there) |")
    out.append(f"| K2 supplementary: one integer t per eval clip at t_rel ~ 0 (not a captured window) | 100 % | "
               f"**{k2d['n_clips_with_t_in_band_and_time_s_le_6'] - k2d['n_unequal']}/{k2d['n_clips_with_t_in_band_and_time_s_le_6']}** | {'PASS' if k2d['PASS'] else 'FAIL'} |")
    for tag in ("eval_s0g", "eval_s1", "train_s0"):
        k = c[f"K0_join_bank_nav_eq_record_clip_token_{tag}"]
        out.append(f"| K0 join: bank `nav` == record `nav_command` side, {tag} | 100 % | {k['n_equal']}/{k['n']} | {'PASS' if k['PASS'] else 'FAIL'} |")
    k0 = c["K0_sidecar_coverage"]
    out.append(f"| K0 sidecar coverage | the 3 eval clips with no row are the config's 3 `tactical_excluded_sids`; 0 train-diag clips uncovered | "
               f"{k0['eval_clips_without_sidecar_row']} uncovered, sids equal = {k0['sids_equal_the_3_config_tactical_excluded_sids']}; train-diag uncovered {k0['train_diag_clips_without_sidecar_row']} | {'PASS' if k0['PASS'] else 'FAIL'} |")
    rp = c["reproduction_with_clip_token"]
    for k, v in rp.items():
        if "max_abs_diff" in v:
            out.append(f"| reproduction: {k} | max abs diff <= 2e-4 | {v['max_abs_diff']:.1e} | {'PASS' if v['PASS'] else 'FAIL'} |")
    v = rp["X1_fit_alpha_equal_and_cv_diff"]
    out.append(f"| reproduction: A2 X1 refit (alpha, 5-fold CV ADE) | same alpha, CV diff <= 5e-4 | alpha equal {v['alpha_equal']}, CV diff {v['cv_ade_max_abs_diff']:.1e} | {'PASS' if v['PASS'] else 'FAIL'} |")
    v = rp["navc_term_recomputed_from_record_clip_token_eq_shipped_term_max_abs"]
    out.append(f"| recomputed navc term (clip token) == shipped term | <= 1e-9 | eval {v['eval_s0g']:.1e} / train {v['train_s0']:.1e} | {'PASS' if v['PASS'] else 'FAIL'} |")

    t = R["tally_eval_s0g"]
    out.append("\n## token tally (139 eval clips) and window counts (EVAL grid, 1,112 windows)\n")
    cl = t["clips"]
    out.append(f"* clip token (`nav_command`): {cl['clip_token']}; entry tokens (`nav_30s.entries`): {cl['entry_tokens']}; entries per clip: {cl['n_entries_per_clip']}")
    dis = cl["nav_command_vs_nav_30s_first_entry_disagree"]
    out.append(f"* `nav_command` vs `nav_30s.entries[0]` disagree on {dis['n_clips']} clips: {dis['cases']}")
    out.append(f"* window reason (all {sum(t['window_reason'].values())} windows): {t['window_reason']}; t_rel range {t['t_rel_s']}; clock source {t['clock_source_windows']}")
    out.append("\n| window class | n | nav_tl L / follow / R | clip token L / follow / R | nav_tl non-follow | clip-token non-follow | H=8 non-follow | clip->nav_tl (L->L, L->F, F->L, F->R, R->F, R->R) |\n|---|---|---|---|---|---|---|---|")
    for cn in ("turn", "turnL", "turnR", "straight", "gentle", "unclassified", "every_window"):
        b = t["window_counts"][cn]
        x = b["crosstab_clip_token_x_nav_tl"]
        out.append(f"| {cn} | {b['n']} | {b['nav_tl_side']['left']} / {b['nav_tl_side']['follow']} / {b['nav_tl_side']['right']} | "
                   f"{b['clip_token_side']['left']} / {b['clip_token_side']['follow']} / {b['clip_token_side']['right']} | "
                   f"{b['nav_tl_non_follow']} | {b['clip_token_non_follow']} | {b['nav_tl_H8_non_follow']} | "
                   f"{x['left->left']}, {x['left->follow']}, {x['follow->left']}, {x['follow->right']}, {x['right->follow']}, {x['right->right']} |")
    out.append("\nnav_tl vs the geometric GT direction (descriptive; `straight` rows count follow == straight):\n")
    out.append("| class | n | nav_tl == GT dir | nav_tl follow | nav_tl opposite | clip token == GT dir | clip token follow | clip token opposite |\n|---|---|---|---|---|---|---|---|")
    for cn in ("turn", "turnL", "turnR", "straight", "gentle"):
        b = t["window_counts"][cn]["descriptive_vs_GT_direction"]
        out.append(f"| {cn} | {t['window_counts'][cn]['n']} | {b['nav_tl_eq_GT_dir']} | {b['nav_tl_follow']} | {b['nav_tl_opposite_of_GT']} | "
                   f"{b['clip_token_eq_GT_dir']} | {b['clip_token_follow']} | {b['clip_token_opposite_of_GT']} |")
    out.append("\n## bar (EVAL, paired episode-cluster bootstrap vs V0, B = 2000, the SPEC sec. 5 draws)\n")
    out.append("| arm | picks changed | turn dADE [CI] | turn dir-correct d [CI] | straight dADE [CI] | all dADE [CI] | seed 1: turn dADE [CI] | seed 1: turn dir-correct d [CI] | criteria 1/2/3/4 | bar |\n|---|---|---|---|---|---|---|---|---|---|")
    names = {"T2": "**T2 (REPORTED)**", "T3": "T3 (a: term replaced)", "T3b": "T3b (b: term added, sensitivity)", "T4": "T4 refit re-scorer",
             "T2c": "T2c CONTROL (derangement)", "T2h8": "T2h8 H=8 (sensitivity)"}
    for k in ("T2", "T3", "T3b", "T4", "T2c", "T2h8"):
        b = R["bar"][k]
        cr = "/".join("Y" if b[x] else "N" for x in ("1_turn_ade_and_dircorrect", "2_straight_no_regression", "3_all_ade_not_worse", "4_replicates_seed1"))
        out.append(f"| {names[k]} | {L0[k]['pick_changed_frac']:.3f} | {d(L0, k, 'turn')} | {d(L0, k, 'turn', 'dir_correct')} | {d(L0, k, 'straight')} | {d(L0, k, 'all')} | "
                   f"{d(L1, k, 'turn')} | {d(L1, k, 'turn', 'dir_correct')} | {cr} | **{'CLEARS' if b['CLEARS'] else 'FAILED'}** |")
    for k in ("B1t", "ORACLE"):
        out.append(f"| {k} (label-side bound, not a lever) | {L0[k]['pick_changed_frac']:.3f} | {d(L0, k, 'turn')} | {d(L0, k, 'turn', 'dir_correct')} | {d(L0, k, 'straight')} | {d(L0, k, 'all')} | "
                   f"{d(L1, k, 'turn')} | {d(L1, k, 'turn', 'dir_correct')} | – | bound |")
    f = R["T2_fraction_of_B1t"]
    out.append(f"\nT2 fraction of the A4 B1t bound: dADE_turn(T2) {f['T2_dADE_turn']:+.4f} / ({f['B1t_bound_registered']}) = **{f['fraction_of_registered_bound']:.3f}** "
               f"(B1t recomputed here {f['B1t_bound_recomputed_here']:+.4f}); per-draw ratio bootstrap 95 % {f['fraction_of_B1t_bootstrap_ci95_(recomputed_B1t_ratio_per_draw)']}")
    out.append("\n## four families (SPEC 2.1 format), EVAL seed 0 (eval_s0g)\n")
    cols = [("ade", "ADE"), ("fde", "FDE"), ("along_abs_6s", "LON |along| 6 s"), ("along_signed_6s", "LON along 6 s signed"), ("speed_mae_0_2s", "LON speed MAE 0-2 s"),
            ("cross_abs_6s", "LAT |cross| 6 s"), ("heading_mae_0_2s_deg", "LAT heading MAE 0-2 s deg"), ("curv_mae_0_2s", "LAT curv MAE 0-2 s"),
            ("term_heading_err_deg", "LAT term. heading err deg"), ("dir_correct", "TAC dir correct"), ("nav_complies", "STR nav compliance (vs the CLIP token)")]
    for cls, lab in (("turn", "GT-turn windows (107)"), ("straight", "GT-straight windows (588)"), ("all", "all classified windows (800)")):
        out.append(f"\n**{lab}**\n")
        out.append("| arm | " + " | ".join(n for _, n in cols) + " |\n|---|" + "---|" * len(cols))
        for k in ("V0", "T2", "T3", "T4", "T2c", "T2h8", "B1t", "ORACLE"):
            out.append(f"| {k} | " + " | ".join(mean(L0, k, cls, m) for m, _ in cols) + " |")
    out.append("\n**TACTICAL, `four_families.tactical_from_trajectory` (0-2 s, all 1,112 windows)**\n")
    out.append("| arm | lateral accuracy / kappa | longitudinal accuracy / kappa |\n|---|---|---|")
    for k, v in R["tactical_from_trajectory_eval_s0g"].items():
        out.append(f"| {k} | {v['lateral_decision']['accuracy']:.4f} / {v['lateral_decision']['kappa']:.4f} | {v['longitudinal_decision']['accuracy']:.4f} / {v['longitudinal_decision']['kappa']:.4f} |")
    out.append("\n*Coverage: LONGITUDINAL distance keeping UNAVAILABLE (no lead tracks); STRATEGIC decision UNAVAILABLE (`--no-strategic`).*")
    out.append("\n## T4 fit (TRAIN, 5-fold episode-grouped CV; V0 TRAIN ADE in the row)\n")
    ft = R["fit_T4"]
    fx = R["fit_X1_reproduction"]["banked"]
    out.append(f"* T4 (nav_tl features): alpha {ft['alpha']}, CV ADE by alpha {ft['cv_ade_by_alpha']}, in-sample {ft['train_in_sample_ade']}, V0 {ft['train_V0_ade']}; nav weights navc {ft['theta']['navc']}, agree_nav_side {ft['theta']['agree_nav_side']}")
    out.append(f"* A2 X1 (clip token, banked): alpha {fx['alpha']}, CV ADE by alpha {fx['cv_ade_by_alpha']}, in-sample {fx['train_in_sample_ade']}")
    ph = R["POSTHOC"]
    out.append("\n## POST-HOC (descriptive; NOT part of the bar or the reading rule)\n")
    out.append("**Both sampler draws pooled** (per-window dADE averaged over seeds 0 and 1):\n")
    out.append("| arm | turn | straight | all |\n|---|---|---|---|")
    for k, v in ph["pooled_over_the_two_sampler_draws_dADE"].items():
        out.append(f"| {k} | " + " | ".join(f"{v[cn]['mean']:+.3f}{ci(v[cn]['ci95'])}" for cn in ("turn", "straight", "all")) + " |")
    out.append("\n**True series vs donor series, paired ADE (T2 - T2c; negative = the true series is better)**\n")
    out.append("| draw | turn | straight | all |\n|---|---|---|---|")
    for tag, v in ph["paired_T2_minus_T2c_ADE_(negative = the true series is better)"].items():
        out.append(f"| {tag} | " + " | ".join(f"{v[cn]['mean']:+.3f}{ci(v[cn]['ci95'])}" for cn in ("turn", "straight", "all")) + " |")
    out.append("\n**Where the GT-turn gain sits (seed 0)**\n")
    out.append("| arm | GT-turn windows | picks changed | improved | worse | episodes with a change | sum dADE over changed | sum of the 5 largest gains |\n|---|---|---|---|---|---|---|---|")
    for k in ("T2", "T3"):
        v = ph[f"{k}_turn_windows_seed0"]
        out.append(f"| {k} | {v['n_turn_windows']} | {v['n_pick_changed']} | {v['n_changed_improved']} | {v['n_changed_worse']} | {v['n_episodes_with_a_changed_turn_window']} | {v['sum_dADE_over_changed']} | {v['sum_of_5_largest_gains']} |")
    cv = ph["nav_coverage_of_GT_turn_windows"]
    out.append(f"\n**How many GT-turn windows can a nav signal touch?** GT-turn {cv['GT_turn_windows']}: nav_tl active on {cv['nav_tl_active_on_GT_turn']}, silent (follow) on {cv['GT_turn_windows_where_nav_tl_is_follow']}; clip token active on {cv['clip_token_active_on_GT_turn']}.\n")
    out.append("| arm | nav_tl-active GT-turn (n=52) seed 0 | seed 1 | nav_tl-silent GT-turn (n=55) seed 0 | seed 1 |\n|---|---|---|---|---|")
    for k in ("T2", "T3", "B1t"):
        a, s = cv[f"{k}_dADE_by_nav_tl_coverage"]["nav_tl_active_GT_turn"], cv[f"{k}_dADE_by_nav_tl_coverage"]["nav_tl_silent_GT_turn"]
        out.append(f"| {k} dADE | {a['seed0']['mean']:+.3f}{ci(a['seed0']['ci95'])} | {a['seed1']['mean']:+.3f}{ci(a['seed1']['ci95'])} | "
                   f"{s['seed0']['mean']:+.3f}{ci(s['seed0']['ci95'])} | {s['seed1']['mean']:+.3f}{ci(s['seed1']['ci95'])} |")
    out.append("\n**Pure timing vs extra information: GT-turn windows (107) partitioned by the shipped clip token and nav_tl** "
               "(contribution = sum of per-window dADE / 107, i.e. what the part adds to the GT-turn mean)\n")
    out.append("| part | n | T2 seed 0: sum dADE / contribution | T2 seed 1 | T3 seed 0 | T3 seed 1 |\n|---|---|---|---|---|---|")
    for nm, v in ph["GT_turn_partition_clip_token_vs_nav_tl"].items():
        out.append(f"| {nm} | {v['n']} | " + " | ".join(f"{v[f'{a}_{t}_sum_dADE']:+.2f} / {v[f'{a}_{t}_contribution_to_107_window_mean']:+.3f}"
                                                        for a in ("T2", "T3") for t in ("eval_s0g", "eval_s1")) + " |")
    out.append(f"\nT2c vs T2 series: {ph['T2c_vs_T2_series']}")
    td = R["POSTHOC_train_diag"]
    out.append(f"\n**Second sample for the RULE: the TRAIN-DIAG fans ({td['n_episodes']} other episodes; the checkpoint was TRAINED on them: NOT held-out, no bar verdict)** "
               f"classes {td['class_counts']}; nav_tl active on {td['nav_tl_non_follow_windows']} of 1,112 windows\n")
    out.append("| arm | picks changed | turn dADE [CI] | turn dir-correct d [CI] | straight dADE [CI] | all dADE [CI] |\n|---|---|---|---|---|---|")
    for k, v in td["arms"].items():
        if k == "V0":
            continue
        def _d(c, m):
            e = v[c][m]
            return "–" if e.get("delta_vs_V0") is None else f"{e['delta_vs_V0']:+.3f}{ci(e['delta_ci95'])}"
        out.append(f"| {k} | {v['pick_changed_frac']:.3f} | {_d('turn', 'ade')} | {_d('turn', 'dir_correct')} | {_d('straight', 'ade')} | {_d('all', 'ade')} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
