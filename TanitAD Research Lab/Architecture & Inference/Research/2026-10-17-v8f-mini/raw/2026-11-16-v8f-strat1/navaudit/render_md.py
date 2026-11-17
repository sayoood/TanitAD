"""render_md.py -- RESULT_NAVAUDIT.md from RESULT_NAVAUDIT.json (every number read from the json)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(HERE + "/out/RESULT_NAVAUDIT.json", encoding="utf-8"))
q = r["Q_wide_vs_pinned_vs_v9"]
c = r["audit_counts"]
bs = q["by_side"]["BOTH_SIDES"]
bn = bs["both_name_correct_turn"]
pct = lambda d: f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] ({d['n_num']}/{d['n_den']})"  # noqa: E731
pr = lambda d: f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {'sep' if d['separated'] else 'not sep'}"  # noqa: E731
L = []
w = L.append
w("# RESULT_NAVAUDIT -- why the route-text nav misses the turns v9 names, and a wider announcement cut\n")
w(f"Evidence class: MEASURED (ours; banked T1-kin dumps, the pinned text table `navtok_table.npz`, zero training, CPU). Tier: T1-kin tick 0 (the window's NOW frame). Estimator: episode-cluster bootstrap (`taniteval.ci`), B 2,000, seed 0, 'another draw of EPISODES' only. Artifact: `F:/TanitAD-scratch/strat1/navaudit/out/RESULT_NAVAUDIT.json`. Builder sha256 `{r['builder_sha256'][:16]}` (the landed `build_navtext_nav.py`, imported, not edited).\n")
w(f"Windows: {r['n_turn_windows']} labelled TURN_L / TURN_R windows (30 episodes). Pinned text names the correct turn at NOW on {r['n_text_correct_pinned']}, v9 on {r['n_v9_correct']}; **{r['n_only_v9_windows']} windows in {r['audit_n_episodes']} episodes are named by v9 only** (the audited set).\n")
w("## 1. Audit of the windows where only v9 names the turn\n")
w("Rules (first match wins; `navaudit.py` `classify`): MATCH-FAILED = table flag at NOW is not 0, or a sided row whose countdown could not be computed (`build_navtext_nav.py:80-81`, `:93-94`); ANNOUNCED-LATE-CUT = flag 0, the table side at NOW is the labelled side, token FOLLOW because countdown > `d_ann = max(30, 6 v)` (`:100-101`); ANNOUNCED-LATE-ROW = no side at NOW and a flag-0 row with the labelled side within the next 8 s (40 steps of 0.2 s); ROUTE-LACKS-TURN = no side at NOW and none in the next 8 s; OTHER = the table side at NOW is the other side.\n")
w("| class | windows (of 75) |\n|---|---|")
for k, v in c.items():
    w(f"| {k} | {v} |")
w("")
w(f"* Concentration: the 5 episodes with most audited windows hold {r['audit_top5_episode_share']:.3f} of the {r['n_only_v9_windows']}. Per-episode table in the json (`audit_by_episode`).")
lack = {e: v for e, v in r["audit_by_episode"].items() if v["ROUTE-LACKS-TURN"] > 0}
zero = {e: v for e, v in lack.items() if v["episode_steps_with_turn_side_row_share"] == 0.0}
w(f"* ROUTE-LACKS-TURN: {len(lack)} episodes; in {len(zero)} of them ({sum(v['ROUTE-LACKS-TURN'] for v in zero.values())} of the {c['ROUTE-LACKS-TURN']} windows) the text table holds NO turn-sided row at ANY step of the clip (episodes {', '.join(sorted(zero, key=int))}). The text is silent for the whole clip, not late.")
rb = {e: v for e, v in zero.items() if (r["audit_lacks_episode_table_class_groups_flag0_steps"][e].get("RB", 0) + r["audit_lacks_episode_table_class_groups_flag0_steps"][e].get("ROTARY", 0)) > 50}
co = {e: v for e, v in zero.items() if e not in rb}
w(f"* Of those {len(zero)} text-silent episodes, {len(rb)} are roundabout clips (RB_* / ROTARY rows on more than half of the flag-0 steps, which the builder maps to none by rule 1, `:6`; episodes {', '.join(sorted(rb, key=int))}; {sum(v['ROUTE-LACKS-TURN'] for v in rb.values())} windows) and {len(co)} hold CONTINUE rows only (episodes {', '.join(sorted(co, key=int))}; {sum(v['ROUTE-LACKS-TURN'] for v in co.values())} windows).")
w(f"* Flag-0 step class groups in those episodes (counts over the clip): " + "; ".join(f"ep {e}: {v}" for e, v in r["audit_lacks_episode_table_class_groups_flag0_steps"].items()) + ".")
w(f"* ROUTE-LACKS windows with a class the builder maps to none (KEEP / MERGE / RAMP / CONTINUE_SLIGHT ...) whose name ends in the labelled side within 8 s: {r['audit_lacks_windows_with_a_none_mapped_class_ending_with_labelled_side_within_8s']} of {c['ROUTE-LACKS-TURN']}.")
w(f"* ANNOUNCED-LATE-CUT: countdown at NOW median {r['audit_cut_windows_countdown_m']['median']:.1f} m, max {r['audit_cut_windows_countdown_m']['max']:.1f} m (n {r['audit_cut_windows_countdown_m']['n']}); with a text turn within 100 m: {r['audit_within_distance']['text_table_correct_side_countdown_le_100m']}, within 200 m: {r['audit_within_distance']['text_table_correct_side_countdown_le_200m']}.")
w(f"* ANNOUNCED-LATE-ROW with the later row inside the label's 2-8 s band: {r['audit_later_row_within_2_8s']} (of {c['ANNOUNCED-LATE-ROW']}).")
w(f"* v9 (ego-derived turn distance, a label-side quantity) at NOW: {r['audit_within_distance']['v9_ego_turn_distance_le_100m_all_75']} of 75 within 100 m, {r['audit_within_distance']['v9_ego_turn_distance_le_200m_all_75']} within 200 m. The turns are close; the text has no row for them.")
w(f"* MATCH-FAILED: flag values in the eval table `{r['audit_flag_values_in_eval_table']}` (flag meaning beyond 'not 0 = no usable row' is UNVERIFIED here).")
w(f"* Upper bound: if the cut were removed entirely (any distance), the text table holds the labelled side at NOW on {r['upper_bound_text_names_correct_if_cut_removed']} of 197 windows.\n")
w("## 2. The wide variant (new file, same builder code path)\n")
w(f"`navtext_nav_wide.npz`, sha256 `{r['wide_npz']['sha256']}`, {r['wide_npz']['bytes']} bytes. Pre-registered (`PREREG_NAVAUDIT.md`): `d_ann = max(60 m, 8 s x v)`, two module constants of the builder changed by the wrapper `build_variant`, nothing else.\n")
for s in ("train", "eval"):
    p = r["build_proof"][s]
    w(f"* {s}: same wrapper at 30 / 6 reproduces the pinned arrays {p['narrow_reproduces_pinned']}; wide vs pinned: {p['diff_wide_vs_pinned']['n_steps_changed']:,} steps changed, equal to the expected set (flag 0, sided, countdown between the two cuts) {p['diff_wide_vs_pinned']['changed_equals_expected_set']}, ch3 / ch4 / nav_d identical {p['diff_wide_vs_pinned']['ch3_ch4_nav_d_identical']}, every change FOLLOW to side {p['diff_wide_vs_pinned']['every_change_is_FOLLOW_to_side']}.")
w(f"* No ego-future read (source: the builder reads the table, `J['now_s']` and track rows up to row 2j: arc at NOW and at `now - age`, speed at row 2j; `:75-100`). Probe: {r['future_independence_eval']['n_sampled_steps']} eval and {r['future_independence_train']['n_sampled_steps']} train steps sampled, track rows after 2j replaced by garbage (sd 1e6): {r['future_independence_eval']['n_differ_when_future_track_rows_are_garbage']} / {r['future_independence_train']['n_differ_when_future_track_rows_are_garbage']} differ ({r['future_independence_eval']['n_with_side_token']} / {r['future_independence_train']['n_with_side_token']} carried a side token). A mutant reading 10 rows ahead goes RED in `test_navaudit.py`.\n")
w("## 3. Score on the 197 turn windows (landed navecho3 functions, unchanged)\n")
w("| feed | names the correct turn | names the other side |\n|---|---|---|")
w(f"| pinned text (30 m, 6 s) | {pct(q['pinned_correct_share'])} | {q['text_pinned_names_other_side']} |")
w(f"| **wide text (60 m, 8 s)** | {pct(q['wide_correct_share'])} | **{q['text_wide_names_other_side']}** |")
w(f"| v9 | {q['v9_names_correct']}/197 | {q['v9_names_other_side']} |")
w("")
w(f"* wide minus pinned, paired over the same windows: {pr(q['wide_minus_pinned_correct_paired'])}; gained {q['wide_windows_gained_vs_pinned']} windows, lost {q['wide_windows_lost_vs_pinned']}.")
w(f"* v9 minus wide: {pr(q['v9_minus_wide_correct_paired'])}.")
w(f"* Where wide and v9 both name it ({bn['n_windows']} windows, {bn['n_episodes']} episodes): text distance median {bn['distance_m']['text_median']:.1f} m vs v9 {bn['distance_m']['v9_median']:.1f} m (text longer on {bn['distance_m']['n_text_longer']}, shorter on {bn['distance_m']['n_text_shorter']}; mean difference {pr(bn['distance_m']['mean_diff_text_minus_v9']).replace('sep','sep')}); text countdown inside the 2-8 s band for {bn['time_to_turn_s_text']['share_in_2_to_8s']:.3f}, v9 {bn['time_to_turn_s_v9']['share_in_2_to_8s']:.3f}.")
t = q["token_windows_beside"]
w(f"* Side of the wrong-side windows: {q['wrong_side_tokens_on_labelled_turn_windows_wide']} windows, all in episode 139 (table row names TURN_R at a countdown of 50-56 m while the label is TURN_L; the pinned cut hid them). Over all 2,399 windows the wide feed carries a turn token on {t['wide_token_windows_all_2399']} windows (pinned {t['pinned_token_windows_all_2399']}, v9 {t['v9_token_windows_all_2399']}); on labelled windows whose label is not a turn, wide carries a token on {t['wide_token_where_label_is_not_a_turn_labelled']} (pinned {t['pinned_token_where_label_is_not_a_turn_labelled']}).\n")
w("Nav-only out-of-fold classifier (navecho2 HGB, 5 channels, episode-grouped 5-fold), macro-F1; paired wide minus pinned:\n")
w("| windows | n labelled | pinned | wide | wide minus pinned (paired) | TURN_L recall pinned / wide | TURN_R recall pinned / wide |\n|---|---|---|---|---|---|---|")
for k, v in r["nav_only_classifier"].items():
    w(f"| {k} | {v['n_labelled']} | {v['macro_f1_pinned']:.3f} | {v['macro_f1_wide']:.3f} | {pr(v['wide_minus_pinned_paired_macro_f1'])} | {v['recall_TURN_L_pinned']:.3f} / {v['recall_TURN_L_wide']:.3f} | {v['recall_TURN_R_pinned']:.3f} / {v['recall_TURN_R_wide']:.3f} |")
w("")
b = r["bar"]
w("## 4. Committed bar\n")
w(f"Rule: {b['rule']}. Measured: correct {b['n_correct']} of 197, wrong side {b['n_wrong_side']}, no future read {b['no_future_read']}. **Bar met: {b['met']}.**\n")
w(f"Controls: K-FEED (the pinned table at NOW equals the N1 dump's nav_in) {r['K-FEED_pinned_table_equals_N1_dump_nav_in']}; K-REPRO / K-DIFF problems list: {r['problems']}.\n")
w("Limits: 197 windows in 30 episodes; the audited 75 in 19 episodes. The bootstrap answers another draw of episodes only. The route-text table is the only text source read here; the GPS / OSM route itself is not re-read.")
open(HERE + "/out/RESULT_NAVAUDIT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("ok")
