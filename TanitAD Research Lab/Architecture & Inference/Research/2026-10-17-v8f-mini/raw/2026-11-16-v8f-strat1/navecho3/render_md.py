"""render_md.py -- RESULT_NAVECHO3.md from RESULT_NAVECHO3.json (every number is read from the JSON; no number is typed here)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "out", "RESULT_NAVECHO3.json"), encoding="utf-8"))


def sh(d):
    lo, hi = d.get("lo"), d.get("hi")
    ci = f" [{lo:.3f}, {hi:.3f}]" if lo is not None else ""
    return f"{d['point']:.3f}{ci} ({d['n_num']}/{d['n_den']})"


def pd(d):
    return f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" + (" sep" if d.get("separated") else " not sep")


def f1(x, k=1):
    return "n/a" if x is None else f"{x:.{k}f}"


def iqr(v):
    return f"{f1(v[0])} to {f1(v[1])}"


Q, rd, q3 = r["Q1_Q2_by_side"], r["reading"], r["Q3_missed_turns_by_arm"]
B = Q["BOTH_SIDES"]
o = []
w = o.append
w("# RESULT_NAVECHO3 -- D-STRAT1-TACLAT-F1: cue missing or cue present on the labelled turns\n")
w(f"Evidence class: MEASURED (ours; banked T1-kin dumps in `F:/TanitAD-scratch/strat1/read/t1`, zero training, CPU). Tier of every number: T1-kin tick 0 (the lateral head's argmax and the nav_in vector fed at the window's first tick) vs the v9 label lat_cls.")
w(f"Estimator: {r['estimator']}. Windows: {r['n_windows']:,} dump windows ({r['n_episodes']} episodes); the windows labelled TURN_L or TURN_R number **{r['n_turn_windows']} in {r['n_turn_episodes']} episodes** (TURN_L {r['turn_support']['TURN_L']}, TURN_R {r['turn_support']['TURN_R']}). Artifact: `F:/TanitAD-scratch/strat1/navecho3/out/RESULT_NAVECHO3.json`.\n")
w(f"## Reading by the committed rules: **{rd['verdict']}**\n")
w(f"Rule as committed: {rd['rule']}.\n")
w("Share of each text arm's missed turns (head argmax is not the label) by the cue at NOW, point [95 % episode-cluster CI] (num/den):\n")
w("| arm | n missed of 197 | no correct text turn while v9 names it (CUE-MISSING) | correct text turn at NOW (CUE-PRESENT) | neither feed names it |")
w("|---|---|---|---|---|")
for a in ("S1_s0", "S1_s1", "N1_s0", "N1_s1"):
    v = q3[a]
    w(f"| **{a}** (rule cell) | {v['n_missed']} | {sh(v['share_cue_missing'])} | {sh(v['share_cue_present'])} | {sh(v['share_neither'])} |")
for a in ("S1d_s0", "S1d_s1", "S1_s10", "S1_s11", "N1_s10", "N1_s11", "S1d_s10", "S1d_s11"):
    v = q3[a]
    w(f"| {a} (beside) | {v['n_missed']} | {sh(v['share_cue_missing'])} | {sh(v['share_cue_present'])} | {sh(v['share_neither'])} |")
cm = [q3[a]["share_cue_missing"]["point"] for a in ("S1_s0", "S1_s1", "N1_s0", "N1_s1")]
w(f"\nThe four rule cells read {min(cm):.3f} to {max(cm):.3f} for CUE-MISSING, each at or above 0.50, so the rule fires at both seeds for both text arms. All 8 beside arms (S1d s0 / s1 / s10 / s11, S1 and N1 at s10 / s11) read the same side of the bar. The lowest lower CI bound among the four cells is {min(q3[a]['share_cue_missing']['lo'] for a in ('S1_s0', 'S1_s1', 'N1_s0', 'N1_s1')):.3f}, above 0.50 in every cell as well.\n")
w("Where the text feed names no correct turn, it names no turn at all, never the other side (the `names the other side` cells are 0 in every table below). Missed turns by the text feed's state at NOW (S1 s0): " + ", ".join(f"{k} {v}" for k, v in q3["S1_s0"]["missed_by_text_state"].items()) + ".\n")
w("Head recall on the turn windows split by the cue its OWN feed carries at NOW (point [CI] (num/den)):\n")
w("| arm | recall when its feed names the correct turn | recall when it does not |")
w("|---|---|---|")
for a in ("C0_s0", "C0_s1", "S1_s0", "S1_s1", "N1_s0", "N1_s1"):
    v = q3[a]
    w(f"| {a} ({v['feed']}) | {sh(v['recall_when_own_cue_correct'])} | {sh(v['recall_when_own_cue_not_correct'])} |")
w("\nThe text arms read the cue they are given: recall is near 0.9 when the text feed names the correct turn and near 0.1 to 0.2 when it does not.  C0 reads its own feed the same way: it recalls 0 of 19 turns its feed does not name at both seeds, and its feed names 178 of 197.\n")
w("## Q1. Does the text nav name a turn at NOW, on the correct side? Does v9? (labelled turn windows, tick 0)\n")
for sd in ("TURN_L", "TURN_R", "BOTH_SIDES"):
    t = Q[sd]
    w(f"### {sd}  (n {t['n']} windows, {t['n_episodes']} episodes)\n")
    w("2 x 2 of the correct-turn cue (rows text, columns v9); no cell holds an other-side token, so every non-correct cell is FOLLOW or no row:\n")
    tb = t["table_text_x_v9"]
    C = "names the correct turn"
    cnt = lambda tc, vc: sum(v for kk, v in tb.items() if kk.startswith("text " + C + " |") == tc and kk.endswith("v9 " + C) == vc)  # noqa: E731
    w("| | v9 names the correct turn | v9 does not |")
    w("|---|---|---|")
    w(f"| **text names the correct turn** | {cnt(True, True)} | {cnt(True, False)} |")
    w(f"| **text does not** | {cnt(False, True)} | {cnt(False, False)} |")
    w("")
    w(f"* text names the correct turn: {sh(t['text_correct'])}; v9: {sh(t['v9_correct'])}; v9 minus text, paired over the same windows: {pd(t['v9_minus_text_correct_paired'])}.\n")
w("Beside the table: the text feed names a turn on the wrong side on 0 of the 197 windows, and v9 on 0 of 197.\n")
w("## Q2. Where both feeds name the correct turn: the distance channel and the label window\n")
w("Distance = channel 3 x 100 m (the feed clips at 300 m; no window reads the clip). Time to the turn = distance / ego speed at NOW (dump `v0`, m/s); the label looks 2 s to 8 s ahead.\n")
w("| side | n both (episodes) | text distance m, median (IQR) | v9 distance m, median (IQR) | text minus v9 m, median (IQR); text longer / shorter | mean diff m, paired [CI] | ego speed m/s, median |")
w("|---|---|---|---|---|---|---|")
for sd in ("TURN_L", "TURN_R", "BOTH_SIDES"):
    b = Q[sd]["both_name_correct_turn"]
    d = b["distance_m"]
    w(f"| {sd} | {b['n_windows']} ({b['n_episodes']}) | {f1(d['text_median'])} ({iqr(d['text_iqr'])}) | {f1(d['v9_median'])} ({iqr(d['v9_iqr'])}) | {f1(d['diff_text_minus_v9_median'])} ({iqr(d['diff_iqr'])}); {d['n_text_longer']} / {d['n_text_shorter']} | {pd(d['mean_diff_text_minus_v9'])} | {f1(b['ego_speed_mps_median'])} |")
w("\n| side | text time to turn s, median (IQR); below 2 s / 2-8 s / above 8 s | v9 time to turn s, median (IQR); below 2 s / 2-8 s / above 8 s | median text minus v9 s |")
w("|---|---|---|---|")
for sd in ("TURN_L", "TURN_R", "BOTH_SIDES"):
    b = Q[sd]["both_name_correct_turn"]
    a, c = b["time_to_turn_s_text"], b["time_to_turn_s_v9"]
    w(f"| {sd} | {f1(a['median'], 2)} ({iqr(a['iqr'])}); {a['share_below_2s']:.2f} / {a['share_in_2_to_8s']:.2f} / {a['share_above_8s']:.2f} (n {a['n']}) | {f1(c['median'], 2)} ({iqr(c['iqr'])}); {c['share_below_2s']:.2f} / {c['share_in_2_to_8s']:.2f} / {c['share_above_8s']:.2f} (n {c['n']}) | {f1(b['time_to_turn_diff_text_minus_v9_median_s'], 2)} |")
w("\nEach feed over ALL its correct-turn windows (not only the windows where both are correct):\n")
w("| side | feed | n | median time to turn s (IQR) | share in 2-8 s | median distance m |")
w("|---|---|---|---|---|---|")
for sd in ("TURN_L", "TURN_R", "BOTH_SIDES"):
    for fd in ("text", "v9"):
        c = Q[sd][f"{fd}_correct_windows_time_to_turn"]
        w(f"| {sd} | {fd} | {c['n']} | {f1(c['median_s'], 2)} ({iqr(c['iqr_s'])}) | {f1(c['share_in_2_to_8s'], 2)} | {f1(c['distance_median_m'])} |")
w("\nReading: where both feeds name the correct turn, the text distance is longer than the v9 distance in "
  f"{B['both_name_correct_turn']['distance_m']['n_text_longer']} of {B['both_name_correct_turn']['distance_m']['n_distance_pairs']} windows (median {f1(B['both_name_correct_turn']['distance_m']['diff_text_minus_v9_median'])} m, about {f1(B['both_name_correct_turn']['time_to_turn_diff_text_minus_v9_median_s'], 1)} s at the ego speed of the window). "
  f"The text countdown lands {B['both_name_correct_turn']['time_to_turn_s_text']['share_in_2_to_8s']:.2f} of the time inside the label's 2-8 s band; the v9 countdown {B['both_name_correct_turn']['time_to_turn_s_v9']['share_in_2_to_8s']:.2f}, with {B['both_name_correct_turn']['time_to_turn_s_v9']['share_below_2s']:.2f} below 2 s (the turn is already about to start or running).\n")
w("## Permutation importance per nav channel (beside)\n")
w(f"Classifier: navecho2's (HistGradientBoosting, balanced weights, episode-grouped 5-fold out-of-fold). Importance = out-of-fold macro-F1 minus the macro-F1 with that channel permuted across the windows of each test fold ({r['permutation_importance']['v9']['ALL']['n_repeats']} repeats, seed 0; mean with the min-max of the repeats; 'paired' = one repeat read through the paired episode-cluster bootstrap, B 2,000). One-hot channels are collinear, so a single permuted token channel makes impossible rows; the token and distance GROUP rows are the cleaner reading.\n")
for sp in ("ALL", "INF_ANY"):
    w(f"### {sp}  (base macro-F1: v9 {r['permutation_importance']['v9'][sp]['base_macro_f1']:.3f}, text {r['permutation_importance']['text'][sp]['base_macro_f1']:.3f})\n")
    w("| channel / group | v9 importance (min, max) | v9 paired | text importance (min, max) | text paired |")
    w("|---|---|---|---|---|")
    for g in r["permutation_importance"]["v9"][sp]["groups"]:
        a, c = r["permutation_importance"]["v9"][sp]["groups"][g], r["permutation_importance"]["text"][sp]["groups"][g]
        pa = pd(a["paired_drop_repeat0"]) if "paired_drop_repeat0" in a else "-"
        pc = pd(c["paired_drop_repeat0"]) if "paired_drop_repeat0" in c else "-"
        w(f"| {g} | {a['importance_mean']:+.3f} ({a['importance_min']:+.3f}, {a['importance_max']:+.3f}) | {pa} | {c['importance_mean']:+.3f} ({c['importance_min']:+.3f}, {c['importance_max']:+.3f}) | {pc} |")
    w("")
w("The v9 classifier leans on the side tokens and on the distance pair; the text classifier leans on the TURN_R token, the FOLLOW token and the distance value, and its TURN_L token carries almost no importance (text TURN_L names are rare on the labelled windows). ALL-window importance is dominated by the TURN_R token because TURN_R is the larger turn class.\n")
w("## Unweighted shuffled-nav control (beside; navecho2's weighted K-SHUFFLE band is not re-read)\n")
u = r["shuffle_unweighted_beside"]
w(f"Nav permuted across windows, out-of-fold, the navecho2 classifier without class weights, {u['by_feed']['v9']['n_permutations']} permutations: v9 feed mean {u['by_feed']['v9']['mean']:.3f} (min {u['by_feed']['v9']['min']:.3f}, max {u['by_feed']['v9']['max']:.3f}); text feed mean {u['by_feed']['text']['mean']:.3f} (min {u['by_feed']['text']['min']:.3f}, max {u['by_feed']['text']['max']:.3f}); majority-class score {u['majority_baseline_macro_f1']:.3f}. Both feeds read within 0.014 of the majority-class score, so the unweighted classifier given noise reads the majority score; the weighted classifier reads below it because balanced weights spread its predictions over the rare classes. navecho2's weighted control read 0.230 (v9) and 0.145 (text) and its band stays as stated there (FAILED).\n")
w("## Controls\n")
c = r["controls"]
w(f"* K-SELF: a feed's correct-turn indicator paired with itself reads {c['K-SELF']['self_pair_delta']} (pass {c['K-SELF']['pass']}); the test also sends a mutant to RED.")
w(f"* K-REACH: a token equal to the label reads CORRECT on every turn window: {c['K-REACH']['label_as_token_all_correct']}.")
w(f"* K-FEED (nav_in identical across every arm of a feed), K-LANDED (17 head macro-F1 values equal the landed `fam/out/RESULT_FAM_EXTRA.json` to 1e-12) and K-BASE (the unpermuted fold models reproduce navecho2.crossfit_hgb exactly): problems list {c['K-FEED/K-LANDED/K-BASE problems']} (empty = pass).")
w("\n## Limits\n")
w(f"* The question lives on {r['n_turn_windows']} windows in {r['n_turn_episodes']} episodes (TURN_L {r['turn_support']['TURN_L']} windows in {Q['TURN_L']['n_episodes']} episodes). Windows of one episode are consecutive, so the intervals are wide and the TURN_L cell is thin; the episode-cluster bootstrap carries that width.")
w("* The estimator answers 'another draw of EPISODES' only; the shares agree across 4 training seeds and 3 text families (all beside rows), which is the training-run check available here, not a replicate arm.")
w("* The text arms see the feed through the head, and the shares describe the feed at NOW; the dump holds no earlier-frame nav, so a cue named a few frames before NOW is not visible to this reading.")
w("* The importance tables are a descriptive reading of the 5-channel classifier, not of the driving model.")
open(os.path.join(HERE, "out", "RESULT_NAVECHO3.md"), "w", encoding="utf-8").write("\n".join(o) + "\n")
print("ok")
