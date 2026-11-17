import json
import os

H = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(H + "/out/RESULT_NAVECHO.json", encoding="utf-8"))


def f(d):
    return f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" + (" sep" if d["separated"] else "") + (" >F" if d.get("above_floor") else "")


def P(x):
    return f"{x['point']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"


o = []
rd = r["reading"]
nan = float("nan")
o.append("# RESULT_NAVECHO -- D-STRAT1-TACLAT-F1: nav echo or skill cost\n")
o.append(f"Evidence class: MEASURED (ours; banked T1-kin dumps in `F:/TanitAD-scratch/strat1/read/t1`, zero training, CPU). Tier of every number: {r['tier']}.")
o.append(f"Estimator: {r['estimator']}. 2,399 windows, 141 episodes. Artifact: `F:/TanitAD-scratch/strat1/navecho/out/RESULT_NAVECHO.json`.\n")
o.append(f"## Reading by the committed rules: **{rd['verdict']}**\n")
o.append(f"* ECHO needs the nav-only predictor at >= 90 % of C0 head macro-F1 at s0 and s1. It reads {rd['nav_only_ratio_C0_feed']['s0']:.3f} (s0) and {rd['nav_only_ratio_C0_feed']['s1']:.3f} (s1): not met.")
g = rd["S1_minus_C0_on_UNINF_BOTH"]
o.append(f"* SKILL-COST needs the S1 - C0 head gap separated, same sign, at s0 and s1 on the uninformative windows. It reads s0 {f(g['s0'])}, s1 {f(g['s1'])}: s0 is not separated, so not met.")
o.append("* Neither rule fires, so the reading is MIXED. **The uninformative split cannot carry the test**: it holds 1,736 labelled windows of which only 10 are turns (TURN_L 1, TURN_R 9), C0 and S1 at s0 and s1 recall no turn there (macro-F1 0.332, the majority-class score over the three present classes), and a gap of -0.0003 to -0.0005 is one or two windows, far below the floor F 0.067. It neither supports nor rules out a skill cost on a turn that no feed announces.")
sh = rd["additive_share_of_gap"]
n0, n1 = sh["s0"]["net_correct_windows_C0_minus_S1"], sh["s1"]["net_correct_windows_C0_minus_S1"]
o.append(f"* **Where the gap lives (additive, labelled windows, net correct windows C0 - S1):** s0 total {sh['s0']['total']}: UNINF_BOTH {n0['UNINF_BOTH']}, INF_ANY {n0['INF_ANY']}; s1 total {sh['s1']['total']}: UNINF_BOTH {n1['UNINF_BOTH']}, INF_ANY {n1['INF_ANY']}. About 90 % of the gap sits on the 439 windows (371 labelled) where at least one feed names a turn.")
o.append("* **Why it is not a plain echo either:** a map from the nav token alone reaches macro-F1 0.562 (v9 feed) against the C0 head 0.816 / 0.862; the head uses the image beside the token. On the windows where C0's own token names a turn the head reads 0.736 / 0.805 and the token-only map 0.337, so the head sits far above its nav. What the data cannot separate: the v9 token is built from the ego's own future (section 1), so on those windows C0 holds a cleaner route cue than the text feed holds, and that difference is the cost the text nav carries into the head on announced turns. The text feed names a turn on 223 windows, the v9 feed on 395; the two tokens differ on 261 windows.")
o.append("* Same sign at s10 / s11 and for N1 / S1d: see the tables. s10 is the weak seed (INF_ANY S1 - C0 -0.094, not separated).\n")
o.append("## 1. Provenance (two probes each, commit 0ddfb65)\n")
o.append("| object | derivation (file:line) | function of the ego future? |\n|---|---|---|")
o.append("| `lat_cls_a` | `stack/scripts/build_v9_labels.py:802-891, 969`: heading change `dps = degrees(gpsi[20:] - gpsi[20])` of the LOGGED future from NOW + 2 s to the horizon, `theta` its extreme, `is_junc = big and dom.is_turn` where `dom` is a turn segment of the ego's own track overlapping 2 s to 8 s; class = side of theta (1 L, 2 R); else lane change or lane keep. Test pins: `tests/test_v9_labels_builder.py:50-68, 147` | YES, entirely (ego logged poses 2-8 s ahead) |")
o.append("| v9 `nav_token`, `nav_d_next_m` | `build_v9_labels.py:260-283` (`nav_fields`): the first ANNOUNCED turn `u` of `turns` (segmented from the same ego track, `Turn.announced` :229); token = side of `u` when in progress or `d_next <= d_ann = max(30 m, 6 s x v_now)` (:265, :279); distance = `u.s_start - s_now`. Fed as 5 channels by `stack/tanitad/data/v8f_mini_data.py:219-222` (one-hot token, d/100 clipped to [0,3], valid) | YES, same turn table and track as the label |")
o.append("| text nav (N1, S1, S1d) | `v8f_mini_data.py:373-396` (`nav_text_arrays`) loads `navtext_nav.npz` (sha256 pinned; report `inputs/navtext_nav_report.json`); built by the BUILD_SPEC section 5 rule from NVIDIA route text (GPS trace matched to OSM routing, `navtok.py`; `PREREG_NAVTEXT_DRAFT.md` sections 1-3): the same announcement cut (`d_ann = max(30, 6 v)`, countdown distance, U-turn excluded) applied to the text turn. 8,235 eval steps differ from the v9 token (`navtext_nav_report.json`, eval `n_steps_text_ne_v9`) | NO ego-future turn table; it follows the driven GPS route (a function of where the car drove) on a different clock (G-CLOCK-NAV median error 5.3 s) |")
o.append("\nThe v9 token and the label share a source (the ego track's turn table) and differ in time window: the token looks to the next announced turn (about 6 s of travel or 30 m), the label looks 2 s to 8 s ahead. They are near functions of each other, not identical: of the 335 labelled windows where C0's v9 token names a turn, 157 carry LANE_KEEP and 178 a turn label.\n")
o.append("## 2. The nav-only predictor (macro-F1 against lat_cls, tier T1-kin tick 0)\n")
o.append("Map the nav token (FOLLOW / TURN_L / TURN_R / no row) to the majority label; fit on one half of the EPISODES, score on the other, halves swapped. Majority baseline 0.317.\n")
o.append("| feed | fit A score B | fit B score A | mean | 20 random splits (min / median / max) | token x distance bins (sensitivity) | shuffled nav (20 perms, mean) |\n|---|---|---|---|---|---|---|")
for k, v in r["nav_only_predictor"]["by_feed"].items():
    t, d = v["token_map"], v["token_x_distance_map (sensitivity)"]
    s = t["random_split_spread"]
    o.append(f"| {k} | {t['fit_A_score_B']['macro_f1']:.3f} | {t['fit_B_score_A']['macro_f1']:.3f} | {t['mean']:.3f} | {s['min']:.3f} / {s['median']:.3f} / {s['max']:.3f} | {d['mean']:.3f} | {v['shuffled_nav_control']['mean']:.3f} |")
o.append("\n| arm | feed | head macro-F1 | nav-only (token, mean of halves) | ratio | nav-only (token x distance) | ratio |\n|---|---|---|---|---|---|---|")
for a, t in r["head_vs_nav_only"].items():
    o.append(f"| {a} | {t['feed']} | {t['head_macro_f1']:.3f} | {t['nav_only_token_cross_fit_mean']:.3f} | {t['ratio_nav_only_token_to_head']:.3f} | {t['nav_only_token_x_distance_cross_fit_mean']:.3f} | {t['ratio_nav_only_token_x_distance_to_head']:.3f} |")
o.append("\nNav-only out-of-fold macro-F1 on each split (its own feed): " + "; ".join(f"{k}: " + ", ".join(f"{sp} {x:.3f}" for sp, x in v["token_map_oof_macro_f1_on_splits"].items()) for k, v in r["nav_only_predictor"]["by_feed"].items()) + ".\n")
o.append("## 3. The split\n")
o.append("Splits by the nav feeds at tick 0: UNINF_BOTH = neither the v9 nor the text token names a turn (the committed uninformative set, the only one where no arm holds route information); INF_ANY = its complement; UNINF_V9 / INF_V9 = by C0's own token.\n")
o.append("| split | n windows | n labelled | n episodes (labelled) | label support |\n|---|---|---|---|---|")
for sp, c in r["splits"]["counts"].items():
    sup = ", ".join(f"{k} {v}" for k, v in c["label_support"].items() if v)
    o.append(f"| {sp} | {c['n_windows']} | {c['n_labelled']} | {c['n_episodes_labelled']} | {sup} |")
ARMS = ["C0_s0", "C0_s1", "C0_s10", "C0_s11", "C0_rep", "N1_s0", "N1_s1", "N1_s10", "N1_s11", "S1_s0", "S1_s1", "S1_s10", "S1_s11", "S1d_s0", "S1d_s1", "S1d_s10", "S1d_s11"]
for sp in ("UNINF_BOTH", "INF_ANY", "UNINF_V9", "INF_V9"):
    gp, b = r["split_gaps"][sp], r["split_head_blocks"][sp]
    fl = gp["floor_F_abs_C0rep_minus_C0s0"]
    o.append(f"\n### {sp}  (floor F: macro-F1 {fl.get('macro_f1', nan):.3f}; TURN_R recall {fl.get('recall_TURN_R', nan):.3f})\n")
    o.append("Head macro-F1 and TURN recall per arm (point [95 % episode-cluster CI]; n/a = class absent):\n")
    o.append("| arm | macro-F1 | TURN_L recall | TURN_R recall |\n|---|---|---|---|")
    for a in ARMS:
        x = b[a]
        rl = P(x["recall_TURN_L"]) if x["recall_TURN_L"] and "point" in x["recall_TURN_L"] else "n/a"
        rr = P(x["recall_TURN_R"]) if x["recall_TURN_R"] and "point" in x["recall_TURN_R"] else "n/a"
        o.append(f"| {a} | {P(x['macro_f1'])} | {rl} | {rr} |")
    o.append("\nPaired X - C0 (same seed), macro-F1 [paired CI]; sep = CI excludes 0, >F = above the floor; net = net correct windows X - C0:\n")
    o.append("| X | s0 | s1 | s10 | s11 |\n|---|---|---|---|---|")
    for fam in ("S1", "N1", "S1d"):
        o.append(f"| {fam} - C0 | " + " | ".join(f"{f(gp[f'{fam}_{s}-C0_{s}']['macro_f1'])}; net {gp[f'{fam}_{s}-C0_{s}']['net_correct_windows_x_minus_ref']}" for s in ("s0", "s1", "s10", "s11")) + " |")
    o.append("\nTURN recall gap S1 - C0:\n")
    o.append("| class | s0 | s1 | s10 | s11 |\n|---|---|---|---|---|")
    for c in ("recall_TURN_L", "recall_TURN_R"):
        o.append(f"| {c[7:]} | " + " | ".join((f(gp[f'S1_{s}-C0_{s}'][c]) if c in gp[f'S1_{s}-C0_{s}'] else "n/a") for s in ("s0", "s1", "s10", "s11")) + " |")
o.append("\n## 4. Controls\n")
c = r["controls"]
o.append(f"* K-UNION: pass {c['K-UNION']['pass']} ({c['K-UNION']['reading']}).")
o.append(f"* K1 self-pair: pass {c['K1_self_pair']['pass']} ({c['K1_self_pair']['reading']}).")
o.append(f"* K-SHUFFLE: shuffled nav {c['K-SHUFFLE']['shuffled_mean_macro_f1']:.4f} against majority baseline {c['K-SHUFFLE']['majority_baseline_macro_f1']:.4f}: pass {c['K-SHUFFLE']['pass']}.")
o.append(f"* K-LANDED / K-FEED problems list: {c['problems']} (empty = every arm full-set macro-F1 equals the landed `RESULT_FAM_EXTRA.json` to 1e-12, and every arm of a family carries the identical nav feed).")
o.append("\n## 5. Limits\n")
sp9 = r["nav_only_predictor"]["by_feed"]["v9"]["token_map"]["random_split_spread"]
o.append(f"* The pre-registered echo question asks about windows with no route information, and only 10 labelled turn windows exist there; the MIXED reading follows from that as much as from the numbers. A powered version needs a larger eval grid or a train-split read (not run here).")
o.append(f"* One fixed episode split (seed 0) carries the committed nav-only figure; 20 random splits range {sp9['min']:.2f} to {sp9['max']:.2f} (v9 feed), against head 0.816 / 0.862, so the ratio stays below 0.90 at every split.")
o.append("* A separated interval from a one-seed arm is necessary, not sufficient; the estimator answers 'another draw of EPISODES'. The S1 - C0 gap on INF_ANY is separated at s0, s1 and s11 and has the same sign at s10.")
o.append("* Provenance rows are read from source at commit 0ddfb65.")
open(H + "/out/RESULT_NAVECHO.md", "w", encoding="utf-8").write("\n".join(o) + "\n")
