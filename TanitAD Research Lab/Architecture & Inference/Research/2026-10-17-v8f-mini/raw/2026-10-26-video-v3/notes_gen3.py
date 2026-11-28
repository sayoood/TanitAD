"""notes_gen3.py -- writes viz3/VIDEO_NOTES_V3.md.  Every table is generated from the source files (no transcription); sections for the unchanged v2 segments are sliced from viz2/VIDEO_NOTES.md.
Run after build_video3.py, controls_v2_rerun.py and controls3.py."""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
from v3_lib import SC, OUT, V2, HACC, HBLEND, HSPEC, SOURCES, KEYS, WINDOWS, SEEDS, SEL, getp  # noqa: E402

jl = lambda p: json.load(open(p))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
CT = jl(f"{OUT}/CONTROLS_V3.json")
RD = jl(f"{HACC}/READ.json")
SJ = {sd: jl(f"{HACC}/score_{sd}.json") for sd in SEEDS}
TB = jl(f"{OUT}/data/tables.json")["hblend_hspec"]
AP = jl(f"{HACC}/accel_profile.json")["per_seed"]
DK = jl(f"{HACC}/dk_read.json")
PK = np.load(f"{OUT}/data/plans_pack3.npz")
CH = jl(f"{OUT}/video_chapters.json")
v2n = open(f"{V2}/VIDEO_NOTES.md", encoding="utf-8").read()


def slice_(a, b):
    i = v2n.index(a)
    j = v2n.index(b, i + 1) if b else len(v2n)
    return v2n[i:j].rstrip() + "\n"


L = []
w = L.append
TIER = "T1-class open loop (own held outputs; NOT closed loop)"
EST = "paired episode-cluster bootstrap, B = 2,000, 141 episodes (taniteval/ci.py); another draw of EPISODES only; training variance = the two seeds side by side; inference deterministic"
w("# v8F-mini evidence video v3 (HACC) -- VIDEO_NOTES_V3 (every number, its artifact, evidence class, tier, estimator)\n")
w("Written 2026-10-25 by the viz3 subagent. Dev box, CPU only, ONE render process, nothing staged, committed or pushed. All paths are under "
  "`C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/` (`SC` below). Video: `viz3/v8f_evidence_v3.mp4`, "
  f"{CH['duration_s']:.1f} s, 1920x1080, H.264, 10 fps, 7 segments, {CH['unique_frames']} unique frames.\n")
w("## 0. Status, caveats, and what changed from v2\n")
w(f"* **Tier of every number: {TIER}.** The model reads its own held outputs; the plan is scored against the logged human path; nothing feeds back. Interventions (the nav counterfactual, the LISTEN donor swaps) are diagnostics of dependence.")
w(f"* **Estimator: {EST}.** A separated interval from a one-seed arm is necessary, not sufficient, for a lever claim (H-ESTIM-SEED-1); the control B1 below re-derives the four primary dominance intervals with an independent bootstrap.")
w("* **Baseline:** P10 is the programme's own kinematic floor (it extrapolates ego motion and ignores the scene). The dominance claim is against that floor, not against the state of the art; no external benchmark is shown.")
kp = {sd: RD["per_seed"][sd]["PRIMARY_KEEP"] for sd in SEEDS}
w(f"* **Seed 1's KEEP margin is thin** (on screen in segment 5): ADE6 HACC - H = {kp['M2b_s1']['delta']:+.4f} [{kp['M2b_s1']['lo']:+.4f}, {kp['M2b_s1']['hi']:+.4f}] m is SEPARATED-WORSE than H and passes the F/2 bar "
  f"({RD['KEEP_bar_m']:+.5f} m) by {RD['KEEP_bar_m'] - kp['M2b_s1']['hi']:.4f} m. Seed 0: {kp['M2b_s0']['delta']:+.4f} [{kp['M2b_s0']['lo']:+.4f}, {kp['M2b_s0']['hi']:+.4f}] (not separated).")
w("* **STRATEGIC family: IN FORCE for v8F** (PI-V8F-2026-10-17; R72 / R73; confirmed by `hacc/RESULT_V8F_HACC.md` section 8 and `score_<seed>.json::structural_identity.strategic`). **v2's notes and caption said PAUSED (PI ruling 2026-09-19): that is stale for v8F and is corrected "
  "in every v3 caption** (segment 6's caption and the closing card say IN FORCE; control S finds no 'paused' string among the drawn text). HACC touches the 6 s plan only; S's checkpoints and T's decision heads are the banked B arrays, so the strategic block is unchanged by HACC.")
w("* Never drawn: the words 'drives' and 'SOTA' (control W); 'closed loop' appears only in the mandated tier string 'NOT closed loop'.")
w("* **v2 -> v3 changes.** (1) HACC drawn beside GT / P10 / B / H on the same 4 pre-declared windows (`SELECTION_V2.md`, unchanged, sha256 below); HS (v2's fifth line) is kept as a thin dotted line and a table row so v2's content is not lost. "
  "(2) New segment 5 'Dominance over the kinematic prior'. (3) The three pending v2 source fixes are applied and visible in the v3 render: segment 6's S card has the S-age chart (held vs fresh 50 m error by S's age; both seeds, from `sfresh/sfresh_extra.json::age_profile_50m`), "
  "the two segment-6 window frames carry the shortened caption that points to the card for seed 1, and the segment-3 summary caption says the 23-of-24 T-lateral count is with the nav command at all three layers. (4) v2's stale STRATEGIC caption corrected (above). "
  "(5) The v2 gap card ('what is NOT yet better', which named a second-difference penalty in training as the next step) is replaced by the HACC version: that penalty was realised as HACC's fit objective, with zero training. "
  "(6) Segment 3's plans are the model's own plan B BEFORE the post-map; this is now written on the method card and in the footers (HACC is a post-map of B's plan; the counterfactual is a diagnostic of the model's chain).")
w("* **Evidence classes.** HACC numbers (segments 4 and 5) are **MEASURED (ours)** by PREREG_V8F_HACC's run (`SC/hacc/`, tip c8ea1623, `raw/2026-10-25-hacc/`) and are **re-read here, not re-fitted**: every on-screen number is bound to its JSON source (control N1) and to `RESULT_V8F_HACC.md` (N2); the plan arrays are re-derived from the banked weights (H1); "
  "the per-window and eval-wide ADE means are recomputed from the banked per-window arrays (H2); the four primary intervals are re-derived with an independent bootstrap (B1). The paired intervals themselves are **INHERITED** from `READ.json` / `score_<seed>.json` (taniteval/ci.py, seed 0) apart from that cross-check. "
  "Segment 3 (nav counterfactual) is MEASURED (ours) in v2 and regenerated here (controls V2). Segment 2 (LISTEN) and SFRESH numbers in segment 6 are **INHERITED** from their result files, each checked against the file's table in v2 and re-found in the source corpus here (control N4).")
w("* Nothing is registered, promoted or retracted by this video; landing and the GOALS_AND_CLAIMS register are the Master Mind's step. 'One window is an example; the eval-wide numbers are the evidence.'\n")

# ----------------------------------------------------------------------------------------------------------------- 1. controls
w("## 1. Controls (`viz3/CONTROLS_V3.json`): ALL_PASS = %s\n" % CT["ALL_PASS"])
h1 = CT["control_H1_HACC_as_drawn_equals_P_plus_w_tau_minus_P_and_the_banked_plan"]
h2 = CT["control_H2_ade6_recomputed_from_drawn_plans_and_tables_equal_sources"]
n1 = CT["control_N1_every_registered_on_screen_number_equals_its_json_source"]
n2 = CT["control_N2_dominance_table_accel_bands_and_KEEP_equal_RESULT_md"]
n3 = CT["control_N3_derived_aggregates_recomputed_independently_and_found_on_screen"]
n4 = CT["control_N4_every_numeric_token_drawn_is_a_declared_constant_or_a_source_value"]
b1 = CT["control_B1_primary_row_intervals_reproduced_by_an_independent_bootstrap"]
v2c = CT["control_V2_v2_controls_still_pass"]
p2 = v2c["model_level_rerun_CONTROLS_V3_part2"]
w("| control | result | tolerance |\n|---|---|---|")
w(f"| H1. HACC as drawn = P + w (.) (tau - P) from the banked weights (30 x 2) AND = the banked per-window plan `hacc/score_<seed>_perwindow.npz`; all 2,399 windows x 2 seeds + the 4 drawn windows x 2 seeds | max abs **{h1['max_abs_m']:.1e} m** | <= 1e-5 m |")
w(f"| H1 deliberate regressions (the check must be able to fail): axes of w swapped / w = 1 | {h1['deliberate_regression_arms']['axes_swapped_maxabs_m']:.2f} m / {h1['deliberate_regression_arms']['w_equals_1_maxabs_m']:.2f} m: both detected = {h1['deliberate_regression_arms']['both_detected']} | > 1e-3 m |")
w(f"| H2. ADE6 recomputed from the drawn plan and GT = the banked per-window ADE6, all 5 plans, 8 window-seed pairs; eval-wide per-window ADE1-6 means = the score JSON; B/H/P10 in `hacc/score_*.json` = `hblend/score_*.json` exactly; F = |ADE6(B s0) - ADE6(B s1)| = READ.json F | PASS = {h2['PASS']} | 1e-5 m (means: json rounding 5.1e-5) |")
w(f"| N1. every number the scenes registered ({n1['n_registered_numbers']}) re-resolved from its JSON file in a fresh process, re-formatted, equal to the drawn string, and found in that frame's drawn text | {n1['n_bound_and_drawn']} / {n1['n_registered_numbers']} bound, {n1['n_mismatch_or_not_drawn']} failures | exact string |")
w(f"| N2. dominance table (15 rows x 2 seeds), accel-band rows, KEEP strings = `RESULT_V8F_HACC.md` text, verbatim on screen | {n2['dominance_rows_x_seeds_compared_with_RESULT_md_section_4']} cells, {n2['accel_band_rows_found_on_screen']} / {n2['accel_band_rows_compared_with_RESULT_md_section_6']} band rows, KEEP s0 / s1 on screen = {n2['KEEP_found_on_screen']} | exact string |")
w(f"| N3. derived aggregates recomputed independently and found on screen: separated-better rows {n3['separated_better_rows_of_15']}, separated-worse {n3['separated_worse_rows']}, max abs ADE1-5 HACC - H {n3['max_abs_ADE1_5_HACC_minus_H_m']:.4f} m, KEEP margin s1 {n3['KEEP_margin_s1_m']:.4f} m, non-monotone steps {n3['non_monotone_steps']} | PASS = {n3['PASS']} | exact |")
w(f"| N4. existence scan: every numeric token of every drawn string ({n4['n_numeric_tokens_scanned']} tokens, all 177 frames) is a declared constant or a value present in the source corpus | {n4['n_distinct_unmatched']} unmatched | 0 |")
w(f"| B1. the 4 primary HACC - P10 intervals (ADE2, ADE6, accel MAE, curvature MAE; both seeds) re-derived with an independent numpy bootstrap (B = 2,000, 141 episodes) | PASS = {b1['PASS']}; e.g. ADE6 s0 own {b1['per_seed_row']['M2b_s0/ADE6']['own_mean_lo_hi'][0]:+.4f} [{b1['per_seed_row']['M2b_s0/ADE6']['own_mean_lo_hi'][1]:+.4f}, {b1['per_seed_row']['M2b_s0/ADE6']['own_mean_lo_hi'][2]:+.4f}] vs READ.json {b1['per_seed_row']['M2b_s0/ADE6']['read_json_mean_lo_hi'][0]:+.4f} [{b1['per_seed_row']['M2b_s0/ADE6']['read_json_mean_lo_hi'][1]:+.4f}, {b1['per_seed_row']['M2b_s0/ADE6']['read_json_mean_lo_hi'][2]:+.4f}] | within 3 % of the width (different random stream) |")
w(f"| V2. v2's controls still pass: CONTROLS_V2.json ALL_PASS as recorded = {v2c['ALL_PASS_as_recorded']}, re-derived from its numbers = {all(v2c['rederived'].values())}; control 1 (own nav at S, T, O reproduces `forward_infer`) RE-RUN with v2's harness: max abs **{p2['control_1_v2_own_nav_all_layers_reproduces_forward']['own_nav_all_layers_vs_forward_traj30']:.1f}** over 8 window-seed pairs (exactly 0.0); the {p2['regeneration_of_the_drawn_counterfactual_arrays']['n_counterfactual_cells_regenerated']} counterfactual cells regenerated, max abs vs the drawn arrays {p2['regeneration_of_the_drawn_counterfactual_arrays']['max_abs_m']:.1e} m (<= 1e-4); structural zero {v2c['rerun_on_v3_data']['structural_zero_cells']} cells max {v2c['rerun_on_v3_data']['structural_zero_max_abs']:.1f}; D6 recomputed from the drawn plans max {v2c['rerun_on_v3_data']['D6_and_LR_recomputed_from_drawn_plans_max_abs_m']:.1e} m; v2 pack arrays unchanged {v2c['rerun_on_v3_data']['v3_pack_keys_of_v2_equal_v2_plans_pack_max_abs_m']:.1f}; data copies byte-identical {all(v2c['rerun_on_v3_data']['data_copies_byte_identical_to_viz2'].values())}; frame accessor pixel-identical to v2 on {p2['frame_accessor_pixel_identical_to_v2']['n_frames_compared']} frames; SELECTION_V2.md unchanged (sha256 {v2c['rerun_on_v3_data']['SELECTION_V2.md_sha256'][:16]}) | PASS = {v2c['PASS']} | as v2 |")
w(f"| W / S. no 'drives' / 'SOTA' drawn, 'closed loop' only inside 'NOT closed loop'; no 'paused' string and 'IN FORCE' drawn | PASS = {CT['control_W_forbidden_words']['PASS']} / {CT['control_S_strategic_caption_not_stale']['PASS']} | - |")
w("\nControl 3 of v2 (frame alignment, 404 / 404 frames, min PSNR 48.6 dB) is v1's `align_control` and applies unchanged: v3 draws frames with a byte-identical copy of the accessor (checked pixel-identical to v2's on every frame drawn).\n")

# ----------------------------------------------------------------------------------------------------------------- 2/3. unchanged v2 segments
w("## 2. Segment 3 (nav counterfactual) -- MEASURED (ours) in v2, unchanged; regenerated here (controls V2)\n")
w("The numbers and tables below are v2's (`viz2/VIDEO_NOTES.md` section 2), sliced verbatim; the data file `viz3/data/cf_windows.npz` is a byte-identical copy of v2's and every drawn counterfactual array was regenerated by `viz3/controls_v2_rerun.py` (max abs 1.5e-5 m). "
  "The plans in this segment are the model's own plan B BEFORE the HACC post-map. Paths `viz2/data/...` in the slice are read-only originals; `viz3/data/` holds the copies the render reads.\n")
w(slice_("### 2a. The four pre-declared windows", "## 3. Segment 2"))
w("## 3. Segment 2 (who listens to what) -- INHERITED from LISTEN, unchanged from v2\n")
w(slice_("Source: `SC/listen/LISTEN_RESULT.md`", "## 4. Segment 4"))

# ----------------------------------------------------------------------------------------------------------------- 4. plans with HACC
w("## 4. Segment 4: better plans, now with HACC -- MEASURED (ours), " + TIER + "\n")
w("Artifacts: `viz3/data/plans_pack3.npz` (GT, B, P10, H, HS, HACC of the 4 windows x 2 seeds + per-step and ADE6; v2's arrays unchanged + HACC from `hacc/score_<seed>_perwindow.npz`; builder `viz3/plans_pack3.py`), "
  "`hacc/score_M2b_s{0,1}.json` (single, paired, consistency), `hspec/score_hs2_M2b_s{0,1}.json` (HS via `viz3/data/tables.json`). Estimator: " + EST + ".\n")
w("### 4a. ADE by horizon over all 2,399 windows (m; mean [95 % CI])\n")
w("| seed | plan | 1 s | 2 s | 3 s | 4 s | 5 s | 6 s |\n|---|---|---|---|---|---|---|---|")
for sd in SEEDS:
    for arm in ("B", "H", "HS", "HACC", "P10"):
        cells = []
        for h in range(1, 7):
            v = TB[sd]["ade"]["HS"][f"ade{h}"] if arm == "HS" else SJ[sd]["single"][arm][f"ade{h}"]
            cells.append(f"{v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
        w(f"| {sd[-1]} | {arm} | " + " | ".join(cells) + " |")
w("\n### 4b. Paired differences (m unless stated)\n")
w("| seed | pair | ADE6 | accel MAE (m/s^2) | curvature MAE (1/m) | CONSISTENCY(a) 'any' (share) |\n|---|---|---|---|---|---|")
for sd in SEEDS:
    for pair in ("HACC_minus_P10", "HACC_minus_H", "HACC_minus_B"):
        P = SJ[sd]["paired"][pair]
        cp = SJ[sd]["consistency"]["paired"][pair]["viol_any"]
        f = lambda d, n=4: f"{d['mean']:+.{n}f} [{d['lo']:+.{n}f}, {d['hi']:+.{n}f}] " + ("sep." if d["separated"] else "n.s.")
        w(f"| {sd[-1]} | {pair.replace('_minus_', ' - ')} | {f(P['ade6'])} | {f(P['fw_lon_accel_mae'])} | {f(P['fw_lat_curvature_mae_1pm'], 5)} | {f(cp)} |")
w("\nv2 rows kept for context (from `viz3/data/tables.json`): " + "; ".join(f"seed {sd[-1]} H - P10 accel {TB[sd]['accel']['H-P10']['mean']:+.3f}, ADE6 H - B {TB[sd]['paired_ade6']['H-B']['mean']:+.3f}" for sd in SEEDS) + ".\n")
w("### 4c. The four windows: ADE6 of each plan (m; seed 0 / seed 1), `viz3/data/plans_pack3.npz` (recomputed from the drawn plan and GT, control H2)\n")
w("| window | P10 | B | H | HS | HACC |\n|---|---|---|---|---|---|")
for k in KEYS:
    w(f"| {k} (#{WINDOWS[k]['window']}) | " + " | ".join(f"{float(PK[f'{SEEDS[0]}/{k}/ade6_{a}']):.2f} / {float(PK[f'{SEEDS[1]}/{k}/ade6_{a}']):.2f}" for a in ("P10", "B", "H", "HS", "HACC")) + " |")
w("\nOn the four windows HACC is not uniformly better (it is worse than P10 on FAILURE at both seeds and on CURVE at seed 1, and worse than H on JUNCTION and CURVE s0): the captions say so per window; one window is an example, the eval-wide rows are the evidence.\n")
w("### 4d. The gap card: what HACC closes and what it does not (`hacc/score_<seed>.json`; both seeds on screen)\n")
w("Closed vs P10: accel MAE and curvature MAE (both separated-better at both seeds, table 4b). Not better: CONSISTENCY(a) 'any' vs B (HACC - B +0.0763 / +0.0550 separated-worse), vs H on cross-track MAE (+0.0274 / +0.0138), FDE6 (+0.1195 / +0.1886) and, at seed 1, ADE6; "
  "distance keeping vs P10 not separated; the accel error at 5.2-5.8 s still above P10 (section 5c). Quoted from the JSON by the scene code (control N1).\n")

# ----------------------------------------------------------------------------------------------------------------- 5. dominance
w("## 5. Segment 5 (NEW): Dominance over the kinematic prior -- MEASURED (ours, `hacc/`), " + TIER + "\n")
w("### 5a. The dominance row, HACC - P10 (paired; `hacc/READ.json::per_seed.<seed>.PRIMARY_DOM.rows`; the same numbers verbatim in `hacc/RESULT_V8F_HACC.md` section 4, control N2)\n")
w("Estimator: " + EST + ". Outcome (PREREG outcome table, in order): **DOMINATES P10** (PRIMARY-DOM and PRIMARY-KEEP at both seeds). Rows: `*` = interval excludes 0.\n")
w("| metric | requirement | seed 0: mean [95 % CI] | verdict s0 | seed 1: mean [95 % CI] | verdict s1 |\n|---|---|---|---|---|---|")
for k, row in RD["per_seed"]["M2b_s0"]["PRIMARY_DOM"]["rows"].items():
    cells = []
    for sd in SEEDS:
        r = RD["per_seed"][sd]["PRIMARY_DOM"]["rows"][k]
        d = 5 if k == "curvature MAE" else 4
        cells += [f"{r['mean']:+.{d}f} [{r['lo']:+.{d}f}, {r['hi']:+.{d}f}]", ("better, separated" if (r["separated"] and r["better"]) else ("WORSE, separated" if (r["separated"] and r["worse"]) else "not separated"))]
    w(f"| {k} | {row['need']} | " + " | ".join(cells[:2]) + " | " + " | ".join(cells[2:]) + " |")
dkn = lambda sd, a, f: getp(DK["pairs"][f"HACC_{sd[-2:]}__minus__P10"], f"{a}/{f}")
w("\nDK rows (distance keeping; `hacc/dk_read.json`): only windows with a lead agent in frame; n windows / episodes: headway "
  + ", ".join(f"seed {sd[-1]} {dkn(sd, 'dk/headway_min_m', 'n_used')} / {dkn(sd, 'dk/headway_min_m', 'n_episodes')}" for sd in SEEDS) + "; time-gap "
  + ", ".join(f"seed {sd[-1]} {dkn(sd, 'dk/time_gap_min_s', 'n_used')} / {dkn(sd, 'dk/time_gap_min_s', 'n_episodes')}" for sd in SEEDS) + "; LEAD-VIOL at-risk "
  + ", ".join(f"seed {sd[-1]} {dkn(sd, 'lead_viol/at_risk', 'n_at_risk')} / {dkn(sd, 'lead_viol/at_risk', 'n_episodes')}" for sd in SEEDS) + ". (The 141-episode statement applies to the other rows; the DK rows bootstrap their own episode subsets.) "
  "Decision agreements are derived from the plan (plan-derived). Separated-better rows: " + ", ".join(f"seed {sd[-1]} {sum(1 for k in RD['per_seed'][sd]['PRIMARY_DOM']['rows'] if RD['per_seed'][sd]['PRIMARY_DOM']['rows'][k]['separated'] and RD['per_seed'][sd]['PRIMARY_DOM']['rows'][k]['better'])} of 15" for sd in SEEDS) + "; separated-worse: 0.\n")
w("### 5b. PRIMARY-KEEP (ADE6, HACC - H; bar = F/2 with the exact F)\n")
w("| seed | HACC - H ADE6 [95 % CI] | bar (hi <=) | pass | lambda* (m*) |\n|---|---|---|---|---|")
for sd in SEEDS:
    k = RD["per_seed"][sd]["PRIMARY_KEEP"]
    w(f"| {sd[-1]} | {k['delta']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}] | {k['bar_hi_le']:+.5f} | {k['pass']} | {RD['per_seed'][sd]['lam_star']:,.0f} ({RD['per_seed'][sd]['m_star']}) |")
w(f"\nF = |ADE6(B s0) - ADE6(B s1)| = {RD['F_exact_m']:.5f} m (`READ.json::F_exact_m`; reproduced from the score JSONs by control H2); the seed-1 margin to the bar is {RD['KEEP_bar_m'] - kp['M2b_s1']['hi']:.5f} m (derived: bar - hi).\n")
w("### 5c. The per-step accel-error profile (descriptive; unpaired per-cell means over the 2,399 windows; `hacc/accel_profile.json`, tied to the banked `fw_lon_accel_mae` by control 7 of RESULT_V8F_HACC.md: 9.5e-7 / 4.8e-7)\n")
w("| seed | band | HACC | H | P10 | HACC - P10 | H - P10 |\n|---|---|---|---|---|---|---|")
for sd in SEEDS:
    for key, b in AP[sd]["bands"].items():
        w(f"| {sd[-1]} | {key} | {b['HACC']:.3f} | {b['H']:.3f} | {b['P10']:.3f} | {b['HACC_minus_P10']:+.3f} | {b['H_minus_P10']:+.3f} |")
    a = AP[sd]["all_cells_mean"]
    w(f"| {sd[-1]} | all 29 cells | {a['HACC']:.3f} | {a['H']:.3f} | {a['P10']:.3f} | {a['HACC'] - a['P10']:+.3f} | {a['H'] - a['P10']:+.3f} |")
w("\nReading (on screen): H's gap to P10 sat at 0.8-1.6 s, 2.2-2.6 s and 5.2-5.8 s; HACC is below P10 in the first two (gone) and still above P10 at 5.2-5.8 s (reduced, NOT gone).\n")
w("### 5d. The mechanism card (descriptive; two things changed together, no ablation separates them)\n")
w("Weights: `hacc/weights_HACC_M2b_s{0,1}.json` and `hblend/weights_M2b_s{0,1}.json` (`w_along_by_step`, `w_cross_by_step`; 30 each); lambda* from `READ.json`; mean |tau - P| per step from `viz3/data/aux3.json` (built by `viz3/aux3.py` from `hblend/score_*_perwindow.npz`). "
  "HACC = P + w (.) (tau - P): w fitted on position error + lambda x the second-difference error, bounded [0, 1], no isotonic step; H: position error only, then isotonic. On screen: the along weight at steps 10 -> 14 (H "
  + ", ".join(f"seed {sd[-1]} {jl(SOURCES['hblend_w_s' + sd[-1]])['w_along_by_step'][9]:.2f} -> {jl(SOURCES['hblend_w_s' + sd[-1]])['w_along_by_step'][13]:.2f}" for sd in SEEDS) + "; HACC "
  + ", ".join(f"seed {sd[-1]} {jl(SOURCES['hacc_w_s' + sd[-1]])['w_along_by_step'][9]:.2f} -> {jl(SOURCES['hacc_w_s' + sd[-1]])['w_along_by_step'][13]:.2f}" for sd in SEEDS) + "), the counts of non-monotone steps, and the first cross-axis steps (mean |tau - P| at step 4 is small).\n")
w("### 5e. The caveats card (all numbers from `READ.json`, control N1)\n")
w("Seven items: tier; baseline = the programme's floor; estimator and the seed floor F; seed 1's thin KEEP margin; distance keeping not separated; only the plan moves (T's heads and S are the banked B arrays; HACC - H is exactly 0 on the strategic block, `score_<seed>.json::structural_identity`); not done here (HACC not folded into `forward_infer`; D-V8F-T1KIN-1 needs a PI go for the pooled eval cache).\n")

# ----------------------------------------------------------------------------------------------------------------- 6. strategic
w("## 6. Segment 6: the strategic layer at its own tick (DIAGNOSTIC) -- family IN FORCE for v8F\n")
s5 = slice_("## 5. Segment 5: the strategic layer at its own tick", "## 6. Frames and windows")
s5 = s5.replace("## 5. Segment 5: the strategic layer at its own tick (DIAGNOSTIC; family PAUSED)\n", "")
w("**Status correction vs v2:** the STRATEGIC family is IN FORCE for v8F (PI-V8F-2026-10-17; R72 / R73); the v2 text 'PAUSED' is stale. The diagnostics below are INHERITED (SFRESH; `sfresh/SFRESH_RESULT.md`, `sfresh_numbers.json`) and MEASURED per window (CPU re-run, v2); they do not change with HACC. "
  "S's checkpoint error IS the family's metric: at the deployed (held) ages S loses to the nav-only baseline at 50 m; ticked fresh it beats it.\n")
w(s5.replace("Part of the held-vs-fresh gap is DEFINITIONAL", "Part of the held-vs-fresh gap is DEFINITIONAL").replace("S-age chart (source only, not yet in the mp4)", "S-age chart (NOW in the v3 render)"))
w(slice_("## 6. Frames and windows", "## 7. Manifest").replace("## 6. Frames and windows", "## 6b. Frames and windows (unchanged from v2)"))

# ----------------------------------------------------------------------------------------------------------------- 7. sources
w("## 7. Source files read by the render and the controls (sha256 of the bytes read)\n")
w("| key | path | sha256 |\n|---|---|---|")
for k, p in SOURCES.items():
    w(f"| {k} | `{p.replace(SC, 'SC')}` | `{sha(p)}` |")
for extra in (f"{HACC}/RESULT_V8F_HACC.md", f"{HACC}/PREREG_V8F_HACC.md", f"{V2}/SELECTION_V2.md", f"{V2}/CONTROLS_V2.json", f"{SC}/listen/READ.json", f"{SC}/listen/fig_listen_grid.png"):
    w(f"| (file) | `{extra.replace(SC, 'SC')}` | `{sha(extra)}` |")
w("\nPer-window plan arrays: `hblend/score_M2b_s{0,1}_perwindow.npz`, `hspec/score_hs2_M2b_s{0,1}_perwindow.npz`, `hacc/score_M2b_s{0,1}_perwindow.npz` (read by `plans_pack3.py` and, again, by the controls).\n")
w("## 8. Manifest (all under `SC/viz3/`; nothing in git; list with bytes and raw-bytes git blob sha1 in `FILES.txt`)\n")
w("`v8f_evidence_v3.mp4`, `CONTROLS_V3.json` (+ `CONTROLS_V3_part1.json`, `CONTROLS_V3_part2.json`), `VIDEO_NOTES_V3.md`, `qa_*.png` (6 stills), `video_chapters.json`, `FILES.txt`, `HANDOFF.md` (if present), "
  "`data/` (cf_windows.npz, cf_evalwide.json, tables.json, plans_pack3.npz, aux3.json, numreg.json, onscreen_text.json), scripts `v3_lib.py v3_draw.py v3_scenes.py build_video3.py plans_pack3.py aux3.py controls3.py controls_v2_rerun.py notes_gen3.py`.\n")
open(f"{OUT}/VIDEO_NOTES_V3.md", "w", encoding="utf-8").write("\n".join(L))
print("wrote VIDEO_NOTES_V3.md", sum(len(x) for x in L))
