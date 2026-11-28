# SOURCES for `section_7_20.md`

Every number in the section, mapped to its primary file and the exact line or key it came from.

* All repo reads are from the read-only mirror at commit **`a5a4ee8b7309379c5288be81f3011d36581add9e`**
  (`git -C C:/Users/Admin/tanitad-push show a5a4ee8:<path>`). Line numbers are lines of that blob.
* `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`
* `REG` = `Project Steering/MODEL_REGISTRY.md` (blob of 569,205 bytes)
* `GAC` = `Project Steering/GOALS_AND_CLAIMS.md` (blob of 2,809,790 bytes)
* `RL` = `Project Steering/RETRACTION_LOG.md`
* Class = MEASURED (ours) / PUBLISHED (cited) / INHERITED (another doc, not re-verified here) / ESTIMATED / HYPOTHESIS.
  "RE-MEASURED" = reproduced by this author from the primary artifact in this session.
* **No number below is sourced from a handoff, changelog or weekly report.** Where a number appears only in
  `GAC` (the register) and the raw artifact was also read, both are given.

## 0. The two things this author computed (new numbers, not in the registry)

Both read the harness CSVs on the dev box and use `zero_tail.Boot`'s estimator exactly (ratio of sums over
136 logs, B = 10,000, seed 20260927, percentile 2.5 / 97.5). Controls were run in the same breath and reproduce.

| artifact (staged) | proposed repo path | what it gives | controls |
|---|---|---|---|
| `C:/Users/Admin/paper_pkg/marginal_ci.py` + `marginal_ci.json` | `PK/raw/2026-10-17-paper-s720-recompute/` | marginal PDMS and 95 % interval of six runs | `model_final` 83.2363 [81.7709, 84.6301] = `REG` 5246 83.24 [81.77, 84.63]; teacher 89.7861 [88.5474, 90.9737] = `RESULT_P4_TEACHER_NAVTEST.md` 58 (89.79 [88.55, 90.97]); human 94.5514 [93.781, 95.2328] = same file line 38 col 2 (94.55 [93.78, 95.23]). Route-fix row: point equal, endpoints differ by 0.02 and 0.08 from the banked [83.18, 85.77] (the banked interval used a different resampler); the banked one is quoted |
| `C:/Users/Admin/paper_pkg/gap_decomp_ci.py` + `gap_decomp_ci.json` | same | paired intervals on the four gap terms | prints 6.3045 = 4.1546 + 0.8556 + 1.3053 − 0.0110; `ep_lowtail.out` line 1 prints 6.30 = 4.15 + 0.86 + 1.31 − 0.01; zero picks u400 529, human 0 |

Input CSVs (dev box `D:/Projects/TanitAD/data/refe_navtest/score/<run>/<run>.csv`), sha256 prefixes read now and
equal to the registry's: `refe_navtest_final` fb6a01a2fa95f1df · `refe_navtest_final_routefix`
bec72778cb6e1a41 (`REG` 5276) · `refe_stage2_sft4_B_u400_navtest_full` df005c4750a186de (`REG` 5275) ·
`refe_stage2_p2_full` 61073a61e44a3dcf (`GAC` 16710) · `refe_p4_teacher_navtest` 23b6a41d636fd37e (`REG` 5277) ·
`refe_p4_human_navtest` 5bbe9006e6854246 (`REG` 5278). Token→log map `data/refe_navtest/navtest_full_tokens.json`
(sha256 prefix 927bbe3838dd70af; it differs from the copy ZT3 hashed as c6b5cae6…, but yields 12,146 tokens /
136 logs and reproduces every control, so the logs are the same).

## 1. Opening paragraph

| number / claim in the section | primary file | line / key | class |
|---|---|---|---|
| "§12.4 records that the programme contained no NAVSIM implementation … benchmark, then reward, then RL" | `Paper/TANITAD_PAPER.md` | lines 3908–3922 (heading 3908; "zero `import navsim`" line 3911; "benchmark → reward → RL" line 3915) | paper text (now stale; see CONFLICTS C1) |
| DriveZero = arXiv:2609.06055, library key | `TanitAD Research Lab/Library/library.json` | `entries["2609.06055"]`: sha256 `82a00048648b155ecad83888b94be95a39c3f6511cb8497eb2fb435feb72797b`, 9,392,899 bytes; the PDF in the mirror hashes to the same value | PUBLISHED, banked |
| best REFe checkpoint 88.25 | `REG` | line 5275 (§14.2 table, SFT-4 B u400) | MEASURED |
| bar 94.55 | `REG` 5282; PDF `2609.06055` Table A13 row "DINOv3 ViT-L" | PDMS 94.55 (NC 98.80, DAC 99.25, TTC 95.74, C 99.94, EP 92.72, DDC 95.87) | PUBLISHED, RE-VERIFIED in the banked PDF (pdftotext of the mirror blob) |
| 65.9 % of the gap is the zero tail | `gap_decomp_ci.json` | `gate_share_of_gap_pct` 65.9 (4.1546 / 6.3045) | RE-MEASURED; registry form: `GAC` 16250 "4.15 of 6.30" |
| "P6 has not yet trained" | `GAC` | 16716 (D-REFE-P6-D7: 3-C twin bank decided 2026-10-17, parts build resumed, re-roll next), 16713 (ZT4 "BLOCKED … P6's checkpoint") | MEASURED status at a5a4ee8 |

## 2. (a) The instrument

| number / claim | primary file | line / key | class |
|---|---|---|---|
| 322,070,854 params, 18,991,426 trainable; four cameras at 512×960; 64 proposals; six-component PDM scorer | `REG` | 5258 (Architecture row, `config.json`) | MEASURED |
| paper's 338.46 M full / 18.58 M trainable, 4 cams, 64 proposals, 960×512 (W×H) | PDF `2609.06055` | Table A12 (the pdftotext layout shifts the value column one row; read as Trainable 18.58 M, Full 338.46 M, matching the library note in `LIBRARY.md` line 34) | PUBLISHED |
| navtrain 103,288 tokens / 1,192 logs; DriveRL `checkpoint_2400` teacher | `REG` | 5262 (Data row) | MEASURED |
| 10,075 optimiser steps | `REG` | 5260 (Schedule row) | MEASURED |
| seven ego-kinematic scalars | `PK/REFE_MODEL.md` | line 11 ("ego **7 real kinematic scalars**") and line 16 (table row "architecture … ego 7-D") | MEASURED (as built 2026-09-21; not re-read at u400) |
| no HD-map geometry as planner input (PI 2026-09-28); nav command admissible | `REG` | 5251 (Amendment 8 row, PI quotes) | PI ruling, verbatim in `REG` |
| maps as training labels (PI 2026-10-11); p(lane) is a learned scorer output | `REG` | 5281 | PI ruling + MEASURED design |
| seven declared departures (bf16; growing bank; 10 Hz rollouts; accumulation 4 × 64; scorer fixed-then-on-policy from step 3,708; NAVSIM DAC labels; NAVSIM comfort labels) | `REG` | 5264 (items 1–7); on-policy switch at "step 3,708" = last fixed step 3,707 (`REG` 5247) | MEASURED |
| PDMS = NC × DAC × (5·EP + 5·TTC + 2·C)/12; 4 s horizon; LQR + kinematic bicycle; non-reactive; collision / DAC zero the token | PDF `2406.15349` | §3 "Non-Reactive Simulation" (h = 4 s, LQR, bicycle), Eq. 1, "weightEP = 5, weightTTC = 5, weightC = 2", "scoreNC = 0 or scoreDAC = 0 … PDMS of 0" | PUBLISHED, banked sha256 `d3bc66d321cfceccc4f431113dea3f0f481d74d29dd283a33e204a4326f0403c` (equals `library.json`) |
| tier T1; "not closed loop" | `REG` | 5270 | doctrine stamp |
| 136 logs, 10,000 resamples, seed 20260927 | `PK/raw/2026-10-17-zero-tail-mechanism/code/zero_tail.py` | line 36 `SEED = 20260927`; class `Boot` lines 110–128; `RESULT.json` `estimator` | MEASURED |
| P2 picks pose-bit-identical, PDMS-exact 509/509 | `PK/raw/2026-10-15-p2-zero-tail/p2_own_table/P2_MEASURED.json` | `meta.same_pick_proof` (`pose_n_bit_identical` 509, `pdms_n_exact` 509) | MEASURED |
| P2-s1 − P2-s0 = +0.33 [+0.04, +0.64] | `PK/eval/RESULT_P2REP.md` | table, column RR2, row PDMS; text "F = |Δ_seed| = 0.33" | MEASURED |
| CV 20.65 vs published 20.6 | `PK/eval/RESULT_NAVTEST_FINAL.md` line 17; PDF `2406.15349` Table 1 row "Constant Velocity" PDMS 20.6; `PK/eval/SPEC_NAVTEST.md` line 12 (CV 20.6517) | | MEASURED / PUBLISHED |
| human 94.55 [93.78, 95.23] vs published 94.8 | `PK/eval/RESULT_P4_TEACHER_NAVTEST.md` line 38 (second column); PDF `2609.06055` Table 4 "Human Driver … 94.8" and `2406.15349` Table 1 "Human … 94.8" | | MEASURED / PUBLISHED |
| STOP 61.82; "EP == 1 by rule when compliant progress ≤ 5 m" | `PK/eval/RESULT_NAVTEST_FINAL.md` line 15; `PK/eval/SPEC_NAVTEST.md` line 12 | | MEASURED |
| "41 points" (61.82 − 20.65 = 41.17) | derived | | arithmetic |
| refcv4b A1 = 58.97, below STOP | `PK/eval/RESULT_NAVTEST_FINAL.md` | line 16 | MEASURED (a different, earlier programme model) |
| A7: 60.76 → 76.96, +16.20 [+12.83, +19.52], 923 tokens / 43 logs, epoch-15 snapshot, registered before its data | `REG` | 5252 | MEASURED; `…/eval/RESULT_A7_a7confirm_ep015.md` named there |
| A9: +1.22 [+0.59, +2.07] on the full navtest; adopted on the PI's ruling 2026-10-04; reads the roadblock graph, only the nav goal moves | `REG` | 5246 ("Amendment 9 … DEPLOYED 2026-10-04 on the PI's ruling", "paired +1.22 [+0.59, +2.07]", "it reads roadblock-graph topology … only the nav goal moves") | MEASURED |
| A8 not deployed | `REG` | 5251 | MEASURED / PI ruling |
| selection formula NAVSIM v1; u400 × p(lane) | `REG` | 5250 (DEFAULT_RULE navsim_v1), 5275 (`navsim_v1_lane` = navsim_v1 × p(lane)) | MEASURED |

## 3. (b) The ladder and the gap

| number | primary file | line / key | class |
|---|---|---|---|
| model_final 83.24 [81.77, 84.63] | `REG` 5246; `marginal_ci.json` `rows.model_final` | 83.2363 [81.7709, 84.6301] | MEASURED, RE-MEASURED |
| + A9 84.46 [83.18, 85.77] | `REG` 5246 ("PDMS 84.46 [83.18, 85.77]"); recompute `rows.model_final_routefix_A9` [83.1563, 85.6897] | | MEASURED |
| u400 88.25 [87.05, 89.36] | `REG` 5275 (point); `marginal_ci.json` `rows.sft4_B_u400` 88.2469 [87.0479, 89.3621] | | MEASURED; interval new |
| P2-s0 88.82 [87.86, 89.77] | `GAC` 16702 (point 88.82); `marginal_ci.json` `rows.p2_s0` 88.8184 [87.8611, 89.7701]; `RESULT_P2REP.md` "P2-s0 88.8184" | | MEASURED; interval new |
| P2-s1 89.15 | `PK/eval/RESULT_P2REP.md` "PDMS means … P2-s1 89.1493" | | MEASURED |
| teacher 89.79 [88.55, 90.97] | `REG` 5277 (89.79); `RESULT_P4_TEACHER_NAVTEST.md` line 58 (89.79 [88.55, 90.97]); recompute 89.7861 [88.5474, 90.9737] | | MEASURED |
| human 94.55 [93.78, 95.23] | `REG` 5278 (94.55); `RESULT_P4_TEACHER_NAVTEST.md` line 38; recompute 94.5514 [93.781, 95.2328] | | MEASURED |
| published matched row 94.55; published human 94.8 | PDF `2609.06055` Table A13 / Table 4 | | PUBLISHED |
| SFT-4 over model_final + route fix: +3.79 [+3.05, +4.54]; 0 of 21 adverse | `REG` 5280; `PK/eval/RESULT_SFT4.md` lines 134–136 and 186 | | MEASURED |
| P2-s0 over u400: +0.57 [+0.02, +1.18] | `GAC` 16702; `RESULT_P2REP.md` table col 3 | | MEASURED |
| P2-s1 over u400: +0.90 [+0.36, +1.59]; TTC −0.49 [−0.97, −0.07] adverse; NOT REPLICATED | `PK/eval/RESULT_P2REP.md` table col 1 and "The committed outcome"; `GAC` 16703 | | MEASURED |
| gap 6.30 [5.39, 7.23]; terms 4.15 [3.31, 5.07], 1.31 [1.09, 1.53], 0.86 [0.41, 1.28], −0.01 [−0.03, +0.00]; shares 65.9 / 20.7 / 13.6 / −0.2 % | `gap_decomp_ci.json` `terms.*`; registry statement `REG` 5283, `GAC` 16250; raw `PK/raw/2026-10-11-ep-gap-decomposition/ep_lowtail.out` line 1 | | RE-MEASURED (intervals new; points equal the banked line) |
| 529 of 12,146 = 4.36 %; human 0 zero picks | `gap_decomp_ci.json` `zero_picks_u400` 529, `zero_picks_human` 0; `GAC` 16250 ("529 tokens exactly zeroed") | | RE-MEASURED |
| TTC = 0 on 4.37 % of u400 tokens, human 0 % | `ep_lowtail.out` line 2 | | MEASURED |
| "order-dependent" | `ep_lowtail.py` header comment "exact, sequential: gate first" | | method statement |
| published student profile NC 98.80, DAC 99.25, TTC 95.74, EP 92.72; u400 NC 98.94, DAC 96.58, TTC 95.63, EP 81.20 | PDF Table A13; `REG` 5275 | | PUBLISHED / MEASURED |
| human's published EP 87.5 | PDF `2609.06055` Table 4 "Human Driver … 87.5" | | PUBLISHED |
| "corrects an earlier internal reading: EP is the largest gap" | `GAC` 16250 ("Corrects `P6_CONFIDENCE.md`'s 'EP is the largest gap'") | | register statement |

## 4. (c) The zero tail is selection

| number | primary file | line / key | class |
|---|---|---|---|
| u400: 529 zeros, 455 SEL / 74 PROP, SEL share 86.0 % [80.1, 91.7], ceiling +3.62 [+2.85, +4.55] | `PK/raw/2026-10-15-p2-zero-tail/RESULT.json` | `Q1a_u400_own.SEL_def`: `n_zero` 529, `n_SEL` 455, `n_PROP` 74, `SEL_share` 0.8601 [0.8011, 0.9174], `oracle_on_SEL_gain_pdms_pts` 3.6201 [2.8455, 4.5541] | MEASURED |
| definitions of SEL / PROP / oracle-on-SEL | `PK/raw/2026-10-15-p2-zero-tail/PREREG.md` | lines 51, 55 | method |
| earlier bridged P2 estimate 439 / 70 (INFERRED) | same `RESULT.json` | `Q1c_P2_bridged.variants.V1_hierarchy.SEL_def` (`n_SEL` 439, `n_PROP` 70, 3.4913) | INFERRED, superseded |
| P2 own table: 509 zeros, 445 SEL / 64 PROP, 87.4 % [81.0, 93.2], ceiling +3.54 [+2.90, +4.23] | `PK/raw/2026-10-15-p2-zero-tail/p2_own_table/P2_MEASURED.json` | `Q1_SEL_PROP` (`n_SEL` 445, `n_PROP` 64, `SEL_share` 0.8743 [0.8098, 0.9320], `oracle_on_SEL_gain_pdms_pts` 3.5377 [2.8990, 4.2313]); `GAC` 16711 | MEASURED |
| P2 cause split: DAC-only 376 (73.9 % [66.7, 80.3]), NC-only 129, both 4 | same | `Q2_cause_split.all` | MEASURED |
| NC-type zeros 130 / 133 SEL | same | `Q2_cause_split.SEL` (NC_only 127 + both 3 = 130) over `all` (129 + 4 = 133) | MEASURED, arithmetic on those keys |
| DAC-only SEL carry +2.50 [+1.88, +3.20] | same | `Q2_cause_split.gain_by_cause_pdms_pts_SEL.DAC_only` 2.5018 [1.8814, 3.1951] | MEASURED |
| 59 of 64 PROP tokens have all 64 off-road | same | `PROP_failure_kinds.all_DAC0` 59 | MEASURED |
| PROP ceiling ≈ 0.49 | same | `context.proposal_side_ceiling_pts_ESTIMATED` 0.4885 | ESTIMATED |
| median 19 non-zero proposals of 64 | same | `fan_cleanliness_POSTHOC.median_nonzero_of_64` 19.0 | MEASURED, POST-HOC |
| turning enrichment: left ×1.59 [1.12, 2.18], right ×1.89 [1.30, 2.70], straight ×0.65 [0.47, 0.81] | `PK/raw/2026-10-17-zero-tail-mechanism/RESULT.json` | `arms.A.Q2.command.{left,right,straight}.enrichment` (1.5858 [1.1151, 2.1778]; 1.8886 [1.3004, 2.7044]; 0.645 [0.4727, 0.8073]) | MEASURED |
| TTC: 394 non-zeroed tokens with TTC = 0; 96.7 % have a TTC-ok proposal; ceiling +1.44 [+1.17, +1.72] | `PK/raw/2026-10-15-p2-zero-tail/REPORT.md` section "Q4: TTC"; `RESULT.json` `Q4_TTC` | | 394 MEASURED; 96.7 % and ceiling INFERRED (proxy fan) |
| "pre-registered before any count" | `GAC` 16710 ("`PREREG.md`, written and amended before any count"); `REPORT.md` provenance (file order PREREG 21:40:00 < zero_tail.py 21:43:51 < RESULT.json 21:44:04) | | MEASURED (file order) |

## 5. (d) Near-twin grazes

All from `PK/raw/2026-10-17-zero-tail-mechanism/RESULT.json` unless a file is named (`arms.A` = u400,
`n_target` 343 = DAC-only SEL zeros, of `n_DAC_only_zero` 413 and `n_zero` 529).

| number | key | class |
|---|---|---|
| gates: 2,996/2,996; tracked-dump identity max 0.0 m; human DAC = 1 on 1,224/1,224 | `gates.G1_all_scored_trajectories`, `gates.G2_tracked_dump_identity`, `gates.G3_human_DAC1` | MEASURED |
| PREREG hashed before any number; two amendments before any target value | `GAC` 16712; `MANIFEST.json` `prereg_chain` (amendment1 13:46:19Z, amendment2 13:47:57Z); `PREREG_ZT3.md` lines 165, 177 (headings) | MEASURED |
| 60.1 % [51.4, 68.0] ON_PATH (206 of 343) | `arms.A.Q3.class_share.ON_PATH` 60.0583 [51.4105, 67.9881]; `readings.A.top_class_share` | MEASURED |
| median 0.29 m lateral, length ratio 1.01, heading diff 0.04°; ROUTE_DIFF 1.7 % [0.3, 3.6]; manoeuvre match 95.3 % | `report.mechanism.near_twin` | MEASURED |
| 1.55 [1.06, 2.15] of 2.55 [1.83, 3.41] PDMS at stake | `arms.A.Q3.class_pdms_at_stake.ON_PATH` 1.5508 [1.0593, 2.1538]; `total_pdms_at_stake` 2.5456 [1.8322, 3.4106] | MEASURED |
| P2 replicates, 63.2 % | `readings.replication_P2` (`same_top_class` true, `share_diff_pts` 3.12); `report.headline` "P2 63.2 %" | MEASURED |
| depth < 0.5 m on 74.1 % [66.8, 80.5], median 0.23 m | `arms.A.Q1.GRAZE_depth_lt_0.5m` 74.0525 [66.8218, 80.5447]; `depth_m.median` 0.234 | MEASURED |
| t* median 3.1 s; d* median 17.9 m; NEAR 38.5 % [28.2, 48.6]; FAR 61.5 % | `arms.A.Q1.t_star_s.median`, `d_star_m.median` 17.869, `NEAR` 38.484 [28.1858, 48.6074], `FAR` 61.516 | MEASURED |
| DAC_INVERTED 72.0 % [63.9, 78.5] | `arms.A.Q4.DAC_INVERTED` 72.0117 [63.9098, 78.5126] | MEASURED |
| within-token AUC 0.46 [0.36, 0.57] vs 0.92 [0.90, 0.94] | `arms.A.Q7.target_mean_auc` 0.4621 [0.3638, 0.5671]; `random_mean_auc` 0.9243 [0.9032, 0.9449] | MEASURED, POST-HOC (conditioned on failure) |
| near-twin pair accuracy 0.39 [0.27, 0.57] vs 0.96 [0.94, 0.98] | `PK/raw/2026-10-17-zero-tail-mechanism/ZT4_BASELINE.json` `A_u400.target_DAC_SEL.pairwise_DAC_accuracy` 0.3921 [0.2677, 0.5706]; `random_stratum` 0.9625 [0.9372, 0.9821] | MEASURED, POST-HOC |
| readings R1 GEOMETRY, R2 EXECUTION_ARM, R3 HEAD_WRONG, R4 FAR, R5 NEAR_BLIND; R4 meaning "anticipation problem" | `readings.A` (R1–R5); committed meanings `PREREG_ZT3.md` lines 139–155 (§5 "Committed readings") | MEASURED |
| 42.0 % [33.9, 49.4] tracker-made (144 of 343) | `arms.A.Q6_origin.TRACKER_MADE` 41.9825 [33.8843, 49.4383] | MEASURED |
| lever "reach" (gross): 58.0 % / 1.48 [1.04, 1.95]; 42.0 % / 1.07 [0.71, 1.55]; 25.7 % / 0.65 [0.43, 0.89]; 2.9 % / 0.06 [0.01, 0.13] | `arms.A.Q5.P6_trunk_DAC_gradient.Y_share` 58.0175, `.Y_pdms_at_stake` 1.4754 [1.0393, 1.9455]; `proposal_side` 41.9825 / `report.lever_coverage_u400.proposal_side` "Y-PDMS 1.07 [0.71,1.55]"; `P7_1_map_supervision` 25.656 [19.9184, 32.3371] / 0.6516 [0.427, 0.8865]; `P6_lane_rank_twins` 2.9155 [0.6023, 5.4124] / 0.0647 [0.0116, 0.1318] | MEASURED, **gross** (R47) |

## 6. (e) The execution side and the correction

All from `PK/raw/2026-10-17-tracking-execution/RESULT.json` unless a file is named.

| number | key | class |
|---|---|---|
| 144 picks / 44 logs, 529 controls, gates 673/673 | `report.mechanism_navtest_DAC_level.units` | MEASURED |
| rescue: heading 56.9 % [45.1, 66.0]; actuator lag 47.2 % [38.5, 57.3]; steering 18.1 % [12.0, 26.3]; speed 7.6 % [3.3, 13.2]; controls broken (heading) 1.7 % [0.2, 3.8] | `report.mechanism_navtest_DAC_level.rescue_single_factor` (H, A, S, V) and `.controls_broken_weighted.H` | MEASURED |
| "under-turns and runs wide late" | `report.mechanism_navtest_DAC_level.form` | MEASURED |
| dev set 2,466 sets / 13 logs; baseline 89.28 [85.53, 91.61] | `report.dev.set`, `report.dev.baseline_N-s0_PDMS_x100` | MEASURED |
| F_c −0.30 [−0.72, +0.04], 19 rescued / 32 broken | `report.dev.F_c` (`dPDMS`, `rescued`, `broken`); `dev.effects` | MEASURED |
| F_b −0.03 [−0.15, +0.07]; F_bc −0.19 [−0.68, +0.27]; F_m +0.00 [−0.06, +0.05]; F_a −0.19 [−0.76, +0.63] | `report.dev.F_b`, `.F_bc`, `.F_m_amendment3`, `.F_a_ceiling`; `dev.eligibility` (all not eligible) | MEASURED |
| definitions of F_a, F_b, F_c, F_bc, F_m | `PREREG_ZT5.md` lines 103–112 (F_c, F_b, F_bc, F_a), 205 (F_m, Amendment 3) | method |
| eligibility bar (lower bound > 0 and point ≥ +0.05, positive on N-s1 and N-s2, no adverse sub-score) | `PREREG_ZT5.md` lines 138–141 (§6) | method |
| F_c "generalises A7" | `PREREG_ZT5.md` line 103 | method |
| no navtest confirm | `report.navtest_confirm` ("NOT ISSUED: PREREG section 6 …") | MEASURED |
| one dev set failed the strict gate, EP only, 0.0286, weight 1/2,466 | `report.dev.DG1_note` | MEASURED |
| tracker saves 27, makes 20; +0.28 pp [−0.88, +1.24] | `posthoc_tracker_saved.dev.N-s0` (`TS_picks` 27, `TM_picks` 20, `TS_minus_TM_pct` 0.2839 [−0.8844, 1.2425]) | MEASURED, POST-HOC |
| navtest ≈ 132 [44, 237] vs 144 | `posthoc_tracker_saved.navtest.TS_tokens_ESTIMATED` 131.76 [43.84, 236.60]; `TM_tokens_MEASURED` 144 | ESTIMATED vs MEASURED |
| 1.07 PDMS gross reach, quoted as a reach by D-REFE-ZEROTAIL-3 and the report to the PI | `RL` line 18242 onward (R47 "Claimed" / "True" / "Class"); `GAC` 16712 (⚠️ CORRECTED 2026-10-17) and 16713 | register / retraction |
| "13 development logs … wide intervals" | `report.limitations` (11 held-out log DBs absent) | MEASURED |

## 7. (f) Four families

All from `PK/eval/RESULT_SFT4.md`, section "Four families (families6, on the same 12,146 tokens, 136 logs)",
table at lines 164–181 (u400 column), JSON `sft4_B_u400_navtest_full_families.json`.

| number | line | class |
|---|---|---|
| EP 81.20, TTC 95.63, NC 98.94, C 99.97, DAC 96.58, DDC 97.66 | `RESULT_SFT4.md` line 148 (sub-score row); `REG` 5275 | MEASURED |
| speed MAE 0.740; along-track MAE 1.035; acceleration MAE 0.519; progress ratio 1.003 | lines 166, 167, 168, 171 | MEASURED |
| heading MAE 2.532; cross-track MAE 0.378; curvature MAE 0.0199; yaw-rate MAE 2.983 | lines 173, 174, 176, 177 | MEASURED |
| tactical: lateral 0.925 / 0.863; longitudinal 0.679 / 0.516; 5-way 0.783 / 0.718 | lines 178–180 | MEASURED, no interval |
| distance-keeping UNAVAILABLE (no lead-agent track in the seam) | line 172 | MEASURED status |
| STRATEGIC UNAVAILABLE in NAVSIM + PAUSED | line 181 | status |
| PI ruling 2026-09-19, quote, ITEM 25 | `Project Steering/PI_DECISION_QUEUE.md` line 1446 (verbatim quote); heading line 1321 | PI ruling |
| anti-echo BEATS_HOLDV0, copy detector clean, echo index 0.0013 vs human 0.0338 | `RESULT_SFT4.md` lines 196–197 (the "Anti-echo" bullets of the full-navtest section) | MEASURED |
| SFT-4: every MAE improves | `RESULT_SFT4.md` bullet "Every MAE point estimate improves" (line 188) | MEASURED |
| P2-s0 vs u400: 3 of 21 longitudinal components adverse, all speed tracking | `GAC` 16702 ("Families vs u400 …") | MEASURED, register (artifact: `PK/raw/p2_readout/p2_full/r2_result.json`, not opened here) |
| P2-s1 vs u400: 21 components, none adverse | `PK/eval/RESULT_P2REP.md` "Families (RR3 …)" | MEASURED |

## 8. (g) Next

| number / claim | primary file | line / key | class |
|---|---|---|---|
| P6 = paper-faithful full REFe run; un-detached scorer context; PDM targets online on own proposals; ≈ 7 A40-days | `PK/SPEC_P6_PAPER_FAITHFUL.md` | lines 1–12 (header), 118–120 (design table), §0 ("Only with the zero-extra-forward labeller … 9.2–13.5 days otherwise") | design / ESTIMATED |
| P6 only lever with Y on ON_PATH; gross 58 % / 1.48 | `RESULT.json` | `report.lever_coverage_u400.P6_trunk_DAC_gradient` | MEASURED, gross |
| ZT4 design: zeros ≤ 240 (−30 %) with paired CI excl. 0 and near-twin accuracy ≥ 0.70 → TRUNK-REACHES; reduction < 15 % or CI includes 0 or accuracy ≤ 0.50 → TRUNK-MISSES; PARTIAL otherwise; ≤ 1 dev-box GPU-h; control 0.39 | `PK/raw/2026-10-17-zero-tail-mechanism/RESULT.json` | `report.next_experiment_ZT4` (`TRUNK_REACHES`, `TRUNK_MISSES`, `PARTIAL`, `cost`, `primary`); control = `ZT4_BASELINE.json` | pre-registered |
| S1 not promoted: head DAC AUC 0.772 / 0.765 / 0.781 vs scorer 0.787; navtest P1 IoU [0,40) 0.496 / 0.521 / 0.486 vs 0.80 bar | `GAC` 16709 (D-REFE-S1-READ); `PK/raw/2026-10-15-s1-read/REPORT.md` (condition table, rows 1 and K2 text) and `S1_READ.json` | | MEASURED (T0-like held-out selection and perception, never driving) |
| P5-C needs the PI's GPU go | `GAC` 16713 ("Training-side margin targets (P5-C) need the PI's GPU go") | | register |
| 3-C twin bank, PI 2026-10-17 | `GAC` 16716 ("PI DECIDED 3-C on 2026-10-17") | | register |
| P6 replicate with the +0.33 floor | `RESULT_P2REP.md` ("F = 0.33 is a VALID read" for P6); `GAC` 16703 | | register |

## 9. Not sourced from the registry, and why

`MODEL_REGISTRY.md` §14.2 carries no P2 (88.82), P2-s1 (89.15), P6 or zero-tail row at a5a4ee8
(`grep -c "88\.82"` = 0; `grep -ci "zero tail"` = 0). Those numbers are quoted from the raw artifacts and the
register rows that cite them, which is the best primary available; see `CONFLICTS.md` C3.
