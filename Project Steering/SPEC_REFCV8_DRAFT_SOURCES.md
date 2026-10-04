# SPEC_REFCV8_DRAFT — sources: every number, its artifact, its evidence class

*Companion to `Project Steering/SPEC_REFCV8_DRAFT.md`. Written 2026-10-04 ~21:45 Berlin by the design agent. Paths are
repo-relative to `D:/Projects/TanitAD/` unless marked otherwise. Abbreviations used for package roots:*

| key | package root |
|---|---|
| `WPA` | `FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/` |
| `WPB` | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-tactical-conditioning/` |
| `WPC` | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-fixes/` |
| `WPD` | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-perception-architecture/` |
| `R1` | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv8-r1-head-only/` |
| `ROUTE` | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-route-following/` |
| `DIAG` | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-map-box-diagnostics/` |
| `AUDIT` | `TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/` |
| `BAT` | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/` |
| `NAV` | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/` |

**Evidence classes:** MEASURED (programme measurement; artifact named) · PUBLISHED · INHERITED (another doc, not
re-verified against its raw artifact by me) · ESTIMATED · ANALYTIC. The column "opened" says whether I opened the raw
artifact and found the value (**raw**), or read it from the package's own RESULT / design document (**doc**).

⚠️ **Registry gap (SPEC §13 Q6).** `Project Steering/MODEL_REGISTRY.md` on D: (mtime 2026-10-01; repo HEAD
`37645fc`) carries **no refcv7 row**. PLAN_REFCV8 cites the block `REFCV7-2026-10-04-FINAL`, which exists only at the
lander tip, outside D:. No number below is quoted from the registry; every refcv7 number is from raw JSON or a package
document on D:, as listed.

---

## 1. Models, checkpoints, releases, registrations

| number | SPEC § | artifact | class | opened |
|---|---|---|---|---|
| refcv7-50,400 model-only export md5 `b418d0fc4a92a6848c246a6a7c50207b` | 3.1 | `WPB/raw/r8_fullsize_warmstart.json` (`warm_start_from.md5`) | MEASURED | raw |
| v9 train release md5 `f63ece410b725febb8a5242cf2b01d3c`; eval139 md5 `6b5c7f207cffc3b7eb3cd527fd433599` | 1, 3.2 | `WPA/RESULT.md` §S3.0; same values in the WP-B launch-gate overlay `WPB/code/fix/stack/scripts/launch_gate.py` (`required_values`) | MEASURED | doc + source |
| WP-B overlay base tip `5388a438611e8c401becdacbf66bb40d7f5bd247` | 3.2, 8.3 | `WPB/code/fix/BASE_BLOBS.txt` | MEASURED | raw |
| refcv8 launch-gate profile: required flags, required values, forbidden levers, four open items | 3.2, 10.3 | `WPB/code/fix/stack/scripts/launch_gate.py` lines 621–680 | MEASURED (source read) | raw |
| smoke argv: 197 tokens, incl. `--speed-max-sidecar-v6 …/refcv6_speed_max_v8_train.jsonl`, `--trunk-compile`, `--w-r8-sat 0.1 --w-r8-subscore 0.5 --r8-lat-prior-dropout 0.3 --r8-n-alloc 32 --r8-alloc-emit` | 3.2, 8.3, 10.3 | `WPB/code/fix/stack/ops/runs.d/refcv8-wpb-smoke.argv.json` | MEASURED | raw |
| no `--r8-*` flag for N2 / N3 or unknown-row dropout | 8.3 | `WPB/code/fix/stack/scripts/refc_v3_train.py` (grep of `add_argument("--r8`) | MEASURED (source read) | raw |
| SPEC_WPB sha256 `c952d4b45a55dfd422c920264f900057e4876780d81d01b859b6695c2d60e04f` | 1, 10.3 | `WPB/raw/SPEC_SHA256.txt` | MEASURED | raw |
| SPEC_D0 sha256 `7c7e4f52…` | 14 | `WPB/raw/SPEC_SHA256.txt` | MEASURED | raw |
| SPEC_R1 sha256 `e9873e12…`; SPEC_R1_A1 sha256 `cf3caf0b…` | 10.2 | `R1/raw/SPEC_SHA256.txt` | MEASURED | raw |
| PREREG_WPD_PROBES sha256 `c054190b…`; PREREG_WPD_A1 sha256 `eb6de12b…` | 10.1 | `WPD/raw/SPEC_SHA256.txt` | MEASURED | raw |
| WP-A SPEC sha256 `d26ed9e8…`; S2A2 sha256 `a44bd428…` | 4.1, 4.3 | `WPA/raw/SPEC_SHA256.txt` | MEASURED | raw |
| route SPEC A6 sha256 `b4099578…`, A7 `8e846a84…` | 2.1 | `ROUTE/raw/SPEC_SHA256.txt` | MEASURED | raw |
| battery A7 registered 2026-10-04T17:11:57+02:00 (sha256 of the text `46a82d1e…`) | 5 | `BAT/SPEC.md` (A7 header) | MEASURED (doc is the registration) | doc |
| D6 SPEC_P1P2_A2_P1PRIME registered 15:14:40Z, sha256 `0ccd6c5a…` | 6.2 | `AUDIT/D6_navsim_failure_anatomy/RESULT.md` §13.1 | INHERITED | doc |
| PI cost ceiling `max_s_per_step` 10.5 | 10.3 | `Project Steering/SPEC_REFCV7.md` §25 (file `work/refcv7/launch/pi_cost_approval.json`, not on D:) | INHERITED | doc |

## 2. Warm start (SPEC §3.1)

| number | artifact | class | tier | opened |
|---|---|---|---|---|
| variant A (seams, no allocation): 4 / 4 windows `traj`, `sel_idx`, base fan equal; base score max diff 0.0; 1,131 keys loaded; 28 new keys all refcv8 seams; 122,513 new params; 0 G-DVB mismatches; eval v9 join 23,772 / 23,772, 3 tactically excluded clips | `WPB/raw/r8_fullsize_warmstart.json` | MEASURED (dev-box CPU) | identity check | raw |
| variant B (allocation 32, emission off): `traj` and base fan NOT equal on 4 / 4; `sel_idx` equal; base score max diff up to 5.72e-6; `identity_PASS: false` | same | MEASURED | identity check | raw |
| Thor I-W: seams-only bit-identical on 218 windows; allocation + prior-free: traj differs on 205 / 218, base fan on 218, base scores up to 2.1e-5 | `WPB/code/iw_diag.py` docstring (artifact `arms/I_W.json` on Thor, not on D:) | INHERITED | identity check | doc |
| refcv7-r101-s0 total params 100,468,987 | `WPB/DESIGN.md` stamps (from `D:/refcv7_eval_kit/ckpt/config.json`, outside the repo) | INHERITED | — | doc |

## 3. Labels and data audit (SPEC §4.1, §4.2, §8)

| number | artifact | class | opened |
|---|---|---|---|
| refcv7 tactical labels on 173,409 / 746,946 = 23.22 % train windows; 5,404 / 23,772 = 22.73 % eval | `AUDIT/D1_label_census/COVERAGE.md` rows 1 and table (raw: `D1_label_census/tables/label_truth_*.npz`) | MEASURED (D1) | doc |
| refcv7 logged tactical rows per window 0.2357 [0.2295, 0.2426] vs census 0.2322 (control K2) | `AUDIT/D1_label_census/COVERAGE.md` control K2 | MEASURED (D1) | doc |
| clip-token "turn but no turn starts within 6 s": 205,292 = 74.3 % of 276,434 L/R windows | `AUDIT/D1_label_census/COVERAGE.md` row 3 | MEASURED (D1) | doc |
| `LANE_CHANGE_L` scored on 173,409 windows vs census 12,145 | `AUDIT/D1_label_census/COVERAGE.md` goal table; WPB `DESIGN.md` §3.6 | MEASURED (D1) | doc |
| v9 V1 coverage train 95.09 % (710,256) [95.04, 95.14]; eval139 94.57 %; 95.58 % on 136 tactically scored clips | `WPA/RESULT.md` §S3.1 (raw `WPA/raw/s3_validation_{train,eval139}.json`) | MEASURED (WP-A) | doc |
| V2 TURN side train 0.980 / 0.988 (n 113,946); eval 0.943 / 0.966 (n 3,332); STOP 0.991 / 0.994 train, 0.999 / 0.998 eval | same | MEASURED (WP-A) | doc |
| V5 lane change side-correct 14 / 101; V7 ⇒ `H_ABS_MIN` 8.0 | same | MEASURED (WP-A) | doc |
| V8 spatial token "no turn within 6 s" 22.2 % (35,640 / 160,669) train, 30.4 % eval; realised announced turns matched 99.55 % (125,029 / 125,591) train, 99.34 % eval | same | MEASURED (WP-A) | doc |
| E1 trivial planner RC-A50: direction 0.916, heading-15 0.654 [0.531, 0.763] (1,071-window grid); 0.640 on all eval139; deranged RC direction 0.178 | `WPA/RESULT.md` §S2.1 (raw `WPA/raw/s2_echo_leak.json`) | MEASURED (WP-A) | doc |
| E2′ RC-A50 noised Δ −0.035 (bar ≤ 0.01) PASS; instrument C1 −0.0045, C2 0.997, O1 0.379; clean point +0.021 | `WPA/RESULT.md` §S2.7 (raw `WPA/raw/s2_e2prime.json`) | MEASURED (WP-A) | doc |
| 693 ego-footprint boxes in 18 train clips | `WPC/raw/join_masks_acceptance.json` (`fix3.footprint_boxes_before` 693 → after 0); also `WPA/raw/refcv8_corpus_manifest.json`, `AUDIT/D3_raw_plausibility/RESULT.md` D-2 | MEASURED | raw |
| largest \|v_rel\| 764.1 → 77.9 m/s; rows > 100 m/s 5 → 0; 3,470 rate rows masked (40 jump clips) | `WPC/raw/join_masks_acceptance.json` (`fix4`) | MEASURED | raw |
| 3 (image-confirmed) to 5 eval clips share a recording with train; 2 duplicate train videos | `AUDIT/D3_raw_plausibility/RESULT.md` D-1 | MEASURED (D3; MM transcription) | doc |
| poses lead the image 0–34 ms, mean 16.5 ms ⇒ 0.19 m mean; ~8 % of clips dark vs 46 % "night"; left turns on 13 eval clips | `AUDIT/D3_raw_plausibility/RESULT.md` MINOR | MEASURED (D3) | doc |
| NUDGE 23.5 % of the lateral mass; 10.3 % / 12.1 % corroborated | `AUDIT/D4_usage_and_design/USAGE_AUDIT.md` line 70; PLAN K10 (D2) | MEASURED (D2 / D4) | doc |
| 3 of 8 lateral and 1 of 8 longitudinal classes with zero windows | PLAN_REFCV8 X5 (D1) | INHERITED | doc |
| speed-input leak (recovered future-speed share, 576,555 train windows): N0 0.0, N2 0.0346, N3 0.0193, N2 + unknown p 0.45 0.0093, window oracle O1 0.5698 | `AUDIT/D4_usage_and_design/raw/d4_vmax_nonoracle.json` | MEASURED (D4) | raw |
| refcv7 per-clip value recovers 23.8 % | `AUDIT/D4_usage_and_design/USAGE_AUDIT.md` lines 93–94, 268 | MEASURED (D4) | doc |
| the unknown speed row fed on 45 % of navtest tokens | `USAGE_AUDIT.md` line 106 ("INHERITED, NavSim" there) | INHERITED | doc |
| agent store x ≤ 61 m excludes 84 % of joined agents | PLAN_REFCV8 K17 / X7 (D4) | INHERITED | doc |
| 110 / 2,059 reel windows with the plan above the fed ceiling | PLAN_REFCV8 X3 | INHERITED | doc |
| 342 MB of streaming per clip for LiDAR; ≈ 1.5 TB over 4,369 clips | PLAN_REFCV8 decision 9; `WPD/PREREG_WPD_PROBES.md` P-DEPTH (MEASURED 2026-09-11 there) | 342 MB INHERITED; 1.5 TB ESTIMATED | doc |

## 4. Route following, selection, tactical (SPEC §4.2–4.4, §8.1–8.2)

| number | artifact | class | tier | opened |
|---|---|---|---|---|
| S-ROUTE: 4,634 windows; 2,317 GT-turn (849 left / 1,468 right); 1,347 straight, 276 gentle, 694 unclassified in the sample; turn windows from 43 episodes, all 139 episodes present | `ROUTE/raw/a6_window_list.json` (counted) | MEASURED | — | raw |
| refcv7 V0 on S-ROUTE, seed 0: turn direction 0.8153 [0.7299, 0.8847]; turn ADE 3.2967 m; straight ADE 1.6783 m; straight direction 0.9629; nav complies 0.6853; turn speed MAE 0–2 s 0.3465, straight 0.2515; \|along\| 6 s 7.3676 / 4.9952 m; terminal heading 23.7707° / 2.2889°; curvature MAE 0.0098 / 0.0016; \|cross\| 6 s 6.3295 / 1.0478 m | `ROUTE/raw/a6_dense_score.json` (`dense_seed0.V0`) | MEASURED | T1 open loop | raw |
| oracle-117 on S-ROUTE: turn ADE 1.225; ΔADE vs V0 −2.0718 [−2.4477, −1.7426] (⇒ REGRET 2.072 m); oracle direction 0.9525 | same (`dense_seed0.ORACLE`) | MEASURED | T0 ceiling | raw |
| refcv7 on S-GRID: E9 pick direction 0.8411 [0.7423, 0.9197] (n 107); heading-15 0.514 [0.4228, 0.6027]; random-in-reach 0.3849; fan-contains 1.0 | `ROUTE/raw/route_analysis.json` (`chain_eval_s0`) | MEASURED | T1 | raw |
| D_core +0.1589 [+0.0803, +0.2577] | `ROUTE/raw/route_analysis.json` (`drops`) | MEASURED | T1 | raw |
| D_core seed 1 +0.150 | `ROUTE/RESULT.md` line 32 | MEASURED (route) | T1 | doc |
| refcv7 S-ROUTE heading-15 0.505 [0.432, 0.579] seed 0 / 0.510 seed 1; direction 0.813 seed 1; fan-contains 1.000 | `R1/RESULT_R1.md` §3 "H0" (raw on Thor `/home/nvidia/refcv8_r1/`) | INHERITED | T1 | doc |
| A6 T3a (announced nav) turn ΔADE −0.131 [−0.207, −0.062] etc. (context only) | `ROUTE/RESULT_A6.md` §A.4 (raw `ROUTE/raw/a6_dense_score.json`) | MEASURED (route) | T1 | doc |
| tactical side-correct on GT turns 0.4019; lat3 side-collapsed 0.7375 on 800 | `WPB/raw/d0_dose_response.json` (`acc3_turn`, `acc3_all_classified`) | MEASURED (D0) | T1 | raw |
| refcv7's own progress predictor conditioned on: ΔADE all +0.2127 [+0.1207, +0.3131] (seed 1 +0.239) | `WPB/raw/d0_dose_response.json` (`H_E8.eval_s0.dADE_all`) | MEASURED (D0) | T1 | raw |
| refcv7's progress head median relative chord error 0.176 | `WPB/DESIGN.md` §2.3 | INHERITED | — | doc |
| D0 σ\* 0.10 ⇒ median relative error 6.74 %; q\* 0.95; D0c heading 10–15° | `WPB/DESIGN.md` §2; `WPB/SPEC_WPB.md` §5 B-TAC | MEASURED (D0) / post hoc (D0c) | label-side simulation | doc |
| lat3 accuracy 0.856115 vs majority 0.864508 (834 windows); macro-F1 0.492324; lon accuracy 0.418465, macro-F1 0.351357; lon majority 0.4556 | `WPB/raw/r84_baseline_refcv7.json` | MEASURED | T1 | raw |
| pick vs tactical argmax 0.88625 (all classified, 800) / 0.439252 (GT-turn, 107); candidate share at the tag 0.416 (lat3, turn) / 0.3518 (side, turn); permuted-tag null 0.3615 (turn) | `WPB/raw/r84_baseline_refcv7.json` (`iii_consistency`) | MEASURED | T1 | raw |
| residual prior: pick correct 0.6739 when the prior is wrong (n 46), 0.9672 when right; 17 wrong turn picks, 14 on the prior's side; seed 1: 0.6957, 16 / 13 | `WPB/raw/d0b_prior_follow.json` (`eval_s0g`, `eval_s1`) | MEASURED | T1 | raw |

## 5. Perception (SPEC §7)

| number | artifact | class | tier | opened |
|---|---|---|---|---|
| F1 map IoU (refcv7 + F1): nocls 0.6082, drivable 0.5764, lane 0.1643, crosswalk 0.1052, arrow 0.0588, edge 0.0421, hatched 0.0616, sidewalk 0.5001 (\|Δ\| 0.0 vs diagnostics) | `WPC/raw/f1_f4_f4b_reproduction.json` (`F1_map_thresholds`) | MEASURED | perception diagnostic | raw |
| lane prior_corrected (as trained) 0.068 [0.056, 0.078]; edge 0.000 | `DIAG/RESULT.md` line 102 (raw `DIAG/raw/M_c.json`) | MEASURED (diagnostics) | perception diagnostic | doc |
| box3d AP@2 m 0.2483 → 0.3494 after NMS (r 2.5 m, gate 0.214477); agent 0.1314 → 0.2996 (r 3.0 m, gate 0.180872); boxes per object at gate 0.2589: box3d 2.1198 → 1.0673, agent 2.596 → 1.0123; conf_ratio 0.995 / 1.0681 | `WPC/raw/f1_f4_f4b_reproduction.json` (`F4b_nms.reproduced`) | MEASURED | perception diagnostic | raw |
| box3d AP@1 m 0.1298, AP@2 m 0.2483 (T0); NMS r 2 m AP@2 m 0.3385 (+0.0903 [+0.0756, +0.1073]), AP@1 m 0.1111 | `WPD/raw/box_probe0.json` (`Q4_ceilings`) | MEASURED | perception diagnostic | raw |
| AP@2 m 30k → 50.4k: 0.199 → 0.248 | `WPD/raw/box_probe0.json` (`Q1_ap_curve.ap2_all`) | MEASURED | perception diagnostic | raw |
| boxes per detected object at the TRAIN P = R gate: box3d 1.9675 (gate 0.2567), agent 3.1230 (gate 0.2307) | `WPD/raw/box_probe0.json` (`Q2_duplicates`) | MEASURED | perception diagnostic | raw |
| box3d median \|Δlon\| 0.960 / 1.565 m, \|Δlat\| 0.379 / 0.456 m at 0–20 / 20–40 m | `WPD/raw/box_probe0.json` (`Q5_loc_by_range`) | MEASURED | perception diagnostic | raw |
| agent AP@2 m 0.1314 | `WPC/raw/f1_f4_f4b_reproduction.json`; `DIAG/raw/B_box.json` | MEASURED | perception diagnostic | raw (WPC) |
| map IoU_2 (`thr_phat`, EVAL): edge 0–20 m 0.23507, edge 20–40 m 0.11763, lane 20–40 m 0.30773 | `DIAG/raw/M_b.json` (`eval/thr_phat/...`) | MEASURED | perception diagnostic | raw |
| P-GRAD whole-trunk projection shares: agent 0.50139, box3d 0.35717, traj 0.00016, map_hires 0.00014, tac_v6 −0.0005; cos(agent, tac_v6) −0.29698 | `WPD/raw/pgrad.json` (`proj_share_on_total`, cosines) | MEASURED | gradient census | raw |
| P-GRAD shared 0.25 m encoder: box3d 0.973, map −0.004 (projection) | `WPD/RESULT.md` §7.2 | MEASURED (WP-D) | gradient census | doc |
| PB0 vs refcv7 step 0: AP@2 m NMS 0.3382 vs 0.3082, +0.0301 [+0.0190, +0.0412] | `WPD/RESULT.md` §7.3 (raw `WPD/raw/pbox_step0_vs_PB0_descriptive.json`) | MEASURED (descriptive, not pre-registered) | perception diagnostic | doc |
| 40–100 m decoded from ~1.3 stride-8 rows; 903 label rows per stride-8 row at 80–100 m | `WPD/RESULT.md` §0, §3 (raw `WPD/raw/geom_bound.json`) | ANALYTIC | — | doc |
| B1 expected +0.03 … +0.07 AP@2 m over 30k steps | `WPD/PERCEPTION_DESIGN.md` §2.0 | ESTIMATED | — | doc |

## 6. Battery and NavSim (SPEC §5, §6)

| number | artifact | class | tier | opened |
|---|---|---|---|---|
| S2: 4,754 windows / 139 episodes | `BAT/SPEC.md` §3.1 | MEASURED (battery SPEC) | — | doc |
| step 50,400: G0-A6 FAIL as coded; G0-A2 PASS; no family numbers | `BAT/raw/step50400/battery_summary.json` (`stages.g0`) | MEASURED | — | raw |
| step 50,400 G0-A6 PASS under the A7 text ruling | `BAT/SPEC.md` A7 ("`raw/step50400/g0_A6_text.json`") | MEASURED (per the registered A7 text) | — | doc |
| H-ESTIM-SEED-1: 6 of 42 cells separated on a same-seed replicate (14.3 %) | `CLAUDE.md` (artifact `…/2026-09-05-withheld-bank-panel/` `panel_report.json`) | INHERITED | — | doc |
| 9.5 % (4 / 42) on another tiny rig | `WPB/SPEC_WPB_LADDER_DRAFT.md` §3 (`…/2026-09-26-yaw-rate-mask/raw/claims/replicate_fp_rate_fixed.json`) | INHERITED | — | doc |
| navtest PDMS ×100: step 30k R7_A1 71.8849 vs STOP 61.8202, Δ +0.1006 [+0.0829, +0.1178], PASS; step 5k 65.5976, PASS; seed floor 0.2483 / 0.0849 | `NAV/raw/milestones/step30000/BARS.json`, `step5000/BARS.json` | MEASURED | NavSim open-loop | raw |
| navhard official EPDMS: step 30k 0.22693 vs STOP 0.29853, CV 0.11482, ECHO 0.14294; Δ −0.0716 [−0.1112, −0.0346], FAILED; step 5k 0.16243, FAILED | same | MEASURED | NavSim open-loop | raw |
| warmup S2-EPDMS-u: step 30k 0.52237 vs STOP 0.52125, NOT PROVEN; step 5k 0.44868, FAILED | same | MEASURED | NavSim open-loop | raw |
| step 50,400: no BARS.json; navtest scoring aborted by the RAM guard (last line 2026-10-04T19:03:48Z) | `NAV/raw/milestones/step50400/runner.log` | MEASURED | — | raw |
| refcv7 A1 arm fed the MAP speed limit | `NAV/SPEC.md` §3 arms table | MEASURED (SPEC) | — | doc |
| NavSim `AgentInput` = ego statuses, cameras, lidars (`navsim/common/dataclasses.py:150-155`); mapping left/forward/right/unknown → 1/0/2/0 | `WPA/INTEGRATION.md` §5 | INHERITED (WP-A read the devkit) | — | doc |
| D6 (step 30k): DAC-zero 26.0 % and NC-zero 15.6 % of stage-2 scenes; 1,563 DAC-zero scenes; NC-zero 94.8 % front collisions; plan +3.87 m/s faster at 4 s (median); premature turns ~59 % of 724 scenes; P3 ceilings: route / lateral +0.113 EPDMS, speed +0.061, deficit to STOP −0.072, speed fix clears 46.0 % [40.4, 51.2] of NC-zero; STOP control max abs 8.3e-17; 76 navhard / 136 navtest logs | `AUDIT/D6_navsim_failure_anatomy/RESULT.md` §0, §12 (raw `raw/d6_p3_navhard.json`, `raw/d6_p3_navtest.json`, score frames in `NAV`) | MEASURED (D6) | NavSim open-loop | doc |

## 7. Plan-level figures quoted as context

| number | artifact | class |
|---|---|---|
| ~30,000 steps, ≈ 3.5–4 days at ~10–11 s/step; from scratch ~6 days | `Project Steering/PLAN_REFCV8.md` §0.1, §6 | ESTIMATED |
| refcv7 ran at 9.876 s/step | `Project Steering/SPEC_REFCV7.md` §25 | INHERITED |
| allocation +27 % sampler candidates (117 → 149) | `WPB/DESIGN.md` §3.1.2 | ESTIMATED |

## 8. Values chosen by this draft (not measurements)

R8-1-REACH 0.93 (and ±0.02 control band); NAV-COMPLY 0.95 / NAV-PREMATURE 0.05; R8-3 bar 2 (+0.02) and the RC-OFF ADE
margin reuse of +0.05 / +0.10 m; TAC-LAT-F1 0.80; TAC-LON-F1 0.60; R8-4 (iv) bar 4 (0.90 / 0.60); I-2 literals (1e-3 m,
1e-4); R8-5-BOX 1.20; the NMS radius grid {2.0, 2.5, 3.0} m; R8-6-BOX-4 0.248 (= refcv7's box3d level); X1 REGRET 1.55 m;
X2 0.90; X3 LEAK 0.05, CEIL-OBEY 0.01, UNKNOWN +0.10 / +0.20 m/s; X7 sign test; X10 1 ms. Each is listed with the
refcv7 value that was visible in SPEC §14.
