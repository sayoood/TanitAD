# SCORER-FT Stage 1 read: T vs C (2026-11-24 18:27Z), outcome O3

Card `PK/raw/2026-11-20-scorer-ft/PREREG_SCORER_FT.md` (sha256 `adc1b2ae…`), look 1 of `H-SCORERFT-1`. Read by `PK/eval/scorerft/scorerft_read.py` at the export of `429086ac`
(byte-equal `eval/` to the tree whose `test_scorerft_read.py` read 40/40), driven unattended by `sft_read_s1.sh` (v2, sequential). Evidence class: MEASURED (this folder).
Tier T1 (NAVSIM v1 PDMS, open-loop pseudo-simulation of the executed LQR-tracked plan). Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 136 logs, 12,146 tokens;
it answers "another draw of LOGS", not "another training run" (one seed per arm: training variance NOT measured) and not "another inference run" (deterministic argmax, CTRL-DET).

## Verdict (`VERDICT_stage1.txt`)
**O3, FAIL: the heads rank, the pick does not.** P1 FAIL, G1 FAIL (adverse NC, DAC, TTC). NOT ADOPTED. Stage 2 (Cs, Ts) does not run: its trigger is P1 and G1.
Every control PASS: CTRL-SET, CTRL-T (T, C), CTRL-CSV (T, C), CTRL-SEAM, CTRL-ARGV, CTRL-W0, CTRL-DET (T's FULL seam re-run, bit-identical), CTRL-ID (T, C).

| tracked PDMS x100 | point [95 % CI] |
|---|---|
| T | 91.77 [90.95, 92.56] |
| C | 92.67 [91.90, 93.40] |
| **T - C (P1)** | **-0.9015 [-1.2021, -0.5874]**, separated worse |
| NC | -0.7698 [-1.0052, -0.5417] |
| DAC | -0.5599 [-0.8101, -0.3119] |
| TTC | -2.1077 [-2.5977, -1.6179] |
| EP | +0.9386 [+0.6040, +1.2841] |
| C (comfort) | -0.0082 [-0.0279, 0.0000] |

Both arms are BELOW the goal 94.55.

## Mechanism (64-proposal tables, `read_stage1.json` families.TACTICAL.(b))
| T - C | point [95 % CI] |
|---|---|
| M1 EP Spearman | +0.3534 [0.3340, 0.3748] (moved) |
| regret (oracle - pick) | +0.9320 [0.6091, 1.2408] |
| regret_ep | -0.9107 [-1.0028, -0.8221] |
| regret_gate | +1.2708 [+0.9755, +1.5671] |
| regret_ttc | +0.5214 [+0.3330, +0.7163] |
| zeroed picks (pp) | +1.2597 [0.9670, 1.5496] (C 2.81 %, T 4.07 %) |
| oracle-of-64 | +0.0304 [-0.0311, 0.0901] (not damaged) |

Reading: the rank term worked on the head it targeted (EP ranks far better, EP regret falls by 0.91), and the pick paid for it in the gates: it moves to higher-progress
proposals that NC / DAC / TTC then zero. That is the card's O3 case: head-level ranking does not transfer to the decision.

## Every separated-adverse family row (T - C; 25, per the PI's every-row rule)
| row | T - C [95 % CI] | n |
|---|---|---|
| LONGITUDINAL.headway_min_m | -0.1032 [-0.1807, -0.0214] | 6,192 |
| LONGITUDINAL.time_gap_min_s | -0.0422 [-0.0621, -0.0229] | 5,660 |
| LONGITUDINAL.min_ttc_s | -0.4979 [-0.6174, -0.3798] | 6,192 |
| LONGITUDINAL.lead_viol | +0.0109 [+0.0062, +0.0154] | 6,435 |
| LONGITUDINAL.lead_progress_ratio | +0.0757 [+0.0670, +0.0858] | 5,331 |
| LONGITUDINAL.speed_mae_mps | +0.0324 [+0.0168, +0.0485] | 12,146 |
| LONGITUDINAL.target_speed_acc.within_0.5_mps | -0.0244 [-0.0309, -0.0176] | 12,146 |
| LONGITUDINAL.target_speed_acc.within_1.0_mps | -0.0158 [-0.0228, -0.0085] | 12,146 |
| LONGITUDINAL.target_speed_acc.within_2.0_mps | -0.0077 [-0.0139, -0.0018] | 12,146 |
| LONGITUDINAL.along_mae_m | +0.0578 [+0.0379, +0.0775] | 12,146 |
| LONGITUDINAL.ego_progress.progress_ratio_mean | +0.0488 [+0.0424, +0.0556] | 12,053 |
| LONGITUDINAL.ego_progress.progress_error_mean | +0.0198 [+0.0157, +0.0240] | 12,053 |
| LATERAL.heading_mae_deg | +0.1409 [+0.0863, +0.1961] | 12,102 |
| LATERAL.yaw_rate_mae_degps | +0.0848 [+0.0547, +0.1162] | 11,958 |
| LATERAL.cross_mae_m | +0.0394 [+0.0278, +0.0523] | 12,146 |
| LATERAL.cross_final_mae_m | +0.1392 [+0.1037, +0.1803] | 12,146 |
| LATERAL.last_pose_heading_err_deg | +0.2500 [+0.1401, +0.3627] | 12,146 |
| TACTICAL.lateral_decision.accuracy | -0.0077 [-0.0126, -0.0035] | 12,146 |
| TACTICAL.lateral_decision.kappa | -0.0116 [-0.0204, -0.0041] | 12,146 |
| TACTICAL.longitudinal_decision.accuracy | -0.0103 [-0.0196, -0.0002] | 12,146 |
| TACTICAL.maneuver_5way_collapsed.accuracy | -0.0093 [-0.0162, -0.0026] | 12,146 |
| TACTICAL.maneuver_5way_collapsed.kappa | -0.0096 [-0.0184, -0.0010] | 12,146 |
| TACTICAL.selection.regret | +0.9320 [+0.6091, +1.2408] | 12,146 |
| TACTICAL.selection.zeroed | +1.2597 [+0.9670, +1.5496] | 12,146 |
| TACTICAL.selection.pick | -0.9015 [-1.2021, -0.5874] | 12,146 |

STRATEGIC: reported in full in `strategic_family_T_C.json` (no bar; PI ruling 2026-11-16). TACTICAL (c) goal-setting: NOT COMPUTABLE (n = 12,146; no goal label on navtest).

## Next arm (card NEXT_ARM["O3"], committed before data)
Decision-level term: pick-anchored pairwise on the `navsim_v1` aggregate (every j with PDMS_j > PDMS_pick vs the current pick, weight PDMS_j - PDMS_pick), a new head key
`pick` in the same `rank_loss.py`; C reused (same init / seed / steps), ~14.2 L40-h. Own card; the run needs a PI go.

## What is here, and what is not
Here: the read's outputs (`read_stage1.json`, `report_stage1.txt`, `VERDICT_stage1.txt`), its inputs' small files (`argv_t.json` from SFTT's pod `config.json` sha256 `829586de…`,
`argv_c.json`, `w0_c.json`, `seam_ledger.json`, `convert_report.json`, `pod_config_{t,c}.json`, `pod_summary_c.json`), the family / tactical / strategic outputs
(`famci_T_vs_C.json`, `families_summary.json`, `tactical.json`, `strategic_family_T_C.json`), the chain scripts (`sft_read_s1.sh` v2, `sft_read_s1.v1_overlap.sh`,
`run_sft_read_s1.cmd`) and the dev-box `CHAIN.log`. Every file was LF already: the copies here are byte-equal to the read's files.
NOT here (re-creatable, large): the transp_score rows (`pass1/`), plans `.npz` (8.9 MB each), per-arm family JSONs (~50 MB each). They are rebuilt by `sft_read_s1.sh` from the seams
`E:/Projects/TanitAD/data/refe_navtest/seams/stage2_scorerft_{t,c}.npz` (+ `.proposals.npz`), which are themselves rebuilt from the HF models (`Sayood/tanitad-refe-p5c`
prefixes `scorerft/t`, `scorerft/c`) by `sft_seam_chain.sh`.
