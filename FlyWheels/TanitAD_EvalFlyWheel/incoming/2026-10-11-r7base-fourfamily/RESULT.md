# X8 / X7 -- the refcv7-50,400 four-family baseline row, incl. LEAD-VIOL at 6 s (EvalFlyWheel, 2026-10-11)

**Verdict: the row EXISTS, was produced on the dev box CPU in ~25 min with NO GPU and NO Colab, and is admissible for the launch gate (X8 precondition 7), with the gaps in section 5.**
Tier: **T1 self-action OPEN LOOP** on logged eval139 frames (status UNRULED for an action-free model; never closed loop). Estimator: FULL-SET mean; interval = episode-cluster bootstrap (B 2000, seed 0, 95 %), paired for deltas. Tip: `9826a6a` (FETCH_HEAD agent/arch-inf-20260803). Checkpoint md5 `b418d0fc4a92a6848c246a6a7c50207b` (MEASURED, == the banked battery_summary.json ckpt_md5).

## 1. What the launch gate requires (MEASURED: `Project Steering/SPEC_REFCV8_DRAFT.md` @ 9826a6a)
* Launch precondition 7 (l.184) + **X8** (s8.8): the battery on refcv7-50,400 **under the A7 ruling, all four families on S2 / S6, both inference seeds**. FAIL => every s5 bar reads PENDING.
* **Grid = S2 + S6, not EVAL-DIAG K=8 and not the full 23,772-window eval139.** S2 = 4,754 windows / 139 episodes (stride 5, instants 0.5..2.0 s); S6 = the S2 windows with a 60-step future = **3,668 windows / 139 episodes**, instants 1..6 s at dt 1.0 (battery SPEC s3.1). (EVAL-DIAG 1,112 is S-PERC/S-GRID; S-ROUTE 4,634 is the route/tactical surface; neither is the X8 grid.)
* **X7** (s8.7): LEAD-VIOL on S2 and S-ROUTE windows with a lead, refcv8 - refcv7 < 0 separated at both seeds; baseline "through the same instrument (the battery's distance-keeping block, X8)". Denominator pinned by the PI: **count a violation only where the human keeps >= 1.0 s** (`share_of_at_risk`).
* Producing script: battery `run_battery_r7.py` (roll -> dump) -> `r7_panel.build_s6_dump` / `analyze` -> `refcv3_arm.py --analyze-only` (+ `lead_metrics.lead_viol`, which that path does NOT call, see s4). 

## 2. Why no GPU was needed (the finding that changes the plan)
The X7 hand-back says "a 6 s row needs a GPU re-roll with `refcv3_arm --grid 6s`" (PI compute decision). **That is wrong for the registered S6.** MEASURED, `D:/refcv7_eval_kit/battery/step50400_a6text/dump_s{0,1}/decisions/epNNN.npz`: `plan_full_nav_true` is **[N, 8, 2] = slots 5,10,15,20,30,40,50,60 steps (0.5..6 s) for all 4,754 windows, both seeds**; `os` (top level, 4 slots) equals `plan_full_nav_true[:, :4]` with **max |diff| = 0** on all 139 episodes, both seeds; 3,668 windows carry a valid 60-step GT future. `r7_panel.build_s6_dump` is exactly how S6 is defined (BAR-R7-3 was always scored from it). The old banked S6 analysis only lacked the 6 s *lead block* (it used the 2 s block: instants [1,2]); the X7 k12/dt0.5 block (md5 `c346dff3b3d356f55ef2786f327c6c91`) shares all six S6 instants.
* Data sizes (for the record): eval139 cache **11.7 GB** on the dev box (`D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139`, 139 x ~85 MB); ckpt 0.40 GB (`D:/refcv7_eval_kit/ckpt/ckpt_50400.pt`). A frame-slice is NOT < 3 GB: ESTIMATED (not built) -- obs window 8 frames at stride 5 overlap, so the union is a contiguous run from frame 0; <~10 % saving. A Colab run would have needed ~12 GB up at ~1.2 MB/s (INHERITED, not re-measured) = ~2.8 h. **No Colab session was created; nothing to stop.** The dev-box RTX 4060 lock was free and was not used either.

## 3. The four-family row (MEASURED; `raw/analysis_s{2,6}_s{0,1}.json`, `raw/family_rows.json`, `raw/cross_paired_*`)

**S2** (4754 windows / 139 episodes; horizon 0.5-2.0 s, dt 0.5). `os` = refcv7-50,400 deployed plan (nav_true); `ha0_ext` = echo control. Interval = episode-cluster bootstrap (question answered: another draw of EPISODES).

| family | metric | os seed 0 | os seed 1 | ha0_ext (model-free) |
|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.2588 [0.2379, 0.2788] (n 4754/139) | 0.2595 [0.2388, 0.2797] (n 4754/139) | 0.2548 [0.2364, 0.2737] (n 4754/139) |
| LONGITUDINAL | along-track MAE (m) | 0.226 [0.2034, 0.2532] (n 4754/139) | 0.2266 [0.2046, 0.2532] (n 4754/139) | 0.2347 [0.216, 0.2583] (n 4754/139) |
| LONGITUDINAL | target-speed acc (+-0.5 m/s) | 0.8534 [0.8361, 0.8711] (n 4754/139) | 0.8519 [0.8343, 0.8693] (n 4754/139) | 0.8655 [0.8497, 0.8822] (n 4754/139) |
| LONGITUDINAL | accel MAE (m/s2) | 0.3715 [0.3445, 0.3974] (n 4754/139) | 0.3723 [0.3451, 0.3993] (n 4754/139) | 0.3181 [0.2873, 0.3483] (n 4754/139) |
| LATERAL | heading MAE (deg) | 1.348 [0.7344, 2.52] (n 4538/139) | 1.337 [0.7296, 2.493] (n 4538/139) | 1.452 [0.8259, 2.623] (n 4513/139) |
| LATERAL | curvature MAE (1/m) | 0.003662 [0.002893, 0.004558] (n 4500/139) | 0.003529 [0.002821, 0.004337] (n 4499/139) | 0.003767 [0.003024, 0.00457] (n 4475/139) |
| LATERAL | yaw-rate MAE, masked (deg/s) | 1.08 [0.9315, 1.236] (n 4500/139) | 1.064 [0.9207, 1.214] (n 4499/139) | 1.357 [1.128, 1.609] (n 4475/139) |
| LATERAL | cross-track MAE (m) | 0.103 [0.0903, 0.1169] (n 4754/139) | 0.1031 [0.0905, 0.1167] (n 4754/139) | 0.1079 [0.0918, 0.1252] (n 4754/139) |
| LONGITUDINAL | distance keeping (os, seed 0): mean min headway / time gap / min TTC; n, n_closing | 27.96 m / 4.23 s / 23.62 s; n 1108, n_time_gap 1008, n_closing 488 (TTC censored at 30 s: quote n_closing beside it) | seed 1: n_closing 495 | see s4 |
| LONGITUDINAL | 6-s/horizon progress ratio plan/human (mean; median) | 1.015; 1.001 | 1.015 | |
| TACTICAL | TACTICAL lateral decision (traj-derived: lane_keep/turn_left/turn_right) acc; kappa | 0.9617; 0.8418 (n 4754) | 0.9621; 0.8427 | |
| TACTICAL | TACTICAL longitudinal decision (traj-derived: brake_stop/steady/accelerate) acc; kappa | 0.8305; 0.5518 (n 4754) | 0.8357; 0.5636 | |
| STRATEGIC | -- | **NOT APPLICABLE, n = 0, reason: strategic layer OFF (PI ruling R5; `--no-strategic`)** | same | |

Paired `os - ha0_ext` (S2; negative = os better; `raw/cross_paired_s2_s{0,1}.json`):
* seed 0: ade_m -0.0102 [-0.026, +0.0067]; fde_m -0.036 [-0.0759, +0.0054]; LON_speed_mae_mps +0.004 [-0.0102, +0.0198]; LON_along_mae_m -0.0086 [-0.0225, +0.0057]; LAT_cross_mae_m -0.0049 [-0.0158, +0.0059]; LAT_heading_mae_deg -0.0961 [-0.2042, +0.0146]; LAT_yaw_rate_mae_radps_valid -0.0047 [-0.0075, -0.0021] SEP
* seed 1: ade_m -0.0096 [-0.0253, +0.0069]; fde_m -0.0347 [-0.0754, +0.006]; LON_speed_mae_mps +0.0048 [-0.0095, +0.0202]; LON_along_mae_m -0.008 [-0.0214, +0.0065]; LAT_cross_mae_m -0.0049 [-0.0153, +0.0056]; LAT_heading_mae_deg -0.1071 [-0.2109, -0.0042] SEP; LAT_yaw_rate_mae_radps_valid -0.005 [-0.0077, -0.0024] SEP

**S6** (3668 windows / 139 episodes; horizon 1-6 s, dt 1.0). `os` = refcv7-50,400 deployed plan (nav_true); `ha0_ext` = echo control. Interval = episode-cluster bootstrap (question answered: another draw of EPISODES).

| family | metric | os seed 0 | os seed 1 | ha0_ext (model-free) |
|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9329 [0.8486, 1.017] (n 3668/139) | 0.9357 [0.8503, 1.021] (n 3668/139) | 1.096 [0.9889, 1.199] (n 3668/139) |
| LONGITUDINAL | along-track MAE (m) | 2.19 [1.998, 2.389] (n 3668/139) | 2.189 [2, 2.39] (n 3668/139) | 2.646 [2.376, 2.912] (n 3668/139) |
| LONGITUDINAL | target-speed acc (+-0.5 m/s) | 0.5368 [0.5077, 0.5667] (n 3668/139) | 0.5338 [0.5057, 0.5629] (n 3668/139) | 0.5249 [0.4978, 0.5544] (n 3668/139) |
| LONGITUDINAL | accel MAE (m/s2) | 0.4306 [0.3934, 0.4668] (n 3668/139) | 0.4322 [0.3946, 0.4684] (n 3668/139) | 0.5055 [0.4541, 0.5552] (n 3668/139) |
| LATERAL | heading MAE (deg) | 3.163 [2.318, 4.502] (n 3579/139) | 3.151 [2.312, 4.481] (n 3581/139) | 5.941 [4.6, 7.53] (n 3521/139) |
| LATERAL | curvature MAE (1/m) | 0.005002 [0.003871, 0.006244] (n 3552/139) | 0.004961 [0.003876, 0.006171] (n 3551/139) | 0.00685 [0.005367, 0.008462] (n 3472/139) |
| LATERAL | yaw-rate MAE, masked (deg/s) | 1.632 [1.378, 1.899] (n 3552/139) | 1.628 [1.374, 1.896] (n 3551/139) | 2.775 [2.199, 3.389] (n 3472/139) |
| LATERAL | cross-track MAE (m) | 0.8412 [0.7119, 0.9792] (n 3668/139) | 0.8471 [0.7164, 0.9862] (n 3668/139) | 1.668 [1.38, 1.974] (n 3668/139) |
| LONGITUDINAL | distance keeping (os, seed 0): mean min headway / time gap / min TTC; n, n_closing | 25.77 m / 3.88 s / 22.33 s; n 1020, n_time_gap 925, n_closing 475 (TTC censored at 30 s: quote n_closing beside it) | seed 1: n_closing 475 | see s4 |
| LONGITUDINAL | 6-s/horizon progress ratio plan/human (mean; median) | 1.104; 1.004 | 1.102 | |
| TACTICAL | TACTICAL lateral decision (traj-derived: lane_keep/turn_left/turn_right) acc; kappa | 0.8770; 0.7108 (n 3668) | 0.8768; 0.7111 | |
| TACTICAL | TACTICAL longitudinal decision (traj-derived: brake_stop/steady/accelerate) acc; kappa | 0.6039; 0.3811 (n 3668) | 0.6009; 0.3766 | |
| STRATEGIC | -- | **NOT APPLICABLE, n = 0, reason: strategic layer OFF (PI ruling R5; `--no-strategic`)** | same | |

Paired `os - ha0_ext` (S6; negative = os better; `raw/cross_paired_s6_s{0,1}.json`):
* seed 0: ade_m -0.9804 [-1.294, -0.6815] SEP; fde_m -2.874 [-3.714, -2.051] SEP; LON_speed_mae_mps -0.1628 [-0.2384, -0.0805] SEP; LON_along_mae_m -0.456 [-0.6715, -0.2365] SEP; LAT_cross_mae_m -0.8268 [-1.05, -0.6143] SEP; LAT_heading_mae_deg -2.692 [-3.525, -1.896] SEP; LAT_yaw_rate_mae_radps_valid -0.0199 [-0.0269, -0.0129] SEP
* seed 1: ade_m -0.9796 [-1.286, -0.6842] SEP; fde_m -2.868 [-3.707, -2.065] SEP; LON_speed_mae_mps -0.1599 [-0.2338, -0.0789] SEP; LON_along_mae_m -0.4568 [-0.6722, -0.2354] SEP; LAT_cross_mae_m -0.8209 [-1.043, -0.6119] SEP; LAT_heading_mae_deg -2.713 [-3.522, -1.934] SEP; LAT_yaw_rate_mae_radps_valid -0.0201 [-0.027, -0.0133] SEP
**Controls (MEASURED).** (a) The tip's S2 seed-0 `os` cells reproduce the banked `analysis_s0.json` **exactly** on 8 of 8 cells (speed MAE 0.2588, along MAE 0.226, target-speed acc 0.8534, accel MAE 0.3715, heading 1.3478, curvature 0.003662, yaw-rate 1.08, cross-track 0.103; |diff| 0.0, `raw/control_vs_banked_s2_s0.json`). (b) The paired cells reproduce the banked bars: S2 `ade_m` -0.0102 [-0.0260, +0.0067] (BAR-R7-1, seed 1 -0.0096), S6 `ade_m` -0.9804 [-1.2940, -0.6815] SEP (BAR-R7-3, seed 1 -0.9796). (c) `os` == `plan_full[:, :4]` bit-exact (s2). (d) Model-free arms and GT are the same arrays in both seeds (ha0_ext identical across seeds above).
**Inference-seed floor** (would another INFERENCE run say this?): S2 ade_m |s0 - s1| 0.0006 m (banked); S6 paired delta moves 0.0008 m. **Training-seed floor: NOT measured (single training seed)** -- any refcv8 - refcv7 claim is restricted as SPEC s9 says.
**Tactical caveat.** The TACTICAL cells here are trajectory-derived decisions on the whole grid. The dense-v9 R8-4 baseline (lat/lon macro-F1, goal AP, controllability on S-V9 / S-ROUTE) is NOT in this deliverable: it stays with WP-B (`r84_baseline_refcv7.py` on the R1 capture). The banked decoder-head numbers (v6_behaviour_decoder lat acc 0.7401 / kappa 0.4566, lon 0.481 / 0.3185, n 1,081 in-band, seed 0) are INHERITED from `battery/raw/step50400_a6text/RESULT_SECTION_step50400_a6text.md`, not re-derived here.

## 4. LEAD-VIOL (X7) -- the 6 s block, MEASURED (`raw/lead_viol_s{0,1}.json`, `code/r7base_stageB.py`)
Definition (SPEC s8.7(a), `lead_metrics.lead_viol`): share of lead windows whose plan's minimum time gap is < 1.0 s while the human's is >= 1.0 s. **Lead window = join state `LEAD` of the X7 k12/dt0.5 block** (md5 `c346dff3...`), joined with `refcv3_arm.lead_block_common_grid` + `refav1_arm.join_lead_block` exactly as the battery does (S6 instants 1..6 s, dt 1.0; S2 instants 0.5..2.0 s). `share_of_at_risk` = the PI's ruling (denominator = lead windows where the human keeps >= 1.0 s); its interval is the episode-cluster bootstrap over those windows (my addition; `lead_viol` itself emits only the all-lead-window CI). **Progress control** = plan along-track x at the last instant over the human's; the progress-matched share counts only at-risk windows where the plan covered >= 90 % of the human's distance. The lead-block `LEAD` set here is NOT the v9 `lead_valid` (see s5).

**Headline (S6, 6 s, os): LEAD-VIOL 0.0426 [0.0197, 0.0734] (49/1,151 lead windows, seed 0) and 0.0434 [0.0217, 0.0715] (50/1,151, seed 1); share_of_at_risk 0.0590 [0.0272, 0.1002] / 0.0602 [0.0295, 0.0977] over 830 at-risk windows. Plan lost the lead (undefined time gap, not a violation): 28 windows; sensitivity share including them 0.0652 / 0.0660.** Inference floor |s0 - s1| = 0.0009 (S6), 0.0025 (S2). The 6 s block is informative; the 2 s block is nearly blind (S2 os 5-8 violations in 1,188 lead windows) -- bar X7 on S6, as the X7 package said.
* Progress control: os covers **1.108x** the human's 6-s distance on lead windows (ratio mean), so the 0.043 is not bought by braking; progress-matched share 0.0679 (n 707). **STOP reads exactly 0 (progress 0.000)**, hold-v0 0.0765, echo `ha0_ext` 0.0313. A refcv8 LEAD-VIOL drop is admissible only beside a progress ratio that stays ~1.
* os is **not** better than the echo here (os - ha0_ext +0.0113 [-0.0074, +0.0352], not separated); os - hold-v0 -0.0339 [-0.0645, -0.0052] separated. Regression control present: human-as-plan reads 0 by construction (`tests/test_lead_viol.py`, 31 passed per the X7 package; I did not re-run it).
* **Grid caution:** the X7 package's hold-v0 control **0.096** is on the 12-instant (0.5 s) grid over all block rows; on the S6 grid (6 instants) hold-v0 reads 0.0765. refcv8 must be scored on the S6 grid used here or the delta compares different grids.

**S6**  (n_lead_windows / windows below)

| seed | arm | LEAD-VIOL share (all lead windows) | 95% CI | n_viol/n_lead | share_of_at_risk (PI ruling) | CI | n_at_risk | plan lost lead | share incl. lost | progress ratio plan/human | progress-matched share (ratio>=0.9) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | os | 0.0426 | [0.0197, 0.0734] | 49/1151 | 0.0590 | [0.0272, 0.1002] | 830 | 28 | 0.0652 | 1.108 | 0.0679 |
| 0 | ha0 | 0.0765 | [0.0415, 0.1214] | 88/1151 | 0.1060 | [0.0590, 0.1672] | 830 | 43 | 0.1112 | 1.059 | 0.1424 |
| 0 | ha0_ext | 0.0313 | [0.0183, 0.0462] | 36/1151 | 0.0434 | [0.0253, 0.0629] | 830 | 33 | 0.0573 | 0.943 | 0.0501 |
| 0 | stop | 0.0000 | [0.0000, 0.0000] | 0/1151 | 0.0000 | [0.0000, 0.0000] | 830 | 43 | 0.0348 | 0.000 | NA |
|  | paired seed 0 os_minus_ha0 | -0.0339 | [-0.0645, -0.0052] | 49 vs 88 | | | | | | | separated=True |
|  | paired seed 0 os_minus_ha0_ext | +0.0113 | [-0.0074, +0.0352] | 49 vs 36 | | | | | | | separated=False |
|  | paired seed 0 os_minus_stop | +0.0426 | [+0.0197, +0.0734] | 49 vs 0 | | | | | | | separated=True |
| 1 | os | 0.0434 | [0.0217, 0.0715] | 50/1151 | 0.0602 | [0.0295, 0.0977] | 830 | 28 | 0.0660 | 1.104 | 0.0697 |
| 1 | ha0 | 0.0765 | [0.0415, 0.1214] | 88/1151 | 0.1060 | [0.0590, 0.1672] | 830 | 43 | 0.1112 | 1.059 | 0.1424 |
| 1 | ha0_ext | 0.0313 | [0.0183, 0.0462] | 36/1151 | 0.0434 | [0.0253, 0.0629] | 830 | 33 | 0.0573 | 0.943 | 0.0501 |
| 1 | stop | 0.0000 | [0.0000, 0.0000] | 0/1151 | 0.0000 | [0.0000, 0.0000] | 830 | 43 | 0.0348 | 0.000 | NA |
|  | paired seed 1 os_minus_ha0 | -0.0330 | [-0.0651, -0.0037] | 50 vs 88 | | | | | | | separated=True |
|  | paired seed 1 os_minus_ha0_ext | +0.0122 | [-0.0056, +0.0334] | 50 vs 36 | | | | | | | separated=False |
|  | paired seed 1 os_minus_stop | +0.0434 | [+0.0217, +0.0715] | 50 vs 0 | | | | | | | separated=True |

S6: windows 3668, episodes 139, lead windows 1151 (join state counts {'LEAD': 1151, 'NO_LEAD': 864, 'NOT_STRAIGHT': 1444, 'NO_LABEL': 209}), instants [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] s, dt 1.0.

**S2**  (n_lead_windows / windows below)

| seed | arm | LEAD-VIOL share (all lead windows) | 95% CI | n_viol/n_lead | share_of_at_risk (PI ruling) | CI | n_at_risk | plan lost lead | share incl. lost | progress ratio plan/human | progress-matched share (ratio>=0.9) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | os | 0.0042 | [0.0008, 0.0090] | 5/1188 | 0.0054 | [0.0011, 0.0113] | 927 | 15 | 0.0168 | 1.206 | 0.0056 |
| 0 | ha0 | 0.0177 | [0.0061, 0.0334] | 21/1188 | 0.0227 | [0.0077, 0.0419] | 927 | 24 | 0.0379 | 1.007 | 0.0272 |
| 0 | ha0_ext | 0.0034 | [0.0000, 0.0080] | 4/1188 | 0.0043 | [0.0000, 0.0096] | 927 | 10 | 0.0118 | 0.952 | 0.0046 |
| 0 | stop | 0.0000 | [0.0000, 0.0000] | 0/1188 | 0.0000 | [0.0000, 0.0000] | 927 | 24 | 0.0202 | 0.000 | NA |
|  | paired seed 0 os_minus_ha0 | -0.0135 | [-0.0262, -0.0043] | 5 vs 21 | | | | | | | separated=True |
|  | paired seed 0 os_minus_ha0_ext | +0.0008 | [-0.0018, +0.0040] | 5 vs 4 | | | | | | | separated=False |
|  | paired seed 0 os_minus_stop | +0.0042 | [+0.0008, +0.0090] | 5 vs 0 | | | | | | | separated=True |
| 1 | os | 0.0067 | [0.0018, 0.0128] | 8/1188 | 0.0086 | [0.0024, 0.0160] | 927 | 19 | 0.0227 | 1.186 | 0.0089 |
| 1 | ha0 | 0.0177 | [0.0061, 0.0334] | 21/1188 | 0.0227 | [0.0077, 0.0419] | 927 | 24 | 0.0379 | 1.007 | 0.0272 |
| 1 | ha0_ext | 0.0034 | [0.0000, 0.0080] | 4/1188 | 0.0043 | [0.0000, 0.0096] | 927 | 10 | 0.0118 | 0.952 | 0.0046 |
| 1 | stop | 0.0000 | [0.0000, 0.0000] | 0/1188 | 0.0000 | [0.0000, 0.0000] | 927 | 24 | 0.0202 | 0.000 | NA |
|  | paired seed 1 os_minus_ha0 | -0.0109 | [-0.0214, -0.0033] | 8 vs 21 | | | | | | | separated=True |
|  | paired seed 1 os_minus_ha0_ext | +0.0034 | [+0.0008, +0.0070] | 8 vs 4 | | | | | | | separated=True |
|  | paired seed 1 os_minus_stop | +0.0067 | [+0.0018, +0.0128] | 8 vs 0 | | | | | | | separated=True |

S2: windows 4754, episodes 139, lead windows 1188 (join state counts {'LEAD': 1188, 'NO_LEAD': 889, 'NOT_STRAIGHT': 1494, 'NO_LABEL': 1183}), instants [0.5, 1.0, 1.5, 2.0] s, dt 0.5.
## 5. Admissibility for the launch gate, and the gaps (read before quoting)
**Admissible for X8 precondition 7 on S2 and S6, both seeds:** same 4,754 / 3,668 windows and 139 episodes as refcv8's battery will use; same dumps (`D:/refcv7_eval_kit/battery/step50400_a6text/dump_s0|s1`, OFF git, dev box only, 291 / 283 MB) are what refcv8's paired bootstrap pairs against; tip instruments (`9826a6a`) reproduce the banked numbers bit-for-bit (s3 controls); G0 under the A7 text ruling is PASS (`battery/raw/step50400/g0_A6_text.json`, banked).
**Not covered -- do not claim them:**
1. **S-ROUTE LEAD-VIOL** (SPEC s8.7 names "S2 and S-ROUTE"): not computed; the R1/A6 dense capture is a different 4,634-window list and was not rolled here. Needs the A6 window list + a plan capture on it.
2. **v9 `lead_valid`** is not joined: my lead windows are the B1 block's `LEAD` state (same definition as the X7 control). n_lead 1,151 (S6) / 1,188 (S2) is therefore not the SPEC's v9 count. The refcv8 run must use the same definition on both sides.
3. **R8-4 dense-v9 tactical, X3 ceiling-compliance, X5/X10 instruments** (`ceil_obey.py`, `night_luma.py`, `nav_compliance.py`) are NOT in this row; they need their own inputs (v9 labels, vmax sidecars).
4. `refcv3_arm._distance_keeping` replaces the family block with one that has **no `lead_viol`** (MEASURED: `lead_viol` is wired only in `four_families._distance_keeping`, which the refcv3 analysis path overrides). The X7 UNVERIFIED question is answered: LEAD-VIOL silently does NOT appear in a refcv3 analysis. `code/r7base_stageB.py` is the working route; it should be merged into the battery (integration escalated to the Master Mind, not left as a note).
5. Single training seed: a separated interval answers "another draw of episodes" only (H-ESTIM-SEED-1).
6. Raw clip UUIDs in the staged JSONs were replaced by `sha12:<12 hex>` (1,322 substitutions); the work copies with raw ids stay in the scratchpad.

## 6. Deliverable manifest
All under `C:/Users/Admin/qland/hold_r7base/` (nothing `git add`ed, committed or pushed; nothing on Thor, Colab, or G:):
* `RESULT.md`, `msg.txt`
* `stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-r7base-fourfamily/RESULT.md` (same text) + `code/r7base_stageA.py` (S6 rebuild + tip analysis, S2/S6 x seeds), `r7base_stageB.py` (LEAD-VIOL), `r7base_stageC.py` (paired cells + banked-control) + `raw/*.json` (analysis_s{2,6}_s{0,1}, lead_viol_s{0,1}, family_rows, cross_paired_*, control_vs_banked_s2_s0, s6_build_s*).
* Work state (not staged): `C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f0c4bd12-5464-4952-995c-062072b3b8d0/scratchpad/r7base/` (tip tree via `git archive`, battery code, `out/` incl. rebuilt `s6_s0|s1` dumps). Reproduce: PowerShell, `PYTHONPATH=<tree>\stack;<tree>\taniteval`, `OMP_NUM_THREADS=2`, venv `C:/Users/Admin/venvs/tanitad`, `python r7base_stageA.py <seed> <out>` then `B`, `C` (~5 min per seed for A, <1 min for B).
* Owed to the Master Mind: SPEC_REFCV8_DRAFT s8.7(b)/s8.8 and GOALS_AND_CLAIMS rows: X8 baseline row = EXISTS (this file); X7 baseline = 0.0426 (S6, seed 0) / 0.0434 (seed 1); correct the claim that a 6 s row needs a GPU re-roll.
