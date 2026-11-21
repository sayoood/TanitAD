# READ_DAG_HACC: PREREG_V8F_DAG_HACC executed as written (2026-10-27)

**OUTCOME: COSTS POSITION** (KEEP fails at >= 1 seed)

* Committed next (PREREG section 4, verbatim): *"not adopted; DAG stays plain. Next: as NOT FIXED"*
* Flags: KEEP at both seeds = False; D1 both = True; D2 both = False; D3 both = True; M1 separated-worse at >= 1 seed = True; a LONGITUDINAL row separated-worse at >= 1 seed = False; every registered control passes = True.
* Evidence class: MEASURED (ours); every number is copied from `raw/` of this package. Tier: **T1-kin** (self-action OPEN loop; m = 10, causal bucket, dead-reckoned nav, logged scene): never closed loop, never "drives". Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2000, seed 0, per seed, over the 141 episodes (2,399 windows); it answers 'would another draw of EPISODES say this?' only. F = 0.1207 m, 2F = 0.2415 m. The planner is deterministic (no inference-seed question). Zero training; dev-box CPU.

## 1. KEEP (DAG-HACC - DAG, ADE6, T1-kin): upper bound must be <= +0.0604 m at both seeds

| seed | delta [lo, hi] | separated | DAG-HACC mean | DAG mean | n | bar (hi <=) | holds |
|---|---|---|---|---|---|---|---|
| s0 | +0.1264 [+0.0522, +0.2004] | yes | 2.4041 | 2.2777 | 2399/141 | +0.0604 | **False** |
| s1 | -0.1047 [-0.1834, -0.0334] | yes | 2.3517 | 2.4565 | 2399/141 | +0.0604 | **True** |

## 2. Controls

| control | result |
|---|---|
| C1_comps_L2_M1 | PASS |
| C1_t1kf_unmodified_on_HACC_banked_dumps | PASS |
| C1_spos_path_reproduces_DAG_READ | PASS |
| C2_identity | PASS |
| C3_floor | PASS |
| C4_same_windows | PASS |
| reachability_post_map_moves_the_executed_plan | PASS |
| C_H1_recipe_identity | PASS |
| C_H2_train_only | PASS |
| C_H3_bake_identity | PASS |
| C_H4_bake_scope | PASS |
| rollouts_clean | PASS |

All registered controls pass: **True**. Failed: [].
* C-H1 (recipe identity): `v8f_hacc.fit_seed` on M2b's banked fit inputs: max |dw| vs HACC's banked weights = M2b_s0: 0.0e+00, lambda* 9998.35 == banked 9998.35, M2b_s1: 0.0e+00, lambda* 36931.4 == banked 36931.4.
* C-H3 (bake identity, w == 1): max |d| traj30 / traj60 on the first 24 eval windows = s0: 0.0 / 0.0 m; s1: 0.0 / 0.0 m (also 64 windows and through a saved + reloaded checkpoint).
* C-H4 (bake scope): s0: extra keys ['hacc_w'], 276 source tensors, non-identical 0, optimizer/step/cfg non-identical 0; s1: extra keys ['hacc_w'], 276 source tensors, non-identical 0, optimizer/step/cfg non-identical 0.
* Added C-X1 (not in the PREREG): the plan-production route reproduces HBLEND's banked M2b `fit_windows` (windows, ages, GT identical; traj30 max |d| M2b_s0: 3.8e-05, M2b_s1: 3.8e-05 m). My first-draft p30 tolerance (1e-6) was too tight for float32 CPU noise and read False; HBLEND's own bound (1e-4 m) is used and both are in `plans/CX1_route_reproduces_banked_M2b.json`.

## 3. The fit (DAG's own T0 plans on HBLEND's W_fit: 141 train clips, 9,582 windows; fit code unchanged)

| seed | lambda0 | m* | lambda* | rule branch | train accel (arm / P10) | train curvature | train ADE2 | train ADE6 | mean w along / cross |
|---|---|---|---|---|---|---|---|---|---|
| DAG_s0 | 2206.8 | 16 | 35309.0 | qualified: the smallest grid lambda meeting  | 0.6606 / 0.7046 | 0.05067 / 0.05204 | 0.2105 / 0.2186 | 2.0124 / 2.5502 | 0.376 / 0.682 |
| DAG_s1 | 2116.1 | 16 | 33857.0 | qualified: the smallest grid lambda meeting  | 0.6601 / 0.7046 | 0.04914 / 0.05204 | 0.2089 / 0.2186 | 1.9606 / 2.5502 | 0.374 / 0.698 |

D1 sensitivity (BESIDE, decides nothing): lambda0 from M2b's literal banked w instead of the HBLEND recipe on DAG's plans: s0: lambda* 40339 vs 35309, max |dw| 4.1e-03; s1: lambda* 33915 vs 33857, max |dw| 5.5e-05.

## 4. Read A: DAG-HACC - P10 at T1-kin (decisive). Right-hand column: plain DAG - P10 from the banked reread, for the side-by-side

#### Read A, seed 0: DAG-HACC - P10, T1-kin decisive (m = 10, causal bucket, dead-reckoned nav, logged scene)

| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.0392 [-1.1941, -0.9008] | yes | BETTER | 2.4041 | 3.4433 | 2399/141 | -1.1656 [-1.3437, -0.9963] better |
| LONGITUDINAL | speed MAE | -0.6126 [-0.7482, -0.4957] | yes | BETTER | 1.2754 | 1.8880 | 2399/141 | -0.5765 [-0.7791, -0.4040] better |
| LONGITUDINAL | along-track MAE | -0.6262 [-0.7417, -0.5177] | yes | BETTER | 1.8320 | 2.4581 | 2399/141 | -0.6282 [-0.7671, -0.4916] better |
| LONGITUDINAL | accel MAE | -0.6444 [-0.9256, -0.4115] | yes | BETTER | 1.4455 | 2.0899 | 2399/141 | +0.8072 [+0.3253, +1.2728] WORSE |
| LONGITUDINAL | DK-VIOL | +0.0269 [-0.0064, +0.0760] | no | ns | 0.0740 | 0.0471 | 595/70 | +0.0353 [+0.0046, +0.0726] WORSE |
| LONGITUDINAL | DK-HW | -0.0812 [-0.4369, +0.2675] | no | ns | 2.3335 | 2.3869 | 716/72 | -0.0491 [-0.4481, +0.2980] |
| LONGITUDINAL | DK-TG | -0.0393 [-0.0980, +0.0149] | no | ns | 0.3916 | 0.4229 | 654/72 | -0.0035 [-0.0682, +0.0611] |
| LATERAL | heading MAE | -2.7563 [-3.2938, -2.2564] | yes | BETTER | 6.7963 | 9.5525 | 2373/141 | -3.3593 [-4.1043, -2.7104] better |
| LATERAL | yaw-rate MAE | -1.1388 [-3.0578, +0.5428] | no | ns | 18.7718 | 19.9106 | 2366/141 | -0.0400 [-2.3623, +2.1885] |
| LATERAL | curvature MAE | +0.0174 [+0.0091, +0.0270] | yes | **WORSE** | 0.0910 | 0.0736 | 2366/141 | +0.0612 [+0.0415, +0.0839] WORSE |
| LATERAL | cross-track MAE | -0.7736 [-0.9141, -0.6471] | yes | BETTER | 1.1460 | 1.9195 | 2399/141 | -0.9793 [-1.1231, -0.8460] better |
| LATERAL | L1 lateral-accel MAE | -1.4898 [-2.5369, -0.6077] | yes | BETTER | 2.9159 | 4.4057 | 2366/141 | -2.4322 [-3.8282, -1.2766] better |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0112 [+0.0050, +0.0190] | yes | **WORSE** | 0.0277 | 0.0165 | 2106/135 | +0.0288 [+0.0165, +0.0433] WORSE |
| TACTICAL | TAC-LAT | +0.2018 [+0.1651, +0.2397] | yes | BETTER | 0.7949 | 0.5932 | 2399/141 | +0.1926 [+0.1546, +0.2323] better |
| TACTICAL | TAC-LON | +0.1647 [+0.1171, +0.2113] | yes | BETTER | 0.5361 | 0.3714 | 2399/141 | +0.1288 [+0.0651, +0.1917] better |
| STRATEGIC | S-POS@30 | -1.9915 [-2.6029, -1.4745] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -1.7175 [-2.2707, -1.2803] better |
| STRATEGIC | S-POS@50 | -5.1059 [-6.3201, -4.0103] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -4.4832 [-5.6271, -3.4998] better |
| STRATEGIC | S-POS@80 | -10.7722 [-13.1090, -8.6474] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -9.6871 [-11.8435, -7.5669] better |
| crawl | M1 crawl share | +0.0242 [+0.0126, +0.0381] | yes | **WORSE** | 0.0471 | 0.0230 | 2106/135 | +0.0470 [+0.0296, +0.0673] WORSE |

#### Read A, seed 1: DAG-HACC - P10, T1-kin decisive (m = 10, causal bucket, dead-reckoned nav, logged scene)

| family | row | DAG-HACC - P10 [lo, hi] | sep | verdict | DAG-HACC mean | P10 mean | n (win/ep) | plain DAG - P10 (reread) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY | ADE6 | -1.0916 [-1.2535, -0.9322] | yes | BETTER | 2.3517 | 3.4433 | 2399/141 | -0.9868 [-1.1778, -0.8060] better |
| LONGITUDINAL | speed MAE | -0.7449 [-0.9160, -0.5928] | yes | BETTER | 1.1432 | 1.8880 | 2399/141 | -0.3020 [-0.4813, -0.1312] better |
| LONGITUDINAL | along-track MAE | -0.6472 [-0.7803, -0.5163] | yes | BETTER | 1.8109 | 2.4581 | 2399/141 | -0.4571 [-0.6243, -0.2880] better |
| LONGITUDINAL | accel MAE | -0.8469 [-1.1600, -0.5836] | yes | BETTER | 1.2430 | 2.0899 | 2399/141 | +1.4436 [+0.9463, +1.9592] WORSE |
| LONGITUDINAL | DK-VIOL | +0.0034 [-0.0248, +0.0372] | no | ns | 0.0504 | 0.0471 | 595/70 | +0.0050 [-0.0196, +0.0345] |
| LONGITUDINAL | DK-HW | -0.3167 [-0.6545, -0.0282] | yes | BETTER | 2.1495 | 2.3869 | 713/72 | -0.2065 [-0.6444, +0.1422] |
| LONGITUDINAL | DK-TG | -0.1139 [-0.1803, -0.0586] | yes | BETTER | 0.2913 | 0.4229 | 653/72 | -0.0934 [-0.1585, -0.0402] better |
| LATERAL | heading MAE | -2.1159 [-2.6448, -1.6318] | yes | BETTER | 7.4366 | 9.5525 | 2373/141 | -1.5991 [-2.3295, -0.8482] better |
| LATERAL | yaw-rate MAE | +1.2187 [-0.4700, +3.0036] | no | ns | 21.1293 | 19.9106 | 2366/141 | +8.6601 [+5.7957, +11.8486] WORSE |
| LATERAL | curvature MAE | +0.0471 [+0.0284, +0.0683] | yes | **WORSE** | 0.1207 | 0.0736 | 2366/141 | +0.1134 [+0.0787, +0.1525] WORSE |
| LATERAL | cross-track MAE | -0.7943 [-0.9345, -0.6665] | yes | BETTER | 1.1253 | 1.9195 | 2399/141 | -0.9269 [-1.0757, -0.7954] better |
| LATERAL | L1 lateral-accel MAE | -2.0790 [-3.1986, -1.1025] | yes | BETTER | 2.3267 | 4.4057 | 2366/141 | -1.1360 [-2.1320, -0.2847] better |
| LATERAL | L2 curvature MAE (GT>=5.2) | +0.0280 [+0.0127, +0.0488] | yes | **WORSE** | 0.0445 | 0.0165 | 2106/135 | +0.0604 [+0.0356, +0.0930] WORSE |
| TACTICAL | TAC-LAT | +0.1651 [+0.1279, +0.2043] | yes | BETTER | 0.7582 | 0.5932 | 2399/141 | +0.1851 [+0.1461, +0.2255] better |
| TACTICAL | TAC-LON | +0.1792 [+0.1247, +0.2323] | yes | BETTER | 0.5506 | 0.3714 | 2399/141 | +0.1142 [+0.0442, +0.1792] better |
| STRATEGIC | S-POS@30 | -2.0186 [-2.6019, -1.5203] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -1.8734 [-2.3640, -1.4387] better |
| STRATEGIC | S-POS@50 | -5.4377 [-6.7005, -4.2945] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -4.9034 [-5.9596, -3.9235] better |
| STRATEGIC | S-POS@80 | -12.0676 [-14.5739, -9.7827] | yes | BETTER | - | - | ticks>=1, unpaired vs CTRV | -10.3254 [-12.4401, -8.3492] better |
| crawl | M1 crawl share | +0.0390 [+0.0221, +0.0587] | yes | **WORSE** | 0.0619 | 0.0230 | 2106/135 | +0.0614 [+0.0409, +0.0859] WORSE |

Verdicts (rows D2 tests = every row of the table; M1 / L2 are the crawl rows):

| seed | D1 (ADE6 sep-better, <= -2F) | multiple of F | D2 | D3 | separated-worse rows | LONGITUDINAL separated-worse | M1 sep-worse | L2 sep-worse |
|---|---|---|---|---|---|---|---|---|
| s0 | True | 8.6 F | False | True | ['curvature MAE', 'L2 curvature MAE (GT>=5.2)'] | [] | True | True |
| s1 | True | 9.0 F | False | True | ['curvature MAE', 'L2 curvature MAE (GT>=5.2)'] | [] | True | True |

## 5. Beside (never decisive)

### 5.1 DAG-HACC - DAG on every Read-A row (T1-kin decisive; each row's move attributed)

| row | s0 [lo, hi] | sep | s1 [lo, hi] | sep |
|---|---|---|---|---|
| ADE6 | +0.1264 [+0.0522, +0.2004] | yes | -0.1047 [-0.1834, -0.0334] | yes |
| speed MAE | -0.0361 [-0.1404, +0.0647] | no | -0.4429 [-0.5587, -0.3432] | yes |
| along-track MAE | +0.0020 [-0.0728, +0.0697] | no | -0.1901 [-0.2860, -0.1086] | yes |
| accel MAE | -1.4515 [-1.8102, -1.1365] | yes | -2.2905 [-2.7945, -1.8596] | yes |
| heading MAE | +0.6031 [+0.1549, +1.0913] | yes | -0.5168 [-1.1894, +0.0443] | no |
| yaw-rate MAE | -1.0715 [-3.0525, +0.9458] | no | -7.4142 [-9.9141, -5.1883] | yes |
| curvature MAE | -0.0431 [-0.0618, -0.0268] | yes | -0.0655 [-0.0904, -0.0422] | yes |
| cross-track MAE | +0.2057 [+0.1553, +0.2600] | yes | +0.1326 [+0.0806, +0.1981] | yes |
| L1 lateral-accel MAE | +0.9421 [+0.4455, +1.5306] | yes | -0.9436 [-1.4260, -0.4871] | yes |
| L2 curvature MAE (GT>=5.2) | -0.0176 [-0.0289, -0.0076] | yes | -0.0324 [-0.0588, -0.0128] | yes |
| ADE2 (beside) | -0.0716 [-0.0992, -0.0528] | yes | -0.0933 [-0.1228, -0.0735] | yes |
| L3 yaw-rate fast (beside) | -0.2450 [-1.8486, +1.5953] | no | -3.3518 [-5.1572, -1.6913] | yes |
| M1 crawl share | -0.0228 [-0.0339, -0.0133] | yes | -0.0224 [-0.0309, -0.0146] | yes |
| TAC-LAT | +0.0092 [-0.0096, +0.0275] | no | -0.0200 [-0.0421, +0.0008] | no |
| TAC-LON | +0.0358 [-0.0146, +0.0888] | no | +0.0650 [+0.0217, +0.1093] | yes |
| DK-VIOL | -0.0084 [-0.0266, +0.0096] | no | -0.0017 [-0.0135, +0.0106] | no |
| DK-HW | -0.0354 [-0.3400, +0.3074] | no | -0.1237 [-0.3253, +0.1114] | no |
| DK-TG | -0.0360 [-0.0842, +0.0087] | no | -0.0214 [-0.0608, +0.0154] | no |

### 5.2 STRICT (scene frozen, limit causal) and limit-logged, DAG-HACC - P10 verdicts

* STRICT s0: D1 True, D2 False, D3 True; separated-worse ['curvature MAE', 'L2 curvature MAE (GT>=5.2)']; not computed []
* STRICT s1: D1 True, D2 False, D3 True; separated-worse ['curvature MAE', 'L2 curvature MAE (GT>=5.2)']; not computed []
* limit logged (S-POS not computed) s0: D1 True, D2 False, D3 False; separated-worse ['curvature MAE', 'L2 curvature MAE (GT>=5.2)']; not computed ['S-POS@30', 'S-POS@50', 'S-POS@80']
* limit logged (S-POS not computed) s1: D1 True, D2 False, D3 False; separated-worse ['yaw-rate MAE', 'curvature MAE', 'L2 curvature MAE (GT>=5.2)']; not computed ['S-POS@30', 'S-POS@50', 'S-POS@80']

Tables: `raw/READ_A_beside_tables_DAGHACC.md`.

### 5.3 STRATEGIC S-POS@30/50/80 against CTRV (within run, unpaired)

| run | S-POS@30 | S-POS@50 | S-POS@80 |
|---|---|---|---|
| DAGHACC_s0_decisive | -1.99 [-2.60, -1.47] | -5.11 [-6.32, -4.01] | -10.77 [-13.11, -8.65] |
| DAGHACCSTRICT_s0_beside | -1.62 [-2.20, -1.12] | -4.14 [-5.27, -3.13] | -8.40 [-10.44, -6.56] |
| DAG_s0 (plain, DAG_READ.json) | -1.72 [-2.27, -1.28] | -4.48 [-5.63, -3.50] | -9.69 [-11.84, -7.57] |
| DAGHACC_s1_decisive | -2.02 [-2.60, -1.52] | -5.44 [-6.70, -4.29] | -12.07 [-14.57, -9.78] |
| DAGHACCSTRICT_s1_beside | -1.42 [-1.94, -0.97] | -4.01 [-5.06, -3.07] | -9.17 [-11.27, -7.35] |
| DAG_s1 (plain, DAG_READ.json) | -1.87 [-2.36, -1.44] | -4.90 [-5.96, -3.92] | -10.33 [-12.44, -8.35] |

### 5.4 The four families at T0 (fresh ages, eval-141, T1-class open loop): DAG-HACC - DAG

| row | s0 [lo, hi] | sep | s1 [lo, hi] | sep |
|---|---|---|---|---|
| ade1 | -0.0111 [-0.0146, -0.0081] | yes | -0.0146 [-0.0183, -0.0114] | yes |
| ade2 | -0.0873 [-0.1192, -0.0619] | yes | -0.1038 [-0.1347, -0.0803] | yes |
| ade6 | -0.0001 [-0.0491, +0.0474] | no | -0.0714 [-0.1309, -0.0119] | yes |
| fde6 | +0.2722 [+0.1076, +0.4361] | yes | +0.0511 [-0.1454, +0.2584] | no |
| fw_lon_speed_mae | -0.0729 [-0.1110, -0.0397] | yes | -0.1228 [-0.1626, -0.0855] | yes |
| fw_lon_accel_mae | -0.3924 [-0.4714, -0.3222] | yes | -0.4644 [-0.5433, -0.3971] | yes |
| fw_lon_along_mae | -0.0257 [-0.0683, +0.0154] | no | -0.0882 [-0.1449, -0.0314] | yes |
| fw_lat_cross_mae | +0.0346 [+0.0052, +0.0626] | yes | +0.0198 [-0.0079, +0.0473] | no |
| fw_lat_heading_mae_deg | -0.0842 [-0.3763, +0.1628] | no | -0.1019 [-0.4107, +0.1393] | no |
| fw_lat_yaw_rate_mae_degps | -1.7110 [-2.2743, -1.2138] | yes | -2.1639 [-2.8888, -1.5601] | yes |
| fw_lat_curvature_mae_1pm | -0.0235 [-0.0347, -0.0142] | yes | -0.0309 [-0.0456, -0.0194] | yes |
| fw_tac_lat_acc | -0.0075 [-0.0229, +0.0096] | no | +0.0038 [-0.0075, +0.0158] | no |
| fw_tac_lon_acc | +0.0046 [-0.0108, +0.0213] | no | +0.0142 [-0.0042, +0.0346] | no |
| consistency_viol_any | +0.0609 [+0.0400, +0.0846] | yes | +0.0742 [+0.0513, +0.0989] | yes |

TACTICAL decision heads and STRATEGIC checkpoint outputs vs plain DAG, max |diff| (post-map touches the plan only): s0: 0.00e+00; s1: 0.00e+00.

## 6. Declarations, deviations, disclosures

The M2b-side controls (C-H1, C-H2) were the only things run before this file (read-only inspection of the repo export and of the checkpoint structure).

* **D1 (lambda0's reference weight).** HACC's lambda0 = (position term)/(D2 term) "at HBLEND's banked w". HBLEND's w is the output of ONE recipe
  (`v8f_hblend.fit_weights`: closed form per step and axis, clip to [0,1], then isotonic) applied to the arm's own plans. The PREREG changes
  "only the plans (DAG's in place of M2b's)", so the reference weight for DAG is that same recipe applied to DAG's OWN T0 plans on W_fit.
  M2b's banked weights are NOT used for DAG (that would put M2b's plans into DAG's fit). The literal alternative (M2b's banked w as the
  reference) is computed BESIDE, as a sensitivity on lambda*, and decides nothing.
* **D2 (how DAG's T0 plans on W_fit are produced).** Exactly HBLEND's `cmd_fit` route with only the checkpoint changed: the 141 TRAIN clips of
  `fit_clips.json` (k = 32), `train_windows(hist, start)` of the checkpoint's own config (9,582 windows expected), ages = the cycle schedule
  (a_t = i mod 3, a_s = a_t + (i div 3) mod 5, i = index in the window list), `forward_infer` (p_gt 0, T's own code, no label read),
  tau := traj30, P := p10, y := gt. The plan-production route is checked on M2b first: it must reproduce HBLEND's banked
  `fit_windows_M2b_s{0,1}.npz` (traj30, p30, gt, ages) before it is used for DAG (control C-X1, added; it touches no bar).
* **D3 (the code that runs).** `v8f_hacc.py`, `v8f_hblend.py`, `v8f_hspec.py`, `accel_profile.py` exported from tip 2a416ca7. Only their hard-coded
  scratchpad path constants are re-pointed (a patch script with asserted substitution counts; diff in `code/PATCHES.md`). No arithmetic is edited.
* **D4 (the bake script).** `bake_hacc.py` pins M2b's source checkpoint sha256, the M2b weight tables and the banked M2b HACC plan. Those pins are
  M2b-specific and cannot hold for DAG. The bake MECHANISM (`V8FMini.load_hacc_weights`, then `torch.save({**ck, "model": state_dict, "hacc": prov})`) is
  run unchanged; only the pins are replaced by DAG's (source sha256, DAG weight table, hash of the weights JSON). The M2b-banked-HACC agreement check (its step 5)
  has no DAG analogue and is replaced by C-H3 / C-H4. Reported as a deviation in the result.
* **D5 (C-H3's windows).** "24 eval windows" = the first 24 windows of eval-141 in the harness's own order, plus ALL 2,399 beside.
* **D6 (the logged-limit dumps).** Read A's `t1kf_read_a1` reader consumes three dump configs per seed (decisive, `--limit logged`, strict). The PREREG
  names two decisive rollouts and STRICT beside; the `--limit logged` rollouts (same flags, `--limit logged`) are added BESIDE so the unchanged reader can run.
  They decide nothing.

* **The fit-set cache is not in the repo.** It was read in place (read-only) from the previous HBLEND package in the Master Mind scratchpad (`scratchpad/hblend/cache/train/clips`, 141 clips, 2.8 GB; `box2/join_train.npz`, 34.7 MB, sha256 `978f41bf...`): it should be banked (e.g. to E: or box 2) before that scratchpad is cleaned.
* HBLEND's weights JSON blobs in the repo are LF; the pinned sha256 in `v8f_hacc.py` is of the CRLF original. The LF blob converted to CRLF reproduces the pinned sha256 for both seeds (checked) and was used.
* All code runs from `code/` copies of tip 2a416ca7 with only scratchpad path constants re-pointed (`code/PATCHES.md`).
