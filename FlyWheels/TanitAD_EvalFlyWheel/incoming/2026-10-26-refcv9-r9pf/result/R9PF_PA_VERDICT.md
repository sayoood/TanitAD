# R9PF_PA_VERDICT -- VOID

**tier:** T1 self-action OPEN LOOP on logged eval139 frames, not driving performance  
**estimator:** paired episode-cluster bootstrap over EPISODES, B 2000 seed 0 — blind to training and inference variance; fit seeds 0/1/2 = training-variance read. (pair_read_r8.paired_cell -> taniteval.ci, models fixed; inference seeds 0 / 1 reported separately, never pooled; ONE refcv8 trunk seed)  
**questions answered:** another draw of EPISODES (the bootstrap); another TRAINING run = fit seeds 0/1/2 (PA-R); another INFERENCE run = seeds 0 / 1 reported apart (floor beside).

**amendment:** PREREG_REFCV9_INMODEL_PA_AMENDMENT1 (sha256 1c25a0640def6378ed33221d016dad7925a6784d5c39eb2082b8d198cd1714ac) -- controls only; every bar, arm, population and estimator unchanged. Prereg sha256 67d178600a87ba128d8b5e76adbbf32a6fcfe7ac5374e38f1375ddd1fe918da8.

**Disclosed (AMENDMENT1 change 5):**

* **(a) tree drift (NOT an R9 effect; owned by H-R9-TREEDRIFT, not investigated here).** The tip tree's planner without the R9 head selects a different path from the fb47a58-rolled bank on 70 / 70 calibration windows (max 7.24 m; z_tac bit-equal; deterministic run-to-run). So the bars are composed OFFLINE on the BANKED fb47a58 paths (the same path as H2CN_DEPLOY8 / RETIME_DEPLOY8), and the in-forward roll serves K-OFF / K-EQ only.
* **(b) a DEFECT in the banked R9 code, fixed by one line.** `enable_refcv9` called `torch.manual_seed` inside `fork_rng(devices=[])`, which re-seeds the CUDA RNG and does not restore it: merely building the head moved the selected path on 70 / 70 calibration windows (max 8.10 m). Fix: `torch.default_generator.manual_seed` (CPU generator only), pinned in FIX_R9_RNG.sha256; flags-ON == flags-OFF bit for bit after it on those 70 calibration windows; the registered eval139 check is K-OFF in the controls table.
* **CONSTZ fit-side flag.** On the stage-2 fit (calibration split, out-of-fold CV objective), R9PF-CONSTZ -- z_tac replaced by its training mean, i.e. NO vision -- scored BETTER than R9PF: 1.83-1.85 vs 1.95-1.97 over fit seeds 0/1/2 (R9PF_FIT_INFO.json). K-CONSTZ and the INERT branch decide it on eval, as sealed.

**committed outcome (prereg s5):** VOID -> a control failed (prereg s3: 'a failed control VOIDS the bars'); the bars are reported BESIDE as unregistered numbers only.

**every bar, as written (primary R9PF_fs0):** PA-1a FAILS, PA-1b FAILS, PA-2 HOLDS, PA-G FAILS, PA-R FAILS -> PASS = False.

**failed controls:** K-EQ

## Bars (primary R9PF_fs0; S6 compliant unless stated; paired on the SAME draw)

| bar | seed 0 | seed 1 | holds |
|---|---|---|---|
| PA-1a MD vs H2CN_DEPLOY8 (sep below) | -0.0391 [-0.0783, +0.0000] (0.4945 vs 0.5336, n=639/79 ep) | -0.0376 [-0.0772, +0.0015] (0.4945 vs 0.5321, n=639/79 ep) | False |
| PA-1a FD vs H2CN_DEPLOY8 (NOT sep above) | +0.0065 [-0.0125, +0.0243] (0.0668 vs 0.0603, n=1856/135 ep) | +0.0065 [-0.0125, +0.0243] (0.0668 vs 0.0603, n=1856/135 ep) | |
| PA-1b FD vs FACTORED_DEPLOY8 (sep below) | -0.0835 [-0.1208, -0.0497] SEP (0.0668 vs 0.1503, n=1856/135 ep) | -0.0857 [-0.1231, -0.0529] SEP (0.0668 vs 0.1525, n=1856/135 ep) | False |
| PA-1b MD vs FACTORED_DEPLOY8 (NOT sep above) | +0.1049 [+0.0369, +0.1840] SEP (0.4945 vs 0.3897, n=639/79 ep) | +0.1049 [+0.0393, +0.1802] SEP (0.4945 vs 0.3897, n=639/79 ep) | |
| PA-2 S2 ADE vs RETIME_DEPLOY8 (sep below) | -0.0061 [-0.0127, -0.0000] SEP (0.2454 vs 0.2516, n=4159/138 ep) | -0.0067 [-0.0129, -0.0007] SEP (0.2462 vs 0.2529, n=4159/138 ep) | True |
| PA-2 S6 ADE vs RETIME_DEPLOY8 (sep below) | -0.2042 [-0.3553, -0.0545] SEP (2.3373 vs 2.5415, n=3251/138 ep) | -0.1991 [-0.3436, -0.0545] SEP (2.3430 vs 2.5422, n=3251/138 ep) | |
| PA-2 E1 (S2) vs ha0_ext_pose (sep below) | -0.0404 [-0.0518, -0.0291] SEP (0.2454 vs 0.2858, n=4159/138 ep) | -0.0396 [-0.0505, -0.0287] SEP (0.2462 vs 0.2858, n=4159/138 ep) | |
| PA-2 E3 (S6) vs ha0_ext_pose (sep below) | -1.0725 [-1.3617, -0.8062] SEP (2.3373 vs 3.4098, n=3251/138 ep) | -1.0668 [-1.3536, -0.8010] SEP (2.3430 vs 3.4098, n=3251/138 ep) | |
| PA-G exceedance vs H2CN_DEPLOY8 (NOT sep above; full plan, all windows) | +0.0032 [-0.0040, +0.0101] (0.1041 vs 0.1010, n=4754/139 ep) | +0.0032 [-0.0040, +0.0101] (0.1041 vs 0.1010, n=4754/139 ep) | False |
| PA-R (PA-1a, 1b, 2 at fit seeds 0/1/2) | | | False |

PA-G separated-worse rows (primary): ['s0|S2|ADE|ade_m: +0.0067 [+0.0025, +0.0109] SEP (0.2454 vs 0.2387, n=4159/138 ep)', 's0|S2|ADE|fde_m: +0.0107 [+0.0003, +0.0207] SEP (0.5258 vs 0.5151, n=4159/138 ep)', 's0|S2|LONGITUDINAL|LON_speed_mae_mps: +0.0114 [+0.0060, +0.0166] SEP (0.2243 vs 0.2130, n=4159/138 ep)', 's0|S2|LONGITUDINAL|LON_along_mae_m: +0.0078 [+0.0028, +0.0126] SEP (0.1975 vs 0.1898, n=4159/138 ep)', 's0|S2|LONGITUDINAL|LON_along_abs_final_m: +0.0123 [+0.0007, +0.0229] SEP (0.4323 vs 0.4199, n=4159/138 ep)', 's0|S2|LONGITUDINAL|LON_speed_within_0p5: -0.0073 [-0.0134, -0.0015] SEP (0.8803 vs 0.8876, n=4159/138 ep)', 's0|S2|LONGITUDINAL|LON_accel_mae_mps2: +0.0400 [+0.0311, +0.0486] SEP (0.3319 vs 0.2919, n=4159/138 ep)', 's0|S2|LATERAL|LAT_curvature_mae_1pm_valid: +0.0000 [+0.0000, +0.0001] SEP (0.0037 vs 0.0037, n=3892/138 ep)', 's0|S6|ADE|ade_m: +0.1250 [+0.0658, +0.1919] SEP (2.3373 vs 2.2123, n=3251/138 ep)', 's0|S6|ADE|fde_m: +0.3189 [+0.1533, +0.5018] SEP (5.7460 vs 5.4271, n=3251/138 ep)', 's0|S6|LONGITUDINAL|LON_speed_mae_mps: +0.0547 [+0.0250, +0.0871] SEP (0.8374 vs 0.7826, n=3251/138 ep)', 's0|S6|LONGITUDINAL|LON_along_mae_m: +0.1187 [+0.0586, +0.1853] SEP (1.9229 vs 1.8042, n=3251/138 ep)', 's0|S6|LONGITUDINAL|LON_along_abs_final_m: +0.2764 [+0.1098, +0.4653] SEP (4.6591 vs 4.3827, n=3251/138 ep)', 's1|S2|ADE|ade_m: +0.0068 [+0.0026, +0.0109] SEP (0.2462 vs 0.2394, n=4159/138 ep)', 's1|S2|ADE|fde_m: +0.0109 [+0.0004, +0.0208] SEP (0.5274 vs 0.5166, n=4159/138 ep)', 's1|S2|LONGITUDINAL|LON_speed_mae_mps: +0.0114 [+0.0060, +0.0166] SEP (0.2243 vs 0.2129, n=4159/138 ep)', 's1|S2|LONGITUDINAL|LON_along_mae_m: +0.0077 [+0.0027, +0.0126] SEP (0.1977 vs 0.1899, n=4159/138 ep)', 's1|S2|LONGITUDINAL|LON_along_abs_final_m: +0.0123 [+0.0007, +0.0227] SEP (0.4328 vs 0.4205, n=4159/138 ep)', 's1|S2|LONGITUDINAL|LON_speed_within_0p5: -0.0072 [-0.0133, -0.0015] SEP (0.8804 vs 0.8876, n=4159/138 ep)', 's1|S2|LONGITUDINAL|LON_accel_mae_mps2: +0.0400 [+0.0311, +0.0486] SEP (0.3318 vs 0.2918, n=4159/138 ep)', 's1|S2|LATERAL|LAT_curvature_mae_1pm_valid: +0.0000 [+0.0000, +0.0001] SEP (0.0037 vs 0.0037, n=3892/138 ep)', 's1|S6|ADE|ade_m: +0.1241 [+0.0650, +0.1906] SEP (2.3430 vs 2.2189, n=3251/138 ep)', 's1|S6|ADE|fde_m: +0.3142 [+0.1467, +0.4988] SEP (5.7593 vs 5.4451, n=3251/138 ep)', 's1|S6|LONGITUDINAL|LON_speed_mae_mps: +0.0547 [+0.0250, +0.0871] SEP (0.8372 vs 0.7825, n=3251/138 ep)', 's1|S6|LONGITUDINAL|LON_along_mae_m: +0.1163 [+0.0559, +0.1828] SEP (1.9231 vs 1.8068, n=3251/138 ep)', 's1|S6|LONGITUDINAL|LON_along_abs_final_m: +0.2695 [+0.1012, +0.4601] SEP (4.6608 vs 4.3913, n=3251/138 ep)']

## Every head: MD / FD (S6 compliant) and the bars

| head | MD s0 | FD s0 | MD s1 | FD s1 | PA-1a | PA-1b | PA-2 | PA-G | exceed s0 / s1 |
|---|---|---|---|---|---|---|---|---|---|
| R9PF-CONSTZ_fs0 | 0.5086 | 0.1121 | 0.5086 | 0.1121 | False | False | False | False | 0.0782 / 0.0782 |
| R9PF-CONSTZ_fs1 | 0.5321 | 0.1018 | 0.5321 | 0.1018 | False | False | False | False | 0.0780 / 0.0780 |
| R9PF-CONSTZ_fs2 | 0.5681 | 0.0927 | 0.5681 | 0.0927 | False | False | False | False | 0.0793 / 0.0793 |
| R9PF-SHUF_fs0 | 0.4867 | 0.1024 | 0.4867 | 0.1024 | False | False | False | False | 0.1214 / 0.1214 |
| R9PF-SHUF_fs1 | 0.4523 | 0.1083 | 0.4523 | 0.1083 | False | True | True | False | 0.1115 / 0.1115 |
| R9PF-SHUF_fs2 | 0.4601 | 0.1034 | 0.4601 | 0.1034 | False | False | False | False | 0.1308 / 0.1308 |
| R9PF-TL_fs0 | 0.5274 | 0.1164 | 0.5274 | 0.1164 | False | False | False | False | 0.0991 / 0.0991 |
| R9PF-TL_fs1 | 0.5274 | 0.1185 | 0.5274 | 0.1185 | False | False | False | False | 0.1031 / 0.1031 |
| R9PF-TL_fs2 | 0.5383 | 0.1196 | 0.5383 | 0.1196 | False | False | False | False | 0.1018 / 0.1018 |
| R9PF_fs0 | 0.4945 | 0.0668 | 0.4945 | 0.0668 | False | False | True | False | 0.1041 / 0.1041 |
| R9PF_fs1 | 0.4664 | 0.0765 | 0.4664 | 0.0765 | True | False | False | False | 0.1119 / 0.1119 |
| R9PF_fs2 | 0.5446 | 0.0560 | 0.5446 | 0.0560 | False | False | False | False | 0.1054 / 0.1054 |
| H2CN_DEPLOY8 (beside) | 0.5336 | 0.0603 | 0.5321 | 0.0603 | | | | | 0.1010 / 0.1010 |
| FACTORED_DEPLOY8 (beside) | 0.3897 | 0.1503 | 0.3897 | 0.1525 | | | | | 0.0753 / 0.0759 |
| RETIME_DEPLOY8 (beside) | 0.4695 | 0.1369 | 0.4710 | 0.1374 | | | | | 0.0757 / 0.0770 |
| OSONLY = the battery os-only plan, no re-timing (beside) | 0.4695 | 0.1395 | 0.4789 | 0.1395 | | | | | 0.0515 / 0.0513 |

GT-oracle pick from the same fan: MD 0.075 / FD 0.057 (INHERITED from PLAN_REFCV9 s12 via the prereg s0; not re-run here).

n: MD on 639 GT-decel windows, FD on 1856 GT-cruise windows (S6 compliant, seed 0).

## Controls

| control | PASS | note |
|---|---|---|
| compose_ALL_PASS | True |  |
| K-BESIDE-PINNED | True | {} |
| K-ALIGN|s0 | True | {} |
| K-GT|s0 | True | {} |
| K-E13|s0 | True | {} |
| K-ZERO|s0 | True | {} |
| K-EXC|s0 | True | {"mine": 480, "landed": 480} |
| K-ALIGN|s1 | True | {} |
| K-GT|s1 | True | {} |
| K-E13|s1 | True | {} |
| K-ZERO|s1 | True | {} |
| K-EXC|s1 | True | {"mine": 480, "landed": 480} |
| K-CONSTZ | True | {"constz_passes": {"fs0": {"PA-1a": false, "PA-1b": false}, "fs1": {"PA-1a": false, "PA-1b": false}, "fs2": {"PA-1a": false, "PA-1b": false}}, "reading": "CONSTZ passing PA-1a OR PA-1b at ANY fit seed voids the run (the conservative reading of 'must NOT pass PA-1')"} |
| K-SHUF | True | {"shuf_passes_PA-1a": {"fs0": false, "fs1": false, "fs2": false}} |
| K-EQ | False | {"rule": "AMENDMENT1 change 3: in-forward R9 plan vs the offline composition on the forward's OWN selected path (same inputs); per window max |d| <= 2 x ulp_float32(max |coordinate|) AND 100 % decision agreement, >= 500 eval windows; both seeds required (the conservative reading)", "per_seed": {"s0" |
| K-OFF | True | {"rule": "AMENDMENT1 change 2: one tree (tip + patch + pinned fix), flags-ON vs flags-OFF: traj_preretime and every non-R9 output bit-equal, >= 500 eval windows, both inference seeds", "per_seed": {"s0": {"n_calls_on": 515, "n_calls_off": 515, "n_windows": 515, "subset_equal": true, "same_tree": tru |
| K-GEO | True | {"max_abs_m": 0.0} |
| K-LAT | True | {"max_dist_f64_m": 2.864289338439558e-14, "max_dist_after_f32_cast_m": 4.939631284203246e-06} |

## Four families -- R9PF_fs0 minus H2CN_DEPLOY8 (compliant sets; worse = separated in the bad direction)

| surface | family | metric | seed 0 | seed 1 |
|---|---|---|---|---|
| S2 | ADE | ade_m | +0.0067 [+0.0025, +0.0109] SEP (0.2454 vs 0.2387, n=4159/138 ep) | +0.0068 [+0.0026, +0.0109] SEP (0.2462 vs 0.2394, n=4159/138 ep) |
| S2 | ADE | fde_m | +0.0107 [+0.0003, +0.0207] SEP (0.5258 vs 0.5151, n=4159/138 ep) | +0.0109 [+0.0004, +0.0208] SEP (0.5274 vs 0.5166, n=4159/138 ep) |
| S2 | LONGITUDINAL | LON_speed_mae_mps | +0.0114 [+0.0060, +0.0166] SEP (0.2243 vs 0.2130, n=4159/138 ep) | +0.0114 [+0.0060, +0.0166] SEP (0.2243 vs 0.2129, n=4159/138 ep) |
| S2 | LONGITUDINAL | LON_along_mae_m | +0.0078 [+0.0028, +0.0126] SEP (0.1975 vs 0.1898, n=4159/138 ep) | +0.0077 [+0.0027, +0.0126] SEP (0.1977 vs 0.1899, n=4159/138 ep) |
| S2 | LONGITUDINAL | LON_along_abs_final_m | +0.0123 [+0.0007, +0.0229] SEP (0.4323 vs 0.4199, n=4159/138 ep) | +0.0123 [+0.0007, +0.0227] SEP (0.4328 vs 0.4205, n=4159/138 ep) |
| S2 | LONGITUDINAL | LON_speed_within_0p5 (higher better) | -0.0073 [-0.0134, -0.0015] SEP (0.8803 vs 0.8876, n=4159/138 ep) | -0.0072 [-0.0133, -0.0015] SEP (0.8804 vs 0.8876, n=4159/138 ep) |
| S2 | LONGITUDINAL | LON_accel_mae_mps2 | +0.0400 [+0.0311, +0.0486] SEP (0.3319 vs 0.2919, n=4159/138 ep) | +0.0400 [+0.0311, +0.0486] SEP (0.3318 vs 0.2918, n=4159/138 ep) |
| S2 | LATERAL | LAT_heading_mae_deg | +0.0009 [-0.0048, +0.0066] (1.4979 vs 1.4970, n=3936/138 ep) | +0.0008 [-0.0050, +0.0066] (1.5084 vs 1.5076, n=3936/138 ep) |
| S2 | LATERAL | LAT_curvature_mae_1pm_valid | +0.0000 [+0.0000, +0.0001] SEP (0.0037 vs 0.0037, n=3892/138 ep) | +0.0000 [+0.0000, +0.0001] SEP (0.0037 vs 0.0037, n=3892/138 ep) |
| S2 | LATERAL | LAT_yaw_rate_mae_radps_valid | +0.0001 [-0.0001, +0.0003] (0.0176 vs 0.0175, n=3892/138 ep) | +0.0001 [-0.0001, +0.0003] (0.0177 vs 0.0175, n=3892/138 ep) |
| S2 | LATERAL | LAT_cross_mae_m | +0.0003 [-0.0013, +0.0017] (0.0958 vs 0.0955, n=4159/138 ep) | +0.0003 [-0.0012, +0.0017] (0.0968 vs 0.0964, n=4159/138 ep) |
| S2 | LATERAL | LAT_cross_abs_final_m | +0.0002 [-0.0052, +0.0048] (0.1947 vs 0.1945, n=4159/138 ep) | +0.0003 [-0.0051, +0.0049] (0.1969 vs 0.1966, n=4159/138 ep) |
| S2 | TACTICAL | TAC_traj_lat_correct (higher better) | -0.0002 [-0.0017, +0.0012] (0.9635 vs 0.9637, n=4159/138 ep) | -0.0005 [-0.0020, +0.0010] (0.9632 vs 0.9637, n=4159/138 ep) |
| S2 | TACTICAL | TAC_traj_lon_correct (higher better) | -0.0005 [-0.0124, +0.0143] (0.8666 vs 0.8670, n=4159/138 ep) | -0.0005 [-0.0124, +0.0143] (0.8666 vs 0.8670, n=4159/138 ep) |
| S6 | ADE | ade_m | +0.1250 [+0.0658, +0.1919] SEP (2.3373 vs 2.2123, n=3251/138 ep) | +0.1241 [+0.0650, +0.1906] SEP (2.3430 vs 2.2189, n=3251/138 ep) |
| S6 | ADE | fde_m | +0.3189 [+0.1533, +0.5018] SEP (5.7460 vs 5.4271, n=3251/138 ep) | +0.3142 [+0.1467, +0.4988] SEP (5.7593 vs 5.4451, n=3251/138 ep) |
| S6 | LONGITUDINAL | LON_speed_mae_mps | +0.0547 [+0.0250, +0.0871] SEP (0.8374 vs 0.7826, n=3251/138 ep) | +0.0547 [+0.0250, +0.0871] SEP (0.8372 vs 0.7825, n=3251/138 ep) |
| S6 | LONGITUDINAL | LON_along_mae_m | +0.1187 [+0.0586, +0.1853] SEP (1.9229 vs 1.8042, n=3251/138 ep) | +0.1163 [+0.0559, +0.1828] SEP (1.9231 vs 1.8068, n=3251/138 ep) |
| S6 | LONGITUDINAL | LON_along_abs_final_m | +0.2764 [+0.1098, +0.4653] SEP (4.6591 vs 4.3827, n=3251/138 ep) | +0.2695 [+0.1012, +0.4601] SEP (4.6608 vs 4.3913, n=3251/138 ep) |
| S6 | LONGITUDINAL | LON_speed_within_0p5 (higher better) | -0.0096 [-0.0286, +0.0091] (0.5573 vs 0.5670, n=3251/138 ep) | -0.0096 [-0.0286, +0.0091] (0.5573 vs 0.5669, n=3251/138 ep) |
| S6 | LONGITUDINAL | LON_accel_mae_mps2 | -0.0184 [-0.0322, -0.0048] SEP (0.4049 vs 0.4233, n=3251/138 ep) | -0.0184 [-0.0322, -0.0047] SEP (0.4049 vs 0.4233, n=3251/138 ep) |
| S6 | LATERAL | LAT_heading_mae_deg | +0.0751 [-0.0120, +0.2166] (3.3083 vs 3.2332, n=3156/138 ep) | +0.0801 [-0.0075, +0.2212] (3.3393 vs 3.2591, n=3156/138 ep) |
| S6 | LATERAL | LAT_curvature_mae_1pm_valid | -0.0001 [-0.0002, +0.0000] (0.0055 vs 0.0055, n=3123/138 ep) | -0.0000 [-0.0002, +0.0001] (0.0055 vs 0.0055, n=3123/138 ep) |
| S6 | LATERAL | LAT_yaw_rate_mae_radps_valid | +0.0001 [-0.0003, +0.0005] (0.0279 vs 0.0278, n=3123/138 ep) | +0.0002 [-0.0002, +0.0006] (0.0280 vs 0.0279, n=3123/138 ep) |
| S6 | LATERAL | LAT_cross_mae_m | +0.0088 [-0.0121, +0.0298] (0.8479 vs 0.8391, n=3251/138 ep) | +0.0087 [-0.0120, +0.0295] (0.8548 vs 0.8461, n=3251/138 ep) |
| S6 | LATERAL | LAT_cross_abs_final_m | +0.0409 [-0.0213, +0.1056] (2.1589 vs 2.1181, n=3251/138 ep) | +0.0380 [-0.0237, +0.0991] (2.1766 vs 2.1385, n=3251/138 ep) |
| S6 | TACTICAL | TAC_traj_lat_correct (higher better) | -0.0012 [-0.0053, +0.0025] (0.8886 vs 0.8899, n=3251/138 ep) | -0.0015 [-0.0046, +0.0012] (0.8911 vs 0.8926, n=3251/138 ep) |
| S6 | TACTICAL | TAC_traj_lon_correct (higher better) | -0.0055 [-0.0333, +0.0219] (0.6349 vs 0.6404, n=3251/138 ep) | -0.0055 [-0.0333, +0.0219] (0.6349 vs 0.6404, n=3251/138 ep) |

STRATEGIC: n/a (refcv8 runs --no-strategic). Distance-keeping (headway / TTC): not instrumented in this battery (no lead-agent join in the eval139 panel) -- n/a with reason.

## Four families -- R9PF_fs0 minus RETIME_DEPLOY8 (beside)

| surface | metric | seed 0 | seed 1 |
|---|---|---|---|
| S2 | ade_m | -0.0061 [-0.0127, -0.0000] SEP (0.2454 vs 0.2516, n=4159/138 ep) | -0.0067 [-0.0129, -0.0007] SEP (0.2462 vs 0.2529, n=4159/138 ep) |
| S2 | fde_m | -0.0133 [-0.0294, +0.0014] (0.5258 vs 0.5391, n=4159/138 ep) | -0.0139 [-0.0295, +0.0006] (0.5274 vs 0.5413, n=4159/138 ep) |
| S2 | LON_speed_mae_mps | -0.0046 [-0.0134, +0.0038] (0.2243 vs 0.2290, n=4159/138 ep) | -0.0048 [-0.0137, +0.0035] (0.2243 vs 0.2291, n=4159/138 ep) |
| S2 | LON_along_mae_m | -0.0079 [-0.0153, -0.0011] SEP (0.1975 vs 0.2055, n=4159/138 ep) | -0.0081 [-0.0152, -0.0015] SEP (0.1977 vs 0.2058, n=4159/138 ep) |
| S2 | LON_along_abs_final_m | -0.0179 [-0.0357, -0.0013] SEP (0.4323 vs 0.4502, n=4159/138 ep) | -0.0170 [-0.0346, -0.0011] SEP (0.4328 vs 0.4498, n=4159/138 ep) |
| S2 | LON_speed_within_0p5 | +0.0084 [+0.0001, +0.0170] SEP (0.8803 vs 0.8720, n=4159/138 ep) | +0.0064 [-0.0016, +0.0150] (0.8804 vs 0.8740, n=4159/138 ep) |
| S2 | LON_accel_mae_mps2 | +0.0299 [+0.0149, +0.0449] SEP (0.3319 vs 0.3020, n=4159/138 ep) | +0.0304 [+0.0159, +0.0448] SEP (0.3318 vs 0.3015, n=4159/138 ep) |
| S2 | LAT_heading_mae_deg | +0.0020 [-0.0063, +0.0100] (1.4965 vs 1.4944, n=3932/138 ep) | +0.0016 [-0.0070, +0.0099] (1.5088 vs 1.5073, n=3931/138 ep) |
| S2 | LAT_curvature_mae_1pm_valid | -0.0000 [-0.0001, +0.0001] (0.0037 vs 0.0038, n=3893/138 ep) | +0.0000 [-0.0000, +0.0001] (0.0038 vs 0.0038, n=3894/138 ep) |
| S2 | LAT_yaw_rate_mae_radps_valid | +0.0001 [-0.0001, +0.0004] (0.0176 vs 0.0175, n=3893/138 ep) | +0.0002 [-0.0001, +0.0004] (0.0177 vs 0.0175, n=3894/138 ep) |
| S2 | LAT_cross_mae_m | +0.0011 [+0.0001, +0.0023] SEP (0.0958 vs 0.0946, n=4159/138 ep) | +0.0009 [-0.0002, +0.0021] (0.0968 vs 0.0958, n=4159/138 ep) |
| S2 | LAT_cross_abs_final_m | +0.0031 [+0.0001, +0.0063] SEP (0.1947 vs 0.1916, n=4159/138 ep) | +0.0025 [-0.0007, +0.0057] (0.1969 vs 0.1944, n=4159/138 ep) |
| S2 | TAC_traj_lat_correct | -0.0002 [-0.0021, +0.0017] (0.9635 vs 0.9637, n=4159/138 ep) | -0.0010 [-0.0027, +0.0007] (0.9632 vs 0.9642, n=4159/138 ep) |
| S2 | TAC_traj_lon_correct | +0.0125 [-0.0005, +0.0272] (0.8666 vs 0.8541, n=4159/138 ep) | +0.0108 [-0.0015, +0.0240] (0.8666 vs 0.8557, n=4159/138 ep) |
| S6 | ade_m | -0.2042 [-0.3553, -0.0545] SEP (2.3373 vs 2.5415, n=3251/138 ep) | -0.1991 [-0.3436, -0.0545] SEP (2.3430 vs 2.5422, n=3251/138 ep) |
| S6 | fde_m | -0.5986 [-1.0286, -0.1815] SEP (5.7460 vs 6.3446, n=3251/138 ep) | -0.5902 [-0.9886, -0.1861] SEP (5.7593 vs 6.3495, n=3251/138 ep) |
| S6 | LON_speed_mae_mps | -0.1109 [-0.1851, -0.0402] SEP (0.8374 vs 0.9482, n=3251/138 ep) | -0.1095 [-0.1805, -0.0393] SEP (0.8372 vs 0.9468, n=3251/138 ep) |
| S6 | LON_along_mae_m | -0.2747 [-0.4340, -0.1170] SEP (1.9229 vs 2.1976, n=3251/138 ep) | -0.2708 [-0.4206, -0.1191] SEP (1.9231 vs 2.1940, n=3251/138 ep) |
| S6 | LON_along_abs_final_m | -0.8103 [-1.2494, -0.3802] SEP (4.6591 vs 5.4694, n=3251/138 ep) | -0.8076 [-1.2240, -0.4015] SEP (4.6608 vs 5.4684, n=3251/138 ep) |
| S6 | LON_speed_within_0p5 | +0.0237 [-0.0011, +0.0478] (0.5573 vs 0.5336, n=3251/138 ep) | +0.0244 [-0.0007, +0.0485] (0.5573 vs 0.5329, n=3251/138 ep) |
| S6 | LON_accel_mae_mps2 | -0.0391 [-0.0685, -0.0118] SEP (0.4049 vs 0.4440, n=3251/138 ep) | -0.0370 [-0.0648, -0.0106] SEP (0.4049 vs 0.4419, n=3251/138 ep) |
| S6 | LAT_heading_mae_deg | +0.0790 [-0.0200, +0.2216] (3.2325 vs 3.1535, n=3144/138 ep) | +0.0870 [-0.0059, +0.2155] (3.2521 vs 3.1652, n=3142/138 ep) |
| S6 | LAT_curvature_mae_1pm_valid | -0.0000 [-0.0001, +0.0001] (0.0052 vs 0.0052, n=3109/138 ep) | -0.0000 [-0.0001, +0.0001] (0.0051 vs 0.0051, n=3108/138 ep) |
| S6 | LAT_yaw_rate_mae_radps_valid | +0.0000 [-0.0006, +0.0007] (0.0272 vs 0.0271, n=3109/138 ep) | +0.0001 [-0.0006, +0.0008] (0.0273 vs 0.0272, n=3108/138 ep) |
| S6 | LAT_cross_mae_m | +0.0421 [+0.0039, +0.0820] SEP (0.8479 vs 0.8058, n=3251/138 ep) | +0.0419 [+0.0027, +0.0832] SEP (0.8548 vs 0.8130, n=3251/138 ep) |
| S6 | LAT_cross_abs_final_m | +0.1120 [+0.0001, +0.2324] SEP (2.1589 vs 2.0469, n=3251/138 ep) | +0.1159 [-0.0020, +0.2418] (2.1766 vs 2.0606, n=3251/138 ep) |
| S6 | TAC_traj_lat_correct | -0.0009 [-0.0067, +0.0041] (0.8886 vs 0.8896, n=3251/138 ep) | +0.0000 [-0.0049, +0.0046] (0.8911 vs 0.8911, n=3251/138 ep) |
| S6 | TAC_traj_lon_correct | +0.0391 [-0.0006, +0.0797] (0.6349 vs 0.5958, n=3251/138 ep) | +0.0434 [+0.0037, +0.0845] SEP (0.6349 vs 0.5915, n=3251/138 ep) |

## INERT read (R9PF vs R9PF-TL, per fit seed, against the fit-seed floor)

```
{
 "INERT": false,
 "detail": {
  "s0|MD": {
   "fit_seed_floor": 0.07824726134585286,
   "R9PF_minus_TL": [
    -0.03286384976525819,
    -0.061032863849765195,
    0.006259780907668211
   ],
   "within_floor": true
  },
  "s0|FD": {
   "fit_seed_floor": 0.020474137931034475,
   "R9PF_minus_TL": [
    -0.04956896551724138,
    -0.042025862068965525,
    -0.06357758620689655
   ],
   "within_floor": false
  },
  "s1|MD": {
   "fit_seed_floor": 0.07824726134585286,
   "R9PF_minus_TL": [
    -0.03286384976525819,
    -0.061032863849765195,
    0.006259780907668211
   ],
   "within_floor": true
  },
  "s1|FD": {
   "fit_seed_floor": 0.020474137931034475,
   "R9PF_minus_TL": [
    -0.04956896551724138,
    -0.042025862068965525,
    -0.06357758620689655
   ],
   "within_floor": false
  }
 }
}
```

## Inference-seed floor beside the primary effects (|mean_new(s0) - mean_new(s1)|)

```
{
 "MD_vs_H8": 0.0,
 "FD_vs_H8": 0.0,
 "MD_vs_FX": 0.0,
 "FD_vs_FX": 0.0,
 "ADE_S2_vs_D8": 0.0007854771468251975,
 "ADE_S6_vs_D8": 0.00570982599631531
}
```

## Deviations (disclosed)

* **D1** Calibration rows = the fresh S6 COMPLIANT rows (7,147 rows / 297 episodes / 297 clips; S6 is a subset of S2, every slot valid, regime defined), not the 9,135 S2 rows: the regime target needs the 5.0 / 6.0 s slots. Design defect flagged for Phase B: refcv9_progress.r9_losses teacher-forces the profile on r* even where gt_regime's `ok` is False (rows without a 6 s future), where r* is undefined.
* **D2** CV score = the stage-2 objective itself (mean L1 progress over 8 slots + regime CE), pooled out-of-fold; H2cn's h2n_fit scored pooled held-out L1 only. Grid (MLP_WD x MLP_MAX_EP 200), folds (folds_of(eid, fit_seed), 5), AdamW lr 1e-3, batch 256, seeding: identical to h2n_fit.
* **D3** The head IS refcv9_progress.R9ProgressHead, fitted with its BatchNorm in EVAL mode on the training rows' population mean / var (the module's own eval formula is the fitted standardisation); the in-trainer stage 2 would accumulate BN statistics in train mode (unbiased batch variances). Fitted in float64, frozen in float32 (the in-model dtype): K-RELOAD 0 decision flips, max profile difference <= 2.8e-5 m on the fit rows.
* **D4** R9PF-TL shares R9PF's fitted head per fit seed (prereg s1: 'same profiles'), decision 'trunk_lon' from the battery dump's nav_true.tacv6_lon_logits (argmax over LON6 grouped to regimes); not separately fitted.
* **D5** R9PF-SHUF permutation = a seeded episode derangement (rng 20261026 + fit seed; rows position-matched mod donor length); no episode keeps its own labels, but ~40 % of rows keep their class by chance (cruise is 52 %).
* **D6** K-LABSRC-R9 mutation (labels from the fresh v9 lon_v7) reads RED mostly TRIVIALLY: 6,658 of 7,147 rows carry a lon_v7 id outside LON6 (unmapped); on the 489 mapped rows, 173 disagree with the kinematic target -- RED on those too.
* **D7** The prereg s1 EXPLORATORY calibration-tuned decel tau was not run in A2; it was run AFTER the bars (code/x_tau_fit.py: OOF regime logits on the calibration rows, the head's own wd / epochs and folds; tau = max Youden J for decel, ties -> larger tau; fs0/1/2 tau = 0.25 / 0.29 / 0.28) and is reported BESIDE only (result/R9PF_X_ATTRIB.json, arm TAU). It never enters a bar.
* **D8** Clip ids were printed ONCE to this agent's own terminal output while inspecting windows_s1.json's format; no file written by this run contains a raw clip id (sha12 only).
* **D9** The R9 code = a COPY of the patch's base tree 425762053c75 (== tip for refc_v3.py) + the banked patch (4b5f7fb; patch blob 0a165b98 verified; reverse dry-run clean on the copy); the K-EQ launch tree = that stack + taniteval@425762053c75 + the refcv7 battery code (identical tree in fb47a58).
* **D10** SUPERSEDED by AMENDMENT1 change 4: the PRIMARY eval path is the battery tree's BANKED fb47a58 plan (refcv6_extras nav_true.traj; bit-equal to the plan8 inside H2CN_DEPLOY8 and RETIME_DEPLOY8 at both seeds, K-INPUTS). The eval bank's traj_own equals it bit for bit at seed 1 but differs at seed 0 on 4,630 / 4,754 S2 windows (max 35.31 m); R9 compositions on traj_own are banked beside (*_own arrays). FACTORED_DEPLOY8's plan8 equals NEITHER path at either seed (it is the factored decoder's own native path), so PA-1b compares two decoders on their own paths, as the sealed s1 defines FACTORED_DEPLOY8. The OSONLY beside arm = the battery plan without re-timing.
* **D11** PA-2 read as the CONJUNCTION (conservative): paired ADE vs RETIME_DEPLOY8 separated below on S2 AND S6 compliant, AND the SHIP-R8 E1 / E3 form (vs ha0_ext_pose) separated below, at both inference seeds.
* **D12** K-CONSTZ read conservatively: R9PF-CONSTZ passing PA-1a OR PA-1b at ANY fit seed voids the run. PA-G is binding on the primary R9PF_fs0 (PA-R names only 1a / 1b / 2); fs1 / fs2 PA-G are reported beside.
* **D13** PA-G's exceedance = the registered compliance-row rule (planned_max_speed of the FULL 8-slot plan > the fed ceiling, strict) on all 4,754 windows, paired per window vs H2CN_DEPLOY8.
* **D14** SUPERSEDED by AMENDMENT1 changes 2 / 3: K-OFF = flags-ON vs flags-OFF on ONE tree (tip + R9 patch + the pinned one-line RNG fix, F:/refcv9_r9pf_scratch/launch_tree_fix) at inference seeds 0 AND 1; K-EQ = same-input form at seeds 0 AND 1 (both required -- the conservative reading; the amendment names seeds only for K-OFF). Subset = the first clips in sha12 order until >= 500 windows of the battery's windows_s{seed}.json, filtered in memory; the tip loader's pose-sync shift is blanked under R8_NO_POSE_SYNC=1 and asserted not attached. 'Every non-R9 output' = every tensor the model returns per call (sha256 of bytes + dtype + shape, all rows and conditions, nav-null calls included); the K-EQ same inputs = the forward's own traj_preretime, z_tac (lat_head_tac input) and r9_x, plus the scoring prior8 (s_prior agreement reported).
* **D15** PRE-GO diagnostics on the CALIBRATION split (70 fresh windows; not eval139): tip tree without R9 != fb47a58 bank plan on 70/70 (max 7.24 m; z bit-equal; deterministic run-to-run); R9 build reseeds CUDA RNG (torch.manual_seed inside fork_rng(devices=[])) -> path moves on 70/70 (max 8.10 m); one-line fix (torch.default_generator.manual_seed) restores flags-on == flags-off bit for bit; same-input K-EQ 1.53e-5 m (1 float32 ULP at 128-256 m) -- escalated to the MM before GO.
* **D16** ONE composition per window: the S6 compliant rows are a subset of the S2 rows, and the S6 arm := the S2 composition indexed to those windows (every S6 input asserted bit-equal to its S2 row). A separate S6-batch composition differs from the S2-batch one by float32 batch-order rounding only -- 7 / 3,668 rows, <= 7.63e-6 m, 0 decision flips (R9PF_COMPOSE.json, K-S6CONSIST 'separate_batch_composition_beside').
* **D17** The R9 RNG fix (AMENDMENT1 change 1) is a byte-exact one-line replacement at refc_v3.py line 2125 (CRLF preserved; code/FIX_R9_RNG.DIFF.txt; the fixed file, the diff, the unchanged refcv9_progress.py and the unfixed reference are pinned in FIX_R9_RNG.sha256). An earlier sed attempt converted the file's line endings and was discarded unused; the pinned file is the byte-clean one. Every a3 script re-hashes the amendment seal and every FIX_R9_RNG.sha256 line and refuses otherwise.
* **D18** Process note: two shell commands in this run fed a heredoc AND '</dev/null' to python; on Windows NUL is a console device, so python opened an interactive REPL that looped on an invalid handle (~64 MB of error text into the harness's own output file). Both processes were stopped by their task; neither wrote any file in the work dir or scratch, and both edits were re-applied from patch files.
* **D19** EXPLORATORY post-bar attribution (code/x_attrib_r9pf.py -> result/R9PF_X_ATTRIB.json; NOT a bar, NOT pre-registered): arms ORACLE (GT regime + R9 profiles; inadmissible), CRUISE, TAU, SOFT (posterior-weighted profile), HYB (R9 decision gating H2CN_DEPLOY8). Run under RULE ZERO to rank the next lever; every number there is labelled EXPLORATORY and none changes the verdict.
* **D20** K-OFF: `wp_seq` differs ON vs OFF on 515/515 calls at both seeds because it IS the emitted plan (refc_v3.py:2742 out['wp_seq'] = traj, set after the R9 re-time at :2723). The compare first read it as a non-R9 output (K-OFF FAIL), then classified it as a plan alias from data -- digest(wp_seq) == digest(traj) on every call of both runs -- and checks it like traj (ON traj_preretime == OFF wp_seq: 0 / 515 differ). No other output differs. K-OFF then PASSES at both seeds. Disclosed because the classification was made after seeing the first result.
* **D21** K-EQ FAILS as written at both seeds: decision agreement 100 % (515/515), but max |d| 3.05e-5 / 4.58e-5 m exceeds 2 ulp_float32 of the window's own max |coordinate| on 28 / 24 of 515 windows (max ratio 13). The stricter reading of the coordinate scope was used (the brief: conservative on ambiguity). BESIDE, not registered: the residual survives the forward's OWN s_prior (17 / 19 windows over) and a tolerance that includes the prior's coordinates (26 / 22 over), so it is not the prior. Unresolved: GPU-vs-CPU arithmetic in the head is the candidate (the in-forward head runs on CUDA, the offline one on CPU); the discriminating test is the same roll on CPU, which was not run.

## Next lever

**VOID first: the one failed control is K-EQ, a numerical-tolerance failure, not a logic one.** Decisions agree 515/515 at both seeds; max |d| is 3.05e-5 m (seed 0) and 4.58e-5 m (seed 1); 28 and 24 of 515 windows exceed 2 ulp of their own coordinates. Cheapest discriminator, not run: the same ON roll on CPU (~15-30 CPU-min, no GPU). If CPU in-forward == offline within 2 ulp -> the residual is CUDA-vs-CPU head arithmetic, and a Master-Mind ruling on the tolerance (amendment 2) would turn the bars below into the registered read. If not -> a real composition mismatch to find. **Blocked on:** that roll + the MM ruling. The bars are BESIDE until then. The provisional word on them, had K-EQ held, is FAIL-GUARD.

**RULE ZERO -- the failed bars leave the next arm, and the attribution that picks it was RUN (EXPLORATORY, post-hoc on eval139, result/R9PF_X_ATTRIB.json; T1 open loop; paired episode-cluster bootstrap, B 2000 seed 0; inference seed 0 shown, seed 1 within the same CIs; NOT a bar, NOT a claim).**

1. **The loss is the DECISION, not the profiles.** ORACLE (R9PF_fs0's own profiles with the GT regime -- inadmissible, attribution only) vs H2CN_DEPLOY8 on S6 compliant: ADE -0.3977 [-0.4761, -0.3188] m, MD 0.059 vs 0.534, 0 family rows separated worse (all three fit seeds: 0). The registered argmax decision gets 308 / 639 GT-decel and 355 / 756 GT-accel windows right (300 decel and 389 accel called cruise; 273 / 1856 cruise windows called decel/accel); OOF decel recall on the calibration split is 0.52-0.55 at a false-decel rate of 0.07-0.08 (x_tau_fit.py).
2. **Calibrated tau (prereg s1 beside, run now) makes it WORSE:** fs0 tau 0.25 -> MD -0.1768 [-0.2257, -0.1319] but FD +0.1277 [+0.0910, +0.1678], 15 rows worse. A threshold only moves along the same weak classifier's ROC.
3. **The soft decode removes the guard failure.** SOFT = sum_r p(r) dS[r] on the registered heads, no refit: fs0 MD -0.0751 [-0.1144, -0.0377], FD +0.0259 [+0.0061, +0.0442] vs H2CN_DEPLOY8; family rows separated worse 13 -> 1 (fs1 1, fs2 2 / 1); PA-2-like holds at every fit seed; PA-1b-like holds for fs0 (seed 0) and fs1 (both seeds). It still misses PA-1a on FD (+0.022 to +0.026, separated above) -- not a pass.

**Next arm (cheapest first; each needs a Master-Mind seal BEFORE it runs -- prereg s5 says the constrained stage-2 objective is 'pre-registered first'):**
* **R9PF-SOFT2 (0 GPU, ~15 CPU-min):** stage 2 refit with the loss evaluated through the decode it is used with -- L1 of s_P + sum_r p(r) dS[r] against s_H, plus the regime CE -- same split, grid, folds and fit seeds; decode = the posterior-weighted profile; bars PA-1a/1b/2/G/R as sealed. This is prereg s5's 'constrained stage-2 objective' aimed at the measured term (decision errors), with K-CONSTZ / K-SHUF carried.
* **The decision itself (the dominant lever):** a better regime classifier needs either more clean calibration clips (V-C; availability UNVERIFIED -- the blocker is a corpus check) or the lon decision trained in the trunk with richer supervision (Phase B; ~240 Thor-h ESTIMATED -- the blocker is the PI's compute decision D-R9-INMODEL Q3).
**Blocked on:** the Master Mind's seal for R9PF-SOFT2 (prereg s5); the PI for Phase B; a corpus availability check for V-C.
