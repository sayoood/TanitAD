# THE LATENT LINEAR LADDER — rendered tables

**Eval tier: T0-DIAGNOSTIC.** A frozen-latent readout is a world-model diagnostic and is NEVER driving performance.

Estimator `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000, seeds [0, 1, 2], fit_mode `pc6`. Features: cells [16, 128] flattened -> 2049 incl. bias.

**Pose-grid binding:** `poses[frame_idx + off]` with offset **2** (offset == n_stack - 1 == 2), accepted on EXACT equality of poses[.,3] and banked v0 over 5617 rows (max mismatch 0.0). Scan: {'-4': 1.977205, '-3': 1.65485, '-2': 1.309958, '-1': 0.985365, '0': 0.667246, '1': 0.344891, '2': 0.0}

## 1. THE LADDER — correlation with the truth, arm vs its controls

`r` = corr(prediction, truth). `r_wep` = the same AFTER both are demeaned by their own eval episode — the ANTI-EPISODE-IDENTITY statistic. `r_pv0` = partial correlation with EGO SPEED partialled out — the TRIVIAL-PROXY test.

| target | rung | unit | n | **v6 r** | **NULL r** | C-V0 r | ORACLE r | v6 r_wep | v6 r_pv0 |
|---|---|---|---|---|---|---|---|---|---|
| `ego_v0` | EGO (anchor) | m/s | 3023/70 | **+0.247** | -0.025 | +1.000 | +0.548 | +0.149 | -- |
| `ego_accel` | EGO | m/s^2 | 3023/70 | **+0.187** | -0.011 | -0.112 | +0.021 | +0.115 | +0.169 |
| `ego_yawrate` | EGO | rad/s | 3023/70 | **+0.060** | -0.008 | -0.020 | +0.031 | +0.099 | +0.063 |
| `ego_curv` | EGO | 1/m | 2221/70 | **-0.009** | -0.022 | -0.085 | -0.095 | +0.033 | +0.002 |
| `n_agents_grid` | SCENE | agents | 3023/70 | **+0.141** | -0.015 | +0.322 | +0.874 | +0.042 | +0.094 |
| `n_agents_all` | SCENE | agents | 3023/70 | **+0.275** | +0.027 | +0.335 | +0.660 | +0.101 | +0.194 |
| `lead_present` | OBJECT | prob | 3023/70 | **+0.093** | -0.012 | +0.152 | +0.414 | +0.090 | +0.076 |
| `nearest_any` | OBJECT | m | 3019/70 | **+0.219** | -0.024 | +0.408 | +0.856 | +0.013 | +0.179 |
| `lead_gap` | OBJECT | m | 2721/70 | **+0.159** | -0.018 | +0.683 | +0.979 | +0.166 | +0.052 |
| `lead_closing` | OBJECT-DYNAMICS | m/s | 2721/70 | **-0.006** | +0.005 | +0.017 | +0.432 | +0.067 | -0.004 |
| `lead_inv_ttc` | OBJECT-DYNAMICS | 1/s | 2721/70 | **-0.008** | +0.030 | -0.036 | +0.118 | +0.032 | -0.002 |

## 2. THE SAME LADDER IN ERROR AND K1 — every row with its null

Positive K1 = the arm is WORSE than the constant. `PASS` = separated and negative.

| target | unit | C-CONST | v6 err | v6 K1 | NULL err | NULL K1 | C-V0 err | C-V0 K1 | ORACLE err | ORACLE K1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ego_v0` | m/s | 4.0970 | 5.5256 | +1.429 fail | 5.2149 | +1.118 fail | 0.0000 | -4.097 **PASS** | 3.7530 | -0.344 ns |
| `ego_accel` | m/s^2 | 0.5095 | 0.5049 | -0.005 ns | 0.5051 | -0.004 **PASS** | 0.5029 | -0.007 **PASS** | 0.5126 | +0.003 ns |
| `ego_yawrate` | rad/s | 0.0285 | 0.0332 | +0.005 fail | 0.0288 | +0.000 fail | 0.0285 | +0.000 fail | 0.0326 | +0.004 fail |
| `ego_curv` | 1/m | 0.0069 | 0.0083 | +0.001 fail | 0.0070 | +0.000 fail | 0.0069 | -0.000 ns | 0.0081 | +0.001 fail |
| `n_agents_grid` | agents | 9.7982 | 10.3810 | +0.583 ns | 10.9935 | +1.195 fail | 9.7770 | -0.021 ns | 3.0878 | -6.710 **PASS** |
| `n_agents_all` | agents | 37.9358 | 39.7423 | +1.806 ns | 43.8513 | +5.915 fail | 37.9323 | -0.004 ns | 26.7374 | -11.198 **PASS** |
| `lead_present` | prob | 0.0999 | 0.3389 | +0.239 fail | 0.4263 | +0.326 fail | 0.2046 | +0.105 fail | 0.2288 | +0.129 fail |
| `nearest_any` | m | 5.4835 | 6.5377 | +1.054 fail | 6.5628 | +1.079 fail | 5.4051 | -0.079 ns | 3.0316 | -2.452 **PASS** |
| `lead_gap` | m | 5.1329 | 6.7128 | +1.580 fail | 8.5340 | +3.401 fail | 3.5712 | -1.562 **PASS** | 1.0165 | -4.116 **PASS** |
| `lead_closing` | m/s | 1.1123 | 1.1408 | +0.029 fail | 1.1054 | -0.007 ns | 1.1123 | -0.000 ns | 1.0950 | -0.017 ns |
| `lead_inv_ttc` | 1/s | 0.0802 | 0.0813 | +0.001 ns | 0.0795 | -0.001 ns | 0.0798 | -0.000 **PASS** | 0.0783 | -0.002 ns |

## 3. R^2, and the CEILING an optimally-rescaled linear readout reaches

⚠️ The fit is OVER-DISPERSED — it emits close to full variance at low correlation — so MAE can lose to a constant while `r` is positive. `r2_ceiling` = r^2 is the variance a perfectly-rescaled version of the SAME readout would explain. It is the fairest single number per rung.

| target | v6 R2 | **v6 r2_ceiling** | NULL r2_ceiling | C-V0 r2_ceiling | ORACLE r2_ceiling |
|---|---|---|---|---|---|
| `ego_v0` | -0.547 | **0.0609** | 0.0006 | 1.0000 | 0.3000 |
| `ego_accel` | +0.014 | **0.0350** | 0.0001 | 0.0125 | 0.0004 |
| `ego_yawrate` | -0.042 | **0.0036** | 0.0001 | 0.0004 | 0.0010 |
| `ego_curv` | -0.089 | **0.0001** | 0.0005 | 0.0071 | 0.0089 |
| `n_agents_grid` | -0.242 | **0.0200** | 0.0002 | 0.1038 | 0.7643 |
| `n_agents_all` | -0.341 | **0.0758** | 0.0007 | 0.1124 | 0.4356 |
| `lead_present` | -1.227 | **0.0087** | 0.0001 | 0.0231 | 0.1713 |
| `nearest_any` | -0.474 | **0.0480** | 0.0006 | 0.1667 | 0.7325 |
| `lead_gap` | -0.816 | **0.0254** | 0.0003 | 0.4672 | 0.9583 |
| `lead_closing` | -0.025 | **0.0000** | 0.0000 | 0.0003 | 0.1865 |
| `lead_inv_ttc` | -0.003 | **0.0001** | 0.0009 | 0.0013 | 0.0139 |

## 4. SEED SPREAD — between-condition vs between-seed

| target | v6 err seed-range | v6 K1 seed-range | v6 err | |v6 - NULL| err gap |
|---|---|---|---|---|
| `ego_v0` | 0.0000 | 0.0000 | 5.5256 | 0.3107 |
| `ego_accel` | 0.0000 | 0.0000 | 0.5049 | 0.0002 |
| `ego_yawrate` | 0.0000 | 0.0000 | 0.0332 | 0.0044 |
| `ego_curv` | 0.0000 | 0.0000 | 0.0083 | 0.0013 |
| `n_agents_grid` | 0.0000 | 0.0000 | 10.3810 | 0.6125 |
| `n_agents_all` | 1.1145 | 1.1145 | 39.7423 | 4.1090 |
| `lead_present` | 0.0000 | 0.0000 | 0.3389 | 0.0874 |
| `nearest_any` | 0.8300 | 0.8300 | 6.5377 | 0.0251 |
| `lead_gap` | 0.0000 | 0.0000 | 6.7128 | 1.8212 |
| `lead_closing` | 0.0000 | 0.0000 | 1.1408 | 0.0354 |
| `lead_inv_ttc` | 0.0000 | 0.0000 | 0.0813 | 0.0018 |

## 5. ⭐ THE READOUT'S OWN POSITIVE CONTROL FOR THE ANCHOR

`EGO-ORACLE` = the real cache with `cells` replaced by a DISTRIBUTED random projection of the window's own `v0`, at four noise levels (x the real cells' std). This is what the ladder's anchor row looks like when ego speed IS linearly present.

| noise | ego_v0 err (m/s) | K1 | r | r_wep | R2 |
|---|---|---|---|---|---|
| **0.10x** | 0.0460 | -4.051 **PASS** | **+1.000** | +1.000 | +1.000 |
| **1.0x** | 0.4838 | -3.613 **PASS** | **+0.994** | +0.973 | +0.988 |
| **3.0x** | 1.1477 | -2.949 **PASS** | **+0.968** | +0.871 | +0.933 |
| **10x** | 3.4711 | -0.626 ns | **+0.724** | +0.438 | +0.387 |
| *v6 @11250 (the real arm)* | 5.5256 | +1.429 fail | **+0.247** | +0.149 | -0.547 |

## 6. THE CHECKPOINT TRAJECTORY — `r` per rung

| target | @2000 | @9000 | @9250 | @10000 | @11250 |
|---|---|---|---|---|---|
| `ego_v0` | +0.263 | +0.214 | +0.222 | +0.267 | +0.247 |
| `ego_accel` | +0.141 | +0.175 | +0.188 | +0.189 | +0.187 |
| `ego_yawrate` | +0.022 | +0.054 | +0.058 | +0.066 | +0.060 |
| `ego_curv` | -0.006 | -0.004 | -0.004 | +0.001 | -0.009 |
| `n_agents_grid` | +0.174 | +0.147 | +0.136 | +0.105 | +0.141 |
| `n_agents_all` | +0.274 | +0.178 | +0.190 | +0.220 | +0.275 |
| `lead_present` | +0.065 | +0.091 | +0.065 | +0.060 | +0.093 |
| `nearest_any` | +0.274 | +0.210 | +0.179 | +0.181 | +0.219 |
| `lead_gap` | +0.137 | +0.108 | +0.060 | +0.151 | +0.159 |
| `lead_closing` | -0.003 | +0.014 | -0.008 | +0.030 | -0.006 |
| `lead_inv_ttc` | -0.011 | -0.011 | -0.010 | -0.006 | -0.008 |

## 7. ⛔ THE INTERCEPT REPAIR — pc6 penalises its own bias term

pc6's `ridge_fit` puts the appended ones-column INSIDE `alpha * np.eye(d)`, so as alpha grows the prediction collapses toward ZERO rather than toward the mean: the readout is structurally unable to fall back to the very constant K1 scores it against. `centred` mode centres `y` and leaves the intercept unpenalised. Same caches, same split, same estimator, same seeds.

| target | v6 K1 pc6 | **v6 K1 repaired** | NULL K1 repaired | ORACLE K1 repaired | v6 r repaired |
|---|---|---|---|---|---|
| `ego_v0` | +1.429 fail | **+0.427 ns** | +0.194 ns | -0.825 **PASS** | +0.322 |
| `ego_accel` | -0.005 ns | **+0.016 fail** | +0.018 fail | +0.018 fail | +0.127 |
| `ego_yawrate` | +0.005 fail | **+0.000 fail** | +0.000 fail | +0.000 ns | +0.030 |
| `ego_curv` | +0.001 fail | **+0.000 fail** | +0.000 fail | +0.000 fail | +0.002 |
| `n_agents_grid` | +0.583 ns | **+0.576 ns** | -0.228 ns | -6.916 **PASS** | +0.141 |
| `n_agents_all` | +1.806 ns | **-5.003 **PASS**** | -1.884 **PASS** | -12.720 **PASS** | +0.390 |
| `lead_present` | +0.239 fail | **+0.129 fail** | +0.115 fail | +0.073 fail | +0.109 |
| `nearest_any` | +1.054 fail | **+0.071 ns** | +0.533 fail | -2.723 **PASS** | +0.310 |
| `lead_gap` | +1.580 fail | **+0.736 fail** | +0.043 ns | -4.553 **PASS** | +0.073 |
| `lead_closing` | +0.029 fail | **+0.093 fail** | +0.093 fail | +0.064 ns | -0.036 |
| `lead_inv_ttc` | +0.001 ns | **+0.004 fail** | +0.004 fail | +0.002 ns | -0.029 |

