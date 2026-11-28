n_windows **1497** / n_episodes **35** · d (trainable params) **16,989,725** · rate slots [0, 1, 2, 3] at dt 0.5 s · windows with a lead at t0: **880**

Estimator: `paired_episode_cluster_bootstrap (taniteval.ci)`, 10,000 resamples, 95 %. NOT used: `overlapping_holdout_se -- biases the point estimate`. Tier: T1 self-action open loop (planner rolls its own plan from measured state at t0). NOT closed loop.

### Arm point estimates (mean over the scored windows)

| arm | ade_m | speed_mae_mps | along_mae_m | cross_mae_m | heading_mae | curvature_mae | yaw_rate_mae | dk_mean_headway_m | dk_n |
|---|---|---|---|---|---|---|---|---|---|
| `off` | 5.78334 | 0.61394 | 4.47170 | 2.34531 | 0.05261 | 0.01003 | 0.05260 | 36.51321 | 800 |
| `head` | 5.29750 | 0.69214 | 4.38465 | 1.74074 | 0.04893 | 0.00992 | 0.04869 | 36.41183 | 801 |
| `shuf` | 5.91008 | 0.66231 | 5.07515 | 1.79920 | 0.04875 | 0.00980 | 0.04805 | 36.31833 | 797 |
| `const` | 30.70626 | 11.84068 | 30.41345 | 1.45833 | 0.04632 | 0.00950 | 0.04501 | 45.17160 | 805 |
| `straight` | 4.29227 | 0.49762 | 3.48056 | 1.45833 | 0.04632 | 0.00950 | 0.04501 | 37.25717 | 803 |

### LONGITUDINAL + LATERAL contrasts (delta = A − B; **lower is better**)

⚠️ LATERAL is read on **curvature MAE with the straight-line floor beside it** — see the control table row `straight`.

| contrast | delta | 95 % CI | n win / ep | verdict |
|---|---|---|---|---|
| `head-off:ade_m` | -0.4858 | [-0.9161, -0.1088] | 1497 / 35 | **SEPARATED BETTER** |
| `head-off:speed_mae_mps` | +0.0782 | [+0.0088, +0.1544] | 1412 / 35 | **SEPARATED WORSE** |
| `head-off:along_mae_m` | -0.0870 | [-0.3886, +0.1859] | 1497 / 35 | not separated |
| `head-off:cross_mae_m` | -0.6046 | [-1.1527, -0.1578] | 1497 / 35 | **SEPARATED BETTER** |
| `head-off:heading_mae` | -0.0037 | [-0.0071, -0.0009] | 1412 / 35 | **SEPARATED BETTER** |
| `head-off:curvature_mae` | -0.0001 | [-0.0003, +0.0001] | 1397 / 35 | not separated |
| `head-off:yaw_rate_mae` | -0.0039 | [-0.0078, -0.0007] | 1397 / 35 | **SEPARATED BETTER** |
| `head-shuf:ade_m` | -0.6126 | [-0.9361, -0.3117] | 1497 / 35 | **SEPARATED BETTER** |
| `head-shuf:speed_mae_mps` | +0.0298 | [-0.0307, +0.0909] | 1412 / 35 | not separated |
| `head-shuf:along_mae_m` | -0.6905 | [-0.9817, -0.4155] | 1497 / 35 | **SEPARATED BETTER** |
| `head-shuf:cross_mae_m` | -0.0585 | [-0.4066, +0.2460] | 1497 / 35 | not separated |
| `head-shuf:heading_mae` | +0.0002 | [-0.0016, +0.0018] | 1412 / 35 | not separated |
| `head-shuf:curvature_mae` | +0.000118 | [+0.000002, +0.000246] | 1397 / 35 | **SEPARATED WORSE** |
| `head-shuf:yaw_rate_mae` | +0.0006 | [-0.0014, +0.0025] | 1397 / 35 | not separated |
| `shuf-off:ade_m` | +0.1267 | [-0.3320, +0.5696] | 1497 / 35 | not separated |
| `shuf-off:speed_mae_mps` | +0.0484 | [-0.0052, +0.1015] | 1412 / 35 | not separated |
| `shuf-off:along_mae_m` | +0.6035 | [+0.1703, +1.0109] | 1497 / 35 | **SEPARATED WORSE** |
| `shuf-off:cross_mae_m` | -0.5461 | [-1.0457, -0.1397] | 1497 / 35 | **SEPARATED BETTER** |
| `shuf-off:heading_mae` | -0.0039 | [-0.0071, -0.0012] | 1412 / 35 | **SEPARATED BETTER** |
| `shuf-off:curvature_mae` | -0.0002 | [-0.0004, -0.0001] | 1397 / 35 | **SEPARATED BETTER** |
| `shuf-off:yaw_rate_mae` | -0.0046 | [-0.0082, -0.0015] | 1397 / 35 | **SEPARATED BETTER** |

### DISTANCE-KEEPING contrasts (headway / time-gap / min-TTC; **higher is safer**)

⭐ This is the family the lever is supposed to move.

| contrast | delta | 95 % CI | n win / ep | verdict |
|---|---|---|---|---|
| `head-off:headway_min_m` | -0.4535 | [-0.7915, -0.1639] | 776 / 28 | **SEPARATED WORSE** |
| `head-off:time_gap_min_s` | -0.0351 | [-0.0694, -0.0037] | 716 / 28 | **SEPARATED WORSE** |
| `head-off:min_ttc_s` | -1.2529 | [-2.2522, -0.3694] | 776 / 28 | **SEPARATED WORSE** |
| `head-shuf:headway_min_m` | -0.1486 | [-0.4074, +0.0856] | 776 / 28 | not separated |
| `head-shuf:time_gap_min_s` | -0.0084 | [-0.0395, +0.0191] | 716 / 28 | not separated |
| `head-shuf:min_ttc_s` | -0.2861 | [-0.9028, +0.2332] | 776 / 28 | not separated |
| `shuf-off:headway_min_m` | -0.3049 | [-0.4896, -0.1389] | 776 / 28 | **SEPARATED WORSE** |
| `shuf-off:time_gap_min_s` | -0.0267 | [-0.0567, +0.0049] | 716 / 28 | not separated |
| `shuf-off:min_ttc_s` | -0.9668 | [-1.4928, -0.4709] | 776 / 28 | **SEPARATED WORSE** |

### Controls at their known values

* **constant-only (never moves)** — ADE **30.706260** vs the no-information value mean‖gt‖ **30.706261**, tol 0.0001 ⇒ **PASS**
* **straight-line floor (never steers)** — curvature MAE **0.00950** 1/m. Arms: `off` **0.01003** · `head` **0.00992** · `shuf` **0.00980**. ⛔ Ratio arm/floor: `off` **1.1×** · `head` **1.0×** · `shuf` **1.0×**
* **n vs d** — n = windows scored, d = trainable params. n << d is UNDERPOWERED BY CONSTRUCTION, not a negative result. Here n = 1497 windows / 35 episodes against d = 16,989,725: **n ≪ d**.
* **windows entering each metric** — {'ade_m': 1497, 'cross_mae_m': 1497, 'along_mae_m': 1497, 'speed_mae_mps': 1412, 'heading_mae': 1412, 'curvature_mae': 1397, 'yaw_rate_mae': 1397}
