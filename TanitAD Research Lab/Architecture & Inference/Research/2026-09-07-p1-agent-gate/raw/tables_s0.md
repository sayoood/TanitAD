n_windows **1497** / n_episodes **35** · d (trainable params) **16,989,725** · rate slots [0, 1, 2, 3] at dt 0.5 s · windows with a lead at t0: **880**

Estimator: `paired_episode_cluster_bootstrap (taniteval.ci)`, 10,000 resamples, 95 %. NOT used: `overlapping_holdout_se -- biases the point estimate`. Tier: T1 self-action open loop (planner rolls its own plan from measured state at t0). NOT closed loop.

### Arm point estimates (mean over the scored windows)

| arm | ade_m | speed_mae_mps | along_mae_m | cross_mae_m | heading_mae | curvature_mae | yaw_rate_mae | dk_mean_headway_m | dk_n |
|---|---|---|---|---|---|---|---|---|---|
| `off` | 4.90824 | 0.52829 | 4.05527 | 1.62945 | 0.04799 | 0.00980 | 0.04759 | 36.29252 | 795 |
| `head` | 6.12579 | 0.74337 | 5.42403 | 1.55854 | 0.04743 | 0.00973 | 0.04688 | 36.01010 | 800 |
| `shuf` | 6.35562 | 0.76517 | 5.63760 | 1.69474 | 0.04803 | 0.00981 | 0.04751 | 36.06747 | 797 |
| `const` | 30.70626 | 11.84068 | 30.41345 | 1.45833 | 0.04632 | 0.00950 | 0.04501 | 45.17160 | 805 |
| `straight` | 4.29227 | 0.49762 | 3.48056 | 1.45833 | 0.04632 | 0.00950 | 0.04501 | 37.25717 | 803 |

### LONGITUDINAL + LATERAL contrasts (delta = A − B; **lower is better**)

⚠️ LATERAL is read on **curvature MAE with the straight-line floor beside it** — see the control table row `straight`.

| contrast | delta | 95 % CI | n win / ep | verdict |
|---|---|---|---|---|
| `head-off:ade_m` | +1.2175 | [+0.7274, +1.7505] | 1497 / 35 | **SEPARATED WORSE** |
| `head-off:speed_mae_mps` | +0.2151 | [+0.1332, +0.3016] | 1412 / 35 | **SEPARATED WORSE** |
| `head-off:along_mae_m` | +1.3688 | [+0.8539, +1.9150] | 1497 / 35 | **SEPARATED WORSE** |
| `head-off:cross_mae_m` | -0.0709 | [-0.1675, +0.0107] | 1497 / 35 | not separated |
| `head-off:heading_mae` | -0.0006 | [-0.0013, +0.0001] | 1412 / 35 | not separated |
| `head-off:curvature_mae` | -0.0001 | [-0.0002, +0.0000] | 1397 / 35 | not separated |
| `head-off:yaw_rate_mae` | -0.0007 | [-0.0016, +0.0001] | 1397 / 35 | not separated |
| `head-shuf:ade_m` | -0.2298 | [-0.4742, +0.0075] | 1497 / 35 | not separated |
| `head-shuf:speed_mae_mps` | -0.0218 | [-0.0693, +0.0228] | 1412 / 35 | not separated |
| `head-shuf:along_mae_m` | -0.2136 | [-0.4981, +0.0533] | 1497 / 35 | not separated |
| `head-shuf:cross_mae_m` | -0.1362 | [-0.2110, -0.0660] | 1497 / 35 | **SEPARATED BETTER** |
| `head-shuf:heading_mae` | -0.00060 | [-0.00120, -0.00002] | 1412 / 35 | **SEPARATED BETTER** |
| `head-shuf:curvature_mae` | -0.0001 | [-0.0002, +0.0000] | 1397 / 35 | not separated |
| `head-shuf:yaw_rate_mae` | -0.00062 | [-0.00122, -0.00004] | 1397 / 35 | **SEPARATED BETTER** |
| `shuf-off:ade_m` | +1.4474 | [+0.9736, +1.9516] | 1497 / 35 | **SEPARATED WORSE** |
| `shuf-off:speed_mae_mps` | +0.2369 | [+0.1465, +0.3330] | 1412 / 35 | **SEPARATED WORSE** |
| `shuf-off:along_mae_m` | +1.5823 | [+1.0681, +2.1285] | 1497 / 35 | **SEPARATED WORSE** |
| `shuf-off:cross_mae_m` | +0.0653 | [+0.0128, +0.1167] | 1497 / 35 | **SEPARATED WORSE** |
| `shuf-off:heading_mae` | +0.0000 | [-0.0006, +0.0006] | 1412 / 35 | not separated |
| `shuf-off:curvature_mae` | +0.0000 | [-0.0001, +0.0001] | 1397 / 35 | not separated |
| `shuf-off:yaw_rate_mae` | -0.0001 | [-0.0009, +0.0008] | 1397 / 35 | not separated |

### DISTANCE-KEEPING contrasts (headway / time-gap / min-TTC; **higher is safer**)

⭐ This is the family the lever is supposed to move.

| contrast | delta | 95 % CI | n win / ep | verdict |
|---|---|---|---|---|
| `head-off:headway_min_m` | -0.5015 | [-0.7712, -0.2366] | 779 / 28 | **SEPARATED WORSE** |
| `head-off:time_gap_min_s` | -0.0702 | [-0.1074, -0.0385] | 719 / 28 | **SEPARATED WORSE** |
| `head-off:min_ttc_s` | -2.1959 | [-4.1408, -0.6110] | 779 / 28 | **SEPARATED WORSE** |
| `head-shuf:headway_min_m` | -0.0438 | [-0.2456, +0.1191] | 779 / 28 | not separated |
| `head-shuf:time_gap_min_s` | -0.0022 | [-0.0299, +0.0202] | 719 / 28 | not separated |
| `head-shuf:min_ttc_s` | -0.2625 | [-0.5911, +0.0660] | 779 / 28 | not separated |
| `shuf-off:headway_min_m` | -0.4576 | [-0.7242, -0.1829] | 779 / 28 | **SEPARATED WORSE** |
| `shuf-off:time_gap_min_s` | -0.0681 | [-0.1112, -0.0297] | 719 / 28 | **SEPARATED WORSE** |
| `shuf-off:min_ttc_s` | -1.9334 | [-3.7665, -0.4244] | 779 / 28 | **SEPARATED WORSE** |

### Controls at their known values

* **constant-only (never moves)** — ADE **30.706260** vs the no-information value mean‖gt‖ **30.706261**, tol 0.0001 ⇒ **PASS**
* **straight-line floor (never steers)** — curvature MAE **0.00950** 1/m. Arms: `off` **0.00980** · `head` **0.00973** · `shuf` **0.00981**. ⛔ Ratio arm/floor: `off` **1.0×** · `head` **1.0×** · `shuf` **1.0×**
* **n vs d** — n = windows scored, d = trainable params. n << d is UNDERPOWERED BY CONSTRUCTION, not a negative result. Here n = 1497 windows / 35 episodes against d = 16,989,725: **n ≪ d**.
* **windows entering each metric** — {'ade_m': 1497, 'cross_mae_m': 1497, 'along_mae_m': 1497, 'speed_mae_mps': 1412, 'heading_mae': 1412, 'curvature_mae': 1397, 'yaw_rate_mae': 1397}
