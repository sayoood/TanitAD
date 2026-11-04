# E1 replicate floor -- C0 vs C1 (same flags, training seed 0 vs 1), DEV

Tier: **T0 (label level)** (perception only). Estimator: **paired_episode_cluster_bootstrap_stat**, B = 2000, seed 0 -- it answers *another draw of EPISODES*, not another training run.
Windows: 2308 DEV windows / 70 episodes (clips), identical for both arms (paired alignment drops none). Fixed P=R gate for the census rows: 0.26599645614624023 (--gates-json).

## The keep rule (REFCV9_DESIGN 4.3)

Keep a lever iff its primary is separated in the good direction AND |delta| > 2 x |C1 - C0| on that metric AND no other perception primary is separated worse beyond 2 x its floor (REFCV9_DESIGN 4.3; taniteval.ci.e1_floor_read).

`floor` = abs(C1 - C0) is ONE seed pair (n = 1): a single draw of the training noise, not its spread. A floor of exactly 0 is **degenerate** (two runs agreeing to the last digit), and the keep decision then rests on separation alone (`taniteval.ci.e1_floor_read`; `r9_e1_floor.lever_read` calls it).

## The four metric families (CLAUDE.md, binding)

| family | status |
|---|---|
| LONGITUDINAL | not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0 |
| LATERAL | not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0 |
| TACTICAL | not computable: the E1 rig builds no planner (perception-only, T0 label level), n = 0 |
| STRATEGIC | PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive well; reported as paused, not as a pass |

Perception is the family this rig reads: boxes and the near-field map raster, below.

## Perception -- boxes (T0)

### AP@1 m, all and by range (head output, no NMS)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `ap1m_0_20` | 0.37801 | 0.37367 | -0.004333 | [-0.0254, 0.0169] | no | 0.004333 | 0.008666 | 2308 windows / 70 episodes |
| `ap1m_20_40` | 0.11970 | 0.11742 | -0.002282 | [-0.0113, 0.0068] | no | 0.002282 | 0.004564 | 2308 windows / 70 episodes |
| `ap1m_40_60` | 0.02922 | 0.02995 | +0.000728 | [-0.0041, 0.0053] | no | 0.000728 | 0.001457 | 2308 windows / 70 episodes |
| `ap1m_all` | 0.20993 | 0.20756 | -0.002367 | [-0.0124, 0.0075] | no | 0.002367 | 0.004734 | 2308 windows / 70 episodes |

### AP@2 m, all and by range (head output, no NMS)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `ap2m_0_20` | 0.53568 | 0.53785 | +0.002168 | [-0.009, 0.0118] | no | 0.002168 | 0.004337 | 2308 windows / 70 episodes |
| `ap2m_20_40` | 0.23472 | 0.23056 | -0.004160 | [-0.0157, 0.0085] | no | 0.004160 | 0.008319 | 2308 windows / 70 episodes |
| `ap2m_40_60` | 0.08974 | 0.09173 | +0.001993 | [-0.0078, 0.0105] | no | 0.001993 | 0.003985 | 2308 windows / 70 episodes |
| `ap2m_all` | 0.32988 | 0.33072 | +0.000845 | [-0.0065, 0.007] | no | 0.000845 | 0.001689 | 2308 windows / 70 episodes |

### P=R census at the FIXED gate (same gate for both arms)

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `boxes_per_object_noNMS` | 2.07618 | 2.07369 | -0.002492 | [-0.055, 0.0575] | no | 0.002492 | 0.004983 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_0_20` | 2.21815 | 2.18740 | -0.030750 | [-0.0838, 0.0303] | no | 0.030750 | 0.061501 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_20_40` | 1.73574 | 1.80851 | +0.072767 | [-0.0194, 0.1677] | no | 0.072767 | 0.145533 | 2308 windows / 70 episodes |
| `boxes_per_object_noNMS_40_60` | 1.15152 | 1.33333 | +0.181818 | [-0.0742, 0.3979] | no | 0.181818 | 0.363636 | 2308 windows / 70 episodes |
| `frac_objects_ge2_noNMS` | 0.67113 | 0.66357 | -0.007561 | [-0.0306, 0.0149] | no | 0.007561 | 0.015121 | 2308 windows / 70 episodes |
| `prec_gate_noNMS` | 0.36903 | 0.36876 | -0.000270 | [-0.0122, 0.0111] | no | 0.000270 | 0.000539 | 2308 windows / 70 episodes |
| `rec_gate_noNMS` | 0.35897 | 0.35816 | -0.000810 | [-0.0097, 0.0081] | no | 0.000810 | 0.001619 | 2308 windows / 70 episodes |

### localisation / class of the 2 m greedy TPs

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `dx_0_40_mean` | 0.78979 | 0.78394 | -0.005850 | [-0.0223, 0.0119] | no | 0.005850 | 0.011700 | 2308 windows / 70 episodes |
| `dx_0_40_median` | 0.71954 | 0.71737 | -0.002173 | [-0.0238, 0.0266] | no | 0.002173 | 0.004347 | 2308 windows / 70 episodes |
| `tp_class_acc` | 0.83570 | 0.83670 | +0.001000 | [-0.01, 0.0123] | no | 0.001000 | 0.002000 | 2308 windows / 70 episodes |

### other

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2308 windows / 70 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `n_det_noNMS_0_20` | 1939.00000 | 1937.00000 | -2.000000 | [-42, 41] | no | 2.000000 | 4.000000 | 2308 windows / 70 episodes |
| `n_det_noNMS_20_40` | 719.00000 | 705.00000 | -14.000000 | [-56, 27] | no | 14.000000 | 28.000000 | 2308 windows / 70 episodes |
| `n_det_noNMS_40_60` | 33.00000 | 45.00000 | +12.000000 | [-7, 36] | no | 12.000000 | 24.000000 | 2308 windows / 70 episodes |
| `n_pos` | 7410.00000 | 7410.00000 | +0.000000 | [0, 0] | no | 0.000000 (DEGENERATE: not a floor) | 0.000000 | 2308 windows / 70 episodes |

## Perception -- map raster, corridor |y| < 3 m (T0)

### corridor sidewalk precision, within 0.5 m of a GT sidewalk cell

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_prec05_0_5` | 0.30828 | 0.28784 | -0.020440 | [-0.0359, -0.0057] | yes | 0.020440 | 0.040880 | 2329 raster files / 68 episodes |
| `nf_prec05_10_20` | 0.83318 | 0.83109 | -0.002083 | [-0.0155, 0.0128] | no | 0.002083 | 0.004167 | 2329 raster files / 68 episodes |
| `nf_prec05_20_40` | 0.83192 | 0.82663 | -0.005282 | [-0.0194, 0.0071] | no | 0.005282 | 0.010565 | 2329 raster files / 68 episodes |
| `nf_prec05_5_10` | 0.84763 | 0.82331 | -0.024320 | [-0.0373, -0.012] | yes | 0.024320 | 0.048639 | 2329 raster files / 68 episodes |

### corridor sidewalk precision, exact GT sidewalk cell

Tier T0 (label level) | estimator paired_episode_cluster_bootstrap_stat B=2000 seed=0 | n = 2329 raster files / 68 episodes

| metric | C0 | C1 | delta = C1 - C0 | 95 % CI of delta | separated? | floor = abs(C1 - C0) | keep threshold = 2 x floor | n |
|---|---|---|---|---|---|---|---|---|
| `nf_prec_0_5` | 0.22106 | 0.19992 | -0.021138 | [-0.0327, -0.0101] | yes | 0.021138 | 0.042275 | 2329 raster files / 68 episodes |
| `nf_prec_10_20` | 0.73410 | 0.72869 | -0.005405 | [-0.0204, 0.0093] | no | 0.005405 | 0.010811 | 2329 raster files / 68 episodes |
| `nf_prec_20_40` | 0.75323 | 0.74477 | -0.008455 | [-0.0235, 0.0046] | no | 0.008455 | 0.016910 | 2329 raster files / 68 episodes |
| `nf_prec_5_10` | 0.74724 | 0.71706 | -0.030175 | [-0.0437, -0.0183] | yes | 0.030175 | 0.060349 | 2329 raster files / 68 episodes |

## Reading notes

* 4 of 29 defined metrics have C1 vs C0 separated by the episode bootstrap. A separated C1 - C0 is NOT a failure of the rig: it is the proof (H-ESTIM-SEED-1) that a separated interval alone cannot certify a lever -- which is why the keep rule adds the 2 x floor clause.
* 0 defined metrics have floor exactly 0 (degenerate).
* Map rows are what `r9_e1_eval.py` emits: sidewalk-flag (`psw`) precision on GT-seen cells of the corridor; the lift-valid subset and a sidewalk+edge union are NOT computed here (not emitted by the evaluator).
* The census rows (`prec_gate_*`, `rec_gate_*`, `boxes_per_object_*`, `frac_objects_ge2_*`) are read at the fixed gate in the header for both arms, not at each arm's own P=R point.
* Inputs: eval_c0 sha256 cf6fba26f7d5; eval_c1 sha256 50c3f4dfe530; paired sha256 291dd691eea6.
