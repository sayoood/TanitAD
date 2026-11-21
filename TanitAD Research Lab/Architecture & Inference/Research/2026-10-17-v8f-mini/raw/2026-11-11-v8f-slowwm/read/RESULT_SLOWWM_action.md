# RESULT_SLOWWM (PREREG_V8F_SLOWWM, T0)

Card: PREREG_V8F_SLOWWM.md sha256 7c2775e39f9b1271342a0c588780d3da8b4d63dfb56d06de9fef4c50bc8594ca. Deciding space: **action** ({'LONGITUDINAL': 'acc_mae', 'LATERAL': 'yaw_mae'}). m = mean(HOLD) - mean(G-TRUE); positive = G-TRUE better; floor = |m(SLOW_rep) - m(SLOW_s0)| per read, context, family.
Estimator on every row: EC-BOOT: paired episode-cluster bootstrap, B 2000, seed 0, ratio of per-episode sums, 95 % percentile (plantest_lib.boot_*); the training-run question is answered by the floor.

**Controls: PASS.** **Outcome: CELLS-DISAGREE.** Cells: {'PLANTEST|c1': ['NO-EFFECT'], 'PLANTEST|c3': ['NO-EFFECT', 'SLOW-COSTS'], 'PLANTEST2|c1': ['NO-EFFECT'], 'PLANTEST2|c3': ['NO-EFFECT', 'SLOW-COSTS']}

**Committed by AMEND_SLOWWM_1.md sha256 5d2270d40e5ae86f0b2b552bf5e9f1cc2b244e97e16f4ccbe400e77920df1549: {'outcome': 'CELLS-DISAGREE', 'next_step_of': 'SLOW-COSTS', 'cell_labels': {'PLANTEST|c1': 'NO-EFFECT', 'PLANTEST|c3': 'MIXED', 'PLANTEST2|c1': 'NO-EFFECT', 'PLANTEST2|c3': 'MIXED'}}**

## Families not read

* TACTICAL: n = 0, NOT COMPUTABLE. the planner harness has no decision output (it returns action plans); n = 0
* STRATEGIC: n = 0, PAUSED (PI item 25). PI_DECISION_QUEUE.md ITEM 25, 2026-09-19: the strategic layer is switched off temporarily; reported as paused, never as a pass

## PLANTEST, context 1

Floor: {'LONGITUDINAL': 0.18999905708479792, 'LATERAL': 0.023742695434692124}. Decision status OK: ['NO-EFFECT']

| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |
|---|---|---|---|---|---|---|---|---|
| R35H_s0 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.8341 [-0.8803, -0.7868] | +0.8341 [+0.7868, +0.8803] | -0.4697 [-0.5322, -0.4089] | T0 |
| R35H_s0 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0590 [-0.0641, -0.0543] | +0.0590 [+0.0543, +0.0641] | -0.0406 [-0.0482, -0.0333] | T0 |
| R35H_s1 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.8457 [-0.8941, -0.7942] | +0.8457 [+0.7942, +0.8941] | -0.3647 [-0.4308, -0.2994] | T0 |
| R35H_s1 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0752 [-0.0803, -0.0704] | +0.0752 [+0.0704, +0.0803] | -0.0421 [-0.0490, -0.0349] | T0 |
| SLOW_s0 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.9445 [-0.9897, -0.9012] | +0.9445 [+0.9012, +0.9897] | -0.3381 [-0.4022, -0.2769] | T0 |
| SLOW_s0 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.1102 [-0.1157, -0.1047] | +0.1102 [+0.1047, +0.1157] | -0.0426 [-0.0491, -0.0364] | T0 |
| SLOW_s1 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.7424 [-0.7863, -0.6999] | +0.7424 [+0.6999, +0.7863] | -0.4251 [-0.4906, -0.3627] | T0 |
| SLOW_s1 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0858 [-0.0910, -0.0808] | +0.0858 [+0.0808, +0.0910] | -0.0566 [-0.0632, -0.0492] | T0 |
| PCA-B_s0 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.8992 [-0.9457, -0.8568] | +0.8992 [+0.8568, +0.9457] | -0.2829 [-0.3461, -0.2186] | T0 |
| PCA-B_s0 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0938 [-0.0992, -0.0884] | +0.0938 [+0.0884, +0.0992] | -0.0387 [-0.0468, -0.0305] | T0 |
| PCA-B_s1 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.8362 [-0.8824, -0.7865] | +0.8362 [+0.7865, +0.8824] | -0.3479 [-0.4089, -0.2877] | T0 |
| PCA-B_s1 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.1005 [-0.1049, -0.0958] | +0.1005 [+0.0958, +0.1049] | -0.0315 [-0.0383, -0.0244] | T0 |
| SLOW_rep_s2 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.7545 [-0.7993, -0.7107] | +0.7545 [+0.7107, +0.7993] | -0.3707 [-0.4386, -0.3052] | T0 |
| SLOW_rep_s2 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0865 [-0.0912, -0.0815] | +0.0865 [+0.0815, +0.0912] | -0.0479 [-0.0557, -0.0402] | T0 |

## PLANTEST, context 3

Floor: {'LONGITUDINAL': 0.08160225666227705, 'LATERAL': 0.006636860525873825}. Decision status OK: ['NO-EFFECT', 'SLOW-COSTS']

| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |
|---|---|---|---|---|---|---|---|---|
| R35H_s0 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.8217 [-0.8639, -0.7809] | +0.8217 [+0.7809, +0.8639] | -0.3807 [-0.4319, -0.3304] | T0 |
| R35H_s0 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0700 [-0.0739, -0.0666] | +0.0700 [+0.0666, +0.0739] | -0.0327 [-0.0368, -0.0286] | T0 |
| R35H_s1 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.7896 [-0.8305, -0.7510] | +0.7896 [+0.7510, +0.8305] | -0.3715 [-0.4237, -0.3237] | T0 |
| R35H_s1 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0819 [-0.0855, -0.0784] | +0.0819 [+0.0784, +0.0855] | -0.0364 [-0.0408, -0.0321] | T0 |
| SLOW_s0 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.9244 [-0.9648, -0.8846] | +0.9244 [+0.8846, +0.9648] | -0.3025 [-0.3558, -0.2516] | T0 |
| SLOW_s0 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0975 [-0.1009, -0.0943] | +0.0975 [+0.0943, +0.1009] | -0.0279 [-0.0321, -0.0239] | T0 |
| SLOW_s1 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.8454 [-0.8840, -0.8067] | +0.8454 [+0.8067, +0.8840] | -0.3097 [-0.3637, -0.2603] | T0 |
| SLOW_s1 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.1031 [-0.1064, -0.0997] | +0.1031 [+0.0997, +0.1064] | -0.0356 [-0.0401, -0.0310] | T0 |
| PCA-B_s0 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.9039 [-0.9457, -0.8629] | +0.9039 [+0.8629, +0.9457] | -0.2765 [-0.3312, -0.2267] | T0 |
| PCA-B_s0 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0943 [-0.0975, -0.0913] | +0.0943 [+0.0913, +0.0975] | -0.0275 [-0.0314, -0.0236] | T0 |
| PCA-B_s1 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.8669 [-0.9105, -0.8241] | +0.8669 [+0.8241, +0.9105] | -0.3126 [-0.3693, -0.2598] | T0 |
| PCA-B_s1 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.1053 [-0.1084, -0.1025] | +0.1053 [+0.1025, +0.1084] | -0.0199 [-0.0230, -0.0165] | T0 |
| SLOW_rep_s2 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.8428 [-0.8834, -0.8026] | +0.8428 [+0.8026, +0.8834] | -0.3044 [-0.3590, -0.2543] | T0 |
| SLOW_rep_s2 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0908 [-0.0944, -0.0878] | +0.0908 [+0.0878, +0.0944] | -0.0301 [-0.0338, -0.0263] | T0 |

## PLANTEST2, context 1

Floor: {'LONGITUDINAL': 0.16414029102806138, 'LATERAL': 0.022146189819741383}. Decision status OK: ['NO-EFFECT']

| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |
|---|---|---|---|---|---|---|---|---|
| R35H_s0 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.5856 [-0.6496, -0.5243] | +0.5856 [+0.5243, +0.6496] | -0.6197 [-0.7147, -0.5334] | T0 |
| R35H_s0 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0493 [-0.0553, -0.0434] | +0.0493 [+0.0434, +0.0553] | -0.0383 [-0.0482, -0.0292] | T0 |
| R35H_s1 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.6206 [-0.6833, -0.5563] | +0.6206 [+0.5563, +0.6833] | -0.4516 [-0.5457, -0.3557] | T0 |
| R35H_s1 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0581 [-0.0645, -0.0517] | +0.0581 [+0.0517, +0.0645] | -0.0493 [-0.0587, -0.0400] | T0 |
| SLOW_s0 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.7313 [-0.8042, -0.6623] | +0.7313 [+0.6623, +0.8042] | -0.4181 [-0.5007, -0.3401] | T0 |
| SLOW_s0 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0860 [-0.0933, -0.0790] | +0.0860 [+0.0790, +0.0933] | -0.0490 [-0.0593, -0.0379] | T0 |
| SLOW_s1 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.5817 [-0.6424, -0.5248] | +0.5817 [+0.5248, +0.6424] | -0.4381 [-0.5215, -0.3567] | T0 |
| SLOW_s1 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0638 [-0.0704, -0.0571] | +0.0638 [+0.0571, +0.0704] | -0.0639 [-0.0740, -0.0537] | T0 |
| PCA-B_s0 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.7018 [-0.7617, -0.6393] | +0.7018 [+0.6393, +0.7617] | -0.3559 [-0.4476, -0.2670] | T0 |
| PCA-B_s0 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0713 [-0.0784, -0.0644] | +0.0713 [+0.0644, +0.0784] | -0.0435 [-0.0533, -0.0336] | T0 |
| PCA-B_s1 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.6784 [-0.7457, -0.6154] | +0.6784 [+0.6154, +0.7457] | -0.3964 [-0.4779, -0.3132] | T0 |
| PCA-B_s1 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0815 [-0.0878, -0.0752] | +0.0815 [+0.0752, +0.0878] | -0.0397 [-0.0498, -0.0292] | T0 |
| SLOW_rep_s2 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.5672 [-0.6195, -0.5131] | +0.5672 [+0.5131, +0.6195] | -0.4017 [-0.4821, -0.3168] | T0 |
| SLOW_rep_s2 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0639 [-0.0704, -0.0573] | +0.0639 [+0.0573, +0.0704] | -0.0525 [-0.0639, -0.0415] | T0 |

## PLANTEST2, context 3

Floor: {'LONGITUDINAL': 0.028905142555203267, 'LATERAL': 0.006135971789741679}. Decision status OK: ['NO-EFFECT', 'SLOW-COSTS']

| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |
|---|---|---|---|---|---|---|---|---|
| R35H_s0 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.5066 [-0.5557, -0.4557] | +0.5066 [+0.4557, +0.5557] | -0.5376 [-0.6102, -0.4694] | T0 |
| R35H_s0 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0518 [-0.0559, -0.0479] | +0.0518 [+0.0479, +0.0559] | -0.0418 [-0.0472, -0.0364] | T0 |
| R35H_s1 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.4945 [-0.5394, -0.4477] | +0.4945 [+0.4477, +0.5394] | -0.5153 [-0.5823, -0.4493] | T0 |
| R35H_s1 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0589 [-0.0628, -0.0552] | +0.0589 [+0.0552, +0.0628] | -0.0458 [-0.0523, -0.0396] | T0 |
| SLOW_s0 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.5847 [-0.6322, -0.5398] | +0.5847 [+0.5398, +0.6322] | -0.4750 [-0.5409, -0.4114] | T0 |
| SLOW_s0 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0714 [-0.0753, -0.0674] | +0.0714 [+0.0674, +0.0753] | -0.0405 [-0.0465, -0.0341] | T0 |
| SLOW_s1 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.5627 [-0.6098, -0.5146] | +0.5627 [+0.5146, +0.6098] | -0.4222 [-0.4921, -0.3541] | T0 |
| SLOW_s1 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0717 [-0.0756, -0.0675] | +0.0717 [+0.0675, +0.0756] | -0.0517 [-0.0580, -0.0450] | T0 |
| PCA-B_s0 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.6179 [-0.6710, -0.5642] | +0.6179 [+0.5642, +0.6710] | -0.3920 [-0.4641, -0.3217] | T0 |
| PCA-B_s0 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0686 [-0.0726, -0.0645] | +0.0686 [+0.0645, +0.0726] | -0.0361 [-0.0412, -0.0312] | T0 |
| PCA-B_s1 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.5815 [-0.6331, -0.5286] | +0.5815 [+0.5286, +0.6331] | -0.4215 [-0.4919, -0.3516] | T0 |
| PCA-B_s1 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0790 [-0.0835, -0.0747] | +0.0790 [+0.0747, +0.0835] | -0.0315 [-0.0372, -0.0257] | T0 |
| SLOW_rep_s2 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.5558 [-0.6071, -0.5045] | +0.5558 [+0.5045, +0.6071] | -0.4080 [-0.4677, -0.3448] | T0 |
| SLOW_rep_s2 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0652 [-0.0688, -0.0613] | +0.0652 [+0.0613, +0.0688] | -0.0411 [-0.0464, -0.0354] | T0 |

## IA-0 skill per level (beside, no bar)

```
{
 "band_end_summary": {
  "PCA-B_s0": {
   "basis_kind": "pca",
   "tier": "T0",
   "band_end": {
    "1": {
     "horizon_s": 2.0,
     "readable": true,
     "skill": 0.1842048214066579,
     "lo": 0.14921387499875344,
     "hi": 0.2168553211355065,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    },
    "2": {
     "horizon_s": 6.0,
     "readable": true,
     "skill": 0.4657897540471825,
     "lo": 0.41718264505465,
     "hi": 0.5060111379054442,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    },
    "3": {
     "horizon_s": 12.0,
     "readable": true,
     "skill": 0.5951422932412673,
     "lo": 0.525777289457772,
     "hi": 0.650683459535009,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    }
   },
   "band_end_context1": {
    "1": {
     "horizon_s": 2.0,
     "readable": true,
     "skill": -0.0716790403078933,
     "lo": -0.13197139356394474,
     "hi": -0.0173442387392568,
     "above_copy_last": false,
     "below_copy_last": true,
     "healthy": true
    },
    "2": {
     "horizon_s": 6.0,
     "readable": true,
     "skill": 0.3503818777999488,
     "lo": 0.28812557764398455,
     "hi": 0.40359378243299543,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    },
    "3": {
     "horizon_s": 12.0,
     "readable": true,
     "skill": 0.5960896581401546,
     "lo": 0.5325630445162645,
     "hi": 0.6479145047510539,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    }
   }
  },
  "PCA-B_s1": {
   "basis_kind": "pca",
   "tier": "T0",
   "band_end": {
    "1": {
     "horizon_s": 2.0,
     "readable": true,
     "skill": 0.18489368925646946,
     "lo": 0.1476139575528572,
     "hi": 0.22060936019416322,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    },
    "2": {
     "horizon_s": 6.0,
     "readable": true,
     "skill": 0.449618735715831,
     "lo": 0.397880392886463,
     "hi": 0.4929833066001906,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    },
    "3": {
     "horizon_s": 12.0,
     "readable": true,
     "skill": 0.5401717463448605,
     "lo": 0.4595796389600883,
     "hi": 0.6078945695125807,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    }
   },
   "band_end_context1": {
    "1": {
     "horizon_s": 2.0,
     "readable": true,
     "skill": -0.08994881250444989,
     "lo": -0.154135772373791,
     "hi": -0.03183098390638554,
     "above_copy_last": false,
     "below_copy_last": true,
     "healthy": true
    },
    "2": {
     "horizon_s": 6.0,
     "readable": true,
     "skill": 0.34396967562407194,
     "lo": 0.28056559376923396,
     "hi": 0.40034122873595657,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    },
    "3": {
     "horizon_s": 12.0,
     "readable": true,
     "skill": 0.5334572783254782,
     "lo": 0.45609254210885836,
     "hi": 0.5951311302107445,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    }
   }
  },
  "R35H_s0": {
   "basis_kind": "adapter",
   "tier": "T0",
   "band_end": {
    "1": {
     "horizon_s": 2.0,
     "readable": true,
     "skill": 0.2593902445792321,
     "lo": 0.19842011276738736,
     "hi": 0.3157644550419052,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    },
    "2": {
     "horizon_s": 6.0,
     "readable": true,
     "skill": 0.5457710131054107,
     "lo": 0.4926824425671234,
     "hi": 0.5879264348756112,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    },
    "3": {
     "horizon_s": 12.0,
     "readable": true,
     "skill": 0.6721048810701261,
     "lo": 0.6175932876721663,
     "hi": 0.7128162528488051,
     "above_copy_last": true,
     "below_copy_last": false,
     "healthy": true
    }
   },
   "band_end_conte
```
