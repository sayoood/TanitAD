# RESULT_SLOWWM (PREREG_V8F_SLOWWM, T0)

Card: PREREG_V8F_SLOWWM.md sha256 7c2775e39f9b1271342a0c588780d3da8b4d63dfb56d06de9fef4c50bc8594ca. Deciding space: **traj** ({'LONGITUDINAL': 'speed_err', 'LATERAL': 'heading_err'}). m = mean(HOLD) - mean(G-TRUE); positive = G-TRUE better; floor = |m(SLOW_rep) - m(SLOW_s0)| per read, context, family.
Estimator on every row: EC-BOOT: paired episode-cluster bootstrap, B 2000, seed 0, ratio of per-episode sums, 95 % percentile (plantest_lib.boot_*); the training-run question is answered by the floor.

**Controls: PASS.** **Outcome: CELLS-DISAGREE.** Cells: {'PLANTEST|c1': ['NO-EFFECT', 'SLOW-COSTS'], 'PLANTEST|c3': ['NO-EFFECT', 'SLOW-COSTS'], 'PLANTEST2|c1': ['NO-EFFECT'], 'PLANTEST2|c3': ['NO-EFFECT', 'SLOW-COSTS']}

**Committed by AMEND_SLOWWM_1.md sha256 5d2270d40e5ae86f0b2b552bf5e9f1cc2b244e97e16f4ccbe400e77920df1549: {'outcome': 'CELLS-DISAGREE', 'next_step_of': 'SLOW-COSTS', 'cell_labels': {'PLANTEST|c1': 'MIXED', 'PLANTEST|c3': 'MIXED', 'PLANTEST2|c1': 'NO-EFFECT', 'PLANTEST2|c3': 'MIXED'}}**

## Families not read

* TACTICAL: n = 0, NOT COMPUTABLE. the planner harness has no decision output (it returns action plans); n = 0
* STRATEGIC: n = 0, PAUSED (PI item 25). PI_DECISION_QUEUE.md ITEM 25, 2026-09-19: the strategic layer is switched off temporarily; reported as paused, never as a pass

## PLANTEST, context 1

Floor: {'LONGITUDINAL': 0.2448480058865452, 'LATERAL': 0.005275511493326224}. Decision status OK: ['NO-EFFECT', 'SLOW-COSTS']

| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |
|---|---|---|---|---|---|---|---|---|
| R35H_s0 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.6486 [-0.7507, -0.5575] | +0.6486 [+0.5575, +0.7507] | -1.5256 [-1.7227, -1.3292] | T0 |
| R35H_s0 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0672 [-0.0772, -0.0587] | +0.0672 [+0.0587, +0.0772] | -0.0822 [-0.0994, -0.0655] | T0 |
| R35H_s1 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.7489 [-0.8466, -0.6520] | +0.7489 [+0.6520, +0.8466] | -1.2870 [-1.4732, -1.1142] | T0 |
| R35H_s1 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0831 [-0.0925, -0.0745] | +0.0831 [+0.0745, +0.0925] | -0.0834 [-0.1006, -0.0675] | T0 |
| SLOW_s0 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.9384 [-1.0469, -0.8347] | +0.9384 [+0.8347, +1.0469] | -1.1238 [-1.3271, -0.9476] | T0 |
| SLOW_s0 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.1033 [-0.1134, -0.0944] | +0.1033 [+0.0944, +0.1134] | -0.1103 [-0.1279, -0.0936] | T0 |
| SLOW_s1 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.8067 [-0.9080, -0.7075] | +0.8067 [+0.7075, +0.9080] | -1.2652 [-1.4552, -1.0871] | T0 |
| SLOW_s1 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0988 [-0.1102, -0.0886] | +0.0988 [+0.0886, +0.1102] | -0.0899 [-0.1061, -0.0740] | T0 |
| PCA-B_s0 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.7761 [-0.8758, -0.6794] | +0.7761 [+0.6794, +0.8758] | -1.0600 [-1.2392, -0.8827] | T0 |
| PCA-B_s0 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.1127 [-0.1247, -0.1005] | +0.1127 [+0.1005, +0.1247] | -0.0773 [-0.0943, -0.0606] | T0 |
| PCA-B_s1 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.8197 [-0.9205, -0.7208] | +0.8197 [+0.7208, +0.9205] | -1.1819 [-1.3591, -1.0164] | T0 |
| PCA-B_s1 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.1141 [-0.1229, -0.1049] | +0.1141 [+0.1049, +0.1229] | -0.0734 [-0.0887, -0.0572] | T0 |
| SLOW_rep_s2 | LONGITUDINAL | 1976 / 141 | - | NO-GAIN | -0.6936 [-0.7811, -0.6096] | +0.6936 [+0.6096, +0.7811] | -1.2022 [-1.3885, -1.0355] | T0 |
| SLOW_rep_s2 | LATERAL | 1976 / 141 | - | NO-GAIN | -0.0980 [-0.1089, -0.0879] | +0.0980 [+0.0879, +0.1089] | -0.0917 [-0.1095, -0.0747] | T0 |

## PLANTEST, context 3

Floor: {'LONGITUDINAL': 0.07276408776067378, 'LATERAL': 0.0002632665388665745}. Decision status OK: ['NO-EFFECT', 'SLOW-COSTS']

| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |
|---|---|---|---|---|---|---|---|---|
| R35H_s0 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.7009 [-0.7972, -0.6096] | +0.7009 [+0.6096, +0.7972] | -0.9814 [-1.1721, -0.7954] | T0 |
| R35H_s0 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0691 [-0.0807, -0.0595] | +0.0691 [+0.0595, +0.0807] | -0.0252 [-0.0338, -0.0165] | T0 |
| R35H_s1 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.6490 [-0.7412, -0.5571] | +0.6490 [+0.5571, +0.7412] | -0.8744 [-1.0540, -0.7030] | T0 |
| R35H_s1 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0756 [-0.0864, -0.0659] | +0.0756 [+0.0659, +0.0864] | -0.0224 [-0.0320, -0.0131] | T0 |
| SLOW_s0 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.7220 [-0.8083, -0.6355] | +0.7220 [+0.6355, +0.8083] | -0.9104 [-1.1125, -0.7245] | T0 |
| SLOW_s0 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0860 [-0.0970, -0.0763] | +0.0860 [+0.0763, +0.0970] | -0.0177 [-0.0275, -0.0084] | T0 |
| SLOW_s1 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.7445 [-0.8397, -0.6500] | +0.7445 [+0.6500, +0.8397] | -0.8603 [-1.0360, -0.6959] | T0 |
| SLOW_s1 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0882 [-0.0989, -0.0787] | +0.0882 [+0.0787, +0.0989] | -0.0160 [-0.0261, -0.0058] | T0 |
| PCA-B_s0 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.6945 [-0.7808, -0.6039] | +0.6945 [+0.6039, +0.7808] | -0.8637 [-1.0528, -0.6815] | T0 |
| PCA-B_s0 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0983 [-0.1085, -0.0887] | +0.0983 [+0.0887, +0.1085] | -0.0165 [-0.0254, -0.0073] | T0 |
| PCA-B_s1 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.6223 [-0.7147, -0.5324] | +0.6223 [+0.5324, +0.7147] | -1.0042 [-1.1919, -0.8399] | T0 |
| PCA-B_s1 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0929 [-0.1038, -0.0831] | +0.0929 [+0.0831, +0.1038] | -0.0117 [-0.0209, -0.0020] | T0 |
| SLOW_rep_s2 | LONGITUDINAL | 1835 / 141 | - | NO-GAIN | -0.6492 [-0.7401, -0.5657] | +0.6492 [+0.5657, +0.7401] | -0.8787 [-1.0647, -0.7098] | T0 |
| SLOW_rep_s2 | LATERAL | 1835 / 141 | - | NO-GAIN | -0.0857 [-0.0977, -0.0757] | +0.0857 [+0.0757, +0.0977] | -0.0150 [-0.0242, -0.0059] | T0 |

## PLANTEST2, context 1

Floor: {'LONGITUDINAL': 0.20528317289323839, 'LATERAL': 0.007485780603585748}. Decision status OK: ['NO-EFFECT']

| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |
|---|---|---|---|---|---|---|---|---|
| R35H_s0 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.5379 [-0.6554, -0.4120] | +0.5379 [+0.4120, +0.6554] | -1.4517 [-1.7437, -1.1685] | T0 |
| R35H_s0 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0618 [-0.0737, -0.0500] | +0.0618 [+0.0500, +0.0737] | -0.0649 [-0.0879, -0.0443] | T0 |
| R35H_s1 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.5650 [-0.6845, -0.4439] | +0.5650 [+0.4439, +0.6845] | -1.2532 [-1.5076, -0.9914] | T0 |
| R35H_s1 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0740 [-0.0854, -0.0630] | +0.0740 [+0.0630, +0.0854] | -0.0735 [-0.0927, -0.0549] | T0 |
| SLOW_s0 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.7939 [-0.9395, -0.6498] | +0.7939 [+0.6498, +0.9395] | -1.0606 [-1.3237, -0.8065] | T0 |
| SLOW_s0 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0892 [-0.1010, -0.0776] | +0.0892 [+0.0776, +0.1010] | -0.0965 [-0.1210, -0.0728] | T0 |
| SLOW_s1 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.6989 [-0.8324, -0.5626] | +0.6989 [+0.5626, +0.8324] | -1.0580 [-1.3039, -0.8190] | T0 |
| SLOW_s1 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0771 [-0.0914, -0.0645] | +0.0771 [+0.0645, +0.0914] | -0.0982 [-0.1191, -0.0759] | T0 |
| PCA-B_s0 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.6249 [-0.7450, -0.5094] | +0.6249 [+0.5094, +0.7450] | -0.9924 [-1.2552, -0.7410] | T0 |
| PCA-B_s0 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0879 [-0.1029, -0.0731] | +0.0879 [+0.0731, +0.1029] | -0.0739 [-0.0958, -0.0532] | T0 |
| PCA-B_s1 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.6221 [-0.7581, -0.4895] | +0.6221 [+0.4895, +0.7581] | -1.1666 [-1.4104, -0.9368] | T0 |
| PCA-B_s1 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.1000 [-0.1115, -0.0885] | +0.1000 [+0.0885, +0.1115] | -0.0655 [-0.0853, -0.0453] | T0 |
| SLOW_rep_s2 | LONGITUDINAL | 994 / 71 | 1.0 | NO-GAIN | -0.5886 [-0.7135, -0.4760] | +0.5886 [+0.4760, +0.7135] | -1.0781 [-1.3291, -0.8221] | T0 |
| SLOW_rep_s2 | LATERAL | 994 / 71 | 1.0 | NO-GAIN | -0.0817 [-0.0963, -0.0669] | +0.0817 [+0.0669, +0.0963] | -0.0792 [-0.1030, -0.0561] | T0 |

## PLANTEST2, context 3

Floor: {'LONGITUDINAL': 0.08187882366791005, 'LATERAL': 0.00044056256329188387}. Decision status OK: ['NO-EFFECT', 'SLOW-COSTS']

| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |
|---|---|---|---|---|---|---|---|---|
| R35H_s0 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.3909 [-0.5052, -0.2755] | +0.3909 [+0.2755, +0.5052] | -1.0162 [-1.2499, -0.7962] | T0 |
| R35H_s0 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0435 [-0.0551, -0.0332] | +0.0435 [+0.0332, +0.0551] | -0.0308 [-0.0444, -0.0184] | T0 |
| R35H_s1 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.3093 [-0.4099, -0.2039] | +0.3093 [+0.2039, +0.4099] | -0.8815 [-1.1017, -0.6712] | T0 |
| R35H_s1 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0454 [-0.0549, -0.0364] | +0.0454 [+0.0364, +0.0549] | -0.0317 [-0.0442, -0.0214] | T0 |
| SLOW_s0 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.4603 [-0.5723, -0.3490] | +0.4603 [+0.3490, +0.5723] | -0.9599 [-1.2101, -0.7158] | T0 |
| SLOW_s0 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0510 [-0.0609, -0.0421] | +0.0510 [+0.0421, +0.0609] | -0.0279 [-0.0391, -0.0163] | T0 |
| SLOW_s1 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.4940 [-0.6248, -0.3707] | +0.4940 [+0.3707, +0.6248] | -0.8846 [-1.1097, -0.6664] | T0 |
| SLOW_s1 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0495 [-0.0581, -0.0411] | +0.0495 [+0.0411, +0.0581] | -0.0302 [-0.0421, -0.0185] | T0 |
| PCA-B_s0 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.4467 [-0.5545, -0.3384] | +0.4467 [+0.3384, +0.5545] | -0.7848 [-1.0192, -0.5700] | T0 |
| PCA-B_s0 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0637 [-0.0738, -0.0541] | +0.0637 [+0.0541, +0.0738] | -0.0214 [-0.0321, -0.0116] | T0 |
| PCA-B_s1 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.3830 [-0.4813, -0.2874] | +0.3830 [+0.2874, +0.4813] | -0.8686 [-1.0786, -0.6789] | T0 |
| PCA-B_s1 | LATERAL | 923 / 71 | 1.0 | GOAL-BLIND | -0.0652 [-0.0766, -0.0548] | +0.0652 [+0.0548, +0.0766] | -0.0106 [-0.0256, +0.0032] | T0 |
| SLOW_rep_s2 | LONGITUDINAL | 923 / 71 | 1.0 | NO-GAIN | -0.3785 [-0.4878, -0.2665] | +0.3785 [+0.2665, +0.4878] | -0.8575 [-1.0899, -0.6281] | T0 |
| SLOW_rep_s2 | LATERAL | 923 / 71 | 1.0 | NO-GAIN | -0.0514 [-0.0604, -0.0426] | +0.0514 [+0.0426, +0.0604] | -0.0362 [-0.0495, -0.0239] | T0 |

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
