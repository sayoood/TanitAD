# H-HJ-COLLAPSE-1 numbers (generated from readout.json; n_episodes=40, ckpt step 21109, strict load {'missing_keys': [], 'unexpected_keys': []})

Estimator: episode-cluster bootstrap B=2000 seed 0; answers 'another draw of EPISODES' only (single checkpoint, deterministic forward)

## PREREG outcome (operative predictor, k = 10 = 2.0 s)

* steer-sens = 0.0043 [0.0026, 0.0063]
* v-sens (speed-channel swap) = 0.0590 [0.0474, 0.0716]; v/steer = 13.79
* n = 80 windows / 40 episodes
* **point-estimate outcome: O-COLLAPSE**; steer CI straddles thresholds: none

## sens_k(c), Operative (k = 1/5/10 steps = 0.2/1.0/2.0 s) -- mean of per-window ratios, episode-cluster 95 % CI

| swap | k=1 | k=5 | k=10 | median k=10 | ratio-of-means k=10 |
|---|---|---|---|---|---|
| self | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| all | 0.0476 [0.0363, 0.0602] | 0.0663 [0.0542, 0.0804] | 0.0742 [0.0625, 0.0869] | 0.0687 | 0.0735 |
| a | 0.0202 [0.0126, 0.0319] | 0.0305 [0.0221, 0.0403] | 0.0384 [0.0306, 0.0474] | 0.0323 | 0.0380 |
| steer | 0.0041 [0.0020, 0.0066] | 0.0055 [0.0031, 0.0083] | 0.0043 [0.0026, 0.0063] | 0.0007 | 0.0043 |
| v | 0.0376 [0.0280, 0.0489] | 0.0530 [0.0411, 0.0665] | 0.0590 [0.0474, 0.0716] | 0.0467 | 0.0585 |
| v0 | 0.0376 [0.0280, 0.0489] | 0.0527 [0.0407, 0.0664] | 0.0589 [0.0471, 0.0717] | 0.0475 | 0.0585 |
| d_steer05 | 0.0053 [0.0047, 0.0060] | 0.0063 [0.0056, 0.0072] | 0.0052 [0.0050, 0.0054] | 0.0051 | 0.0052 |
| d_a1 | 0.0293 [0.0252, 0.0336] | 0.0429 [0.0353, 0.0512] | 0.0448 [0.0407, 0.0499] | 0.0385 | 0.0447 |

## sens_k(c), Tactical (k = 1/5/10 steps = 0.6/3.0/6.0 s) -- mean of per-window ratios, episode-cluster 95 % CI

| swap | k=1 | k=5 | k=10 | median k=10 | ratio-of-means k=10 |
|---|---|---|---|---|---|
| self | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| all | 0.2488 [0.1967, 0.3088] | 0.2540 [0.2111, 0.3021] | 0.2134 [0.1695, 0.2643] | 0.1465 | 0.2105 |
| a | 0.2472 [0.1943, 0.3092] | 0.2457 [0.2037, 0.2914] | 0.1983 [0.1602, 0.2418] | 0.1467 | 0.1963 |
| steer | 0.0071 [0.0034, 0.0116] | 0.0062 [0.0033, 0.0097] | 0.0045 [0.0028, 0.0066] | 0.0008 | 0.0043 |
| v | 0.1008 [0.0734, 0.1361] | 0.0924 [0.0721, 0.1150] | 0.0903 [0.0709, 0.1110] | 0.0750 | 0.0884 |
| v0 | 0.1008 [0.0734, 0.1361] | 0.0926 [0.0723, 0.1146] | 0.0911 [0.0717, 0.1115] | 0.0763 | 0.0898 |
| d_steer05 | 0.0075 [0.0069, 0.0082] | 0.0075 [0.0069, 0.0084] | 0.0073 [0.0069, 0.0079] | 0.0073 | 0.0072 |
| d_a1 | 0.3361 [0.3134, 0.3620] | 0.3259 [0.3081, 0.3444] | 0.3240 [0.3075, 0.3405] | 0.3231 | 0.3158 |

## EXPLORATORY: sens after the model's linear decoder to_enc (operative; the space the loss reads)

| swap | k=1 | k=5 | k=10 |
|---|---|---|---|
| self | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| all | 0.0617 [0.0475, 0.0777] | 0.1524 [0.1288, 0.1788] | 0.2131 [0.1816, 0.2473] |
| a | 0.0263 [0.0177, 0.0392] | 0.0981 [0.0785, 0.1200] | 0.1514 [0.1259, 0.1802] |
| steer | 0.0048 [0.0024, 0.0079] | 0.0084 [0.0046, 0.0127] | 0.0097 [0.0054, 0.0148] |
| v | 0.0474 [0.0355, 0.0612] | 0.1089 [0.0861, 0.1330] | 0.1472 [0.1182, 0.1776] |
| v0 | 0.0474 [0.0355, 0.0612] | 0.1089 [0.0860, 0.1335] | 0.1477 [0.1177, 0.1790] |
| d_steer05 | 0.0063 [0.0056, 0.0073] | 0.0094 [0.0087, 0.0104] | 0.0107 [0.0104, 0.0111] |
| d_a1 | 0.0363 [0.0325, 0.0404] | 0.1205 [0.1152, 0.1263] | 0.1768 [0.1721, 0.1818] |

## rho (paper Eq. 10; within-episode variance share), episode-cluster 95 % CI

| quantity | rho | CI | n_ep | n_vec |
|---|---|---|---|---|
| feat_tok | 0.1669 | [0.1494, 0.1934] | 40 | 2681 |
| feat_full | 0.4813 | [0.4565, 0.5203] | 40 | 2681 |
| adp_tok | 0.1067 | [0.0950, 0.1261] | 40 | 2681 |
| adp_full | 0.3530 | [0.3324, 0.3885] | 40 | 2681 |
| adp10_tok | 0.1210 | [0.1072, 0.1441] | 40 | 280 |
| adp10_full | 0.3859 | [0.3622, 0.4263] | 40 | 280 |
| tac0_tok | 0.0161 | [0.0119, 0.0239] | 40 | 280 |
| tac0_full | 0.0161 | [0.0119, 0.0239] | 40 | 280 |
| op1_tok | 0.1259 | [0.1115, 0.1495] | 40 | 280 |
| op1_full | 0.3862 | [0.3622, 0.4264] | 40 | 280 |
| op5_tok | 0.1299 | [0.1148, 0.1540] | 40 | 280 |
| op5_full | 0.3882 | [0.3643, 0.4281] | 40 | 280 |
| op10_tok | 0.1319 | [0.1174, 0.1555] | 40 | 280 |
| op10_full | 0.3899 | [0.3664, 0.4296] | 40 | 280 |
| tacp1_tok | 0.0161 | [0.0119, 0.0239] | 40 | 280 |
| tacp1_full | 0.0161 | [0.0119, 0.0239] | 40 | 280 |
| tacp5_tok | 0.0162 | [0.0120, 0.0238] | 40 | 280 |
| tacp5_full | 0.0162 | [0.0120, 0.0238] | 40 | 280 |
| tacp10_tok | 0.0165 | [0.0122, 0.0241] | 40 | 280 |
| tacp10_full | 0.0165 | [0.0122, 0.0241] | 40 | 280 |

Reference (paper App. L, Table 12, 5 fps, DROID): collapsed rho 0.008-0.016; IDM-on healthy 0.12; time-repulsion 0.17-0.92. (Different encoder, corpus and latent shape.)

## Input perturbation size of each swap (RMS over windows and horizon)

| level | swap | rms d(a) m/s^2 | rms d(steer) rad | rms d(v/30) |
|---|---|---|---|---|
| op | all | 1.1471 | 0.11368 | 0.3287 |
| op | a | 1.1471 | 0.00000 | 0.0000 |
| op | steer | 0.0000 | 0.11368 | 0.0000 |
| op | v | 0.0000 | 0.00000 | 0.3287 |
| op | v0 | 0.0000 | 0.00000 | 0.3329 |
| op | d_steer05 | 0.0000 | 0.05000 | 0.0000 |
| op | d_a1 | 1.0000 | 0.00000 | 0.0000 |
| tac | all | 1.0310 | 0.08578 | 0.3306 |
| tac | a | 1.0310 | 0.00000 | 0.0000 |
| tac | steer | 0.0000 | 0.08578 | 0.0000 |
| tac | v | 0.0000 | 0.00000 | 0.3306 |
| tac | v0 | 0.0000 | 0.00000 | 0.3329 |
| tac | d_steer05 | 0.0000 | 0.05000 | 0.0000 |
| tac | d_a1 | 1.0000 | 0.00000 | 0.0000 |

## Controls

* self_swap_op: {"n_windows": 40, "max_numerator_all_k": 0.0, "all_bit_identical": true, "max_abs_field_diff": 0.0, "reads_exactly_zero": true}
* self_swap_tac: {"n_windows": 80, "max_numerator_all_k": 0.0, "all_bit_identical": true, "max_abs_field_diff": 0.0, "reads_exactly_zero": true}
* stub identity / op / self / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / op / all / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / op / steer / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / op / v / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / op / a / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / tac / self / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / tac / all / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / tac / steer / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / tac / v / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub identity / tac / a / k=10: num_max_abs=0.000e+00 den_min=0.000e+00 sens_mean=n/a [n/a, n/a] n_den_zero=80
* stub drift / op / self / k=10: num_max_abs=0.000e+00 den_min=1.619e+02 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / op / all / k=10: num_max_abs=0.000e+00 den_min=1.619e+02 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / op / steer / k=10: num_max_abs=0.000e+00 den_min=1.619e+02 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / op / v / k=10: num_max_abs=0.000e+00 den_min=1.619e+02 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / op / a / k=10: num_max_abs=0.000e+00 den_min=1.619e+02 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / tac / self / k=10: num_max_abs=0.000e+00 den_min=5.098e+01 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / tac / all / k=10: num_max_abs=0.000e+00 den_min=5.098e+01 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / tac / steer / k=10: num_max_abs=0.000e+00 den_min=5.098e+01 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / tac / v / k=10: num_max_abs=0.000e+00 den_min=5.098e+01 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub drift / tac / a / k=10: num_max_abs=0.000e+00 den_min=5.098e+01 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub linear / op / self / k=10: num_max_abs=0.000e+00 den_min=1.439e+01 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub linear / op / all / k=10: num_max_abs=5.373e+02 den_min=1.439e+01 sens_mean=1.543289 [1.237369, 1.932219] n_den_zero=0
* stub linear / op / steer / k=10: num_max_abs=9.438e+01 den_min=1.439e+01 sens_mean=0.073543 [0.042021, 0.110918] n_den_zero=0
* stub linear / op / v / k=10: num_max_abs=1.280e+02 den_min=1.439e+01 sens_mean=0.480820 [0.362487, 0.633828] n_den_zero=0
* stub linear / op / a / k=10: num_max_abs=5.367e+02 den_min=1.439e+01 sens_mean=1.370671 [1.073126, 1.754450] n_den_zero=0
* stub linear / tac / self / k=10: num_max_abs=0.000e+00 den_min=4.983e+00 sens_mean=0.000000 [0.000000, 0.000000] n_den_zero=0
* stub linear / tac / all / k=10: num_max_abs=1.089e+02 den_min=4.983e+00 sens_mean=1.318416 [1.055032, 1.638325] n_den_zero=0
* stub linear / tac / steer / k=10: num_max_abs=1.480e+01 den_min=4.983e+00 sens_mean=0.060161 [0.038555, 0.083569] n_den_zero=0
* stub linear / tac / v / k=10: num_max_abs=3.814e+01 den_min=4.983e+00 sens_mean=0.517469 [0.387857, 0.684255] n_den_zero=0
* stub linear / tac / a / k=10: num_max_abs=1.078e+02 den_min=4.983e+00 sens_mean=1.121135 [0.874169, 1.424812] n_den_zero=0

## Rollout displacement vs field norm (variant `all`)

{
 "op": {
  "k1": {
   "mean_den": 306.91345192408846,
   "mean_z0_norm": 851.0201931476549,
   "mean_den_over_z0norm": 0.36067451906533404,
   "n": 80
  },
  "k5": {
   "mean_den": 323.20935712836996,
   "mean_z0_norm": 851.0201931476549,
   "mean_den_over_z0norm": 0.37983519541258914,
   "n": 80
  },
  "k10": {
   "mean_den": 355.8742715299093,
   "mean_z0_norm": 851.0201931476549,
   "mean_den_over_z0norm": 0.4181923551767598,
   "n": 80
  }
 },
 "tac": {
  "k1": {
   "mean_den": 1.8231589233368986,
   "mean_z0_norm": 319.4764355577275,
   "mean_den_over_z0norm": 0.006225068367264948,
   "n": 80
  },
  "k5": {
   "mean_den": 9.003447196860964,
   "mean_z0_norm": 319.4764355577275,
   "mean_den_over_z0norm": 0.030730904943885407,
   "n": 80
  },
  "k10": {
   "mean_den": 17.864328367169772,
   "mean_z0_norm": 319.4764355577275,
   "mean_den_over_z0norm": 0.06086323212788687,
   "n": 80
  }
 }
}

## WM-vs-persistence (T0, teacher-forced, frozen target space, S_10 windows)

{
 "model_k1": {
  "mean": 0.298917124367186,
  "n": 280
 },
 "model_k5": {
  "mean": 0.3779370869909014,
  "n": 280
 },
 "model_k10": {
  "mean": 0.42277453632227013,
  "n": 280
 },
 "const_k1": {
  "mean": 0.26060509445544866,
  "n": 280
 },
 "const_k5": {
  "mean": 0.489473915938288,
  "n": 280
 },
 "const_k10": {
  "mean": 0.5930591012484261,
  "n": 280
 }
}

## Exploratory terciles of the swap magnitude (op, k=10)

```
{
 "steer": {
  "channel": "steer",
  "rows": [
   {
    "bin": "low",
    "n": 27,
    "perturb_rms_mean": 0.0025244186264764584,
    "sens_mean": 0.00021982765230745815,
    "sens_median": 0.00021970574367911598
   },
   {
    "bin": "mid",
    "n": 26,
    "perturb_rms_mean": 0.00899331966558328,
    "sens_mean": 0.0007812811606287786,
    "sens_median": 0.0006811416071614466
   },
   {
    "bin": "high",
    "n": 27,
    "perturb_rms_mean": 0.12517924599901395,
    "sens_mean": 0.011709033881548583,
    "sens_median": 0.005941065675489526
   }
  ]
 },
 "v": {
  "channel": "v/30",
  "rows": [
   {
    "bin": "low",
    "n": 27,
    "perturb_rms_mean": 0.08253351985304444,
    "sens_mean": 0.021008594363424345,
    "sens_median": 0.018851059588426682
   },
   {
    "bin": "mid",
    "n": 26,
    "perturb_rms_mean": 0.23181114116540322,
    "sens_mean": 0.048355072072175424,
    "sens_median": 0.0454452032034173
   },
   {
    "bin": "high",
    "n": 27,
    "perturb_rms_mean": 0.4974661436345842,
    "sens_mean": 0.10725009275309053,
    "sens_median": 0.10131779158558421
   }
  ]
 },
 "a": {
  "channel": "a",
  "rows": [
   {
    "bin": "low",
    "n": 27,
    "perturb_rms_mean": 0.34046310517523026,
    "sens_mean": 0.010579195382549282,
    "sens_median": 0.010602635563100517
   },
   {
    "bin": "mid",
    "n": 26,
    "perturb_rms_mean": 0.830500955765064,
    "sens_mean": 0.03617654164715588,
    "sens_median": 0.03309816438584397
   },
   {
    "bin": "high",
    "n": 27,
    "perturb_rms_mean": 1.635259955017655,
    "sens_mean": 0.06833687272817957,
    "sens_median": 0.05168447448093476
   }
  ]
 }
}
```

## Tripwire (probe rollouts vs model.forward())

{"window": ["01be5919-6e61-45ba-a193-5f447efb4b51", 3], "max_abs_diff_op_vs_forward": 0.0, "max_abs_diff_tac_vs_forward": 0.0, "op_pred_absmax": 6.899797439575195}
