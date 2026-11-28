## Gates

| check | result |
|---|---|
| windows | **599 new vs 599 committed** |
| episode clusters | 596 |
| `eid` ordering identical (both dumps) | True |
| `t0` ordering identical | True |
| anchor `a_imagination__own__roSTR` (in `bi_perwindow_compact`) | max \|Δ\| = **0 m** (tol 0.0001) |
| anchor `a_imagination__hold__roSTR` (in `bi_perwindow_compact`) | max \|Δ\| = **0 m** (tol 0.0001) |
| anchor `b_frozenlast__own__roSTR` (in `perwindow_matched_K185`) | max \|Δ\| = **3.05e-05 m** (tol 0.0001) |
| anchor `b_frozenlast__hold__roSTR` (in `perwindow_matched_K185`) | max \|Δ\| = **3.05e-05 m** (tol 0.0001) |
| **WINDOW-SET IDENTITY GATE** | **PASS** |
| self-test `a_selftest__blend0` == `a_imagination__own__roSTR` | max \|Δ\| = **0** — BIT-IDENTICAL |
| self-test `a_selftest__every1` == `a_imagination__own__roSTR` | max \|Δ\| = **0** — BIT-IDENTICAL |
| self-test `a_selftest__blend1` == `a_imagination__hold__roSTR` | max \|Δ\| = **0** — BIT-IDENTICAL |
| anti-no-op: smallest \|Δ\| vs `own` over all filter arms | **226.672363 m** |
| arms identical to `own` (must be empty) | `[]` |
| failing-value probe — identical arms / swapped arms | **1 / 1 steps** (both must be 1) |
| **ALL GATES** | **PASS** |

### Fidelity against the ladder's committed numbers

| quantity | committed | recomputed here |
|---|---:|---:|
| `T_blind_own_str` | 25 | **25** |
| `T_blind_hold_str` | 115 | **115** |
| `de2s_own_str` | 1.8165 | **1.8165** |
| `de2s_hold_str` | 0.6718 | **0.6718** |
| `ade_0_2s_own_str` | 0.871 | **0.871** |
| `ade_0_2s_hold_str` | 0.3351 | **0.3351** |
| `T_useful_1m_own_str_s` | 1.4 | **1.4** |
| `T_useful_1m_hold_str_s` | 2.3 | **2.3** |

`LEVEL_FIDELITY_PASS = True` · `T_BLIND_EXACT_REPRODUCTION = True` · `T_useful_reproduces = True`


## The blend curve — own actions <-> hold-last

| α | eligible | `T_blind` | CI95 (s) | Δ vs own (steps) | `de@2s` | `ade_0_2s` | paired Δ@2s vs comparator | beats CV | `T_useful@1m` |
|---:|---|---:|---|---|---:|---:|---|---:|---:|
| 0 | ⛔ endpoint | **25** (2.5 s) | [2.5, 3.9] | +0 ⛔ | 1.8165 | 0.8710 | +0.4130 [+0.2651, +0.5673] ✅ | 0/185 | 1.4 s |
| 0.125 | ✅ | **55** (5.5 s) | [5.5, 8.4] | +36 ✅ | 1.3755 | 0.6793 | +0.8091 [+0.6706, +0.9498] ✅ | 0/185 | 1.6 s |
| 0.25 | ✅ | **85** (8.5 s) | [8.5, 13.1] | +74 ✅ | 1.0736 | 0.5440 | +1.0751 [+0.9459, +1.2099] ✅ | 43/185 | 1.9 s |
| 0.375 | ✅ | **101** (10.1 s) | [10.1, 15.9] | +94 ✅ | 0.9103 | 0.4611 | +1.2111 [+1.0786, +1.3449] ✅ | 62/185 | 2.0 s |
| 0.5 | ✅ | **111** (11.1 s) | [11.1, 16.9] | +105 ✅ | 0.7924 | 0.4010 | +1.3090 [+1.1838, +1.4416] ✅ | 72/185 | 2.2 s |
| 0.625 | ✅ | **115** (11.5 s) | [11.5, 17.2] | +109 ✅ | 0.7184 | 0.3633 | +1.3663 [+1.2406, +1.5010] ✅ | 78/185 | 2.3 s |
| 0.75 | ✅ | **116** (11.6 s) | [11.6, 17.2] | +109 ✅ | 0.6842 | 0.3437 | +1.3862 [+1.2612, +1.5206] ✅ | 81/185 | 2.3 s |
| 0.875 | ✅ | **116** (11.6 s) | [11.6, 17.3] | +109 ✅ | 0.6724 | 0.3362 | +1.3871 [+1.2598, +1.5210] ✅ | 82/185 | 2.3 s |
| 1 | ⛔ endpoint | **115** (11.5 s) | [11.5, 17.4] | +109 ✅ | 0.6718 | 0.3351 | +1.3785 [+1.2503, +1.5122] ✅ | 83/185 | 2.3 s |

monotone non-decreasing in α: **False** · Spearman(α, `T_blind`) = **0.95**


## The other intervention families

| family | config | eligible | `T_blind` | CI95 (s) | Δ vs own | `de@2s` | `ade_0_2s` | beats CV | `T_useful@1m` |
|---|---|---|---:|---|---|---:|---:|---:|---:|
| clip_steer | `steerclip0.02` | ✅ | **24** (2.4 s) | [2.4, 3.8] | -1 ⛔ | 1.8497 | 0.8833 | 0/185 | 1.4 s |
| clip_steer | `steerclip0.005` | ✅ | **25** (2.5 s) | [2.5, 4.4] | +2 ⛔ | 1.8289 | 0.8757 | 0/185 | 1.4 s |
| clip_steer | `steerclip0` | ⛔ diagnostic | **27** (2.7 s) | [2.7, 5.5] | +8 ⛔ | 1.8143 | 0.8718 | 0/185 | 1.4 s |
| clip_accel | `accelclip1` | ✅ | **47** (4.7 s) | [4.7, 7.3] | +27 ✅ | 1.3513 | 0.6477 | 0/185 | 1.7 s |
| clip_accel | `accelclip0.3` | ✅ | **62** (6.2 s) | [6.2, 10.1] | +46 ✅ | 1.2053 | 0.5774 | 14/185 | 1.8 s |
| clip_accel | `accelclip0` | ⛔ diagnostic | **78** (7.8 s) | [7.8, 12.7] | +67 ✅ | 1.1684 | 0.5608 | 23/185 | 1.8 s |
| smooth_ema | `ema0.5` | ✅ | **38** (3.8 s) | [3.8, 5.5] | +15 ✅ | 1.3862 | 0.6550 | 0/185 | 1.7 s |
| smooth_ema | `ema0.8` | ✅ | **64** (6.4 s) | [6.4, 8.9] | +44 ✅ | 0.9864 | 0.4667 | 34/185 | 2.0 s |
| smooth_ema | `ema0.95` | ✅ | **111** (11.1 s) | [11.1, 16.1] | +101 ✅ | 0.6966 | 0.3464 | 76/185 | 2.3 s |
| update_every | `every2` | ✅ | **9** (0.9 s) | [0.9, 1.1] | -21 ⛔ | 2.8106 | 1.4625 | 0/185 | 1.0 s |
| update_every | `every5` | ✅ | **9** (0.9 s) | [0.9, 1.0] | -22 ⛔ | 3.9867 | 1.9242 | 0/185 | 0.9 s |
| update_every | `every20` | ✅ | **9** (0.9 s) | [0.9, 1.0] | -22 ⛔ | 4.6320 | 2.1458 | 0/185 | 0.9 s |
| channel_decomposition | `chansteer` | ⛔ diagnostic | **90** (9.0 s) | [9.0, 13.8] | +77 ✅ | 0.8797 | 0.4306 | 60/185 | 2.1 s |
| channel_decomposition | `chanaccel` | ⛔ diagnostic | **49** (4.9 s) | [4.9, 7.8] | +31 ✅ | 1.5493 | 0.7580 | 0/185 | 1.5 s |
| convention_and_speed | `gtkin` | ⛔ diagnostic | **185** (18.5 s) | [18.5, 18.5] | +154 ✅ | 0.4361 | 0.2552 | 179/185 | 3.0 s |
| convention_and_speed | `own_vupd` | ✅ | **9** (0.9 s) | [0.9, 1.0] | -21 ⛔ | 23.9351 | 9.6020 | 0/185 | 0.6 s |

## Mechanism
### The penalty curve (own − hold), comparator-free

| step | penalty (m) | own / hold |
|---:|---:|---:|
| 1 | 0.0 | 1.0 |
| 2 | 0.0078 | 1.089 |
| 5 | 0.0656 | 1.617 |
| 10 | 0.2737 | 2.438 |
| 20 | 1.1447 | 2.704 |
| 40 | 3.2722 | 2.009 |
| 80 | 8.7958 | 1.585 |
| 120 | 15.031 | 1.438 |
| 185 | 23.979 | 1.305 |

`loglog_fit_2_20`: exponent **2.098**, R² 0.995, n 19, window [2, 20] — admissible: True

`loglog_fit_20_185`: exponent **1.346**, R² 0.997, n 166, window [20, 185] — admissible: True

`loglog_fit_full`: exponent **1.559**, R² 0.984, n 184, window [2, 185] — admissible: True

### Action statistics (reconstructed over all 599 windows)

| arm | step | mean \|steer\| | mean \|accel\| | frac steer at clamp | frac accel at clamp | jitter(steer) | jitter(accel) | mean speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `a_imagination__own__roSTR` | 5 | 0.00978 | 2.0582 | 0.0928 | 0.4641 | 0.00158 | 3.1524 | 12.99 |
| `a_imagination__own__roSTR` | 20 | 0.01110 | 2.0057 | 0.1011 | 0.4580 | 0.00165 | 2.8091 | 12.95 |
| `a_imagination__own__roSTR` | 40 | 0.01361 | 1.8293 | 0.1431 | 0.3943 | 0.00131 | 2.4429 | 13.20 |
| `a_imagination__own__roSTR` | 185 | 0.01337 | 1.0892 | 0.1331 | 0.2074 | 0.00064 | 1.4125 | 13.77 |
| `a_imagination__hold__roSTR` | 5 | 0.01037 | 1.5088 | 0.1078 | 0.2992 | 0.00084 | 2.1331 | 12.99 |
| `a_imagination__hold__roSTR` | 20 | 0.01155 | 0.9341 | 0.1235 | 0.1171 | 0.00073 | 0.9891 | 12.97 |
| `a_imagination__hold__roSTR` | 40 | 0.01199 | 0.6184 | 0.1236 | 0.0591 | 0.00048 | 0.5000 | 13.01 |
| `a_imagination__hold__roSTR` | 185 | 0.01259 | 0.2717 | 0.1244 | 0.0129 | 0.00017 | 0.1083 | 12.99 |
| `b_frozenlast__own__roSTR` | 5 | 0.00990 | 1.7338 | 0.0992 | 0.3519 | 0.00089 | 2.4447 | 13.07 |
| `b_frozenlast__own__roSTR` | 20 | 0.00991 | 1.3764 | 0.1024 | 0.3042 | 0.00056 | 2.3034 | 13.07 |
| `b_frozenlast__own__roSTR` | 40 | 0.00992 | 1.2825 | 0.1035 | 0.3074 | 0.00048 | 2.2539 | 13.06 |
| `b_frozenlast__own__roSTR` | 185 | 0.00992 | 1.2113 | 0.1039 | 0.3166 | 0.00041 | 2.2341 | 13.06 |
| `a_gtkin` | 5 | 0.00984 | 0.5387 | 0.0805 | 0.0053 | 0.00122 | 0.2976 | 12.97 |
| `a_gtkin` | 20 | 0.01013 | 0.4808 | 0.0665 | 0.0048 | 0.00116 | 0.1349 | 12.93 |
| `a_gtkin` | 40 | 0.01044 | 0.4801 | 0.0722 | 0.0043 | 0.00093 | 0.1151 | 12.87 |
| `a_gtkin` | 185 | 0.01075 | 0.5028 | 0.0830 | 0.0040 | 0.00054 | 0.1031 | 13.05 |
| `a_blend0.5` | 5 | 0.01022 | 1.7983 | 0.1088 | 0.3629 | 0.00084 | 2.8165 | 13.01 |
| `a_blend0.5` | 20 | 0.01155 | 1.4944 | 0.1220 | 0.2893 | 0.00094 | 2.2459 | 13.05 |
| `a_blend0.5` | 40 | 0.01198 | 1.0754 | 0.1256 | 0.1926 | 0.00070 | 1.5043 | 13.18 |
| `a_blend0.5` | 185 | 0.01251 | 0.4938 | 0.1220 | 0.0766 | 0.00028 | 0.5833 | 13.29 |
| `a_ema0.8` | 5 | 0.01028 | 1.5099 | 0.1078 | 0.3008 | 0.00073 | 2.2694 | 13.01 |
| `a_ema0.8` | 20 | 0.01013 | 1.0469 | 0.0895 | 0.1376 | 0.00079 | 1.2508 | 13.11 |
| `a_ema0.8` | 40 | 0.00991 | 0.7963 | 0.0846 | 0.0774 | 0.00057 | 0.6502 | 13.30 |
| `a_ema0.8` | 185 | 0.00855 | 0.3846 | 0.0501 | 0.0213 | 0.00020 | 0.1483 | 14.21 |

### The onset sweep — recovery as a fraction of the own→hold gap

| arm | @5 | @20 | @40 | @185 |
|---|---:|---:|---:|---:|
| `own_first_5_then_hold` | 0.000 | 0.798 | 0.997 | 1.031 |
| `hold_first_5_then_own` | 1.000 | 0.627 | 0.301 | 0.276 |
| `own_first_10_then_hold` | 0.000 | 0.388 | 0.825 | 1.074 |
| `hold_first_10_then_own` | 1.000 | 0.902 | 0.564 | 0.458 |
| `own_first_20_then_hold` | 0.000 | 0.000 | 0.263 | 0.979 |
| `hold_first_20_then_own` | 1.000 | 1.000 | 0.886 | 0.622 |
| `own_first_40_then_hold` | 0.000 | 0.000 | 0.000 | 0.654 |
| `hold_first_40_then_own` | 1.000 | 1.000 | 1.000 | 0.705 |

### Fed actions, dense (audit subset)

| arm | mean \|steer\| | mean \|accel\| | frac steer at clamp | frac accel at clamp | jitter(steer) | jitter(accel) |
|---|---:|---:|---:|---:|---:|---:|
| `own` | 0.01388 | 1.2440 | 0.1062 | 0.2295 | 0.00076 | 1.6833 |
| `hold` | 0.02723 | 0.4235 | 0.1707 | 0.0000 | 0.00000 | 0.0000 |
| `true` | 0.03428 | 0.5954 | 0.1648 | 0.0163 | 0.00189 | 0.1526 |
| `gtkin` | 0.01265 | 0.4780 | 0.1524 | 0.0028 | 0.00101 | 0.1013 |
| `own_op` | 0.02291 | 1.0015 | 0.2338 | 0.2025 | 0.00087 | 1.5384 |
| `blend0.5` | 0.01767 | 0.3855 | 0.1221 | 0.0000 | 0.00017 | 0.2336 |
| `ema0.8` | 0.00838 | 0.3441 | 0.0107 | 0.0000 | 0.00028 | 0.0334 |
| `every5` | 0.00900 | 0.6297 | 0.0351 | 0.0630 | 0.00020 | 0.0411 |
| `accelclip0.3` | 0.00940 | 0.1960 | 0.0276 | 0.0000 | 0.00025 | 0.0395 |
| `own_frozen` | 0.01175 | 1.1914 | 0.0726 | 0.3061 | 0.00040 | 2.1267 |

## Verdict

```json
{
 "rule": "PRE_REGISTRATION.md \u00a75 \u2014 CONFIRM: best ELIGIBLE T_blind >= 50 steps AND paired gain separated; PARTIAL: 31..49 separated, or >=50 not separated; REFUTE: <= 30 steps. Baseline 25 steps, ceiling 115 steps.",
 "n_eligible_arms": 18,
 "baseline_T_blind_steps_prereg": 25,
 "ceiling_T_blind_steps_prereg": 115,
 "baseline_T_blind_steps_this_run": 25,
 "ceiling_T_blind_steps_this_run": 115,
 "best_eligible": "blend0.75",
 "best_T_blind_steps": 116,
 "best_T_blind_s": 11.6,
 "best_T_blind_ci95_s": [
  11.6,
  17.2
 ],
 "paired_gain_vs_own": {
  "median_gain_steps": 109.0,
  "ci95_steps": [
   85.0,
   140.02499999999986
  ],
  "separated_better": true,
  "frac_draws_gain_gt_0": 1.0,
  "n_boot": 2000,
  "estimator": "paired_episode_cluster_bootstrap"
 },
 "bonferroni_required_frac": 0.9972,
 "bonferroni_met": true,
 "frac_of_ceiling_recovered": 1.0111,
 "VERDICT": "CONFIRM",
 "CAPABILITY_CAP": {
  "beats_cv_steps": 81,
  "beats_cv_interval_s": [
   0.7,
   8.7
  ],
  "T_useful_s": {
   "1m": 2.3,
   "1.391m": 2.7,
   "2m": 3.2
  },
  "baseline_T_useful_1m_s": 1.4,
  "capability_verdict": "CONFIRM"
 },
 "dose_response": {
  "monotone_nondecreasing": false,
  "spearman_alpha_vs_T_blind": 0.95,
  "why_it_matters": "a best-of-N selection effect cannot manufacture a monotone dose-response between two independently MEASURED endpoints"
 }
}
```

### Ranking of ELIGIBLE arms by `T_blind`

| rank | arm | steps | s |
|---:|---|---:|---:|
| 1 | `blend0.75` | 116 | 11.6 |
| 2 | `blend0.875` | 116 | 11.6 |
| 3 | `blend0.625` | 115 | 11.5 |
| 4 | `blend0.5` | 111 | 11.1 |
| 5 | `ema0.95` | 111 | 11.1 |
| 6 | `blend0.375` | 101 | 10.1 |
| 7 | `blend0.25` | 85 | 8.5 |
| 8 | `ema0.8` | 64 | 6.4 |
| 9 | `accelclip0.3` | 62 | 6.2 |
| 10 | `blend0.125` | 55 | 5.5 |
| 11 | `accelclip1` | 47 | 4.7 |
| 12 | `ema0.5` | 38 | 3.8 |
| 13 | `steerclip0.005` | 25 | 2.5 |
| 14 | `steerclip0.02` | 24 | 2.4 |
| 15 | `every2` | 9 | 0.9 |
| 16 | `every5` | 9 | 0.9 |
| 17 | `every20` | 9 | 0.9 |
| 18 | `own_vupd` | 9 | 0.9 |
