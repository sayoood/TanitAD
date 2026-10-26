# LHA0 diagnostic -- GO

**Tier:** T0 label-level diagnostic on FRESH-split out-of-fold heads over frozen z_tac (no planner, no plan metric).  **Estimator:** another draw of fresh EPISODES with the OOF scores fixed; blind to training (fold-fit) variance (see beside.fold_seed_replicates) and to inference variance (head deterministic).

**Function class:** ridge multinomial logistic (LH's frozen family), lambda 100 fixed, over standardised features: a NO-GO is 'no >= +0.020 gain for a LINEAR head', not 'a0 / psi_dot0 carry no decel information'.

**Primary:** dAUC(B - A) = +0.0241  95% CI [+0.0172, +0.0320]  (bar: >= +0.020 and CI lower bound > 0)  -> clears bar: True.  A 0.6699  B 0.694.

**C4 (permuted within episode):** dAUC(C4 - A) = +0.0174  [+0.0107, +0.0247]  clears bar: False.

## Arms (seed-0 LH folds, same fit rows, same scored rows)

| arm | AUC | 95% CI (episode bootstrap) | d | lambda | n fit rows | n scored rows | scored episodes | OOF CE (fit rows) |
|---|---|---|---|---|---|---|---|---|
| A | 0.6699 | [+0.6161, +0.7212] | 513 | 100.0 | 8178 | 5038 | 280 | 1.091727 |
| B | 0.694 | [+0.6447, +0.7416] | 515 | 100.0 | 8178 | 5038 | 280 | 1.08598 |
| C2 | 0.5526 | [+0.5011, +0.6053] | 1 | 1.0 | 8178 | 5038 | 280 | 1.372789 |
| C3 | 0.7656 | [+0.7373, +0.7949] | 1 | 1.0 | 8178 | 5038 | 280 | 1.456452 |
| C4 | 0.6873 | [+0.6377, +0.7356] | 515 | 100.0 | 8178 | 5038 | 280 | 1.075963 |
| C5 | 0.6949 | [+0.6460, +0.7425] | 515 | 100.0 | 8178 | 5038 | 280 | 1.085561 |
| B05 | 0.6917 | [+0.6427, +0.7399] | 515 | 100.0 | 8178 | 5038 | 280 | 1.087256 |
| Bnom | 0.694 | [+0.6448, +0.7416] | 515 | 100.0 | 8178 | 5038 | 280 | 1.085973 |
| X1 | 0.7058 | [+0.6711, +0.7393] | 3 | 1.0 | 8178 | 5038 | 280 | 1.331733 |
| C1 | 0.5 | - | 0 | - | 0 | 5038 | 280 | - |
| C1_fold_prior_beside | 0.5142 | - | - | - | - | - | - | - |

## Gates

```
{
 "K_REPRO_A": {
  "oof_mean_ce_A_lam100": 1.0917272119292791,
  "LH_FIT_INFO": 1.0917272119292791,
  "abs_diff": 0.0,
  "pass": true
 },
 "K_POP": {
  "s6_valid": 3668,
  "compliant": 3251,
  "gt_decel": 639,
  "gt_cruise": 1856,
  "union_n": 2495,
  "n_episodes": 136,
  "LH_auc": 0.7304592642059252,
  "want": {
   "s6_valid": 3668,
   "compliant": 3251,
   "gt_decel": 639,
   "gt_cruise": 1856,
   "union_n": 2495,
   "n_episodes": 136,
   "LH_auc": 0.7305
  },
  "pass": true
 },
 "K_ID": {
  "fit_rows_Z_bitwise_equal_bank": true,
  "fit_rows_v0_bitwise_equal_bank": true,
  "bank_v0_in_vs_extras_v0_maxabs": 0.0,
  "dump_v0_vs_bank_v0_maxabs": 0.0,
  "n_U": 10107,
  "n_U_dropped_nonfinite_features": 0,
  "n_fit_rows": 8178,
  "n_fit_episodes": 290,
  "pass": true
 },
 "K_CONST_C1_auc": 0.5,
 "K_CONST_pass": true,
 "K_WAUC_unit_weights": {
  "A": true,
  "B": true,
  "C2": true,
  "C3": true,
  "C4": true,
  "C5": true,
  "B05": true,
  "Bnom": true,
  "X1": true
 },
 "calibration": {
  "verdict": "CALIBRATED",
  "anchor_rule": "k = ws + (0) ; bank v0_in == poses[k,3] for all 10456 windows",
  "max_abs_diff": 0.0,
  "mutation_fwd_neq_bwd_frac": 0.9534238714613619
 },
 "ALL_PASS": true
}
```

## Population

```
{
 "U_rows": 10107,
 "U_episodes": 290,
 "P_rows": 5038,
 "P_decel": 1468,
 "P_cruise": 3570,
 "P_episodes": 280,
 "P_decel_base_rate": 0.2914,
 "Pfit_rows": 4883,
 "Pfit_decel": 1440,
 "Pnocomp_rows": 5645,
 "Pnocomp_decel": 1859,
 "P_rows_in_bank_outside_U": 169,
 "gt_accel_rows_in_U_excluded": 1875,
 "fit_rows": 8178,
 "fit_episodes": 290,
 "fit_class_counts": {
  "0 FOLLOW": 1796,
  "1 CRUISE": 2395,
  "3 BRAKE_TO": 1876,
  "4 CREEP": 103,
  "5 HOLD": 130,
  "7 ACCELERATE": 1878
 },
 "C4_frac_values_moved": 0.9566
}
```

## Beside (never gating)

```
{
 "fold_seed_replicates": {
  "summary": {
   "B_minus_A_per_seed": [
    0.0241,
    0.0249,
    0.0227,
    0.0233,
    0.024
   ],
   "mean": 0.0238,
   "min": 0.0227,
   "max": 0.0249,
   "sd_across_fold_seeds": 0.0008,
   "all_clear_bar": true,
   "C4_minus_A_per_seed": [
    0.0174,
    0.0177,
    0.0175,
    0.0171,
    0.0195
   ]
  },
  "per_seed": {
   "fold_seed_0": {
    "A_auc": 0.6699,
    "B_auc": 0.694,
    "C4_auc": 0.6873,
    "B_minus_A": {
     "point": 0.0241,
     "ci95": [
      0.0172,
      0.032
     ],
     "boot_se": 0.0037
    },
    "C4_minus_A": {
     "point": 0.0174,
     "ci95": [
      0.0107,
      0.0247
     ],
     "boot_se": 0.0036
    },
    "B_minus_C4": {
     "point": 0.0067,
     "ci95": [
      0.0002,
      0.013
     ],
     "boot_se": 0.0033
    },
    "oof_ce_A": 1.09173,
    "oof_ce_B": 1.08598
   },
   "fold_seed_1": {
    "A_auc": 0.6726,
    "B_auc": 0.6975,
    "C4_auc": 0.6903,
    "B_minus_A": {
     "point": 0.0249,
     "ci95": [
      0.0178,
      0.0333
     ],
     "boot_se": 0.0039
    },
    "C4_minus_A": {
     "point": 0.0177,
     "ci95": [
      0.0109,
      0.0246
     ],
     "boot_se": 0.0034
    },
    "B_minus_C4": {
     "point": 0.0072,
     "ci95": [
      0.0009,
      0.0137
     ],
     "boot_se": 0.0033
    },
    "oof_ce_A": 1.07802,
    "oof_ce_B": 1.07038
   },
   "fold_seed_2": {
    "A_auc": 0.6693,
    "B_auc": 0.692,
    "C4_auc": 0.6868,
    "B_minus_A": {
     "point": 0.0227,
     "ci95": [
      0.016,
      0.0306
     ],
     "boot_se": 0.0036
    },
    "C4_minus_A": {
     "point": 0.0175,
     "ci95": [
      0.0111,
      0.0241
     ],
     "boot_se": 0.0033
    },
    "B_minus_C4": {
     "point": 0.0052,
     "ci95": [
      -0.0008,
      0.0112
     ],
     "boot_se": 0.003
    },
    "oof_ce_A": 1.08784,
    "oof_ce_B": 1.08151
   },
   "fold_seed_3": {
    "A_auc": 0.6732,
    "B_auc": 0.6965,
    "C4_auc": 0.6904,
    "B_minus_A": {
     "point": 0.0233,
     "ci95": [
      0.0169,
      0.031
     ],
     "boot_se": 0.0035
    },
    "C4_minus_A": {
     "point": 0.0171,
     "ci95": [
      0.011,
      0.0234
     ],
     "boot_se": 0.0032
    },
    "B_minus_C4": {
     "point": 0.0062,
     "ci95": [
      0.0003,
      0.0121
     ],
     "boot_se": 0.003
    },
    "oof_ce_A": 1.07799,
    "oof_ce_B": 1.07375
   },
   "fold_seed_4": {
    "A_auc": 0.6691,
    "B_auc": 0.6931,
    "C4_auc": 0.6886,
    "B_minus_A": {
     "point": 0.024,
     "ci95": [
      0.0175,
      0.0315
     ],
     "boot_se": 0.0035
    },
    "C4_minus_A": {
     "point": 0.0195,
     "ci95": [
      0.0124,
      0.0269
     ],
     "boot_se": 0.0037
    },
    "B_minus_C4": {
     "point": 0.0045,
     "ci95": [
      -0.0018,
      0.0107
     ],
     "boot_se": 0.0032
    },
    "oof_ce_A": 1.08481,
    "oof_ce_B": 1.07621
   }
  }
 },
 "lambda_or_model_l10": {
  "A_auc": 0.6617,
  "B_auc": 0.6854,
  "B_minus_A": {
   "point": 0.0237,
   "ci95": [
    0.0166,
    0.0317
   ],
   "boot_se": 0.0038
  },
  "oof_ce_A": 1.18057,
  "oof_ce_B": 1.17353
 },
 "lambda_or_model_l100": {
  "A_auc": 0.6699,
  "B_auc": 0.694,
  "B_minus_A": {
   "point": 0.0241,
   "ci95": [
    0.0172,
    0.032
   ],
   "boot_se": 0.0037
  },
  "oof_ce_A": 1.09173,
  "oof_ce_B": 1.08598
 },
 "lambda_or_model_l1000": {
  "A_auc": 0.6817,
  "B_auc": 0.696,
  "B_minus_A": {
   "point": 0.0143,
   "ci95": [
    0.0104,
    0.019
   ],
   "boot_se": 0.0022
  },
  "oof_ce_A": 1.07032,
  "oof_ce_B": 1.06622
 },
 "lambda_or_model_mlp": {
  "A_auc": 0.6768,
  "B_auc": 0.6806,
  "B_minus_A": {
   "point": 0.0038,
   "ci95": [
    -0.0067,
    0.013
   ],
   "boot_se": 0.005
  },
  "oof_ce_A": 1.10031,
  "oof_ce_B": 1.11673
 },
 "contrasts_vs_A_seed0_lam100": {
  "C5": {
   "point": 0.025,
   "ci95": [
    0.0181,
    0.033
   ],
   "boot_se": 0.0038
  },
  "B05": {
   "point": 0.0218,
   "ci95": [
    0.0151,
    0.0294
   ],
   "boot_se": 0.0036
  },
  "Bnom": {
   "point": 0.0241,
   "ci95": [
    0.0172,
    0.0321
   ],
   "boot_se": 0.0037
  },
  "C2": {
   "point": -0.1173,
   "ci95": [
    -0.1723,
    -0.0553
   ],
   "boot_se": 0.0299
  },
  "C3": {
   "point": 0.0957,
   "ci95": [
    0.035,
    0.1576
   ],
   "boot_se": 0.031
  },
  "X1": {
   "point": 0.0358,
   "ci95": [
    -0.0142,
    0.0889
   ],
   "boot_se": 0.026
  }
 },
 "C5_minus_B": {
  "point": 0.0009,
  "ci95": [
   -0.0,
   0.0018
  ],
  "boot_se": 0.0005
 },
 "B_minus_C4": {
  "point": 0.0067,
  "ci95": [
   0.0002,
   0.013
  ],
  "boot_se": 0.0033
 },
 "B_minus_X1": {
  "point": -0.0117,
  "ci95": [
   -0.0619,
   0.0348
  ],
  "boot_se": 0.0243
 },
 "population_P_fit": {
  "n_rows": 4883,
  "n_decel": 1440,
  "n_episodes": 280,
  "A_auc": 0.6714,
  "B_auc": 0.6953,
  "B_minus_A": {
   "point": 0.024,
   "ci95": [
    0.0167,
    0.032
   ],
   "boot_se": 0.0038
  },
  "C4_minus_A": {
   "point": 0.0179,
   "ci95": [
    0.0111,
    0.0252
   ],
   "boot_se": 0.0036
  }
 },
 "population_P_nocomp": {
  "n_rows": 5645,
  "n_decel": 1859,
  "n_episodes": 283,
  "A_auc": 0.6596,
  "B_auc": 0.6845,
  "B_minus_A": {
   "point": 0.0249,
   "ci95": [
    0.0184,
    0.0324
   ],
   "boot_se": 0.0036
  },
  "C4_minus_A": {
   "point": 0.0185,
   "ci95": [
    0.0114,
    0.0262
   ],
   "boot_se": 0.0038
  }
 },
 "B_extra_feature_coefficients_std_units": {
  "classes": [
   "FOLLOW",
   "CRUISE",
   "BRAKE_TO",
   "CREEP",
   "HOLD",
   "ACCELERATE"
  ],
  "a0_mean_over_folds": [
   0.1984,
   0.1617,
   -0.1605,
   -0.2143,
   -0.1351,
   0.1497
  ],
  "psi_dot0_mean_over_folds": [
   -0.0768,
   0.0074,
   -0.0159,
   0.006,
   -0.0061,
   0.0854
  ],
  "a0_sd_over_folds": [
   0.0379,
   0.0311,
   0.0361,
   0.033,
   0.0371,
   0.0289
  ]
 }
}
```

---

## Reading (human-written, appended after the machine verdict; the verdict word above is the sealed rule's, unchanged)

**Verdict (sealed rule): GO.** dAUC(B - A) = +0.0241, 95 % paired episode-cluster CI [+0.0172, +0.0320], bar +0.020 and CI > 0 met; 5 / 5 fold-seed replicates (0.0227 - 0.0249) meet +0.020; all instrument gates pass
(K-REPRO-A reproduces LH's OOF CE 1.0917272119 to 0.0; K-POP reproduces K-HEAD's 3668 / 3251 / 639 / 1856 / 2495 and 0.7305 through the fresh-side code; C1 reads exactly 0.5; calibration exact: k = ws, max|v0_in - pose speed| = 0.0 on all 10,456 windows; forward != backward a0 on 95.3 % of rows).
What the estimator answers: "another draw of EPISODES, OOF scores fixed"; the fold-seed replicates (A, B, C4 re-fit on 4 more partitions) are the only training-variance read.

**Why the GO is weaker than it looks.**
1. C4 (within-episode permuted a0 / psi_dot0) = +0.0174 [+0.0107, +0.0247] over A: **72 % of the gain survives the shuffle** and C4 misses the VOID bar by only 0.0026 (replicate seed 4: 0.0195). The window-specific increment B - C4 = +0.0067 [+0.0002, +0.0130]; two of five fold-seed CIs include 0.
2. The gain is a property of the linear lambda-100 head: lambda 1000 -> +0.0143 [+0.0104, +0.0190]; the LH-recipe MLP -> +0.0038 [-0.0067, +0.0130].
3. The forward-difference contrast (C5) is uninformative: corr(a0_bwd, a0_fwd) = 0.974, C5 - B = +0.0009. It cannot certify "no leak"; the dataset's egomotion is an offline, smooth estimate, not an IMU.

**Beside, pre-registered:** C2 v0-only 0.5526; **C3 a0-only 0.7656 [0.7373, 0.7949]** (above A, B and X1 = 0.7058); B05 (0.5 s a0) 0.6917; Bnom (nominal dt) 0.6940; P-fit and P-nocomp populations give the same picture (+0.0240 / +0.0249).
**Post-hoc, NOT pre-registered (`LHA0_POSTHOC.json`):** raw -a0 AUC 0.7699, +0.100 [+0.042, +0.161] over A; equal-weight fusion A + C3 adds nothing over a0 alone. The B recipe leaves most of a0's information unused.
**Function class:** ridge multinomial logistic over standardised features. A GO is a lower bound on learnability; it says nothing about nonlinear heads (MLP beside: no gain).

**Provenance note.** `code/lha0_diag.py` was edited once between the fit stage and the assemble stage: only `write_md` (output formatting) and a syntax repair of its newline join were added; `fit_job`, `prep`, the bootstrap and the verdict logic ran unchanged (script_sha256 in the JSON is the assemble-time file). The 27 OOF-logit files of the fit stage are copied to `result/oof_logits/` (arrays only, no ids; the working copy is `F:/c_overflow_2026-10-26/tmp/lha0/jobs/`); `universe.npz` (the 10,107 x 512 z_tac table, 43 MB) stays only in that scratch directory and is regenerable by `lha0_diag.py prep` in ~1 min.
