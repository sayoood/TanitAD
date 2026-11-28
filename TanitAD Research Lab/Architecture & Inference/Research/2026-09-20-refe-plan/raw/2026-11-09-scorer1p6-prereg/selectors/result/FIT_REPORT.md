# FIT report -- PREREG_P6_SELECTORS

split sha256 `8a22f2e0c71c6faf56230058272841d8b03559075ac5e4c1930b3f05d260cf2f`; estimator: paired log-cluster bootstrap on FIT-B, 10,000 draws, seed 20260927, 95 % percentile; 'another draw of logs' only

| variant | FIT-B PDMS x100 | P6 rule | delta [95 % CI] | picks changed |
|---|---|---|---|---|
| R1 | 93.786 | 93.707 | +0.079 [-0.009, +0.133] | 216 |
| R2 | 93.804 | 93.707 | +0.097 [-0.102, +0.305] | 516 |
| C_platt | 93.707 | 93.707 | -0.000 [-0.019, +0.019] | 116 |
| C_isotonic | 93.673 | 93.707 | -0.034 [-0.198, +0.101] | 653 |
| C_platt_binned | 93.675 | 93.707 | -0.032 [-0.138, +0.093] | 319 |
| S_gbdt_pdms | 93.846 | 93.707 | +0.139 [-0.133, +0.414] | 1177 |
| S_gbdt_heads | 93.863 | 93.707 | +0.156 [-0.230, +0.527] | 1019 |
| S_mlp_heads | 92.281 | 93.707 | -1.427 [-1.931, -1.003] | 1897 |
| P_pg | 93.694 | 93.707 | -0.013 [-0.334, +0.292] | 1457 |

## Nominated (FIT-B only, CI-lo > 0)
- R rule shape: none. no variant has a paired log-cluster CI-lo > 0 against P6's rule on FIT-B: the family nominates none (reported, not a failure of the tool)
- C calibration: none. no variant has a paired log-cluster CI-lo > 0 against P6's rule on FIT-B: the family nominates none (reported, not a failure of the tool)
- S stackers: none. no variant has a paired log-cluster CI-lo > 0 against P6's rule on FIT-B: the family nominates none (reported, not a failure of the tool)
- P policy gradient: none. no variant has a paired log-cluster CI-lo > 0 against P6's rule on FIT-B: the family nominates none (reported, not a failure of the tool)

MEASURED (ours; fit_pack sha256 and split sha256 above). P6 logits are IN-SAMPLE on these logs (PREREG section 1): FIT numbers are not performance claims.
Tier: FIT-B is a selection set, not a driving tier.
