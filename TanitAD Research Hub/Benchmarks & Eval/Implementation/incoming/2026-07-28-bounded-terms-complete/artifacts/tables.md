# Generated tables — bounded-terms completion (2026-07-28)

*Every number here is produced by `code/tables.py` from the JSON in `raw/`. Nothing is hand-typed.*

## T1 — `lat_heading`'s raw quantity `u = |dpsi| / PSI_TOL`, per arm

| arm | n def | u med | u p90 | u p99 | u max | frac>1 | floor TERM | floor MEAN |
|---|---|---|---|---|---|---|---|---|
| cv_holdv0 | 15092 | 0.7156 | 2.1942 | 3.7783 | 4.9069 | 0.3628 | 0.3638 | 0.1981 |
| nospeed_tactical_oracle | 15269 | 0.7686 | 1.6846 | 3.2034 | 6.3199 | 0.3724 | 0.3730 | 0.2066 |
| oracle_lon_straight | 15190 | 0.7190 | 2.3620 | 3.9732 | 15.4483 | 0.3687 | 0.3694 | 0.1972 |
| refc_base_produced | 15086 | 1.0131 | 1.9106 | 2.7762 | 13.2605 | 0.5048 | 0.5052 | 0.2345 |
| refc_base_v0off | 14849 | 1.0193 | 1.9318 | 2.7897 | 5.7648 | 0.5079 | 0.5087 | 0.2510 |
| refc_base_v0on | 15086 | 1.0131 | 1.9106 | 2.7761 | 13.2605 | 0.5049 | 0.5052 | 0.2345 |
| refc_small_produced | 15106 | 0.9986 | 1.9334 | 3.0174 | 15.7023 | 0.4997 | 0.4999 | 0.2321 |
| refc_xl_produced | 15146 | 1.0112 | 1.8873 | 2.8342 | 6.2470 | 0.5057 | 0.5059 | 0.2331 |
| refc_xl_v0off | 15039 | 1.0271 | 1.9020 | 2.8359 | 6.4660 | 0.5121 | 0.5125 | 0.2497 |
| refc_xl_v0on | 15146 | 1.0112 | 1.8873 | 2.8342 | 6.2472 | 0.5057 | 0.5060 | 0.2331 |
| stand_still | 0 | — | — | — | — | — | — | — |
| v1_ego_double | 15246 | 0.7756 | 1.7038 | 3.3897 | 13.1204 | 0.3694 | 0.3701 | 0.2039 |
| v1_ego_half | 14693 | 0.7158 | 1.3168 | 2.1900 | 13.1204 | 0.3116 | 0.3122 | 0.2213 |
| v1_ego_oracle_lon | 15190 | 0.7788 | 1.7163 | 3.0743 | 15.4004 | 0.3749 | 0.3757 | 0.1971 |
| v1_ego_v0 | 15092 | 0.7678 | 1.6513 | 3.0353 | 13.1204 | 0.3661 | 0.3670 | 0.1984 |
| v1_lat_straight | 15340 | 0.7116 | 2.2044 | 4.3514 | 6.0548 | 0.3605 | 0.3609 | 0.2073 |
| v1_tactical_follow | 15340 | 0.7785 | 1.6706 | 3.2245 | 6.6792 | 0.3743 | 0.3750 | 0.2085 |
| v1_tactical_oracle | 15340 | 0.7776 | 1.6819 | 3.2286 | 6.6792 | 0.3738 | 0.3746 | 0.2062 |
| v4_blind | 15442 | 6.0696 | 12.5688 | 15.2785 | 15.7074 | 0.8429 | 0.8429 | 0.0260 |
| v4_oracle | 15329 | 0.8644 | 2.1407 | 10.4746 | 15.6325 | 0.4357 | 0.4364 | 0.0121 |
| **POOLED (non-probe)** | — | **0.9103** | 2.1537 | 11.6969 | 15.7074 | **0.4629** | — | — |

## T2 — would WIDENING `PSI_TOL` fix it? Floor fraction of the published shape

| arm | psi_tol 0.1 | 0.2 (published) | 0.4 | 0.8 (4x) |
|---|---|---|---|---|
| cv_holdv0 | 0.6572 | **0.3628** | 0.1157 | 0.0056 |
| nospeed_tactical_oracle | 0.6750 | **0.3724** | 0.0606 | 0.0018 |
| oracle_lon_straight | 0.6602 | **0.3687** | 0.1265 | 0.0093 |
| refc_base_produced | 0.7543 | **0.5048** | 0.0727 | 0.0014 |
| refc_base_v0off | 0.7531 | **0.5079** | 0.0829 | 0.0011 |
| refc_base_v0on | 0.7543 | **0.5049** | 0.0727 | 0.0014 |
| refc_small_produced | 0.7467 | **0.4997** | 0.0823 | 0.0032 |
| refc_xl_produced | 0.7611 | **0.5057** | 0.0607 | 0.0015 |
| refc_xl_v0off | 0.7620 | **0.5121** | 0.0670 | 0.0015 |
| refc_xl_v0on | 0.7611 | **0.5057** | 0.0607 | 0.0015 |
| v1_ego_double | 0.6733 | **0.3694** | 0.0649 | 0.0031 |
| v1_ego_half | 0.6221 | **0.3116** | 0.0163 | 0.0005 |
| v1_ego_oracle_lon | 0.6750 | **0.3749** | 0.0651 | 0.0013 |
| v1_ego_v0 | 0.6693 | **0.3661** | 0.0580 | 0.0011 |
| v1_lat_straight | 0.6499 | **0.3605** | 0.1147 | 0.0169 |
| v1_tactical_follow | 0.6722 | **0.3743** | 0.0602 | 0.0020 |
| v1_tactical_oracle | 0.6731 | **0.3738** | 0.0614 | 0.0020 |
| v4_blind | 0.9172 | **0.8429** | 0.7461 | 0.6120 |
| v4_oracle | 0.6944 | **0.4357** | 0.1193 | 0.0261 |

⛔ Widening PSI_TOL_RAD moves the floor and does not remove the one-sidedness: at FOUR TIMES the published tolerance (0.8 rad = 45.8 deg, at which point 'tolerance' means nothing) `v4_blind` is still floored on 61.20 % of its rows.

## T3 — `ego_progress@twosided_v2`: WHERE the floor comes from

| arm | floor total | from r<=0 (UNDER) | from r>=2 (OVER) | median r | max r |
|---|---|---|---|---|---|
| cv_holdv0 | 0.0141 | 0.0050 | 0.0091 | 0.9834 | 4.32 |
| nospeed_tactical_oracle | 0.0781 | 0.0014 | 0.0767 | 1.0210 | 64.63 |
| oracle_lon_straight | 0.0000 | 0.0000 | 0.0000 | 0.9933 | 1.06 |
| refc_base_produced | 0.0114 | 0.0016 | 0.0096 | 0.9784 | 10.46 |
| refc_base_v0off | 0.0174 | 0.0041 | 0.0132 | 0.9071 | 11.38 |
| refc_base_v0on | 0.0114 | 0.0016 | 0.0096 | 0.9784 | 10.46 |
| refc_small_produced | 0.0123 | 0.0016 | 0.0106 | 0.9793 | 9.26 |
| refc_xl_produced | 0.0093 | 0.0005 | 0.0088 | 0.9853 | 7.88 |
| refc_xl_v0off | 0.0155 | 0.0006 | 0.0148 | 0.9477 | 7.88 |
| refc_xl_v0on | 0.0093 | 0.0005 | 0.0088 | 0.9853 | 7.88 |
| stand_still | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 |
| v1_ego_double | 0.2426 | 0.0060 | 0.2342 | 1.9556 | 8.65 |
| v1_ego_half | 0.0062 | 0.0058 | 0.0005 | 0.4904 | 2.16 |
| v1_ego_oracle_lon | 0.0008 | 0.0008 | 0.0000 | 0.9900 | 1.06 |
| v1_ego_v0 | 0.0147 | 0.0059 | 0.0088 | 0.9792 | 4.32 |
| v1_lat_straight | 0.0822 | 0.0000 | 0.0822 | 1.0493 | 70.66 |
| v1_tactical_follow | 0.0805 | 0.0006 | 0.0798 | 1.0407 | 70.52 |
| v1_tactical_oracle | 0.0750 | 0.0006 | 0.0742 | 1.0298 | 70.52 |
| v4_blind | 0.3270 | 0.3178 | 0.0093 | 0.9724 | 4.30 |
| v4_oracle | 0.0117 | 0.0007 | 0.0110 | 0.9870 | 15.03 |

## T4 — THE ACCEPTANCE TEST, every candidate term

| term | resolution | shape | H0 purity | H1 correct | H2 min live_frac | reward bias | SURVIVES |
|---|---|---|---|---|---|---|---|
| `term_lin_q0` | term | lin_q0 | ✅ | 5/10 | 0.1570 | 1.000 | ⛔ no |
| `term_lin_q0p5` | term | lin_q0p5 | ✅ | 10/10 | 0.2536 | 1.000 | ⛔ no |
| `term_lin_q0p6667` | term | lin_q0p6667 | ✅ | 10/10 | 0.3222 | 1.000 | ⛔ no |
| `term_lin_q0p85` | term | lin_q0p85 | ✅ | 9/10 | 0.5250 | 1.000 | ⛔ no |
| `term_lin_q0p9` | term | lin_q0p9 | ✅ | 9/10 | 0.7393 | 1.000 | ⛔ no |
| `term_share_q0p5` | term | share_q0p5 | ✅ | 10/10 | 0.9977 | 3.332 | ⭐ **YES** |
| `term_cos_q0p5` | term | cos_q0p5 | ✅ | 10/10 | 0.2443 | 0.079 | ⛔ no |
| `term_cos_q0p75` | term | cos_q0p75 | ✅ | 10/10 | 0.3068 | 0.064 | ⛔ no |
| `mean_lin_q0` | mean | lin_q0 | ⛔ | 9/10 | 0.7490 | 1.000 | ⛔ no |
| `mean_lin_q0p5` | mean | lin_q0p5 | ⛔ | 10/10 | 0.9993 | 1.000 | ⛔ no |
| `mean_lin_q0p6667` | mean | lin_q0p6667 | ⛔ | 10/10 | 0.9977 | 1.000 | ⛔ no |
| `mean_lin_q0p85` | mean | lin_q0p85 | ⛔ | 10/10 | 0.9885 | 1.000 | ⛔ no |
| `mean_lin_q0p9` | mean | lin_q0p9 | ⛔ | 10/10 | 0.9783 | 1.000 | ⛔ no |
| `mean_share_q0p5` | mean | share_q0p5 | ⛔ | 10/10 | 1.0000 | 3.332 | ⛔ no |
| `mean_cos_q0p5` | mean | cos_q0p5 | ⛔ | 10/10 | 0.9380 | 0.079 | ⛔ no |
| `mean_cos_q0p75` | mean | cos_q0p75 | ⛔ | 10/10 | 0.9190 | 0.064 | ⛔ no |
| `mean1_lin_q0` | mean1 | lin_q0 | ✅ | 9/10 | 0.7464 | 1.000 | ⛔ no |
| `mean1_lin_q0p5` | mean1 | lin_q0p5 | ✅ | 10/10 | 0.9975 | 1.000 | ⭐ **YES** |
| `mean1_lin_q0p6667` | mean1 | lin_q0p6667 | ✅ | 10/10 | 0.9977 | 1.000 | ⭐ **YES** |
| `mean1_lin_q0p85` | mean1 | lin_q0p85 | ✅ | 10/10 | 0.9886 | 1.000 | ⭐ **YES** |
| `mean1_lin_q0p9` | mean1 | lin_q0p9 | ✅ | 10/10 | 0.9791 | 1.000 | ⭐ **YES** |
| `mean1_share_q0p5` | mean1 | share_q0p5 | ✅ | 10/10 | 1.0000 | 3.332 | ⭐ **YES** |
| `mean1_cos_q0p5` | mean1 | cos_q0p5 | ✅ | 10/10 | 0.9383 | 0.079 | ⭐ **YES** |
| `mean1_cos_q0p75` | mean1 | cos_q0p75 | ✅ | 10/10 | 0.9201 | 0.064 | ⭐ **YES** |

**SELECTED: `mean1_lin_q0p5`** — H3 — linear family, smallest q

## T5 — the shipped term, cell by cell

| arm | injection | zero-mean | `term_lin_q0` (PUBLISHED) | `mean1_lin_q0p5` (SHIPPED) |
|---|---|---|---|---|
| cv_holdv0 | `yaw_bias(+5)` |  | -0.0097 [-0.0160, -0.0036] SEP | -0.0322 [-0.0414, -0.0219] SEP |
| cv_holdv0 | `yaw_bias(-5)` |  | -0.0097 [-0.0154, -0.0035] SEP | -0.0369 [-0.0452, -0.0275] SEP |
| cv_holdv0 | `lat_drift(+0.05)` |  | -0.0021 [-0.0054, +0.0010] n.s. | -0.0111 [-0.0164, -0.0055] SEP |
| cv_holdv0 | `lat_drift(-0.05)` |  | -0.0035 [-0.0066, -0.0002] SEP | -0.0144 [-0.0193, -0.0091] SEP |
| cv_holdv0 | `yaw_jitter(+5)` | ⭐ **yes** | -0.0146 [-0.0195, -0.0098] SEP | -0.0337 [-0.0380, -0.0293] SEP |
| v1_tactical_follow | `yaw_bias(+5)` |  | -0.0046 [-0.0100, +0.0007] n.s. | -0.0276 [-0.0330, -0.0221] SEP |
| v1_tactical_follow | `yaw_bias(-5)` |  | -0.0049 [-0.0138, +0.0039] n.s. | -0.0352 [-0.0400, -0.0303] SEP |
| v1_tactical_follow | `lat_drift(+0.05)` |  | -0.0025 [-0.0062, +0.0011] n.s. | -0.0092 [-0.0120, -0.0062] SEP |
| v1_tactical_follow | `lat_drift(-0.05)` |  | -0.0025 [-0.0071, +0.0023] n.s. | -0.0134 [-0.0159, -0.0108] SEP |
| v1_tactical_follow | `yaw_jitter(+5)` | ⭐ **yes** | -0.0098 [-0.0144, -0.0051] SEP | -0.0310 [-0.0351, -0.0270] SEP |

## T6 — the CONTAMINATION panel (a translation must move it LEAST)

| arm | control | `term_lin_q0` | `mean1_lin_q0p5` | `mean_lin_q0` |
|---|---|---|---|---|
| cv_holdv0 | `lat_shift(+2)` | -0.0001 [-0.0003, +0.0001] n.s. | -0.0035 [-0.0073, -0.0003] SEP | -0.0222 [-0.0243, -0.0204] SEP |
| cv_holdv0 | `lat_shift(-2)` | +0.0002 [-0.0000, +0.0004] n.s. | +0.0027 [-0.0007, +0.0066] n.s. | -0.0197 [-0.0218, -0.0175] SEP |
| cv_holdv0 | `lat_jitter(+1)` | +0.0000 [-0.0000, +0.0001] n.s. | -0.0004 [-0.0006, -0.0002] SEP | -0.0158 [-0.0168, -0.0148] SEP |
| cv_holdv0 | `lon_retime(+0.5)` | +0.0306 [+0.0179, +0.0462] SEP | +0.0361 [+0.0219, +0.0521] SEP | +0.0233 [+0.0139, +0.0341] SEP |
| v1_tactical_follow | `lat_shift(+2)` | +0.0012 [-0.0001, +0.0030] n.s. | -0.0024 [-0.0056, +0.0003] n.s. | -0.0213 [-0.0237, -0.0191] SEP |
| v1_tactical_follow | `lat_shift(-2)` | +0.0001 [-0.0007, +0.0011] n.s. | +0.0023 [-0.0010, +0.0058] n.s. | -0.0182 [-0.0207, -0.0156] SEP |
| v1_tactical_follow | `lat_jitter(+1)` | +0.0001 [-0.0001, +0.0004] n.s. | +0.0000 [-0.0001, +0.0002] n.s. | -0.0143 [-0.0150, -0.0136] SEP |
| v1_tactical_follow | `lon_retime(+0.5)` | +0.0309 [+0.0227, +0.0388] SEP | +0.0348 [+0.0275, +0.0426] SEP | +0.0245 [+0.0194, +0.0294] SEP |

## T7 — C47's discriminator on THIS term's density (panel median u = 0.9167)

| shape | |dg/du| @0 | @0.5 | @1 | @2 | @3 | reward bias | H1 (term / mean / mean1) |
|---|---|---|---|---|---|---|---|
| `cos_q0p5` | 0.0001 | 0.5554 | 0.7854 | 0.0000 | 0.0000 | **0.079** | 10/10/10 |
| `cos_q0p75` | 0.0000 | 0.2618 | 0.4534 | 0.4534 | 0.0000 | **0.064** | 10/10/10 |
| `lin_q0` | 1.0000 | 1.0000 | 0.5000 | 0.0000 | 0.0000 | **1.000** | 5/9/9 |
| `lin_q0p5` | 0.5000 | 0.5000 | 0.5000 | 0.2500 | 0.0000 | **1.000** | 10/10/10 |
| `lin_q0p6667` | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.1667 | **1.000** | 10/10/10 |
| `lin_q0p85` | 0.1500 | 0.1500 | 0.1500 | 0.1500 | 0.1500 | **1.000** | 9/10/10 |
| `lin_q0p9` | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | **1.000** | 9/10/10 |
| `share_q0p5` | 0.9999 | 0.4444 | 0.2500 | 0.1111 | 0.0625 | **3.332** | 10/10/10 |

Families by ascending reward bias: ['cos', 'lin', 'share'] · **pass rate monotone in reward bias: False**

## T8 — THE GUARD: what v1 admits and what v2 refuses

|  | gate v1 (published) | gate v2 (shipped) |
|---|---|---|
| `lat_heading@term_lin_q0` (PUBLISHED) | ✅ ADMITTED | ⛔ **REFUSED** |
| `lat_heading@mean1_lin_q0p5` (FIXED) | ✅ ADMITTED | ✅ ADMITTED |

| check | result |
|---|---|
| `OLD_GUARD_ADMITTED_THE_BROKEN_TERM` | ✅ True |
| `GUARD_REFUSES_THE_BROKEN_TERM` | ✅ True |
| `GUARD_ADMITS_THE_FIXED_TERM` | ✅ True |

| arm | live_frac (published term) | v1 | v2 |
|---|---|---|---|
| cv_holdv0 | 0.6339 | True | True |
| nospeed_tactical_oracle | 0.6260 | True | True |
| oracle_lon_straight | 0.6283 | True | True |
| refc_base_produced | 0.4933 | True | False |
| refc_base_v0off | 0.4901 | True | False |
| refc_base_v0on | 0.4933 | True | False |
| refc_small_produced | 0.4986 | True | False |
| refc_xl_produced | 0.4925 | True | False |
| refc_xl_v0off | 0.4864 | True | False |
| refc_xl_v0on | 0.4925 | True | False |
| stand_still | — | None | None |
| v1_ego_double | 0.6290 | True | True |
| v1_ego_half | 0.6863 | True | True |
| v1_ego_oracle_lon | 0.6232 | True | True |
| v1_ego_v0 | 0.6317 | True | True |
| v1_lat_straight | 0.6368 | True | True |
| v1_tactical_follow | 0.6241 | True | True |
| v1_tactical_oracle | 0.6244 | True | True |
| v4_blind | 0.1570 | True | False |
| v4_oracle | 0.5633 | True | True |

## T9 — THE REPRODUCTION GATE

| check | value |
|---|---|
| published `@clamp_v1` composites checked | 16 |
| **max |diff|** | **0.000000** |
| verdict | ✅ PASS |
| published metric id, unchanged | `PSS_recovery_progress@clamp_v1` |
| published `lat_heading` BIT-identical (20/20 arms) | ✅ True |
| published axis id unchanged | ✅ True |
| published suite id | `control_v1@t1s_d1.75m_sref10m_psi0.2rad` |
| new suite id | `control_v1@t1s_d1.75m_sref10m_psi0.2rad+lath_rowmean_v2` |

## T10 — the RANKING STATEMENT and the INSTRUMENT GUARDS

| statement | value |
|---|---|
| `cv_holdv0_rank_realisable_PSS` | 1 |
| `cv_holdv0_rank_realisable_CONTROL_published` | 1 |
| `cv_holdv0_rank_realisable_CONTROL_new` | 1 |
| `cv_holdv0_STILL_RANKS_FIRST_AMONG_REALISABLE_PSS` | True |
| `cv_holdv0_STILL_RANKS_FIRST_AMONG_REALISABLE_CONTROL` | True |
| `whole_v1_tactical_family_below_every_REFC_PSS` | True |
| `whole_v1_tactical_family_below_every_REFC_CONTROL_published` | True |
| `whole_v1_tactical_family_below_every_REFC_CONTROL_new` | True |

| guard | PSS | CONTROL (published lat_heading) | CONTROL (shipped) |
|---|---|---|---|
| `G1_v4_oracle_minus_v4_blind` | +0.2992 [+0.2217, +0.3829] SEP | +0.2353 [+0.1801, +0.2949] SEP | +0.2384 [+0.1815, +0.3004] SEP |
| `G2_v1_ego_half_minus_v1_tactical_follow` | -0.1454 [-0.1758, -0.1149] SEP | -0.1310 [-0.1605, -0.1003] SEP | -0.1318 [-0.1614, -0.1010] SEP |

### CONTROL composite levels

| arm | published `lat_heading` | shipped `lat_heading` |
|---|---|---|
| oracle_lon_straight | 0.7944 | 0.8112 |
| v1_ego_oracle_lon | 0.7874 | 0.8050 |
| cv_holdv0 | 0.7499 | 0.7665 |
| v4_oracle | 0.7439 | 0.7614 |
| v1_ego_v0 | 0.7425 | 0.7598 |
| refc_xl_produced | 0.7268 | 0.7444 |
| refc_xl_v0on | 0.7269 | 0.7444 |
| refc_base_produced | 0.7234 | 0.7407 |
| refc_base_v0on | 0.7234 | 0.7407 |
| refc_small_produced | 0.7198 | 0.7370 |
| refc_xl_v0off | 0.6615 | 0.6784 |
| refc_base_v0off | 0.6462 | 0.6627 |
| v1_tactical_oracle | 0.6225 | 0.6395 |
| v1_lat_straight | 0.6223 | 0.6381 |
| v1_tactical_follow | 0.6183 | 0.6352 |
| nospeed_tactical_oracle | 0.6123 | 0.6290 |
| v4_blind | 0.5089 | 0.5234 |
| v1_ego_half | 0.4872 | 0.5036 |
| v1_ego_double | 0.3121 | 0.3279 |
| stand_still | 0.1170 | 0.1170 |

## T11 — `ego_progress` over-travel acceptance test

| progress term | correct | over-side floor max (scorable) | failing cells |
|---|---|---|---|
| `clamp_v1` | 2/12 | 0.0000 | `cv_holdv0|lon_retime(+1.5)`, `cv_holdv0|lon_retime(+2)`, `cv_holdv0|lon_scale(+1.5)`, `v1_tactical_follow|lon_retime(+1.5)`, `v1_tactical_follow|lon_retime(+2)`, `v1_tactical_follow|lon_scale(+1.5)`, `v1_ego_double|lon_retime(+1.5)`, `v1_ego_double|lon_retime(+2)`, `v1_ego_double|lon_scale(+1.5)`, `v1_ego_double|lon_jitter(+2)` |
| `twosided_v2` | 11/12 | 0.0822 | `v1_ego_double|lon_jitter(+2)` |
| `twosided_asym_w0p5` | 11/12 | 0.0316 | `v1_ego_double|lon_jitter(+2)` |
| `twosided_asym_w0p3333` | 11/12 | 0.0185 | `v1_ego_double|lon_jitter(+2)` |
| `twosided_asym_w2` | 11/12 | 0.1748 | `v1_ego_double|lon_jitter(+2)` |

### the decisive cells, on the already-over-travelling substrate

| cell | `clamp_v1` | `twosided_v2` | `twosided_asym_w0p5` | `twosided_asym_w0p3333` | `twosided_asym_w2` |
|---|---|---|---|---|---|
| `v1_ego_double|lon_retime(+1.5)` | +0.0033 [+0.0010, +0.0062] SEP | -0.0973 [-0.1168, -0.0782] SEP | -0.4286 [-0.4517, -0.4031] SEP | -0.3048 [-0.3167, -0.2902] SEP | -0.0213 [-0.0327, -0.0115] SEP |
| `v1_ego_double|lon_retime(+2)` | +0.0044 [+0.0013, +0.0083] SEP | -0.1071 [-0.1318, -0.0835] SEP | -0.4989 [-0.5150, -0.4791] SEP | -0.5870 [-0.6133, -0.5576] SEP | -0.0262 [-0.0414, -0.0133] SEP |
| `v1_ego_double|lon_scale(+1.5)` | +0.0033 [+0.0010, +0.0062] SEP | -0.0987 [-0.1189, -0.0791] SEP | -0.4323 [-0.4548, -0.4069] SEP | -0.3076 [-0.3193, -0.2930] SEP | -0.0219 [-0.0337, -0.0117] SEP |
| `v1_ego_double|lon_jitter(+2)` | -0.0013 [-0.0037, +0.0010] n.s. | +0.0104 [+0.0054, +0.0158] SEP | -0.0013 [-0.0039, +0.0014] n.s. | -0.0013 [-0.0039, +0.0017] n.s. | +0.0030 [-0.0010, +0.0073] n.s. |

## T12 — the FULL bounded-term audit, on the two-sided `live_frac`

| term | n_sub | max floor | max ceil | min live_frac | gate v1 | gate v2 |
|---|---|---|---|---|---|---|
| `ego_progress@clamp_v1` | 1 | 0.3178 | 0.5703 | 0.2461 | ✅ | ⛔ |
| `ego_progress@twosided_v2` | 1 | 0.3270 | 0.1173 | 0.6704 | ✅ | ✅ |
| `recovery@clamp_v1` | 1 | 0.9219 | 0.0002 | 0.0781 | ✅ | ⛔ |
| `recovery@twosided_v2` | 1 | 0.0833 | 0.0007 | 0.9161 | ✅ | ✅ |
| `comfort` | 1 | 1.0000 | 1.0000 | 0.0000 | ⛔ | ⛔ |
| `lon_track` | 20 | 0.0029 | 0.1070 | 0.8930 | ✅ | ✅ |
| `lat_track` | 20 | 0.0947 | 0.0008 | 0.9051 | ✅ | ✅ |
| `lat_heading@term_lin_q0 (PUBLISHED)` | 1 | 0.8429 | 0.0023 | 0.1570 | ✅ | ⛔ |
| `lat_heading@mean1_lin_q0p5 (SHIPPED)` | 20 | 0.0014 | 0.0010 | 0.9975 | ✅ | ✅ |

## T13 — `comfort`, 100 % saturated by construction

| arm | pass rate | distinct values | observed_range | live_frac |
|---|---|---|---|---|
| cv_holdv0 | 1.0000 | 1 | 0.0000 | 0.0000 |
| nospeed_tactical_oracle | 0.0007 | 2 | 1.0000 | -0.0000 |
| oracle_lon_straight | 0.9361 | 2 | 1.0000 | 0.0000 |
| refc_base_produced | 0.0082 | 2 | 1.0000 | -0.0000 |
| refc_base_v0off | 0.0036 | 2 | 1.0000 | 0.0000 |
| refc_base_v0on | 0.0082 | 2 | 1.0000 | -0.0000 |
| refc_small_produced | 0.0027 | 2 | 1.0000 | -0.0000 |
| refc_xl_produced | 0.0054 | 2 | 1.0000 | -0.0000 |
| refc_xl_v0off | 0.0050 | 2 | 1.0000 | 0.0000 |
| refc_xl_v0on | 0.0053 | 2 | 1.0000 | 0.0000 |
| stand_still | 1.0000 | 1 | 0.0000 | 0.0000 |
| v1_ego_double | 0.0987 | 2 | 1.0000 | 0.0000 |
| v1_ego_half | 0.6372 | 2 | 1.0000 | -0.0000 |
| v1_ego_oracle_lon | 0.2453 | 2 | 1.0000 | 0.0000 |
| v1_ego_v0 | 0.2882 | 2 | 1.0000 | 0.0000 |
| v1_lat_straight | 0.0099 | 2 | 1.0000 | -0.0000 |
| v1_tactical_follow | 0.0004 | 2 | 1.0000 | -0.0000 |
| v1_tactical_oracle | 0.0004 | 2 | 1.0000 | -0.0000 |
| v4_blind | 0.0000 | 1 | 0.0000 | 0.0000 |
| v4_oracle | 0.0000 | 1 | 0.0000 | 0.0000 |

**DECISION.** PUBLISH THE MEASUREMENT, RETIRE THE NAME AND THE SCORE-SHAPED NODE. (a) It must not be emitted as `components.comfort` beside two real scores in [0,1]: a reader comparing 0.5492 (recovery) with 1.0000 (comfort) is comparing a mean score with a PASS RATE, and the program has already published `observed_range = 1.0` for it 20 times as if it were range. (b) It must not be silently deleted either: C46 was found BECAUSE the number was still on the page, and a measurement that refutes its own term is the cheapest instrument in the suite. ⇒ it is emitted as `diagnostics.plan_smoothness_pass_rate`, a RATE with its own units, carrying COMFORT_STATUS and its live_frac = 0.0000, and gate v2 refuses it automatically so no future weight can be attached without the refusal being visible.
