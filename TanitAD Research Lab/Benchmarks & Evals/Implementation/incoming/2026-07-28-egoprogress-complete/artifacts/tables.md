# Generated tables — `ego_progress`, both sides

*Regenerated from `raw/*.json` by `code/tables.py`. Nothing here is hand-typed.*

### T1 — the UNDER side's density, per arm (`term` = published n_sub 1, `mean` = 20-step)

| arm | r≤0 | r<0 strict | r=0 exact | min r | median r \| r≤0 | frac < −0.10 | term floor | term live | mean floor | mean live |
|---|---|---|---|---|---|---|---|---|---|---|
| stand_still | 1.0000 | 0.0000 | 1.0000 | 0.000 | 0.000 | 0.000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| ⛔ v4_blind | 0.3178 | 0.3178 | 0.0000 | -0.657 | -0.134 | 0.819 | 0.3270 | 0.6704 | 0.0010 | 0.9990 |
| v1_ego_double | 0.0060 | 0.0010 | 0.0050 | -5.252 | 0.000 | 0.141 | 0.2427 | 0.7573 | 0.0995 | 0.9005 |
| v1_ego_v0 | 0.0059 | 0.0009 | 0.0050 | -2.661 | 0.000 | 0.099 | 0.0147 | 0.9749 | 0.0063 | 0.9937 |
| v1_ego_half | 0.0058 | 0.0008 | 0.0050 | -1.365 | 0.000 | 0.056 | 0.0062 | 0.9938 | 0.0057 | 0.9943 |
| cv_holdv0 | 0.0050 | 0.0000 | 0.0050 | 0.000 | 0.000 | 0.000 | 0.0141 | 0.9742 | 0.0058 | 0.9942 |
| refc_base_v0off | 0.0041 | 0.0041 | 0.0000 | -1.971 | -0.294 | 0.812 | 0.0174 | 0.9790 | 0.0100 | 0.9896 |
| refc_base_produced | 0.0016 | 0.0016 | 0.0000 | -1.971 | -0.283 | 0.560 | 0.0114 | 0.9668 | 0.0035 | 0.9870 |
| refc_small_produced | 0.0016 | 0.0016 | 0.0000 | -1.377 | -0.026 | 0.360 | 0.0123 | 0.9669 | 0.0040 | 0.9849 |
| refc_base_v0on | 0.0016 | 0.0016 | 0.0000 | -1.971 | -0.283 | 0.560 | 0.0114 | 0.9668 | 0.0035 | 0.9870 |
| nospeed_tactical_oracle | 0.0014 | 0.0014 | 0.0000 | -2.369 | -0.348 | 0.809 | 0.0781 | 0.9194 | 0.0615 | 0.9385 |
| v1_ego_oracle_lon | 0.0008 | 0.0008 | 0.0000 | -0.997 | -0.520 | 0.923 | 0.0008 | 0.8818 | 0.0006 | 0.8858 |
| v4_oracle | 0.0007 | 0.0007 | 0.0000 | -0.853 | -0.798 | 1.000 | 0.0117 | 0.9745 | 0.0045 | 0.9955 |
| v1_tactical_follow | 0.0006 | 0.0006 | 0.0000 | -1.721 | -0.564 | 0.800 | 0.0805 | 0.9167 | 0.0621 | 0.9378 |
| v1_tactical_oracle | 0.0006 | 0.0006 | 0.0000 | -1.721 | -0.564 | 0.700 | 0.0750 | 0.9220 | 0.0573 | 0.9426 |
| refc_xl_v0off | 0.0006 | 0.0006 | 0.0000 | -0.034 | -0.011 | 0.000 | 0.0155 | 0.9778 | 0.0076 | 0.9920 |
| refc_xl_produced | 0.0005 | 0.0005 | 0.0000 | -0.033 | -0.011 | 0.000 | 0.0093 | 0.9723 | 0.0027 | 0.9871 |
| refc_xl_v0on | 0.0005 | 0.0005 | 0.0000 | -0.034 | -0.011 | 0.000 | 0.0093 | 0.9723 | 0.0027 | 0.9871 |
| oracle_lon_straight | 0.0000 | 0.0000 | 0.0000 | 0.978 | nan | nan | 0.0000 | 0.8699 | 0.0000 | 0.8825 |
| v1_lat_straight | 0.0000 | 0.0000 | 0.0000 | 0.102 | nan | nan | 0.0822 | 0.9143 | 0.0626 | 0.9373 |

### T2 — ⛔ THE SAME 42 ROWS, PUSHED FURTHER AND FURTHER BACKWARDS (rows with r <= 0 after lon_shift(-1 m), on v1_tactical_follow — HELD FIXED at every deeper shift so no row is re-selected)

| injection | ground-truth \|s_plan−s_human\| (m) | mean ratio r | clamp_v1 | twosided_v2 | sqrtlin_w0p3333 | hyp_w1 |
|---|---|---|---|---|---|---|
| lon_shift(-1) | 2.216 | -1.054 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| lon_shift(-3) | 4.198 | -3.592 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| lon_shift(-10) | 11.136 | -12.475 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| lon_shift(-30) | 30.956 | -37.856 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### T3 — ⭐ EVERY CONTROL'S DIRECTION, VERIFIED AGAINST A METRIC-INDEPENDENT GROUND TRUTH (`|s_plan − s_human|`, metres)

| cell | Δ ground-truth error (m) | admissible as a degradation? |
|---|---|---|
| cv_holdv0\|lon_retime(+0.7) | +7.1567 [+5.4200, +8.9487] SEP | ✅ |
| cv_holdv0\|lon_retime(+0.5) | +12.2846 [+9.4731, +15.2052] SEP | ✅ |
| cv_holdv0\|lon_scale(+0.5) | +12.2846 [+9.4731, +15.2052] SEP | ✅ |
| cv_holdv0\|lon_shift(-2) | +1.4850 [+1.2981, +1.6572] SEP | ✅ |
| cv_holdv0\|lon_shift(-5) | +4.3282 [+4.0603, +4.5766] SEP | ✅ |
| cv_holdv0\|lon_jitter(+2) | +0.8509 [+0.7686, +0.9328] SEP | ✅ |
| v1_tactical_follow\|lon_retime(+0.7) | +2.5582 [+0.6901, +4.4777] SEP | ✅ |
| v1_tactical_follow\|lon_retime(+0.5) | +6.8613 [+3.9305, +9.8637] SEP | ✅ |
| v1_tactical_follow\|lon_scale(+0.5) | +6.9233 [+3.9969, +9.9212] SEP | ✅ |
| v1_tactical_follow\|lon_shift(-2) | +0.0216 [-0.2564, +0.3264] n.s. | ⛔ **REFUSED** |
| v1_tactical_follow\|lon_shift(-5) | +0.8948 [+0.2422, +1.5946] SEP | ✅ |
| v1_tactical_follow\|lon_jitter(+2) | +0.2290 [+0.1810, +0.2796] SEP | ✅ |
| v4_blind\|lon_retime(+0.7) | +2.8149 [+0.7746, +4.9840] SEP | ✅ |
| v4_blind\|lon_retime(+0.5) | +7.1516 [+3.9096, +10.6158] SEP | ✅ |
| v4_blind\|lon_scale(+0.5) | +7.2985 [+4.0747, +10.7548] SEP | ✅ |
| v4_blind\|lon_shift(-2) | +0.5588 [+0.0795, +1.0213] SEP | ✅ |
| v4_blind\|lon_shift(-5) | +2.0871 [+0.9706, +3.1147] SEP | ✅ |
| v4_blind\|lon_jitter(+2) | +0.2519 [+0.1481, +0.3582] SEP | ✅ |

### T4 — the UNDER-SIDE acceptance test, every term (verified cells only)

| term | correct | separated WRONG WAY | all correct |
|---|---|---|---|
| clamp_v1 | 17/17 | 0 | ✅ |
| ⭐ twosided_v2 | 14/17 | 0 | ⛔ |
| twosided_asym_w0p5 | 17/17 | 0 | ✅ |
| twosided_asym_w0p3333 | 17/17 | 0 | ✅ |
| twosided_asym_w2 | 10/17 | 0 | ⛔ |
| hyp_w1 | 17/17 | 0 | ✅ |
| hyp_w0p5 | 17/17 | 0 | ✅ |
| hyp_w2 | 13/17 | 0 | ⛔ |
| exp_w1 | 17/17 | 0 | ✅ |
| exp_w0p5 | 17/17 | 0 | ✅ |
| sqrtlin_w1 | 17/17 | 0 | ✅ |
| sqrtlin_w0p5 | 17/17 | 0 | ✅ |
| sqrtlin_w0p3333 | 17/17 | 0 | ✅ |
| sqrtlin_w0p25 | 17/17 | 0 | ✅ |
| mean\|clamp_v1 | 17/17 | 0 | ✅ |
| mean\|twosided_v2 | 16/17 | 0 | ⛔ |
| mean\|sqrtlin_w0p3333 | 17/17 | 0 | ✅ |
| mean\|hyp_w1 | 17/17 | 0 | ✅ |

### T5 — ⭐ THE DECISIVE SUBSTRATE `v4_blind` (31.78 % of its rows at r ≤ 0), cell by cell

| term | lon_shift(−2 m) | lon_shift(−5 m) | ⭐ lon_jitter(σ=2 m) ZERO-MEAN |
|---|---|---|---|
| clamp_v1 | -0.0572 [-0.0833, -0.0359] SEP | -0.1326 [-0.1836, -0.0910] SEP | -0.0070 [-0.0131, -0.0014] SEP |
| twosided_v2 | -0.0214 [-0.0484, +0.0006] n.s. | -0.0667 [-0.1218, -0.0183] SEP | -0.0135 [-0.0243, -0.0050] SEP |
| mean\|clamp_v1 | -0.1806 [-0.2146, -0.1520] SEP | -0.3119 [-0.3558, -0.2718] SEP | -0.0272 [-0.0401, -0.0143] SEP |
| mean\|twosided_v2 | -0.1173 [-0.1463, -0.0934] SEP | -0.2339 [-0.2703, -0.2020] SEP | -0.0688 [-0.0861, -0.0529] SEP |
| sqrtlin_w0p3333 | -0.0491 [-0.0746, -0.0285] SEP | -0.1193 [-0.1701, -0.0773] SEP | -0.0100 [-0.0176, -0.0031] SEP |

### T6 — ⛔⛔ THE REVERSING SUBSTRATE (0.9994 of rows at r ≤ 0, median r -0.5223) — SYNTHETIC, never votes

| injection | ground-truth error (m) | mean clamp_v1 | mean twosided_v2 | paired Δ @twosided_v2 |
|---|---|---|---|---|
| base (no injection) | 39.948 | 0.000206 | 0.000206 | — |
| lon_shift(-5) | 44.899 | 0.000000 | 0.000000 | -0.000200 [-0.000500, +0.000000] n.s. |
| lon_scale(+2) | 53.540 | 0.000340 | 0.000262 | +0.000100 [-0.000000, +0.000200] n.s. |
| lon_jitter(+2) | 39.978 | 0.003107 | 0.002165 | +0.002000 [+0.000600, +0.003700] SEP |

### T7 — ⛔ THE UNDER-SIDE RANGE BUDGET, PRICED (`p` = the score a plan making zero progress receives)

| arm | p=0 | p=0.05 | p=0.1 | p=0.25 |
|---|---|---|---|---|
| ⛔ stand_still | 0.000000 | 0.050000 | 0.100000 | 0.250000 |
| v4_blind | 0.599887 | 0.617629 | 0.635371 | 0.688597 |
| cv_holdv0 | 0.940738 | 0.943701 | 0.946664 | 0.955553 |
| v1_tactical_follow | 0.908109 | 0.912686 | 0.917264 | 0.930996 |
| v1_ego_half | 0.486352 | 0.512026 | 0.537701 | 0.614724 |

### T8 — ⛔⛔ THE OVER-SIDE ZERO-MEAN CELLS, DIRECTION-VERIFIED (this is what E-2's failing cell rests on)

| cell | Δ ground-truth error (m) | admissible? |
|---|---|---|
| cv_holdv0\|lon_jitter(+2) | +0.8509 [+0.7686, +0.9328] SEP | ✅ |
| v1_tactical_follow\|lon_jitter(+2) | +0.2290 [+0.1810, +0.2796] SEP | ✅ |
| v1_ego_double\|lon_jitter(+2) | +0.0101 [-0.0258, +0.0477] n.s. | ⛔ **REFUSED** |

### T9 — the OVER-SIDE grid and ⛔ THE FAILING CELL (`v1_ego_double × lon_jitter(σ=2 m)`, zero-mean)

| term | correct | WRONG WAY | over-side floor max (scorable) | the failing cell |
|---|---|---|---|---|
| clamp_v1 | 2/12 | 9 | 0.0000 | -0.0013 [-0.0037, +0.0010] n.s. |
| ⭐ twosided_v2 | 11/12 | 1 | 0.0822 | +0.0104 [+0.0054, +0.0158] SEP |
| twosided_asym_w0p5 | 11/12 | 0 | 0.0316 | -0.0013 [-0.0039, +0.0014] n.s. |
| twosided_asym_w0p3333 | 11/12 | 0 | 0.0185 | -0.0013 [-0.0039, +0.0017] n.s. |
| twosided_asym_w2 | 11/12 | 0 | 0.1748 | +0.0030 [-0.0010, +0.0073] n.s. |
| hyp_w1 | 11/12 | 0 | 0.0000 | +0.0006 [-0.0018, +0.0029] n.s. |
| hyp_w0p5 | 11/12 | 0 | 0.0000 | -0.0003 [-0.0027, +0.0019] n.s. |
| hyp_w2 | 11/12 | 0 | 0.0000 | +0.0016 [-0.0010, +0.0041] n.s. |
| exp_w1 | 11/12 | 0 | 0.0082 | +0.0011 [-0.0015, +0.0036] n.s. |
| exp_w0p5 | 11/12 | 0 | 0.0020 | -0.0004 [-0.0029, +0.0020] n.s. |
| sqrtlin_w1 | 11/12 | 0 | 0.0822 | +0.0030 [-0.0030, +0.0090] n.s. |
| sqrtlin_w0p5 | 11/12 | 0 | 0.0315 | -0.0022 [-0.0056, +0.0013] n.s. |
| sqrtlin_w0p3333 | 11/12 | 0 | 0.0185 | -0.0016 [-0.0045, +0.0016] n.s. |
| sqrtlin_w0p25 | 9/12 | 0 | 0.0139 | -0.0015 [-0.0044, +0.0013] n.s. |
| mean\|clamp_v1 | 3/12 | 9 | 0.0000 | -0.0236 [-0.0309, -0.0169] SEP |
| mean\|twosided_v2 | 11/12 | 1 | 0.0626 | +0.0590 [+0.0497, +0.0678] SEP |
| mean\|sqrtlin_w0p3333 | 12/12 | 0 | 0.0132 | -0.0379 [-0.0494, -0.0275] SEP |
| mean\|hyp_w1 | 12/12 | 0 | 0.0000 | -0.0062 [-0.0109, -0.0018] SEP |

### T10 — the CONTAMINATION panel: max|Δ| pure-lateral vs min|Δ| longitudinal (lateral must move this axis LEAST)

| term | cv_holdv0 | v1_tactical_follow | pure? |
|---|---|---|---|
| clamp_v1 | 0.0001 vs 0.0246 (246.0×) | 0.0004 vs 0.0086 (21.5×) | ✅ |
| twosided_v2 | 0.0008 vs 0.0468 (58.5×) | 0.0010 vs 0.0037 (3.7×) | ✅ |
| twosided_asym_w0p5 | 0.0004 vs 0.0367 (91.8×) | 0.0008 vs 0.0111 (14.8×) | ✅ |
| twosided_asym_w0p3333 | 0.0003 vs 0.0333 (111.0×) | 0.0005 vs 0.0105 (21.0×) | ✅ |
| twosided_asym_w2 | 0.0015 vs 0.0584 (38.9×) | 0.0009 vs 0.0138 (15.3×) | ✅ |
| hyp_w1 | 0.0007 vs 0.0423 (60.4×) | 0.0007 vs 0.0109 (15.6×) | ✅ |
| hyp_w0p5 | 0.0004 vs 0.0349 (87.2×) | 0.0006 vs 0.0102 (17.0×) | ✅ |
| hyp_w2 | 0.0011 vs 0.0537 (48.8×) | 0.0007 vs 0.0001 (0.1×) | ⛔ **FAILS** |
| exp_w1 | 0.0007 vs 0.0442 (63.1×) | 0.0008 vs 0.0077 (9.6×) | ✅ |
| exp_w0p5 | 0.0004 vs 0.0358 (89.5×) | 0.0006 vs 0.0105 (17.5×) | ✅ |
| sqrtlin_w1 | 0.0005 vs 0.0378 (75.6×) | 0.0009 vs 0.0089 (10.0×) | ✅ |
| sqrtlin_w0p5 | 0.0003 vs 0.0320 (106.7×) | 0.0005 vs 0.0107 (21.4×) | ✅ |
| sqrtlin_w0p3333 | 0.0002 vs 0.0300 (150.0×) | 0.0004 vs 0.0100 (25.0×) | ✅ |
| sqrtlin_w0p25 | 0.0002 vs 0.0166 (83.0×) | 0.0004 vs 0.0025 (6.2×) | ✅ |
| mean\|clamp_v1 | 0.0128 vs 0.0333 (2.6×) | 0.0043 vs 0.0559 (13.0×) | ✅ |
| mean\|twosided_v2 | 0.0265 vs 0.2035 (7.7×) | 0.0082 vs 0.0280 (3.4×) | ✅ |
| mean\|sqrtlin_w0p3333 | 0.0152 vs 0.0469 (3.1×) | 0.0050 vs 0.0321 (6.4×) | ✅ |
| mean\|hyp_w1 | 0.0239 vs 0.1713 (7.2×) | 0.0063 vs 0.0735 (11.7×) | ✅ |

### T11 — ✅ THE REPRODUCTION GATE

| check | result |
|---|---|
| published `@clamp_v1` composites checked | 16 |
| **max \\|diff\\|** | **0.000000** |
| verdict | ✅ **PASS** |
| published metric id | `PSS_recovery_progress@clamp_v1` |
| default metric id | `PSS_recovery_progress@twosided_v2` |
| every new term bit-identical to `clamp_v1` for r ≤ 1 | ✅ True |

### T12 — the RANKING STATEMENT and the two INSTRUMENT GUARDS

| term | cv_holdv0 rank (realisable) | v1 tactical < every REF-C | ego_progress min live_frac | gate v2 | guard `v4_oracle − v4_blind` | guard `v1_ego_half − v1_tactical_follow` |
|---|---|---|---|---|---|---|
| clamp_v1 | 1 | ⛔ | 0.2461 | ⛔ | +0.2824 [+0.1968, +0.3736] SEP | -0.2687 [-0.2889, -0.2499] SEP |
| twosided_v2 | 1 | ✅ | 0.6704 | ✅ | +0.2992 [+0.2217, +0.3829] SEP | -0.1454 [-0.1758, -0.1149] SEP |
| twosided_asym_w0p5 | 1 | ⛔ | 0.6754 | ✅ | +0.2903 [+0.2082, +0.3766] SEP | -0.1932 [-0.2154, -0.1698] SEP |
| sqrtlin_w0p5 | 1 | ⛔ | 0.6703 | ✅ | +0.2860 [+0.2013, +0.3747] SEP | -0.2178 [-0.2370, -0.1967] SEP |
| sqrtlin_w0p3333 | 1 | ⛔ | 0.6668 | ✅ | +0.2846 [+0.1997, +0.3745] SEP | -0.2335 [-0.2513, -0.2151] SEP |
| hyp_w1 | 1 | ⛔ | 0.6796 | ✅ | +0.2957 [+0.2155, +0.3791] SEP | -0.1850 [-0.2058, -0.1637] SEP |
| mean\|clamp_v1 | 1 | ⛔ | 0.4477 | ⛔ | +0.2577 [+0.1793, +0.3412] SEP | -0.2710 [-0.2913, -0.2524] SEP |
| mean\|twosided_v2 | 1 | ✅ | 0.8858 | ✅ | +0.2882 [+0.2141, +0.3677] SEP | -0.1452 [-0.1737, -0.1166] SEP |
| mean\|sqrtlin_w0p3333 | 1 | ⛔ | 0.8707 | ✅ | +0.2642 [+0.1859, +0.3473] SEP | -0.2358 [-0.2518, -0.2198] SEP |
| mean\|hyp_w1 | 1 | ⛔ | 0.8857 | ✅ | +0.2800 [+0.2043, +0.3597] SEP | -0.1855 [-0.2048, -0.1663] SEP |

### T13 — paired contrasts that FLIP sign against `clamp_v1` (out of 10)

| term | flipped |
|---|---|
| twosided_v2 | 3 |
| twosided_asym_w0p5 | 3 |
| sqrtlin_w0p5 | 3 |
| sqrtlin_w0p3333 | 3 |
| hyp_w1 | 3 |
| mean\|clamp_v1 | 0 |
| mean\|twosided_v2 | 3 |
| mean\|sqrtlin_w0p3333 | 3 |
| mean\|hyp_w1 | 3 |

### T14 — ⚠️ `lat_heading`'s CONTROL_WEIGHTS entry, re-derived

| term | pooled sd | p05–p95 span | between-arm sd | between-arm span |
|---|---|---|---|---|
| term_lin_q0 | 0.3359 | 0.9531 | 0.0590 | 0.2594 |
| mean1_lin_q0p5 | 0.2505 | 0.8360 | 0.0715 | 0.3125 |
| **ratio shipped/published (between-arm span)** |  |  |  | **1.2047** |
| **weight preserving the published influence** |  |  |  | **0.83** (today: 1.0) |
