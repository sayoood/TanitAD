# NUMBERS_V46: numbers of sections 7.51 and 7.52 and of edits_v46.md, each re-read from its source by `check_numbers_v46.py`

Rewritten on every run. Rows: check, read from source, quoted in the text, result, source.

## UNSOURCED

None: every numeric token of both sections is held by a scoped source, a recomputation or is structural.

| # | check | read from source | quoted in the text | result | source |
|---|---|---|---|---|---|
| 1 | derived 0.29 recomputed | True | True | PASS | arithmetic [E] |
| 2 | derived 35 recomputed | True | True | PASS | arithmetic [E] |
| 3 | derived 45 recomputed | True | True | PASS | arithmetic [E] |
| 4 | derived -0.1187 recomputed | True | True | PASS | arithmetic [E] |
| 5 | derived -0.0114 recomputed | True | True | PASS | arithmetic [E] |
| 6 | derived 6 recomputed | True | True | PASS | arithmetic [E] |
| 7 | band loss gain is 0.19 to 0.27 of the acceleration term's operative ADE gain (text: a fifth to a quarter) | (0.271, 0.189) | (0.271, 0.189) | PASS | arithmetic [E]; 0.27 is slightly above a quarter |
| 8 | tactical replicate spread is 4.8 to 12.0 times the band loss's tactical ADE effects (text: five to twelve) | (4.8, 12.0) | (4.8, 12.0) | PASS | arithmetic [E] |
| 9 | sec 7.51: every numeric token (141) is in the scoped sources, derived, or structural; unsourced: [] | 0 | 0 | PASS | scoped corpus 751 |
| 10 | sec 7.52: every numeric token (193) is in the scoped sources, derived, or structural; unsourced: [] | 0 | 0 | PASS | scoped corpus 752 |
| 11 | sec 7.51: 23 interval phrases and pair members found verbatim in the scoped sources (not found: []) | 0 | 0 | PASS | phrase audit; the listed rounded cells are recomputed from JSON below |
| 12 | sec 7.52: 41 interval phrases and pair members found verbatim in the scoped sources (not found: []) | 0 | 0 | PASS | phrase audit; the listed rounded cells are recomputed from JSON below |
| 13 | OB-CTL operative ADE s0 | 0.2571 | 0.2571 | PASS | READ_FULL.json |
| 14 | OB-CTL operative ADE s1 | 0.2573 | 0.2573 | PASS | READ_FULL.json |
| 15 | replicate floor ADE | 0.0074 | 0.0074 | PASS | FLOOR_TABLE.json |
| 16 | replicate floor FDE | 0.0008 | 0.0008 | PASS | FLOOR_TABLE.json |
| 17 | replicate floor along | 0.0088 | 0.0088 | PASS | FLOOR_TABLE.json |
| 18 | replicate floor speed | 0.0390 | 0.0390 | PASS | FLOOR_TABLE.json |
| 19 | replicate floor heading | 0.0176 | 0.0176 | PASS | FLOOR_TABLE.json |
| 20 | Finding 1 ADE (m) s0 delta | -0.0168 | -0.0168 | PASS | FLOOR_TABLE.json BAND-CTL |
| 21 | Finding 1 ADE (m) s1 delta | -0.0149 | -0.0149 | PASS | FLOOR_TABLE.json BAND-CTL |
| 22 | Finding 1 FDE at 2 s (m) s0 delta | -0.0393 | -0.0393 | PASS | FLOOR_TABLE.json BAND-CTL |
| 23 | Finding 1 FDE at 2 s (m) s1 delta | -0.0267 | -0.0267 | PASS | FLOOR_TABLE.json BAND-CTL |
| 24 | Finding 1 along-track MAE (m) s0 delta | -0.0173 | -0.0173 | PASS | FLOOR_TABLE.json BAND-CTL |
| 25 | Finding 1 along-track MAE (m) s1 delta | -0.0157 | -0.0157 | PASS | FLOOR_TABLE.json BAND-CTL |
| 26 | Finding 1 speed MAE (m/s) s0 delta | -0.0184 | -0.0184 | PASS | FLOOR_TABLE.json BAND-CTL |
| 27 | Finding 1 speed MAE (m/s) s1 delta | -0.0258 | -0.0258 | PASS | FLOOR_TABLE.json BAND-CTL |
| 28 | Finding 1 heading MAE (deg) s0 delta | +0.0311 | +0.0311 | PASS | FLOOR_TABLE.json BAND-CTL |
| 29 | Finding 1 heading MAE (deg) s1 delta | +0.0313 | +0.0313 | PASS | FLOOR_TABLE.json BAND-CTL |
| 30 | Finding 2 tactical ADE (m) / step1 schedule: value and bold mark | ('-0.187 / +0.032', False) | ('-0.187 / +0.032', False) | PASS | FLOOR_TABLE.json |
| 31 | Finding 2 tactical ADE (m) / step2 DAgger rows: value and bold mark | ('-0.819 / -0.615', True) | ('-0.819 / -0.615', True) | PASS | FLOOR_TABLE.json |
| 32 | Finding 2 tactical ADE (m) / step3 w_acc: value and bold mark | ('-0.022 / -0.165', False) | ('-0.022 / -0.165', False) | PASS | FLOOR_TABLE.json |
| 33 | Finding 2 tactical heading MAE (deg) / step1 schedule: value and bold mark | ('-0.49 / +0.31', False) | ('-0.49 / +0.31', False) | PASS | FLOOR_TABLE.json |
| 34 | Finding 2 tactical heading MAE (deg) / step2 DAgger rows: value and bold mark | ('-4.63 / -2.57', True) | ('-4.63 / -2.57', True) | PASS | FLOOR_TABLE.json |
| 35 | Finding 2 tactical heading MAE (deg) / step3 w_acc: value and bold mark | ('+0.97 / -0.74', False) | ('+0.97 / -0.74', False) | PASS | FLOOR_TABLE.json |
| 36 | Finding 2 tactical cross-track MAE (m) / step1 schedule: value and bold mark | ('-0.02 / -0.11', False) | ('-0.02 / -0.11', False) | PASS | FLOOR_TABLE.json |
| 37 | Finding 2 tactical cross-track MAE (m) / step2 DAgger rows: value and bold mark | ('-0.44 / -0.46', True) | ('-0.44 / -0.46', True) | PASS | FLOOR_TABLE.json |
| 38 | Finding 2 tactical cross-track MAE (m) / step3 w_acc: value and bold mark | ('+0.03 / -0.07', False) | ('+0.03 / -0.07', False) | PASS | FLOOR_TABLE.json |
| 39 | Finding 2 operative ADE (m) / step1 schedule: value and bold mark | ('-0.022 / +0.006', False) | ('-0.022 / +0.006', False) | PASS | FLOOR_TABLE.json |
| 40 | Finding 2 operative ADE (m) / step2 DAgger rows: value and bold mark | ('+0.007 / +0.016', False) | ('+0.007 / +0.016', False) | PASS | FLOOR_TABLE.json |
| 41 | Finding 2 operative ADE (m) / step3 w_acc: value and bold mark | ('-0.062 / -0.079', True) | ('-0.062 / -0.079', True) | PASS | FLOOR_TABLE.json |
| 42 | Finding 2 operative acceleration MAE (m/s^2) / step1 schedule: value and bold mark | ('-0.37 / +0.19', False) | ('-0.37 / +0.19', False) | PASS | FLOOR_TABLE.json |
| 43 | Finding 2 operative acceleration MAE (m/s^2) / step2 DAgger rows: value and bold mark | ('+0.01 / +0.09', False) | ('+0.01 / +0.09', False) | PASS | FLOOR_TABLE.json |
| 44 | Finding 2 operative acceleration MAE (m/s^2) / step3 w_acc: value and bold mark | ('-0.97 / -1.14', True) | ('-0.97 / -1.14', True) | PASS | FLOOR_TABLE.json |
| 45 | Finding 2 tactical acceleration MAE (m/s^2) / step1 schedule: value and bold mark | ('-0.17 / -0.05', False) | ('-0.17 / -0.05', False) | PASS | FLOOR_TABLE.json |
| 46 | Finding 2 tactical acceleration MAE (m/s^2) / step2 DAgger rows: value and bold mark | ('-0.05 / +0.64', False) | ('-0.05 / +0.64', False) | PASS | FLOOR_TABLE.json |
| 47 | Finding 2 tactical acceleration MAE (m/s^2) / step3 w_acc: value and bold mark | ('-2.01 / -2.56', True) | ('-2.01 / -2.56', True) | PASS | FLOOR_TABLE.json |
| 48 | Finding 2 TAC-LAT row cells equal the READ result table (+0.011 / -0.045, +0.100 / +0.192, -0.002 / +0.012) | 1 | 1 | PASS | RESULT_FULL.md |
| 49 | DAG-ORC pair numbers present in DAG_FLOOR.json text | 1 | 1 | PASS | DAG_FLOOR.json / RESULT_FULL.md |
| 50 | top-3 numbers (-0.3645, -0.3326, 0.572, 0.585, 0.895, 0.901) present in TOPK_FULL.json or RESULT_FULL.md | 1 | 1 | PASS | TOPK_FULL.json |
| 51 | outcome BAND-COST at both seeds; STRATEGIC PAUSED; card prefix fd81c66e | True | True | PASS | RESULT_FULL.md |
| 52 | ladder C0 level 1 skill s0 / s1 | 0.064 / 0.080 | 0.064 / 0.080 | PASS | RESULT_LADDER_READ.json |
| 53 | ladder C0 level 2 skill s0 / s1 | 0.278 / 0.267 | 0.278 / 0.267 | PASS | RESULT_LADDER_READ.json |
| 54 | ladder C0 level 3 skill s0 / s1 | 0.359 / 0.401 | 0.359 / 0.401 | PASS | RESULT_LADDER_READ.json |
| 55 | ladder S35 level 1 skill s0 / s1 | 0.069 / 0.096 | 0.069 / 0.096 | PASS | RESULT_LADDER_READ.json |
| 56 | ladder S35 level 2 skill s0 / s1 | 0.294 / 0.339 | 0.294 / 0.339 | PASS | RESULT_LADDER_READ.json |
| 57 | ladder S35 level 3 skill s0 / s1 | 0.472 / 0.480 | 0.472 / 0.480 | PASS | RESULT_LADDER_READ.json |
| 58 | ladder R35 level 1 skill s0 / s1 | 0.178 / 0.190 | 0.178 / 0.190 | PASS | RESULT_LADDER_READ.json |
| 59 | ladder R35 level 2 skill s0 / s1 | 0.419 / 0.430 | 0.419 / 0.430 | PASS | RESULT_LADDER_READ.json |
| 60 | ladder R35 level 3 skill s0 / s1 | 0.535 / 0.541 | 0.535 / 0.541 | PASS | RESULT_LADDER_READ.json |
| 61 | ladder R35H level 1 skill s0 / s1 | 0.259 / 0.267 | 0.259 / 0.267 | PASS | RESULT_LADDER_READ.json |
| 62 | ladder R35H level 2 skill s0 / s1 | 0.546 / 0.555 | 0.546 / 0.555 | PASS | RESULT_LADDER_READ.json |
| 63 | ladder R35H level 3 skill s0 / s1 | 0.672 / 0.687 | 0.672 / 0.687 | PASS | RESULT_LADDER_READ.json |
| 64 | every arm BAND-MET | ['BAND-MET', 'BAND-MET', 'BAND-MET', 'BAND-MET', 'BAND-MET'] | ['BAND-MET', 'BAND-MET', 'BAND-MET', 'BAND-MET', 'BAND-MET'] | PASS | RESULT_LADDER_READ.json |
| 65 | largest seed-to-seed spread per level (0.027, 0.045, 0.042) | ['0.027', '0.045', '0.042'] | ['0.027', '0.045', '0.042'] | PASS | RESULT_LADDER_READ.json |
| 66 | k-step +0.091 to +0.125 at levels 1-2 | ('0.091', '0.125') | ('0.091', '0.125') | PASS | RESULT_LADDER_READ.json |
| 67 | k-step +0.061 to +0.063 at level 3 | ('0.061', '0.063') | ('0.061', '0.063') | PASS | RESULT_LADDER_READ.json |
| 68 | multi-rate +0.077 to +0.146 | ('0.077', '0.146') | ('0.077', '0.146') | PASS | RESULT_LADDER_READ.json |
| 69 | SHRINK vs Copy-Last ranges 0.05-0.08 / 0.14-0.20 / 0.18-0.26 | {'L1': (0.05, 0.08), 'L2': (0.14, 0.2), 'L3': (0.18, 0.26)} | {'L1': (0.05, 0.08), 'L2': (0.14, 0.2), 'L3': (0.18, 0.26)} | PASS | DIAG_SHRINK.md |
| 70 | R35b identity counts 276 / 399 and the control 252 differ | True | True | PASS | R35B_CKPT_IDENTITY.json |
| 71 | PLANTEST2 FLAT1 1.0430 / 1.0033 and HOLD 0.3348 in the result file | True | True | PASS | RESULT_PLANTEST2.md |
| 72 | R^2 0.778 / 0.702 held-out speed (best of four readers) in RESULT_FULLN.md and RESULT_PT3F.md | (True, True) | (True, True) | PASS | RESULT_FULLN.md row 'speed R0 s1', RESULT_PT3F.md |
| 73 | edits: every numeric token is in a section or the v45 base text; not found: [] | 0 | 0 | PASS | edits_v46.md |
| 74 | edits: 17 interval phrases occur verbatim in a section (not found: []) | 0 | 0 | PASS | edits_v46.md |
| 75 | edits: pair members occur in a section (not found: []) | 0 | 0 | PASS | edits_v46.md |
| 76 | banned words in sec_751 | [] | [] | PASS |  |
| 77 | em or en dash in sec_751 | False | False | PASS |  |
| 78 | 'drives' / 'driving performance' in sec_751 | False | False | PASS |  |
| 79 | no CR in sec_751 | False | False | PASS |  |
| 80 | banned words in sec_752 | [] | [] | PASS |  |
| 81 | em or en dash in sec_752 | False | False | PASS |  |
| 82 | 'drives' / 'driving performance' in sec_752 | False | False | PASS |  |
| 83 | no CR in sec_752 | False | False | PASS |  |
| 84 | banned words in edits_v46.md | [] | [] | PASS |  |
| 85 | em or en dash in edits_v46.md | False | False | PASS |  |
| 86 | 'drives' / 'driving performance' in edits_v46.md | False | False | PASS |  |
| 87 | no CR in edits_v46.md | False | False | PASS |  |
| 88 | no learning-curve exponent quoted | False | False | PASS |  |
| 89 | STRATEGIC reported as PAUSED in both sections, never as a pass | True | True | PASS |  |
| 90 | tiers named: T0 in 7.51, T1-kin in 7.52 | (True, True) | (True, True) | PASS |  |
| 91 | mutation arm: -0.0169 is NOT in the OTBAND sources while -0.0168 is | True | True | PASS |  |
| 92 | mutation arm: 0.2162 is NOT a printed value in the 7.51 sources while 0.2161 is | True | True | PASS |  |

## Derived numbers

| number | derivation | class |
|---|---|---|
| 35 % and 45 % (ORACLE-END share of the HOLD error) | (0.3348-0.2161)/0.3348 and (0.0256-0.0142)/0.0256, recomputed above | [E] |
| 0.29 m (tactical replicate spread of the band recipe) | 0.2869 rounded | [E] |
| about 6 % (band loss gain over OB-CTL operative ADE) | 0.0168/0.2571 and 0.0149/0.2573 | [E] |
| a fifth to a quarter; five to twelve times | ratios 0.19 to 0.27 and 4.8 to 11.9, recomputed above | [E] |
| 0.0390 (speed floor) | FLOOR_TABLE.json prints 0.039 | [M] |
| seed spreads, differences and shrink ranges of 7.51 | recomputed from RESULT_LADDER_READ.json / DIAG_SHRINK.md; the text itself says two seeds, a rough floor, post-hoc | [M/E] |
