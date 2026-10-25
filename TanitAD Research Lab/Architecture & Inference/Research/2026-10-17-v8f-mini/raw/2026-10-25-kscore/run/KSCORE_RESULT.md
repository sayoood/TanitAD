# KSCORE_RESULT: PREREG_V8F_KSCORE executed exactly as written

**Outcome (the PREREG table applied verbatim): NOT PREDICTABLE**. PRIMARY-B per seed {0: False, 1: False}; PRIMARY-SEL per seed {0: False, 1: False}; SECONDARY (speed/accel) per seed {0: True, 1: True}; SECONDARY (CONSISTENCY, LEAD-VIOL) per seed {0: False, 1: True}.

**Committed next step, quoted from the PREREG:** *"the best hypothesis is not predictable from these inputs at this scale. The longitudinal lever moves to O's speed profile (OACC's line)"*

Tiers: ORACLE = T0 diagnostic (reads the GT; headroom, never a result); selectors = T1-class open loop (own held outputs, no GT read at inference). Estimator: paired episode-cluster bootstrap, B=2000, seed 0, 141 episodes (taniteval/ci.py); answers 'another draw of EPISODES' only, so both seeds are shown. F = 0.1207 m (PREREG), exact |ADE6(s0)-ADE6(s1)| of B = 0.12074 m.


## Seed 0

| read | value (m) | CI | bar |
|---|---|---|---|
| PRIMARY-B ADE6 KSCORE - B | -0.1001 | -0.1001 [-0.1463, -0.0583] * | <= -0.1207 and separated |
| PRIMARY-SEL ADE6 KSCORE - SEL-CONST | -0.0005 | -0.0005 [-0.0188, +0.0170] | < 0 and separated |
| SECONDARY speed MAE KSCORE - B | -0.0820 | -0.0820 [-0.1103, -0.0552] * | not separated-worse (worse = False) |
| SECONDARY accel MAE KSCORE - B | -0.1193 | -0.1193 [-0.1418, -0.0976] * | not separated-worse (worse = False) |
| SECONDARY CONSISTENCY(a) 'any' KSCORE - B | +0.0246 | +0.0246 [+0.0083, +0.0400] * | not separated-worse (worse = True) |
| SECONDARY LEAD-VIOL (at-risk share) KSCORE - B | 0.0084 | +0.0084 [+0.0016, +0.0176] * | not separated-worse (worse = True) |

| arm (ADE6, m) | level [CI] | arm - B (paired) | arm - SEL-CONST (paired) |
|---|---|---|---|
| B | 2.1753 [2.0071, 2.3568] | - | +0.0996 [+0.0621, +0.1412] * |
| ARGMAX | 2.1838 [2.0144, 2.3679] | +0.0085 [-0.0022, +0.0195] | +0.1081 [+0.0718, +0.1478] * |
| CONST | 2.0757 [1.9087, 2.2540] | -0.0996 [-0.1412, -0.0621] * | - |
| KSCORE | 2.0752 [1.9074, 2.2552] | -0.1001 [-0.1463, -0.0583] * | -0.0005 [-0.0188, +0.0170] |
| PERM | 2.0757 [1.9087, 2.2540] | -0.0996 [-0.1412, -0.0621] * | +0.0000 [+0.0000, +0.0000] |
| FORCED | 2.1838 [2.0144, 2.3679] | - | - |

Four families, KSCORE - B (paired; lower is better except the two tactical-agreement rows):

| family | metric | KSCORE - B |
|---|---|---|
| LONGITUDINAL | speed MAE | -0.0820 [-0.1103, -0.0552] * |
| LONGITUDINAL | accel MAE | -0.1193 [-0.1418, -0.0976] * |
| LONGITUDINAL | along MAE | -0.1139 [-0.1606, -0.0717] * |
| LATERAL | cross-track MAE | +0.0062 [-0.0070, +0.0196] |
| LATERAL | heading MAE (deg) | -0.4929 [-0.7815, -0.2503] * |
| LATERAL | yaw-rate MAE (deg/s) | -0.6098 [-0.9727, -0.3215] * |
| LATERAL | curvature MAE (1/m) | -0.0125 [-0.0239, -0.0041] * |
| TACTICAL (plan-derived) | lateral decision agreement | +0.0050 [-0.0025, +0.0133] |
| TACTICAL (plan-derived) | longitudinal decision agreement | +0.0067 [-0.0096, +0.0233] |
| STRATEGIC | - | PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25): O/T/S unchanged here, reported as paused, not as a pass |

Selector accuracy vs k*: KSCORE 0.5206, PERM 0.5035, ARGMAX 0.1434, SEL-CONST (class 5) 0.5035, majority train class 5 0.5035. k* counts (eval) [290, 208, 314, 82, 130, 1208, 167]; KSCORE picks [84, 5, 232, 0, 0, 2078, 0].

Controls (seed 0): {"alignment": {"gt_maxabs": 7.628115213265119e-06, "n": 2399, "n_expected": 2399}, "harness_identity_maxabs_m": 0.0, "harness_identity_tol_m": 0.0001, "control3_B_reproduces_banked": {"ade6_mean_mine": 2.175306120638704, "ade6_banked_4dp": 2.1753, "pass": true}, "control4_oracle_le_vertex_selectors": {"n_violations": {"ARGMAX": 0, "CONST": 0, "KSCORE": 0, "PERM": 0, "FORCED": 0}, "pass": true}, "hypotheses_distinct": true, "control1_forced_is_sel_argmax": {"picks_equal": true, "plans_bit_equal": true, "paired_ade6_forced_minus_argmax": {"mean": 0.0, "lo": 0.0, "hi": 0.0, "separated": false, "lower_is_better": true, "better": false, "worse": false, "n_windows": 2399, "n_episodes": 141, "estimator": "paired_episode_cluster_bootstrap", "point_unrounded": 0.0}, "pass": true}, "control2_perm_scorer_vs_sel_const": {"perm_minus_const_ade6": {"point_unrounded": 0.0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "separated": false}, "pass": true, "bar": "a scorer trained on permuted k* must NOT separate from SEL-CONST"}}

Stack: H {"status": "NOT-RUN", "reason": "no --h-weights given (descriptive row)"}; HS NOT-INSTRUMENTED (HS is PREREG_V8F_HSPEC's coefficient-shrunk operator; this harness imports no HSPEC code. Descriptive row only: a work item, not a pass.)

## Seed 1

| read | value (m) | CI | bar |
|---|---|---|---|
| PRIMARY-B ADE6 KSCORE - B | +0.0036 | +0.0036 [-0.0273, +0.0318] | <= -0.1207 and separated |
| PRIMARY-SEL ADE6 KSCORE - SEL-CONST | -0.0072 | -0.0072 [-0.0419, +0.0238] | < 0 and separated |
| SECONDARY speed MAE KSCORE - B | -0.0046 | -0.0046 [-0.0202, +0.0115] | not separated-worse (worse = False) |
| SECONDARY accel MAE KSCORE - B | -0.0235 | -0.0235 [-0.0415, -0.0073] * | not separated-worse (worse = False) |
| SECONDARY CONSISTENCY(a) 'any' KSCORE - B | -0.0029 | -0.0029 [-0.0146, +0.0079] | not separated-worse (worse = False) |
| SECONDARY LEAD-VIOL (at-risk share) KSCORE - B | 0.005 | +0.0050 [-0.0018, +0.0129] | not separated-worse (worse = False) |

| arm (ADE6, m) | level [CI] | arm - B (paired) | arm - SEL-CONST (paired) |
|---|---|---|---|
| B | 2.0546 [1.8859, 2.2323] | - | -0.0108 [-0.0245, +0.0030] |
| ARGMAX | 2.0637 [1.8942, 2.2415] | +0.0092 [+0.0020, +0.0165] * | -0.0016 [-0.0141, +0.0111] |
| CONST | 2.0653 [1.8957, 2.2463] | +0.0108 [-0.0030, +0.0245] | - |
| KSCORE | 2.0581 [1.8940, 2.2357] | +0.0036 [-0.0273, +0.0318] | -0.0072 [-0.0419, +0.0238] |
| PERM | 2.0653 [1.8957, 2.2463] | +0.0108 [-0.0030, +0.0245] | +0.0000 [+0.0000, +0.0000] |
| FORCED | 2.0637 [1.8942, 2.2415] | - | - |

Four families, KSCORE - B (paired; lower is better except the two tactical-agreement rows):

| family | metric | KSCORE - B |
|---|---|---|
| LONGITUDINAL | speed MAE | -0.0046 [-0.0202, +0.0115] |
| LONGITUDINAL | accel MAE | -0.0235 [-0.0415, -0.0073] * |
| LONGITUDINAL | along MAE | +0.0066 [-0.0238, +0.0351] |
| LATERAL | cross-track MAE | -0.0003 [-0.0139, +0.0144] |
| LATERAL | heading MAE (deg) | +0.2210 [+0.0610, +0.4188] * |
| LATERAL | yaw-rate MAE (deg/s) | +0.2770 [-0.0610, +0.7801] |
| LATERAL | curvature MAE (1/m) | +0.0053 [-0.0025, +0.0172] |
| TACTICAL (plan-derived) | lateral decision agreement | +0.0104 [+0.0000, +0.0233] |
| TACTICAL (plan-derived) | longitudinal decision agreement | -0.0054 [-0.0196, +0.0083] |
| STRATEGIC | - | PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25): O/T/S unchanged here, reported as paused, not as a pass |

Selector accuracy vs k*: KSCORE 0.3931, PERM 0.3556, ARGMAX 0.1409, SEL-CONST (class 5) 0.3556, majority train class 5 0.3556. k* counts (eval) [442, 272, 382, 56, 173, 853, 221]; KSCORE picks [529, 52, 293, 0, 60, 1369, 96].

Controls (seed 1): {"alignment": {"gt_maxabs": 7.628115213265119e-06, "n": 2399, "n_expected": 2399}, "harness_identity_maxabs_m": 0.0, "harness_identity_tol_m": 0.0001, "control3_B_reproduces_banked": {"ade6_mean_mine": 2.0545660143297426, "ade6_banked_4dp": 2.0546, "pass": true}, "control4_oracle_le_vertex_selectors": {"n_violations": {"ARGMAX": 0, "CONST": 0, "KSCORE": 0, "PERM": 0, "FORCED": 0}, "pass": true}, "hypotheses_distinct": true, "control1_forced_is_sel_argmax": {"picks_equal": true, "plans_bit_equal": true, "paired_ade6_forced_minus_argmax": {"mean": 0.0, "lo": 0.0, "hi": 0.0, "separated": false, "lower_is_better": true, "better": false, "worse": false, "n_windows": 2399, "n_episodes": 141, "estimator": "paired_episode_cluster_bootstrap", "point_unrounded": 0.0}, "pass": true}, "control2_perm_scorer_vs_sel_const": {"perm_minus_const_ade6": {"point_unrounded": 0.0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "separated": false}, "pass": true, "bar": "a scorer trained on permuted k* must NOT separate from SEL-CONST"}}

Stack: H {"status": "NOT-RUN", "reason": "no --h-weights given (descriptive row)"}; HS NOT-INSTRUMENTED (HS is PREREG_V8F_HSPEC's coefficient-shrunk operator; this harness imports no HSPEC code. Descriptive row only: a work item, not a pass.)
