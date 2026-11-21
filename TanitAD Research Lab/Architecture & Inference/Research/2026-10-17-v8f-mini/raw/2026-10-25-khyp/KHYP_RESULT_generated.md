# KHYP_RESULT: PREREG_V8F_KHYP executed exactly as written

**Outcome (the PREREG table applied verbatim): HEADROOM, NO SELECTOR.** PRIMARY per seed {0: False, 1: False}; SECONDARY per seed {0: True, 1: True}; HEADROOM <= -2F per seed {0: True, 1: True}.

Controls: CONTROL 3 (deliberate regression) FAILED - read flagged.

**Committed next step, quoted from the PREREG:** *"the alternatives exist but the cheap features cannot pick them. Next: a scorer that reads the scene (a trained head; box 2)"*

Tiers: ORACLE = T0 diagnostic (reads the GT; headroom, never a result); selectors = T1-class open loop (own held outputs, no GT read at inference). Estimator: paired episode-cluster bootstrap, B=2000, seed 0, 141 episodes (taniteval/ci.py); answers 'another draw of EPISODES' only, so both seeds are shown. F = 0.1207 m (PREREG), exact |ADE6(s0)-ADE6(s1)| of B = 0.12074 m.


## Seed 0

| read | value (m) | CI | bar |
|---|---|---|---|
| HEADROOM ADE6 ORACLE - B | -0.4233 | -0.4233 [-0.4656, -0.3833] * | <= -0.2414 for the headroom outcomes |
| PRIMARY ADE6 SEL-LEARN - B | -0.0982 | -0.0982 [-0.1410, -0.0586] * | <= -0.1207 and separated |
| SECONDARY speed MAE SEL-LEARN - B | -0.0920 | -0.0920 [-0.1195, -0.0672] * | not separated-worse (worse = False) |
| SECONDARY accel MAE SEL-LEARN - B | -0.1247 | -0.1247 [-0.1464, -0.1039] * | not separated-worse (worse = False) |

| arm (ADE6, m) | level [CI] | arm - B (paired) | ADE2 arm - B | speed MAE | accel MAE | CONS(a) any | CONS(b) any |
|---|---|---|---|---|---|---|---|
| B | 2.1753 [2.0071, 2.3568] | - | - | - | - | 0.1596 | 0.2297 |
| ORACLE | 1.7520 [1.5974, 1.9152] | -0.4233 [-0.4656, -0.3833] * | -0.0188 [-0.0229, -0.0145] * | -0.1829 [-0.2077, -0.1586] * | -0.0692 [-0.0935, -0.0462] * | 0.1059 | 0.2409 |
| ARGMAX | 2.1838 [2.0144, 2.3679] | +0.0085 [-0.0022, +0.0195] | +0.0008 [+0.0000, +0.0016] * | -0.0001 [-0.0061, +0.0057] | -0.0034 [-0.0078, +0.0007] | 0.1642 | 0.2288 |
| MIX | 2.1746 [2.0064, 2.3558] | -0.0007 [-0.0035, +0.0016] | -0.0004 [-0.0006, -0.0002] * | +0.0001 [-0.0012, +0.0013] | +0.0002 [-0.0005, +0.0009] | 0.1592 | 0.2293 |
| LEARN | 2.0771 [1.9065, 2.2553] | -0.0982 [-0.1410, -0.0586] * | -0.0169 [-0.0203, -0.0135] * | -0.0920 [-0.1195, -0.0672] * | -0.1247 [-0.1464, -0.1039] * | 0.1863 | 0.2559 |
| LEARN_PERM | 2.0770 [1.9069, 2.2543] | -0.0983 [-0.1399, -0.0600] * | -0.0183 [-0.0219, -0.0148] * | -0.0942 [-0.1208, -0.0696] * | -0.1279 [-0.1477, -0.1071] * | 0.2051 | 0.2672 |
| H_B | 1.9685 [1.8106, 2.1306] | -0.2068 [-0.2771, -0.1423] * | -0.1033 [-0.1407, -0.0754] * | -0.1965 [-0.2466, -0.1506] * | -0.4644 [-0.5542, -0.3796] * | 0.2601 | 0.3697 |
| H_LEARN | 1.9403 [1.7820, 2.1069] | -0.2350 [-0.3183, -0.1573] * | -0.1036 [-0.1411, -0.0755] * | -0.2265 [-0.2845, -0.1743] * | -0.4921 [-0.5859, -0.4026] * | 0.2759 | 0.3747 |

Four families, SEL-LEARN - B (paired; lower is better except the two tactical-agreement rows):

| family | metric | SEL-LEARN - B |
|---|---|---|
| LONGITUDINAL | speed MAE | -0.0920 [-0.1195, -0.0672] * |
| LONGITUDINAL | accel MAE | -0.1247 [-0.1464, -0.1039] * |
| LONGITUDINAL | along MAE | -0.1120 [-0.1565, -0.0712] * |
| LATERAL | cross-track MAE | +0.0080 [-0.0015, +0.0176] |
| LATERAL | heading MAE (deg) | -0.2247 [-0.5642, +0.2045] |
| LATERAL | yaw-rate MAE (deg/s) | -0.4736 [-0.8047, -0.1036] * |
| LATERAL | curvature MAE (1/m) | -0.0083 [-0.0180, +0.0017] |
| TACTICAL (plan-derived) | lateral decision agreement | +0.0071 [-0.0004, +0.0150] |
| TACTICAL (plan-derived) | longitudinal decision agreement | +0.0125 [-0.0033, +0.0279] |
| STRATEGIC | - | PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25): O/T/S unchanged here, reported as paused, not as a pass |

Selector accuracy vs k*: ARGMAX 0.1434, LEARN 0.5127 (in-sample train 0.4938), LEARN_PERM 0.5035, majority-class 0.5035. k* counts (eval) [290, 208, 314, 82, 130, 1208, 167]; LEARN picks [58, 13, 169, 0, 0, 2143, 16]; ARGMAX picks [89, 25, 120, 545, 434, 534, 652].

Controls (seed 0): {"alignment": {"gt_maxabs": 7.628115213265119e-06, "limit_ms_equal": true, "lead_gap_h_equal": true, "s_h_maxabs": 0.0, "n": 2399}, "own_vs_diag_identity_maxabs_m": 0.0, "control2_reproduction": {"banked_kind_own": {"path": "C:\\Users\\Admin\\AppData\\Local\\Temp\\claude\\E--Projects-TanitAD\\bfe78918-15e8-439b-8566-593ca3297bd8\\scratchpad/hblend/inputs/banked/eval_M2b_s0-kind-own.npz", "ade6_mean_mine": 2.1753060537342375, "ade6_mean_banked": 2.175306138079546, "ade6_abs_diff_m": 8.434530851531008e-08, "traj30_maxabs_vs_banked_pred_m": 4.1961669921875e-05, "stop_bar_m": 0.001, "pass": true}, "sfresh_cycle_rerun": {"path": "C:\\Users\\Admin\\AppData\\Local\\Temp\\claude\\E--Projects-TanitAD\\bfe78918-15e8-439b-8566-593ca3297bd8\\scratchpad/sfresh/eval/eval_M2b_s0-sfresh-cycle.npz", "ade6_mean_mine": 2.1753060537342375, "ade6_mean_banked": 2.1753060638040864, "ade6_abs_diff_m": 1.0069848954685767e-08, "traj30_maxabs_vs_banked_pred_m": 3.0517578125e-05, "stop_bar_m": 0.001, "pass": true}}, "train_window_list": {"n_mine": 9582, "n_hblend": 9582, "ci": true, "t": true, "a_t": true, "a_s": true, "gt_maxabs": 0.0, "own_traj30_maxabs_vs_hblend_fit_m": 3.814697265625e-05, "pass": true}, "control1": {"n_exact_onehot_windows": 0, "max_abs_delta_m": null, "pass": null, "note": "vacuous on this data (no exactly one-hot softmax)", "harness_identity_maxabs_m": 0.0}, "control4": {"ARGMAX": {"n_violations": 0, "max_excess_m": 0.0}, "LEARN": {"n_violations": 0, "max_excess_m": 0.0}, "pass": true, "MIX_reported": {"n_windows_mix_beats_oracle": 56, "max_excess_m": 0.12814712524693617}}, "hypotheses_distinct": true, "stack_H_on_B_vs_hblend_banked_H_maxabs_m": 3.783620204700355e-05, "control3": {"perm_minus_argmax_ade6": {"point_unrounded": -0.10677097981516681, "mean": -0.1068, "lo": -0.1467, "hi": -0.0696, "separated": true}, "pass": false, "bar": "SEL-LEARN fitted on permuted labels must NOT separate from SEL-ARGMAX"}}

## Seed 1

| read | value (m) | CI | bar |
|---|---|---|---|
| HEADROOM ADE6 ORACLE - B | -0.3594 | -0.3594 [-0.4072, -0.3147] * | <= -0.2414 for the headroom outcomes |
| PRIMARY ADE6 SEL-LEARN - B | +0.0204 | +0.0204 [-0.0059, +0.0467] | <= -0.1207 and separated |
| SECONDARY speed MAE SEL-LEARN - B | +0.0037 | +0.0037 [-0.0125, +0.0198] | not separated-worse (worse = False) |
| SECONDARY accel MAE SEL-LEARN - B | +0.0009 | +0.0009 [-0.0165, +0.0183] | not separated-worse (worse = False) |

| arm (ADE6, m) | level [CI] | arm - B (paired) | ADE2 arm - B | speed MAE | accel MAE | CONS(a) any | CONS(b) any |
|---|---|---|---|---|---|---|---|
| B | 2.0546 [1.8859, 2.2323] | - | - | - | - | 0.2097 | 0.1430 |
| ORACLE | 1.6952 [1.5463, 1.8524] | -0.3594 [-0.4072, -0.3147] * | -0.0188 [-0.0250, -0.0132] * | -0.1232 [-0.1459, -0.1021] * | -0.0102 [-0.0233, +0.0027] | 0.1517 | 0.1421 |
| ARGMAX | 2.0637 [1.8942, 2.2415] | +0.0092 [+0.0020, +0.0165] * | +0.0011 [+0.0005, +0.0017] * | +0.0016 [-0.0020, +0.0054] | -0.0005 [-0.0024, +0.0013] | 0.2147 | 0.1430 |
| MIX | 2.0554 [1.8864, 2.2328] | +0.0008 [-0.0007, +0.0025] | +0.0000 [-0.0001, +0.0002] | +0.0002 [-0.0005, +0.0008] | -0.0002 [-0.0008, +0.0003] | 0.2130 | 0.1421 |
| LEARN | 2.0750 [1.9101, 2.2512] | +0.0204 [-0.0059, +0.0467] | -0.0066 [-0.0139, -0.0001] * | +0.0037 [-0.0125, +0.0198] | +0.0009 [-0.0165, +0.0183] | 0.2051 | 0.1376 |
| LEARN_PERM | 2.0653 [1.8955, 2.2452] | +0.0107 [-0.0029, +0.0244] | -0.0122 [-0.0153, -0.0093] * | -0.0180 [-0.0274, -0.0092] * | -0.0429 [-0.0510, -0.0350] * | 0.2393 | 0.1530 |
| H_B | 1.9848 [1.8199, 2.1587] | -0.0697 [-0.1011, -0.0419] * | -0.0777 [-0.1148, -0.0525] * | -0.0723 [-0.1000, -0.0507] * | -0.2277 [-0.2894, -0.1740] * | 0.2851 | 0.2172 |
| H_LEARN | 1.9993 [1.8373, 2.1673] | -0.0552 [-0.0915, -0.0236] * | -0.0781 [-0.1159, -0.0525] * | -0.0700 [-0.1014, -0.0447] * | -0.2176 [-0.2814, -0.1615] * | 0.2818 | 0.2093 |

Four families, SEL-LEARN - B (paired; lower is better except the two tactical-agreement rows):

| family | metric | SEL-LEARN - B |
|---|---|---|
| LONGITUDINAL | speed MAE | +0.0037 [-0.0125, +0.0198] |
| LONGITUDINAL | accel MAE | +0.0009 [-0.0165, +0.0183] |
| LONGITUDINAL | along MAE | +0.0038 [-0.0210, +0.0279] |
| LATERAL | cross-track MAE | +0.0268 [+0.0132, +0.0416] * |
| LATERAL | heading MAE (deg) | +0.2031 [+0.1085, +0.3161] * |
| LATERAL | yaw-rate MAE (deg/s) | +0.2010 [-0.0031, +0.4669] |
| LATERAL | curvature MAE (1/m) | +0.0036 [-0.0014, +0.0099] |
| TACTICAL (plan-derived) | lateral decision agreement | +0.0083 [-0.0017, +0.0209] |
| TACTICAL (plan-derived) | longitudinal decision agreement | -0.0104 [-0.0271, +0.0054] |
| STRATEGIC | - | PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25): O/T/S unchanged here, reported as paused, not as a pass |

Selector accuracy vs k*: ARGMAX 0.1409, LEARN 0.3581 (in-sample train 0.4186), LEARN_PERM 0.3543, majority-class 0.3556. k* counts (eval) [442, 272, 382, 56, 173, 853, 221]; LEARN picks [364, 45, 463, 1, 19, 1442, 65]; ARGMAX picks [22, 12, 52, 593, 351, 562, 807].

Controls (seed 1): {"alignment": {"gt_maxabs": 7.628115213265119e-06, "limit_ms_equal": true, "lead_gap_h_equal": true, "s_h_maxabs": 0.0, "n": 2399}, "own_vs_diag_identity_maxabs_m": 0.0, "control2_reproduction": {"banked_kind_own": {"path": "C:\\Users\\Admin\\AppData\\Local\\Temp\\claude\\E--Projects-TanitAD\\bfe78918-15e8-439b-8566-593ca3297bd8\\scratchpad/hblend/inputs/banked/eval_M2b_s1-kind-own.npz", "ade6_mean_mine": 2.0545660204036604, "ade6_mean_banked": 2.0545660167647672, "ade6_abs_diff_m": 3.6388931867747942e-09, "traj30_maxabs_vs_banked_pred_m": 9.918212890625e-05, "stop_bar_m": 0.001, "pass": true}, "sfresh_cycle_rerun": {"path": "C:\\Users\\Admin\\AppData\\Local\\Temp\\claude\\E--Projects-TanitAD\\bfe78918-15e8-439b-8566-593ca3297bd8\\scratchpad/sfresh/eval/eval_M2b_s1-sfresh-cycle.npz", "ade6_mean_mine": 2.0545660204036604, "ade6_mean_banked": 2.054566010448041, "ade6_abs_diff_m": 9.955619439949714e-09, "traj30_maxabs_vs_banked_pred_m": 3.0517578125e-05, "stop_bar_m": 0.001, "pass": true}}, "train_window_list": {"n_mine": 9582, "n_hblend": 9582, "ci": true, "t": true, "a_t": true, "a_s": true, "gt_maxabs": 0.0, "own_traj30_maxabs_vs_hblend_fit_m": 3.0517578125e-05, "pass": true}, "control1": {"n_exact_onehot_windows": 0, "max_abs_delta_m": null, "pass": null, "note": "vacuous on this data (no exactly one-hot softmax)", "harness_identity_maxabs_m": 0.0}, "control4": {"ARGMAX": {"n_violations": 0, "max_excess_m": 0.0}, "LEARN": {"n_violations": 0, "max_excess_m": 0.0}, "pass": true, "MIX_reported": {"n_windows_mix_beats_oracle": 30, "max_excess_m": 0.06131967016199502}}, "hypotheses_distinct": true, "stack_H_on_B_vs_hblend_banked_H_maxabs_m": 8.685091061977346e-05, "control3": {"perm_minus_argmax_ade6": {"point_unrounded": 0.0015312171954646736, "mean": 0.0015, "lo": -0.0111, "hi": 0.0145, "separated": false}, "pass": true, "bar": "SEL-LEARN fitted on permuted labels must NOT separate from SEL-ARGMAX"}}

Distance keeping proper (headway / time-gap / min-TTC): {"status": "ERROR", "error": "Refusal: B_s0: ||pred - gt|| differs from its saved ade6 by 1.35e-05 m (> 1e-05): the rebuilt ground truth is not this run's grid / frame / units"}
