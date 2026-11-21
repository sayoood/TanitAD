# KHYP_RESULT: PREREG_V8F_KHYP, Phase B executed (zero training, CPU, banked M2b s0 / s1)

## OUTCOME (the PREREG outcome table applied verbatim): **HEADROOM, NO SELECTOR**

| | seed 0 | seed 1 | bar |
|---|---|---|---|
| **HEADROOM** ADE6 ORACLE - B (T0 diagnostic, never a result) | **-0.4233** [-0.4656, -0.3833] * | **-0.3594** [-0.4072, -0.3147] * | <= -2F = -0.2414 at BOTH seeds: **MET** (ORACLE closes 19.5 % / 17.5 % of B's ADE6 of 2.1753 / 2.0546 m; it is 1.75x / 1.49x the 2F bar) |
| **PRIMARY** ADE6 SEL-LEARN - B (T1-class open loop) | -0.0982 [-0.1410, -0.0586] * | +0.0204 [-0.0059, +0.0467] | <= -F = -0.1207 AND separated at BOTH seeds: **FAILS at both** (s0 separated but 0.0225 m short of the bar; s1 not separated and the wrong sign) |
| SECONDARY speed MAE SEL-LEARN - B | -0.0920 [-0.1195, -0.0672] * | +0.0037 [-0.0125, +0.0198] | not separated-worse: **met** at both |
| SECONDARY accel MAE SEL-LEARN - B | -0.1247 [-0.1464, -0.1039] * | +0.0009 [-0.0165, +0.0183] | not separated-worse: **met** at both |

Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes (`taniteval/ci.py`), the eval's own `evaluate()` / `families()`; it answers "would another draw of EPISODES say this?" only, so both seeds ride side by side with F (0.1207 m PREREG; exact |ADE6(s0) - ADE6(s1)| of B = 0.12074 m). Tiers: ORACLE = T0 diagnostic; selectors = T1-class open loop. Evidence class: MEASURED (ours), artifacts below.

**Committed next step, quoted from the PREREG:** *"the alternatives exist but the cheap features cannot pick them. Next: a scorer that reads the scene (a trained head; box 2)"*

## What the outcome means, and what the controls say about it

* The lever is real in size: a perfect choice among the seven one-hot longitudinal plans is worth 0.36-0.42 m of ADE6 (T0), separated, at both seeds. The cheap selector does not capture it.
* **CONTROL 3 (the deliberate regression) FAILED at seed 0**; the table is printed verbatim and flagged, as the PREREG and the handoff require. SEL-LEARN fitted on PERMUTED labels gives ADE6 - B = -0.0983 against SEL-LEARN's -0.0982; SEL-LEARN - SEL-LEARN_PERM = +0.0000 [-0.0112, +0.0112]. The permuted fit collapses to the majority class, and so does the real fit (it picks class 5 on 2,143 / 2,399 = 89.3 % of eval windows; accuracy 0.5127 against 0.5035 for always predicting the train-majority class). Seed 0's near-miss PRIMARY is therefore a **constant override, not selection** (post-hoc, descriptive, below). At seed 1 the control passes (+0.0015 [-0.0111, +0.0145]) and SEL-LEARN is flat-to-worse. Neither seed shows a learned-selection effect.
* T's own longitudinal argmax is a poor picker of the ADE-best class (SEL-ARGMAX agrees with k* on 14.3 % / 14.1 % of windows) and SEL-ARGMAX - B = +0.0085 [-0.0022, +0.0195] (s0) / +0.0092 [+0.0020, +0.0165] * (s1). SEL-MIX - B = -0.0007 / +0.0008 (not separated): O's plan is close to linear in the longitudinal slice of the code, so the soft code already is the mixture.
* Strategic family: **PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25), never a pass.

## Controls (values from `READ.json`; full dicts in the generated appendix)

| control | seed 0 | seed 1 |
|---|---|---|
| 2 REPRODUCTION (stop bar 1e-3 m), B's own-code ADE6 on CPU | **2.1753061 vs banked 2.1753061 (abs diff 8.4e-8 m); max traj30 diff 4.2e-5 m: PASS** (also vs the sfresh cycle rerun, 1.0e-8 m) | **2.0545660 vs banked 2.0545660 (abs diff 3.6e-9 m); max traj30 diff 9.9e-5 m: PASS** |
| harness identity: `forward_diag(T's own code)` == `forward_infer`, every eval and train window | max abs **0.0** (exact): PASS | max abs **0.0** (exact): PASS |
| 1 one-hot identity | **vacuous on real data: 0 exactly-one-hot softmax windows** (as predicted); the bit-exact case is pinned by `test_khyp.py` | same |
| 4 ORACLE <= selector on every window | ARGMAX 0 violations, LEARN 0: PASS (hard). MIX reported: beats ORACLE on 56 windows, max 0.128 m (a convex combination can beat the best vertex; declared interpretation 8) | ARGMAX 0, LEARN 0: PASS. MIX beats ORACLE on 30 windows, max 0.061 m |
| 3 deliberate regression: LEARN_PERM - ARGMAX must NOT separate | **-0.1068 [-0.1467, -0.0696] *: FAILED** | +0.0015 [-0.0111, +0.0145]: PASS |
| alignment of stored windows with the eval context; train window list == HBLEND's (9,582 windows, ci / t / a_t / a_s equal); H on B == HBLEND's banked H | all PASS (H diff 3.8e-5 m) | all PASS (H diff 8.7e-5 m) |
| the 7 hypotheses pairwise distinct (override live) | yes | yes |

No stop-control fired (`status` is not STOPPED).

## Beside rows (descriptive; full tables in the appendix)

* **Four families, SEL-LEARN - B (paired).** LONGITUDINAL s0: speed MAE -0.0920 *, accel MAE -0.1247 *, along MAE -0.1120 * (all better); s1: +0.0037, +0.0009, +0.0038 (none separated). LATERAL s0: cross-track +0.0080, heading -0.2247 deg, yaw-rate -0.4736 deg/s * (better), curvature -0.0083; s1: cross-track +0.0268 *, heading +0.2031 deg * (both separated WORSE), yaw-rate +0.2010, curvature +0.0036. TACTICAL (plan-derived agreement): s0 lat +0.0071, lon +0.0125; s1 lat +0.0083, lon -0.0104 (none separated). STRATEGIC: PAUSED. No family row is separated in the same direction at both seeds, so nothing here is a stable effect of the selector.
* **Distance keeping** (headway / time-gap / min-TTC / LEAD-VIOL; programme reader `v8f_dk_read`, K = 30; post-hoc rerun, see item 3 below), SEL-LEARN - B: s0 headway -0.161 m [-0.317, +0.003], time-gap -0.069 s [-0.168, -0.008] * (closer), min-TTC +0.213 s [-0.093, +0.539], lead-violation share of all lead windows +0.0067 [+0.0013, +0.0141] * (26 vs 21 of 743; separated WORSE); s1 headway +0.007, time-gap -0.017, min-TTC +0.203, violation share -0.0067 [-0.0167, +0.0015], none separated. ORACLE - B: min-TTC +0.351 * (s0), +0.498 * (s1); violation share -0.0135 * at s1.
* **CONSISTENCY(a) / (b)** rates are in the arm tables (unpaired rates; SEL-LEARN s0 (a) 0.1863 vs B 0.1596, (b) 0.2559 vs 0.2297; s1 (a) 0.2051 vs 0.2097, (b) 0.1376 vs 0.1430).
* **SEL-LEARN stacked with H's banked weights** (HBLEND `weights_M2b_s*.json`, descriptive): H_LEARN - B = -0.2350 [-0.3183, -0.1573] * (s0), -0.0552 [-0.0915, -0.0236] * (s1). Against H alone (H_B - B = -0.2068 *, -0.0697 *): H_LEARN - H_B = **-0.0282 [-0.0512, -0.0063] * (s0)** and **+0.0145 [-0.0071, +0.0360] (s1)**, so the stack adds nothing stable beyond H (and the s0 gain is the constant-class override).
* **The selector against k\*.** Accuracy on eval: ARGMAX 0.143 / 0.141; SEL-LEARN 0.513 / 0.358 (in-sample train 0.494 / 0.419); LEARN_PERM 0.504 / 0.354; always the train-majority class 0.504 / 0.356. k\* counts on eval (classes 0..6): s0 [290, 208, 314, 82, 130, 1208, 167]; s1 [442, 272, 382, 56, 173, 853, 221]. SEL-LEARN picks s0 [58, 13, 169, 0, 0, 2143, 16]; s1 [364, 45, 463, 1, 19, 1442, 65]. SEL-LEARN beats the majority class by +0.009 (s0) and +0.003 (s1) in accuracy.

## POST-HOC DESCRIPTIVE (run AFTER the outcome was fixed; changes no class set, feature, regularisation, selector or bar; nothing here is promoted without its own pre-registration)

From the banked `hyps/*.npz` only (`post_read.py`, `posthoc.json`; it reproduces `READ.json`'s pick counts exactly, both seeds).

1. **The seven CONSTANT selectors ("always class k") vs B, ADE6 paired** (the discriminating look after CONTROL 3 failed):

| k | seed 0 | seed 1 |
|---|---|---|
| 0 | +0.2673 [+0.2052, +0.3270] * | +0.1144 [+0.0456, +0.1825] * |
| 1 | +0.3214 [+0.2561, +0.3898] * | +0.0981 [+0.0437, +0.1556] * |
| 2 | +0.2243 [+0.1743, +0.2749] * | +0.0743 [+0.0307, +0.1172] * |
| 3 | -0.0348 [-0.0501, -0.0201] * | +0.0052 [-0.0039, +0.0141] |
| 4 | +0.0263 [+0.0001, +0.0509] * | +0.0026 [-0.0209, +0.0243] |
| **5** | **-0.0996 [-0.1412, -0.0621] \*** | +0.0108 [-0.0030, +0.0245] |
| 6 | -0.0569 [-0.0762, -0.0385] * | **-0.0194 [-0.0352, -0.0050] \*** |

   "Always class 5" at seed 0 (-0.0996) is the whole of SEL-LEARN's seed-0 gain (-0.0982). At seed 1 the best constant is a different class (6, -0.0194, about a sixth of F). Which constant helps depends on the checkpoint, and none reaches -F at seed 1.
2. **SEL-LEARN - SEL-LEARN_PERM**, ADE6: +0.0000 [-0.0112, +0.0112] (s0), +0.0097 [-0.0174, +0.0365] (s1): the learned selector is not distinguishable from a label-blind one.
3. **Distance-keeping rerun at TOL_ADE = 1e-4 m (DECLARED DEVIATION in the reader's frame/units sanity check only).** `khyp.py read` recorded `ERROR: Refusal: B_s0 ... differs from its saved ade6 by 1.35e-05 m (> 1e-05)`: float32 rounding in the banked per-window ade6 (the mean agrees to 8.4e-8 m, Control 2). The reader's tolerance was raised to 1e-4 in `post_read.py` only; every statistic is the reader's own (`dk_read_tol1e4.json`). The `khyp.py` ERROR stays recorded in `READ.json`. WORK ITEM: the reader's 1e-5 float64-vs-saved-ade6 tolerance should be 1e-4 (or compare in float32), or every float32 plan will be refused.
4. **Caveat on HEADROOM (interpretation, not a registered control):** ORACLE is a min over seven plans scored against the GT, so part of its gain is selection from a spread, not information a selector could use. The constant rows show how wide the spread is (k = 0..2 are 0.07-0.32 m WORSE than B), so the headroom is not merely a noise minimum, but the PREREG has no random-pick control for it.

## What would move this (RULE ZERO: a refutation is a waypoint; nothing below was run, the PREREG's zero-training scope ends here)

* The PREREG's committed next is a trained, scene-reading scorer (box 2). The measured case for it: headroom -0.36 to -0.42 m at both seeds, cheap plan-statistics features add nothing over the majority class (+0.009 / +0.003 accuracy), and T's own argmax is 14 % accurate against k*.
* The cheapest discriminating precursor, **not run** (the features are frozen by the PREREG, so it needs its own pre-registration): the same logistic regression on T's frozen scene latent instead of plan statistics, fitted on HBLEND's train windows, scored on the same 141 episodes with a permuted-label control and both seeds. If a linear read of the scene cannot beat the majority class either, the box-2 scorer's prior is weaker than this headroom suggests.
* A second cheap check, also not run: a confidence gate (use class k only when the classifier beats the majority class), since a constant override helps on one seed and hurts on the other.

## Run record

* **Guard** (edited in `run_phaseB.sh`, replacing the 3 GB rule): C: free >= 0.7 GB AND commit headroom >= 5 GB, re-checked once a minute for at most 4 min. Never tripped: at starts C: free 1.085 / 1.058 / 0.992 / 0.987 / 0.907 GB (4.175 GB at `post`), commit headroom 9.98 / 11.87 / 11.65 / 11.38 / 10.09 GB (5.79 GB at `post`). Torch threads 6 (the brief); OpenBLAS/MKL 1. A `post` stage and `run_chain.sh` were added to the runner.
* **Peak private commit** (psutil `peak_pagefile`, `logs/peak_memory.jsonl`): eval s0 1.476 GB, train s0 1.579 GB, eval s1 1.476 GB, train s1 1.578 GB, read 1.137 GB, post 1.165 GB (peak working set <= 1.083 GB). Wall: 87 s / 326 s / 85 s / 334 s / 188 s / 62 s.
* `khyp.py` and `test_khyp.py` were not modified in Phase B. One python process at a time. CPU only (`CUDA_VISIBLE_DEVICES=-1`). Nothing staged, committed, pushed or deleted; no box-2 access.

---
# APPENDIX: the generated report, verbatim (`khyp.py read`; unedited copy in `KHYP_RESULT_generated.md`)

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
