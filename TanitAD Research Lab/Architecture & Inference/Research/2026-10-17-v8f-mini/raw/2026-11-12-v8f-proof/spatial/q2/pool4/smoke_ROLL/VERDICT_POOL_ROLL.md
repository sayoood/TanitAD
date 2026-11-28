# Pooled lead read over seeds, arm ROLL (note PREREG_V8F_PROOF_ADD18_N2, section 4; calibrated forms beside)

MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Estimator: two-level bootstrap: seeds with replacement, then episodes with replacement inside each drawn seed; rng seed 0; percentiles 2.5 and 97.5, B 2000.

## Controls (first)

* p4_each_seed_once_same_windows: PASS {"seeds": [0, 1], "n_lead_windows": [441, 441], "n_episodes": [139, 139], "same_windows_and_labels_mask": true}
* p1_one_seed_equals_rb_trained: PASS {"bar": 1e-09}
* p2_copy_as_rollout_is_0: PASS {"A": 0.0, "A_boot_abs_max": 0.0, "V": 0.0, "A3_abs": 1.6653345369377348e-16}
* p3_permuted_labels: PASS {"A": 0.05748663116654262, "ci95": [-0.0440533928501579, 0.1636489930347404], "why": "with labels that carry nothing, A is the difference of the spreads of the two reads over the spread of the label"}
* ALL_PASS: PASS
* beside, calibrated forms under permuted labels: A3 -0.013 [-0.022, -0.005], A3r -0.007

## Per seed (each used once)

| seed | lead windows | spread of the reads (label / rollout / copy / true future, m) | R2 rollout / copy / true future | A: rollout minus copy | A2: rollout minus the ego floor | A3: rollout added to the calibrated copy | A3r: calibrated rollout minus calibrated copy | A3c: true future added to the calibrated copy (headroom) | A3e: rollout added to the calibrated copy and ego floor | A3ec: true future added to the calibrated copy and ego floor (headroom) | V: change skill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 441 | 10.02 / 6.41 / 6.61 / 6.32 | 0.431 / 0.450 / 0.475 | -0.019 | +0.252 | +0.024 | -0.025 | +0.035 | +0.009 | +0.033 | -0.115 |
| 1 | 441 | 10.02 / 6.13 / 6.63 / 6.13 | 0.420 / 0.451 / 0.508 | -0.032 | +0.241 | +0.022 | -0.023 | +0.075 | +0.010 | +0.071 | -0.053 |

## Pooled

* scope all, seeds [0, 1]: A: rollout minus copy -0.025 [-0.100, +0.045]; A2: rollout minus the ego floor +0.247 [+0.083, +0.447]; A3: rollout added to the calibrated copy +0.023 [-0.016, +0.062]; A3r: calibrated rollout minus calibrated copy -0.024 [-0.103, +0.053]; A3c: true future added to the calibrated copy (headroom) +0.055 [+0.014, +0.103]; A3e: rollout added to the calibrated copy and ego floor +0.009 [-0.026, +0.045]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.052 [+0.012, +0.099]; V: change skill -0.084 [-0.258, +0.040]; rollout read, arm minus base -0.014 [-0.105, +0.073]; A3, arm minus base +0.004 [-0.045, +0.048]; A3e, arm minus base -0.002 [-0.042, +0.037]
* base arm G read the same way, seeds [0, 1]: A: rollout minus copy -0.045 [-0.122, +0.030]; A2: rollout minus the ego floor +0.261 [+0.104, +0.463]; A3: rollout added to the calibrated copy +0.020 [-0.014, +0.066]; A3r: calibrated rollout minus calibrated copy -0.016 [-0.106, +0.079]; A3c: true future added to the calibrated copy (headroom) +0.078 [+0.036, +0.127]; A3e: rollout added to the calibrated copy and ego floor +0.011 [-0.020, +0.050]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.076 [+0.035, +0.123]; V: change skill -0.105 [-0.237, +0.010]

## Controls of note N3 (calibrated read)

* n1_p1_p2_p4: PASS {}
* n2_permuted_labels_calibrated: PASS {"A3": -0.013246280186177573, "A3_ci95": [-0.02186137013550495, -0.005058252496484928], "A3e": -0.012053463287540755, "A3e_ci95": [-0.019331878733370597, -0.004862579297007235]}
* n3_true_future_as_rollout_above_0: PASS {"A3c": 0.05508245564164921, "ci95": [0.013922305535551302, 0.1027782552377137]}
* n4_copy_as_rollout_is_0: PASS {"A3_abs": 1.6653345369377348e-16, "A3e_abs": 4.996003610813204e-16, "bar": 1e-09}
* ALL_PASS: PASS

## Names

* note N2: not named: the names of the note need four seeds with two new ones; this run reads seeds [0, 1]
* note N3: not named: the names of note N3 need four seeds with two new ones; this run reads seeds [0, 1]
* share of the headroom reached: {"all": {"A3_over_A3c": 0.421, "A3e_over_A3ec": 0.181}}

Limits: one training run per seed; probes show what is linearly readable; T0 reads on logged frames; one corpus.
