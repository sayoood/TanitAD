# Pooled lead read over seeds, arm RR (note PREREG_V8F_PROOF_ADD18_N2, section 4; calibrated forms beside)

MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Estimator: two-level bootstrap: seeds with replacement, then episodes with replacement inside each drawn seed; rng seed 0; percentiles 2.5 and 97.5, B 2000.

## Controls (first)

* p4_each_seed_once_same_windows: PASS {"seeds": [0, 1, 2, 3], "n_lead_windows": [441, 441, 441, 441], "n_episodes": [139, 139, 139, 139], "same_windows_and_labels_mask": true}
* p1_one_seed_equals_rb_trained: PASS {"bar": 1e-09}
* p2_copy_as_rollout_is_0: PASS {"A": 0.0, "A_boot_abs_max": 0.0, "V": 0.0, "A3_abs": 1.3877787807814457e-16}
* p3_permuted_labels: FAIL {"A": 0.06403471479412104, "ci95": [0.03321166499364995, 0.09825311470631783], "why": "with labels that carry nothing, A is the difference of the spreads of the two reads over the spread of the label"}
* ALL_PASS: FAIL
* beside, calibrated forms under permuted labels: A3 -0.006 [-0.012, +0.001], A3r -0.001

## Per seed (each used once)

| seed | lead windows | spread of the reads (label / rollout / copy / true future, m) | R2 rollout / copy / true future | A: rollout minus copy | A2: rollout minus the ego floor | A3: rollout added to the calibrated copy | A3r: calibrated rollout minus calibrated copy | A3c: true future added to the calibrated copy (headroom) | A3e: rollout added to the calibrated copy and ego floor | A3ec: true future added to the calibrated copy and ego floor (headroom) | V: change skill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 441 | 10.02 / 6.44 / 6.95 / 6.58 | 0.485 / 0.435 / 0.502 | +0.050 | +0.306 | +0.046 | +0.052 | +0.073 | +0.032 | +0.067 | +0.091 |
| 1 | 441 | 10.02 / 6.62 / 6.89 / 6.41 | 0.493 / 0.451 / 0.529 | +0.042 | +0.315 | +0.040 | +0.046 | +0.096 | +0.025 | +0.089 | +0.070 |
| 2 (new) | 441 | 10.02 / 6.42 / 6.95 / 6.54 | 0.519 / 0.509 / 0.559 | +0.010 | +0.340 | +0.006 | +0.017 | +0.063 | -0.008 | +0.065 | +0.091 |
| 3 (new) | 441 | 10.02 / 5.93 / 6.33 / 6.05 | 0.466 / 0.462 / 0.528 | +0.004 | +0.287 | +0.010 | +0.006 | +0.077 | +0.008 | +0.076 | +0.101 |

## Pooled

* scope all, seeds [0, 1, 2, 3]: A: rollout minus copy +0.027 [-0.001, +0.058]; A2: rollout minus the ego floor +0.312 [+0.201, +0.464]; A3: rollout added to the calibrated copy +0.025 [-0.000, +0.057]; A3r: calibrated rollout minus calibrated copy +0.030 [+0.001, +0.061]; A3c: true future added to the calibrated copy (headroom) +0.077 [+0.044, +0.112]; A3e: rollout added to the calibrated copy and ego floor +0.014 [-0.009, +0.042]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.074 [+0.046, +0.106]; V: change skill +0.088 [+0.044, +0.148]; rollout read, arm minus base +0.025 [-0.031, +0.086]; A3, arm minus base +0.009 [-0.019, +0.040]; A3e, arm minus base +0.008 [-0.019, +0.035]
* scope new, seeds [2, 3]: A: rollout minus copy +0.007 [-0.021, +0.038]; A2: rollout minus the ego floor +0.314 [+0.151, +0.534]; A3: rollout added to the calibrated copy +0.008 [-0.016, +0.031]; A3r: calibrated rollout minus calibrated copy +0.012 [-0.020, +0.046]; A3c: true future added to the calibrated copy (headroom) +0.070 [+0.029, +0.114]; A3e: rollout added to the calibrated copy and ego floor -0.000 [-0.023, +0.022]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.070 [+0.033, +0.113]; V: change skill +0.096 [+0.034, +0.177]; rollout read, arm minus base +0.001 [-0.073, +0.073]; A3, arm minus base -0.004 [-0.034, +0.026]; A3e, arm minus base -0.001 [-0.037, +0.033]
* base arm G read the same way, seeds [0, 1, 2, 3]: A: rollout minus copy -0.028 [-0.080, +0.017]; A2: rollout minus the ego floor +0.287 [+0.178, +0.426]; A3: rollout added to the calibrated copy +0.016 [-0.008, +0.043]; A3r: calibrated rollout minus calibrated copy -0.012 [-0.067, +0.042]; A3c: true future added to the calibrated copy (headroom) +0.072 [+0.041, +0.104]; A3e: rollout added to the calibrated copy and ego floor +0.006 [-0.017, +0.029]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.070 [+0.040, +0.102]; V: change skill -0.071 [-0.169, +0.006]

## Controls of note N3 (calibrated read)

* n1_p1_p2_p4: PASS {}
* n2_permuted_labels_calibrated: PASS {"A3": -0.005810677430327205, "A3_ci95": [-0.012149655481830118, 0.0007878262164822801], "A3e": -0.0057152557164396045, "A3e_ci95": [-0.012168141975793155, 0.0011883684408047343]}
* n3_true_future_as_rollout_above_0: PASS {"A3c": 0.07710984241258598, "ci95": [0.04435547837676244, 0.11170957164449113]}
* n4_copy_as_rollout_is_0: PASS {"A3_abs": 1.3877787807814457e-16, "A3e_abs": 2.7755575615628914e-17, "bar": 1e-09}
* ALL_PASS: PASS

## Names

* note N2: not named: a control failed: nothing is named
* note N3: MOTION-ADDS-NOTHING-4
* share of the headroom reached: {"all": {"A3_over_A3c": 0.329, "A3e_over_A3ec": 0.193}, "new": {"A3_over_A3c": 0.113, "A3e_over_A3ec": -0.002}}

Limits: one training run per seed; probes show what is linearly readable; T0 reads on logged frames; one corpus.
