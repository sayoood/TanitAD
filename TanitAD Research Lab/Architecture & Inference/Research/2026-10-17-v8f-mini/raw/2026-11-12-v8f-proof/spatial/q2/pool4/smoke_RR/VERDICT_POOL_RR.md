# Pooled lead read over seeds, arm RR (note PREREG_V8F_PROOF_ADD18_N2, section 4; calibrated forms beside)

MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Estimator: two-level bootstrap: seeds with replacement, then episodes with replacement inside each drawn seed; rng seed 0; percentiles 2.5 and 97.5, B 2000.

## Controls (first)

* p4_each_seed_once_same_windows: PASS {"seeds": [0, 1], "n_lead_windows": [441, 441], "n_episodes": [139, 139], "same_windows_and_labels_mask": true}
* p1_one_seed_equals_rb_trained: PASS {"bar": 1e-09}
* p2_copy_as_rollout_is_0: PASS {"A": 0.0, "A_boot_abs_max": 0.0, "V": 0.0, "A3_abs": 1.6653345369377348e-16}
* p3_permuted_labels: FAIL {"A": 0.062183817971961486, "ci95": [0.026769616894721536, 0.10154740772238202], "why": "with labels that carry nothing, A is the difference of the spreads of the two reads over the spread of the label"}
* ALL_PASS: FAIL
* beside, calibrated forms under permuted labels: A3 -0.008 [-0.018, +0.006], A3r -0.003

## Per seed (each used once)

| seed | lead windows | spread of the reads (label / rollout / copy / true future, m) | R2 rollout / copy / true future | A: rollout minus copy | A2: rollout minus the ego floor | A3: rollout added to the calibrated copy | A3r: calibrated rollout minus calibrated copy | A3c: true future added to the calibrated copy (headroom) | A3e: rollout added to the calibrated copy and ego floor | A3ec: true future added to the calibrated copy and ego floor (headroom) | V: change skill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 441 | 10.02 / 6.44 / 6.95 / 6.58 | 0.485 / 0.435 / 0.502 | +0.050 | +0.306 | +0.046 | +0.052 | +0.073 | +0.032 | +0.067 | +0.091 |
| 1 | 441 | 10.02 / 6.62 / 6.89 / 6.41 | 0.493 / 0.451 / 0.529 | +0.042 | +0.315 | +0.040 | +0.046 | +0.096 | +0.025 | +0.089 | +0.070 |

## Pooled

* scope all, seeds [0, 1]: A: rollout minus copy +0.046 [+0.015, +0.082]; A2: rollout minus the ego floor +0.311 [+0.138, +0.534]; A3: rollout added to the calibrated copy +0.043 [+0.003, +0.086]; A3r: calibrated rollout minus calibrated copy +0.049 [+0.015, +0.086]; A3c: true future added to the calibrated copy (headroom) +0.084 [+0.037, +0.142]; A3e: rollout added to the calibrated copy and ego floor +0.029 [-0.006, +0.066]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.078 [+0.036, +0.129]; V: change skill +0.080 [+0.011, +0.165]; rollout read, arm minus base +0.050 [-0.025, +0.140]; A3, arm minus base +0.023 [-0.022, +0.067]; A3e, arm minus base +0.017 [-0.020, +0.054]
* base arm G read the same way, seeds [0, 1]: A: rollout minus copy -0.045 [-0.122, +0.030]; A2: rollout minus the ego floor +0.261 [+0.104, +0.463]; A3: rollout added to the calibrated copy +0.020 [-0.014, +0.066]; A3r: calibrated rollout minus calibrated copy -0.016 [-0.106, +0.079]; A3c: true future added to the calibrated copy (headroom) +0.078 [+0.036, +0.127]; A3e: rollout added to the calibrated copy and ego floor +0.011 [-0.020, +0.050]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.076 [+0.035, +0.123]; V: change skill -0.105 [-0.237, +0.010]

## Controls of note N3 (calibrated read)

* n1_p1_p2_p4: PASS {}
* n2_permuted_labels_calibrated: PASS {"A3": -0.007682900269353143, "A3_ci95": [-0.017713066166141937, 0.005641845610892036], "A3e": -0.00720456553785187, "A3e_ci95": [-0.01645792919061746, 0.005975314023233885]}
* n3_true_future_as_rollout_above_0: PASS {"A3c": 0.08430720784838064, "ci95": [0.0370398972252582, 0.14230866482731444]}
* n4_copy_as_rollout_is_0: PASS {"A3_abs": 1.6653345369377348e-16, "A3e_abs": 2.220446049250313e-16, "bar": 1e-09}
* ALL_PASS: PASS

## Names

* note N2: not named: a control failed: nothing is named
* note N3: not named: the names of note N3 need four seeds with two new ones; this run reads seeds [0, 1]
* share of the headroom reached: {"all": {"A3_over_A3c": 0.509, "A3e_over_A3ec": 0.369}}

Limits: one training run per seed; probes show what is linearly readable; T0 reads on logged frames; one corpus.
