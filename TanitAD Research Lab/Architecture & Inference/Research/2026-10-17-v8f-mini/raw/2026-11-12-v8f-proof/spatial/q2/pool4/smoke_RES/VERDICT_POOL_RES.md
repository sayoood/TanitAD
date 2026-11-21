# Pooled lead read over seeds, arm RES (note PREREG_V8F_PROOF_ADD18_N2, section 4; calibrated forms beside)

MEASURED, tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25).
Estimator: two-level bootstrap: seeds with replacement, then episodes with replacement inside each drawn seed; rng seed 0; percentiles 2.5 and 97.5, B 2000.

## Controls (first)

* p4_each_seed_once_same_windows: PASS {"seeds": [0, 1], "n_lead_windows": [441, 441], "n_episodes": [139, 139], "same_windows_and_labels_mask": true}
* p1_one_seed_equals_rb_trained: PASS {"bar": 1e-09}
* p2_copy_as_rollout_is_0: PASS {"A": 0.0, "A_boot_abs_max": 0.0, "V": 0.0, "A3_abs": 5.551115123125783e-17}
* p3_permuted_labels: PASS {"A": 0.04005497175485251, "ci95": [-0.00045389727744421515, 0.0919210067020844], "why": "with labels that carry nothing, A is the difference of the spreads of the two reads over the spread of the label"}
* ALL_PASS: PASS
* beside, calibrated forms under permuted labels: A3 -0.007 [-0.018, +0.001], A3r -0.003

## Per seed (each used once)

| seed | lead windows | spread of the reads (label / rollout / copy / true future, m) | R2 rollout / copy / true future | A: rollout minus copy | A2: rollout minus the ego floor | A3: rollout added to the calibrated copy | A3r: calibrated rollout minus calibrated copy | A3c: true future added to the calibrated copy (headroom) | A3e: rollout added to the calibrated copy and ego floor | A3ec: true future added to the calibrated copy and ego floor (headroom) | V: change skill |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 441 | 10.02 / 6.39 / 6.68 / 6.34 | 0.383 / 0.366 / 0.425 | +0.017 | +0.205 | -0.001 | +0.014 | +0.055 | -0.015 | +0.042 | +0.029 |
| 1 | 441 | 10.02 / 6.31 / 6.66 / 6.10 | 0.486 / 0.475 / 0.540 | +0.012 | +0.308 | +0.019 | +0.023 | +0.086 | -0.000 | +0.081 | +0.061 |

## Pooled

* scope all, seeds [0, 1]: A: rollout minus copy +0.014 [-0.017, +0.046]; A2: rollout minus the ego floor +0.256 [+0.084, +0.461]; A3: rollout added to the calibrated copy +0.009 [-0.021, +0.044]; A3r: calibrated rollout minus calibrated copy +0.018 [-0.011, +0.053]; A3c: true future added to the calibrated copy (headroom) +0.070 [+0.025, +0.121]; A3e: rollout added to the calibrated copy and ego floor -0.008 [-0.037, +0.024]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.061 [+0.017, +0.113]; V: change skill +0.045 [-0.005, +0.108]; rollout read, arm minus base -0.005 [-0.113, +0.098]; A3, arm minus base -0.010 [-0.046, +0.023]; A3e, arm minus base -0.019 [-0.051, +0.015]
* base arm G read the same way, seeds [0, 1]: A: rollout minus copy -0.045 [-0.122, +0.030]; A2: rollout minus the ego floor +0.261 [+0.104, +0.463]; A3: rollout added to the calibrated copy +0.020 [-0.014, +0.066]; A3r: calibrated rollout minus calibrated copy -0.016 [-0.106, +0.079]; A3c: true future added to the calibrated copy (headroom) +0.078 [+0.036, +0.127]; A3e: rollout added to the calibrated copy and ego floor +0.011 [-0.020, +0.050]; A3ec: true future added to the calibrated copy and ego floor (headroom) +0.076 [+0.035, +0.123]; V: change skill -0.105 [-0.237, +0.010]

## Controls of note N3 (calibrated read)

* n1_p1_p2_p4: PASS {}
* n2_permuted_labels_calibrated: PASS {"A3": -0.0073430612232990855, "A3_ci95": [-0.01787786231555041, 0.0013661458559209875], "A3e": -0.005231229410341598, "A3e_ci95": [-0.016796899768885383, 0.0047206114858606525]}
* n3_true_future_as_rollout_above_0: PASS {"A3c": 0.0704702232993153, "ci95": [0.025030845003817526, 0.120570197756986]}
* n4_copy_as_rollout_is_0: PASS {"A3_abs": 5.551115123125783e-17, "A3e_abs": 2.7755575615628914e-16, "bar": 1e-09}
* ALL_PASS: PASS

## Names

* note N2: not named: the names of the note need four seeds with two new ones; this run reads seeds [0, 1]
* note N3: not named: the names of note N3 need four seeds with two new ones; this run reads seeds [0, 1]
* share of the headroom reached: {"all": {"A3_over_A3c": 0.129, "A3e_over_A3ec": -0.123}}

Limits: one training run per seed; probes show what is linearly readable; T0 reads on logged frames; one corpus.
