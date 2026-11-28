# PROBEGATE RESULT (mechanical; PREREG_PROBEGATE sha256 20a643303207)

**Verdict: FAIL (P1, P3, G)**

estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'would another draw of EPISODES say this?' ONLY; NOT another training run (one refcv8 training seed), NOT another inference run (one inference seed, 0)
tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)

## rates (S6 compliant, with the shipped stage; false-decel on GT-cruise n 1,856, missed-decel on GT-decel n 639)

| arm | false-decel | missed-decel |
|---|---|---|
| BASE-os | 0.1395 | 0.4695 |
| BASE-bat | 0.1369 | 0.4695 |
| ORACLE | 0.1379 | 0.3412 |
| PROBEGATE | 0.1428 | 0.4726 |
| RANDGATE | 0.1584 | 0.4773 |

## bars

| clause | reading | cell (delta [lo, hi]) |
|---|---|---|
| P1 PROBEGATE missed-decel separated BELOW base | False | +0.0031 [-0.0177, +0.0241] separated=False |
| P2 PROBEGATE false-decel NOT separated above base | True | +0.0032 [-0.0070, +0.0128] separated=False |
| P3 PROBEGATE - RANDGATE missed-decel separated below 0 | False | -0.0047 [-0.0298, +0.0203] separated=False |
| G no METRICS row / S6 ADE separated worse | False | rows: LAT_cross_abs_final_m, LAT_curvature_mae_1pm_valid, LAT_yaw_rate_mae_radps_valid, TAC_traj_lon_correct |
| (context) rows the ORACLE force itself degrades | - | LAT_curvature_mae_1pm_valid, LAT_heading_mae_deg, LAT_yaw_rate_mae_radps_valid |
| (context) rows RANDGATE degrades | - | LAT_cross_abs_final_m, LAT_cross_mae_m, LAT_curvature_mae_1pm_valid, LAT_heading_mae_deg, LAT_yaw_rate_mae_radps_valid, LON_speed_mae_mps, LON_speed_within_0p5, TAC_traj_lon_correct, ade_m, fde_m |

ceiling share (MD_base - MD_probegate)/(MD_base - MD_oracle) = -0.024 [-0.238, 0.127] (DESCRIPTIVE; MD base 0.4695, probegate 0.4726, oracle 0.3412)

## descriptive

```
{
 "ceiling_share_missed_decel": {
  "point": -0.02439024390243894,
  "lo": -0.23825396825396822,
  "hi": 0.12679667750090268,
  "md": {
   "base": 0.4694835680751174,
   "new": 0.4726134585289515,
   "oracle": 0.3411580594679186
  },
  "randgate": {
   "point": -0.06097560975609757,
   "lo": -0.3333333333333329,
   "hi": 0.10810810810810811,
   "md": {
    "base": 0.4694835680751174,
    "new": 0.47730829420970267,
    "oracle": 0.3411580594679186
   }
  }
 },
 "PROBEGATE_changed_windows_vs_base": {
  "missed_decel (GT-decel)": {
   "fixed": 12,
   "broken": 14,
   "n": 639
  },
  "false_decel (GT-cruise)": {
   "fixed": 15,
   "broken": 21,
   "n": 1856
  }
 },
 "RANDGATE_changed_windows_vs_base": {
  "missed_decel (GT-decel)": {
   "fixed": 17,
   "broken": 22,
   "n": 639
  },
  "false_decel (GT-cruise)": {
   "fixed": 17,
   "broken": 52,
   "n": 1856
  }
 },
 "flips_on_S6_compliant_from_labels_check": {
  "probegate": {
   "up": {
    "total": 489,
    "in_S6_compliant": 338,
    "GT_decel": 123,
    "GT_cruise": 125,
    "GT_accel": 90
   },
   "down": {
    "total": 489,
    "in_S6_compliant": 315,
    "GT_decel": 68,
    "GT_cruise": 196,
    "GT_accel": 51
   }
  },
  "randgate": {
   "up": {
    "total": 489,
    "in_S6_compliant": 338,
    "GT_decel": 40,
    "GT_cruise": 218,
    "GT_accel": 80
   },
   "down": {
    "total": 489,
    "in_S6_compliant": 301,
    "GT_decel": 139,
    "GT_cruise": 134,
    "GT_accel": 28
   }
  }
 }
}
```

## controls

```
{
 "K1_stage_bit_equal_banked_s0": true,
 "K2_population_s0": true,
 "K0_seals": {
  "prereg": true
 },
 "K3_hook_and_identity_ORACLE": {
  "order_equals_extras": true,
  "n_forced": 3672,
  "n_unforced": 1082,
  "unforced_plan8_max_abs_vs_BASE-os": 0.0,
  "unforced_sel_equal_BASE-os": true,
  "counters": {
   "calls": 4754,
   "hook_applied": 3672,
   "hook_native_fallthrough": 1082,
   "hook_calls_unarmed": 0,
   "multi_hook_in_call": 0,
   "batch_gt1": 0,
   "navnull_calls": 0,
   "oracle_labelled": 3672
  },
  "counters_ok": true,
  "applied_onehot_on_forced": true,
  "applied_equals_raw_on_unforced": true,
  "raw_equals_ORACLE_raw": true,
  "pass": true
 },
 "K3_hook_and_identity_PROBEGATE": {
  "order_equals_extras": true,
  "n_forced": 978,
  "n_unforced": 3776,
  "unforced_plan8_max_abs_vs_BASE-os": 0.0,
  "unforced_sel_equal_BASE-os": true,
  "counters": {
   "calls": 4754,
   "hook_applied": 978,
   "hook_native_fallthrough": 3776,
   "hook_calls_unarmed": 0,
   "multi_hook_in_call": 0,
   "batch_gt1": 0,
   "navnull_calls": 0,
   "oracle_labelled": 978
  },
  "counters_ok": true,
  "applied_onehot_on_forced": true,
  "applied_equals_raw_on_unforced": true,
  "raw_equals_ORACLE_raw": true,
  "pass": true
 },
 "K3_hook_and_identity_RANDGATE": {
  "order_equals_extras": true,
  "n_forced": 978,
  "n_unforced": 3776,
  "unforced_plan8_max_abs_vs_BASE-os": 0.0,
  "unforced_sel_equal_BASE-os": true,
  "counters": {
   "calls": 4754,
   "hook_applied": 978,
   "hook_native_fallthrough": 3776,
   "hook_calls_unarmed": 0,
   "multi_hook_in_call": 0,
   "batch_gt1": 0,
   "navnull_calls": 0,
   "oracle_labelled": 978
  },
  "counters_ok": true,
  "applied_onehot_on_forced": true,
  "applied_equals_raw_on_unforced": true,
  "raw_equals_ORACLE_raw": true,
  "pass": true
 },
 "K4_labels_rederived": {
  "probegate_equal": true,
  "randgate_equal": true,
  "n_up": 489,
  "n_down": 489,
  "pass": true
 },
 "K5_labels_checks": {
  "LABELS_CHECK_ALL_PASS": true,
  "LABELS_VERIFY_ALL_PASS": true,
  "sha256_equal_recorded": true,
  "prereg_sha_in_check": true,
  "pass": true
 },
 "K6_identity_BASE_vs_BASE_reads_zero": {
  "pass": true,
  "n_rows": 18
 },
 "K7_oracle_vs_base_reproduces_L2": {
  "pass": true,
  "missed_decel_cell": {
   "delta": -0.1283,
   "lo": -0.1802,
   "hi": -0.0762,
   "separated": true
  }
 },
 "K8_oracle_rates_equal_L2": {
  "pass": true
 }
}
```

## next lever (pre-registered mapping, section 8)

GATE INERT (P1 failed and the missed-decel point estimate did not improve): flipping the head's decel/cruise group at probe quality does not move the plan.  Next lever = stop gating post hoc; attribute the oracle gap per forced class (BRAKE_TO / CREEP / HOLD) at 0 GPU from the banked oracle trace, then train toward the probe's posterior (L3) only if one class carries the gap.
