# L2 RESULT (mechanical, PREREG_L2 sha256 4267cd684f38, Amendment 1 c463ba5556ab)

**Verdict on the bars: FAIL (seed 0 fails a clause common to both PASS labels; the other seed cannot change it)**  |  **Outcome: O5 CHANNEL-INERT (K6: forcing CRUISE does not move the plan)**

tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)
estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: answers 'another draw of episodes' only; NOT another training run (single training seed), NOT another inference run (seeds 0 / 1 shown separately, never pooled)
primary comparator: BASE-os; inference-seed floor on FD: {'per_baseline': {'BASE-os': 0.0021551724137931216, 'BASE-bat': 0.0005387931034482596}, 'F': 0.0021551724137931216}

pending: ['cal s1 (not rolled)', 'const_cruise s1 (not rolled)', 'oracle s1 (not rolled)']

## rates (S6 compliant; false-decel on GT-cruise n 1,856, missed-decel on GT-decel n 639)

| seed | arm | stage | false-decel | missed-decel |
|---|---|---|---|---|
| 0 | BASE-bat | stage | 0.1369 | 0.4695 |
| 0 | BASE-bat | nostage | 0.1395 | 0.4695 |
| 0 | BASE-os | stage | 0.1395 | 0.4695 |
| 0 | BASE-os | nostage | 0.1422 | 0.4710 |
| 0 | cal | stage | 0.1460 | 0.4710 |
| 0 | cal | nostage | 0.1482 | 0.4726 |
| 0 | const_cruise | stage | 0.1352 | 0.4194 |
| 0 | const_cruise | nostage | 0.1374 | 0.4210 |
| 0 | oracle | stage | 0.1379 | 0.3412 |
| 0 | oracle | nostage | 0.1401 | 0.3396 |
| 0 | r7 | stage | 0.0911 | 0.5587 |
| 0 | r7 | nostage | 0.0911 | 0.5587 |
| 1 | BASE-bat | stage | 0.1374 | 0.4710 |
| 1 | BASE-bat | nostage | 0.1395 | 0.4789 |
| 1 | BASE-os | stage | 0.1374 | 0.4710 |
| 1 | BASE-os | nostage | 0.1395 | 0.4789 |
| 1 | r7 | stage | 0.0938 | 0.5681 |
| 1 | r7 | nostage | 0.0938 | 0.5681 |

## bars

| seed | FD | MD | P1 | P2 | P3 | P3L | P4 | G | rows separated worse |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1460 | 0.4710 | False | False | False | True | False | False | LAT_curvature_mae_1pm_valid, LAT_yaw_rate_mae_radps_valid, LON_accel_mae_mps2, LON_along_abs_final_m, LON_along_mae_m, LON_speed_mae_mps, ade_m, fde_m |
| 1 | PENDING | - | - | - | - | - | - | - | - | |

## controls

```
{
 "K1_stage_bit_equal_banked_s0": true,
 "K2_population_s0": true,
 "K1_stage_bit_equal_banked_s1": true,
 "K2_population_s1": true,
 "K3_ident_vs_native": {
  "n": 308,
  "max_abs_m": 0.0,
  "sel_equal": true,
  "hook_calls_armed": 308,
  "pass": true
 },
 "K4_cal_trace_vs_apply8_s0": {
  "max_abs": 4.899648891054653e-07,
  "pass": true
 },
 "K4_const_all_cruise_s0": {
  "frac_argmax_cruise": 1.0,
  "pass": true
 },
 "K5_base_os_vs_battery_rates": {
  "false_decel": {
   "BASE-os": {
    "0": 0.13954741379310345,
    "1": 0.13739224137931033
   },
   "BASE-bat": {
    "0": 0.13685344827586207,
    "1": 0.13739224137931033
   },
   "tol": 0.01,
   "pass": true
  },
  "missed_decel": {
   "BASE-os": {
    "0": 0.4694835680751174,
    "1": 0.47104851330203446
   },
   "BASE-bat": {
    "0": 0.4694835680751174,
    "1": 0.47104851330203446
   },
   "tol": 0.01,
   "pass": true
  }
 },
 "K7_fit_selftest": true,
 "K6_const_potency": {
  "cell": {
   "delta": -0.0043,
   "lo": -0.0335,
   "hi": 0.0228,
   "separated": false,
   "n_windows": 1856,
   "n_episodes": 135,
   "mean_new": 0.13523706896551724,
   "mean_ref": 0.13954741379310345,
   "family": "LONGITUDINAL",
   "label": "false-decel rate (GT-cruise)",
   "unit": "frac",
   "lower_is_better": true,
   "worse": false
  },
  "pass": false
 }
}
```

Full four-family tables for every pair: `L2_RESULT.json` -> seeds -> comparisons.