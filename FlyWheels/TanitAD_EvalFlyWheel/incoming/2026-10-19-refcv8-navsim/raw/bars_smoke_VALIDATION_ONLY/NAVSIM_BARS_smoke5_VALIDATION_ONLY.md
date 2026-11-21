# NAVSIM_BARS_smoke5_VALIDATION_ONLY

**VALIDATION ONLY (not refcv8's NavSim score): the run's own label says VALIDATION ONLY; checkpoint step 40000 != 48852**

Tier: NavSim open-loop benchmark (T1-family), zero-shot PhysicalAI-AV -> nuPlan cameras, non-parity; NEVER closed loop. NOT leaderboard-submitted (PI 2026-10-04).
Run: `C:\Users\Admin\qland\landed_hold_r8navsim2\stage\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-10-19-refcv8-navsim\raw\smoke_cpu_ema40k_VALIDATION_ONLY` step 40000 weights ema; n = 5 tokens / 5 logs.

> **VALIDATION ONLY -- n = 5, NO VERDICT.** the run's own label says VALIDATION ONLY; checkpoint step 40000 != 48852. Numbers exercise the path; they are never refcv8's NavSim performance.

## Estimator
`navsim_log_cluster_bootstrap` / `paired_navsim_log_cluster_bootstrap` (IMPORTED from `C:\lgs\Lfb47a58376\taniteval\adapters\navsim_ci.py`, sha256 `00e0e39ef6cfc9de`); cluster unit **log_name** (sensitivity unit nuplan_drive), B = 2000, seed 0, percentile 95 %. Answers: *another draw of LOGS?* only. Blind to training variance (one training seed) and to inference variance, which is read from R8_LEGAL vs R8_LEGAL_s1 (below). `separated` = interval excludes 0 at the log_name level AND at the nuplan_drive level (SPEC 5).

## Per-arm (x100; NC / DAC / TTC / EP / C and PDMS -- never PDMS alone)

| arm | role | n | NC | DAC | TTC | EP | C | DDC | PDMS | PDMS 95 % (log bootstrap) | zeroed NC/DAC | stop-rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PRIOR_ha0p | model-free reference (this run) | 5 | 80.00 | 20.00 | 80.00 | 6.88 | 100.00 | 40.00 | **14.5337** | UNAVAILABLE (5 logs < 8: cannot rule) | 4 | 0.000 |
| R8_LEGAL_s1 | inference-seed floor arm | 5 | 100.00 | 60.00 | 100.00 | 14.26 | 100.00 | 90.00 | **40.9421** | UNAVAILABLE (5 logs < 8: cannot rule) | 2 | 0.200 |
| R8_LEGAL | BAR ARM | 5 | 100.00 | 60.00 | 100.00 | 15.93 | 100.00 | 100.00 | **41.6371** | UNAVAILABLE (5 logs < 8: cannot rule) | 2 | 0.200 |
| R8_PRIV | DIAGNOSTIC ONLY (never a bar) | 5 | 100.00 | 60.00 | 100.00 | 28.78 | 100.00 | 100.00 | **46.9929** | UNAVAILABLE (5 logs < 8: cannot rule) | 2 | 0.000 |
| R8_RT | beside; carries a bar only if SHIP-R8 passed | 5 | 100.00 | 40.00 | 100.00 | 14.88 | 100.00 | 100.00 | **29.5351** | UNAVAILABLE (5 logs < 8: cannot rule) | 3 | 0.200 |
| STOP | reference (banked floor, INHERITED) | 5 | 100.00 | 80.00 | 100.00 | 14.08 | 60.00 | 100.00 | **49.2011** | UNAVAILABLE (5 logs < 8: cannot rule) | 1 | BY CONSTRUCTION |
| CV | reference (banked floor, INHERITED) | 5 | 70.00 | 40.00 | 60.00 | 8.30 | 100.00 | 60.00 | **15.5345** | UNAVAILABLE (5 logs < 8: cannot rule) | 3 | 0.200 |
| HUMAN | reference (banked floor, INHERITED) | 5 | 100.00 | 100.00 | 100.00 | 83.99 | 100.00 | 100.00 | **93.3310** | UNAVAILABLE (5 logs < 8: cannot rule) | 0 | 0.000 |
| REFCV7_50400_LEGAL | reference (banked floor, INHERITED) | 5 | 100.00 | 60.00 | 100.00 | 44.17 | 100.00 | 80.00 | **53.4057** | UNAVAILABLE (5 logs < 8: cannot rule) | 2 | - |

## Paired differences (a - b, x100 PDMS; same tokens, same resampled logs)

| pair | delta | 95 % | log-level | drive-level | positive (both) | W/T/L |
|---|---|---|---|---|---|---|
| PRIOR_ha0p__minus__STOP | -34.6674 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 1/1/3 |
| PRIOR_ha0p__minus__CV | -1.0008 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 1/3/1 |
| PRIOR_ha0p__minus__HUMAN | -78.7973 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/0/5 |
| PRIOR_ha0p__minus__REFCV7_50400_LEGAL | -38.8720 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/2/3 |
| R8_LEGAL_s1__minus__STOP | -8.2589 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 3/0/2 |
| R8_LEGAL_s1__minus__CV | 25.4076 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 2/2/1 |
| R8_LEGAL_s1__minus__HUMAN | -52.3889 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/0/5 |
| R8_LEGAL_s1__minus__REFCV7_50400_LEGAL | -12.4635 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/2/3 |
| R8_LEGAL_s1__minus__PRIOR_ha0p | 26.4085 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 2/2/1 |
| R8_LEGAL__minus__STOP | -7.5639 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 3/0/2 |
| R8_LEGAL__minus__CV | 26.1026 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 2/2/1 |
| R8_LEGAL__minus__HUMAN | -51.6939 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/0/5 |
| R8_LEGAL__minus__REFCV7_50400_LEGAL | -11.7685 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/2/3 |
| R8_LEGAL__minus__PRIOR_ha0p | 27.1035 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 2/2/1 |
| R8_PRIV__minus__STOP | -2.2082 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 3/0/2 |
| R8_PRIV__minus__CV | 31.4583 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 3/2/0 |
| R8_PRIV__minus__HUMAN | -46.3382 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/0/5 |
| R8_PRIV__minus__REFCV7_50400_LEGAL | -6.4128 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/2/3 |
| R8_PRIV__minus__PRIOR_ha0p | 32.4592 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 3/2/0 |
| R8_RT__minus__STOP | -19.6660 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 2/0/3 |
| R8_RT__minus__CV | 14.0006 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 2/2/1 |
| R8_RT__minus__HUMAN | -63.7959 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/0/5 |
| R8_RT__minus__REFCV7_50400_LEGAL | -23.8706 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 0/2/3 |
| R8_RT__minus__PRIOR_ha0p | 15.0014 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 1/3/1 |
| R8_LEGAL__minus__R8_LEGAL_s1 | 0.6950 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 2/2/1 |
| R8_PRIV__minus__R8_LEGAL | 5.3557 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 1/4/0 |
| R8_RT__minus__R8_LEGAL | -12.1020 | UNAVAILABLE (5 logs < 8: cannot rule) | no | - | - | 2/2/1 |

**Inference-seed floor** (R8_LEGAL vs R8_LEGAL_s1) = 0.6950 PDMS x100. a difference below this floor is not an effect (one pair of inference draws: a lower bound on the noise, not an estimate of it).

## Bars

| bar | verdict | margin (x100) | floor | note |
|---|---|---|---|---|
| BAR-R8-N1 | **NOT_EVALUATED -- VALIDATION ONLY** | -7.5639 | 0.6950 | ONE training seed: separated is necessary, not sufficient (H-ESTIM-SEED-1). |
| BAR-R8-N2 | **UNAVAILABLE (summary_navhard.json not supplied: needs official two-stage EPDMS for R8_LEGAL, R8_LEGAL_s1, CV_official, STOP_zero, ECHO_ha0_ext and the paired interval vs STOP_zero)** | - | - |  |
| BAR-R8-N3 | **UNAVAILABLE (summary_warmup.json not supplied)** | - | - |  |
| BAR-R8-N4 | **NOT_EVALUATED -- VALIDATION ONLY** | -11.7685 | 0.6950 | MIXED-BACKEND: refcv7-50,400 LEGAL was scored on the Thor backend (INHERITED, MODEL_REGISTRY 15.4.1); the refcv8 backend is stated in the arm table -- comparabl |

R8_RT: NO_BAR (SHIP-R8 has not passed on the per-frame feed; R8_RT is reported beside, D-R8-SHIP-RETIME / D-NS8-3)
R8_PRIV: NO_BAR (diagnostic only, never a bar, never submitted)

## Strata (PDMS x100; delta vs STOP / refcv7-LEGAL; bold = an EFFECT: outside the inference floor AND separated at log and drive level)

### city

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| Singapore | 2 | 2 | 73.52 | 32.79 | 88.28 | 40.73 (no interval) | -14.77 (no interval) | 0.00 (no interval) | 1.73 |
| US | 3 | 3 | 20.39 | 60.14 | 30.15 | -39.76 (no interval) | -9.77 (no interval) | 8.93 (no interval) | 0.00 |

### command

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| straight | 2 | 2 | 30.58 | 56.92 | 45.23 | -26.35 (no interval) | -14.65 (no interval) | 13.39 (no interval) | 0.00 |
| turn | 3 | 3 | 49.01 | 44.05 | 58.86 | 4.96 (no interval) | -9.85 (no interval) | 0.00 (no interval) | 1.16 |
| unknown | 0 | - | - | - | - | - | - | - | - |
| malformed | 0 | - | - | - | - | - | - | - | - |

### t0_speed_band

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| [0,2) | 1 | 1 | 61.16 | 61.10 | 90.46 | 0.06 (no interval) | -29.31 (no interval) | 26.78 (no interval) | 0.01 |
| [2,8.33) | 3 | 3 | 49.01 | 44.05 | 58.86 | 4.96 (no interval) | -9.85 (no interval) | 0.00 (no interval) | 1.16 |
| [8.33,13.89) | 1 | 1 | 0.00 | 52.75 | 0.00 | -52.75 (no interval) | 0.00 (no interval) | 0.00 (no interval) | 0.00 |
| >=13.89 | 0 | - | - | - | - | - | - | - | - |
| invalid | 0 | - | - | - | - | - | - | - | - |

### ep_clause

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| fired | 0 | - | - | - | - | - | - | - | - |
| not_fired | 5 | 5 | 41.64 | 49.20 | 53.41 | -7.56 (no interval) | -11.77 (no interval) | 5.36 (no interval) | 0.69 |

## Controls
```
{
 "city_census": {
  "status": "OK",
  "sg_tokens_inputs": 1914,
  "sg_tokens_census": 1914,
  "n_tokens_inputs": 12146,
  "n_tokens_census": 12146
 },
 "ep_clause_source_consistent": {
  "max_abs_diff_m": 0.0,
  "n_compared": 30,
  "status": "OK"
 },
 "v0_inputs_vs_scorer_rows": {
  "max_abs_diff_mps": 2.2774418440008048e-07,
  "n": 5,
  "status": "OK"
 },
 "banked_STOP": {
  "status": "SKIPPED (subset)",
  "banked": 61.8202
 },
 "banked_CV": {
  "status": "SKIPPED (subset)",
  "banked": 20.6517
 },
 "banked_HUMAN": {
  "status": "SKIPPED (subset)",
  "banked": 94.5514
 },
 "banked_REFCV7_50400_LEGAL": {
  "status": "SKIPPED (subset)",
  "banked": 70.2048
 }
}
```

## Inputs each bar needs
- N1: R8_LEGAL + R8_LEGAL_s1 + banked STOP csv, FULL 12,146 tokens, EMA step 48,852.
- N4: as N1 + the refcv7-50,400 LEGAL csv (Thor backend: MIXED-BACKEND stated).
- N2: `summary_navhard.json` (parse7 schema, R8 arm names, official two-stage EPDMS) -- not rebuilt here (the CSV drops the stage-2 weights).
- N3: `summary_warmup.json` incl. R8_BLIND (must FAIL, else the panel is VOID).
- R8_RT: `--ship-r8-passed 1` AND the per-frame table (never b208c9c1...). R8_PRIV: never a bar.
