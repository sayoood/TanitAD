# NAVSIM_BARS_control_refcv7_standin_FULLSPLIT

**VALIDATION ONLY -- PIPELINE CONTROL: refcv7-50,400 arms wearing refcv8 names, NOT a refcv8 result**

Tier: NavSim open-loop benchmark (T1-family), zero-shot PhysicalAI-AV -> nuPlan cameras, non-parity; NEVER closed loop. NOT leaderboard-submitted (PI 2026-10-04).
Run: `None` step 50400 weights ema; n = 12146 tokens / 136 logs.

> **VALIDATION ONLY -- n = 12146, NO VERDICT.** control run. Numbers exercise the path; they are never refcv8's NavSim performance.

## Estimator
`navsim_log_cluster_bootstrap` / `paired_navsim_log_cluster_bootstrap` (IMPORTED from `C:\lgs\Lfb47a58376\taniteval\adapters\navsim_ci.py`, sha256 `00e0e39ef6cfc9de`); cluster unit **log_name** (sensitivity unit nuplan_drive), B = 2000, seed 0, percentile 95 %. Answers: *another draw of LOGS?* only. Blind to training variance (one training seed) and to inference variance, which is read from R8_LEGAL vs R8_LEGAL_s1 (below). `separated` = interval excludes 0 at the log_name level AND at the nuplan_drive level (SPEC 5).

## Per-arm (x100; NC / DAC / TTC / EP / C and PDMS -- never PDMS alone)

| arm | role | n | NC | DAC | TTC | EP | C | DDC | PDMS | PDMS 95 % (log bootstrap) | zeroed NC/DAC | stop-rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R8_LEGAL | BAR ARM | 12146 | 89.81 | 87.37 | 79.13 | 68.27 | 99.94 | 94.71 | **70.2048** | [68.55, 71.82] | 2624 | - |
| R8_LEGAL_s1 | inference-seed floor arm | 12146 | 89.32 | 87.23 | 78.83 | 68.09 | 99.92 | 94.66 | **69.8506** | [68.18, 71.50] | 2691 | - |
| R8_PRIV | DIAGNOSTIC ONLY (never a bar) | 12146 | 89.59 | 87.21 | 78.73 | 68.35 | 99.93 | 94.67 | **69.9904** | [68.29, 71.64] | 2664 | - |
| REFCV7_50400_LEGAL | reference (banked floor, INHERITED) | 12146 | 89.81 | 87.37 | 79.13 | 68.27 | 99.94 | 94.71 | **70.2048** | [68.55, 71.82] | 2624 | - |
| STOP | reference (banked floor, INHERITED) | 12146 | 97.40 | 96.53 | 96.40 | 31.00 | 69.44 | 99.59 | **61.8202** | [60.70, 63.08] | 732 | - |
| CV | reference (banked floor, INHERITED) | 12146 | 68.02 | 57.84 | 50.03 | 19.44 | 100.00 | 79.91 | **20.6517** | [19.20, 22.22] | 8130 | - |
| HUMAN | reference (banked floor, INHERITED) | 12146 | 100.00 | 100.00 | 100.00 | 86.96 | 99.90 | 98.83 | **94.5514** | [93.79, 95.23] | 0 | - |

## Paired differences (a - b, x100 PDMS; same tokens, same resampled logs)

| pair | delta | 95 % | log-level | drive-level | positive (both) | W/T/L |
|---|---|---|---|---|---|---|
| R8_LEGAL__minus__STOP | 8.3846 | [6.36, 10.24] | sep | [6.00, 10.32] | True | 8207/778/3161 |
| R8_LEGAL__minus__CV | 49.5532 | [47.29, 51.76] | sep | [47.11, 51.53] | True | 9013/2560/573 |
| R8_LEGAL__minus__HUMAN | -24.3466 | [-26.17, -22.62] | sep | [-26.71, -22.21] | False | 3792/2093/6261 |
| R8_LEGAL__minus__REFCV7_50400_LEGAL | 0.0000 | [0.00, 0.00] | no | [0.00, 0.00] | False | 0/12146/0 |
| R8_LEGAL_s1__minus__STOP | 8.0304 | [5.97, 9.94] | sep | [5.70, 9.96] | True | 8185/743/3218 |
| R8_LEGAL_s1__minus__CV | 49.1990 | [46.91, 51.39] | sep | [46.75, 51.22] | True | 8997/2602/547 |
| R8_LEGAL_s1__minus__HUMAN | -24.7008 | [-26.59, -22.93] | sep | [-27.07, -22.56] | False | 3807/2091/6248 |
| R8_LEGAL_s1__minus__REFCV7_50400_LEGAL | -0.3542 | [-0.69, 0.00] | no | [-0.67, -0.06] | False | 3399/5376/3371 |
| R8_PRIV__minus__STOP | 8.1702 | [6.12, 10.07] | sep | [5.79, 10.18] | True | 8185/757/3204 |
| R8_PRIV__minus__CV | 49.3387 | [47.06, 51.55] | sep | [46.87, 51.34] | True | 9006/2595/545 |
| R8_PRIV__minus__HUMAN | -24.5610 | [-26.45, -22.81] | sep | [-26.97, -22.35] | False | 3844/2104/6198 |
| R8_PRIV__minus__REFCV7_50400_LEGAL | -0.2144 | [-0.40, -0.04] | sep | [-0.39, -0.05] | False | 301/11704/141 |
| R8_LEGAL__minus__R8_LEGAL_s1 | 0.3542 | [-0.00, 0.69] | no | [0.06, 0.67] | False | 3371/5376/3399 |
| R8_PRIV__minus__R8_LEGAL | -0.2144 | [-0.40, -0.04] | sep | [-0.39, -0.05] | False | 301/11704/141 |

**Inference-seed floor** (R8_LEGAL vs R8_LEGAL_s1) = 0.3542 PDMS x100. a difference below this floor is not an effect (one pair of inference draws: a lower bound on the noise, not an estimate of it).

## Bars

| bar | verdict | margin (x100) | floor | note |
|---|---|---|---|---|
| BAR-R8-N1 | **NOT_EVALUATED -- VALIDATION ONLY** | 8.3846 | 0.3542 | ONE training seed: separated is necessary, not sufficient (H-ESTIM-SEED-1). |
| BAR-R8-N2 | **UNAVAILABLE (summary_navhard.json not supplied: needs official two-stage EPDMS for R8_LEGAL, R8_LEGAL_s1, CV_official, STOP_zero, ECHO_ha0_ext and the paired interval vs STOP_zero)** | - | - |  |
| BAR-R8-N3 | **UNAVAILABLE (summary_warmup.json not supplied)** | - | - |  |
| BAR-R8-N4 | **NOT_EVALUATED -- VALIDATION ONLY** | 0.0000 | 0.3542 | MIXED-BACKEND: refcv7-50,400 LEGAL was scored on the Thor backend (INHERITED, MODEL_REGISTRY 15.4.1); the refcv8 backend is stated in the arm table -- comparabl |

R8_RT: NO_BAR (R8_RT not in this run)
R8_PRIV: NO_BAR (diagnostic only, never a bar, never submitted)

## Strata (PDMS x100; delta vs STOP / refcv7-LEGAL; bold = an EFFECT: outside the inference floor AND separated at log and drive level)

### city

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| Singapore | 1914 | 13 | 71.68 | 58.42 | 71.68 | **13.26 [10.88, 15.94]** | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.18 |
| US | 10232 | 123 | 69.93 | 62.46 | 69.93 | **7.47 [5.23, 9.63]** | 0.00 [0.00, 0.00] | -0.25 [-0.47, -0.04] | 0.39 |

### command

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| straight | 8070 | 134 | 72.23 | 62.97 | 72.23 | **9.26 [6.70, 11.61]** | 0.00 [0.00, 0.00] | -0.25 [-0.45, -0.06] | 0.06 |
| turn | 4076 | 113 | 66.20 | 59.55 | 66.20 | **6.66 [3.76, 9.28]** | 0.00 [0.00, 0.00] | -0.15 [-0.48, 0.14] | 0.95 |
| unknown | 0 | - | - | - | - | - | - | - | - |
| malformed | 0 | - | - | - | - | - | - | - | - |

### t0_speed_band

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| [0,2) | 2539 | 130 | 76.29 | 65.15 | 76.29 | **11.14 [7.21, 14.89]** | 0.00 [0.00, 0.00] | **0.39 [0.05, 0.76]** | 0.18 |
| [2,8.33) | 7354 | 131 | 66.48 | 64.84 | 66.48 | 1.64 [-0.86, 3.99] | 0.00 [0.00, 0.00] | -0.42 [-0.65, -0.21] | 0.54 |
| [8.33,13.89) | 2190 | 113 | 75.08 | 48.36 | 75.08 | **26.72 [23.22, 29.97]** | 0.00 [0.00, 0.00] | -0.24 [-0.64, 0.13] | 0.38 |
| >=13.89 | 63 | 9 | 89.99 | 42.77 | 89.99 | **47.22 [33.04, 65.10]** | 0.00 [0.00, 0.00] | 0.26 [-1.41, 3.89] | 0.28 |
| invalid | 0 | - | - | - | - | - | - | - | - |

### ep_clause

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| fired | 1006 | 111 | 51.84 | 93.73 | 51.84 | **-41.89 [-47.03, -36.33]** | 0.00 [0.00, 0.00] | -1.08 [-1.73, -0.46] | 1.64 |
| not_fired | 11140 | 136 | 71.86 | 58.94 | 71.86 | **12.92 [11.27, 14.45]** | 0.00 [0.00, 0.00] | -0.14 [-0.32, 0.04] | 0.24 |

## Controls
```
{
 "note": "stand-in control; arm names are borrowed (R8_LEGAL=R7_VMAXOFF, R8_LEGAL_s1=R7_A1_s1, R8_PRIV=R7_A1)"
}
```

## Inputs each bar needs
- N1: R8_LEGAL + R8_LEGAL_s1 + banked STOP csv, FULL 12,146 tokens, EMA step 48,852.
- N4: as N1 + the refcv7-50,400 LEGAL csv (Thor backend: MIXED-BACKEND stated).
- N2: `summary_navhard.json` (parse7 schema, R8 arm names, official two-stage EPDMS) -- not rebuilt here (the CSV drops the stage-2 weights).
- N3: `summary_warmup.json` incl. R8_BLIND (must FAIL, else the panel is VOID).
- R8_RT: `--ship-r8-passed 1` AND the per-frame table (never b208c9c1...). R8_PRIV: never a bar.
