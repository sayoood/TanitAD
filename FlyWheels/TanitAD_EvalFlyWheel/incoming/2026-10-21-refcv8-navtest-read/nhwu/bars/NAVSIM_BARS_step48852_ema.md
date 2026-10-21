# NAVSIM_BARS_step48852_ema

**RESULT (EMA 48,852, the bar read)**

Tier: NavSim open-loop benchmark (T1-family), zero-shot PhysicalAI-AV -> nuPlan cameras, non-parity; NEVER closed loop. NOT leaderboard-submitted (PI 2026-10-04).
Run: `C:\Users\Admin\refcv8_pf\navtest_read\raw\full` step 48852 weights ema; n = 12146 tokens / 136 logs.

## Estimator
`navsim_log_cluster_bootstrap` / `paired_navsim_log_cluster_bootstrap` (IMPORTED from `C:\lgs\Lfb47a58376\taniteval\adapters\navsim_ci.py`, sha256 `00e0e39ef6cfc9de`); cluster unit **log_name** (sensitivity unit nuplan_drive), B = 2000, seed 0, percentile 95 %. Answers: *another draw of LOGS?* only. Blind to training variance (one training seed) and to inference variance, which is read from R8_LEGAL vs R8_LEGAL_s1 (below). `separated` = interval excludes 0 at the log_name level AND at the nuplan_drive level (SPEC 5).

## Per-arm (x100; NC / DAC / TTC / EP / C and PDMS -- never PDMS alone)

| arm | role | n | NC | DAC | TTC | EP | C | DDC | PDMS | PDMS 95 % (log bootstrap) | zeroed NC/DAC | stop-rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PRIOR_ha0p | model-free reference (this run) | 12146 | 88.50 | 74.05 | 78.66 | 54.16 | 99.62 | 83.50 | **58.6613** | [56.01, 61.53] | 3955 | 0.060 |
| R8_LEGAL | BAR ARM | 12146 | 85.31 | 76.84 | 77.15 | 52.07 | 99.52 | 91.36 | **57.8426** | [55.79, 59.86] | 4027 | 0.050 |
| R8_LEGAL_s1 | inference-seed floor arm | 12146 | 85.34 | 76.68 | 77.41 | 52.07 | 99.56 | 91.35 | **57.9198** | [55.74, 60.05] | 4033 | 0.049 |
| R8_PRIV | DIAGNOSTIC ONLY (never a bar) | 12146 | 85.57 | 78.23 | 77.12 | 53.96 | 99.76 | 91.89 | **59.2322** | [57.04, 61.31] | 3879 | 0.042 |
| R8_RT | beside; carries a bar only if SHIP-R8 passed | 12146 | 84.83 | 74.47 | 76.39 | 51.90 | 99.46 | 90.21 | **56.6700** | [54.62, 58.66] | 4247 | 0.040 |
| R8_RT_s1 | extra arm (no bar) | 12146 | 84.91 | 74.41 | 76.82 | 52.01 | 99.48 | 90.22 | **56.8582** | [54.72, 58.90] | 4237 | 0.040 |
| STOP | reference (banked floor, INHERITED) | 12146 | 97.40 | 96.53 | 96.40 | 31.00 | 69.44 | 99.59 | **61.8202** | [60.70, 63.08] | 732 | BY CONSTRUCTION |
| CV | reference (banked floor, INHERITED) | 12146 | 68.02 | 57.84 | 50.03 | 19.44 | 100.00 | 79.91 | **20.6517** | [19.20, 22.22] | 8130 | 0.051 |
| HUMAN | reference (banked floor, INHERITED) | 12146 | 100.00 | 100.00 | 100.00 | 86.96 | 99.90 | 98.83 | **94.5514** | [93.79, 95.23] | 0 | 0.019 |
| REFCV7_50400_LEGAL | reference (banked floor, INHERITED) | 12146 | 89.81 | 87.37 | 79.13 | 68.27 | 99.94 | 94.71 | **70.2048** | [68.55, 71.82] | 2624 | - |

## Paired differences (a - b, x100 PDMS; same tokens, same resampled logs)

| pair | delta | 95 % | log-level | drive-level | positive (both) | W/T/L |
|---|---|---|---|---|---|---|
| PRIOR_ha0p__minus__STOP | -3.1589 | [-5.53, -0.65] | sep | [-5.81, -0.09] | False | 6743/1211/4192 |
| PRIOR_ha0p__minus__CV | 38.0096 | [35.10, 41.07] | sep | [34.55, 41.48] | True | 7476/3683/987 |
| PRIOR_ha0p__minus__HUMAN | -35.8901 | [-38.28, -33.33] | sep | [-38.84, -32.67] | False | 2739/1975/7432 |
| PRIOR_ha0p__minus__REFCV7_50400_LEGAL | -11.5435 | [-13.86, -9.04] | sep | [-14.33, -8.25] | False | 3531/3104/5511 |
| R8_LEGAL__minus__STOP | -3.9777 | [-6.37, -1.65] | sep | [-6.82, -1.11] | False | 6872/918/4356 |
| R8_LEGAL__minus__CV | 37.1909 | [34.55, 39.80] | sep | [34.43, 39.90] | True | 7370/3368/1408 |
| R8_LEGAL__minus__HUMAN | -36.7088 | [-38.89, -34.63] | sep | [-39.46, -33.95] | False | 2200/1848/8098 |
| R8_LEGAL__minus__REFCV7_50400_LEGAL | -12.3623 | [-13.90, -10.87] | sep | [-14.11, -10.55] | False | 2900/3667/5579 |
| R8_LEGAL__minus__PRIOR_ha0p | -0.8187 | [-3.46, 1.82] | no | [-3.87, 2.08] | False | 4270/3114/4762 |
| R8_LEGAL_s1__minus__STOP | -3.9005 | [-6.41, -1.55] | sep | [-6.90, -0.90] | False | 6870/931/4345 |
| R8_LEGAL_s1__minus__CV | 37.2681 | [34.51, 39.99] | sep | [34.38, 40.14] | True | 7359/3374/1413 |
| R8_LEGAL_s1__minus__HUMAN | -36.6316 | [-38.92, -34.52] | sep | [-39.55, -33.73] | False | 2194/1844/8108 |
| R8_LEGAL_s1__minus__REFCV7_50400_LEGAL | -12.2851 | [-13.93, -10.78] | sep | [-14.05, -10.52] | False | 2919/3634/5593 |
| R8_LEGAL_s1__minus__PRIOR_ha0p | -0.7415 | [-3.49, 1.92] | no | [-3.88, 2.36] | False | 4272/3121/4753 |
| R8_PRIV__minus__STOP | -2.5880 | [-5.24, -0.17] | sep | [-5.72, 0.64] | False | 7042/879/4225 |
| R8_PRIV__minus__CV | 38.5806 | [35.90, 41.26] | sep | [35.81, 41.43] | True | 7572/3282/1292 |
| R8_PRIV__minus__HUMAN | -35.3192 | [-37.57, -33.27] | sep | [-38.15, -32.56] | False | 2338/1854/7954 |
| R8_PRIV__minus__REFCV7_50400_LEGAL | -10.9726 | [-12.66, -9.41] | sep | [-13.03, -8.84] | False | 2961/3759/5426 |
| R8_PRIV__minus__PRIOR_ha0p | 0.5709 | [-2.21, 3.25] | no | [-2.58, 3.67] | False | 4429/3059/4658 |
| R8_RT__minus__STOP | -5.1503 | [-7.60, -2.85] | sep | [-8.13, -2.26] | False | 6606/912/4628 |
| R8_RT__minus__CV | 36.0183 | [33.46, 38.62] | sep | [33.25, 38.75] | True | 7173/3541/1432 |
| R8_RT__minus__HUMAN | -37.8815 | [-40.02, -35.88] | sep | [-40.67, -35.08] | False | 2262/1846/8038 |
| R8_RT__minus__REFCV7_50400_LEGAL | -13.5349 | [-15.17, -11.98] | sep | [-15.39, -11.60] | False | 2862/3658/5626 |
| R8_RT__minus__PRIOR_ha0p | -1.9913 | [-4.67, 0.60] | no | [-5.05, 1.00] | False | 4162/3282/4702 |
| R8_RT_s1__minus__STOP | -4.9620 | [-7.45, -2.61] | sep | [-8.06, -1.93] | False | 6612/928/4606 |
| R8_RT_s1__minus__CV | 36.2066 | [33.51, 38.88] | sep | [33.31, 39.07] | True | 7180/3529/1437 |
| R8_RT_s1__minus__HUMAN | -37.6932 | [-39.95, -35.66] | sep | [-40.64, -34.80] | False | 2249/1854/8043 |
| R8_RT_s1__minus__REFCV7_50400_LEGAL | -13.3466 | [-15.00, -11.83] | sep | [-15.22, -11.47] | False | 2882/3622/5642 |
| R8_RT_s1__minus__PRIOR_ha0p | -1.8031 | [-4.52, 0.89] | no | [-4.99, 1.34] | False | 4177/3270/4699 |
| R8_LEGAL__minus__R8_LEGAL_s1 | -0.0772 | [-0.50, 0.38] | no | [-0.45, 0.35] | False | 3024/6103/3019 |
| R8_PRIV__minus__R8_LEGAL | 1.3897 | [0.84, 1.95] | sep | [0.78, 2.12] | True | 1612/9123/1411 |
| R8_RT__minus__R8_LEGAL | -1.1726 | [-1.52, -0.88] | sep | [-1.53, -0.81] | False | 3352/6514/2280 |

**Inference-seed floor** (R8_LEGAL vs R8_LEGAL_s1) = 0.0772 PDMS x100. a difference below this floor is not an effect (one pair of inference draws: a lower bound on the noise, not an estimate of it).

## Bars

| bar | verdict | margin (x100) | floor | note |
|---|---|---|---|---|
| BAR-R8-N1 | **FAILED** | -3.9777 | 0.0772 | ONE training seed: separated is necessary, not sufficient (H-ESTIM-SEED-1). |
| BAR-R8-N2 | **FAILED** | -0.1223 | -0.0080 |  |
| BAR-R8-N3 | **FAILED** | -0.0340 | 0.0173 | OK (R8_BLIND fails N3 as it must) |
| BAR-R8-N4 | **FAILED** | -12.3623 | 0.0772 | MIXED-BACKEND: refcv7-50,400 LEGAL was scored on the Thor backend (INHERITED, MODEL_REGISTRY 15.4.1); the refcv8 backend is stated in the arm table -- comparabl |

R8_RT: carries a bar (see JSON)
R8_PRIV: NO_BAR (diagnostic only, never a bar, never submitted)

## Strata (PDMS x100; delta vs STOP / refcv7-LEGAL; bold = an EFFECT: outside the inference floor AND separated at log and drive level)

### city

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| Singapore | 1914 | 13 | 58.02 | 58.42 | 71.68 | -0.40 [-4.41, 3.99] | **-13.66 [-17.72, -9.88]** | 0.00 [0.00, 0.00] | 0.77 |
| US | 10232 | 123 | 57.81 | 62.46 | 69.93 | **-4.65 [-7.24, -2.05]** | **-12.12 [-13.72, -10.46]** | **1.65 [1.04, 2.26]** | 0.05 |

### command

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| straight | 8070 | 134 | 56.90 | 62.97 | 72.23 | **-6.06 [-8.84, -3.56]** | **-15.32 [-17.13, -13.34]** | **1.74 [1.04, 2.48]** | 0.09 |
| turn | 4076 | 113 | 59.70 | 59.55 | 66.20 | 0.15 [-2.87, 3.24] | **-6.50 [-8.47, -4.65]** | **0.69 [0.17, 1.26]** | 0.06 |
| unknown | 0 | - | - | - | - | - | - | - | - |
| malformed | 0 | - | - | - | - | - | - | - | - |

### t0_speed_band

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| [0,2) | 2539 | 130 | 76.01 | 65.15 | 76.29 | **10.85 [7.44, 14.07]** | -0.28 [-1.97, 1.37] | 0.27 [-0.22, 0.72] | 0.07 |
| [2,8.33) | 7354 | 131 | 50.87 | 64.84 | 66.48 | **-13.97 [-17.02, -11.11]** | **-15.61 [-17.73, -13.47]** | **0.82 [0.31, 1.36]** | 0.04 |
| [8.33,13.89) | 2190 | 113 | 59.45 | 48.36 | 75.08 | **11.09 [7.72, 14.51]** | **-15.63 [-18.52, -12.44]** | **4.45 [2.96, 6.03]** | 0.25 |
| >=13.89 | 63 | 9 | 83.79 | 42.77 | 89.99 | **41.02 [22.53, 60.42]** | -6.20 [-19.28, 1.42] | **6.45 [2.36, 14.57]** | 4.15 |
| invalid | 0 | - | - | - | - | - | - | - | - |

### ep_clause

| level | n | logs | R8_LEGAL PDMS | STOP | REFCV7_50400_LEGAL | LEGAL-STOP | LEGAL-R7 | PRIV-LEGAL | s0-s1 floor |
|---|---|---|---|---|---|---|---|---|---|
| fired | 1296 | 116 | 56.02 | 93.07 | 51.44 | **-37.05 [-41.34, -32.92]** | 4.58 [-0.15, 8.86] | **-1.61 [-3.04, -0.27]** | 0.03 |
| not_fired | 10850 | 136 | 58.06 | 58.09 | 72.45 | -0.03 [-2.31, 2.23] | **-14.39 [-16.08, -12.74]** | **1.75 [1.13, 2.38]** | 0.09 |

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
  "max_abs_diff_m": 9.823679647524841e-10,
  "n_compared": 85022,
  "definition": "PDM reference proposal alone (oracle_compliant_progress_m)",
  "status": "OK"
 },
 "mcp_hook_identity": {
  "max_abs_diff_m": 0.0,
  "n": 85022,
  "agent_raised_max": {
   "PRIOR_ha0p": 2446,
   "R8_LEGAL": 2530,
   "R8_LEGAL_s1": 2522,
   "R8_PRIV": 2599,
   "R8_RT": 2521,
   "R8_RT_s1": 2524,
   "STOP": 606
  },
  "status": "OK"
 },
 "v0_inputs_vs_scorer_rows": {
  "max_abs_diff_mps": 4.765258783123727e-07,
  "n": 12146,
  "status": "OK"
 },
 "banked_STOP": {
  "status": "OK",
  "read": 61.8202,
  "banked": 61.8202
 },
 "banked_CV": {
  "status": "OK",
  "read": 20.6517,
  "banked": 20.6517
 },
 "banked_HUMAN": {
  "status": "OK",
  "read": 94.5514,
  "banked": 94.5514
 },
 "banked_REFCV7_50400_LEGAL": {
  "status": "OK",
  "read": 70.2048,
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
