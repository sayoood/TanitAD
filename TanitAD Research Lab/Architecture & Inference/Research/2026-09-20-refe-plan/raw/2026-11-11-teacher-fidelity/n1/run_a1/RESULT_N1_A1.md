# TF-N1 (Amendment A1, pinned W3 harness): navtest lever CONFIRMED, paper level NOT reached (BELOW)

2026-11-12, Master Mind. Sources: `read.log`, `n1_a1_summary.json` (a compact copy of `n1_a1_result.json`, sha256 `cf438bca…`), and the harness CSVs in `csv/`.

## Provenance
- **Pre-registration:** `../PREREG_TF_N1_A1.md`, sha256 `c26f7b8a…`, frozen 10:56:49Z. It amends `PREREG_TF_N1.md` (sha256 `a5dd339e…`).
- **K:** ledger entry n = 42. Bonferroni intervals use K = 42. The void first read (n = 41) stays counted.
- **Chain:** launched 10:57:49Z, ended `ZZN1A1_CHAIN DONE` at 12:15:34Z. CPU only, 0 GPU.
- **Evidence class:** MEASURED.
- **Tier:** T1, executed and LQR-tracked by the pinned NAVSIM v1.1 harness on navtest.
- **Population:** 12,110 primary tokens over 136 logs. All 12,146 tokens are reported as a sensitivity check.
- **Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile. It answers whether another draw of logs would give the same result. The plans and the harness are deterministic: the T seam re-score was bit-identical.
- **Scorer of record:** the pinned W3 harness CSVs (`score_p4_a1.py` → `score_e.py`). The `primary.contrast` string in the JSON still says "W-TRANSP"; that is a stale label.
  - MM check: the N1-C CSV's all-token mean is 94.0798, which equals N1-T 89.7861 plus Δ 4.2937 (the all-token sensitivity).
  - N1-T reads 89.806 on the 12,110. That is the banked harness value; W-TRANSP read 89.7998.

## Controls: all PASS
| control | result |
|---|---|
| C-ID-H (early and in the read) | harness re-score of the T seam vs the banked CSV: 0 / 12,146 tokens over 1e-6; max \|Δ\| 0.0 on all 7 metrics |
| C-COV | 12,146 tokens, 12,110 primary, 0 error lines |
| C-BUILD, C-PIN, C-EST | PASS |
| C-REG | N1-W 52.38 against N1-T 89.81 |
| Human diagnostic | 0 tokens over 1e-6 against the banked NAVSIM human CSV |

## Verdicts (base card §6 bars, unchanged)
- **PRIMARY: CONFIRMED.**
  - ΔPDMS (N1-C − N1-T) = **+4.29 [3.33, 5.32]**; Bonferroni (K = 42) [2.77, 6.04].
  - ΔDAC +3.78 [2.85, 4.76].
  - N1-C DAC **99.01** [98.60, 99.36].
  - Bars: Δ ≥ +3.0, CI lower bound > 0, DAC ≥ 98.5. All three are met.
- **PAPER LEVEL: BELOW.**
  - N1-C PDMS **94.10** [93.60, 94.59] and DAC 99.01, against the paper's 95.8 / 99.9 (DriveRL Table 4).
  - Gap −1.70 PDMS.
- **SECONDARY CERT** (teacher + map-certified selection; decides nothing):
  - PDMS 94.75 [94.30, 95.19], DAC 99.73.
  - CERT − N1-C = +0.65 [0.44, 0.89].
  - Choice on the primary set: N1-C 12,023, N1-M 61, N1-T 26; none pass on 33.

### Sub-scores (x100, primary 12,110; harness)
| arm | NC | DAC | DDC | EP | TTC | C | **PDMS** |
|---|---|---|---|---|---|---|---|
| N1-T (teacher, 8-pose plan) | 99.37 | 95.23 | 98.97 | 85.18 | 97.15 | 99.92 | **89.81** |
| **N1-C (teacher → Tc20s)** | 99.68 | 99.01 | 99.09 | 88.76 | 98.76 | 99.81 | **94.10** |
| N1-M | 98.98 | 98.68 | 99.10 | 88.03 | 97.43 | 99.53 | 92.98 |
| CERT (secondary) | 99.67 | 99.73 | 99.10 | 89.41 | 98.73 | 99.80 | 94.75 |
| N1-W (deliberate regression) | 87.08 | 72.15 | 89.82 | 53.51 | 68.59 | 87.23 | 52.38 |

**Where the 1.70 to the paper sits** (descriptive, never a bar). The PDMS formula is applied to the sub-score means, and each sub-score is swapped in turn for the paper's Table 4 value:

| swapped sub-score | PDMS gain |
|---|---|
| EP | +1.13 |
| DAC | +0.84 |
| TTC | +0.14 |
| NC | +0.11 |
| C | −0.13 |

The formula applied to the means reads 93.53, a residual of 0.57 below the actual 94.10. So the gap is mostly **ego progress**, then **drivable-area compliance**.

**Human anchor** (descriptive). Our NAVSIM human reads 94.55, against the paper's 94.8. N1-C − human = −0.45 [−1.21, +0.35].

## Four metric families
- **Basis.** Executed, tracked, N1-C against N1-T, on W-TRANSP scenery, which is NOT harness-identical (objects shift 1–15 cm, see `../a1/WTRANSP_DIAG.md`).
- **Point values only.** The read computes no CI for the family metrics. A CI is a missing instrument: a work item, not an omission to hide.

| family | metric | N1-T | N1-C | human |
|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) / within 1 m/s | 0.73 / 0.749 | 0.77 / 0.731 | 0.13 / 1.000 |
| | accel MAE (m/s²) | 0.43 | 0.49 | 0.13 |
| | progress ratio vs human | 1.107 | 1.097 | 1.024 |
| | min headway (m) / min time gap (s) / mean min TTC (s, n closing) | 16.63 / 4.94 / 12.99 (4,773) | 16.89 / 5.00 / 13.65 (4,677) | 16.77 / 5.01 / 13.66 (4,639) |
| LATERAL | heading MAE (deg) / yaw-rate MAE (deg/s) | 2.95 / 3.00 | 2.61 / 2.83 | 1.55 / 1.97 |
| | curvature MAE (1/m) / cross-track MAE (m) | 0.0125 / 0.389 | 0.0127 / 0.391 | 0.0100 / 0.147 |
| TACTICAL | lateral decision acc (κ) | 0.931 (0.874) | 0.933 (0.877) | — |
| | longitudinal decision acc (κ) | 0.684 (0.528) | 0.683 (0.525) | — |
| | 5-way collapsed acc (κ) | 0.800 (0.746) | 0.799 (0.745) | — |
| | goal-point error (m) / bearing MAE (deg) | 2.84 / 2.39 | 2.97 / 2.42 | — |
| STRATEGIC | — | PAUSED (PI 2026-09-19; also unavailable in NAVSIM) | PAUSED | PAUSED |

**Reading.** The Tc20s conversion buys DAC (+3.8) and TTC by pre-distorting the plan for the tracker. It barely moves how the car drives against the human:
- heading is better;
- speed and acceleration are slightly worse;
- tactical decisions are unchanged.

The residual gap to the paper is mostly ego progress.

## What this does NOT show
- That a STUDENT trained on Tc20s labels gains this much. That is P5-C's question (C1 vs C0).
- That the privileged teacher reaches the paper. It does not, on this protocol: 94.10 against 95.8. E3 (outcome A) attributes the teacher-vs-paper gap to protocol and input, not to the checkpoint.

## Next lever (Rule Zero; decided in the MM's next turn)
- The gap is EP-first (+1.13), then DAC (+0.84).
- CERT's map-certified selection already gets DAC to 99.73. EP is the larger and untested lever.
- The next TF arm targets EP: the teacher's progress against PDM-Closed's normaliser on the tracked plan. It gets its own pre-registration.
