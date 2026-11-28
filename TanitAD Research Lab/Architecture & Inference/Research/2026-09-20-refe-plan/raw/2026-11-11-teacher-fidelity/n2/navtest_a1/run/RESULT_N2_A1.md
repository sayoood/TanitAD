# TF-N2 A1 (navtest, pinned W3 harness): RJ5 + P5-R CONFIRMED, +1.51 PDMS over N1-C. Paper level: BELOW, and the remaining gap is now DAC, not progress

2026-11-12, Master Mind. Sources: `read.log`, `n2_result.json` (this folder), the harness CSVs in `harness_csv_n2.tar.xz`.

## Provenance
- **Pre-registration:** `PREREG_TF_N2_A1.md`, sha256 `e51af742…`, frozen 2026-11-12T14:51:12Z. It amends `PREREG_TF_N2.md`, whose first read was VOID at C-R0 (K entry n = 43, still counted).
- **K:** ledger entry n = 44. Bonferroni intervals use K = 44.
- **Chain:**
  - Launched 15:52Z.
  - `score_T` failed at first (the MM's own E: archive copy shadowed F:; see `raw/2026-11-12-instruments/archive_root/NOTE.md`). It resumed 19:57Z with the same command.
  - `read` DONE 21:30Z. CPU only, 0 GPU.
- **Evidence class:** MEASURED.
- **Tier:** T1, executed and LQR-tracked by the pinned NAVSIM v1.1 harness on navtest.
- **Population:** 12,110 primary tokens over 136 logs. All 12,146 tokens are reported as a sensitivity check (+1.53).
- **Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile.
  - It answers whether another draw of LOGS would give the same result.
  - The rollouts are deterministic: the 2-thread re-roll reproduces the bank, 240 / 240 tokens at 0.0, C-R0. The harness is deterministic too: C-ID-H read 0 / 12,146 tokens over 1e-6, max |d| 0.0.
  - So no inference or training variance is hidden behind this interval.
- **Naming:** RJ5 (positive jerk limit 5.0 instead of the registered 1.0) is a different teacher setting. Every number below is "RJ5 teacher + P5-R", never "the teacher". The PI accepted RJ5's extra speed on 2026-11-12, with no 5 % speed warp.

## Controls: all PASS (`ZZN2_CONTROLS PASS`)
| control | result |
|---|---|
| C-R0 | the R0 identity subset re-rolled at 2 threads equals the bank: 240 / 240 tokens, max \|d\| 0.0 |
| C-ID-H | harness re-score of the T seam vs the banked CSV: 0 / 12,146 tokens over 1e-6, max \|d\| 0.0 on all 7 metrics |
| C-ID-C | re-derived N1-C plans vs the banked N1-C CSV, 200-token subset: 0 over 1e-6 |
| C-COV, C-BUILD, C-PIN, C-EST | PASS |
| C-REG | N2-W (deliberate regression) 52.40 against N2-R 95.61 |
| human diagnostic | 0 / 12,146 tokens over 1e-6 against the banked NAVSIM human CSV |

## Verdicts (frozen card bars)
- **PRIMARY: CONFIRMED.**
  - N2-R − N1-C = **+1.51 [1.15, 1.85]** PDMS; Bonferroni (K = 44) [0.95, 2.04].
  - ΔDAC +0.03 [−0.14, +0.18].
  - Bars: Δ ≥ +0.5, CI lower bound > 0, ΔDAC ≥ −0.25. All three are met.
- **PAPER LEVEL: BELOW.**
  - N2-R PDMS **95.61** [95.08, 96.08] with DAC **99.04** [98.58, 99.42], against the paper's 95.8 / 99.9 (DriveRL Table 4).
  - Rule: PAPER LEVEL needs PDMS ≥ 95.8 AND DAC ≥ 99.9; NEAR needs ≥ 95.3 AND ≥ 99.5. DAC misses both.
  - PDMS alone is −0.19 from the paper, and its CI contains 95.8.

### Sub-scores (x100, primary 12,110; harness)
| arm | NC | DAC | DDC | EP | TTC | C | **PDMS** | zero-DAC tokens |
|---|---|---|---|---|---|---|---|---|
| T (registered teacher seam) | 99.37 | 95.23 | 98.97 | 85.18 | 97.15 | 99.92 | **89.81** | 578 |
| N1-C (teacher → Tc20s; TF-N1 A1) | 99.68 | 99.01 | 99.09 | 88.76 | 98.76 | 99.81 | **94.10** | 120 |
| N2-C (RJ5 → Tc20s) | 99.72 | 98.85 | 99.03 | 92.33 | 98.39 | 99.60 | **95.36** | 139 |
| **N2-R (RJ5 → Tc20s warm start → P5-R)** | 99.74 | 99.04 | 99.02 | 92.50 | 98.51 | 99.69 | **95.61** | 116 |
| CERT2 (secondary, decides nothing) | 99.74 | 99.35 | 99.02 | 92.79 | 98.51 | 99.69 | **95.91** | 79 |
| N2-W (deliberate regression) | 86.78 | 71.10 | 88.96 | 54.91 | 67.56 | 86.00 | 52.40 | 3,500 |
| human (NAVSIM, banked) | 100.0 | 100.0 | 98.82 | 86.94 | 100.0 | 99.96 | 94.55 | 0 |

**Where the 0.19 to the paper sits** (descriptive, never a bar). Each sub-score mean is swapped in turn for the paper's Table 4 value:

| swapped sub-score | PDMS gain |
|---|---|
| DAC | **+0.82** |
| TTC | +0.24 |
| NC | +0.05 |
| C | −0.11 |
| EP | **−0.41** |

The formula applied to the means reads 95.04, a residual of 0.57 below the actual 95.61.
- **The progress gap that dominated TF-N1 (+1.13) is closed.** RJ5's EP 92.50 is now above the paper's.
- **The remaining lever is drivable-area compliance.**

## Secondary contrasts (pre-registered; decide nothing)
| contrast | Δ PDMS | 95 % CI | Bonferroni (K = 44) |
|---|---|---|---|
| N2-C − N1-C (the jerk unlock alone, both through Tc20s) | +1.26 | [0.90, 1.62] | [0.67, 1.84] |
| N2-R − N2-C (P5-R on RJ5 rollouts) | +0.25 | [0.12, 0.39] | [0.04, 0.48] |
| CERT2 − N2-R (map-certified selection) | +0.30 | [0.13, 0.52] | [0.07, 0.65] |
| CERT2 − N1-C | +1.81 | [1.49, 2.12] | [1.30, 2.32] |

- **CERT2's choice on the primary set:** N2-R 12,073, N1-C 21, T 10, N2-C 6. No candidate passes on 79 tokens.
- **CERT2's descriptive paper read:** PDMS 95.91 [95.46, 96.31], DAC 99.35 [99.01, 99.63].
  - That is BELOW by the joint rule, because DAC < 99.9, even though its PDMS point is above 95.8.
  - It is quoted as "RJ5 teacher + P5-R + map-certified selection", never as the teacher.

**Human anchor** (descriptive). Our NAVSIM human reads 94.55 against the paper's 94.8. N2-R − human = +1.06 [0.12, 2.04].

## Four metric families (DONE 2026-11-13 01:46 local; `out/n2_families.json`, `out/famread.log`, per-arm `fam/fam_*.json` in `fam_json.tar.xz`)
- **Basis:** executed_tracked frame, W-TRANSP scenery from the pinned `transp_score.py` (**not harness-identical**, as the frozen card labels it). Primary 12,110 tokens / 136 logs. T1.
- **Intervals:** per-arm LEVEL intervals, episode-cluster bootstrap over the 136 logs, 2,000 draws. These are not paired: no paired family CI was pre-registered for this card, and that is a work item (TF-N4 carries `fam_ci.py` paired). Two arms' intervals overlapping is no evidence of equality.
- **Pre-registered guard (TF-N2 §10, against N1-C): NO FLAG** (`ZZN2_GUARD no flag`):

| guard check | N2-R − N1-C | bound |
|---|---|---|
| speed bias | +0.3287 m/s | 0.5 |
| speed MAE | +0.2130 m/s | 0.3 |
| accel MAE | +0.0833 m/s² | 0.15 |
| abs speed bias of N2-R | 0.6634 m/s | 1.0 |

| family | metric | N2-R | N1-C | human |
|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.986 [0.902, 1.078] | 0.773 [0.732, 0.815] | 0.129 [0.123, 0.136] |
| | speed bias (m/s) | +0.663 [0.521, 0.802] | +0.335 [0.238, 0.424] | +0.039 [0.030, 0.047] |
| | accel MAE (m/s²) | 0.576 [0.537, 0.620] | 0.493 [0.473, 0.513] | 0.134 [0.127, 0.141] |
| | progress ratio (mean) | 1.195 | 1.097 | 1.024 |
| | target speed within 1 m/s | 0.652 | 0.731 | 1.000 |
| | distance keeping: mean min TTC (s) | 12.91 | 13.65 | 13.66 |
| | distance keeping: mean min headway (m) / time gap (s) | 16.77 / 4.94 | 16.89 / 5.00 | 16.77 / 5.01 |
| | time gap < 1 s, share of 6,423 lead windows | 12 (0.19 %) | 10 (0.16 %) | 53 (0.83 %) |
| LATERAL | heading MAE (deg) | 2.83 [2.57, 3.07] | 2.61 [2.40, 2.80] | 1.55 [1.40, 1.69] |
| | yaw-rate MAE (deg/s) | 2.91 [2.65, 3.15] | 2.83 [2.61, 3.03] | 1.97 [1.81, 2.12] |
| | curvature MAE (1/m) | 0.0118 [0.0110, 0.0126] | 0.0127 [0.0118, 0.0135] | 0.0100 [0.0092, 0.0108] |
| | cross-track MAE (m) | 0.493 [0.442, 0.545] | 0.391 [0.361, 0.421] | 0.147 [0.134, 0.160] |
| TACTICAL | lateral decision accuracy / kappa | 0.927 / 0.868 | 0.933 / 0.877 | 0.979 / 0.961 |
| | longitudinal decision accuracy / kappa | 0.670 / 0.507 | 0.683 / 0.525 | 0.953 / 0.929 |
| | 5-way manoeuvre (collapsed) accuracy | 0.793 | 0.799 | 0.959 |
| STRATEGIC | | PAUSED (PI 2026-09-19); also unavailable in NAVSIM | | |

**Reading (descriptive; the guard is the only pre-registered family decision):**
- **RJ5 buys its PDMS with speed.** N2-R drives about 0.66 m/s faster than the human log on average (progress ratio 1.19, against N1-C's 1.10). That is where EP rose. The guard passed: the extra speed is inside the bounds the PI's ruling allowed for. It is still a real behavioural change and travels with every "RJ5 teacher" number.
- **Distance keeping did not degrade materially:** time-gap violations 12 vs 10 of 6,423 lead windows (both far below the human log's 53). Mean min TTC is 0.7 s shorter.
- **Lateral:** heading, yaw rate and curvature are close to N1-C. Cross-track MAE is larger (0.49 vs 0.39 m), consistent with covering more ground per window.
- **Tactical:** decision agreement with the human log is about 1 point lower than N1-C on both axes.

## What this does NOT show
- That the registered teacher protocol (jerk 1.0) reaches the paper. RJ5 is a changed setting, accepted by the PI as a lever. Whether the paper's teacher ran with a different jerk limit is not established.
- That a STUDENT trained on these labels gains this much. That is P5-C's question.
- That CERT2 reaches paper level. It is a secondary read, below on DAC, and it was seen here: any primary claim for it needs its own card and fresh evidence.

## Next lever (Rule Zero)
- The gap is DAC-first (+0.82 swap gain). CERT2's map-certified selection over 4 candidates already lifts DAC to 99.35 and PDMS +0.30, but 79 tokens have no passing candidate.
- The next arm attacks those tokens: a wider certified candidate set (for example RJ3 rollouts, already banked on navtrain, plus R0), and a DAC repair for tokens where no candidate passes.
- It is developed on navtrain held-out and confirmed once on navtest under its own pre-registration.
