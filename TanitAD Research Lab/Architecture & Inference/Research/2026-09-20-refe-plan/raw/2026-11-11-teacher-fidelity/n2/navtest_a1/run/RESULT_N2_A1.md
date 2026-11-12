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

## Four metric families
- **PENDING.** The chain's `families` stage, W-TRANSP on the DB route and labelled "not harness-identical" by the frozen card, started 22:30 local and needs about 80 min.
- This note is updated with the families table, the longitudinal guard and their CIs when it ends.
- STRATEGIC is PAUSED (PI 2026-09-19) and is also unavailable in NAVSIM.

## What this does NOT show
- That the registered teacher protocol (jerk 1.0) reaches the paper. RJ5 is a changed setting, accepted by the PI as a lever. Whether the paper's teacher ran with a different jerk limit is not established.
- That a STUDENT trained on these labels gains this much. That is P5-C's question.
- That CERT2 reaches paper level. It is a secondary read, below on DAC, and it was seen here: any primary claim for it needs its own card and fresh evidence.

## Next lever (Rule Zero)
- The gap is DAC-first (+0.82 swap gain). CERT2's map-certified selection over 4 candidates already lifts DAC to 99.35 and PDMS +0.30, but 79 tokens have no passing candidate.
- The next arm attacks those tokens: a wider certified candidate set (for example RJ3 rollouts, already banked on navtrain, plus R0), and a DAC repair for tokens where no candidate passes.
- It is developed on navtrain held-out and confirmed once on navtest under its own pre-registration.
