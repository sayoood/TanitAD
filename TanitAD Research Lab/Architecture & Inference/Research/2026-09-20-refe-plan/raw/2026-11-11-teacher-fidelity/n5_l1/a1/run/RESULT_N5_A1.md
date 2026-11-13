# TF-N5 A1 (navtest, pinned W3 harness): L1 NOT CONFIRMED. It repaired 4 of the 20 non-guard REPAIR-D residuals (bar 5); paper level stays NEAR (PDMS 96.37, DAC 99.83, 20 zero-DAC tokens)

2026-11-13, Master Mind. Sources (this folder): `out/read.log`, `out/n5_result.json`, `out/n5_families.json`, `out/fam_ci_N5A_vs_N4D.json`, `build/n5_build_report.json`, `chain.log`, harness CSVs in `harness_csv_n5.tar.xz`. The run tree, including the TF-N5 void, is banked privately on HF (`evidence/2026-11-13-tf-n5-a1`).

## Provenance
- **Card:** `../PREREG_TF_N5_A1.md`, sha256 `96a65416…`, frozen 2026-11-13.
  - It amends `PREREG_TF_N5.md` (sha256 `7411444c…`). That card's first run was VOID at build: the merge ran out of memory before any control or number (K entry n = 47, still counted).
- **K:** ledger entry n = 48. Bonferroni intervals use K = 48.
- **Chain (all times Z):** guards PASS 10:49; derive 11:11; build 11:11 (all 10 build controls PASS); scoring 11:40; read 11:40; families 12:02; `ZZN5_CHAIN DONE`. CPU only.
- **Evidence class:** MEASURED.
- **Tier:** T1. The pinned NAVSIM v1.1 harness executed every plan with its LQR tracker on the full 12,146-token seam.
- **Population:** primary 12,110 tokens over 136 logs.
- **Estimator:** paired log-cluster bootstrap over 136 logs, 10,000 draws, seed 20260927.
  - It answers whether another draw of LOGS would give the same result.
  - The repair count uses an exact binomial interval.
  - Arms are deterministic: C-DET re-ran all 116 tokens, 116 / 116 bit-identical.
- **Naming:** "RJ5 teacher + P5-R + REPAIR-D + L1 on the REPAIR-D residual", with a scorer no-harm guard that reads logged agent futures (teacher-only).

## Controls: all PASS
- **Build:** C-COVB, C-DACMAP (12,146 / 12,146), C-DERIV (200 tokens, max 0.0 m), C-LABELEGO (316 / 316 DB ego), C-DET (116 / 116; N5-A = N5-R on 25 / 25), C-SET (116 = 91 + 25; split 5 / 20), C-CERT-B (0 violations), C-SEAMFN (0 / 12,146), C-UNCH, C-REACH.
- **Read:** C-COV, C-PIN, C-CERT, C-ID-C, C-EST, C-REG (N5-X, the sign-flipped arm: 0 repairs, Δ 0.000).

## Verdict (frozen card bars)
- **PRIMARY: NOT CONFIRMED.** Branch `repair_count_below_5`.

  | bar | result |
  |---|---|
  | 1. repair count ≥ 5 of 20 non-guard | **4 of 20 FAILED**, exact 95 % [0.06, 0.44] |
  | 2. no harm | PASS: every paired Δ ≥ 0; NC/TTC/C 0.000, DDC +0.004; identity outside the L1 set |
  | 3. certificate integrity | PASS |
  | 4. no new failure | PASS (0) |

  - Overall L1 repaired **5 of 25**: the 4 non-guard tokens plus 1 of the 5 guard-blocked ones, on a different candidate that passes the full certificate.
  - ΔPDMS N5-A − N4-D = +0.041 [0.008, 0.090] (not a bar).
- **PAPER LEVEL: NEAR.** N5-A PDMS **96.37** [96.03, 96.68], DAC **99.83** [99.69, 99.94], 20 zero-DAC tokens. The paper needs DAC ≥ 99.9, at most 12 zero-DAC tokens.
- **Reading:** the frozen bar was set to beat what extra iterations alone bought REPAIR-D on these tokens (cap 20 certified 4 of 25, seen on navtest). L1 reached exactly that level and did not beat it.
- **N5-R (L1 replacing REPAIR-D, builder only):** 86 of 116, against REPAIR-D's 91. **L1 is not a better general repair.**

### Sub-scores (x100, primary 12,110; harness)
| arm | NC | DAC | DDC | EP | TTC | C | **PDMS** | zero-DAC |
|---|---|---|---|---|---|---|---|---|
| N4-D (comparator, TF-N4 A1) | 99.74 | 99.79 | 99.05 | 93.21 | 98.51 | 99.69 | **96.33** | 25 |
| **N5-A (N4-D + L1 on the 25)** | 99.74 | 99.83 | 99.05 | 93.25 | 98.51 | 99.69 | **96.37** | 20 |

Swap decomposition for N5-A (descriptive): TTC +0.24, DAC +0.06, NC +0.05, C −0.12, EP −0.73.

## Per-token L1 status on the 25 (builder, descriptive; these tokens are now read twice)
| status | tokens |
|---|---|
| CERTIFIED | 5 |
| CAP_EXHAUSTED (5 iterations, still outside) | 13 |
| INFEASIBLE (at some step the corridor is narrower than the footprint plus margin: no lateral offset exists) | 5 |
| CERT_BLOCKED (a DAC = 1 candidate refused by the NC / DDC / TTC / C guard) | 2 |

## Four metric families (executed_tracked, W-TRANSP, paired `fam_ci`; tip `families_run.py`, end to end without repair)
**Pre-registered guard (N5-A against N4-D): NO FLAG.**

| check | Δ | bound |
|---|---|---|
| speed bias | +0.00005 m/s | 0.25 |
| speed MAE | +0.00005 m/s | 0.15 |
| cross-track MAE | −0.00013 m | 0.05 |

15 of 61 paired metrics are separated from 0, all of magnitude ≤ 1e-3 on 5 changed tokens.

| family | metric | N4-D | N5-A | Δ [95 % CI] |
|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9856 | 0.9857 | +0.00005 [+0.000004, +0.0001] |
| | speed bias (m/s) | +0.663 | +0.663 | +0.00005 [+0.000004, +0.0001] |
| | accel MAE (m/s²) | 0.5763 | 0.5763 | +0.00002 [−0.000001, +0.00006] |
| | progress ratio | 1.194 | 1.195 | +0.00002 [+0.000001, +0.00004] |
| | min TTC (s) / min time gap (s) | 12.90 / 4.945 | 12.90 / 4.945 | −0.0001 / −0.0001 (CIs contain 0) |
| LATERAL | heading MAE (deg) | 2.821 | 2.820 | −0.0006 [−0.0014, −0.0001] |
| | yaw-rate MAE (deg/s) | 2.906 | 2.905 | −0.0004 [−0.0018, +0.0005] |
| | curvature MAE (1/m) | 0.0118 | 0.0118 | −1.2e-6 [−4e-6, +6e-7] |
| | cross-track MAE (m) | 0.4918 | 0.4916 | −0.0001 [−0.0004, −0.000002] |
| TACTICAL | lateral / longitudinal decision accuracy, 5-way | 0.9273 / 0.6701 / 0.7931 | same | 0 |
| | goal-point error (m) | 3.819 | 3.819 | +0.00005 (CI contains 0) |
| STRATEGIC | | PAUSED (PI 2026-09-19) | | |

## What this does NOT show
- **That lateral repair cannot reach the paper.** It shows that L1's two-sided interval repair, at REPAIR-D's cap and margin, repairs no more than extra iterations of REPAIR-D do.
- **Anything about the student.**

## Next lever (Rule Zero; the card's branch "NOT CONFIRMED, repair count < 5")
Remaining gap to paper level: 8 tokens (20 → at most 12).
1. **Cap / margin schedule (L2) and a longitudinal repair (slow down) for the 5 INFEASIBLE tokens.**
   - Both need a development set NOT drawn from these 25, which have now been read twice.
   - The RJ5 navtrain labels of the 198 local logs (16,133 rows) are being built now; REPAIR-D on them shows whether navtrain has residuals to develop on.
2. **PI decisions (named blockers):**
   - **L3:** a no-harm tolerance for guard-blocked tokens.
   - **A certified human-log fallback on the residual tokens.** Every human trajectory has DAC = 1, so this very likely reaches DAC 99.9. It replaces the teacher's output on about 0.17 % of tokens, and whether that may count toward the teacher paper-level goal is the PI's call.
