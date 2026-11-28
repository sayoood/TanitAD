# TF-N7 (navtest, pinned W3 harness): PRIMARY CONFIRMED. The teacher ALONE reaches paper level: no human-log token, no deep slow-down tier.

Master Mind, 2026-11-14.

**Sources** (this folder):
- `out/n7_result.json`, `out/read.log`, `out/famread.log`, `out/fam_ci_*.json`, `out/n7_families.json`, `fam/fam_*.json`;
- `build/out/n7_build_report.json`, `chain.log`, `logs/`;
- harness CSVs in `harness_csv_n7.tar.xz`.

The whole run tree is banked privately on HF (`Sayood/tanitad-refe-scorer1p6`, `evidence/2026-11-14-tf-n7`).

## Provenance
- **Card:** `../chain/PREREG_TF_N7.md`, sha256 `7eab0ff0…`, frozen 2026-11-13 18:18Z before any N7 build. Section 10 holds the MM rulings: PRIMARY = N7-T3, with no deep slow-down; N7-T, which includes T4, is secondary and decides nothing.
- **K:** ledger entry `TF-N7:7eab0ff0…:E3`, n = 50. Bonferroni intervals use K = 50.
- **Pins:** 970 (`N7_PINS.json`; C-PIN 0 mismatches, TIP `families_run`).
- **Chain:** CPU only, on the dev box. All times local (UTC + 1) as the chain logs them.
  - launched 2026-11-13 19:31Z;
  - derive jobs 21:12 and 23:47;
  - harness scoring 00:41 to 03:32;
  - read 03:33;
  - families after that;
  - families stopped at 04:25 local: 12,107 of 12,110 primary rows (see Disclosures);
  - relaunched 2026-11-14 03:27Z;
  - `ZZN7_CHAIN DONE` 05:07:43 local (04:07:43Z).
- **Evidence class:** MEASURED.
- **Tier:** T1. The pinned NAVSIM v1.1 harness executed every plan with its LQR tracker on the full 12,146-token seam.
- **Population:** primary 12,110 tokens over 136 logs.
- **Estimator:** paired log-cluster bootstrap over 136 logs, 10,000 draws, seed 20260927.
  - It answers "would another draw of LOGS say this?" only.
  - The arms are deterministic (C-DET), so it is silent on training and inference variance.
- **Naming (binding):** "RJ5 teacher + P5-R + REPAIR-D + L1 + wider generator (tiers T0 to T3) + tolerance (NC/DDC kept, TTC/C free), teacher only, N_teacher_tol = 18 of 20 residual tokens".
  - **There is no human-log token in N7-T3.**
  - T3 changes speed by at most s = 0.8. The T4 deep slow-down (30 to 50 %) is used only by N7-T.

## Disclosures
- **The navtest residual is read for the FOURTH time**: in the TF-N4 A1 anatomy, TF-N5 A1, TF-N6, and now here. This is a navtest-seen design, so the read confirms it and is not an independent test.
  - The generator's levers, grids, tier order and T4 floor were fixed on 17 navtrain residual tokens (`n7/gen/DEV_N7.md`; n = 17 on 7 logs).
  - No navtest token set any parameter.
- **Harness retries.** The pinned harness wrapper's first chain attempt failed six internal tries for N7-T3: each run stopped with no "Number of successful scenarios" summary, at 2,204 valid rows or fewer. The chain's second attempt scored all 12,146 rows: status PASS, `C1_max_abs_delta` 0.0, 2,087 s.
  - N7-T behaved the same way.
  - The partial CSVs are kept under `quarantine_*` in `harness_csv_n7.tar.xz`. **No number comes from them.**
  - The same failure mode was seen on N6-T's first attempt. It is most likely memory pressure from concurrent jobs on the dev box; that is a HYPOTHESIS, not measured.

- **Families-stage memory stop.** The first families pass scored 12,143 of 12,146 tokens. Three tokens raised `MemoryError` in the transparent scorer while the dev box was under memory load from concurrent jobs: `d9cf5623b4f85cad`, `e0cc59c7dc8c50bd` and `7a61f0a7cbe15387`.
  - The chain's structural check (12,107 of 12,110 primary rows) stopped it with `families_rows.FAIL`. That marker is kept in the run's `markers/`.
  - The relaunch re-scored only those 3 tokens (`transp_score.py` retries error rows) under the chain's free-memory gate. famprep, which skips error rows, then gave 12,110 primary rows.
  - The verdict (bars 1 to 3) was computed before the families stage and does not use these rows.

## Controls: all 9 PASS (`out/read.log`)
| control | result |
|---|---|
| C-COV | 12,146 rows / 12,110 primary, as expected |
| C-BUILD | PASS |
| C-UNCH | plan20 and seam files of the tokens off the applied set are bit-identical to N5-A |
| C-PIN | 970 pins, 0 mismatches |
| C-CERT | every certified token reads harness DAC = 1, in both arms |
| C-ID-C | every token outside the applied set equals its banked N5-A row on all 7 metrics, within 1e-6 (the 200 sentinels included) |
| C-N6SAME | the 4 tokens N6-T changed are identical in N7-T3 and N7-T |
| C-EST | estimator max abs deviation 0.0 |
| C-REG | the deliberate-regression arm N7-X changed 0 tokens and is bit-identical to N5-A |

## Verdict (frozen card bars)

**PRIMARY: CONFIRMED.** Bars 1, 2 and 3 hold for N7-T3.

| bar | result |
|---|---|
| 1. paper level for N7-T3 (PDMS ≥ 95.8 AND DAC ≥ 99.9; ≤ 12 zero-DAC tokens) | **PASS:** PDMS **96.499** [96.224, 96.772], Bonferroni K50 [96.054, 96.963]; DAC **99.9835** [99.9558, 100.0], Bonferroni [99.9293, 100.0]; **2 zero-DAC tokens** |
| 2. no harm to NC and DDC, and no regression vs N6-T | **PASS:** ΔNC 0.000 [0, 0], ΔDDC +0.029 [0, +0.074]; 0 changed tokens with lower harness NC or DDC; the 4 N6-T tokens are unchanged |
| 3. certificate integrity | **PASS:** 18 of 18 changed tokens read harness DAC = 1 |
| 4. four-family guard (reported; a flag goes to the PI and does not flip a bar) | **NO FLAG.** N7-T3 − N5-A, executed_tracked: speed bias −0.0007 m/s [−0.0025, +0.0001] (bound 0.25), speed MAE −0.0004 m/s (bound 0.15), cross-track MAE −0.0003 m (bound 0.05). N7-T: no flag |

- **Paired N7-T3 − N5-A:** PDMS **+0.132** [+0.041, +0.255]; DAC +0.149 [+0.048, +0.283].
- **The declared tolerance (reported, not a bar):** ΔTTC −0.017 [−0.043, 0], ΔC −0.033 [−0.110, 0], ΔEP +0.139 [+0.045, +0.265].
- **Tiers used by N7-T3:** T0 (the N6 generator) 4, T1 (L2 cap × margin) 11, T2 0, T3 (offsets, s ≥ 0.8) 3. **N_teacher_tol = 18 of 20.**
- **The two tokens still at DAC 0:**
  - `12b32c24d2cf5309`: no T0–T3 candidate reaches DAC = 1;
  - `7cb9260c8be35c5c`: its DAC-1 candidates exist in T1 but fail the certificate, on NC and PDMS (one also on DDC), per `remaining_zero_dac.blocking`.

| arm (12,110 primary) | PDMS [95 % CI] | DAC [95 % CI] | zero-DAC | level |
|---|---|---|---|---|
| **N7-T3** (PRIMARY) | **96.499** [96.224, 96.772] | **99.984** [99.956, 100.0] | 2 | **PAPER LEVEL** |
| N7-T (secondary; includes T4; decides nothing) | 96.507 [96.233, 96.780] | 99.992 [99.972, 100.0] | 1 | PAPER LEVEL |
| N6-T (teacher only, TF-N6) | 96.390 [96.056, 96.697] | 99.868 [99.746, 99.959] | 16 | NEAR |
| N5-A (comparator) | 96.367 [96.027, 96.680] | 99.835 [99.695, 99.941] | 20 | NEAR |

**Attribution** (descriptive, decides nothing):
| contrast | ΔPDMS [95 % CI] |
|---|---|
| N7-T3 − N6-T | +0.109 [+0.030, +0.220] |
| N7-T − N7-T3 (what T4 adds) | +0.008 [0, +0.028] |
| N7-T3 − N2-R | +0.890 [+0.533, +1.315] |

N7-T's secondary branch ("the result rests on T4") does **not** apply: N7-T3 reaches paper level without T4.

## The four metric families (N7-T3 − N5-A; executed_tracked; paired log-cluster bootstrap; `out/n7_families.json`, `out/fam_ci_N7T3_vs_N5A.json`)
| family | metrics | separated | max \|d\| |
|---|---|---|---|
| LONGITUDINAL | speed bias, speed MAE (guard), accel MAE, along-track MAE, min headway, min TTC | 0 of 6 | 0.0024 |
| LATERAL | cross-track MAE (guard), heading MAE, curvature MAE, yaw-rate MAE, cross bias | 0 of 5 | 0.0011 |
| TACTICAL | 5-way manoeuvre (collapsed), lateral decision, longitudinal decision, goal setting | 0 of 4 | 0.0032 |
| STRATEGIC | **PAUSED** (PI 2026-09-19); not a pass and not dropped | n/a | n/a |

**The changed tokens' own longitudinal change** (card bar 4, reported):
- On the 18 changed tokens, speed MAE moves by −0.290 m/s on average, and by at most 1.750 m/s in absolute value.
- By tier: T0 −0.624, T1 −0.244, T3 −0.014 m/s.
- So the changed plans track the logged speed slightly BETTER than N5-A did. N7-T's T4 token moves by −1.521 m/s.

Over the whole set the effect is too small to separate, as it should be: only 18 of 12,110 tokens changed.

## What this settles, and what it does not
- **Settled:** the privileged DriveRL teacher reaches the paper's navtest level (PDMS ≥ 95.8, DAC ≥ 99.9) **as teacher output alone** under the pinned harness.
  - This supersedes TF-N6's human-mix wording for the paper-level claim.
  - It removes the need to decide whether T4 is teacher output for this claim.
- **Not settled:**
  - This is a TEACHER score on navtest. It says nothing yet about the REFe student, which reads 91.196 (P6).
  - The 20-token residual is navtest-seen (see the disclosures).

## Next (Rule Zero; the card's REACHED row)
1. **Apply the same cascade to the navtrain labels** (the residual DAC-0 labels of the full set plus the rolled logs), so that the P5-C successor trains on a teacher label set with no zero-DAC labels.
   - It needs the RJ5 bulk labels of navtrain: two download-and-roll chains are running on the dev box, chain A at batch 124 of 220 and chain B at batch 20 of 219 at 03:35Z.
   - It also needs the pod's navtrain metric-cache parts for the builder scoring.
2. **Run the tracker-aware check on the T1/T3-edited labels before any student sees them.**
3. The PI's T4 question no longer gates this claim. It returns only if a navtrain cascade wants T4.
