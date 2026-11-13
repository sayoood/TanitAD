# TF-N6 (navtest, pinned W3 harness): PRIMARY CONFIRMED. Paper level reached with a human-log fallback on 16 of 12,110 tokens. Teacher only stays NEAR.

Master Mind, 2026-11-13.

**Sources** (this folder):
- `out/n6_result.json`, `out/read.log`, `out/famread.log`, `out/fam_ci_N6H_vs_N5A.json`;
- `build/n6_build_report.json`, `chain.log`, `logs/`;
- harness CSVs in `harness_csv_n6.tar.xz`.

The whole run tree is banked privately on HF (`evidence/2026-11-13-tf-n6`).

## Provenance
- **Card:** `../PREREG_TF_N6.md`, sha256 `d693f68a…`, frozen 2026-11-13 before any N6 build.
- **K:** ledger entry n = 49. Bonferroni intervals use K = 49.
- **Chain (all times local):**
  - guards PASS 15:02;
  - derive 15:42 (0.38 CPU-h);
  - build 15:42, all 14 build controls PASS;
  - scoring 16:55;
  - read 16:55;
  - families 17:22;
  - `ZZN6_CHAIN DONE`.

  CPU only, on the dev box.
- **Evidence class:** MEASURED.
- **Tier:** T1. The pinned NAVSIM v1.1 harness executed every plan with its LQR tracker on the full 12,146-token seam.
- **Population:** primary 12,110 tokens over 136 logs.
- **Estimator:** paired log-cluster bootstrap over 136 logs, 10,000 draws, seed 20260927.
  - It answers whether another draw of LOGS would give the same result.
  - The arms are deterministic (C-DET).
- **Naming (binding):** "RJ5 teacher + P5-R + REPAIR-D + L1 + tolerance (NC/DDC kept, TTC/C free) + human-log fallback on 16 of 12,110 tokens".
  - N_teacher_tol = 4 and N_human = 16.
  - **The human-log replacement is NOT teacher output.**
- **Disclosure (from the card):**
  - The 20 residual tokens were read twice before: in the TF-N4 A1 anatomy and in TF-N5 A1. This is a navtest-seen design, so this read confirms it and is not an independent test.
  - N6-T's first harness attempt stopped at 5,964 of 12,146 rows with no summary line. The pinned harness wrapper's own retry scored all 12,146 rows, status PASS, C1 0.0.
  - The partial CSV is kept in `harness_csv_n6.tar.xz` under `quarantine_*`. No number comes from it.

## Controls: all PASS
- **Build:** C-COVB, C-DACMAP6, C-DERIV, C-LABELEGO, C-SET, C-REPLAY, C-DET, C-SELECT, C-CERT-B6, C-ORDER, C-X0, C-SEAMFN, C-UNCH, C-REACH.
- **Read:** C-COV, C-BUILD, C-UNCH, C-PIN (958 pins, TIP `families_run`), C-CERT, C-ID-C, C-EST, C-REG.
  - C-CERT: the harness read DAC 1 on all 20 tokens N6-H certified and all 4 that N6-T certified.
  - C-ID-C: on the tokens that did not change (12,126 in N6-H, 12,142 in N6-T), 0 tokens differ from N5-A by more than 1e-6 on any metric.
  - C-REG: N6-X changed 0 tokens and is bit-identical to N5-A.

## Verdict (frozen card bars)

**PRIMARY: CONFIRMED.**

| bar | result |
|---|---|
| 1. paper level for N6-H (PDMS ≥ 95.8 AND DAC ≥ 99.9) | **PASS:** PDMS **96.52** [96.25, 96.79], Bonferroni K49 [96.07, 96.98]; DAC **100.00** [100.00, 100.00]; **0 zero-DAC tokens** |
| 2. no harm to NC and DDC | **PASS:** ΔNC +0.000 [0, 0], ΔDDC +0.029 [0, +0.074]; 0 changed tokens with lower harness NC or DDC |
| 3. certificate integrity | **PASS:** 20 of 20 changed tokens read harness DAC = 1 |

- **ΔPDMS N6-H − N5-A:** +0.150 [+0.052, +0.281].
- **The declared tolerance (reported, not a bar):** ΔTTC −0.017 [−0.043, 0] and ΔC −0.017 [−0.055, 0].

### Attribution: teacher vs human (secondary, decides nothing)
| arm | NC | DAC | DDC | EP | TTC | C | **PDMS** | zero-DAC |
|---|---|---|---|---|---|---|---|---|
| N5-A (comparator, TF-N5 A1) | 99.74 | 99.83 | 99.05 | 93.25 | 98.51 | 99.69 | **96.37** | 20 |
| **N6-T (teacher only + tolerance)** | 99.74 | 99.87 | 99.06 | 93.29 | 98.50 | 99.68 | **96.39** | 16 |
| **N6-H (+ human fallback)** | 99.74 | **100.00** | 99.08 | 93.41 | 98.50 | 99.68 | **96.52** | **0** |

- N6-T − N5-A: +0.023 [0, +0.060].
- N6-H − N6-T: +0.127 [+0.040, +0.243].
- **N6-T alone is NEAR, not paper level.** DAC 99.87 with 16 zero-DAC tokens; the paper level allows at most 12.
- **The human replacements carry the bar:** the 16 tokens left after the teacher are the difference.

### Why each residual token went where it did (`build/n6_build_report.json`, descriptive)
- **15 of 20: no teacher iterate reaches DAC = 1.** These are the REPAIR-D and L1 iterates, 12 candidates per token. They are blocked by the candidate generator, not by the certificate.
- **5 of 20 have a DAC = 1 teacher candidate:**
  - the tolerance certified 4 (the N6-T changes);
  - on 1, the only DAC = 1 candidate failed NC (0.5 against N5-A's 1.0). Its TTC was 0, so it also scored builder PDMS 0.16. It went to the human log.
- **Human futures on the 16:** builder PDMS 0.88 to 1.00 (median 0.96). None needed the TTC/C tolerance. Human futures were available for all 20 tokens.

## Four metric families (executed_tracked, paired `fam_ci`, N6-H vs N5-A)
**Pre-registered guard: NO FLAG.** 18 of 61 paired metrics are separated from 0. All have |Δ| ≤ 0.0093, from 20 changed tokens, and all point toward improvement.

| family | metric | N5-A | N6-H | Δ [95 % CI] |
|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9857 | 0.9837 | −0.0020 [−0.0041, −0.0005] |
| | speed bias (m/s) | +0.6630 | +0.6616 | −0.0014 [−0.0036, +0.0002] |
| | accel MAE (m/s²) | 0.5763 | 0.5756 | −0.0007 [−0.0015, −0.0002] |
| | min TTC (s) | 12.900 | 12.901 | +0.0009 [−0.0001, +0.0030] |
| LATERAL | heading MAE (deg) | 2.820 | 2.816 | −0.0036 [−0.0096, +0.0020] |
| | curvature MAE (1/m) | 0.0118 | 0.0118 | −1e-5 [−2e-5, 0] |
| | cross-track MAE (m) | 0.4916 | 0.4904 | −0.0012 [−0.0023, −0.0004] |
| TACTICAL | goal-point error (m) | 3.819 | 3.810 | −0.0086 [−0.0179, −0.0022] |
| | goal bearing MAE (deg) | 2.913 | 2.905 | −0.0082 [−0.0160, −0.0027] |
| STRATEGIC | | PAUSED (PI 2026-09-19) | | |

Not computed:
- `goal_bearing_bias_deg`: the instrument emits no per-token array.
- `ego_progress.gt_*`: human-only metrics.

## What this does NOT show
- **That the TEACHER reaches paper level.** The teacher alone, with the tolerance, is DAC 99.87 with 16 zero-DAC tokens (NEAR).
- **That the human fallback is admissible as a teacher claim.** The PI approved it for overall performance: "yes you can replace if it will help the overall perfroamnce". Whether a 4 + 16 teacher + human mix may be quoted as "teacher fidelity at paper level" is the PI's call.
- **Anything about the student.**
- **Anything beyond navtest.** This is a navtest-seen confirmation read.

## Next (Rule Zero; the card's CONFIRMED branch)
1. **PI:** may the teacher + human mix be quoted as the teacher-fidelity result? The naming above is used until the PI rules.
2. **Teacher-only paper level** needs 4 more of the 16 tokens. All 16 are blocked by the candidate generator, not the certificate. The lever is a wider candidate generator (cap and margin schedule, or a lateral search), developed on navtrain, never on these tokens.
   - The navtrain dev set is being built now: REPAIR-D on the 16,133 RJ5 navtrain labels (`F:/tanitad_scratch/rj5nt/repair_nt`).
3. **Student labels:**
   - apply the same tolerance and human fallback to navtrain's residual DAC-0 labels (from the same REPAIR-D run), so the P5-C successor trains on a label set with no zero-DAC target;
   - the card asks for a tracker-aware check of the human-sourced labels before training.
