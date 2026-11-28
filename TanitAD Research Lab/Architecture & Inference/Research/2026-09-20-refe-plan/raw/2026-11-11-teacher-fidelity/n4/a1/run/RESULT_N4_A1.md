# TF-N4 A1 (navtest, pinned W3 harness): REPAIR-D CONFIRMED, +0.72 PDMS over N2-R. Paper level: NEAR. PDMS 96.33 is above the paper's 95.8; DAC 99.79 is below its 99.9 (25 zero-DAC tokens remain, at most 12 allowed)

2026-11-13, Master Mind. Sources (this folder): `out/read.log`, `out/n4_result.json`, `out/n4_families.json`, `out/fam_ci_N4D_vs_N2R.json`, `build/n4_build_report.json`, `chain.log`, and the harness CSVs in `harness_csv_n4.tar.xz`. The full run tree is banked privately on HF: `Sayood/tanitad-refe-scorer1p6`, `evidence/2026-11-13-tf-n4-a1/n4a1_run.tar` plus its manifest.

## Provenance
- **Pre-registration:** `../PREREG_TF_N4_A1.md`, sha256 `319bedbf…`, frozen 2026-11-13 before the rerun.
  - It amends `PREREG_TF_N4.md` (sha256 `1d051cd1…`). That card's first read was VOID at C-DERIV (K entry n = 45, still counted).
  - A1's one change: labels are built from the nuPlan-DB ego, not the harness-cache ego.
- **K:** ledger entry n = 46. Bonferroni intervals use K = 46.
- **Chain:**
  - Guards PASS at 01:01Z (card, k, 946 pins, reach).
  - Build DONE 02:23Z, scoring DONE 07:51Z, read DONE 07:52Z.
  - The families stage FAILED at 08:20Z on an instrument wiring defect and was repaired at 08:25Z (section "Families", below).
  - CPU only, 0 GPU.
- **Evidence class:** MEASURED.
- **Tier:** T1. The pinned NAVSIM v1.1 harness executed every plan with its LQR tracker and scored it on navtest.
- **Population:** 12,110 primary tokens over 136 logs.
- **Estimator:** paired log-cluster bootstrap over the 136 logs, 10,000 draws, seed 20260927, 95 % percentile.
  - It answers whether another draw of LOGS would give the same result.
  - The arms are deterministic: C-DET re-ran 50 tokens, 50 / 50 equal; C-ID-C read 0 tokens over 1e-6 against the banked N2-R rows. So no inference or training variance hides behind the interval.
- **Naming:** every number below is "RJ5 teacher + P5-R + REPAIR-D", never "the teacher".
  - REPAIR-D shifts the teacher's target along the scorer's return vector and re-runs the unchanged label pipeline.
  - Its certificate guards NC, DDC, TTC and C against the base label using **logged agent futures** (`n4_repair.py:24-25`). That is admissible for a privileged teacher's labels, and it is named in any teacher claim. It is not admissible at a student's inference.

## Controls: all PASS
| control | result |
|---|---|
| C-DERIV (build) | 200 tokens re-derived equal the banked N2-R labels, max 0.0 m (bar 1e-9) |
| C-LABELEGO (build, new in A1) | 316 label-building rows, 316 on the DB ego, 0 not |
| C-COVB, C-DACMAP, C-DET, C-CERT-B, C-UNCH (build) | PASS. C-DACMAP: 12,146 / 12,146 builder DAC equal to the harness. C-DET: 50 / 50. C-CERT-B: 0 violations |
| C-COV, C-PIN, C-EST (read) | 12,146 tokens, 12,110 primary, 0 error rows; 946 pins match; estimator max \|dev\| 0.0 |
| C-UNCH (read) | every unrepaired token's plan and seam is bit-identical to N2-R |
| C-CERT (read) | every builder-certified token has harness DAC = 1: 91 / 91 (N4-D), 96 / 96 (N4-DM), 3 / 3 (N4-X) |
| C-ID-C (read) | every unrepaired token's harness row equals the banked N2-R row on all 7 metrics, max \|d\| 0.0 |
| C-REG (deliberate-regression arm N4-X) | repairs 3 of 116 (bar ≤ 10 %); PDMS 95.63 vs N2-R 95.61 (bar ≤ +0.05) |

## Verdicts (frozen card bars)
- **PRIMARY: CONFIRMED.**
  - N4-D − N2-R = **+0.717 [0.420, 1.065]** PDMS; Bonferroni (K = 46) [0.289, 1.352].
  - The four bars:

    | bar | result |
    |---|---|
    | Δ ≥ +0.50 | met |
    | CI lower bound > 0 | met |
    | repair rate ≥ 0.75 | met: **91 / 116 = 0.784** |
    | no-harm: every NC, DDC, TTC, C paired Δ ≥ −0.10 | met: NC 0.000, DDC +0.025, TTC 0.000, C 0.000 |

  - The gain comes from the repaired tokens. Each moves from PDMS 0 (DAC multiplier) to its full score, and EP rises by +0.71 [0.42, 1.06].
- **PAPER LEVEL: NEAR.**
  - N4-D PDMS **96.33** [95.98, 96.64] and DAC **99.79** [99.64, 99.91], against the paper's 95.8 / 99.9 (DriveRL Table 4).
  - Rule: PAPER LEVEL needs PDMS ≥ 95.8 AND DAC ≥ 99.9; NEAR needs ≥ 95.3 AND ≥ 99.5.
  - **PDMS clears the paper by +0.53. DAC does not:** 25 zero-DAC tokens remain, and 99.9 allows at most 12.
  - The card's estimate if navtrain transferred whole (ESTIMATED, not a bar) was PDMS about 96.5 and DAC about 100. Navtest repaired 78 % of the zero-DAC tokens, not navtrain's 100 % (5 / 5 and 30 / 30).

### Sub-scores (x100, primary 12,110; harness)
| arm | NC | DAC | DDC | EP | TTC | C | **PDMS** | zero-DAC tokens |
|---|---|---|---|---|---|---|---|---|
| N2-R (RJ5 → Tc20s → P5-R; TF-N2 A1) | 99.74 | 99.04 | 99.02 | 92.50 | 98.51 | 99.69 | **95.61** | 116 |
| **N4-D (N2-R + REPAIR-D, full certificate)** | 99.74 | 99.79 | 99.05 | 93.21 | 98.51 | 99.69 | **96.33** | 25 |
| N4-DM (REPAIR-D, DAC-only certificate; secondary) | 99.74 | 99.83 | 99.06 | 93.25 | 98.50 | 99.67 | **96.35** | 20 |
| N4-P2 (CERT2 row where a candidate passes, else N4-D; derived) | 99.74 | 99.80 | 99.05 | 93.21 | 98.51 | 99.69 | **96.33** | 24 |
| CERT2 (TF-N2's map-certified selection) | 99.74 | 99.35 | 99.02 | 92.79 | 98.51 | 99.69 | **95.91** | 79 |
| N4-X (deliberate regression) | 99.74 | 99.07 | 99.02 | 92.52 | 98.51 | 99.69 | 95.63 | 113 |
| human (NAVSIM, banked) | 100.0 | 100.0 | 98.82 | 86.94 | 100.0 | 99.96 | 94.55 | 0 |

**Where the remaining gap sits** (descriptive, never a bar). Each N4-D sub-score mean is swapped in turn for the paper's Table 4 value:

| swapped sub-score | PDMS gain |
|---|---|
| TTC | **+0.24** |
| DAC | +0.10 |
| NC | +0.05 |
| C | −0.12 |
| EP | **−0.71** (N4-D's EP 93.21 is above the paper's 91.5) |

- **DAC is no longer the dominant PDMS lever. It is the binding constraint of the paper-level rule.**
- **TTC is now the largest remaining PDMS lever.**

## Secondary contrasts (pre-registered; decide nothing)
| contrast | Δ PDMS | 95 % CI | Bonferroni (K = 46) |
|---|---|---|---|
| N4-D − CERT2 | +0.42 | [0.24, 0.63] | [0.15, 0.80] |
| N4-DM − N2-R | +0.74 | [0.44, 1.10] | [0.30, 1.40] |
| N4-D − N4-DM (value of the full no-harm guard) | −0.027 | [−0.066, −0.001] | [−0.097, 0.000] |
| N4-P2 − N2-R | +0.72 | [0.43, 1.07] | [0.29, 1.35] |
| CERT2 − N2-R | +0.30 | [0.13, 0.52] | [0.07, 0.65] |

- **N4-DM** drops the NC / DDC / TTC / C guard and keeps DAC only. It repairs 5 more tokens (96 of 116) and reads +0.027 PDMS higher.
  - It costs NC 99.744 → 99.740 and TTC 98.514 → 98.497. That is the guard doing its job: it refuses repairs that trade a collision or TTC margin for drivable area.
  - Being seen here, N4-DM is no primary claim.
- **N4-P2 ≈ N4-D.** CERT2's candidate swap adds nothing once REPAIR-D runs.
- **Human anchor:** N4-D − human = +1.77 [0.97, 2.66].

## Four metric families (executed_tracked, W-TRANSP on the harness metric cache, T1; `out/fam_ci_N4D_vs_N2R.json`, `out/n4_families.json`)
**Instrument repair, disclosed.** The chain's families stage FAILED, `fam_ci.py`: *"REFUSED: JSON carries no tokens/logs and no --tokens file was given"*.
- **Root cause:** the run tree pinned a `families_run.py` (sha256 `b5ad959c…`) older than 2207ef03.
  - 2207ef03 (2026-11-12 17:04 local) added the 3 lines that write the top-level `tokens` / `logs` arrays under `--per-window`.
  - The pinned `n4/fam_ci.py` needs those arrays to pair arms by token identity.
  - Class: **"built, tested, unreachable from its caller"**. The pairing was tested against the new producer while the chain pinned the old one.
- **Repair:** `fam_tokens_inject.py` (this folder) writes `tokens` / `logs` into COPIES of the two family JSONs. The values are exactly what the fixed `families_run.py` writes: read_rows over the same `rows_primary.jsonl`, filtered by arm, sorted by token. The originals are untouched.
  - The pinned `fam_ci.py` and the pinned famread then ran with the chain's own argv. No bar, bound, arm or estimator changed, and no family number had been computed before the repair.
- **Checks that the token order is right:**
  - `fam_ci.py` refuses any mismatch. It raises if the human's executed-manoeuvre ids differ between arms, and both arms carry them per token. It did not raise.
  - The N2-R column reproduces TF-N2 A1's banked N2-R families exactly: speed MAE 0.986, bias +0.663, accel MAE 0.576, heading 2.83, yaw rate 2.91, curvature 0.0118, cross-track 0.493, lateral / longitudinal decision accuracy 0.927 / 0.670.

**Pre-registered guard (against N2-R): NO FLAG** (`ZZN4_GUARD no flag`):

| guard check | N4-D − N2-R [95 % CI] | bound |
|---|---|---|
| speed bias | −0.0004 [−0.0016, +0.0002] m/s | 0.25 |
| speed MAE | −0.0002 [−0.0007, +0.0001] m/s | 0.15 |
| cross-track MAE | −0.0007 [−0.0013, −0.0003] m | 0.05 |

**Paired family table** (N4-D − N2-R; 61 paired metrics, 11 separated from 0, all of them tiny improvements on the 91 changed tokens):

| family | metric | N2-R | N4-D | Δ [95 % CI] |
|---|---|---|---|---|
| LONGITUDINAL | speed MAE (m/s) | 0.9858 | 0.9856 | −0.0002 [−0.0007, +0.0001] |
| | speed bias (m/s) | +0.6634 | +0.6630 | −0.0004 [−0.0016, +0.0002] |
| | accel MAE (m/s²) | 0.5763 | 0.5763 | +0.0000 [−0.0002, +0.0003] |
| | progress ratio (mean) | 1.1946 | 1.1945 | −0.0001 [−0.0005, +0.0001] |
| | target speed within 1 m/s | 0.6521 | 0.6522 | +0.0000 [−0.0000, +0.0001] |
| | distance keeping: min TTC (s), n 6,187 | 12.906 | 12.903 | −0.003 [−0.009, +0.000] |
| | min headway (m) / time gap (s) | 16.768 / 4.945 | 16.769 / 4.945 | +0.001 / +0.000 (both CIs contain 0) |
| | lead time-gap violations (share of 6,423) | 0.00187 | 0.00187 | 0 [0, 0] |
| LATERAL | heading MAE (deg) | 2.825 | 2.821 | **−0.005 [−0.008, −0.002]** |
| | yaw-rate MAE (deg/s) | 2.907 | 2.906 | **−0.001 [−0.003, −0.000]** |
| | curvature MAE (1/m) | 0.01182 | 0.01180 | **−1.4e-5 [−2.6e-5, −3.8e-6]** |
| | cross-track MAE (m) | 0.4925 | 0.4918 | **−0.0007 [−0.0013, −0.0003]** |
| TACTICAL | lateral decision accuracy / kappa | 0.9272 / 0.8683 | 0.9273 / 0.8684 | +0.0001 / +0.0001 (CI lower bound 0) |
| | longitudinal decision accuracy / kappa | 0.6701 / 0.5067 | 0.6701 / 0.5067 | 0 / −0.0000 |
| | 5-way manoeuvre (collapsed) accuracy | 0.7931 | 0.7931 | 0 [0, 0] |
| | goal-point error (m) | 3.821 | 3.819 | **−0.003 [−0.008, −0.000]** |
| STRATEGIC | | PAUSED (PI 2026-09-19); also unavailable in NAVSIM | | |

**Reading:**
- REPAIR-D buys its PDMS with drivable-area compliance on 91 tokens. Every family is unchanged or very slightly better.
- RJ5's longitudinal character carries over unchanged: speed bias +0.66 m/s against the human log, progress ratio 1.19. TF-N2 A1 measured it and the PI accepted it.

## What this does NOT show
- **That the registered teacher protocol reaches the paper.** This is RJ5 (jerk 5.0) + P5-R + REPAIR-D. REPAIR-D's certificate reads logged agent futures.
- **That a STUDENT trained on these labels gains this much.** That is P5-C's question, and P5-C is approved on P5-R labels, not on REPAIR-D labels.
- **That N4-DM is better than N4-D.** It is a secondary read and was seen here.

## Next lever (Rule Zero; the card's pre-registered branch "CONFIRMED, NEAR")
1. **Anatomy of the 25 residual zero-DAC tokens by cause:** iteration cap, margin, an NC / TTC trade blocking the certificate, or teacher intent beyond reach.
   - The builder logged only `applied: false` for them, with no reason. That is an instrument gap; the anatomy re-runs REPAIR-D on them with a per-iteration trace.
   - It is descriptive, never a bar.
2. **Raise the iteration cap / margin, developed on navtrain first,** then its own card, confirmed once on navtest.
   - Reaching PAPER needs 13 or more of the 25 repaired.
   - N4-DM's 5 extra repairs show that some of the 25 are guard-blocked, not unreachable.
3. **The TTC lever (+0.24 swap gain):** now the largest PDMS term.
