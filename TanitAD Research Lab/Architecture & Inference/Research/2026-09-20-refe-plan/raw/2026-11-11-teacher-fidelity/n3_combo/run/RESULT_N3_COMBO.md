# TF-N3 combo (navtrain, EXPLORATORY): jerk 5 + 5 Hz is NOT carried to navtest. Jerk 5 alone stays the lever.

2026-11-12, Master Mind. Source: `TFN3_COMBO_RESULT.json` (this folder), chain `ALL.done` 16:19 local (15:19Z).

- **Arm:** RJ5H5 = `--pos-jerk-limit 5.0 --policy-interval-s 0.2`, rolled at 1 thread on the TF-N2 re-roll's 469 navtrain tokens / 13 logs; the other arms are the banked re-roll rows.
- **Control:** R0 identity, 30 tokens at 1 thread vs the banked R0 rows: max |dxy| 0.0 m, PASS.
- **Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927. It answers "another draw of LOGS" only.
- **Scorer:** W-TRANSP on the DB-rebuilt scenery (navtrain has no harness cache), so the paired contrasts are valid but the levels are not harness-identical.
- **Bar:** fixed in `tfn3_combo_analyze.py`, last modified 16:06:40 local, before the chain launch (16:07:44) and before any RJ5H5 row.

## Verdict (computed by the analyzer): WORTH_CARRYING_TO_NAVTEST = false
| bar | value | result |
|---|---|---|
| A: CI lower bound of dPDMS (RJ5H5.R − RJ5.R) > 0 | **−0.10 [−0.72, +0.33]** | FAIL |
| B: DAC point delta ≥ 0 and worse-flips ≤ better-flips | −0.21 [−0.68, 0.0]; flips 1 worse / 0 better | FAIL |
| B: NC | 0.0; 0 / 0 | pass |
| B: C | 0.0; 0 / 0 | pass |

## Paired deltas, form R (PDMS ×100, [95 % CI])
| contrast | PDMS | EP | DAC | TTC |
|---|---|---|---|---|
| RJ5H5 − R0 | +0.48 [−0.89, +1.70] | +2.34 | −0.43 | −0.43 |
| RJ5H5 − RJ5 | −0.10 [−0.72, +0.33] | −0.15 | −0.21 | +0.21 |
| RJ5H5 − R5HZ | +0.23 [−0.48, +0.99] | +1.36 | −0.21 | −0.43 |

**Longitudinal mechanism (RJ5H5.R − RJ5.R):**
- speed bias +0.027 m/s
- speed MAE +0.049 m/s
- accel MAE +0.022 m/s²

The two knobs do not add. The 5 Hz policy interval adds no progress on top of the jerk unlock, costs DAC, and drives slightly further from the human.

## Next lever (Rule Zero)
- The teacher's navtest gap after TF-N1 was EP-first (+1.13), then DAC (+0.84).
- Jerk 5 (TF-N2 A1, running) is the EP lever.
- The measured DAC lever is CERT's map-certified selection. On N1 it gave +0.65 PDMS and DAC 99.73, and TF-N2 A1 reads it as the pre-registered secondary CERT2.
- The next card is decided on the TF-N2 A1 result.
- Meanwhile, a jerk-limit dose check (3.0 and 8.0 against 5.0) on these 469 navtrain tokens tests whether 5.0 is the right point on the EP curve.
