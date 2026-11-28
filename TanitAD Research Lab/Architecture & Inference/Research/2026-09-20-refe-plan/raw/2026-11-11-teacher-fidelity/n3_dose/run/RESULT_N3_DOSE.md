# TF-N3 jerk dose (navtrain, EXPLORATORY): 5.0 stays. 8.0 is INERT, because a second cap (`max_jerk_long` 5.0) binds first.

2026-11-12, Master Mind. Source: `TFN3_DOSE_RESULT.json` (this folder), chain `ALL.done` 16:02Z.

- **Arms:** RJ3 (jerk 3.0) and RJ8 (jerk 8.0), rolled at 1 thread on the same 469 navtrain tokens / 13 logs.
- **Comparator:** the banked RJ5 and R0 rows.
- **Control:** R0 identity, 30 tokens, PASS.
- **Estimator:** paired log-cluster bootstrap, B 10,000, seed 20260927. It answers "another draw of LOGS" only.
- **Scorer:** W-TRANSP on the DB-rebuilt scenery, so the contrasts are valid but the levels are not harness-identical.
- **Bar:** written in the analyzer before any RJ3 or RJ8 row existed.

## Verdict (mechanical): DECISION = "5.0 stays"
| arm | dPDMS vs RJ5.R | A (CI lower bound > 0) | B (DAC, NC, C) | C (longitudinal guard) | beats RJ5 |
|---|---|---|---|---|---|
| RJ3 | +0.10 [−0.10, +0.39] | FAIL | pass (DAC +0.21, 1 better / 0 worse flips) | pass (speed bias −0.04 m/s) | no |
| RJ8 | **0.0 [0.0, 0.0]** | FAIL | pass (all 0) | pass (all 0) | no |

**RJ8 is structurally identical to RJ5, not merely "not better" (MM check).**
- Its `traj20` equals RJ5's on 469/469 tokens, max |d| **0.0** m. RJ3 against RJ5 differs by up to 1.32 m, so the comparison can see a change.
- The read-back confirms why: the built planner holds `positive_jerk_limit_when_nonnegative_acc` 8.0 **and** `max_jerk_long` 5.0, and the second cap binds.
- ⇒ The positive-jerk knob saturates at 5.0. A dose above 5 would also have to raise `max_jerk_long`. That is untested, and it risks NAVSIM's comfort (C) jerk bound. It is not run now.

## Monotone picture (form R, descriptive only)
| jerk | EP | DAC | TTC | PDMS | speed bias vs human (m/s) |
|---|---|---|---|---|---|
| 1.0 (R0) | 90.88 | 99.15 | 97.01 | 94.42 | 0.35 |
| 3.0 | 93.32 | 99.15 | 96.38 | 95.10 | 0.60 |
| 5.0 | 93.37 | 98.93 | 96.38 | 95.00 | 0.64 |
| 8.0 | = 5.0 (saturated) | | | | |

The EP gain is almost entirely reached by 3.0. Between 3 and 5 it is flat, and 3.0 is a little closer to the human's speed. The bar keeps 5.0 because 3.0 does not separate.

## Framing correction (MM)
On navtrain, jerk 5 against R0 with **both in form R** is **+0.58 [−0.25, +1.41]**, not separated. The +1.12 [0.40, 1.98] quoted for the re-roll is RJ5.R against **R0.C**: the jerk unlock plus the P5-R conversion. TF-N2's PRIMARY (N2-R − N1-C) is that same combined contrast, so the card is consistent. But "the jerk unlock alone is worth +1.1" is not what navtrain shows.

## Next (Rule Zero)
- The progress lever stays jerk 5 → P5-R, read on navtest by TF-N2 A1 (running).
- If it lands below the paper, the next arm is the DAC lever, CERT2: map-certified selection, read as the secondary of TF-N2 A1.
- The remaining untested progress direction (jerk above 5 with `max_jerk_long` raised) carries a comfort risk and goes to the backlog.
