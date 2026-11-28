# MM reading: the SHUF permutation null for PROBE-N1D-FEAT (beside; moves no committed band), 2026-11-19

**Inputs**
- Parent: `../probe/MM_READING_N1DWPROBE.md`. Its committed SHUF control FAILED: max |acc - 0.338215| was 0.0403 against a band of 0.02, with 6 of 40 cells out of band, all at permutation seed 1.
- Result: `SHUF_NULL.json` (sha256 fbe99610), run log `shuf.log`, code `code/n1dwt0t1_shufnull.py` and `code/run_shuf2.sh`.
- Pod CPU, zero training.

**What changed before any null value was read**
- The run was planned at 50 permutation seeds per null with 24 workers. It was stopped at 200 of 5,000 fits, with no null value read.
- It was relaunched at **25 seeds per null** (seeds 2 to 26) with 6 workers, because the pod container is capped at 15.3 CPU cores and other jobs share them.
- With 25 draws, the "95 % range" is close to the null's min and max, so it is a coarse range.

**Evidence class:** MEASURED. It answers one question only: what accuracy does this probe design give when the training-fold labels carry no information?

## Result
| | tree probe (decides in the parent) | logistic probe (beside) |
|---|---|---|
| null mean, over the 20 nulls | 0.3262 to 0.3330 | 0.3287 to 0.3336 |
| null SD | 0.0084 to 0.0127 | 0.0100 to 0.0300 |

* **The null is centred BELOW the global majority share 0.338215**, by 0.005 to 0.012. The fold-wise train-majority predictor reads 0.3040.
* This is the parent's HYPOTHESIS, now MEASURED:
  * under episode-grouped folds, the class shares differ by fold;
  * so a label-free predictor's score sits away from the global share, and spreads with an SD of 0.008 to 0.030.
  * A band of ±0.02 around 0.338215 is narrower than this design's null and centred in the wrong place.
* **36 of the 40 committed SHUF cells lie inside the null's 95 % range.** The 4 outside are all at permutation seed 1, all tree:

| cell | committed | null range (p2.5 to p97.5) | null min to max |
|---|---|---|---|
| N1D_s0 F-TEXT and N1D_s1 F-TEXT (the same data, so one cell counted twice) | 0.3502 | 0.3098 to 0.3465 | 0.3065 to 0.3485 |
| N1D_s0 F-VIS | 0.3161 | 0.3207 to 0.3523 | 0.3203 to 0.3536 |
| N1D_s1 F-PLAN+TEXT | 0.3401 | 0.3116 to 0.3396 | 0.3110 to 0.3403 |

* All four sit within 0.0042 of the null's extreme draw. Expected by chance at 95 %: about 2 of 40. The 4 seen include one duplicated cell, so about 3 independent ones.

## Reading
* **The committed SHUF control stays FAILED as written.** This null does not turn it into a pass, and it moves no band of the parent card.
* The failure is explained by the band's design, not by leakage. Under the permutation null this probe design reads 0.326 to 0.333 on average, with a SD of about 0.01 for the tree. The committed SHUF values match that null, apart from about 3 independent cells at the edges of a coarse 25-draw range.
* **For the deciding contrasts:** F-PLAN - F-TEXT is +0.0983 / +0.0893. That is about 7 to 12 times the tree null's SD (0.0084 to 0.0127), and the two arms share folds and windows. The PROBE-PRESENT-UNUSED reading stands.
* F-PLAN's 0.7028 / 0.6937 is quotable as a T0 decodability estimate by this probe. It is not a driving number and not a proven ceiling. PREREG_N1D_CEIL measures the input ceiling separately.
* **Root-cause class**, for later probe cards: a SHUF band fixed around the GLOBAL class share, under group-wise folds whose class shares differ. The design check that catches it: derive the band from the label-free predictor's own fold-wise score, or run the permutation null before freezing the band.
