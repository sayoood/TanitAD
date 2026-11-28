# VERDICT_DECN (Master Mind, 2026-11-09)
Card `PREREG_V8F_DECN.md` (`cfa5bb1f`) with its amendment 1 (in the card's Log). Tier **T0**, a readout diagnostic, never a driving result. MEASURED: `RESULT_DECN.md`, `runs/*.json`, `null_shuffled.json`. Pooled held-out R2, no interval, as the card says; differences under 0.05 carry no weight. The same 17 held-out episodes (238 windows) at every N. LONGITUDINAL, LATERAL and TACTICAL have no metric here (a readout R2 only); STRATEGIC is PAUSED (PI item 25).

## Validity
The first run is **INVALID as run**: control 3 (shuffled targets at or below 0.05) failed on 10 of 216 reader rows, +0.052 to +0.158. All 10 are MLP rows; all 108 ridge rows pass. Controls 1, 2, 4, 5 and 6 pass (target tie R2 1.00000, reproduction 4 of 4, planted 54 of 54, one held-out digest over 27 units, no md5 mismatch).

A diagnostic written after that failure (`null_shuffled.py`, it decides nothing) refitted the ten rows on 8 fresh permutations each. Over the 80 fits the shuffled R2 has mean -0.126, sd 0.125 and maximum +0.160; 3 of the 80 are above 0.05, and every row's mean is below 0. So the exceedances are the spread of a single permutation at 17 held-out episodes, with no positive offset that a leak would give.

Amendment 1, chosen after seeing that failure and the first-run values: control 3 and its bar do not move; a row that fails it leaves the deciding maximum, in place of invalidating the read. None of the 10 rows is the maximum of a deciding cell (R0, R2 at N53 or N1053, both targets), so the amendment changes no deciding value. The read below is valid under amendment 1. The first-run line stays published beside it.

## The read: best held-out R2 over the readers (optimistic, a maximum over four readers)
| target (bar) | input | seed | N53 | N1053 | rise (cell value) |
|---|---|---|---|---|---|
| speed (0.70) | R0 M2b z1 | s0 | 0.420 | 0.695 | +0.274 |
| | | s1 | 0.387 | 0.686 | +0.299 |
| | R2 pooled token | s0 | 0.498 | 0.721 | +0.223 |
| | | s1 | 0.391 | 0.736 | +0.345 |
| yaw rate (0.50) | R0 M2b z1 | s0 | 0.423 | 0.570 | +0.147 |
| | | s1 | 0.268 | 0.441 | +0.173 |
| | R2 pooled token | s0 | 0.502 | 0.451 | -0.051 |
| | | s1 | 0.329 | 0.507 | +0.178 |

The ridge alone, the reader PLANTEST3 uses (R0, one frame): speed 0.271 / 0.314 at N53 and 0.680 / 0.647 at N1053; yaw rate -0.122 / +0.268 and -0.043 / +0.105. Ladder z1 rows (`RESULT_DECN.md`, deciding nothing): speed 0.713 to 0.723 at N1053 on all five checkpoints.

## Outcome per target (mechanical rules, under amendment 1)
* **Speed: N-LIMITED**, through R2 (0.721 / 0.736 at N1053, both seeds at or above 0.70). The margins are 0.021 and 0.036, under the 0.05 that carries weight, and R0 sits just below the bar (0.695 / 0.686). The firm reading is the rise: +0.22 to +0.35 on every deciding input and seed, so the 53-episode read was limited by fit size. DECINFO's NOT-READ-FROM-VISION was a 53-episode read; at 1,053 the read reaches the bar.
* **Yaw rate: RISES**, on R0 (+0.147 / +0.173; 0.570 / 0.441 at N1053, s1 below the bar). It needs the MLP: the ridge stays at or below 0.109 at N1053.

## Limits
* N53 to N1053 changes the amount of data and also the population (TUNE eval clips plus train clips). This card cannot separate those two.
* Yaw rate at N194 reads negative on 17 of the 18 MLP rows (-0.108 to -0.783), between positive MLP reads at N53 and N1053. This is not explained here; the deciding N is N1053.
* The MLP stops at the first useful epoch on most N1053 rows (epochs 13 of 60), so its N1053 values may be below what a better-tuned MLP would read.
* A readout bound for these readers. The PLANTEST3 gate's own DECODER-WEAK bars are still missed by the ridge at N1053 (speed 0.680 / 0.647, yaw rate -0.043 / +0.105).

## Next (the card's triggers, Rule Zero)
1. **Speed, N-LIMITED:** PLANTEST3's FLAT1 DEC-END with the N1053 decoder. Running under its own frozen card, `PREREG_V8F_PT3N.md` (`de94f72c`), so its validity does not rest on amendment 1 (controls 1 to 3 already pass).
2. **Yaw rate, RISES:** the full-n probe on the deciding inputs (N4565: every trained clip). Running in `fulln/`, streamed read-only to the dev box one clip at a time, as this card did, in place of the box-2 job after the ladder (amendment 1 states it). S35_s1, overlapping DECN's stream, ran 2 h 23 min against 2 h 18 min for S35_s0.
3. **FLAT did not occur**, so the measured-speed world-model card stays a draft behind these two.
