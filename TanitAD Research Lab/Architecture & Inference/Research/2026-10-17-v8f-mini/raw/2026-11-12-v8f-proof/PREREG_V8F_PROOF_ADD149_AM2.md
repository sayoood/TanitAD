# PREREG_V8F_PROOF_ADD149_AM2 — LEADSPEEDFULL10K, amendment 2: the measured cost per clip, and read clips loaded at read time

Frozen after the second launch of card ADD149 stopped at its own per-clip guard during the load and before any training step, outcome number or head output exists. Amendment 1 (c122e315) stands. The card's other sections are unchanged.

## What happened (MEASURED)

* Second launch (2026-11-25T19:08Z, `train_full_box2.py` 7ba6239d, the guard of amendment 1): it passed the pre-load guard and stopped as `LEADSPEEDFULL10K-STOPPED-MEMORY` during the load of the held-out clips, about 1,511 of 1,943 clips in: commit available 7.18 GB against 7.31 GB (5.5 GB + the about 432 clips still to load at the card's 4.19 MB). Control c2 passed. No training step ran. `rulefc/leadspeed/box2_full/FULL_launch2_stopped.json`.
* The card's 4.19 MB per clip is the token bytes only. Measured on box 2 in separate processes, 120 clips each (`rulefc/leadspeed/memprobe_box2.py`, `rulefc/leadspeed/box2_full/MEMPROBE2.out`): the process's own commit grows by 5.075, 5.086 and 5.071 MB per clip (memory-mapped, plain, memory-mapped load), against 4.136 MB of token bytes. A first probe that ran both load modes in one process (`MEMPROBE.out`) is order-confounded (the second mode reused heap the first had committed) and is not used.

## The changes

1. The memory budget per loaded clip is 5.2 MB (the measured 5.07 to 5.09 MB, rounded up), in place of 4.19 MB, in both guards.
2. Only the 1,500 fit clips stay in memory during training. The 443 held-out clips and the 141 eval clips are loaded one at a time at read time, after training and after the fit clips are released: each is labelled, read by the 30,000-step checkpoint, and released. The read quantities, the clips, the labels and the order are the same as in the card.
3. So the pre-load guard is: commit available at least 5.5 GB + 1,500 × 5.2 MB (7.8 GB) + max(0, 2.5 GB − own). Before every fit clip: at least 5.5 GB plus the fit clips still to load at 5.2 MB. Before every read clip: at least 5.5 GB plus one clip. A failed guard stops the run as `LEADSPEEDFULL10K-STOPPED-MEMORY`, with the values recorded; the fit set is never shrunk.

## Unchanged

The head, labels, split, 1,500 fit and 443 held-out clips, 30,000 steps, schedule, checkpoints, reads S and A, controls c1 to c4, the bars (0.30), the names and the consequence.
