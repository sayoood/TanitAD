# PREREAD_V8F_STOPTAU: a hardening threshold set on TRAINING windows only (Master Mind, 2026-11-11 ~19:05; frozen before any training-window number)

## Why (MEASURED, `VERDICT_STOPHARD.md`)
Hardening T's own stop decision at 0.5 (goal) or argmax (lon) gains 0.04-0.09 m tactical ADE on the 117 stop-point windows and costs 6-17 mm on all windows, at both seeds of both levers (HARD-TRADES). The committed next step: a threshold that fires the hard stop less often, chosen on training windows only. Its ceiling is the S gain above. Choosing the threshold on eval windows would tune on the scored data, so it is refused.

## Threshold (fixed here, per lever arm and seed)
* Training windows: first-tick windows of the 141 WMACT train clips (`theory/wmact/train_stream/clips141.json`, streamed read-only from box 2 as `theory/wmact/HANDOFF_WMACT_TRAIN.md` does), built with the same window code as eval-141. GT STOP_POINT from the training labels. The arm's own p(STOP_POINT) (goal) and p(lon = STOP) (lon) per window.
* **tau** = the smallest value on the grid 0.50, 0.55, ..., 0.95 at which the arm's training false-positive rate (firing on training windows that are not stop-point windows) is at most the matching NULL arm's training false-positive rate at the unhardened rule (goal 0.5; lon argmax). In words: the hard stop may fire no more often by mistake than the untouched NULL decision did. If no grid value meets it, tau = 0.95, and that case is reported.
* The transform on eval is STOPHARD's with tau in place of 0.5 / argmax: goal entry 1.0 where p(STOP_POINT) >= tau; lon one-hot STOP where p(lon = STOP) >= tau. The NULL arms get their own tau by the same rule, for the floor.

## Read, cells and estimator
Exactly STOPHARD's (`stophard_read.py`, imported): cells S (117 stop-point windows) and A (2,351 windows), tactical ADE knots 11-30, TAU minus OWN, paired episode-cluster bootstrap B 2,000 seed 0, floors |delta(NULL_rep) - delta(NULL_s0)| per variant and cell, four families beside, TACTICAL n = 0, STRATEGIC PAUSED (PI item 25). Tier: T1-kin first tick, never driving performance.

## Controls
1. tau = 0.5 (goal) and tau = argmax-equivalent replay reproduces STOPHARD's cells exactly (the lon rule at argmax is checked against STOPHARD's HARD-LON plans window by window).
2. The training windows share no clip with eval-141 (asserted by clip id).
3. A planted probability vector gives the planted tau.

## Outcomes, per lever (both seeds must agree, else SEEDS-DISAGREE)
| outcome | rule | next step |
|---|---|---|
| **TAU-HELPS** | S a gain at both seeds, A not a cost at either | Pre-register stage 2: the tau transform inside the T1-kin rollout (the tier of STOPBAL's P). The second dose is then read with and without it. |
| **TAU-TRADES** | A a cost at either seed | Hardening is closed as a lever with its measured rows. The second dose proceeds unhardened; the perception card is put to the PI. |
| **TAU-NULL** | S not a gain at either seed | Same as TAU-TRADES. |
