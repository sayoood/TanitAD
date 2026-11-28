# Note N2 to card ADD79 (ESTCLOSE10K): a clause of the reader's own making in control e2, and note N1 to card ADD80 for the same clause in its control k2

Written after the first read of card ADD79 ended with control e2 not passed (logged in `TIMES_NOTE.md` before the detail was read and before this note), from the detail of the control and the index columns of the logs only. The reader wrote no name and no value; none has been seen. No clause of a control of either card changes.

## 1. What happened

The read of card ADD79 ended at 15:04:51Z: 17 control lines pass, `e2_FRESH` does not.

* What the card asks in e2: on the FRESH runs, at the looks with a full history, the closing column of the rule log equals the saved estimator applied to the score dump at the same clip and frame to within 1e-3 m/s. This holds on every compared row of all eight logs: rows failing 0; rows compared 6,774 and 6,915 (N10, replan log and mid-look log, the same in both seeds), 5,364 and 6,069 (L25); largest difference 1.5e-05 m/s; the estimate column is NaN on every row without a full history.
* What failed: a clause that the reader adds and the card does not hold, "rows of the first replan tick (tick 0) are among the compared rows", applied to each log. The four replan logs have them (1,976 rows for N10, 1,412 for L25, none failing). The four mid-look logs have no row at tick 0 at all.

## 2. Cause

The mid-look log holds the second looks, at the ticks 5, 15 and 25 of a window; the replan log holds the ticks 0, 10 and 20. Measured on the index column of the logs of two runs of this card and of the banked run `I10RQ50KF5Hb20_s0` of row 137 (7,197 rows per log, 2,399 at tick 0 in the replan log, 0 in the mid-look log). The clause cannot hold for a mid-look log, whatever the estimate is. The reader's self-test did not catch it because its planted mid-look logs carried rows at tick 0, which a log of the wrapper never has.

## 3. Repair

`make_read_n2.py` writes `read_estclose10k_n2.py` from the reader (826bacc8d0e36be1) by one counted text edit: the first tick of the replan log is tick 0, the first tick of the mid-look log is its smallest tick. The clause stays in force for both logs in the form it was meant to have: the earliest looks have to be among the compared rows. Nothing else of the reader changes. The read is repeated once into a new folder; the first result file stays where it is.

The same clause stands in control k2 of the reader of card ADD80, whose read has not run: `read_constclose10k_n1.py` is written from it (0eed065b41f04777) by the same one edit. This paragraph is note N1 to card ADD80.

## 4. Why this is a repair and not a changed control

Rule of row 136: where the form of a control is in question the card is reported as failed and a changed form belongs to a new card. Here the form of e2 as the card has it is not in question and is not changed: it passed on every row at the first read. What is changed is a clause of the tool beyond the card that asks a log for rows it cannot hold. Because a name would be written after a second read, the row of this card is given to the other session to read before it is landed, as that rule asks.
