# PREREG_V8F_PROOF_ADD122_N3 — note 3 to card ADD122 (TELEGAP-T0): the row clause of control t5 in the reader

Written while the fit queue runs (the identity fit D2 and the permuted-label fit FP are done, the fits of the arms M and F are not), before the read of record and before any gap error of the arms M and F is opened. The card (91c3b8283730030b) and the notes N1 (c12a3e5fde570cec) and N2 (ed95169935da5503) are unchanged.

## 1. What was found

A rehearsal of the reader's control stage on the pod (output under `telegap/tests_tmp/reh_read2`, not a read of record) used the real dump of the permuted-label fit for FP and, standing in for the arms M and F, the same dump. The controls t0 to t4 and t6 passed. Control t5 failed on one clause that the reader carries and neither the card nor a note asks for: "FP_0 has a finite gap on every row of control t3". The dump has a finite gap on 3,817 of the 3,841 rows; the other 24 rows are frames with no matched tele frame, which note N2 section 2 drops from the arms M and F and marks in the dump. The clause of the card and of note N2 section 3 (the error of the permuted-label head is at least 0.95 of the error of the constant, on the frames the head holds) passed in the rehearsal.

So the reader's row clause fails by construction on any head that reads tele tokens: it asks for rows the head is defined not to hold.

## 2. The rule from this note on

The row clause of control t5 reads: FP_0 has a finite gap on every row of control t3 that its own dump does not mark as dropped for a missing tele frame. A row that is neither marked nor finite fails the control as before. The error clause is unchanged and is read, as note N2 says, on the frames the head holds. The reader writes both counts (rows of t3, rows marked as dropped) into the control's record.

The reader is changed in this one clause, with a test for both directions (marked rows pass; an unmarked row without a finite gap fails), and its tests are run again on the pod before the read. No bar, no other control, no measure and no name is touched.

## 3. Limits

The note was written knowing that the error clause of t5 passes on the real permuted-label dump and that t4 passes on the real identity dump (both are control values), and knowing no gap error of the arms M and F: their fits had not ended.
