# Note N1 to card ADD79 (ESTCLOSE10K): a float tolerance inside the save tool

Written after the first run of the save tool stopped (logged in `TIMES_NOTE.md` before the repair was made and before this note was frozen) and before any estimator file exists. No clause of a control of the card changes.

## 1. What happened

`closeread_save.py` (c15066be6b1bd95d) ran on the pod at 13:47Z and returned rc 1; it wrote nothing. Its check lines (differences only, no outcome value):

* The clauses of control e0 as the card has them all pass. Per-seed r of the saved estimator against the result of card ADD78: differences 5.29e-10 and 6.86e-10 (N10), 1.67e-16 and 0 (L25); bar 0.005. L25 against the estimate of the fit tool row by row: 0 and 0; bar 1e-6. Beside them the tool's own clauses: repeated training equals the fit tool's estimate (0 for all four), numpy forward of the saved bytes against the fit tool's estimate for N10 3.18e-07 and 2.38e-07 (the tool's bar 1e-4), the table rows contain the fit tool's rows (all four).
* One sub-check that the tool adds and the card does not hold failed for N10 in both reader seeds: the estimate in the all-rows table against the same saved estimator applied to the scored rows alone, the same inputs, read 1.59e-07 and 1.19e-07 m/s against a bar of 1e-12. For L25 it read 2.22e-16 and 4.44e-16.

## 2. Cause

The network arm computes in float32 (as the fit tool does). The two sides of the sub-check apply the same weights to the same rows in two batches of different size; a float32 matrix product does not give bit-equal sums for two batch sizes. Differences of 1e-7 m/s are what float32 gives; a bar of 1e-12 belongs to the float64 path of the ridge arm, where it holds.

## 3. Repair

`closeread_save_n1.py` is written from the tool by one counted text edit: the bar of that sub-check is 1e-12 for the ridge arm and 1e-5 m/s for the network arm. 1e-5 m/s is a hundredth of the bar of control e2 of the card (1e-3 m/s) and far below any closing speed the rule acts on. Everything else is the tool as reviewed. The tool's self-test (with its planted faults: r off by 0.01, shifted normalisation, a result file without cells) is run again on the pod before the save.

This is a float tolerance and not the form of a control (rule of row 136): what the sub-check asks, that the table and the saved estimator agree on the scored rows, stays; only the size of a float32 rounding difference is admitted.
