# PREREG_V8F_PROOF_ADD120_N1 — note 1 to card ADD120 (PRESROLL10K): definitions the build fixed where the card left them open

Card: `PREREG_V8F_PROOF_ADD120.md` (sha256 prefix d419232ed3cccbc2). Written after the queue start (2026-11-22T17:11:57Z, pod clock) while the identity run is the queue's running first rollout, before the reader of the card has run on anything and before any control or value of the card is read. This note changes no name, no pair, no bar and no committed reading of the card. Read so far of the new runs: nothing. Read of banked runs for this note: key names, dtypes and row counts of the rule logs and of one dump file, and two formula checks stated in section 5 (counts of rows that do not fit, no rate).

## 1. Control q8, the gap clause, and rows the presence input raised

Control k7 of card ADD117, which q8 names for the gap, checks every row with a lead: where the run's gap file has an output at the row's clip and frame, the gap the rule received is that output; where it has none, the gap is checked against the lead rows of the reference run RR of the same setting and model seed at the same clip and frame. A row that the presence input raised (reader probability below 0.5, used probability at least 0.5) has no lead row in the run RR by construction. Where such a row also has no output in the gap file, k7 has nothing to compare it with.

As built: a row that is raised, has no output in the run's gap file and has no lead row in RR is left out of the gap clause and counted (`n_raised_rows_without_reference`, printed per run with the control). Every other row stays in the clause, the raised rows with a gap-file output among them. The clauses on `p_used`, `present`, `p_src` and the counts cover every row.

## 2. Control q1, which arrays are compared

Dump files: every array of every file (dtype, shape, values; NaN equal to NaN). A banked dump file holds numeric arrays only. Rule logs: the key names have to be equal; every numeric array is compared in the same sense; entries of text dtype are not compared as values (the run tag, paths, hashes and the count texts of the run record are text entries of the log file and name the run). The side file of the identity run has to exist and carry no column `p_used`.

The queue has its own gate on the same question, as the card orders: after the identity run it compares the dump folders with `cmp_dumps.py` of the build folder and the numeric rule log arrays (text entries of one value not compared, key names compared), and it stops on a difference. The lead reads its pass or stop marker only.

## 3. Control q8, the rows of the counts

The wrapper counts a row at each tick-level look and at each later look (the looks of the rule every 5 steps under readout K); each of these has a row in a side file. The looks at the earlier frames of the closing run are not counted and have no row. The reader recomputes the counts over the rows of the side files (the tick log's, and under readout K the later-look log's as well) and compares them with the run record of the tick log. With no pre-roll in this card the side rows and the log rows are the same rows.

## 4. Controls q0, q2, q3 and q5 as read

* q0: in addition to what the card states, the first line of the run text of each new run and of the identity run has to name the hash of the study wrapper (the queue writes the hash of the wrapper it runs).
* q2 and the last clause of q3: the first line of a new run carries ` vis-pres-file <path> vis pres sha256 <hash> vis-pres-mode or`, written by the queue; the rule log record has to name the same hash and mode, and `seed` of `pres_meta.json` has to equal the model seed.
* q5: "before" is read as "not later than" in whole seconds. The lead's queue-kit test output was copied to its place in the second of the queue start (both 1795367517); the test itself ran before, in the call that then started the queue.

## 5. Checks on banked runs, stated for the record

On the banked runs `I10RQ50KF5Hb20gGEc_s0` (tick log and later-look log) and `I10RQ50Hb20gA90_s1` (tick log), 7,197 rows each: `present` equals (`p` at least 0.5 and `ok` and not `src_bad`) on every row, and `p_src` equals `p` within 1e-6 on every row. None of these rows has `src_bad` set, so the stand-ins do not exercise that term. In the later-look log the column `j` is the frame of the look (tick frame plus 5), which is the frame the wrapper reads the presence file at.

## 6. What stays as the card states

All nine controls have to pass before any name or value is opened; with one control failed nothing of the card is opened and it is reported as failed with no name. A changed form is a new card.
