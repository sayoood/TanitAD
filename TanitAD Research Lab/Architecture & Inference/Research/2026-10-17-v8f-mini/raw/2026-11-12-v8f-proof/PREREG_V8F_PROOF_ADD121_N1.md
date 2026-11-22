# PREREG_V8F_PROOF_ADD121_N1 — note 1 to card ADD121 (REPL10K)

Written before the two further models exist, before any rollout of this card and before any value is opened. The card (e87dc5b05f6848b5) is unchanged; this note states what the card left open and one statement of the card that the banked run lines do not bear out.

## 1. The wrapper of the banked reference runs

Section 2 of the card says that the wrapper is that of the banked runs. Read from the first lines of the banked run logs after the freeze (arguments only, no outcome value): the candidate runs GEc and A90 of the first pair went through `v8f_t1_eval_reader_rule_rzgfhqetjv.py` (28bc31f21ecb4ccb); the reference runs R0, RLc and RR at CONV and K0 went through an earlier variant of the same wrapper (373ffe72f2655c35), and RLc and RR at K30 through another earlier variant (92a8034cc0f7b0d2). Each later variant adds inputs behind switches that are off in the reference forms. All 26 new runs go through 28bc31f21ecb4ccb, as the card says.

So against the first pair a new reference run differs in the training run of the model and in the wrapper variant. To read the second apart from the first, the queue has an identity stage before the main stage.

## 2. The identity stage of the queue

Three banked reference runs of model seed 0 are made again through 28bc31f21ecb4ccb under new names and compared with the banked runs: `I10R0Hb20IDR` against `I10R0Hb20` (no rule, CONV), `I10RLcKF5Hb20IDR` against `I10RLcKF5Hb20` (label form, K0), `I10RQ50Hb20IDR` against `I10RQ50Hb20` (reader-fed form, CONV). They cover the three reference forms, both readouts and both rule rates.

* Dumps: `cmp_dumps.py` has to report 141 files with none differing.
* Rule logs (tick log, mid log and their side files): every file the banked run has must exist in the new run, every row array of the banked file must exist in the new file and be equal (not-a-number at the same places counts as equal). Files and keys that only the new run has come from the later variant; they are counted and printed and are not a failure. Entries with no rows (text and scalars of the header) are not compared.
* The stage is read as pass or stop only. No level is computed from the three runs and they enter no estimate.
* It is a gate of the queue and not one of the eight controls of the card: the main stage refuses to start without the done marker of the identity stage. If a compare stops the stage, that is reported as written, the main stage is not started under this card, and a changed form is a new card.

The stage runs on a model of the first pair and so may run before the two further models exist. K30 is not in it: the K30 reference variant differs from 28bc31f21ecb4ccb in the same kind of added inputs, and control r3 reads the banked K30 levels by the reader's code path; this limit is stated in the row.

## 3. Control r2, the first lines of the run logs

* The compared fields are form, replan steps, rule rate, readout, heading bound, pre-roll, executor, nav, grid, model seed, reader seed and checkpoint hash. Model seed and checkpoint hash have to differ from the banked line of model seed m - 2; the reader seed has to equal m - 2; every other field both lines carry has to be equal.
* Banked first lines of the reference runs at CONV and K0 carry no pre-roll, executor, nav and grid field (the queue form that made them did not write them). For those the value that this queue form wrote is taken as in note N1 of card ADD113: pre-roll 0, the pinned grid, nav by dead reckoning; the executor field is compared only where both lines carry it.
* The wrapper hash of a first line is the subject of control r1 (every new run names 28bc31f21ecb4ccb) and is not compared with the banked line in r2, for the reason of section 1.

## 4. Rollouts and time

26 rollouts in the main stage and 3 in the identity stage. The card's estimate of 4 pod-h and its stop rule of 7 pod-h are for the main stage from its start line; the identity stage is ESTIMATED at 0.5 pod-h with its own stop at the queue's limit.

## 5. Order

Queue kit: generator 7bfddf67d33c77b5, queue c0bbaa9b9e9f7af4, test 70e1e8852d5af2a1; the lead's test run on the pod passed (6 tests) before this note and before any start. The identity stage starts after this note is frozen. The main stage starts only after the note of control r0 is written; it does not wait for the reader kit's tests. As the card says, no value of a new run is opened before the reader kit's tests pass and the controls are logged.
