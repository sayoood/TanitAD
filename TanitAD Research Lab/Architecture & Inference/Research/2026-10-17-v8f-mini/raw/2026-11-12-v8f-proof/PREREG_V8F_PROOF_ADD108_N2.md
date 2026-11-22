# Note N2 to card ADD108 (LOOKGAP-T0): open definitions of stages C and D and one limit of control m6, fixed before any fit of stage C

Card: `PREREG_V8F_PROOF_ADD108.md` (sha256 begins d3df16d6eee65c9f); note N1 (sha256 begins 3b35543e6112becd). This note is frozen with its own hash file while the encode of stage B runs on the pod (started 07:15:59Z on 2026-11-22 after the PI's go; clause i of control m1 passed 07:16:30Z), before any fit of stage C has started and before any value of the card is opened (the score lines of stage A, every fit record, dump and read value are unread). It changes no bar and no name of the card. Kit of stage C: `lookgap/fit` (`pod_lookgap_fit_queue.sh` fc392a29e8ef7f40, `cross_dump.py` 68ee133b4551bf8a; lead run of its tests 10 passed, logged 07:28:07Z). The kit of stage D (`lookgap/read`) is being built at the time of this note; its hashes and the lead's test run are logged in `TIMES_NOTE.md` before the read.

## 1. Control m6: a limit caused by the order of work (stated, not repaired)

The card says: "the tests of the kit pass in the lead's run on the dev box before stage B". At the start of stage B two kits existed, the ones note N1 names (`lookgap/dirs`, `lookgap/enc`); the lead's test runs of both passed before stage B (17 passed and 1 skipped; 19 passed). The kits of stages C and D were built after the start of stage B, while the encode ran; this was the lead's order of work, not a need of the study. For these two kits the time clause "before stage B" cannot hold.

How m6 is read and reported:

* m6 is coded as: the lead's test output of each of the four kits (`dirs`, `enc`, `fit`, `read`) ends in a line with passed tests and no failure.
* The row states for each kit when its tests ran: `dirs` and `enc` before stage B; `fit` before the first fit; `read` before the read. No test of the two later kits can use a value of the card: none is open when they run.
* By the letter of the time clause, m6 holds for the two kits that make the tokens and does not hold for the two later kits. The row says so in these words beside the control line, and the row is read by the other session before it is landed. If that read holds m6 to its letter for all four kits, the card is reported as failed on m6 with no name, and the same read with the kits built first is a new card.

## 2. Open definitions of stage C, as coded

* The fit tool of card ADD99 knows the arm names D and B only and treats both alike (it reads `<root>/tok/<arm>/<clip>.npy`). Arm P is handed to it in a run root of this card (`lookgap/run`) whose `tok/B` is a symbolic link to the token folder of arm P; the lists of that run root are copies of the pilot's lists (hashes compared). The fit records therefore say arm B. Control m2 reads "head H5 on arm P" as: the record says head H5 and arm B, the root in its arguments is this run root, and `tok/B` of this run root resolves to the token folder of arm P (in the record of the run root and at read time).
* Dumps on the evaluation join: fit seeds 0 and 1 with the present reader's dumps of seed index 0 and 1 as in card ADD99; fit seed 2 with the dump of seed index 0 under the name of seed 2, as card ADD104 did; the permuted fit with seed index 0.
* At most three fits at a time on the one GPU; dumps and cross dumps one at a time. Stop rule of the queue: before each job start the summed seconds of the finished jobs have to be at most 5,400 (1.5 hours); with three fits side by side this sum runs ahead of the wall time, so the rule as coded is the stricter one, and it does not end a running job.
* Cross dump (printed with no name): the importing tool `cross_dump.py` uses the head-output and assembly code of the pilot's dump tool by import. "Reproduces the dump of B0 on the tokens of arm B" is: on the rows where both have a finite gap the largest absolute difference is at most 1e-3 m, and the pattern of missing outputs and the valid flags are equal. This is a printed fact, not a control; when it does not hold, the cross-read values are printed with that statement beside them.

## 3. Open definitions of stage D, as coded

* Base rows: the join rows of the 141 evaluation clips with a lead in the label (present, finite gap; read from the dump of D0) and a finite gap of the five fits D0 to D4. Row set of the card: the base rows that also have a finite gap of P0, P1, P2, B0, B1. The reader prints both counts and counts the rows on which the label fields of the dumps differ; m4 asks for 3,841 rows in the row set and no such difference.
* R and its interval: per clip the sums of the absolute error on the row set; numerator = mean over P0, P1, P2 of these sums, denominator = mean over D0, D1, D2; 2,000 draws of the 141 clips with replacement from `numpy.random.RandomState(108).randint`; R per draw = summed numerator / summed denominator; the 2.5 and 97.5 percentiles. Fewer than 1,000 finite draws: no interval.
* m3: on the rows of the row set where the permuted fit has a finite gap, the same bootstrap for E(permuted) / E(P0).
* m5: on the base rows, with the label's box centre from the pilot's box files and the direction from the direction files of the evaluation clips, the measure code of card ADD107 (`read_lookat._stats`, imported unchanged) gives S4 and `S8_out`.
* Bins of the label gap: 0-10, 10-20, 20-40 m with the upper edge open, 40-60 m with 60 included. "More than 5 m short": gap minus label below -5 m. Bins of the absolute azimuth of the label's box centre as in the measure code of card ADD107 (0-4 with 4 included, 4-8, 8-16, above 16 degrees).
* The share of the bound's gain uses D0, D1, B0, B1, P0, P1 as the card says.
* The score lines of stage A are copied from the logs of its queue into the names log by the read script after the controls pass; they are on another frame set (note N1, section 1, point 4).
