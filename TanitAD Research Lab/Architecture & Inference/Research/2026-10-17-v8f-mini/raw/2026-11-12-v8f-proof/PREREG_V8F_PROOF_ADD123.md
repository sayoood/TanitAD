# Card PREREG_V8F_PROOF_ADD123 (DSCALE-LEAD-2X): the union probe with twice the steps at four seeds, read against the corpus probes, with the training dose read on held-out data

Card of the v8F proof study, written by the coordinator fork. Frozen before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); not driving performance. It is the run that row H-V8F-PROOF-136 names as follow-up (2), under the fork rule of that row: a changed form of a control is a new card and a new run.

Go of the PI, verbatim, 2026-11-22: to this fork "4 yes" (item 4 of the list of that day: the union probe of card ADD45, about 0.7 pod-h on the pod of the other session); to the other session "Yes, after the seeds". The GPU stages run on the pod of the other session, by that session, under its lock, after its two model seeds have ended. A projection above 1 pod-h goes back to the PI.

## 1. Why

Card ADD45 asked whether more clips lift the read of the lead gap through pooled tokens. Its matched-step design put the union probes (14,512 clips, 15,392 steps per pass) at 1.25 passes against 4.0 passes of the corpus probes (4,512 clips), control d4 as written did not pass on them, and the card has no name (rows 135 and 136). MEASURED there, one seed, naming nothing: the union probe with twice the steps reads gap R2 0.719 against 0.712 of the corpus probe of the same seed, and +0.0897 above the union probe of the same seed at matched steps. The question stands open at four seeds, and the dispute of row 136 was about how to tell a probe that has learned what it can from one that has stalled. This card asks both on held-out data.

## 2. Arms (probes on frozen tokens; no model is trained)

* Banked, used as they are (run of 2026-11-20 on the pod of the other session, card ADD45): the corpus probes `P16C_s0` to `P16C_s3` (19,164 steps, 4.0 passes), the union probes at matched steps `P16U_s0` to `P16U_s3`, the union probe with twice the steps `P16U2_s0` (one-arm folder), and the probes with permuted gap labels `P16CSH_s0` and `P16USH_s0`.
* New, one GPU process: `P16Q_s0` to `P16Q_s3`, the union probe at 38,328 steps (2.5 passes) with the banked tool, flags, inputs and schedule of `P16U2_s0`, four seeds on the same batches. `P16Q_s0` has the seed and the flags of the banked `P16U2_s0`: it is a replicate with no lever moved.
* The new process writes, beside the last state, the state of each probe after 19,164 and after 28,746 steps. This is done by a tool that imports the banked probe tool unchanged and saves a copy of the weights at those steps; it draws no random number and changes no update (control q1).
* Evaluation states are extracted for the four new probes at the three steps (twelve extracts), on the evaluation set of card ADD45 through the view of its note N3.
* Budget: ESTIMATED 0.70 pod-h for the training process (the banked four-probe run of 19,164 steps took 1,252 s; twice that), plus a 200-step smoke and the extracts, about 0.85 pod-h of lock time in all. Stop rule: the smoke's projection of the training process above 0.9 h stops the kit before the run; the training process runs under a time limit of 3,300 s; either stop goes back to the PI.

## 3. Reads

* Evaluation set: the 429 scored windows of the 48 evaluation episodes of card ADD45, unchanged.
* Per probe: M2 (gap R2 and mean absolute gap error), M4 (presence AUROC).
* S: per seed, paired on the same windows, `P16Q_sK` minus `P16C_sK` on M2 (gap R2) and on the gap error; two-level bootstrap (seeds, then episodes), B 2,000, the estimator of the reader of card ADD45. It answers what another draw of episodes and of these four seeds would say.
* L, the late gain: per seed, gap R2 of `P16Q_sK` at step 38,328 minus at step 28,746, the same two-level form.
* F, the replicate floor: the absolute difference of gap R2 between `P16Q_s0` and the banked `P16U2_s0`, one number, with its paired episode interval reported beside. The largest absolute difference between any two of the three logs of batch hashes is not part of F (control q1).
* Reported beside, judged by nothing: the gain from step 19,164 to step 28,746 inside the run; `P16Q_sK` minus the banked `P16U_sK` (twice the steps against matched steps, each on its own schedule); the fall of the training loss that control d4 of card ADD45 asked for, per probe; M4 per probe.
* Four metric families: LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL and TACTICAL are not measured by this read; STRATEGIC is not computed (n = 0; a token probe reads no strategic quantity).

## 4. Names (one of four; committed before the data)

The dose clause holds when the upper end of the two-level interval of L is at most +0.020 of gap R2. Reason for the bar, set before the run: doubling the steps gave +0.0897 at seed 0; at an even rate the last quarter of the longer run (9,582 steps) would add about +0.045, so a probe that still gains at less than half of that rate has left the rising part of its curve, and 0.020 is half of the range of the four corpus probes (0.694 to 0.734).

* `LEAD-READ-2X-SCALES-GAP`: the two-level interval of S on gap R2 lies above 0, at least three of the four seeds are above 0, the mean gap error of `P16Q` over the seeds is at most 0.90 times that of `P16C`, and the mean of S is larger than F. The dose clause is not asked: a probe short of training reads lower, which works against this name.
* `LEAD-READ-2X-WORSE`: the two-level interval of S on gap R2 lies below 0, the mean of S is below minus F, and the dose clause holds.
* `LEAD-READ-2X-FLAT`: neither of the two above, and the dose clause holds.
* `LEAD-READ-2X-SHORT-OF-TRAINING`: not `SCALES-GAP`, and the dose clause does not hold. Nothing is said about data scale.

## 5. Controls (all have to pass before any name or value of section 3 is opened)

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* q0 (inputs): the banked folders are those of the run of 2026-11-20 (checkpoint hashes of its done file), and the prechecks of the kit of card ADD45 are run again in the new output folder and pass with their planted failures (no evaluation clip in a training list, labels on one clock, the evaluation set and its order).
* q1 (the saving tool changes nothing): the log of batch hashes of the new folder equals that of the banked folder of `P16U2_s0` entry for entry, and the first logged record of `P16Q_s0` (step 1, before any update) equals that of the banked `P16U2_s0` on every loss term within 1e-4. Whether the two full logs agree to the last digit is printed and not judged (the GPU sums are not promised to repeat).
* q2 (live probes, the form of control d3 of card ADD45): each of the eight probes of S reads a gap R2 whose episode interval lies above 0 and a presence AUROC of at least 0.85.
* q3 (negative control, the form of control d5 with note N1): the two banked probes with permuted gap labels read a gap R2 whose interval includes 0 or lies below it, while their presence AUROC stays at least 0.80 (the presence labels are intact).
* q4 (the runs are complete and the labels reach the loss): for every new probe the header fits the arm, the log ends at step 38,328, every full-batch record (note N5 of card ADD45) has more than 0 gap-labelled frames, and the three states carry the steps 19,164, 28,746 and 38,328. The fall clause of control d4 of card ADD45 is not a control of this card; the dose clause of section 4, read on held-out data, stands in its place, and the values of the fall clause are printed beside.
* q5 (estimator and reader): the reader controls of card ADD45 (a planted difference read back in the two-level form, a pair with itself reads exactly 0, a swap reads the negative), and the tests of the reader kit have passed before the read (file times).
* q6 (one evaluation set): the states of all probes and steps carry the same evaluation ids in the same order and the same count of scored windows.
* Tool checks before any run: unit tests of the saving tool on a tiny stream (the saved weights at a step equal those of a run stopped at that step; the run with and without the saving tool gives the same last weights on CPU), of the run script (dry run prints every command and runs nothing; the time limit and the smoke bar stop it), and of the reader on synthetic states with every control made to fail once.

## 6. What follows (committed before the data)

* `LEAD-READ-2X-SCALES-GAP`: data scale lifts the gap read through pooled tokens. The lead head at 10k becomes a lever of the integrated candidate beside the camera line, and a run at equal passes (61,568 steps, about 1.1 pod-h) is put to the PI to read how much more it gives.
* `LEAD-READ-2X-FLAT`: at up to 2.5 passes more clips of this kind do not lift the gap read through pooled tokens; the lever of the gap line stays the camera crop (rows 170 to 172 and card ADD122). The bound "up to 2.5 passes on this schedule" is part of the reading; the run at equal passes is offered to the PI as the arm that would lift the bound, with its cost, and is not asked for on this card's strength.
* `LEAD-READ-2X-SHORT-OF-TRAINING`: the run at equal passes (four seeds, one process, about 1.1 pod-h) goes to the PI as the next arm, with the kit of this card and the step count changed.
* `LEAD-READ-2X-WORSE`: reported as written; the domain difference between the corpus and the 10k clips is read first (the corpus probe on 10k evaluation clips) before any model arm with a lead head at 10k.
* The row that reports this card is read by the other session before it is landed.

## 7. Order of work

The kit (saving tool, run script, reader with its tests) is built after this freeze and handed to the other session with its hashes; it runs precheck, smoke, run, extract and its log check; the output without the checkpoints comes back to this fork; the reader runs on the pod of this fork where the reader of card ADD45 ran. The controls are read and logged in `TIMES_NOTE.md` first; names and values are opened after that line.

## 8. Limits

One frozen encoder; pooled tokens; the evaluation windows come from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds; the corpus probes are banked runs of two days before, on the same pod and tool. The learning rate of the tool falls on a cosine to a tenth of its peak, so the last quarter of a run is taken at about a seventh of the peak rate on average: a small late gain is partly the schedule's doing, and the dose clause can say that this run had left the rising part of its curve, not that a longer schedule would add nothing. The run at equal passes is the read that would say that. A probe is not a model.
