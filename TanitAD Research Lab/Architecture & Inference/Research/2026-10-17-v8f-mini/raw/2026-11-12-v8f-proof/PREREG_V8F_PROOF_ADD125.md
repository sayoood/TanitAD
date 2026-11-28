# PREREG_V8F_PROOF_ADD125 — DSCALE-LEAD-EQP: the union probe of the lead read at equal passes

Tier T0: a probe on cached tokens of logged frames. It says what the tokens allow, not what a model does with them, and it is not driving performance. Written after row 192 (card ADD123, name `LEAD-READ-2X-SHORT-OF-TRAINING`) and before any tool of this card and before any run. It is the next arm that card ADD123 commits for that name. The run needs the go of the PI; it is run by the other session on its pod, under its lock and after its own queue.

## 1. Question

At equal passes over the clips (4.0 passes on both sides), do the union clips (14,512) give a better read of the gap to the lead through pooled tokens than the corpus clips (4,512)?

## 2. What is known and seen before this card (row 192)

* Union at 2.5 passes against corpus at 4.0 passes, gap R2: -0.0286 [-0.1608, +0.0726], above 0 on 2 of 4 seeds.
* Union at 2.5 passes against union at 1.25 passes, end states: +0.0466 [-0.1181, +0.2197].
* The dose clause of card ADD123 compared a state inside a run with the end state of the same run. On a cosine learning rate the states inside a run read several tenths of gap R2 lower and scatter between seeds, so that clause measures the schedule more than the length of training. This card asks no such clause: it compares end states of full schedules only.

## 3. Runs

* Banked, used as they are: the corpus probes `P16C_s0` to `P16C_s3` (19,164 steps, 4.0 passes; run of 2026-11-20), the union probes `P16U_s0` to `P16U_s3` (19,164 steps, 1.25 passes; same run), the union probes `P16Q_s0` to `P16Q_s3` (38,328 steps, 2.5 passes; run of 2026-11-23, card ADD123), and the probes with permuted gap labels `P16CSH_s0` and `P16USH_s0`.
* New, one GPU process: `P16E_s0` to `P16E_s3`, the union probe at 61,568 steps (4.0 passes at 15,392 steps per pass), four seeds. The landed run script of card ADD123 (`dscale_lead/add123/run/pod_dscale_lead_2x.sh` a01c5ca8cb43a2c5) and its saving tool (`tok_probe_u_ck.py` 5888527b5c37003b) are used unchanged; three settings come from the environment: `STEPS_U2=61568`, `TRAIN_LIMIT_S=6600`, `MAX_H=1.5`, and `OUT` is a new folder. Inside that folder the script names the run folder `runQ` as before; the arm names of this card are `P16E_sK` = `<new out>/runQ/TK16_sK`. The script also saves states at 19,164 and 28,746 steps inside the run; they are not read for a name.
* Evaluation set: the 429 scored windows of the 48 evaluation episodes of card ADD45, unchanged.

## 4. Measures

* E: gap R2 of `P16E_sK` minus gap R2 of `P16C_sK`, per seed, at the last step of both. Interval: the two-level bootstrap of the reader of card ADD123 (seed indices, then episodes), 2,000 draws, unchanged. It treats the four seeds as the draw of seeds and holds each fit fixed. The same on the gap error in metres.
* T, beside: `P16E_sK` minus `P16Q_sK` (4.0 against 2.5 passes, end states), with the same interval, printed next to the known value of 2.5 against 1.25 passes.

## 5. Names

All controls of section 6 have to pass before any name or value is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* `LEAD-READ-EQP-SCALES-GAP`: the interval of E on gap R2 lies above 0, at least three of the four seeds are above 0, and the mean gap error of `P16E` over the seeds is at most 0.90 times that of `P16C`.
* `LEAD-READ-EQP-R2-ONLY`: the interval of E on gap R2 lies above 0 and at least three of the four seeds are above 0, but the gap error bar is not met.
* `LEAD-READ-EQP-WORSE`: the interval of E on gap R2 lies below 0 and at least three of the four seeds are below 0.
* `LEAD-READ-EQP-FLAT`: none of the three above.

## 6. Controls

* e0 inputs: the done files of the three runs are present with their hashes; the precheck of the new run passed with 14,512 union clips and 4,512 corpus clips; the hashes of the run script and of the saving tool in the new done file equal the landed ones; every arm of the new run ended at step 61,568.
* e1 same start: the record of step 1 of each arm `TK16_sK` of the new run agrees with that of the same arm of the run of card ADD123 within 1e-4 (same seed, same first batch, before any update). Whether the batch sequence of the first 38,328 steps is the same is printed beside where the files allow it and is not a control.
* e2 live probes: each of the eight probes of E has an interval of gap R2 above 0 and an AUROC of the lead presence of at least 0.85.
* e3 dead probes (the form of control q3 of card ADD123): the two banked probes with permuted gap labels read a gap R2 whose interval includes 0 or lies below it, while their presence AUROC stays at least 0.80. (Banked values, seen: gap R2 -0.227 and +0.018; the control passed in this form in row 192.)
* e4 complete runs: the new run is complete on all four arms, the gap labels reach the loss in the form of control q4 of card ADD123, and the last state of each arm is present with its step.
* e5 tests: the tests of the estimator and of the reader of this card pass before the read, with the test output older than the read.
* e6 one evaluation set: all probes are read on the same 429 windows of the same 48 episodes.

Each control clause is rehearsed on the banked runs before the read; a clause found to fail by construction goes into a note written before any value is opened.

## 7. Cost and stop rule

MEASURED on the run of card ADD123: 2,687 s for 38,328 steps of the four arms in one process. Projected for 61,568 steps: about 1.2 h for the training process, about 1.3 pod-h of lock with data build and extracts. The smoke stage of the script projects the time; a projection above 1.5 h stops the run and goes back to the PI. The training process is under a time limit of 6,600 s.

## 8. Committed readings

* `SCALES-GAP`: at equal passes data scale lifts the gap read through pooled tokens. The lead head on the 10k data becomes a lever of the integrated candidate beside the camera line; which model carries it is put to the PI and the other session.
* `R2-ONLY`: a gain in R2 that does not reach the metre bar. No lever is claimed; the gap line stays the camera line.
* `FLAT`: at equal passes more clips of this kind do not lift the gap read through pooled tokens of this encoder, up to the upper end of the interval of E, which is printed as the gain the read leaves open. The data-scale line of the lead read through pooled tokens ends there; the lever of the gap line stays the camera crop and the box read at pixel level (rows 172 and 191).
* `WORSE`: as written. The evaluation windows come from the corpus only, so a domain effect of the added clips is the first reading; no further run is asked on this card's strength.

## 9. Families

LONGITUDINAL (distance keeping) is the family this lever serves and is not measured by a token probe. LATERAL and TACTICAL are not measured. STRATEGIC is not computed, n = 0: a token probe reads no strategic quantity.

## 10. Limits

One frozen encoder; pooled tokens; the evaluation windows come from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds; the corpus probes are banked runs of 2026-11-20 on the same pod and tool. An interval that lies above or below 0 on four seeds is necessary for a name and not sufficient for a claim about training runs at large. A probe is not a model.
