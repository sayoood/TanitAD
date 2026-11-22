# Card ADD111 (GAPERR10K): which part of the camera gap error carries the price in falling behind, and do two smoothing forms with no label recover it

Pre-registration of the v8proof fork, written 2026-11-22 after card ADD110 was read (controls a0 to a5 passed, names opened; row 176 in writing) and before any tool of this card is written. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record `v8f_t1_eval_hb.py` with the heading bound 0.2 1/m), pre-roll of 30 steps (6 s), the two 10k models `UE_s0` and `UE_s1`. No training, no fit, no encode: rollouts of study arms on existing models with gap files composed from files that exist.

## 1. Question and why

Card ADD110: with the label gap on every label row and the reader's own presence (arm LA), falling behind drops by 0.0277 [-0.0539, -0.0060] against the camera-fed form GEc and equals the label form RLc (LA minus RLc -0.0031 [-0.0103, +0.0000]); name `FALLATTR10K-GAP-CARRIES-THE-PRICE`. No range of the gap carried it alone (`FALLATTR10K-NO-RANGE-SEPARATED`), and by that card's committed reading the gap error as a whole is the lever; a finer split by range needs more model seeds, which is a PI decision. This card takes the next lever that needs none: it splits the error of the camera gap in time (a slow part, the level over about two seconds, and a fast part, the frame-to-frame rest) and it measures two smoothing forms of the camera gap that use no label.

## 2. Arms

Base: GEc of card ADD106 (banked rollouts; gap file `VGAPLE.npz`, the mean gap of the three large fits). All arms have the form VD5c of card ADD102 (wrapper variant 28bc31f21ecb4ccb unchanged; presence from the reader, the gap of a gap file, the closing input as the slope of the gap the rule receives). Each arm has its own gap file on the frames of the evaluation join (14,237 rows; 3,986 label rows; the base file has an output on 13,673 rows, and 145 label rows have none).

Terms. Label rows: valid flag 1, finite label gap from 0 to 60 m. Error rows: label rows where the base file has an output. Error e: base gap minus label gap on an error row. Error run: a largest set of error rows of one clip on consecutive frames. Slow part s of a row: the mean of e over the rows of its error run whose frame lies within 5 frames before and 5 frames after the row, the row included (fewer near the ends of the run). Fast part f: e minus s.

Attribution arms (label values in the input; not admissible at inference, never candidates):

* NR (fast part removed): label gap plus s on every error row.
* BR (slow part removed): label gap plus f on every error row.
* In both, a label row with no output of the base file has the label gap and an output, as in arm LA of card ADD110; every other row has gap and output flag of the base file.

With these, GEc is label plus s plus f on the error rows, LA is the label alone, NR keeps the slow part and BR keeps the fast part.

Smoothing arms (the base file only; no label enters; admissible at inference as a form, and measured here as study arms):

* Trailing run of a row with a base output: the row and the frames before it in the same clip, going back while each frame has a base output, is the frame before the next and differs from the next by at most 3.0 m in the base gap; at most 5 frames (the run rule of the slope of note N1 of card ADD102, on the base gap).
* SM: the median of the base gap over the trailing run.
* SL: the end value of the least-squares line through the base gap over the trailing run when the run has 3 or more frames; the base gap itself when it has fewer.
* The output flag is that of the base file on every row; a row with no base output keeps its base gap field.

Four arms, both model seeds: eight rollouts, run folders `I10RQ50KF5P30Hb20b<ARM>_s<S>`.

References, banked and recomputed by the reader: GEc (contact 0.0062, falling behind 0.5154, unsafe 0.0839), RLc (0.0000, 0.4908, 0.0714), RR (0.0954, 0.4954, 0.1339), LA of card ADD110 (0.0000, 0.4877, 0.0839).

## 3. Measures and names

Measures as in cards ADD101 to ADD110 on the 1,271 windows per model seed (650 pooled windows with a lead): CONTACT, UNSAFE, FALLING-BEHIND (behind the logged ego by more than 10 m). Paired differences by the two-level bootstrap over model seeds and evaluation episodes, 2,000 draws, the rules of those cards.

* Per arm X of NR, BR, SM, SL and measure M of FALLING-BEHIND, CONTACT, UNSAFE, against GEc: `GAPERR10K-<X>-LOWERS-<M>` (upper end of the interval of X minus GEc below 0), `GAPERR10K-<X>-RAISES-<M>` (lower end above 0), else `GAPERR10K-<X>-<M>-NOT-SEPARATED`.
* The parts, from falling behind of NR and BR against GEc: `GAPERR10K-FAST-PART-CARRIES` (NR lowers, BR does not lower); `GAPERR10K-SLOW-PART-CARRIES` (BR lowers, NR does not lower); `GAPERR10K-BOTH-PARTS-CARRY` (both lower); every other case `GAPERR10K-PARTS-NOT-SEPARATED`.
* Each smoothing arm X of SM, SL: `GAPERR10K-<X>-PAYS-SAFETY` when X raises contact or raises unsafe against GEc; else `GAPERR10K-<X>-RECOVERS` when X lowers falling behind against GEc; else `GAPERR10K-<X>-NO-GAIN`.

Printed with no name: levels of every arm per model seed and pooled; every arm against RLc, against RR and against LA on the three measures; per gap file the count of rows that differ from the base file and the count of rows with a label value; per rollout the share of rule steps with a lead and of them with a non-zero closing input. From the gap-file tool, at tier T0 on the evaluation frames (recorded frames, not driving performance), per range of the label gap (below 20 m, 20 to below 40 m, 40 m and above) and over all error rows: the count of error rows and the root mean square of e, of s, of f, and of the error of SM and of SL against the label.

Stated before the data: the whole price is about 0.028 of 650 windows, about 18 windows, and its interval reached -0.006. A part of it can stay inside the resolution of two model seeds, as the range arms did, so `GAPERR10K-PARTS-NOT-SEPARATED` is a likely reading unless one part carries nearly all. The guess written down before the data is the fast part: the closing input is the slope of the received gap over at most five frames, so frame noise of the gap reads as closing and the rule brakes. SM lags a closing lead by two frames, so it can pay in safety; SL has no lag on a straight course of the gap. These are guesses, not findings.

Readings committed in advance. A smoothing arm RECOVERS: reported as a study arm; a smoothed gap as an input of the rule is a design decision that goes to the PI and the other session with the cost card, and the next card reads the same form with the gap of a single fit and at the pre-roll of 60 steps (12 s). FAST-PART-CARRIES and no smoothing arm recovers: the two forms fixed here are too weak or their lag costs what they gain; the next lever is a gap head with frame history fitted on tokens that exist (its own card). SLOW-PART-CARRIES: the next lever is a recalibration of the camera gap fitted out of fold on train clips with tokens that exist (its own card). BOTH-PARTS-CARRY or PARTS-NOT-SEPARATED: the next lever is both together as one study arm (recalibrated and smoothed); a finer split needs more model seeds, which is a PI decision. A smoothing arm that PAYS-SAFETY is not carried further in that form.

## 4. Controls

All have to pass before any name or any value of this card is opened; logged in `TIMES_NOTE.md` before the names. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* b0 (the gap files are what the card says): by a second routine, each saved file equals the definition of section 2 on every join row (largest absolute difference at most 1e-9 m, equal output flags); on every error row label plus s plus f equals the base gap within 1e-9 m; NR and BR equal the file LA of card ADD110 on every row that is not an error row; SM and SL come out byte for byte the same when the label fields of the join are replaced by other values (no label enters them), have the output flags of the base file, give a constant course back unchanged, SL gives a straight course back within 1e-9 m and SM gives it back two frames late once five frames are in the run; the base file is the file whose hash the gap-file record of card ADD106 names; the evaluation join is the pinned one.
* b1 (wrapper): the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass, and every run text names it.
* b2 (each arm is what the card says): in the rule log of every rollout, on every logged step where the rule had a lead and the gap file has an output, the gap the rule received equals the gap of its file at that clip and frame within 1e-3 m; a step with a lead and no output equals a gap of the control run V0 of card ADD102 within 1e-3 m; the meta names the gap file and the closing switch; the closing input is non-zero on a positive share of the steps with a lead.
* b3: the reader recomputes contact, falling behind and unsafe of GEc, RLc, RR and LA as written in section 2 (within 1e-4; the values there are rounded to four places) from the banked rollouts.
* b4: the tests of every kit of this card pass in the lead's run on the dev box before the first rollout starts; the tests of the read kit run the line of the read driver that parses the queue start on the first line of the start marker of card ADD110 as the pod wrote it.
* b5: the engine's own controls of the grid read and of the strategic grid pass.

Limits by construction, stated before the freeze: none of b0 to b5 depends on the subject. The time clause of b4 holds only when no file of a kit changes after the first rollout starts; a change after that is reported with the row as in note N1 of card ADD110. The lead rehearses both stages of the read on the pod with banked rollouts before the queue starts and logs it; the rehearsal is a step of the lead and not a control. In NR the gap is smooth inside an error run and steps at its ends; in BR the fast part sits on the label; where a run of error rows starts or ends the source of the gap changes, a step of more than 3.0 m ends the run of the slope and the closing input is 0 until three frames are in the new run. The slow part is taken with frames after the row and so is not causal; that is admissible for an attribution arm only.

## 5. Four metric families

LONGITUDINAL (distance keeping): contact, unsafe rate, falling behind, with the engine's pair read. LATERAL, TACTICAL: the engine's pair read against the logged drive. STRATEGIC: read at T1-kin inside the reader under its own engine controls for NR, BR, SM, SL against GEc and against RLc.

## 6. Cost, stop rule, limits

Eight rollouts on the fork pod, one after the other, ESTIMATED 1.4 pod-hours from the eight rollouts of card ADD110 (MEASURED there: 84 minutes); stop rule 3 pod-hours from the start of the queue. No PI decision is needed for rollouts of study arms on existing models; nothing of this card enters the candidate spec.

Limits: T1-kin only, no re-perception; the lead moves as logged whatever the ego does; two model seeds of one line; the label gap is the reference and not ground truth; an attribution with label values in the input says where the price sits, not how to remove it; the two smoothing forms were fixed before the data and are two of many; differences rest on few windows; a separated interval from two model seeds is necessary, not sufficient.
