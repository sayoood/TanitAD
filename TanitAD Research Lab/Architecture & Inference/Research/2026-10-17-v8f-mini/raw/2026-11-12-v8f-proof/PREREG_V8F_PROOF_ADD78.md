# PREREG_V8F_PROOF_ADD78 (CLOSEREAD2): can a closing speed be read from the token reader's own per-frame outputs over a history? Second card, after card ADD77 did not pass its controls

Tier T0 (a probe on cached tokens of logged frames; it says what the reader's outputs allow, not what a model does with them). Evidence class of every value it will give: MEASURED. No estimate of any cell of card ADD77 was computed or seen: the fit tool of that card stops before the cells when a control fails, and its negative control writes a flag only.

## 1. Why a second card

Card ADD77 (with note N1) is reported as written: control c2 did not pass, so it has no name (row 138). The diagnosis, from labels and row indices only (`TIMES_NOTE.md`, 2026-11-20):

* c2 as written compares the two-point change of the label gap over five frames with the label closing speed on every frame with a lead on all five frames. It read r 0.582 on 3,578 frames (bar 0.70). On the 3,501 of them whose trailing label run has all five frames the two readings agree (r 0.9987, slope 0.999). On the other 77 frames the label gap steps by more than 3 m inside the five frames (the label source cuts its run there and takes the slope after the step, the two-point change spans the step: values up to 53.6 m/s); those frames pull the pooled correlation down. Sign, unit and frame clock of the target are not what failed.
* c3 as the tool had it failed on a flag the card does not ask for: 141 rows of the rule log (frame 3, the first tick, one per clip) carry a read where the dump has no five-frame window in the clip. On the 7,056 rows with the window in the clip the values agree (largest difference 1.4e-06 in presence and 7.6e-05 m in the gap after the map; bar 1e-3).

By the rule of this fork (row 136) a control whose form is in question is not changed inside its card. The changed form of c2 is fixed here, on a new card, before any estimate exists; and a row that gives a name under this card is read by the other session before it is landed.

## 2. Question, data, arms, read and names

As card ADD77 sections 1 to 4 with its note N1 (sections 2 and 3), with these fixed points:

* Data: the dumps of the run of card ADD77 are used as they are (per reader seed the fit and score files and the list file in `closeread/dump/both` on the pod; their hashes are written into the result of this card). They hold reads and labels per frame and no estimate. No new dump, no GPU job.
* Target: the closing speed of the label source of the study wrapper (least-squares slope of the label gap over the trailing run of at most five frames, at least three), as note N1 section 3 has it. Frames whose run has three or four frames stay in the fit and score sets, as in card ADD77: they are targets the label-fed rule was given.
* Arms L and N, histories 5, 10, 25 frames, the two beside rows, the two-level bootstrap (reader seeds, then evaluation episodes; B 2,000; the episode unit is the clip, measured in note N1), the bar 0.30: unchanged. Arm N is the mean of its three nets; the pooled r is over the rows of both reader seeds; c4 is run at history 25.
* Names carry the prefix `CLOSEREAD2`: `CLOSEREAD2-<ARM>-H<H>-READS-CLOSING` (pooled r at least 0.30 with the lower end of its interval above 0, r at least 0.30 in both reader seeds, and the paired difference of the absolute errors, estimate minus constant 0, with its interval below 0), else `CLOSEREAD2-<ARM>-H<H>-NO-CLOSING`; `CLOSEREAD2-A-CLOSING-IS-READABLE` when at least one cell reads READS-CLOSING, else `CLOSEREAD2-NOT-READABLE-FROM-READER-OUTPUTS`. Six cells are read; a single passing cell is named and marked as one of six.
* Four families: this probe measures none of them. LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity).

## 3. Controls (all have to pass; else no name and no outcome value is opened)

* c1, c4, c5, c6, c7: as card ADD77.
* c2 (definition, unit and clock of the target): on score frames whose trailing label run has all five frames, the two-point change of the label gap over those five frames against the label closing speed: r at least 0.70 with positive sign, the regression slope of the label closing speed on the two-point change between 0.8 and 1.2, at least 500 frames, and the correlation lower when the labels are shifted by plus or minus 10 frames. The slope band is new and makes the control stricter where it is meant to bite: a target in another unit or on another frame period fails it at r 1. Said plainly: the value of this form on the score labels was measured in the diagnosis (r 0.9987, slope 0.999) before this card was frozen, so its pass is known; what the control still does is refuse a tool whose target has another sign, unit or clock (planted arms of the self-test: sign turned, unit times 5, labels of another clock). Beside it, reported and not judged: c2 of card ADD77 as written (r 0.582, 3,578 frames).
* c3 (the dump is the reader the rule gets), the text of card ADD77 with note N1: on the replan-tick rule log of `I10RQ50KF5Hb20` of both seeds, on every row with the five-frame window in the clip, presence and gap read after the q50 map equal the dump at the same clip and frame (largest difference at most 1e-3), with rows of the first replan tick among them, and no such row without a read in the rule log. Rows of the rule log where the dump has no five-frame window are counted and reported, not judged (the wrapper reads there with the frames that exist; such frames are in no fit or score row).

A control that does not pass: the card is reported with that control failed and has no name. No form of a control of this card is changed inside the card.

## 4. Tool

`closeread/closeread_fit_n2.py`: imports the fit tool of card ADD77 unchanged (`closeread_fit.py`, sha256 b71f481cdc57e6ec, self-test and 26 mutations of its kit) and replaces the two control functions and the name prefix. Its own self-test (planted arms for both controls and an end-to-end run on synthetic data) has to pass on the pod before the read. CPU only.

## 5. What follows (committed before the data)

As card ADD77 section 6. `CLOSEREAD2-A-CLOSING-IS-READABLE`: the estimator of the passing cell with the largest pooled r becomes a closing source of the study wrapper (a new generation with its pod check) and is read in T1-kin at age and fresh with the second look against the forms of row 137 (own card); only that rollout read can say that it lowers contact. `CLOSEREAD2-NOT-READABLE-FROM-READER-OUTPUTS`: the next probe is a temporal head on the token features (own card), then the cost card to the PI for a lead-region crop or a new encode; beside it a rule form that needs no closing speed. In both cases the row of this card goes to the other session before it is landed.

## 6. Limits

As card ADD77 section 7, with its sentence on the label closing speed read as note N1 section 3 has it: the target is a slope of the label gap over at most one second. One frozen encoder and pooled tokens; two reader seeds; 432 fit clips; logged frames only; six cells.
