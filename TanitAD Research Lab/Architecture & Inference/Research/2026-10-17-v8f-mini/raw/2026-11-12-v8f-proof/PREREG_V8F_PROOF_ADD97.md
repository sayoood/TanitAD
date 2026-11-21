# PREREG_V8F_PROOF_ADD97 (SMOOTHGAP10K): does processing the present reader's gap, with no new encode, deliver what the lead rule needs

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), the two frozen 10k checkpoints UE_s0 and UE_s1. Evidence class of every value it will give: MEASURED. A study arm: no form of this card is proposed for the candidate, and nothing here changes the evaluation convention. Written after the read of card ADD96 (row 158, not landed at the time of writing) and before any tool, join or rollout of this card exists.

## 1. Why

Row 158: with the present reader's gap error cut by 2 the lead rule is within half the gap to the label, with no separated price in falling behind; with the reader's gap as it is (arm K100) contact is 0.0477 against 0.0954 for the form without a closing input, and the rule falls behind more. The cut by 2 was made with the label. This card asks whether two operations that need no label and no new encode move the reader's gap far enough: the mean of the two reader seeds, and a causal smoothing over time. The frames on which a lead exists are still the label's, as in card ADD96, so this card isolates the gap and its slope.

Known to the lead: the levels and the error statistics of row 158. Not known: any statistic of the mean of the two reader seeds or of a smoothed gap, and any rollout value with them.

## 2. Arms

All arms are the label-fed form with closing speed of the study wrapper of cards ADD95 and ADD96 (unchanged), with the arguments of the banked label-fed arm, and differ only in the join that the rule reads its lead from. The scoring of every arm uses the unchanged evaluation join. g_s is the reader's gap as the rule reads it (reader seed s, after the q50 map of that seed), as in card ADD96. A frame is a mix frame when it has a valid label gap and a reader read; on a mix frame the join carries the processed gap P, on a frame with a valid label gap and no reader read the label gap, and every other frame is unchanged. No limit is applied; the numbers of values below 0 and above the rule's largest gap are recorded.

* E (mean of the seeds): P = (g_0 + g_1) / 2. One join for both model seeds.
* S50_s and S25_s (causal smoothing of the reader seed s, model seed s): over each run of neighbouring mix frames of a clip, P_t = a * g_t + (1 - a) * P_(t-1), with P = g at the first frame of the run; a = 0.5 and a = 0.25. A run ends where the next frame of the clip is no mix frame.
* ES25 (both): the same smoothing with a = 0.25 on the mean of the seeds. One join for both model seeds.
* For E and ES25 a mix frame needs a read of both reader seeds. Eight join files, one per arm and model seed, of which the two of E and the two of ES25 are equal in every array (six different joins); eight rollouts (four arms times two model seeds).
* References, banked: RR, RLc, and arm K100 of card ADD96 (the reader's gap of seed s with no processing, row 158).

## 3. Measures and names (frozen)

Measures and intervals as in card ADD96 (contact C, unsafe U, falling behind by more than 10 m F on the windows with a lead; paired differences, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws, percentile 2.5 and 97.5). The four metric families are printed for each arm against RR and against RLc, the STRATEGIC reading inside the reader under its own engine controls.

For each arm X in E, S50, S25, ES25, with H = half of the point value of C(RR) minus C(RLc):

* `SMOOTHGAP10K-<X>-CUTS-CONTACT` when the upper end of C(X) minus C(RR) is below 0; `-CONTACT-NOT-SHOWN` otherwise.
* `SMOOTHGAP10K-<X>-WITHIN-HALF-THE-GAP` when the upper end of C(X) minus C(RLc) is at or below H; `-OUTSIDE-HALF-THE-GAP` when its lower end is above H; `-HALF-THE-GAP-NOT-DECIDED` otherwise.
* `SMOOTHGAP10K-<X>-FALLS-BEHIND-MORE` when the lower end of F(X) minus F(RR) is above 0; `-FALLS-BEHIND-LESS` when its upper end is below 0; `-FALLING-BEHIND-NOT-SEPARATED` otherwise.
* Against the reader's gap with no processing: `SMOOTHGAP10K-<X>-LESS-CONTACT-THAN-K100` when the upper end of C(X) minus C(K100) is below 0; `-MORE-CONTACT-THAN-K100` when its lower end is above 0; `-CONTACT-NOT-SEPARATED-FROM-K100` otherwise.

One name for the card: `SMOOTHGAP10K-A-PROCESSED-GAP-SUFFICES` when at least one arm is WITHIN-HALF-THE-GAP and is not FALLS-BEHIND-MORE; `SMOOTHGAP10K-NO-PROCESSED-GAP-SUFFICES` otherwise.

Reported with no name: C, U and F of every arm per model seed and pooled; the differences of U and of F against K100; the share of the contact gap that each arm removes, with its interval; per join the statistics of P minus the label gap on the mix frames in the terms of row 158 (mean, standard deviation, root mean square about the clip mean, standard deviation of the clip means, change between neighbouring frames) and the counts below 0 and above the rule's largest gap.

## 4. Controls

All have to pass before any name or any value of this card is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* p0 (the dump is on the frames of the join): control e0 of card ADD96, computed again for this build.
* p1 (processed joins): the saved gap of every join on every mix frame equals the processed gap computed a second time by a separate routine from the dumps and maps, within 1e-4 m; the smoothed value at a frame does not change when the reader's gap of any later frame is changed (checked on the build by changing the last mix frame of every run and computing again); the frames with a valid label gap and no reader read carry the label gap and are at most 5 percent; every other field of the join equals the evaluation join; the two map files are the q50 maps of the two reader seeds.
* p2 (wrapper and argument): control e2 of card ADD96.
* p3 (references): the reader recomputes C of RR, of RLc and of K100 from the banked rollouts and finds 0.0954, 0.0000 and 0.0477 (rows 157 and 158).
* p4: the tests of the kit, run by the lead on the dev box before the rollouts: planted faults (a smoothing that uses a later frame, a wrong a, a mean that takes one seed twice, a smoothing that runs on across a frame with no read, a join with another field changed, a name written with a control failed) each turn their check red.
* p5: the engine's own controls of the grid read and of the strategic grid pass.

## 5. What follows, fixed now

* `A-PROCESSED-GAP-SUFFICES`: the next card takes the label out of the marking of the lead as well: the reader's own presence marks the lead, with the processing of the best arm that passed. That needs a generation of the study wrapper in which the rule's join may differ from the evaluation join in the lead flag, with its own pod check. Its result goes to the PI and the other session as a possible change of the rule's input; this fork adopts nothing.
* `NO-PROCESSED-GAP-SUFFICES`: the mean of the seeds and a causal smoothing do not deliver the cut by 2 of row 158; the cost card names the cut by 2 as the target of a new read of the lead, with these arms as what was tried with no new encode.
* An arm that is LESS-CONTACT-THAN-K100 and is not within half the gap: reported as a partial step, with its place on the ladder of row 158.
* No arm of this card is a candidate form: every join uses the label to mark the lead.

## 6. Limits

T1-kin only: kinematic rollout, no re-perception. The lead moves as logged whatever the ego does. The frames on which a lead exists are the label's, and the smoothing restarts where the label's lead ends. Four arms are read against one bar. Two model seeds of one line; the mean uses both reader seeds, so the two model seeds of arms E and ES25 share one join. Contact differences rest on few windows (row 158: 0.0092 is 6 of 650). Designed after the reads of cards ADD95 and ADD96.
