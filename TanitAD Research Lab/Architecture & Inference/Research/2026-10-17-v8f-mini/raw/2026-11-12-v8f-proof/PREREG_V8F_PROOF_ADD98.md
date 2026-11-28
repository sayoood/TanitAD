# PREREG_V8F_PROOF_ADD98 (ERRPART10K): which part of the present reader's gap error carries the contact inside the lead rule

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), the two frozen 10k checkpoints UE_s0 and UE_s1. Evidence class of every value it will give: MEASURED. A study arm with the label: no form of this card is proposed for the candidate, and nothing here changes the evaluation convention. Written after the read of card ADD97 (names and values opened on 2026-11-21, row 159 not written at the time of writing) and before any tool, join or rollout of this card exists.

## 1. Why

Row 158: the lead rule needs the present reader's gap error cut by 2. Card ADD97: the mean of the two reader seeds and a causal smoothing do not deliver that (`SMOOTHGAP10K-NO-PROCESSED-GAP-SUFFICES`), and more smoothing gave more contact, not less. A new read of the lead is a decision of the PI (a new encode). Before that decision this card asks what such a read has to fix: the part of the error that stays with a clip, the part that drifts within a clip, or the part that changes from frame to frame. Row 157 planted Gaussian error of these kinds; this card takes the reader's own error apart with the label and gives the rule one part at a time.

Known to the lead: the values of rows 157 and 158 and of card ADD97, and the baseline of row 154 (the reader's error has 6.87 m about the clip mean, 2.96 m between neighbouring frames). Not known: any statistic of the parts defined below, and any rollout value with them.

## 2. Parts and arms

All arms are the label-fed form with closing speed of the study wrapper of cards ADD95 to ADD97 (unchanged), with the arguments of the banked label-fed arm, and differ only in the join that the rule reads its lead from. The scoring of every arm uses the unchanged evaluation join. Model seed s uses reader seed s. g is the reader's gap as the rule reads it (after the q50 map of that seed), L the label gap. A frame is a mix frame when it has a valid label gap and a reader read, as in cards ADD96 and ADD97. On the mix frames e = g - L.

* m (clip part): the mean of e over the mix frames of the clip.
* r = e - m (the part within the clip).
* s (slow part): the mean of r over the mix frames at most 5 frames before and 5 frames after the frame (1 s on each side) inside the same run. A run is a set of neighbouring mix frames of one clip in which the label gap changes by at most 3.0 m from one frame to the next (the boundary of the label rule's own run rule). The window is cut at the ends of the run.
* f = r - s (the fast part).

Arms, each a join per model seed; on a mix frame the join carries P, on a frame with a valid label gap and no reader read the label gap, every other frame is unchanged; no limit is applied:

* M: P = L + m. W: P = L + r. S: P = L + s. F: P = L + f.
* Eight join files and eight rollouts (four arms times two model seeds).
* References, banked: RR, RLc (P = L), and arm K100 of card ADD96 (P = L + e, all parts).

## 3. Measures and names (frozen)

Measures and intervals as in cards ADD96 and ADD97 (contact C, unsafe U, falling behind by more than 10 m F on the windows with a lead; paired differences, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws, percentile 2.5 and 97.5). The four metric families are printed for each arm against RLc and against RR, the STRATEGIC reading inside the reader under its own engine controls.

For each arm X in M, W, S, F:

* `ERRPART10K-<X>-CARRIES-CONTACT` when the lower end of C(X) minus C(RLc) is above 0; `-NO-CONTACT-SHOWN` otherwise.
* `ERRPART10K-<X>-FALLS-BEHIND-MORE` when the lower end of F(X) minus F(RLc) is above 0; `-FALLS-BEHIND-LESS` when its upper end is below 0; `-FALLING-BEHIND-NOT-SEPARATED` otherwise.
* `ERRPART10K-<X>-LESS-CONTACT-THAN-K100` when the upper end of C(X) minus C(K100) is below 0; `-MORE-CONTACT-THAN-K100` when its lower end is above 0; `-CONTACT-NOT-SEPARATED-FROM-K100` otherwise.

One name for the card, from the names of M and W: `ERRPART10K-THE-PART-WITHIN-THE-CLIP-CARRIES-IT` when W is CARRIES-CONTACT and M is NO-CONTACT-SHOWN; `ERRPART10K-THE-CLIP-PART-CARRIES-IT` when M is CARRIES-CONTACT and W is NO-CONTACT-SHOWN; `ERRPART10K-BOTH-PARTS-CARRY-IT` when both are CARRIES-CONTACT; `ERRPART10K-NO-PART-SHOWN` when neither is.

Reported with no name: C, U and F of every arm per model seed and pooled; every difference against RR; the differences of U and of F against K100; C(X) over C(K100) with the interval of the difference over the point value; per join the statistics of P minus L on the mix frames in the terms of row 158 (mean, standard deviation, root mean square about the clip mean, standard deviation of the clip means, change between neighbouring frames), the counts of P below 0 and above the rule's largest gap, and the number of runs with the median run length.

## 4. Controls

All have to pass before any name or any value of this card is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* q0 (the dump is on the frames of the join): control e0 of card ADD96, computed again for this build.
* q1 (part joins): the saved gap of every join on every mix frame equals the part computed a second time by a separate routine from the dumps, the maps and the label gap, within 1e-4 m; on every mix frame the saved values obey (P_M - L) + (P_W - L) = g - L and (P_S - L) + (P_F - L) = P_W - L within 1e-4 m; the mean of P_W - L over the mix frames of every clip is 0 within 1e-4 m; the frames with a valid label gap and no reader read carry the label gap and are at most 5 percent; every other field of the join equals the evaluation join; the two map files are the q50 maps of the two reader seeds. The counts of P below 0 and above the rule's largest gap are recorded and are no pass criterion.
* q2 (wrapper and argument): control e2 of card ADD96.
* q3 (references): the reader recomputes C of RR, of RLc and of K100 from the banked rollouts and finds 0.0954, 0.0000 and 0.0477 (rows 157 and 158).
* q4: the tests of the kit, run by the lead on the dev box before the rollouts: planted faults (the clip mean taken from another clip, the slow mean running across a run boundary, the slow and the fast part swapped, a window of another length, the other reader seed, a join with another field changed, a name written with a control failed) each turn their check red.
* q5: the engine's own controls of the grid read and of the strategic grid pass.

## 5. What follows, fixed now

* `THE-PART-WITHIN-THE-CLIP-CARRIES-IT`: the cost card states the target of a new read of the lead as steadiness on one lead over time and not as absolute range; the names of S and F say whether the drift or the change from frame to frame is the part to cut, and the pilot's pass criterion keeps its form (the slope of the pilot's own gap).
* `THE-CLIP-PART-CARRIES-IT`: an offset that stays with a clip has a price in the rule; the cost card adds the absolute gap per clip to the target, with the size of the present reader's clip part beside it.
* `BOTH-PARTS-CARRY-IT`: the target stays the cut by 2 of row 158 over the whole error.
* `NO-PART-SHOWN`: the parts are too small to be read one at a time on these windows; reported as that, and the target stays the cut by 2 of row 158.
* In every outcome the result goes into the cost card for the PI before it is sent, and no arm is a candidate form: every join is built with the label.

## 6. Limits

T1-kin only: kinematic rollout, no re-perception. The lead moves as logged whatever the ego does. The parts are taken apart with the label and with a centred window, so no part is something a reader delivers. Contact need not add up over the parts. Arm M moves the distance input and leaves the label's slope as it is; an offset can take a frame out of the rule's range (counted). Four arms; contact differences rest on few windows (row 158: 0.0477 is 31 of 650). Two model seeds of one line, one reader seed per model seed. Designed after the reads of cards ADD95, ADD96 and ADD97.
