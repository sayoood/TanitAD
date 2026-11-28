# PREREG_V8F_PROOF_ADD96 (ERRCUT10K): by what factor the present reader's gap error has to be cut for the lead rule, on the model's own rollout

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), the two frozen 10k checkpoints UE_s0 and UE_s1. Evidence class of every value it will give: MEASURED. A study arm: no form of this card is proposed for the candidate, and nothing here changes the evaluation convention. Written after the read of card ADD95 (row 157, not landed at the time of writing and waiting for the other session's read) and before any tool, join or rollout of this card exists.

## 1. Why

Row 154 puts the present reader's gap error at 4 to 8 times the bar of row 153, from statistics at T0 and with Gaussian planted error on the other side of the comparison. Row 157 shows that Gaussian error at the bar is carried inside the rule. Neither says what the reader's own error does inside the rule once it is made smaller, with its own time course, its own dependence on range and its own steps. This card measures that: the gap the rule reads is a mix of the label gap and the reader's gap, and the share of the reader's error is cut step by step.

Known to the lead: the levels of row 157 (contact 0.0954 for the reader-fed form without a closing input, 0.0000 for the label gap; falling behind 0.4954 and 0.4908), the T0 statistics of rows 152 and 154 (on the reader's raw gap), and that 3,834 of the 3,986 frames with a valid label gap carry a reader read (row 154). Not known: any rollout value with a mixed gap, and the statistics of the reader's gap as the rule reads it (after the map of the run).

## 2. Arms

All arms are the label-fed form with closing speed of the study wrapper of card ADD95 (the generation with the rule's join argument, unchanged), with the arguments of the banked label-fed arm, and differ only in the join that the rule reads its lead from. The scoring of every arm uses the unchanged evaluation join.

* Reader gap as the rule reads it: g = min(r, Q(r)), with r the raw gap read of the per-frame dump of card ADD78 (reader seed s) and Q the q50 map of that reader seed, applied as the tool of card ADD78 applies it. Model seed s uses reader seed s, as the reader-fed runs do.
* Mixed gap on a frame with a valid label gap L and a reader read: M_k = L + k * (g - L). On a frame with a valid label gap and no reader read: L. Frames without a valid label gap are unchanged. No limit is applied: a mixed value above the rule's largest gap is left as it is (the rule then sees no lead on that frame, as it would with that reader), and the number of such values is recorded per arm.
* Arms: K100 (k = 1, the reader's gap on the frames the label marks), K50 (k = 1/2), K25 (k = 1/4), K12 (k = 1/8). Eight joins (four arms times two reader seeds), eight rollouts.
* References, banked: RR (reader-fed, no closing input) and RLc (label-fed with closing speed) at the same tier and pre-roll; arm G0 of card ADD95 (the label gap through the rule's join argument, equal to RLc in every window, row 157) stands for k = 0.

## 3. Measures and names (frozen)

Measures at the pre-roll of 30 steps (6 s), as the reader of card ADD85 defines them: contact C, unsafe U, falling behind by more than 10 m F, on the windows with a lead; paired differences with a two-level bootstrap over model seeds and evaluation episodes, 2,000 draws, percentile 2.5 and 97.5. The four metric families are printed for each arm against RR and against RLc: the LONGITUDINAL, LATERAL and TACTICAL readings of the engine's pair read and the STRATEGIC reading of the strategic grid of the reader of card ADD85, this time inside the reader of the card and under its own engine controls.

For each arm X in K100, K50, K25, K12, with H = half of the point value of C(RR) minus C(RLc):

* `ERRCUT10K-<X>-CUTS-CONTACT` when the upper end of C(X) minus C(RR) is below 0; `ERRCUT10K-<X>-CONTACT-NOT-SHOWN` otherwise.
* `ERRCUT10K-<X>-WITHIN-HALF-THE-GAP` when the upper end of C(X) minus C(RLc) is at or below H; `ERRCUT10K-<X>-OUTSIDE-HALF-THE-GAP` when its lower end is above H; `ERRCUT10K-<X>-HALF-THE-GAP-NOT-DECIDED` otherwise.
* `ERRCUT10K-<X>-FALLS-BEHIND-MORE` when the lower end of F(X) minus F(RR) is above 0; `ERRCUT10K-<X>-FALLS-BEHIND-LESS` when its upper end is below 0; `ERRCUT10K-<X>-FALLING-BEHIND-NOT-SEPARATED` otherwise.

One name for the ladder, judged in this order: `ERRCUT10K-LADDER-NOT-MONOTONE` when some arm is WITHIN-HALF-THE-GAP while an arm of a larger cut is not (then the per-arm names stand and no factor is named); else `ERRCUT10K-CUT-BY-8-NOT-SHOWN-TO-SUFFICE` when K12 is not WITHIN-HALF-THE-GAP; else `ERRCUT10K-CUT-BY-<n>-SUFFICES`, with n the smallest of 1, 2, 4, 8 whose arm is WITHIN-HALF-THE-GAP (cuts 1, 2, 4, 8 are the arms K100, K50, K25, K12).

Reported with no name: C, U and F of every arm per model seed and pooled; the differences of U; the share of the contact gap that each arm removes, with its interval; per arm and reader seed the statistics of the planted error M_k minus L on the frames with a reader read (mean, standard deviation, root mean square about the clip mean, standard deviation of the change between neighbouring frames, as row 154 defines them) and the number of values above the rule's largest gap.

## 4. Controls

All have to pass before any name or any value of this card is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* e0 (the dump is on the frames of the join): on every frame of the dump that carries a label gap, that label gap equals the lead gap of the evaluation join at the same clip and frame (largest absolute difference at most 1e-4 m), for both reader seeds.
* e1 (mixed joins): on every frame with a valid label gap and a reader read, M_k minus L equals k * (g - L) within 1e-4 m; the frames with a valid label gap and no reader read carry L and are at most 5 percent of the frames with a valid label gap; every other field of the join equals the evaluation join; no mixed value is below 0; the map file used for reader seed s is `QMAP_LR16_s<s>_q50.json` of the folder that the queue of the banked reader-fed runs takes its map from, and its sha256 is recorded.
* e2 (wrapper and argument): the wrapper file on the pod has the hash that the check record of card ADD95 holds (control j2 of that card, passed), and the queue admits the rollouts only with that record.
* e3 (references): the reader recomputes C of RR and of RLc from the banked rollouts and finds 0.0954 and 0.0000 (row 157).
* e4: the tests of the kit, run by the lead on the dev box before the rollouts: planted faults (a mix with the wrong k, the raw gap in place of the mapped gap, the reader seed of the other model seed, a join with another field changed, a rule join passed to the scoring, a name written with a control failed, a ladder name from a ladder that is not monotone) each turn their check red.
* e5: the engine's own controls of the grid read and of the strategic grid pass.

## 5. What follows, fixed now

* `CUT-BY-n-SUFFICES` with n of 2, 4 or 8: the cost card gives n as the cut of the present reader's gap error that the rule needs, measured on the reader's own error, beside the 4 to 8 of row 154.
* `CUT-BY-1-SUFFICES` (K100 is within half the gap): the reader's gap is good enough for the rule once the lead is marked by the label and the closing speed is the slope of that gap; the limit of the candidate's form is then where the reader sees a lead and the closing input, and not the gap error. The cost card is rewritten around that, and the next card tests the reader's own marking of the lead with the slope.
* `CUT-BY-8-NOT-SHOWN-TO-SUFFICE`: an error of the present reader's kind has to be cut by more than 8 or cannot be judged at this number of windows; the cost card says so and the next card adds a cut by 16.
* No arm of this card is a candidate form: every mixed gap uses the label.

## 6. Limits

T1-kin only: kinematic rollout, no re-perception. The lead moves as logged whatever the ego does. A mix of the label gap and the reader's gap is no reader: cutting an error by a factor keeps its time course and its dependence on range, which a better reader need not do. The frames on which a lead exists are the label's in every arm. Two model seeds of one line, each with its own reader seed. Contact differences rest on few windows (row 157: 0.0169 is 11 of 650). Designed after the reads of cards ADD91 to ADD95.
