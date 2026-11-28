# PREREG_V8F_PROOF_ADD95 (GAPRULE10KB): planted gap error inside the lead rule, on the model's own rollout; the form of card ADD94 with the limit clause of control j1 changed

Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), the two frozen 10k checkpoints UE_s0 and UE_s1. Evidence class of every value it will give: MEASURED. A study arm: no form of this card is proposed for the candidate, and nothing here changes the evaluation convention.

## 0. Why a new card, and what was seen before it was written

Card ADD94 failed at its control j1 at the build of the planted joins, before any wrapper check and before any rollout: for arm GC20, 0.0296 of the frames with a valid lead gap were changed by the limits of its section 2, against a bar of 0.02. By its section 4 card ADD94 is reported as failed with no name, and a changed form is a new card: this one. Seen by the lead before this card was written: that one line of the build output of card ADD94, and that no other arm had a failing line. Not seen: any other value of the build record, any rollout value with a planted gap (none exists). This card is written before any tool of it exists.

What is changed against card ADD94, and why: only the limit clause of control j1. The upper limit is the rule's largest gap (60 m). A scale error of 20 percent that stays with one lead runs into it at the far end of the range, and there the planted error is smaller than its level. The bar of 2 percent is kept where contact is decided, on the frames with a label gap below 40 m (the two lower range bins of row 152); over all frames with a valid lead gap the bar is 5 percent, and the share is printed per arm in the row. The bar of 5 percent was chosen after the 0.0296 was seen; this is stated in the row. Arms, noise seed, measures, name rules and the other controls are those of card ADD94, repeated here in full so that this card stands alone. Names carry the prefix GAPRULE10KB.

## 1. What the card asks

Rows 151 and 153 set a bar on the gap error of a new read of the lead at T0, on the closing read alone: a scale error that stays with one lead is carried up to 20 percent, a slow error up to 1 m, a white frame-to-frame error up to 0.5 m. Both rows state the limit that a pass at T0 is not a pass in the rule. In the rule the gap is also the distance input, so a scale error of the gap that a slope does not see may still move the contact and the falling behind. This card plants the same kinds of error on the label gap and gives the planted gap to the rule as its only source of the lead: the distance is the planted gap, and the closing speed is the slope of the planted gap by the rule's own run rule (at most 5 frames, a step of more than 3.0 m ends the run, a run below 3 frames gives 0.0), exactly as the label-fed form reads the label gap.

Known to the lead from rows on the remote: at this tier and pre-roll the contact rate is 0.0954 for the reader-fed form without a closing input and 0.0138 for the reader's gap with the label closing speed, with falling behind 0.4954 and 0.4738 (rows 145 and 146).

## 2. Arms

All arms are the label-fed form with closing speed of the study wrapper, with the arguments of the banked label-fed arm of card ADD85 (same checkpoint, cycle, readout, pre-roll, heading bound), and differ only in the join that the rule reads its lead from. The scoring of every arm uses the unchanged evaluation join.

* Unit noise, one draw per clip and kind, seeded by the clip and the number 94, shifted and scaled to mean 0 and standard deviation 1 exactly over all frames of the evaluation join that have a valid lead gap: W white; R first-order autoregression of 0.94 per frame; C constant over the clip.
* Planted joins (the lead gap replaced on the frames with a valid lead gap, everything else unchanged): G0, the label gap itself; GR1, label gap + 1 m of R; GR2, label gap + 2 m of R; GW05, label gap + 0.5 m of W; GC10, label gap * (1 + 0.10 * C); GC20, label gap * (1 + 0.20 * C). A planted value below 0.5 m is set to 0.5 m and one above the rule's largest gap to that gap; the number of values changed this way is recorded per arm, over all frames with a valid lead gap and over those with a label gap below 40 m.
* The joins are built again in a new folder by a generated variant of the tool of card ADD94 in which only the record and the check differ; the arrays of the six joins have to be equal to those of the build of card ADD94 (control j1).
* References, banked rollouts of card ADD85 at the same tier and pre-roll: RR (reader-fed, no closing input), RLc (label-fed with closing speed).
* Twelve rollouts: six planted joins times two model seeds.

## 3. Measures and names (frozen)

Measures at the pre-roll of 30 steps (6 s), as the reader of card ADD85 defines them: contact C, unsafe U, falling behind by more than 10 m F, on the windows with a lead; paired differences with a two-level bootstrap over model seeds and evaluation episodes, 2,000 draws, percentile 2.5 and 97.5. Every section that the reader of card ADD85 prints for an arm, the four metric families included, is printed for each arm of this card; a family that the rule does not move is stated as such with its reason.

For each planted arm X in GR1, GR2, GW05, GC10, GC20:

* Contact against the candidate's form: `GAPRULE10KB-<X>-CUTS-CONTACT` when the upper end of C(X) minus C(RR) is below 0; `GAPRULE10KB-<X>-CONTACT-NOT-SHOWN` otherwise.
* Contact against the label: with H = half of the point value of C(RR) minus C(RLc), `GAPRULE10KB-<X>-WITHIN-HALF-THE-GAP` when the upper end of C(X) minus C(RLc) is at or below H; `GAPRULE10KB-<X>-OUTSIDE-HALF-THE-GAP` when its lower end is above H; `GAPRULE10KB-<X>-HALF-THE-GAP-NOT-DECIDED` otherwise.
* Price: `GAPRULE10KB-<X>-FALLS-BEHIND-MORE` when the lower end of F(X) minus F(RR) is above 0; `GAPRULE10KB-<X>-FALLS-BEHIND-LESS` when its upper end is below 0; `GAPRULE10KB-<X>-FALLING-BEHIND-NOT-SEPARATED` otherwise.
* Reported with no name: C, U and F of every arm per model seed and pooled; the differences of U; the share of the contact gap C(RR) minus C(RLc) that each arm removes, with its interval; the realised spread of each planted join and its numbers of changed values.

## 4. Controls

All have to pass before any name or any value of this card is opened. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* j0 (the new argument changes nothing else): arm G0, the label gap given through the rule's join argument, has per-window measures equal to those of the banked arm RLc of the same model seed (largest absolute difference 0).
* j1 (planted joins): for every planted join the standard deviation of the planted error before the limits of section 2 is the level within 1e-9 and its mean is 0 within 1e-9 (absolute error for GR1, GR2, GW05; planted over label gap minus 1 for GC10, GC20); the lag-1 autocorrelation of the unit noise is within 0.05 of zero for W, between 0.89 and 0.97 for R and at least 0.999 for C; every field of the join other than the lead gap is equal to the evaluation join; in any arm at most 2 percent of the frames with a valid lead gap and a label gap below 40 m are changed by the limits, and at most 5 percent of all frames with a valid lead gap; every array of each of the six joins is equal to the same array of the build of card ADD94 (largest difference 0).
* j2 (wrapper): the new generation of the study wrapper without the rule's join argument gives the same per-window measures as the generation it was written from, on the pod check of the earlier generations, here on the first 8 clips of the pre-roll grid.
* j3 (references): the reader recomputes C of RR and of RLc from the banked rollouts and finds for RR the 0.0954 of rows 145 and 146.
* j4: the tests of the kit, run by the lead on the dev box before the rollouts: planted faults (a noise that is not scaled, a proportional level planted as an absolute one, a join with another field changed, a rule join passed to the scoring, a name written with a control failed, a share of limited frames above each of the two bars, a join that differs from the earlier build) each turn their check red.

## 5. What follows, fixed now

* GR1 and GW05 (the levels at the T0 bar) cut contact and lie within half the gap, GR2 does less: the bar of rows 151 and 153 holds in the rule, and the cost card states it with this confirmation.
* GR1 or GW05 do not cut contact: the T0 bar does not carry over to the rule; the cost card then says that the pilot is judged in T1-kin directly and gives the values of this card as what planted error of that size does.
* GC10 and GC20 say what a scale error that stays with one lead does through the distance input, which T0 cannot see. If they cut contact with no price in falling behind, the cost card keeps the statement that the pilot need not reach the absolute scale; if they do not cut contact or fall behind more, the cost card puts a bar on the scale error at the largest level of the two that passes, or below 10 percent if none does.
* No arm of this card is a candidate form: the planted gap is built from the label.

## 6. Limits

T1-kin only: kinematic rollout, no re-perception. The lead moves as logged whatever the ego does. One draw of planted error per clip and kind; Gaussian; the three kinds are planted one at a time. Two model seeds of one line. The rule's own run of at most 5 frames is used and not the run of 10 of rows 151 and 153. Where the limit of 60 m acts, the planted error is smaller than its level: for GC20 on 0.0296 of the frames with a valid lead gap (build of card ADD94). Designed after the reads of cards ADD89 to ADD93 and after the failed control j1 of card ADD94; the limit clause of j1 was written with the 0.0296 known.
