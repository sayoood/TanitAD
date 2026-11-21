# PREREG_V8F_PROOF_ADD101 (CROPRULE10K): the pilot's gap inside the lead rule

Fork v8proof, 2026-11-21. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `stack/scripts/v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, 1,271 windows per model seed. Evidence class: MEASURED. Designed after the read of card ADD99 (row 161). Frozen with its hash file before any tool, join or rollout of it exists. A study of a changed input of the rule; no arm is a candidate form, and adoption of any form is the PI's and the other session's decision.

## 1. Question

Row 158 put the cut of the present reader's gap error that the rule needs at 2. Card ADD99 measured at T0 a gap from crops: the bound arm (windows on the lead from the label) cuts the error to about a third, the fixed windows to 0.80. What is each worth in contact inside the rule?

## 2. Arms

The rule's source of the lead is the label source of cards ADD95 to ADD98 (the label marks the frames with a lead; the rule's closing input is the slope of the gap it reads), with the join's gap replaced, on the frames with a valid label gap and a pilot output, by the pilot's gap of card ADD99 (the head's output as dumped, no map, no minimum); frames with a valid label gap and no pilot output keep the label gap. Model seed k reads the pilot's fit seed k.

* KB5: bound arm, head H5. KB1: bound arm, head H1. KD5: fixed windows, head H5. KD1: fixed windows, head H1. Eight rollouts.
* References, banked on the pod and recomputed by the reader: the reader-fed form without a closing input (RR, contact 0.0954), the label gap (RLc, 0.0000), the present reader's gap on the same frames (K100, 0.0477). Half the gap H = 0.0477.

## 3. Names

Per arm, by the rules of card ADD97 with the arm in place of its arms (paired differences, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws):

* contact against RR: `CROPRULE10K-<arm>-CUTS-CONTACT` when the upper end of the difference is below 0, else `-CONTACT-NOT-SEPARATED-FROM-RR` (or `-MORE-CONTACT` when the lower end is above 0);
* contact against RLc: `-WITHIN-HALF-THE-GAP` when the upper end of the difference is at or below H, `-HALF-THE-GAP-NOT-REACHED` when the lower end is above H, else `-HALF-THE-GAP-NOT-DECIDED`;
* contact against K100: `-LESS-CONTACT-THAN-K100` when the upper end is below 0, `-MORE-CONTACT-THAN-K100` when the lower end is above 0, else `-CONTACT-NOT-SEPARATED-FROM-K100`;
* falling behind by more than 10 m against RR: `-FALLS-BEHIND-MORE`, `-FALLS-BEHIND-LESS` or `-FALLING-BEHIND-NOT-SEPARATED` by the same rule.

Card names: from KB5 `CROPRULE10K-BOUND-WITHIN-HALF-THE-GAP`, `-BOUND-HALF-THE-GAP-NOT-REACHED` or `-BOUND-HALF-THE-GAP-NOT-DECIDED`; from KD5 the same with `FIXED`. Printed with no name: unsafe rate, values per model seed, the size of the pilot's error on the join's frames, the frames on which the gap leaves the rule's range.

Reading committed in advance: KB5 within half the gap says a crop on the lead with a 500-clip head is enough for the rule, so the open piece is where the crop sits and whether the lead is present (card ADD100 and a form with no label at inference). KD5 within half the gap says the fixed windows already carry it and the next card reads the form with no label at inference. Neither: the T0 cut does not carry into the rule at this size, and the result goes into the cost card for the PI.

## 4. Controls

All have to pass before any name or any value of this card is opened; logged in `TIMES_NOTE.md` before the names. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* q0: the pilot's dumps are on the frames of the evaluation join (every dump row has its join row; clip order and frame counts agree).
* q1: the saved gap of every join equals a second routine's value within 1e-4 m on every frame; frames with a valid label gap and no pilot output carry the label gap and are counted; every other field equals the evaluation join; the lead marking (`lead_valid`) is unchanged.
* q2: the study wrapper cbf7063d867c9f4e and the rule-join argument are the ones of card ADD95 (its pod check record), unchanged.
* q3: the reader recomputes contact 0.0954 for RR, 0.0000 for RLc and 0.0477 for K100 from the banked rollouts.
* q4: the tests of the kit pass in the lead's run on the dev box before the rollouts.
* q5: the engine's own controls of the grid read and of the strategic grid pass.

## 5. Four metric families

LONGITUDINAL (distance-keeping): contact, unsafe rate, falling behind, with the engine's pair read. LATERAL, TACTICAL: the engine's pair read against the logged drive, as in cards ADD95 to ADD98. STRATEGIC: read inside the reader under its own engine controls for every arm against RR and RLc, as in cards ADD95 to ADD98.

## 6. Limits

T1-kin only, no re-perception; the lead moves as logged whatever the ego does; the label marks the frames with a lead in every arm, and the bound arm also takes the crop position from the label; two model seeds of one line, one fit seed per model seed; contact differences rest on few windows; designed after the read of card ADD99.
