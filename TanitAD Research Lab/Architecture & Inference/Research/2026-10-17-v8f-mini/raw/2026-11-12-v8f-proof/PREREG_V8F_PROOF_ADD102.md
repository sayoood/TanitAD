# PREREG_V8F_PROOF_ADD102 (VISRULE10K): the lead rule with no label at inference, gap from the fixed-window head

Fork v8proof, 2026-11-21. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `stack/scripts/v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, 1,271 windows per model seed. Evidence class: MEASURED. Designed after the read of card ADD101 (row 162: with the label marking the frames with a lead, the gap of the fixed-window head of card ADD99 gives contact 0.0169, within half the gap to the label gap). Frozen with its hash file before any tool, file or rollout of it exists. A study of a changed input of the rule; adoption of any form is the PI's and the other session's decision.

## 1. Question

Does the contact stay within half the gap when no label enters at inference: the frames with a lead come from the present reader's presence output, the gap from the fixed-window head?

## 2. Arms

The reader-fed path of the study wrapper (the path of the form RR of the candidate spec: a lead on a frame when the present reader's presence output says so under the wrapper's own rule), with one change: the gap the rule receives on such a frame is the gap of head H5 on the fixed windows of card ADD99 (D-H5, the head's output as dumped, no map, no minimum; model seed k reads fit seed k). On a frame where the reader marks a lead and the head has no output (frames 0 to 3 of a clip) the rule receives the reader's own gap as in RR.

* VD5c: the rule's closing input is the slope of the gap it receives, by the run rule of the label source (least squares over the trailing run of at most 5 frames, a step of more than 3.0 m ends the run, a run below 3 frames gives 0).
* VD5: no closing input, as in RR.
* Four rollouts, plus the two of control v1.
* References, banked and recomputed by the reader: RR (0.0954), RLc (0.0000), KD5 of card ADD101 (0.0169, the same gap with the label marking the lead). Half the gap H = 0.0477.

## 3. Names

Per arm (paired differences, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws; the rules of card ADD101):

* contact against RR: `VISRULE10K-<arm>-CUTS-CONTACT`, `-CONTACT-NOT-SEPARATED-FROM-RR` or `-MORE-CONTACT`;
* contact against RLc: `-WITHIN-HALF-THE-GAP` (upper end at or below H), `-HALF-THE-GAP-NOT-REACHED` (lower end above H), else `-HALF-THE-GAP-NOT-DECIDED`;
* contact against KD5: `-PRESENCE-COSTS-CONTACT` when the lower end is above 0, else `-CONTACT-NOT-SEPARATED-FROM-KD5` (or `-LESS-CONTACT-THAN-KD5` when the upper end is below 0);
* falling behind by more than 10 m against RR: `-FALLS-BEHIND-MORE`, `-FALLS-BEHIND-LESS` or `-FALLING-BEHIND-NOT-SEPARATED`.

Card name, from VD5c: `VISRULE10K-WITHIN-HALF-THE-GAP`, `VISRULE10K-HALF-THE-GAP-NOT-REACHED` or `VISRULE10K-HALF-THE-GAP-NOT-DECIDED`. Printed with no name: unsafe rate, values per model seed, the share of rule steps with a lead in each arm and in RR, the share of frames with a valid label lead that the reader marks and the share of marked frames with no label lead.

Reading committed in advance: WITHIN says a form with no label at inference lies within half the gap; it goes to the PI and the other session as a possible input of the rule, with its price in the family reads beside it. NOT-REACHED names the presence as the next lever (a presence output on the crop tokens), with a card to follow. NOT-DECIDED goes to the PI with the number.

## 4. Controls

All have to pass before any name or any value of this card is opened; logged in `TIMES_NOTE.md` before the names. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* v0: the head's dumps are on the frames of the evaluation join (141 clips in the same order, every dump row with its join row).
* v1 (the variant is the wrapper): the wrapper variant with the gap change switched off, run for both model seeds, gives the rates of the banked RR rollouts: contact, unsafe rate and falling behind each within 2 windows of the 650 with a lead.
* v2 (the change does what the card says): in the rule log of every rollout of VD5c and VD5, on every logged step where the rule had a lead and the head has an output for the frame, the gap the rule received equals the head's dumped gap of that clip and frame within 1e-3 m; on the steps where the head has no output it equals the reader's own gap as RR feeds it; in VD5 the closing input is 0 on every step.
* v3: the reader recomputes contact 0.0954 for RR, 0.0000 for RLc and 0.0169 for KD5 from the banked rollouts.
* v4: the tests of the kit pass in the lead's run on the dev box before the rollouts.
* v5: the engine's own controls of the grid read and of the strategic grid pass.

## 5. Four metric families

LONGITUDINAL (distance-keeping): contact, unsafe rate, falling behind, with the engine's pair read. LATERAL, TACTICAL: the engine's pair read against the logged drive. STRATEGIC: read inside the reader under its own engine controls for every arm against RR and RLc.

## 6. Limits

T1-kin only, no re-perception; the lead moves as logged whatever the ego does; the head reads crops of logged frames, so the crop does not move with the rolled-out ego; two model seeds of one line, one fit seed per model seed; contact differences rest on few windows; the head is fit on 500 clips; designed after the reads of cards ADD99 and ADD101.
