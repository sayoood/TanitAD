# PREREG_V8F_PROOF_ADD104 (VISSEED10K): the form with no label at inference over three new fits of the head

Fork v8proof, 2026-11-21. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `stack/scripts/v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, 1,271 windows per model seed. Evidence class: MEASURED. Designed after the read of card ADD103 (`VISREP10K-REPLICATION-NOT-DECIDED`: with the other fit seed of the head the form has a contact rate of 0.0200 against 0.0169, it cuts contact against RR, and its interval against the label gap ends at +0.0498, above half the gap of 0.0477). Frozen with its hash file before any tool, fit, file or rollout of it exists. A study of a changed input of the rule; adoption of any form is the PI's and the other session's decision.

## 1. Question

Does the form of card ADD102 hold over new fits of the head: does each of three new fit seeds cut the contact of RR, and do they lie within half the gap to the label gap?

## 2. Fits and arms

* Fits: head D-H5 of card ADD99 (the five-frame head on the fixed windows), the fit tool, settings, tokens and the 500 train clips of that card unchanged, with fit seeds 2, 3 and 4. No new encode. Dumps on the frames of the evaluation join by the dump tool of card ADD99; that tool takes the rows of its output from the present reader's dump of the same seed index, which exists for seeds 0 and 1 only, so a row template folder gives the new seeds the rows of the present reader's dump of seed 0 (only the clip, the frame and the label fields of a template row are used; the gap is the new fit's).
* Arms: the form VD5c of card ADD102 (the frames with a lead from the present reader's presence output; the gap of the head where the reader marks a lead and the head has an output; the slope of the gap the rule receives as closing input by the run rule of note N1 of card ADD102), wrapper variant 28bc31f21ecb4ccb unchanged, with the gap file of fit seed s for s in 2, 3, 4 and both model seeds: arms F2, F3, F4, six rollouts.
* References, banked and recomputed by the reader: RR (0.0954), RLc (0.0000), VD5c of card ADD102 (0.0169), VD5cx of card ADD103 (0.0200). Half the gap H is half of the point value of the contact of RR minus that of RLc (0.0477).

## 3. Names

Paired differences in the contact rate unless said, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws (the rules of cards ADD101 to ADD103). Per arm X of F2, F3, F4:

* against RR: `VISSEED10K-<X>-CUTS-CONTACT` (upper end below 0), `-MORE-CONTACT` (lower end above 0), else `-CONTACT-NOT-SEPARATED-FROM-RR`;
* against RLc: `-WITHIN-HALF-THE-GAP` (upper end at or below H), `-HALF-THE-GAP-NOT-REACHED` (lower end above H), else `-HALF-THE-GAP-NOT-DECIDED`;
* against VD5c: `-LESS-CONTACT-THAN-VD5c`, `-MORE-CONTACT-THAN-VD5c` or `-CONTACT-NOT-SEPARATED-FROM-VD5c`;
* falling behind by more than 10 m against RR: `-FALLS-BEHIND-MORE`, `-FALLS-BEHIND-LESS` or `-FALLING-BEHIND-NOT-SEPARATED`.

Card name: `VISSEED10K-HOLDS` when all three arms are CUTS-CONTACT, at least two of the three are WITHIN-HALF-THE-GAP and none is HALF-THE-GAP-NOT-REACHED; `VISSEED10K-DOES-NOT-HOLD` when at least one arm is not CUTS-CONTACT or at least one is HALF-THE-GAP-NOT-REACHED; else `VISSEED10K-NOT-DECIDED`.

Printed with no name: the contact rate of every fit of the head on record per model seed (fit seeds 0 and 1 of cards ADD102 and ADD103, fit seeds 2, 3, 4 of this card), their mean and range; unsafe rate; falling behind against RLc; values per model seed; the selection loss and the kept epoch of each new fit; the share of rule steps with a lead and of them with a non-zero closing input per rollout.

Readings committed in advance. HOLDS: the form goes to the PI and the other session as holding over five fits of the head on 500 clips. DOES-NOT-HOLD: the result of card ADD102 is reported as depending on the fit of the head, and the next lever is the head itself (more train clips, the PI's call). NOT-DECIDED goes to the PI with the numbers of all five fits.

## 4. Controls

All have to pass before any name or any value of this card is opened; logged in `TIMES_NOTE.md` before the names. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* x0 (the fits): each of the three fit records names its seed (2, 3, 4), the head H5 on arm D, the settings of card ADD99, the 500 train clips and no evaluation clip; the train list hash equals that of card ADD99's fits; the three head files differ from one another and from those of seeds 0 and 1.
* x1 (the gap files): the dumps of the three fits are on the frames of the evaluation join (141 clips in the same order, every dump row with its join row), each dump's meta names the head file of its fit, and each saved gap file equals a second routine's value on every row.
* x2 (the wrapper): the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass, and every run text names it.
* x3 (each arm is what the card says): in the rule log of every rollout, on every logged step where the rule had a lead and the head has an output, the gap the rule received equals the dumped gap of fit seed s at that clip and frame within 1e-3 m; a step with a lead and no head output equals a gap of the control run V0 of card ADD102 at the same clip and frame within 1e-3 m (none is expected); the meta of every rule log names the gap file of its arm and the closing switch; the closing input is non-zero on a positive share of the steps with a lead.
* x4: the reader recomputes contact 0.0954 for RR, 0.0000 for RLc, 0.0169 for VD5c and 0.0200 for VD5cx from the banked rollouts.
* x5: the tests of the kit pass in the lead's run on the dev box before the rollouts.
* x6: the engine's own controls of the grid read and of the strategic grid pass.

## 5. Four metric families

LONGITUDINAL (distance-keeping): contact, unsafe rate, falling behind, with the engine's pair read. LATERAL, TACTICAL: the engine's pair read against the logged drive. STRATEGIC: read inside the reader under its own engine controls for every arm against RR, RLc and VD5c.

## 6. Limits

T1-kin only, no re-perception; the lead moves as logged whatever the ego does; the heads read crops of logged frames; two model seeds of one line; the fits vary the fit seed and not the 500 train clips; contact differences rest on few windows, so a name can turn on two windows; designed after the reads of cards ADD102 and ADD103.
