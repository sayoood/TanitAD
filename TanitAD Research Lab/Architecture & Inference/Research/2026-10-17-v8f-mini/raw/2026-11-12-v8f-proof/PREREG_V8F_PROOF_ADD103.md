# PREREG_V8F_PROOF_ADD103 (VISREP10K): the form with no label at inference, a second fit seed and what carries it

Fork v8proof, 2026-11-21. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `stack/scripts/v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, 1,271 windows per model seed. Evidence class: MEASURED. Designed after the read of card ADD102 (VISRULE10K-WITHIN-HALF-THE-GAP: the reader-fed form with the gap of the fixed-window head and the slope of that gap as closing input has a contact rate of 0.0169 against 0.0954 for the reader-fed form RR; the same form without the closing input has 0.0769). The levels of card ADD102 were seen before this card was written; no pair of this card has been computed. Frozen with its hash file before any tool, file or rollout of it exists. A study of a changed input of the rule; adoption of any form is the PI's and the other session's decision.

## 1. Questions

1. Does the result of card ADD102 hold with the other fit seed of the head (a replicate at the level of the head fit)?
2. What carries it: the gap of the fixed-window head, or the slope closing input alone on the present reader's own gap?
3. Is the head on five frames needed, or does the head on one frame do the same?

## 2. Arms

The reader-fed path of the wrapper variant of card ADD102 (28bc31f21ecb4ccb, unchanged, with its pod check record of that card), the arguments of the banked RR rollouts. Per model seed k:

* VD5cx: as VD5c of card ADD102 (the gap of head D-H5 where the reader marks a lead and the head has an output, the slope of the gap the rule receives as closing input by the run rule of note N1 of card ADD102), with the gap file of the other fit seed: model seed k reads fit seed 1 - k. The gap files are those of card ADD102, unchanged.
* VRc: no gap file, the closing switch only: the rule receives the present reader's own gap after its map and minimum, as in RR, and the slope of that gap as closing input by the same run rule.
* VD1c: as VD5c with the gap of the head on one frame on the fixed windows (D-H1 of card ADD99; model seed k reads fit seed k); new gap files built from its dumps by the tool of card ADD102 with the dump folder as its only change.
* Six rollouts. References, banked and recomputed by the reader: RR (0.0954), RLc (0.0000), KD5 of card ADD101 (0.0169), VD5c (0.0169) and VD5 (0.0769) of card ADD102. Half the gap H is half of the point value of the contact of RR minus that of RLc (0.0477).

## 3. Names

Paired differences in the contact rate unless said, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws (the rules of cards ADD101 and ADD102). Per arm X of VD5cx, VRc, VD1c:

* against RR: `VISREP10K-<X>-CUTS-CONTACT` (upper end below 0), `-MORE-CONTACT` (lower end above 0), else `-CONTACT-NOT-SEPARATED-FROM-RR`;
* against RLc: `-WITHIN-HALF-THE-GAP` (upper end at or below H), `-HALF-THE-GAP-NOT-REACHED` (lower end above H), else `-HALF-THE-GAP-NOT-DECIDED`;
* against VD5c: `-LESS-CONTACT-THAN-VD5c` (upper end below 0), `-MORE-CONTACT-THAN-VD5c` (lower end above 0), else `-CONTACT-NOT-SEPARATED-FROM-VD5c`;
* falling behind by more than 10 m against RR: `-FALLS-BEHIND-MORE`, `-FALLS-BEHIND-LESS` or `-FALLING-BEHIND-NOT-SEPARATED`.

Card names:

* the replicate, from VD5cx: `VISREP10K-REPLICATED` when VD5cx is CUTS-CONTACT and WITHIN-HALF-THE-GAP; `VISREP10K-NOT-REPLICATED` when VD5cx is HALF-THE-GAP-NOT-REACHED, or is not CUTS-CONTACT; else `VISREP10K-REPLICATION-NOT-DECIDED`;
* the gap, from the pair VD5c minus VRc: `VISREP10K-CROP-GAP-CUTS-CONTACT` (upper end below 0), `VISREP10K-CROP-GAP-MORE-CONTACT` (lower end above 0), else `VISREP10K-CROP-GAP-NOT-SEPARATED-FROM-READER-GAP`;
* the slope, from the pair VD5c minus VD5 (both banked; their levels were seen, the pair was not computed): `VISREP10K-SLOPE-CUTS-CONTACT` (upper end below 0), `VISREP10K-SLOPE-MORE-CONTACT` (lower end above 0), else `VISREP10K-SLOPE-NOT-SEPARATED`.

Printed with no name: unsafe rate, values per model seed, falling behind against RLc, the share of rule steps with a lead and the share of them with a non-zero closing input per rollout, the pairs VD5cx, VRc and VD1c against KD5.

Readings committed in advance. REPLICATED: the form of card ADD102 goes to the PI and the other session with a second fit seed behind it; its price in falling behind against the label gap is the next lever. NOT-REPLICATED: the result of card ADD102 is reported as resting on one head fit per model seed and not confirmed; the next card fits further head seeds on the tokens that exist (no encode) before any form goes forward. REPLICATION-NOT-DECIDED goes to the PI with the numbers. VRc WITHIN-HALF-THE-GAP with CROP-GAP-NOT-SEPARATED-FROM-READER-GAP says the slope closing input on the present reader's own gap is enough, a form that needs no new encode; that form then goes to the PI and the other session beside the crop form. CROP-GAP-CUTS-CONTACT says the gap of the fixed-window head is needed. VD1c WITHIN-HALF-THE-GAP and CONTACT-NOT-SEPARATED-FROM-VD5c says one frame per look is enough for the head.

## 4. Controls

All have to pass before any name or any value of this card is opened; logged in `TIMES_NOTE.md` before the names. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* w0 (the gap files of VD1c): the dumps of D-H1 are on the frames of the evaluation join (141 clips in the same order, every dump row with its join row), and the saved gap file equals a second routine's value on every row. The gap files of VD5cx are those of card ADD102, their hashes equal to that card's record.
* w1 (the wrapper): the wrapper file is 28bc31f21ecb4ccb and the pod check record of card ADD102 names it with a pass.
* w2 (each arm is what the card says): in the rule log of every rollout, on every logged step where the rule had a lead: in VD5cx the gap the rule received equals the dumped gap of D-H5 of fit seed 1 - k at that clip and frame within 1e-3 m; in VD1c that of D-H1 of fit seed k; in VRc it equals a gap of the control run V0 of card ADD102 logged at the same clip and frame within 1e-3 m (V0 logs a frame more than once on a few frames, with values that differ in the float noise of the reader; any of them counts). A step with a lead and no head output is held to the same test against V0 (none is expected). The meta of every rule log names the gap file and the closing switch of its arm. In every arm the closing input is non-zero on a positive share of the steps with a lead.
* w3: the reader recomputes contact 0.0954 for RR, 0.0000 for RLc, 0.0169 for KD5, 0.0169 for VD5c and 0.0769 for VD5 from the banked rollouts.
* w4: the tests of the kit pass in the lead's run on the dev box before the rollouts.
* w5: the engine's own controls of the grid read and of the strategic grid pass.

## 5. Four metric families

LONGITUDINAL (distance-keeping): contact, unsafe rate, falling behind, with the engine's pair read. LATERAL, TACTICAL: the engine's pair read against the logged drive. STRATEGIC: read inside the reader under its own engine controls for every arm against RR, RLc and VD5c.

## 6. Limits

T1-kin only, no re-perception; the lead moves as logged whatever the ego does; the heads read crops of logged frames, so the crop does not move with the rolled-out ego; two model seeds of one line; two fit seeds of one selection of 500 train clips, so the replicate varies the fit seed and not the train clips; contact differences rest on few windows; the slope pair is designed after its two levels were seen.
