# PREREG_V8F_PROOF_ADD105 (VISENS10K): the form with no label at inference fed by the mean gap of five fits of the head

Fork v8proof, 2026-11-21. Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions, heading bound 0.2 1/m, executor of record `stack/scripts/v8f_t1_eval_hb.py`), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, 1,271 windows per model seed. Evidence class: MEASURED. Designed after the read of card ADD103 (the name within half the gap holds for one fit of the head and is open for the other by two windows of 650) and while the three fits of card ADD104 run: no fit record, dump or rollout of card ADD104 has been read (one look at the tail of one fit log printed nothing). Frozen with its hash file before any tool, file or rollout of it exists. A study of a changed input of the rule; adoption of any form is the PI's and the other session's decision.

## 1. Question

A single fit of the head carries the noise of its fit. Does the form of card ADD102, fed by the mean gap of the five fits of the head on record, cut the contact of RR and lie within half the gap to the label gap?

## 2. Arm

* Gap file: on every frame of the evaluation join where all five fits of head D-H5 have an output (fit seeds 0 and 1 of card ADD99, fit seeds 2, 3 and 4 of card ADD104; the dumps as they are, no new fit, no new encode), the arithmetic mean of the five dumped gaps; no output on any other frame. No map, no clip, no minimum, no weight.
* Arm E5c: the form VD5c of card ADD102 (the frames with a lead from the present reader's presence output; the gap of the gap file where the reader marks a lead and the file has an output; the slope of the gap the rule receives as closing input by the run rule of note N1 of card ADD102), wrapper variant 28bc31f21ecb4ccb unchanged, the same gap file for both model seeds. Two rollouts.
* References, banked and recomputed by the reader: RR (0.0954), RLc (0.0000), VD5c of card ADD102 (0.0169). Half the gap H is half of the point value of the contact of RR minus that of RLc (0.0477).

## 3. Names

Paired differences in the contact rate unless said, two-level bootstrap over model seeds and evaluation episodes, 2,000 draws (the rules of cards ADD101 to ADD104).

* against RR: `VISENS10K-E5c-CUTS-CONTACT` (upper end below 0), `-MORE-CONTACT` (lower end above 0), else `-CONTACT-NOT-SEPARATED-FROM-RR`;
* against RLc: `-WITHIN-HALF-THE-GAP` (upper end at or below H), `-HALF-THE-GAP-NOT-REACHED` (lower end above H), else `-HALF-THE-GAP-NOT-DECIDED`;
* against VD5c: `-LESS-CONTACT-THAN-VD5c`, `-MORE-CONTACT-THAN-VD5c` or `-CONTACT-NOT-SEPARATED-FROM-VD5c`;
* falling behind by more than 10 m against RR: `-FALLS-BEHIND-MORE`, `-FALLS-BEHIND-LESS` or `-FALLING-BEHIND-NOT-SEPARATED`.

Card name, the first that applies: `VISENS10K-DOES-NOT-CUT` when the arm is not CUTS-CONTACT; `VISENS10K-WITHIN-HALF-THE-GAP` when it is WITHIN-HALF-THE-GAP; `VISENS10K-HALF-THE-GAP-NOT-REACHED` when it is HALF-THE-GAP-NOT-REACHED; else `VISENS10K-HALF-THE-GAP-NOT-DECIDED`.

Printed with no name: contact, unsafe rate and falling behind per model seed and pooled; falling behind against RLc; the pooled contact of every single-fit arm on record beside it (VD5c, VD5cx of card ADD103 and, when card ADD104 has been read with all controls passed, F2, F3, F4; else those three are left out and the line says so); at tier T0 (a probe on logged frames, a diagnostic of a reader), on the frames with a lead in the label and an output of all five fits: the mean absolute gap error of the mean gap and of each single fit; the share of rule steps with a lead and of them with a non-zero closing input per rollout.

Readings committed in advance. WITHIN-HALF-THE-GAP: the mean of five fits is the form of the gap input that goes to the PI and the other session, beside the single-fit record of card ADD104. HALF-THE-GAP-NOT-DECIDED or NOT-REACHED: averaging the fits does not settle it; what limits the form is the head as fit on 500 clips, and the next lever is more train clips (the PI's call). DOES-NOT-CUT: reported as a failure of the mean gap; no changed form under this card.

## 4. Controls

All have to pass before any name or any value of this card is opened; logged in `TIMES_NOTE.md` before the names. This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

* e0 (the gap file): the dumps of fit seeds 0 and 1 have the hashes that the gap-file record of card ADD102 names for them; the dumps of fit seeds 2, 3 and 4 have the hashes that the gap-file record of card ADD104 names for them, and that record reads x0 PASS and x1 PASS; the five dumps are on the same rows (clip and frame); the saved gap equals a second routine's mean on every row within 1e-6 m; a row has an output exactly when all five dumps have one.
* e1 (the wrapper): the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass, and both run texts name it.
* e2 (the arm is what the card says): in the rule log of both rollouts, on every logged step where the rule had a lead and the gap file has an output, the gap the rule received equals the gap of the file at that clip and frame within 1e-3 m; a step with a lead and no output of the file equals a gap of the control run V0 of card ADD102 at the same clip and frame within 1e-3 m; the meta of both rule logs names the gap file and the closing switch; the closing input is non-zero on a positive share of the steps with a lead.
* e3: the reader recomputes contact 0.0954 for RR, 0.0000 for RLc and 0.0169 for VD5c from the banked rollouts.
* e4: the tests of the kit pass in the lead's run on the dev box before the rollouts.
* e5: the engine's own controls of the grid read and of the strategic grid pass.

## 5. Four metric families

LONGITUDINAL (distance-keeping): contact, unsafe rate, falling behind, with the engine's pair read. LATERAL, TACTICAL: the engine's pair read against the logged drive. STRATEGIC: read inside the reader under its own engine controls for the arm against RR, RLc and VD5c.

## 6. Limits

T1-kin only, no re-perception; the lead moves as logged whatever the ego does; the heads read crops of logged frames; two model seeds of one line; five fits of one head on one selection of 500 train clips; the mean of five heads costs five head passes at inference (the encoder pass is shared); contact differences rest on few windows, so a name can turn on two windows; the arm includes the two fits whose single-fit results were seen (cards ADD102 and ADD103).
