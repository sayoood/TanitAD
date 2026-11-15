# PREREG_V8F_PROOF_ADD33: NOCLOSE. The vision-fed distance rule without the head's closing-speed read (the lead is taken to move at the logged ego speed). Zero training.

Frozen before any number of this card is computed and before any rollout of cards ADD30 and ADD31 has been read (two rollouts of card ADD30 are done on the pod, none is pulled or read). Addendum to `PREREG_V8F_PROOF.md` and to cards ADD26 (note N1), ADD29, ADD30, ADD31. Tier T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react; a diagnostic, never driving performance). Evidence class: MEASURED once run. LONGITUDINAL is the family measured; LATERAL and TACTICAL are read as cost cells; STRATEGIC: PAUSED (PI item 25).

## 1. Why (MEASURED today, card ADD29, 642 lead windows of 56 episodes, controls passed)

With presence and gap from the labels, the rule fed with the head's closing speed is not better on the unsafe rate U than the rule fed with a closing speed of zero: LS -0.0009 [-0.0180, +0.0139] (seeds -0.0089 / +0.0071), LSD +0.0160 [+0.0043, +0.0299] (seeds +0.0196 / +0.0125; the head's read is worse there). This comparison was made after the numbers of card ADD29 were seen and names nothing. If it holds for the vision-fed rule too, the weakest of the three reads can be left out of the rule.

## 2. Arms (checkpoints LS_s0, LS_s1, LSD_s0, LSD_s1)

| code | presence | gap | closing speed |
|---|---|---|---|
| RZ | head, threshold 0.5 | head, as read | 0 |
| QZ | head, threshold P* of card ADD30 | head through the map of card ADD31 | 0 |

Eight rollouts, after the queue of card ADD31, one at a time, never as a fourth GPU job, memory check before each. Tools: `v8f_t1_eval_rule_margin.py` at threshold 0.5 and margin 0 with the mix `head head zero` (RZ); `v8f_t1_eval_rule_qcal.py` with the mix `head head zero` (QZ).

## 3. Read

`headway_read.py`, paired on windows, two-level bootstrap (seeds, then episodes), B 2,000, rng 0: RZ minus the head-fed rule of card ADD26 (`RH`), QZ minus the arm QC of card ADD31; each also against no rule and against the label-fed rule (share reached as card ADD30). `q3_read.py` four families against no rule, and RZ against RH, QZ against QC for the cost difference.

## 4. Controls

* z1: no plan is moved ahead in any rule log of this card (lift 0).
* z2: in every rule log of this card the closing speed the rule used is exactly 0 at every tick, and the gap the rule used at the first tick equals that of the partner run (RH for RZ, QC for QZ) exactly.
* z3: the reader controls of every read; control g5 of card ADD30 carries the comparison outcomes of this card too.

## 5. Outcomes (per arm type, both seeds needed; one for each of RZ against RH and QZ against QC)

* `ZERO-CLOSING-BETTER`: U of the zero arm minus U of its partner is below 0 in both seeds with a two-level interval excluding 0. `ZERO-CLOSING-WORSE`: above 0 likewise. `ZERO-CLOSING-EQUAL`: otherwise.
* Reported: contact, falling behind, gap bias, the four families and the position error over 6 s, zero arm minus partner; the share of the label-fed fall reached by QZ.

## 6. What follows (committed before the data)

* `ZERO-CLOSING-BETTER` or `-EQUAL` for both arm types: the vision-fed rule is carried forward without the head's closing-speed output (two reads, presence and gap), and the closing-speed question goes to the PI only as part of the finer lead read (row H-V8F-PROOF-56), with the size card ADD29 measured for a perfect closing speed.
* `ZERO-CLOSING-WORSE` for an arm type: the head's closing speed helps the vision-fed rule there although it does not help beside label presence and gap; the three-read form stays, and the interaction is reported.
