# Card PREREG_V8F_PROOF_ADD44 (INTEG-PRE): the confirmed levers together on the arm of record, four bases, no model training

Card of the v8F proof study, written by the coordinator fork. Frozen before any tool of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). At the freeze the reader of note ADD40_N7 (planner with the gap head) had not run; no number of it was seen. Tier of every rate and family below: T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.

## 1. Why (MEASURED, with the artifact)

* Row H-V8F-PROOF-88: the consistency term at weight 0.0001 (arm RLAm) turns the response of the imagined gap and keeps the level, confirmed on two new bases; the families read no cost.
* Row H-V8F-PROOF-90: the distance rule fed by the trained 0.20-quantile gap head reaches the label-fed rule on the unsafe rate on two lead-supervised arm types (two training runs each), with a cost against no rule and a contact rate above the label-fed rule.
* Rows 94 and 97: the read of the lead is the first limit, and the finer token grid does not lift it at this data size.
* No row reads these levers together on one arm. The joint plan trains an integrated candidate at 10k clips (step 3); this card is its pre-test on the present corpus with checkpoints that exist. It costs no model training.

## 2. Arms and inference forms

* Candidate C: `RLAm` at bases 0, 1, 2, 3 (6,000-step fine-tune with the consistency weight 0.0001 from the RL base of that seed). Control K: `RLFT` at bases 0, 1, 2, 3 (the same fine-tune without the term). Eight checkpoints, all banked on the pod.
* Per checkpoint three rollouts with the banked rule tool, unchanged: `R0` (no rule), `RLb` (the rule fed by labels), `RQ` (the rule with `g_used = min(g_head, q_head)`, `q_head` = the quantile head of card ADD42 fitted for this checkpoint on the states of all training clips in the fit form of note ADD42_N2; a head whose coverage on its check clips lies outside [0.15, 0.25] is not read and its base counts as missing).
* Planner form, conditional and fixed here: if and only if note ADD40_N7 reads `QPLAN-CUTS-MORE-CONFIRMED` together with `QPLAN-NO-COST` on both pairs, the source `cand-cvq` is run on C and K at the four bases with the same heads and reported beside the rule (it names nothing in this card). In every other case no planner run belongs to this card.
* Budget (ESTIMATED; the build states the measured volume of the state dumps before they are written): eight state extracts, eight head fits on the CPU, 24 rollouts and 2 control rollouts, one GPU job at a time behind the three-job limit and the memory guard; under 4 pod-h.

## 3. Reads (paired on windows; 2,399 windows of 141 episodes; U = model-induced unsafe rate of the headway read on the 642 lead windows of 56 episodes; two-level bootstrap over the four bases, then episodes, B 2,000; per-base values and the count of bases with the sign of the pooled value are printed beside every difference)

* D1 = U(C with RQ) minus U(K with R0): the small integrated candidate against the control without any lever. Primary.
* D2 = U(C with RQ) minus U(C with R0): the part of the rule on the arm of record. Share reached = D2 divided by (U(C with RLb) minus U(C with R0)), per base.
* D3 = U(C with RQ) minus U(C with RLb): against the label-fed rule. Contact rate of the same pair.
* D4 = U(C with RQ) minus U(K with RQ): what the consistency term changes under the rule.
* D5 = U(C with R0) minus U(K with R0): the consistency term alone (known from row 88 at the family level; reported).
* Families: C with RQ against K with R0, and against C with R0, by the rule of card ADD12 Q3 with the same-sign rule of ADD13; the 6 s position error difference in metres with its interval. The four metric families are reported for both pairs. STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).

## 4. Names

* `INTEG-PRE-CUTS-UNSAFE`: D1 is below 0 at at least three of the four bases and its two-level interval excludes 0. Otherwise `INTEG-PRE-NO-CUT`.
* `INTEG-PRE-REACHES-LABEL-RULE`: the share reached is at least 0.5 at at least three of the four bases and the two-level interval of D2 lies below 0. Otherwise not named.
* `INTEG-PRE-LEVERS-COMPATIBLE`: the two-level interval of D4 includes 0 or lies below 0. `INTEG-PRE-LEVERS-INTERFERE`: it lies above 0.
* Price of the candidate against the control without levers: `INTEG-PRE-NO-COST` or `INTEG-PRE-COSTS-<family>` (C with RQ against K with R0).
* A base that is missing (head outside the band, a failed control of that base) is reported; with fewer than three bases read nothing is named.

## 5. Controls (all must pass; else nothing is named)

* c1 (heads): for every checkpoint the sha256 of the head file equals its fit record, the fit form is the one of note ADD42_N2, and the check-clip coverage is in the band.
* c2 (known positive): the label-fed rule cuts U against no rule on C and on K, pooled interval excluding 0, and at every base the sign is below 0.
* c3 (identity of the tool on this arm type): on C base 0, with `q_head = g_head`, the dump of `RQ` equals the dump of the head-fed rule of the same checkpoint bit for bit.
* c4 (counts): 2,399 windows, 642 lead windows of 56 episodes in every dump; the same window set in all 24 rollouts.
* c5: the controls h1 to h6 of the headway read and q1 to q5 of the family read.
* c6 (same model as banked): the `R0` dump of every checkpoint equals the banked dump of the run without a planner of the same checkpoint (rows 91 and 95, source `IPo`) on the position outputs, or, if the two tools write different files, the unsafe rate U of `R0` equals the banked U of `IPo` to 1e-9.
* Tool checks before any run: the changed queue passes its tests and a counted mutation list (a wrong checkpoint for a head, a head of another base, the rule form swapped) goes red; a dry run prints every command.

## 6. What follows (committed before the data)

* `INTEG-PRE-CUTS-UNSAFE` with `-LEVERS-COMPATIBLE`: the lever set of the cover note (lead head, consistency term at 0.0001, rule with the trained gap head) goes into the card of the integrated candidate with this measured small-scale effect and its price; the 10k candidate is read with the same pairs.
* `INTEG-PRE-NO-CUT`: reported as written. The cut of row 90 does not carry over to the arm of record; D2 and the share reached say whether the rule or the head loses it; the next arm is the same read with the hand-calibrated map of card ADD31 on C, which needs one calibration on training clips.
* `INTEG-PRE-LEVERS-INTERFERE`: reported as written; the rule is offered to the integrated tree as an inference option only, and the pair K with RQ is the form carried forward until an arm removes the interference.
* Any `INTEG-PRE-COSTS-<family>`: reported beside the cut with its size in metres; the price is set against the price of row 90.

## 7. Limits

One corpus of 2,376 training episodes; four bases are four training runs per arm type, which bounds run-to-run variance for these arm types only; T1-kin (the lead does not react, no re-perception); the heads are fitted per checkpoint on training clips and read on evaluation episodes; the planner and the rule are deterministic, so there is no inference-seed spread; a pre-test on fine-tunes of 6,000 steps does not stand for a model trained from the start with every lever.
