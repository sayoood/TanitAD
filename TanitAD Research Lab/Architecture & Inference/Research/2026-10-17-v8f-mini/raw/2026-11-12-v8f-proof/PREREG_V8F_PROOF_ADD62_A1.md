# Amendment 1 to card PREREG_V8F_PROOF_ADD62 (RULEFAST10K): the second clause of control k4 is reported, not a gate

Written by the coordinator fork before any run of the card and before its queue was started (time of the freeze: the line of this amendment in `TIMES_NOTE.md`; hash in the sidecar file). No rollout of the card exists; no number of the card has been read.

## 1. What the tool check showed (MEASURED on synthetic plans, no model, no data of the card; `integ10k/read_rulefast/CHECK_RULE_EVERY.txt`, sha256 c9c86156ad1e8a08, tool `integ10k/check_rule_every.py`, sha256 bf85ea12dcfa9c64)

* The mechanism does what section 2 of the card says: the later look at step 5 reads what a real tick at that frame reads (label form and reader form, largest difference 0.0); no later look moves a step farther along the path (largest excess 0.0); with a lead that brakes after the tick the car ends the block farther from the lead in 48 of 48 synthetic windows (median 1.44 m); with a lead holding its speed the plans are equal; the planted fault (the later look fed the tick's own inputs) fails the direction check; `--rule-every 10` is the rule at the tick only.
* The second clause of control k4 ("with the rule every 2 steps the plan is never ahead of the plan with the rule every 5 steps at the first tick's block") is not a property of that mechanism. On the synthetic windows with a braking lead the plan with the rule every 2 steps is ahead at steps 6 to 9 in 38 of 48 windows (by up to 0.48 m) and behind at step 10: a rule that looks earlier brakes earlier and more gently, so in the middle of the block it has lost less way than a rule that brakes late and hard. The clause was a wrong expectation of the card's author, not a fault of the tool.

## 2. The change

* Control k4 is its first clause alone: in every later look of every run, no step of the remaining plan is farther along the path than in the plan before the look (largest excess at most 1e-9 m). This is the cap-only property and it stays a gate.
* The comparison of the plans with the rule every 2 and every 5 steps at the first tick's block is reported without a bar (largest excess per step, windows ahead at some step, windows ahead at step 10) and gates nothing.
* Nothing else of the card changes: arms, reads, names, the other controls and what follows stay as frozen. The reader `integ10k/read_rulefast10k.py` carries this change (sha256 in `TIMES_NOTE.md`).

## 3. Why this is admissible

The clause is dropped as a gate before any data of the card exists, on a measurement that involves no model and no evaluation window, and for a stated reason. Keeping it would withhold every name of the card on a property the mechanism does not have.
