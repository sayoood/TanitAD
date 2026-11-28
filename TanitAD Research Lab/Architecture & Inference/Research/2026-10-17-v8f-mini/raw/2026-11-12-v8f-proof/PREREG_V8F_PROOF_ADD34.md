# PREREG_V8F_PROOF_ADD34 : RULEOWN, the distance rule applied to the model's own plan as a training term

Card of the v8F proof programme. Frozen before the trainer change is built, before any gate, step or read of its arms. Tier of every read: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react); never driving performance. STRATEGIC: PAUSED (PI item 25).

## 1. Why (MEASURED, with the artifact)

* The label-fed distance rule of card ADD26 (note N1: keep 3.0 m + 1.2 s, brake at most 6.0 m/s2, never move a plan ahead, cap-only), checking the plan at inference, cuts the unsafe rate U on the fine-tuned base by -0.0676 [-0.1008, -0.0362] and takes the contact rate to 0.0000 (row H-V8F-PROOF-61, `ruletrain/read/hw_rule`).
* Putting the rule into the training target (the logged future replaced by its rule-capped form, card ADD32) is `RULE-TRAINED-FLAT` at 2,000 steps: GRT minus GFT +0.0000 [-0.0187, +0.0164]; the rule at inference still changes the plan in 0.233 / 0.238 of the lead windows at the first tick (0.240 / 0.246 without the relabel).
* The relabel changed 9 percent of the training samples: those where the logged driver came closer than the rule allows. The rule at inference changes the model's plan in 24 percent of the lead windows: those where the model's own plan runs into the rule. These are not the same events. Hypothesis of this card: the unsafe steps of T1-kin come from the model's own plan, so the training signal has to act on the model's own plan.

## 2. The lever (to be built as a flag; default off must be bit for bit the trainer of card ADD32)

`--w-rule-own W` (default 0). For every training sample with a label lead (the mask and the inputs of `--rule-relabel`: label gap, label closing speed, logged speed, lead speed max(0, speed + closing); a sample with a non-finite raw speed is left out and not counted):

* cap = `dist_rule.distance_rule(plan, v0, present, gap, gap, v_L, m=10, floor_lifts=False)` with plan = the model's own predicted trajectory of that sample, detached (no gradient through the rule);
* the term is W times the mean over these samples of the squared distance between the predicted trajectory and its cap, on every trajectory output that is trained against the logged future and that the T1-kin tool rolls out;
* where the predicted plan already keeps the rule, cap equals the plan and the term and its gradient are exactly 0. The term never moves a plan ahead (the rule is cap-only).
* Inference is unchanged: no rule, no lead input. The model has to take the lead from its own state.

Stage 2 of note `PREREG_V8F_PROOF_ADD32_N1.md` is built in the same change: `--rule-relabel-weight W` (default 1), the weight of the samples whose target the relabel changed in the trajectory loss.

## 3. Arms

Fine-tunes from the banked checkpoints G_s0 and G_s1, `--steps 2000 --lr 1e-4 --warmup 100`, the argv of GFT plus the flag; each through the gate of its chain. Control and floor: GFT and GFTb of card ADD32 (same bases, same steps; the new tree with the flags off is bit for bit the tree they were trained on, control o1).

* GOW: `--w-rule-own 1.0`, both bases. The named arm.
* GOW4: `--w-rule-own 4.0`, both bases. Dose arm, reported, names nothing.
* GOWRL, GOW4RL: the label-fed rule at inference on these four checkpoints (first-tick share; does the checked rule still add).
* Cost (ESTIMATED from the arms of card ADD32 at 7.6 min): four fine-tunes and four rollouts, about 0.8 L40-h. Stage 2 of note ADD32_N1 (two fine-tunes of 6,000 steps with the weight 4) about 0.9 L40-h.

## 4. Read

`headway/headway_read.py` (U primary, contact, falling behind, gap bias) and `spatial/q3/q3_read.py` (four families), paired on windows, two-level bootstrap (bases, then episodes), B 2,000, rng 0; 642 lead windows of 56 episodes for the distance cells, 2,399 windows of 141 episodes for the cost cells.

* Lever: GOW minus GFT. Floor: the larger absolute difference in U between GFTb and GFT over the two bases (0.0036, row H-V8F-PROOF-61).
* Share reached = (U of GFT minus U of GOW) / (U of GFT minus U of GFTRL), per base.
* First tick: the share of lead windows whose plan the label-fed rule changes at the first tick, from the rule logs of GOWRL (GFTRL: 0.2404 / 0.2455).

## 5. Controls (a failed control is reported first and the outcomes it carries are not named)

* o1: both flags at their defaults, new tree against the tree of card ADD32, same seed, 100 steps on the CPU: every logged field and the final weights equal (max abs difference 0.0).
* o2: on a batch, the cap equals `dist_rule.distance_rule` called by hand on the detached prediction (exact); the rule applied twice equals the rule applied once (max abs difference at most 1e-5); samples without a label lead and samples with a non-finite raw speed contribute exactly 0.
* o3: reach, in three parts. A plan that runs into the lead gives a term above 0 and a non-zero gradient on the trajectory output; a plan that already keeps the rule gives exactly 0 for both; a batch with no lead gives exactly 0. On a real batch the term is partial (some lead samples contribute, some do not; both counts printed). The line that adds the term is deleted in a mutation test and the test must fail.
* o4: the gate. Declared flag equals the built value; the real-data live step reports the term (finite), the count of lead samples and the count of samples with a term above 0; a wrong expectation makes the selfcheck fail; the train log carries the three per logged step.
* o5: the eval tool loads a GOW checkpoint with the config of its base (strict load); the eval tool is the pinned one (sha256).
* o6: the reader sees a known effect (GFTRL against GFT named as cutting U) and does not name the replicate GFTb different from GFT (both read again in the same call).
* o7: the reader controls of every read.
* For `--rule-relabel-weight`: w1 default 1 is bit for bit (part of o1); w2 on a batch the weighted loss equals the hand computation and only changed samples carry the weight; w3 mutation test; w4 gate entry (declared equals built, wrong expectation refused).

## 6. Outcomes (both bases needed; named for GOW)

* `OWN-RULE-CUTS-UNSAFE`: U(GOW) minus U(GFT) is below 0 in both bases, the two-level interval excludes 0, and the mean difference is larger in size than the floor. `OWN-RULE-WORSE`: the same above 0. `OWN-RULE-FLAT`: otherwise.
* `OWN-RULE-REACHES-ORACLE`: the share reached is at least 0.5 in both bases (named only with `OWN-RULE-CUTS-UNSAFE`).
* Cost of GOW against GFT by the rule of card ADD11: `OWN-RULE-NO-COST` if no cell of the four families is named worse, else `OWN-RULE-COSTS` with the cells. Reported beside in every case: position error over 6 s, along-track error, share of windows more than 10 m behind, gap bias, contact rate; the same cells for GOW4; GOWRL against GFTRL.

A named outcome says the lever moved U beyond a second fine-tune on these two checkpoints; it is not yet a statement about training runs in general (training variance over bases is the next test, section 7).

## 7. What follows (committed before the data)

* `OWN-RULE-CUTS-UNSAFE`: the same arms on G_s2 to G_s5 (six bases); an arm trained from the start with the term; the vision-fed checked rule of cards ADD30 and ADD31 on top (do the trained rule and the checked rule add up); with `OWN-RULE-COSTS` the two weights give the trade.
* `OWN-RULE-FLAT` with the first-tick share fallen: the model keeps the rule on logged states and loses it on its own states; next is the term on own-state samples (the own-state bank of the trainer).
* `OWN-RULE-FLAT` with the first-tick share unchanged at both weights: the plan head does not use the lead that is in the state; next is the lead read as an explicit input of the plan head (the head outputs of arm LS fed to the planner, vision only, no ego state), and the finer lead read stays the decision of the PI named in row H-V8F-PROOF-56.
* `OWN-RULE-WORSE` or any cost cell: reported as written; the arm is not promoted; the next lever of its U outcome still runs.

## 8. Limits

Two bases with one replicate of the control; 2,000 steps; the term uses the label lead at training (labels may use privileged signals, inference is vision only); T1-kin with logged frames (the model cannot see its own drift in the image) and a lead that does not react, so braking for a lead is never rewarded by the lead; one corpus.
