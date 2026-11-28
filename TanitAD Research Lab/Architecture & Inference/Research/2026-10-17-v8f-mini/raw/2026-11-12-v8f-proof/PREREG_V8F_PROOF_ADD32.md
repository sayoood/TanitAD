# PREREG_V8F_PROOF_ADD32: RULETRAIN. The distance rule in the training loop: the planner's trajectory target is passed through the label-fed rule. Does the model then keep its distance from vision alone, with no rule at inference?

Frozen before any number of this card is computed and before its trainer change is built. Addendum to `PREREG_V8F_PROOF.md` and to card ADD26 with its note N1 (which commits "then the training card with the rule in the loop"). Tier of every driving read: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react; a diagnostic, never driving performance). Evidence class of every result: MEASURED once run. LONGITUDINAL is the family measured (distance keeping and the four-family cost read); LATERAL and TACTICAL are read as cost cells; STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it

* MEASURED (row H-V8F-PROOF-54): the label-fed rule on the plan cuts the model-induced unsafe rate U of arm G from 0.1296 to 0.0498 on six seeds; lead supervision alone leaves U equal; the vision-fed rule reaches about half of the label-fed fall on LS and about a fifth on LSD.
* MEASURED (pod table of 2026-11-15 from the rule logs of card ADD26; a description): at the FIRST tick of a window, where the model stands on the logged state and has not drifted, the label-fed rule already changes the model's plan in 7.7 to 8.1 percent of all windows, which is 24 percent of the windows with a lead (G_s0, G_s1, LS_s0, LSD_s0); at the third tick in 8.7 to 9.5 percent. The model's own state lies on average 0.3 m behind the logged one at the second tick and 2.4 to 3.7 m behind at the third. So the plans break the rule on logged states already, and a training signal on logged states exists.
* The trainer already holds a hinge on the minimum time gap to the lead (`consistency_penalties`, LEAD, against the driver's own minimum time gap, weight 0.5 x 0.5); the numbers above are measured with it on. It teaches "not closer than the driver was", not the rule.
* The inputs at inference are camera frames and the measured ego motion with its history. The lead labels are used for training targets only (labels may use privileged signals; inference is vision only: ruling of the PI of 2026-08-03).
* Seen before the freeze: everything above and the results of cards ADD24 to ADD28. Not seen: any number of cards ADD29, ADD30, ADD31 on distance or cost, and nothing of this card.

## 2. The lever

Trainer flag `--rule-relabel on` (default off = every earlier arm, bit for bit). For a training sample whose current frame has a label lead (present, finite gap, finite closing-speed label), the 30-knot trajectory target of the planner output that the T1 rollout executes (`traj30`) is replaced by

`distance_rule(target, v0, present, gap, gap, v_L)` with `v_L = max(0, v_logged + closing)`,

the function of `distrule/dist_rule.py` of card ADD26 (one implementation, its sha256 pinned in the arms file), with the constants of note N1 (keep 3.0 m plus 1.2 s, brake at most 6.0 m/s2, never move a plan ahead, cap-only). Samples without a label lead keep their target. No module, no input and no inference path changes: a checkpoint of this lever loads in the pinned eval tool as its base arm does.

The builder lists every loss term whose target comes from the logged future of the sample and states for each whether it is relabelled; the card requires the relabel for the target of `traj30` (and of the same output before its post-map, if that has its own term) and leaves the world-model terms, the 6 s tactical target and the lead head unchanged. Any other decision goes back to the lead before training.

## 3. Arms (fine-tune from banked checkpoints; nothing is trained from the start)

Bases: G_s0 and G_s1 (grid state 640, card ADD12). From each base, 2,000 steps with the base's argv, batch 64, peak learning rate 1e-4 (warm-up 100 steps, cosine to a tenth):

| arm | targets | fine-tune seed |
|---|---|---|
| GFT_s0, GFT_s1 | unchanged (control) | the base seed |
| GFTb_s0, GFTb_s1 | unchanged (replicate of the control) | the base seed plus 100 |
| GRT_s0, GRT_s1 | relabelled by the rule | the base seed |

Six fine-tunes. Each needs its own gate token (the selfcheck of the queue) before it trains. If the trainer has no path to start from a checkpoint with these flags, the builder reports it and nothing is trained until the lead has decided.

Rollouts (pinned tool, as card ADD26): each of the six with no rule; GFT and GRT also with the label-fed rule (`GFTRL`, `GRTRL`), for the share and for the first-tick read.

## 4. Read

`headway/headway_read.py` (U primary, contact, falling behind, gap bias) and `spatial/q3/q3_read.py` (four families), paired on windows, two-level bootstrap (seeds, then episodes), B 2,000, rng 0; 642 lead windows of 56 episodes for the distance cells, 2,399 windows of 141 episodes for the cost cells.

* Lever: GRT minus GFT. Floor: the absolute difference in U between GFTb and GFT, per base; the larger of the two is the floor.
* Share reached = (U of GFT minus U of GRT) / (U of GFT minus U of GFTRL), per base.
* First tick (reported, from the rule logs of GFTRL and GRTRL): the share of lead windows whose plan the label-fed rule changes at the first tick.

## 5. Controls (a failed control is reported first and the outcomes it carries are not named)

* t1: flag off, patched tree against base tree, same seed, 100 steps on the CPU: every logged field and the final weights equal (max abs difference 0.0).
* t2: on a batch, the relabelled target equals the output of `dist_rule.distance_rule` called by hand on the same inputs (exact), and every sample without a label lead keeps its target (exact).
* t3: reach. On a batch with a lead the trajectory loss and its gradient differ between flag on and off; on a batch with no lead they are exactly equal. The line that applies the relabel is deleted in a mutation test and the test must fail.
* t4: the gate. Declared flag equals the built value; the real-data live step reports the count of relabelled samples and it is above 0; a wrong expectation makes the selfcheck fail. The train log carries the count per logged step.
* t5: the eval tool loads a GRT checkpoint with the config of its base (strict load), and the rollout of GFT's tool path is the pinned one (sha256).
* t6: the replicate GFTb is not named different from GFT by the reader on U (if it is, the floor is that difference and it is reported first); the label-fed rule on GFT (GFTRL against GFT) is named as cutting U (the reader sees a known effect).
* t7: the reader controls of every read.

## 6. Outcomes (both bases needed)

* `RULE-TRAINED-CUTS-UNSAFE`: U(GRT) minus U(GFT) is below 0 in both bases, the two-level interval excludes 0, and the mean difference is larger in size than the floor. `RULE-TRAINED-WORSE`: the same above 0. `RULE-TRAINED-FLAT`: otherwise.
* `RULE-TRAINED-REACHES-ORACLE`: the share reached is at least 0.5 in both bases (named only with `RULE-TRAINED-CUTS-UNSAFE`).
* Cost of GRT against GFT by the rule of card ADD11: `RULE-TRAINED-NO-COST` if no cell of the four families is named worse, else `RULE-TRAINED-COSTS` with the cells. Reported beside in every case: position error over 6 s (the target is moved away from the logged future on purpose, so a rise up to that of the label-fed rule, +0.18 m, is the expected price), along-track error, share of windows more than 10 m behind, gap bias, contact rate.

Two bases with a replicate each: a named outcome says the lever moved U beyond the noise of a second fine-tune on these two checkpoints; it is not yet a statement about training runs in general.

## 7. What follows (committed before the data)

* `RULE-TRAINED-CUTS-UNSAFE`: the same three fine-tunes on G_s2 to G_s5 (six bases), the vision-fed rule of cards ADD30 and ADD31 on top of GRT (do the trained rule and the checked rule add up), and an arm trained from the start with the flag on.
* `RULE-TRAINED-FLAT` with the first-tick share fallen: the model follows the rule on logged states and loses it on its own states; next is the same relabel on own-state samples (the own-state bank of the trainer, with the gap corrected by the offset along the track).
* `RULE-TRAINED-FLAT` with the first-tick share unchanged: the dose is too small; next is 6,000 steps and a weight of 4 on the relabelled samples, then training from the start.
* In every case the limit of the tier is stated beside the result: in T1-kin the frames are the logged ones, so the model cannot see its own drift in the image.
