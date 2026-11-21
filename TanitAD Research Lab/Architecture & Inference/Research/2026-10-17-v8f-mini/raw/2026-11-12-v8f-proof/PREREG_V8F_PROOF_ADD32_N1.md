# PREREG_V8F_PROOF_ADD32_N1 : RULETRAIN, the larger dose in two stages (6,000 steps now, the weight on the relabelled samples when it is built)

Note to card `PREREG_V8F_PROOF_ADD32.md`. Frozen before any gate, step or read of the arms named here. The card and its names stay as written.

## 1. What was seen (MEASURED, T1-kin: kinematic rollout on the model's own actions, logged frames, a lead that does not react; 642 lead windows of 56 episodes)

Six fine-tunes of 2,000 steps from G_s0 and G_s1 (chain ruletrain_a), controls t1 to t7 PASS:

* Unsafe rate U, GRT minus GFT: +0.0036 / -0.0036, +0.0000 [-0.0187, +0.0164]; floor (GFTb against GFT) 0.0000 / 0.0036: `RULE-TRAINED-FLAT`. Cost: `RULE-TRAINED-NO-COST`.
* The label-fed rule on GFT cuts U by -0.0676 [-0.1008, -0.0362] (the reader sees the known effect).
* First tick: the label-fed rule changes the plan in 0.2404 / 0.2455 of the lead windows on GFT and in 0.2326 / 0.2378 on GRT: unchanged.
* Dose given: 37,694 / 36,074 samples with a label lead, of them 11,578 / 11,131 targets changed, of about 127,400 samples.
* Reported beside: gap bias at 2 s, GRT minus GFT, +0.066 m [+0.041, +0.097], against +0.435 m for the label-fed rule on GFT; at 4 s and 6 s the shift (+0.58 m, +1.90 m) is of the size the replicate shows (+0.44 m, +1.37 m).

Section 7 of the card commits for `RULE-TRAINED-FLAT` with the first-tick share unchanged: the dose is too small; next is 6,000 steps and a weight of 4 on the relabelled samples, then training from the start.

## 2. Why two stages

The trainer has the flag `--rule-relabel on` and no weight on the relabelled samples; the weight needs a trainer change and its own gate entry. The 6,000 steps need none. Stage 1 runs now with the pinned trainer of the card; stage 2 (the weight of 4 at 6,000 steps) runs when the change is built and checked. The committed step of the card is complete only with stage 2; stage 1 alone does not close it.

## 3. Stage 1 arms (this note)

From the banked checkpoints of G_s0 and G_s1, tree `tree_ruletrain` (pins of the card), `--steps 6000 --lr 1e-4 --warmup 100`, everything else as the arms of the card:

* GFT6 (control, seeds 0 and 1), GFT6b (replicate of the control, seeds 100 and 101), GRT6 (`--rule-relabel on`, seeds 0 and 1): six fine-tunes, two chains (one per base), each arm through the gate of the chain (kind of 6,000 steps, init checkpoint applied).
* Rollouts with the label-fed rule (card ADD26, note N1, cap-only) on GFT6 and GRT6: GFT6RL, GRT6RL, both bases.
* Cost (ESTIMATED from the 2,000-step arms at 7.6 min): about 26 min per arm, about 2.6 L40-h in all.

## 4. Read, controls, outcomes

The read, the floor, the share reached and the first-tick share are those of sections 4 to 6 of the card with GFT6, GFT6b, GRT6, GFT6RL, GRT6RL in place of the 2,000-step arms. Controls t4 (gate and relabel count per arm), t6 (replicate not named different; label-fed rule on GFT6 named as cutting U) and t7 are read on the new arms; t1, t2, t3, t5 are those of the card (same trainer, same pins).

Names of this note (both bases needed): `RULE-TRAINED-CUTS-UNSAFE-6K`, `RULE-TRAINED-WORSE-6K`, `RULE-TRAINED-FLAT-6K`; `RULE-TRAINED-REACHES-ORACLE-6K`; `RULE-TRAINED-NO-COST-6K` or `RULE-TRAINED-COSTS-6K` with the cells; by the rules of section 6 of the card.

Reported beside, names nothing: the dose response from 2,000 to 6,000 steps on U, on the gap bias at 2 s and on the first-tick share (GRT6 minus GFT6 beside GRT minus GFT); GFT6 against GFT (what the longer fine-tune alone does).

## 5. What follows (committed before the data)

* `RULE-TRAINED-CUTS-UNSAFE-6K`: section 7 of the card for `RULE-TRAINED-CUTS-UNSAFE` (six bases, the vision-fed rule on top, an arm trained from the start); stage 2 is then run as a dose point, not as a rescue.
* `RULE-TRAINED-FLAT-6K`: stage 2 runs (weight 4 at 6,000 steps), and beside it, on its own card, the rule applied to the model's own plan as a training term: the unsafe steps of T1-kin come from the model's own plan, and a relabelled logged target only changes the samples where the driver came closer than the rule allows.
* `RULE-TRAINED-WORSE-6K` or any cost cell: reported as written; the arm is not promoted; stage 2 and the own-plan card still run.

## 6. Limits

Two bases with one replicate each; T1-kin with logged frames (the model cannot see its own drift in the image) and a lead that does not react; one corpus; a named outcome says the lever moved U beyond a second fine-tune on these two checkpoints.
