# MM rulings on PREREG_V8F_CKPT120 (2026-11-16 ~22:55Z, at the freeze)

The card was drafted by an Opus architecture agent before any build, label table or arm. These rulings answer its section 0 and add four items found in review. They bind the builder and the read exactly as the card's own text does.

## Section 0, decisions 1-8: accepted as drafted

* **1. Ladder.** A two-rung ladder, one lever per rung: H1 adds the output, H2 adds the input.
* **2. Isolation.** 30/50/80 m are unchanged. The far point reaches neither T nor O.
* **3. H2 in the same chain.** H2 runs in the same chain without waiting for H1's read. RULE ZERO applies: the labels already predict H1's shortfall (K-NAVINF, MEASURED).
* **4. Deciding form.** The deciding contrast is balanced − 0.5 separated, AND the raw point is above 0. The raw separated form is the STRONG label.
  * This is fixed before any 120 m output exists.
  * The power simulation (ESTIMATED, `CKPT120_POWER.json`) shows the raw form cannot decide at 27 episodes for any follower short of near-perfect.
  * STRAT1's 80 m outcome is not re-read.
* **5. Floor.** F is the larger of the two replicate pairs.
* **6. Labels.** The label build runs on the train release of record, md5 83e30bd3. K-LABEL repeats there before any arm is minted.
* **7. Gate.** The token is `ckpt120-gate-token-v1`, and the runner refuses without it.
* **8. Register note.** STRAT1's S-PARTIAL mechanism goes into the register as a beside diagnostic. The outcome stands as written (row landed with this card).

## Added in review (binding)

9. **Arm priority, reordered so a stopped chain yields both rungs at both seeds.**
   * Order: H1_s0, H1_s1, H2_s0, H2_s1, H1_rep, R1_s0, R2_s0.
   * After four arms, the floor is |S1_s10 − S1_s0|. S1_s10 is banked tonight; the read says which pair set F.
   * The full floor (decision 5) applies once H1_rep exists.
   * The must-fail arms R1 and R2 are still required before any outcome is quoted as final. Until they read, every number is PROVISIONAL.
10. **Echo wording for H2.**
    * H2's S reads the supplied route side directly. So H2's S_CKPT120 measures whether S *follows the route text it is given*. It is worded "follows the supplied route text", never as a perception or decision skill.
    * Any claim that S *sets* the route beyond its input rests on route setting: FAR-SETS against EXTRAP, scored on the driven path.
    * The STRAT1 work item "nav-echo split" applies here: route-following rows are printed with the echo share beside them.
11. **Optimism.** Every STRATEGIC number of this card is worded "with the route supplied along the driven path" (card section 3).
    * The text's next turn is the turn the drive took.
    * A predicted or map-routed goal is a later card under the goal-input rule.
12. **Queue (the PI's slot question, Q3).**
    * The build starts on the PI's go.
    * The arms run on the LoRA pod one job at a time, in the first GPU gap after the go: either between DSCALE10K block 0 (pilot) and block 1 while the block-1 gate items run on CPU, or between two data-scale blocks.
    * A projected chain above 4 h goes back to the MM (card section 9).
