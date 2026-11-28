# PREREG_V8F_PROOF_ADD37_N4 : the low weight of the consistency term on RL bases of new seeds (the confirmation that note N3 section 5 commits)

Note to card `PREREG_V8F_PROOF_ADD37.md` after note N3. Frozen before any arm of this note is trained and before any value of base 1 of note N3 exists (chain `actcons_c1` has not started; chain `actcons_c0` trains its last arm, RLFTb_s0). Tier of the reads: T0 for the imagination reads (diagnostics on logged frames), T1-kin for the families (kinematic rollout on the model's own actions, logged frames, the lead does not react); never driving performance. STRATEGIC: PAUSED (PI item 25).

## 1. What is known at the freeze (MEASURED, one base, an early look that names nothing; `actcons/read_rl_early/`)

Reader of card ADD36 on base 0 (356 lead windows for the response, 441 for the level):

| arm (base 0) | response slope at 2.0 s | rollout read, mean absolute error | present read | H1 |
|---|---|---|---|---|
| RLFT_s0 (control fine-tune) | -0.911 [-1.018, -0.805] | 5.475 m | 6.284 m | +0.046 [-0.010, +0.106] |
| RLA_s0 (weight 1.0) | -6.821 [-10.594, -3.834] | 8,093.860 m | 6.701 m | -0.003 [-0.022, +0.017] |
| RLAm_s0 (weight 0.0001) | +0.805 [+0.748, +0.862] | 5.899 m | 6.272 m | +0.026 [-0.007, +0.064] |

At base 0 the weight 1.0 on the base with the rollout loss breaks the rollout read, and the weight 0.0001 gives the response with the level of the control (error ratio 1.08). The names of note N3 are read by the waiter at both bases as written; this note does not touch them. Note N3 section 5 commits for this case: "the low weight is the candidate; it is confirmed on RL bases of new seeds before any card builds on it".

## 2. Arms (pod, one slot at a time per chain; never more than three GPU jobs; a chain starts only with 28 GB of the container memory free; each arm passes the gate of its chain, and the dry gates named below pass before a chain starts)

* Chain `imag_b`: RL_s2 and RL_s3 = the line of RL_s0 of `arms_imag_a.txt` (pinned tree `tree_leadsup`, 19,164 steps from scratch) with seed 2 and seed 3. Dry gate: RL_s2.
* Chains `actcons_e2`, `actcons_e3` (each after its base is done): RLFT_s<k> (the control fine-tune) and RLAm_s<k> (`--w-act-cons 0.0001`) = the lines of `arms_actcons_c0.txt` with the checkpoint of RL_s<k> as the start and seed k (6,000 steps, lr 1e-4, pinned tree `tree_actcons`). Dry gates: both arms. No arm with weight 1.0 and no second control run (the floor of two checkpoints is the one of bases 0 and 1, RLFTb against RLFT).
* Budget (ESTIMATED from the measured arms of chains `imag_a` and `actcons_c0`): 2.0 to 2.2 h per RL base and 0.7 to 1.2 h per fine-tune, about 8 L40-h in total.

## 3. Reads and names (each needs BOTH new bases, 2 and 3; reader, estimator and controls of card ADD36; families by the rule of ADD12 Q3 with the same-sign rule of ADD13, spread arm G on six seeds)

* `LOW-WEIGHT-RESPONDS-CONFIRMED`: RLAm reads `IMAG-RESPONDS` (card ADD36, H2, pooled over the two new bases) and the slope at 2.0 s is at least 0.5 at each new base, while RLFT reads `IMAG-DEAF`. Else `LOW-WEIGHT-RESPONSE-NOT-CONFIRMED`.
* `LOW-WEIGHT-KEEPS-LEVEL-CONFIRMED`: at each new base the mean absolute error of the rollout read of RLAm is at most 1.15 times that of RLFT, and the pair difference H1(RLAm) minus H1(RLFT) (two-level interval) does not lie below 0. Else `LOW-WEIGHT-LOSES-LEVEL`.
* Price: RLAm against RLFT on the families at T1-kin: `LOW-WEIGHT-NO-COST` or `LOW-WEIGHT-COSTS-<family>`.
* `LOW-WEIGHT-CONFIRMED`: the three positive names together.
* Beside (names nothing): the slope at 1.0 s; H1 of each arm (if RLFT does not read `IMAG-HEAD-ADDS` on the new bases the base has no imagination to add on this read, stated); the same reads on bases 0 and 1 from the read of record of note N3; the train-log course of the term and of the rollout loss; the four bases pooled.

## 4. Controls (all must pass; else nothing is named)

* The controls c1 to c5 of the reader pass on every dump; the controls h1 to h5 and q1 to q5 of the family reads pass.
* Declared against built: the selfcheck of every arm shows the weight 0.0001 (RLAm) or no term (RLFT) and the start checkpoint of its own base (sha256 stated); the selfcheck of RL_s2 and RL_s3 equals that of RL_s0 in every lever.
* The term is alive at this weight: the train log of RLAm shows the term with a finite value on every logged step, and the slope of RLAm differs from that of RLFT at the same base beyond the floor of two checkpoints (the difference RLFTb minus RLFT of the slope at bases 0 and 1, from the read of record).
* A new base is a base: RL_s2 and RL_s3 read `IMAG-DEAF` or a reversed response like RL_s0 and RL_s1 (row H-V8F-PROOF-77); if a new base responds by itself, it is reported and the pair read stands beside it.

## 5. What follows (committed before the data)

* `LOW-WEIGHT-CONFIRMED`: RLAm is the imagination arm X of card ADD40 (its section 3: "the confirmed low weight"). The card is read on bases 0 and 1 if the read of record of note N3 shows the response with the level kept for RLAm at both; else on the new bases 2 and 3. The tool controls p5b and p5r of note ADD40_N4 are read on the bases used, with RLFT as the control arm.
* `LOW-WEIGHT-RESPONDS-CONFIRMED` with `LOW-WEIGHT-LOSES-LEVEL`: the source `cand-imagkin` of note ADD40_N4 on RLFT is the line that is read; next lever a level term inside the consistency loss (the rollout read of the unshifted actions anchored to the label), own note.
* `LOW-WEIGHT-RESPONSE-NOT-CONFIRMED`: reported as written; next lever the weights 0.001 and 0.01 on the new bases (the weight 0.1 without the rollout loss read `IMAG-RESPONDS-WEAKLY`, the weight 1.0 with it breaks the rollout read at base 0), own note.
* Any `LOW-WEIGHT-COSTS-<family>`: reported beside the confirmation; the arm is then not promoted without the PI.

## 6. Limits

Two new seeds, one training run per seed; a fine-tune of 6,000 steps; the response is read around the logged action sequences (note ADD40_N4 section 1 shows that it need not carry over to the model's own plan when the level is lost; the tool control p5b is the test of that on arm X); a response in imagination is a property of the world model, not a driving result.
