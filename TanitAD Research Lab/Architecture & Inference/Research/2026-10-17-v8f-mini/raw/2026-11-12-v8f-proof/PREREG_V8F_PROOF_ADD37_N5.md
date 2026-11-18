# PREREG_V8F_PROOF_ADD37_N5 : the read of record of note N3, the layout of the new-seed chains, and a dose ladder between the two weights

Note to card `PREREG_V8F_PROOF_ADD37.md` after notes N3 and N4. Frozen after the read of record of note N3 (both bases, `actcons/read_rl/`, read on 2026-11-18) and before any arm of note N4 or of this note is trained: the launcher of note N4 never started (its own already-running test matched its own command line; the pod was without a job from 2026-11-16T03:39Z to the start of the chains of this note), so no chain `imag_b*`, `actcons_e*` or `actcons_f` exists. Tier of the reads: T0 for the imagination reads, T1-kin for the families (kinematic rollout on the model's own actions, logged frames, the lead does not react); never driving performance. STRATEGIC: PAUSED (PI item 25). Pod GPU: A40 (as `nvidia-smi` names it); the budget of note N4 is in A40-h.

## 1. The read of record of note N3 (MEASURED, bases 0 and 1; reader of card ADD36, controls c1 to c5 pass on all eight dumps)

| arm | response slope at 2.0 s | H1 | rollout read, mean absolute error | names of the reader |
|---|---|---|---|---|
| RLFT (control) | -0.895 [-0.977, -0.817] | +0.069 [+0.017, +0.134] | 5.483 m | `IMAG-HEAD-ADDS`, `IMAG-DEAF` |
| RLA (weight 1.0) | -10.050 [-17.301, -5.145] | -0.010 [-0.028, +0.006] | 27,597 m | `IMAG-HEAD-FLAT`, `IMAG-DEAF` |
| RLAm (weight 0.0001) | +0.811 [+0.759, +0.866] | +0.050 [+0.006, +0.112] | 5.953 m | `IMAG-HEAD-ADDS`, `IMAG-RESPONDS` |
| RLFTb (second control run) | -0.891 [-0.974, -0.811] | +0.068 [+0.015, +0.132] | 5.510 m | `IMAG-HEAD-ADDS`, `IMAG-DEAF` |

Pair reads: RLAm minus RLFT `RL-RESPONDS-BETTER`; RLA minus RLFT, RLFTb minus RLFT, RLA minus RLAm: nothing named. By the definitions of note N3: the named arm RLA does not respond and loses the imagination (`ACT-CONS-LOSES-IMAGINATION-RL`); the response with the level kept is read for the dose arm RLAm only. Note N3 section 5, second bullet, applies: the low weight is the candidate and is confirmed on new seeds (note N4). The family and headway names of the pairs are written into the row with their numbers.

## 2. Layout of the chains of note N4 (the arms, argv, reads, names and bars of note N4 are unchanged)

* RL_s2 and RL_s3 train at the same time, each in its own chain (`imag_b2`, `imag_b3`), so that the pod's slots are used; dry gate RL_s2 as in note N4.
* Chains `actcons_e2`, `actcons_e3`: RLFT_s<k>, then RLAm_s<k> (the two arms of note N4, first), then the two dose arms of section 3.
* Never more than three GPU jobs; a chain starts only with 28 GB of the container memory free; each arm passes the gate of its chain.

## 3. The dose ladder (new arms; they name the response of a weight, they do not replace the confirmation of note N4)

The weight 1.0 breaks the rollout read, the weight 0.0001 responds at 0.81 of the kinematic size, and on the model's own plans the response of RLAm_s0 reads 0.495 against a bar of 0.5 (note ADD40_N5). Is there a weight between them that responds nearer to 1 with the level kept?

* Arms: RLA3 (`--w-act-cons 0.001`) and RLA2 (`--w-act-cons 0.01`), the line of RLAm with only the weight changed: at bases 0 and 1 in chain `actcons_f` (RLA3_s0, RLA2_s0, RLA3_s1, RLA2_s1), and at the new bases 2 and 3 as the third and fourth arm of chains `actcons_e2`, `actcons_e3`. Dry gates: RLA3_s0 and RLA2_s0.
* Reads per weight w, at bases 0 and 1 and separately at bases 2 and 3: the reader of card ADD36; the pair against RLFT and the pair against RLAm; families and headway against RLFT.
* A weight KEEPS THE LEVEL if at each base of the pair the mean absolute error of its rollout read is at most 1.15 times that of RLFT and the pair difference H1(arm) minus H1(RLFT) does not lie below 0 (the definition of note N4).
* `DOSE-STRONGER-<w>`: at bases 0 and 1 the arm reads `IMAG-RESPONDS`, keeps the level, and its pair read against RLAm names its response as the better one (the pair rule of the reader: the distance of the slope from 1 is smaller, interval excluding 0). `DOSE-STRONGER-<w>-CONFIRMED`: the same three conditions hold at bases 2 and 3. If no weight is named: `DOSE-NO-GAIN-ABOVE-LOW`. If a weight loses the level or breaks the rollout read: `DOSE-BREAKS-AT-<w>` (reported with its numbers).
* Beside (names nothing): slope against the logarithm of the weight over the four weights at bases 0 and 1; the slope at 1.0 s; the families of each weight.
* Controls: those of note N4 section 4 for every arm (selfcheck shows the weight of the arm's name and the start checkpoint of its own base; the term is finite on every logged step).
* Budget (ESTIMATED from the measured arms): 8 fine-tunes at 0.7 to 1.2 h, about 8 A40-h, in slots that the chains of note N4 leave free.

## 4. What follows (committed before the data)

* `LOW-WEIGHT-CONFIRMED` (note N4) and no confirmed stronger weight: arm X of card ADD40 is RLAm, as note N4 section 5 says.
* `DOSE-STRONGER-<w>-CONFIRMED` with `NO-COST` of that weight against RLFT at both pairs of bases: that weight is arm X of card ADD40 (it then has four bases), and RLAm is read beside it; the tool control p5b of note ADD40_N4 is read on it under the unchanged bars.
* `DOSE-STRONGER-<w>` at bases 0 and 1 without the confirmation at 2 and 3: reported as written; the weight is not promoted.
* `DOSE-BREAKS-AT-0.001`: the usable range of the weight is below 0.001; next lever a level term inside the consistency loss (note N4 section 5), own note.

## 5. Limits

Two decades of the weight are sampled at two points; one training run per seed and weight; a fine-tune of 6,000 steps; the response is read around the logged action sequences (the tool control p5b is the test on the model's own plans).
