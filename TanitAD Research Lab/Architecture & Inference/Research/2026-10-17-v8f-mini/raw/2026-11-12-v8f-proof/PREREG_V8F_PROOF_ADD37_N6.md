# PREREG_V8F_PROOF_ADD37_N6 : the dose read at bases 0 and 1, and the level term inside the consistency loss

Note to card `PREREG_V8F_PROOF_ADD37.md` after notes N4 and N5. Frozen after the dose read at bases 0 and 1 (`actcons/read_f/`, read on 2026-11-19 with both versions of the family and headway readers) and before the level term of section 2 is built, before any arm of this note is trained, and before any read at the new bases 2 and 3 exists (their control and low-weight arms are trained, their dose arms are training, no read of them has run). Tier of the reads: T0 for the imagination reads (diagnostics on logged frames), T1-kin for the families and the headway read (kinematic rollout on the model's own actions, logged frames, the lead does not react); never driving performance. STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument). Pod GPU: A40 (as `nvidia-smi` names it).

## 1. The dose read at bases 0 and 1 (MEASURED; reader of card ADD36, controls c1 to c5 pass on all four dumps; `actcons/read_f/`)

| arm (weight) | response slope at 2.0 s (two-level interval) | rollout read, mean absolute error, base 0 / base 1 | ratio to RLFT, base 0 / base 1 | H1(arm) minus H1(RLFT) | pair against RLAm |
|---|---|---|---|---|---|
| RLFT (0, control; note N5 section 1) | -0.895 [-0.977, -0.817] | 5.475 m / 5.492 m | 1 / 1 | 0 | |
| RLAm (0.0001; note N5 section 1) | +0.811 [+0.759, +0.866] | 5.899 m / 6.008 m | 1.077 / 1.094 | | |
| RLA3 (0.001) | +0.971 [+0.907, +1.047] | 9.362 m / 6.875 m | 1.710 / 1.252 | -0.076 [-0.147, -0.022] | `RL-RESPONDS-BETTER` (distance of the slope from 1 smaller by 0.154 [0.079, 0.197]) |
| RLA2 (0.01) | +0.862 [+0.816, +0.906] | 14.553 m / 9.081 m | 2.658 / 1.654 | -0.067 [-0.145, -0.005] | nothing named (-0.051 [-0.113, +0.017]) |

Both dose arms read `IMAG-RESPONDS` and `IMAG-HEAD-FLAT`. Neither keeps the level by the definition of note N4 (ratio above 1.15 at both bases; the pair difference of H1 lies below 0). Names of note N5 section 3, as written: `DOSE-BREAKS-AT-0.001`, `DOSE-BREAKS-AT-0.01`, and, since no weight is named stronger, `DOSE-NO-GAIN-ABOVE-LOW`. The response itself is nearest to the kinematic size at the weight 0.001 (interval contains 1).

Families at T1-kin (2,399 windows of 141 episodes, second-version reader, controls pass): `NO-COST` for RLA3 minus RLFT, RLA3 minus RLAm, RLA2 minus RLFT, RLA2 minus RLAm. Headway (642 lead windows of 56 episodes, second-version reader, controls pass): the model-induced unsafe rate of RLA3 minus RLFT is +0.0018 [-0.0199, +0.0242]; nothing is named for these pairs by the headway card.

Note N5 section 4 applies: `DOSE-BREAKS-AT-0.001`, so the usable range of the weight alone is below 0.001, and the next lever is a level term inside the consistency loss.

## 2. The lever: a level term (training only; no module, parameter, state-dict key or config field is added)

The consistency term compares the offset rollout with the reference rollout (logged actions) and leaves the reference free; its read drifts away from the label as the weight grows (section 1). The level term anchors the reference read to the label.

* For every valid (sample, start s) pair of the consistency term and each of its two read steps k (5 = 1.0 s, 10 = 2.0 s): if the lead label at frame s + k is valid (presence above 0.5, finite gap, 0 <= gap <= 60 m, `frame_ok` at that frame), `e = | gap_reference(s, k) - gap_label(s + k) |` in metres, where `gap_reference` is the lead head's read of the reference rollout, now computed with gradient.
* `level = mean of e over the valid (pair, step) entries`; no valid entry: zero without a graph.
* The consistency term is unchanged; it uses the reference read detached, as before.
* Added to the loss: `W x consistency + V x level`, with `V = --w-act-level` (default 0.0: the block is not entered and the training step is the one of the dose arms' tree). `V > 0` with `W = 0` is refused.
* Gradient of the level term: predictor and action embedding only. The lead head's parameters are detached (as in the consistency term); the start state is computed without gradient.

## 3. Arms (fine-tunes of 6,000 steps from the RL base of their seed, the line of RLA3 with only the named flag changed)

| arm | `--w-act-cons` | `--w-act-level` | bases |
|---|---|---|---|
| RLVa | 0.001 | 0.001 | 0, 1 |
| RLVb | 0.001 | 0.01 | 0, 1 |
| RLAh (a third dose point, no level term) | 0.0003 | 0 | 0, 1 |

All six on the new tree (the tree of the dose arms plus the level term). RLAh has the level block switched off; control v3 shows that this tree with `V = 0` gives the training step of the dose arms' tree. Two chains (base 0, base 1), each RLVa, RLVb, RLAh in this order; never more than three GPU jobs; a chain starts only with 28 GB of the container memory free; every arm passes the dry gate of its chain. The chains start after the queue of card ADD42 has ended. Budget (ESTIMATED from the measured dose arms, 0.9 h each with three jobs at once; the level arms carry a second backward pass): 6 fine-tunes, about 6 to 8 A40-h.

## 4. Reads and names (bases 0 and 1; readers and bars unchanged)

Reads per arm: the reader of card ADD36 (controls c1 to c5 first); the pair against RLFT, against RLAm and against RLA3; families and headway against RLFT and RLAm with the second-version readers. KEEPS THE LEVEL is the definition of note N4 (at each base the mean absolute error of the rollout read is at most 1.15 times that of RLFT, and the two-level interval of H1(arm) minus H1(RLFT) does not lie below 0).

For each level arm X (RLVa with v = 0.001, RLVb with v = 0.01) exactly one name, tested in this order:

1. `LEVEL-TERM-LOSES-RESPONSE-<v>`: X does not read `IMAG-RESPONDS`.
2. `LEVEL-TERM-LOSES-LEVEL-<v>`: X responds and does not keep the level.
3. `LEVEL-TERM-WORKS-<v>`: X responds, keeps the level, and the pair read X minus RLAm names `RL-RESPONDS-BETTER`.
4. `LEVEL-TERM-KEEPS-LEVEL-NO-GAIN-<v>`: X responds and keeps the level, and the pair read against RLAm does not name it the better one.

RLAh is named by note N5 section 3 with w = 0.0003 (`DOSE-STRONGER-0.0003`, `DOSE-BREAKS-AT-0.0003`, or no name).

Beside (names nothing): per base, the mean absolute error of the rollout read of X against that of RLA3 (does the term act on the quantity it anchors); the slope at 1.0 s; the families of each arm.

## 5. Controls

* v1 (gate): each arm's selfcheck shows the two weights of its name and the start checkpoint of its own base; the dry gate passes; the trainer refuses `V > 0` with `W = 0`.
* v2 (gradient): the live probe reads a gradient norm above 0 on the predictor and on the action embedding for the level term, exactly 0.0 on the lead head and on every other parameter, and exactly 0.0 for a batch with no valid entry.
* v3 (identity): with `V = 0` the loss of a training step equals that of the dose arms' tree exactly (same seed, same batch, CPU test).
* v4 (planted value and mutations): on a constructed case where the reference read is the label plus a constant c, the level term reads |c|; each of these changes turns a test red: the add line removed, the reference read detached in the level term, the label taken one frame early, the head parameters left attached.
* v5 (log): `act_level` is finite on every logged step and `act_level_n` is above 0 on at least 99 percent of the logged steps.
* The reader controls c1 to c5, the family controls and the headway controls, as before. A failed control ends the read of its arm with no name.

## 6. What follows (committed before the data)

* `LEVEL-TERM-WORKS-<v>` for at least one v: the smaller such v is trained at bases 2 and 3 (two fine-tunes); the same three conditions there give `LEVEL-TERM-WORKS-<v>-CONFIRMED`. With `NO-COST` against RLFT at both pairs of bases that arm becomes arm X of card ADD40 beside RLAm, and the tool control p5b of note ADD40_N4 is read on it under the unchanged bars.
* No v works and RLAh is named `DOSE-STRONGER-0.0003`: RLAh is trained at bases 2 and 3 for its confirmation.
* `LEVEL-TERM-KEEPS-LEVEL-NO-GAIN` at both v and no stronger dose: the low weight stays the arm (subject to the confirmation of note N4), and the response is used as it is by the planner (the candidates of note ADD40_N5 section 4).
* `LEVEL-TERM-LOSES-RESPONSE` or `LEVEL-TERM-LOSES-LEVEL` at both v: reported as written. The anchor and the response then trade against each other in this predictor at this size; the lever after that is a change of the action path of the predictor, which is an architecture change and goes to the PI.

## 7. Limits

Two level weights at one consistency weight; two bases, one training run per seed and arm, so a separated pair read says what another draw of episodes would say, not what another training run would say; a fine-tune of 6,000 steps; the level term is trained on the quantity whose error on held-out episodes the level bar reads (train clips against validation episodes); the response is read around the logged action sequences.
