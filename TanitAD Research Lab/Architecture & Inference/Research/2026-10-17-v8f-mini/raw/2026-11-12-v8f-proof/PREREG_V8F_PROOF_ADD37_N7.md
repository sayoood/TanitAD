# PREREG_V8F_PROOF_ADD37_N7 : two interior weights of the level term (bases 0 and 1)

Note to card `PREREG_V8F_PROOF_ADD37.md` after note N6. Frozen before the arms of this note are built or trained (time of the freeze: the line of this note in `TIMES_NOTE.md`; hash in the sidecar file). Stated plainly: this note is written AFTER the early reads of the two level arms of note N6 at bases 0 and 1 were seen, so the weights below are chosen with those reads in view. The names of note N6 stand as read and are not changed by this note. The read of RLAh (third dose point of note N6) had not run when this note was frozen. Tier of the reads: T0 for the imagination reads (diagnostics on logged frames), T1-kin for the families and the headway read (kinematic rollout on the model's own actions, logged frames, the lead does not react); never driving performance. STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).

## 1. What was read (MEASURED, T0, bases 0 and 1; rules of note N6 section 4; `actlevel/read/NAMES_RLVa.json`, `actlevel/read/NAMES_RLVb.json`; controls c1 to c5 pass on all four dumps)

| arm | `--w-act-cons` W | `--w-act-level` V | response slope at 2.0 s (two-level interval), per base | rollout MAE, ratio to RLFT, base 0 / base 1 (bar 1.15) | name (note N6) |
|---|---|---|---|---|---|
| RLVa | 0.001 | 0.001 | +0.9034 [+0.8554, +0.9535]; 0.8788 / 0.9279 | 1.1060 / 1.1977 | `LEVEL-TERM-LOSES-LEVEL-0.001` |
| RLVb | 0.001 | 0.01 | -0.4032 [-0.5492, -0.2681]; -0.3582 / -0.4483 | 1.0395 / 1.0525 | `LEVEL-TERM-LOSES-RESPONSE-0.01` |

Beside (note N6 section 1): the same consistency weight without a level term (RLA3) reads ratios 1.710 / 1.252. So the level term at 0.001 moves the level most of the way to the bar and keeps the response; at 0.01 it holds the level and the response is gone. By note N6 section 6 the term loses at both weights; that is reported as written, and note N6 names the action path (an architecture change, a card for the PI) as the lever after it. That stays.

What this note adds is the cheapest experiment that could still make the weight of this term work: the response is lost somewhere inside one decade of V, and the level bar is missed at one base by 0.048 at the lower end of that decade.

## 2. Arms (fine-tunes of 6,000 steps from the RL base of their seed; the line of RLAm of the base with the tree, the name and the two named flags changed; tree `tree_actlevel` with its pins, unchanged)

| arm | `--w-act-cons` | `--w-act-level` | bases |
|---|---|---|---|
| RLVc | 0.001 | 0.002 | 0, 1 |
| RLVd | 0.001 | 0.004 | 0, 1 |

Two chains (`actlevel2_0`, `actlevel2_1`), each RLVc then RLVd. Every arm passes the dry gate of its chain and the check of its selfcheck (both weights as named, start checkpoint `imag_a/RL_s<k>/`, 6,000 steps) before its chain starts. Never more than three GPU jobs on the pod; a chain starts only with 28 GB of the container memory limit free. Budget (ESTIMATED from the measured level arms of note N6, about 1 h each beside other jobs): 4 fine-tunes, about 4 A40-h (the pod GPU as in note N6).

## 3. Reads and names (bases 0 and 1; reader, controls and bars of note N6 section 4, unchanged)

* Controls first: c1 to c5 of the reader of card ADD36 on every dump; control v5 of note N6 (the level term is finite on every logged step and has valid entries on at least 99 % of them).
* Per arm, in this order, with the level weight as suffix: `LEVEL-TERM-LOSES-RESPONSE-<v>` (the reader does not name `IMAG-RESPONDS`); else `LEVEL-TERM-LOSES-LEVEL-<v>` (KEEPS THE LEVEL of note N4 fails: a rollout MAE above 1.15 times that of RLFT at a base, or the two-level interval of H1(arm) minus H1(RLFT) below 0); else `LEVEL-TERM-WORKS-<v>` (responds, keeps the level, and the pair against RLAm names `RL-RESPONDS-BETTER`); else `LEVEL-TERM-KEEPS-LEVEL-NO-GAIN-<v>`.
* Beside, naming nothing: the families and the headway read against RLFT and RLAm (second-version readers), the slope at 1.0 s, the ratio per base.

## 4. What the late choice of the weights costs (committed now)

* A `LEVEL-TERM-WORKS-<v>` of this note at bases 0 and 1 is a candidate and is written as a candidate. It is stated as a result only with the suffix `-CONFIRMED`, after the same arm trained at the bases of the new seeds 2 and 3 (start checkpoints `imag_b2/RL_s2/`, `imag_b3/RL_s3/`) reads the same name with the same bars. No bar is moved for the new bases.
* The margin is reported with the name: the ratio per base against 1.15, beside the spread of the ratio between the two bases (MEASURED 0.092 for RLVa). A ratio that clears the bar by less than that spread is called narrow in the row.
* No third interior weight is run after these two.

## 5. What follows (committed before the data)

* `LEVEL-TERM-WORKS-<v>` for at least one v: the smaller such v is trained at bases 2 and 3 (two fine-tunes). If `-CONFIRMED`: arm X of card ADD40 (the planner with source `cand-imag` on that arm) with control p5b on it.
* No weight works and the pattern is monotone (the lower weight responds and misses the level, or the higher weight keeps the level and does not respond): `LEVEL-WEIGHT-NO-WINDOW`. The weight of this term cannot give response and level together on these bases. Named now as the lever after it, in this order: (a) a level calibration outside the predictor (a shift of the imagined gap per horizon, fitted on training clips only, read with the same two bars on the arm that responds; no training, own note before any data, first a read of how much of the level error is a constant offset); (b) the action-path change of note N6 (architecture, a card for the PI).
* Both weights keep the level and respond without gain over RLAm (`LEVEL-TERM-KEEPS-LEVEL-NO-GAIN` at both): the low consistency weight stays the arm of record; reported as written.
* A control fails: no name; the failing control is repaired under a note before any repeat.

## 6. Limits

Two bases, one seed per base for these arms (the replicate at bases 2 and 3 is the condition of section 4); weights chosen after two reads of the same family; the imagination reads are T0; the rollout read uses logged frames and a lead that does not react.
