# PREREG_V8F_PROOF_ADD37_N3 : ACTCONS on the base with the rollout loss (RL): arms, added dose arm, rerun of the control, order

Note to card `PREREG_V8F_PROOF_ADD37.md` (notes N1, N2). Frozen while RL_s0 trains (about step 8,000 of 19,164) and before RL_s1 has started: no finished checkpoint of an RL arm exists, no arm of this note is gated or trained, and no read of an RL arm exists. The card and its names stay as written. Tier of the imagination reads: T0 (a world-model diagnostic on logged frames); of the family reads: T1-kin (kinematic rollout on the model's own actions, logged frames, a lead that does not react). No number is a driving-performance number. STRATEGIC: PAUSED (PI item 25).

## 1. What is known when this note is written (MEASURED, base LS, two bases; `actcons/read/`)

* LA (weight 1.0): slope of the imagined gap against the kinematic expectation at 2.0 s 0.912 and 0.799 (`IMAG-RESPONDS`); the control LFT reads -0.406 and -0.497 (`IMAG-DEAF`), its rerun LFTb the same within the interval, nothing named for that pair (control a5 holds on base LS). H1 of LA -0.0003 and -0.035 (`IMAG-HEAD-FLAT`) against +0.071 and +0.077 for LFT: the imagined gap is shifted by a near-constant amount (mean absolute error of the rollout read 25 m and 14,719 m against 6.5 and 6.8 m for the present read). Families: `NO-COST`.
* LAm (weight 0.0001): slope at 2.0 s 0.859 and 1.066, at 1.0 s 0.862 and 1.216 (`IMAG-RESPONDS`); H1 -0.003 and -0.014 (`IMAG-HEAD-FLAT`; mean absolute error of the rollout read 149 m and 55 m). Families: `NO-COST`.
* LA01 (weight 0.1) is not read yet.
* So on base LS the response is learned at a weight 10,000 times lower as well, and the level of the imagined gap is lost at both weights. By section 6 of the card (`ACT-CONS-LOSES-IMAGINATION`) the term is combined with the rollout loss on base RL, so that the reference rollout stays tied to the true future. Section 3 of the card names the arms RLFT and RLA.

## 2. Arms (fine-tunes of 6,000 steps, learning rate 1e-4, warm-up 100, tree `tree_actcons`, from RL_s0 and RL_s1 of card ADD36; the argv of RL with its rollout loss kept on)

| arm | flag | role |
|---|---|---|
| RLFT | none | control |
| RLA | `--w-act-cons 1.0` | the named arm of section 3 of the card |
| RLAm | `--w-act-cons 0.0001` | dose arm, names nothing |
| RLFTb | none | rerun of the control (control a5 on this base) |

Order: per base RLFT, RLA, RLAm, RLFTb; base 0 as soon as RL_s0 has ended, base 1 after RL_s1 has ended and the arms of base 0 are done. Each arm passes the gate of its chain; the dry gates of RLFT and RLA and the probe of the term must pass before a chain starts. Never more than three GPU jobs; a chain starts only with 28 GB of the container memory free.

## 3. Reads and names (both bases needed; the rule of section 4 of the card on this base, names with the suffix -RL)

* `ACT-CONS-RESPONDS-RL`: RLA reads `IMAG-RESPONDS` and RLFT does not. `ACT-CONS-PARTLY-RL` and `ACT-CONS-FLAT-RL` by the rule of the card with RLFTb minus RLFT as the floor. If RLFT itself reads `IMAG-RESPONDS`, the response is a property of the base and no response name is given to the term; the slopes are reported.
* `ACT-CONS-KEEPS-IMAGINATION-RL` if RLA reads `IMAG-HEAD-ADDS`; else `ACT-CONS-LOSES-IMAGINATION-RL`. If RLFT does not read `IMAG-HEAD-ADDS`, the base has no imagination to keep on this read and neither name is given; H1 and the mean absolute errors of the three reads are reported for all arms.
* Families at T1-kin, RLA against RLFT: `NO-COST`, `COST-<family>`, `GAIN-<family>` by the rule of the card.
* Beside, naming nothing: RLAm with the same reads; the mean absolute error of the rollout read against that of the present read (the level); the slope at 1.0 s; the train-log course of the term and of the rollout loss.

## 4. Controls

a1 to a3 of the card are those of the tree (unchanged). a4 per arm on the pod (gate, start checkpoint applied, term finite and logged with its count; for the arms without the flag the lever reads 0). a5: RLFTb minus RLFT is not named different on the slope and on H1. a6: c1 to c5 of the read tool for every arm read. a7: the family read keeps q1 to q5.

## 5. What follows (committed before the data)

* `ACT-CONS-RESPONDS-RL` with `ACT-CONS-KEEPS-IMAGINATION-RL` and `NO-COST`: planning by imagination at T1-kin on RLA (candidate plans rolled out with their own actions, the distance rule applied to the imagined gap), own card, as section 6 of the card commits.
* The response and the level only for RLAm and not for RLA: the low weight is the candidate; it is confirmed on RL bases of new seeds before any card builds on it (a dose arm names nothing).
* `ACT-CONS-RESPONDS-RL` with `ACT-CONS-LOSES-IMAGINATION-RL`: the rollout loss does not hold the level against the term; next a level term inside the lever (the head's read of the reference rollout tied to the label gap at the same future frame), built and checked before any arm.
* `ACT-CONS-FLAT-RL`: as section 6 of the card for `ACT-CONS-FLAT`.

## 6. Limits

Two bases; one fine-tune per arm and base, one rerun of the control per base; logged frames; a lead that keeps its motion; one corpus. The base RL is itself read only when its chain has ended; this note is written without that read.

## 7. Cost (ESTIMATED)

Eight fine-tunes of about 40 to 50 min each on one pod slot: about 6 L40-h.
