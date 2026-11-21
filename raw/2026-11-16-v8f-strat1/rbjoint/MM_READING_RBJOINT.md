# MM reading: RB-JOINT, what decodes the roundabout lateral decision, with power (2026-11-18)

Source: `out/RESULT_RBJOINT.json`. The card is `PREREG_RBJOINT.md` (sha256 499860a2), hashed before any number; the run asserts the hash and the MM re-hashed it (equal). Agent build, MM review.
* MEASURED, tier T0 decodability (not driving performance), CPU, about 1.7 h.
* Primary instrument: 5-fold episode-grouped cross-validation over the train RB windows. 12,201 strided windows / 426 episodes, pooled out-of-fold predictions.
* Paired episode-cluster bootstrap, B 2,000, seed 0, over the 426 episodes. It answers "another draw of EPISODES" only.

## Validity: all held (this answers the RB-VIS caveat)

* **POS-YAW passes by a wide margin.** The same probe decodes the current yaw-rate sign at:
  * 0.844 (4x10 grid), +0.268 [+0.233, +0.299] over its majority rate 0.576;
  * 0.848 (8x20 grid), +0.272 [+0.237, +0.303].
* So the probe can read these features. A vision shortfall below is about the features at this budget, not a dead probe.
* R-SHUF-TRAIN: ED −0.0028 (below M); RAW4 and VED8 0.0000. K-CONST: S = 0.6575 = the out-of-fold majority share. Folds are clip-disjoint.

## Outcome as committed: TEXT (the only label that holds)

| pair (5-way TAC-LAT accuracy, CV) | Δ [95 %] | separated | >= M 0.0383 |
|---|---|---|---|
| **ED − S** (state + exit number + countdown distance) | **+0.0581 [+0.0287, +0.0857]** | yes | **yes → TEXT** |
| RAW8 − S (8x20 grid) | +0.0266 [+0.0043, +0.0493] | yes | no |
| RAW4 − S (4x10 grid) | +0.0195 [−0.0051, +0.0433] | no | no |
| RAW8 − RAW4 | +0.0071 [−0.0046, +0.0191] | no | no → not GRID |
| VED8 − S | +0.0235 [+0.0005, +0.0462] | yes | no |
| VED8 − ED | −0.0346 [−0.058, −0.009] | yes, worse | no → not JOINT |
| VED8 − RAW8 | −0.0031 [−0.0122, +0.0064] | no | – |

Secondary (descriptive):
* Timing gain over S: ED +0.080, RAW4 +0.075, RAW8 +0.076, VED8 +0.068.
* Turn windows named with the right side (4,179 turn windows): ED 0.449, RAW4 0.338, RAW8 0.332, VED8 0.329.
* Vision reads WHEN about as well as the text does. The text reads WHICH SIDE better.

Eval-202 (beside, no label; 16 episodes): ED −0.069 [−0.246, +0.078], RAW4 −0.050, RAW8 −0.030, none separated.
* The 16-episode eval subset cannot show an effect of this size. Its intervals are about 0.3 wide against an in-distribution gain of 0.058.
* Its exit-number mix also differs from train (RB-DIST addendum).

## MM caveats

1. **JOINT was not tested fairly.** ED is a gradient-boosted tree on the text features. VED8 is the neural probe with the same features concatenated after pooling.
   * VED8 − RAW8 reads −0.0031: the neural head made no use of the text features at all.
   * "Not JOINT" therefore says this combiner failed. It does not say that vision and text together carry nothing more.
   * A fair combiner, stacking the vision probe's out-of-fold probabilities into the tree, is a cheap CPU check. It does not change the next lever.
2. **RB-DIST (eval-202, TEXT-SILENT) and RB-JOINT (CV, TEXT) are both MEASURED, under different instruments.** One fits all train and scores 16 episodes; the other cross-validates over 426. The CV result is the powered one. The pair shows that the eval subset is too small for this question.
3. One label family, one stride, one probe budget (6 epochs). The vision gain is below the bar here, not absent.

## Next lever (Rule Zero, largest measured effect): the RB-DIST training arm, with a powered evaluation

* **The lever.** The text carries +0.058 over the roundabout state alone, and the trained N1R arms receive only the state channel (NAVRB2). The arm feeds the exit number and the countdown distance as nav channels.
  * This is the NAVRB2 channel table with an exit/distance variant, read against N1R at the same seeds.
* **The evaluation must be powered.** D1 on the 16 eval episodes cannot confirm 0.058.
  * The 10,000 data-scale clips are unseen by every STRAT1 arm and carry the same route text.
  * Their RB windows are a held-out set roughly twice the train RB pool: about 930 episodes (ESTIMATED from the train rate).
  * The card adds a T0 read of TAC-LAT on them, beside D1.
* **Needs:** its own card, a build, and the PI's go for pod time. Cost ESTIMATED from NAVRB2: about 0.16 L40-h per arm, plus the held-out T0 eval on CPU.

## Addendum (2026-11-18, MM): the held-out count is MEASURED

* The NAVRBD feasibility probe (`../navrbd/out/RB10K_COUNTS.json`) counted the RB windows of the 10k clips with rbdist's rule: 548 labelled episodes; 7,594 grid windows / 545 episodes. The "about 930 episodes" above was an estimate from the train rate; the measured pool is 545 episodes, 34 times the 16 of eval-141.
