# MM reading: RB-VIS, do the frozen vision features carry the roundabout lateral decision? (2026-11-18)

Source: `out/RESULT_RBVIS.json` (card `PREREG_RBVIS.md`, sha256 412ebd74, hashed at 10:48 local, run started 10:53; MM re-hashed it: equal). Agent build, MM review.
* MEASURED, tier T0 decodability (not driving performance), CPU.
* Paired episode-cluster bootstrap, B 2,000, seed 0, 202 scored RB windows / 16 episodes. It answers "another draw of EPISODES" only.

## Outcome as committed: VIS-SILENT (no HEAD-GAP; RAND does not carry)

| arm | accuracy (5-way) | − S, paired | timing (3-way) | side given turn (43) |
|---|---|---|---|---|
| S (state only, constant LANE_KEEP) | 0.7871 | – | 0.787 | 0.000 |
| V_s0 (frozen N1R_s0 adapter + probe) | 0.7673 | −0.0198 [−0.0777, +0.0410] | 0.782 | 0.163 |
| V_s1 (frozen N1R_s1 adapter + probe) | 0.7228 | −0.0644 [−0.1237, −0.0137], separated worse | 0.767 | 0.070 |
| RAND (random adapter + probe) | 0.7228 | −0.0644 [−0.1813, +0.0186] | 0.748 | 0.093 |
| H_s0 (the trained N1R_s0 head, T0) | 0.7673 | −0.0198 [−0.068, +0.024] | 0.842 | 0.140 |
| H_s1 (the trained N1R_s1 head, T0) | 0.8218 | +0.0347 [−0.0283, +0.1148] | 0.847 | 0.349 |

* Controls held: K-CONST (0.7871 = the eval LANE_KEEP share), K-DISJOINT 0 / 0 / 0, R-SHUF-TRAIN +0.0000 [0, 0].
* V − H: s0 +0.0000 [−0.0481, +0.0533], s1 −0.0990 [−0.1991, −0.0104].

## MM caveats (they bound what VIS-SILENT means)

1. **The probe barely learned in distribution.** Its train-clip holdout accuracy at the chosen epochs is 0.663 (V_s0), 0.670 (V_s1) and 0.661 (RAND). The train LANE_KEEP share is 0.657 (8,022 of 12,201 fit windows; the holdout's own share is not recorded).
   * A probe at its majority rate cannot separate "the features lack it" from "this probe at this budget cannot read it". The traps catalogue warns about exactly this: a negative from a small probe is not a negative about learnability.
   * The run had no positive control on these windows, so the reading is **VIS-SILENT for this probe**, not yet "the encoder lacks it".
2. **Power.** 16 eval episodes, 43 turn windows in 6 episodes. Every interval spans 0.10-0.20, wider than the 0.0383 bar.
3. **One question was never asked: vision AND the exit number together.** Neither the scene alone (which exits exist) nor the text alone (which exit to take) fixes the side. The joint may.
4. **Grid.** The probe read the 4x10 pooled grid. Both caches also hold an 8x20 grid (MM checked one clip in each), which is finer than what the probe saw.

## Next lever (Rule Zero): RB-JOINT, zero training, own card

* Same RB windows, read with more power: 5-fold, episode-grouped cross-validation over the 426 train RB episodes (about 24k windows). The eval-202 numbers are printed beside.
* Arms:
  * S and ED (text), as RB-DIST;
  * RAW4 and RAW8 (raw DINOv3 4x10 / 8x20 tokens, the same probe);
  * VED8 (RAW8 plus the ED features, joint).
* **Positive control, required.** The same probe must decode the current yaw-rate sign (3-way) on the same windows well above its majority rate. Otherwise the read is INVALID. This answers caveat 1.
* Readings, committed in the card before any number:
  * **JOINT**: the scene plus the exit number fixes the side ⇒ the RB-DIST training arm (exit and countdown channels) comes back with that evidence.
  * **GRID**: only the 8x20 grid carries it ⇒ an input-resolution lever.
  * **SILENT with a passing positive control**: the decision is not decodable from these inputs at T0 ⇒ audit the TAC-LAT label on roundabout windows (zero training). Over a 2-6 s horizon a roundabout is entry right, around left, exit right, so the label may be ill-defined there.
