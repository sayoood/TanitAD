# MM reading: RB-DIST zero-training, does the route text carry the roundabout side? (2026-11-18)

Source: `out/RESULT_RBDIST_ZT.json` (card `PREREG_RBDIST_ZT.md`, sha256 f00a7725, MM re-hashed it on disk: equal). Agent build, MM review.
* Evidence class MEASURED, tier T0 nav-only decodability (not driving performance), CPU, zero training.
* Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, over the 16 eval episodes. It answers "another draw of EPISODES" only.

## Outcome as committed: TEXT-SILENT

* Rule: TEXT-CARRIES-SIDE iff accuracy(ED) − accuracy(S) >= 0.0383 (the NAVRB2 D1 floor) with a separated paired CI.
* Observed ED − S = −0.0792 [−0.2422, +0.0532], not separated, 202 scored RB windows / 16 episodes. The point is below zero.
* All four controls held (K-CONST equals the eval LANE_KEEP share 0.7871 exactly; K-REACH 1.0; eval-shuffle 0 of 10; train-shuffle 0 of 5).

## MM addendum (descriptive, from the counts in the same JSON; no new fit)

The committed 5-way accuracy mixes two questions: WHEN a lateral manoeuvre happens (lane keep against turn) and WHICH side. LANE_KEEP is 159 of the 202 scored windows, so the rule is mostly a timing test. Split from the per-exit label counts:
* **Side given a turn, train (8,463 turn windows):** the majority side per exit number is right on 5,946 = 70.3 %, against 59.2 % for the constant majority side (L). The text carries a weak side prior.
* **Side given a turn, eval (43 turn windows, 6 episodes):** the same train rule is right on 28 = 65.1 %, against 81.4 % for a constant R. The prior does not transfer to these episodes (exit 3 is L 708 / R 302 on train and R 7 of 7 on eval).
* **Timing:** no text feature separates turn from lane-keep windows; ED predicts 50 turns, 17 of them on the 43 turn windows.
* So the next lever is the same as the card's on both questions: the side and the timing must come from the scene.

## Next lever (Rule Zero): RB-VIS, a frozen-feature probe

* Zero-training, own card, before any data.
* Question: do the model's own frozen vision features carry TAC-LAT on roundabout windows, and does the trained model's head use what they carry?
  * Fit on train RB windows once.
  * Score the same 202 eval windows by the same rule, paired against S and against the trained head's own accuracy.
  * Report timing and side-given-turn separately as secondary cells.
  * Add a random-adapter control (the weather probe found a random adapter also detects).
* Reading:
  * features carry it and the head does not use it ⇒ a training-signal lever (roundabout window weight);
  * features do not carry it ⇒ a data or encoder lever.
