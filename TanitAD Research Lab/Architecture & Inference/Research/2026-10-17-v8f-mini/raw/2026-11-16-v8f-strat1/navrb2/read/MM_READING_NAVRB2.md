# MM reading: NAVRB2, the roundabout state channel (2026-11-18)

Source: `out/RESULT_NAVRB2.json` (card `PREREG_V8F_NAVRB2.md`, sha256 e238ba8d; problems none; K-RE 96 rows, 0 mismatches).
* Arms N1R_s0, N1R_s1, S1R_s0 and S1R_s1 (commit e71782a3), each paired with its banked N1 / S1 at the same seed.
* The four arms are banked on private HF (`Sayood/tanitad-v8f-strat1-runs`, rev 48098c45), with sha256 equal at three copies.
* Every interval is the paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES"; the training-run question is the floor F2 = max(F, F_rep).

## G-EVAL identity before the read (MM, MEASURED)

* The build plan required the identity pair on a real checkpoint before the read (BUILD_NAVRB2.md step 3b). Neither the pod chain nor the read ran it, so the MM ran it on N1R_s0 (`out/geval_pair/`).
  * A: the study wrapper `v8f_mini_eval_rb.py` on zon (`e_nav_rb.weight` exactly 0.0).
  * B: the PINNED `v8f_mini_eval.py` on zoff (the channel keys dropped, cfg nav_rb off).
* Result: **all 22 output arrays over 2,399 windows are bit-equal.**
* So R − banked contrasts (R through the wrapper, banked through the pinned evaluator) carry no wrapper effect.

## Outcome: RB-READ-NO-GAIN (both pairs); S1R also RB-SEEDS-DISAGREE

Deciding cell D1 = TAC-LAT on the roundabout windows (235 windows / 16 episodes), T1-kin, F2_rb = 0.0383:

| pair | s0 | s1 |
|---|---|---|
| N1R − N1 | +0.0043 [−0.0274, +0.0504] | −0.0128 [−0.0600, +0.0362] |
| S1R − S1 | +0.0298 [+0.0119, +0.0511], separated, below F2 | −0.0085 [−0.0675, +0.0495] |

* **Z-USE is separated at both seeds for both pairs: the model READS the channel.**
  * It does not turn the reading into a better tactical decision on roundabout windows.
  * Most likely reason (H): the state channel says "a roundabout is ahead" but carries no exit side, and the side is what TAC-LAT scores.
* **No-cost guard (all windows, four families, under F2):**
  * N1R holds at both seeds.
  * S1R has one-seed costs only: operative curvature, operative headway, tactical time gap.
  * Beside, inside the floor but separated and worse at both seeds for N1R: tactical_2_6s yaw-rate MAE +4.06 [+2.41, +5.86] and +2.06 [+0.39, +4.00] deg/s. It is reported, not labelled.
* **STRATEGIC (T0, in full):**
  * no route cost for either pair, and no separated R − banked cell;
  * route following S-PARTIAL, as for the banked arms (real − donor separated at both seeds, not above the best constant);
  * detected info unchanged: country, surface, illumination and period of day detected; weather not detected; fog unscored per the PI ruling.
  * Named deviation: the T0 donor passes hold the channel at zero beside the donor nav.

## The next lever (Rule Zero)

* The card's RB-READ-NO-GAIN row names it: **RB-DIST**, the exit number and countdown distance as channels.
* **Zero-training first.** Does the route text carry the exit side at all? A nav-only classifier (the navecho2 tool) on the state + exit channels, CPU only.
  * If the text cannot decode the side, no nav channel can teach it, and the lever moves to vision (the roundabout geometry in the scene).
* RB-SEEDS-DISAGREE's E4 (N1R / S1R at s10 / s11, about 0.63 L40-h, E) needs the PI's go. It is not recommended before RB-DIST's zero-training answer: D1 sits inside the floor at every seed.
